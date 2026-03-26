import logging
import numpy as np
import pandas as pd
from itertools import combinations

from utils.spatial import MIN_CELLS, is_dominated
from utils.rules import filter_rule_by_scores_mask
from utils.stats import run_permutation_test

from typing import Optional

logger = logging.getLogger(__name__)


def run(neighborhoods, coords, cell_types, config, method, functional_subtypes: Optional[np.ndarray] = None):
    """
    Entry point for Weighted FP-Growth with Gaussian distance-decay.

    Args:
        neighborhoods: Output of utils.spatial.get_neighborhoods()
        coords:        (N, 2) array of cell coordinates
        cell_types:    (N,) array of cell type labels
        config:        Pipeline config dict
        method:        Spatial method name — must be CN or KNN_R
        functional_subtypes: (N,) array of lists of functional marker strings

    Returns:
        mined_rules: DataFrame of association rules (may be empty)
        validate_fn: Callable validate_fn(rules_df) -> rules_df with p_value columns
        stats:       Dict with transaction statistics (sizes, cell_counts, orig, kept)
    """
    if method not in ("CN", "KNN_R"):
        raise ValueError(
            f"weighted_fpgrowth only supports CN and KNN_R methods (got {method!r}). "
            "Distance-decay requires a well-defined center cell. "
            "Use method='CN' or 'KNN_R', or switch ALGO to 'fpgrowth'."
        )

    # Precompute geometry once — distances and Gaussian weights are fixed for this
    # sample regardless of how many times cell type labels are shuffled.
    geo_cache = _precompute_geometry(neighborhoods, coords, config)

    # PERFORMANCE OPTIMIZATION: Pre-calculate spatial labels once to avoid 
    # millions of f-string operations inside the permutation loop.
    neighbor_labels = np.array([f"{ct}_NEIGHBOR" for ct in cell_types])
    center_labels = np.array([f"{ct}_CENTER" for ct in cell_types])
    
    fn_labels = fc_labels = None
    if functional_subtypes is not None:
        fn_labels = np.array([[f"{s}_NEIGHBOR" for s in st] for st in functional_subtypes], dtype=object)
        fc_labels = np.array([[f"{s}_CENTER" for s in st] for st in functional_subtypes], dtype=object)

    trans, sizes, cell_counts = _build_transactions_from_cache(
        geo_cache, cell_types, neighbor_labels, center_labels, fn_labels, fc_labels
    )
    stats = {
        "sizes": sizes,
        "cell_counts": cell_counts,
        "orig": len(neighborhoods),
        "kept": len(trans),
    }
    mined_rules = _mine(trans, config)

    def validate_fn(rules_df):
        return _validate_rules(
            geo_cache, cell_types, neighbor_labels, center_labels, 
            rules_df, config, fn_labels, fc_labels
        )

    return mined_rules, validate_fn, stats


# ---------------------------------------------------------------------------
# Phase A: Spatial Preprocessing & Weight Calculation
# ---------------------------------------------------------------------------

def _precompute_geometry(neighborhoods, coords, config):
    """
    Pre-computes per-neighborhood geometry: neighbor indices and Gaussian weights.

    This is called once per sample. The results are reused across all permutation
    iterations in _validate_rules — distances and decay weights are invariant to
    cell type label shuffling, so there is no need to recompute them per permutation.

    Args:
        b (bandwidth): config["BANDWIDTH"] or config["RADIUS"].
                       At d=b, Gaussian weight = exp(-0.5) ≈ 0.6.

    Returns:
        List of (center_i, all_idxs_arr, neighbor_idxs_arr, gaussian_weights_arr)
        Only neighborhoods that pass MIN_CELLS and have at least one neighbor are included.
    """
    b = config.get("BANDWIDTH", config["RADIUS"])
    cache = []

    for center_i, idxs in neighborhoods:
        if len(idxs) < config.get("MIN_CELLS_PER_PATCH", MIN_CELLS):
            continue

        neighbor_idxs = np.array([n for n in idxs if n != center_i])
        if len(neighbor_idxs) == 0:
            continue

        dists = np.linalg.norm(coords[neighbor_idxs] - coords[center_i], axis=1)
        weights = np.exp(-0.5 * (dists / b) ** 2)
        cache.append((center_i, np.array(idxs), neighbor_idxs, weights))

    return cache


def _build_transactions_from_cache(geo_cache, cell_types, neighbor_labels, center_labels, fn_labels=None, fc_labels=None):
    """
    Builds weighted transactions using pre-computed geometry and pre-calculated labels.
    """
    transactions = []
    sizes = []
    cell_counts = []

    for center_i, all_idxs, neighbor_idxs, weights in geo_cache:
        raw_types = cell_types[all_idxs]
        if is_dominated(raw_types):
            continue

        trans = {}
        for n_idx, w in zip(neighbor_idxs, weights):
            ct = neighbor_labels[n_idx]
            trans[ct] = trans.get(ct, 0.0) + w
            if fn_labels is not None:
                for st in fn_labels[n_idx]:
                    trans[st] = trans.get(st, 0.0) + w

        # Center cell
        ct_center = center_labels[center_i]
        trans[ct_center] = 1.0
        if fc_labels is not None:
            for st in fc_labels[center_i]:
                trans[st] = 1.0

        # Cap each neighbor weight at 1.0
        trans = {k: min(v, 1.0) for k, v in trans.items()}

        transactions.append(trans)
        sizes.append(len(trans))
        cell_counts.append(len(all_idxs))

    return transactions, sizes, cell_counts


# ---------------------------------------------------------------------------
# Phase B: Custom Weighted FP-Tree
# ---------------------------------------------------------------------------

class _FPNode:
    __slots__ = ["item", "weight", "parent", "children", "link"]

    def __init__(self, item, parent):
        self.item = item
        self.weight = 0.0
        self.parent = parent
        self.children = {}
        self.link = None


class WeightedFPTree:
    """
    FP-Tree that accumulates float weights instead of integer counts.

    Header table: {item: [total_weight, first_node]}
      - total_weight: sum of item's weights across all transactions (weighted support)
      - first_node:   head of the linked list of nodes for this item
    """

    def __init__(self):
        self.root = _FPNode(None, None)
        self.header = {}

    def first_pass(self, transactions):
        """Scan transactions to compute per-item total weight."""
        for trans in transactions:
            for item, w in trans.items():
                if item not in self.header:
                    self.header[item] = [0.0, None]
                self.header[item][0] += w

    def prune(self, min_support):
        """Remove items whose total weight is below min_support."""
        self.header = {
            item: v for item, v in self.header.items()
            if v[0] >= min_support
        }

    def insert(self, transaction):
        """Insert a transaction dict into the tree, incrementing float node weights."""
        items = [item for item in transaction if item in self.header]
        # Sort by total weight descending for tree compression
        items.sort(key=lambda x: self.header[x][0], reverse=True)

        node = self.root
        for item in items:
            w = transaction[item]
            if item in node.children:
                node.children[item].weight += w
            else:
                new_node = _FPNode(item, node)
                new_node.weight = w
                node.children[item] = new_node
                # Prepend to header linked list
                new_node.link = self.header[item][1]
                self.header[item][1] = new_node
            node = node.children[item]


# ---------------------------------------------------------------------------
# Phase C: Mining with min-based weighted support
# ---------------------------------------------------------------------------

def _mine(transactions, config):
    """Build WeightedFPTree and mine frequent weighted itemsets → rules DataFrame."""
    if not transactions:
        return pd.DataFrame()

    N = len(transactions)
    min_support_raw = max(
        config["MIN_SUPPORT"] * N,
        config.get("MIN_ABS_SUPPORT", 0),
    )

    tree = WeightedFPTree()
    tree.first_pass(transactions)
    tree.prune(min_support_raw)

    for trans in transactions:
        tree.insert(trans)

    itemsets_raw = _mine_tree(tree, min_support_raw, [])

    if not itemsets_raw:
        return pd.DataFrame()

    # Prune based on size before recomputing exact support
    max_itemset_size = config["MAX_RULE_LENGTH"]
    itemsets_raw = [(fs, w) for fs, w in itemsets_raw if len(fs) <= max_itemset_size]

    if not itemsets_raw:
        return pd.DataFrame()

    # Recompute support using min-based semantics: the FP-tree's accumulated
    # weights are an upper bound on min-based support, not the exact value.
    # Any itemset whose true min-based support falls below the threshold is discarded.
    effective_min_support = min_support_raw / N
    W, item_idx_map = _build_weight_matrix(transactions)
    
    itemsets = []
    for fs, _ in itemsets_raw:
        # Uses fast NumPy math instead of Python loops
        sup = _calculate_weighted_support_vectorized(fs, W, item_idx_map)
        if sup >= effective_min_support:
            itemsets.append((fs, sup))

    if not itemsets:
        return pd.DataFrame()

    support_map = {fs: sup for fs, sup in itemsets}
    return _itemsets_to_rules(itemsets, support_map, config)


def _mine_tree(tree, min_support, prefix):
    """
    Recursively mine the FP-tree via conditional pattern bases.

    Pruning: item i is skipped if its accumulated conditional weight is below
    min_support. Under min-based support, downward closure holds, so this is
    an exact condition (not just an upper bound).
    """
    frequent = []
    for item in sorted(tree.header.keys(), key=lambda x: tree.header[x][0]):
        if tree.header[item][0] < min_support:
            continue

        new_prefix = prefix + [item]
        frequent.append((frozenset(new_prefix), tree.header[item][0]))

        # Collect conditional pattern base (prefix paths to each node of this item)
        cond_patterns = []
        node = tree.header[item][1]
        while node is not None:
            path = {}
            parent = node.parent
            while parent.item is not None:
                path[parent.item] = path.get(parent.item, 0.0) + node.weight
                parent = parent.parent
            if path:
                cond_patterns.append(path)
            node = node.link

        if not cond_patterns:
            continue

        # Build conditional FP-tree. Under min-based support, downward closure
        # holds (support({A,B}) ≤ support({A})), so standard pruning is exact.
        cond_tree = WeightedFPTree()
        cond_tree.first_pass(cond_patterns)
        cond_tree.prune(min_support)

        for pattern in cond_patterns:
            cond_tree.insert(pattern)

        frequent.extend(_mine_tree(cond_tree, min_support, new_prefix))

    return frequent


def _itemsets_to_rules(itemsets, support_map, config):
    """Generate and filter association rules from frequent weighted itemsets."""
    rows = []

    for itemset, _ in itemsets:
        if len(itemset) < 2:
            continue
        for r in range(1, len(itemset)):
            for ant in combinations(sorted(itemset), r):
                ant = frozenset(ant)
                con = itemset - ant

                s_both = support_map.get(itemset, 0)
                s_ant = support_map.get(ant, 0)
                s_con = support_map.get(con, 0)

                assert s_ant > 0 and s_con > 0, f"Closure violation for {itemset}"

                confidence = s_both / s_ant
                lift = confidence / s_con
                leverage = s_both - s_ant * s_con
                conviction = float("inf") if confidence >= 1.0 else (1 - s_con) / (1 - confidence)

                if not filter_rule_by_scores_mask(config, lift, leverage, conviction, confidence):
                    continue

                rows.append({
                    "antecedents": tuple(sorted(ant)),
                    "consequents": tuple(sorted(con)),
                    "support": s_both,
                    "antecedent support": s_ant,
                    "consequent support": s_con,
                    "confidence": confidence,
                    "lift": lift,
                    "leverage": leverage,
                    "conviction": conviction,
                })

    if not rows:
        return pd.DataFrame()

    rules = pd.DataFrame(rows)
    rules["len_ant"] = rules["antecedents"].apply(len)
    rules["len_con"] = rules["consequents"].apply(len)
    
    # weighted_fpgrowth always uses CN/KNN_R: antecedent must contain center cell
    rules = rules[rules["antecedents"].apply(lambda x: any("_CENTER" in i for i in x))]
    return rules.drop_duplicates().reset_index(drop=True)


# ---------------------------------------------------------------------------
# Phase D: High-Performance Vectorized Utilities
# ---------------------------------------------------------------------------

def _build_weight_matrix(transactions):
    """
    Converts a list of weighted transaction dicts into a dense NumPy matrix.

    Returns:
        W:        float64 array of shape (N_transactions, N_unique_items)
                  W[t, j] = weight of item j in transaction t (0 if absent)
        item_idx: dict mapping item label -> column index in W
    """
    if not transactions:
        return np.array([]), {}
        
    all_items = sorted({item for trans in transactions for item in trans})
    item_idx = {item: i for i, item in enumerate(all_items)}
    W = np.zeros((len(transactions), len(all_items)), dtype=np.float64)
    for t_idx, trans in enumerate(transactions):
        for item, w in trans.items():
            W[t_idx, item_idx[item]] = w
    return W, item_idx


def _calculate_weighted_support_vectorized(itemset, W, item_idx_map):
    """Computes min-based weighted support using NumPy matrix operations."""
    if W.size == 0:
        return 0.0
        
    items = list(itemset)
    cols = [item_idx_map[i] for i in items if i in item_idx_map]
    
    if len(cols) < len(items):
        return 0.0
        
    # extract columns, find min per row, sum and normalize
    return float(np.min(W[:, cols], axis=1).sum()) / W.shape[0]


def _check_all_rules_vectorized(rules_df, transactions, config):
    """Vectorized check of all discovered rules against a transaction list."""
    if not transactions:
        return np.zeros(len(rules_df), dtype=bool)

    N = len(transactions)
    W, item_idx = _build_weight_matrix(transactions)
    results = np.zeros(len(rules_df), dtype=bool)
    # Mirror the same effective threshold used during mining
    effective_min_support = max(
        config["MIN_SUPPORT"],
        config.get("MIN_ABS_SUPPORT", 0) / N,
    )

    for r_idx, (_, row) in enumerate(rules_df.iterrows()):
        ant_cols = [item_idx[i] for i in row["antecedents"] if i in item_idx]
        con_cols = [item_idx[i] for i in row["consequents"] if i in item_idx]
        
        if len(ant_cols) < len(row["antecedents"]) or len(con_cols) < len(row["consequents"]):
            continue

        # Mathematical logic: find min weight per row for ant, con, and combined
        ant_W = W[:, ant_cols]
        con_W = W[:, con_cols]
        all_W = W[:, ant_cols + con_cols]

        has_ant = np.all(ant_W > 0, axis=1)
        has_con = np.all(con_W > 0, axis=1)
        has_both = has_ant & has_con

        sum_ant = float(ant_W[has_ant].min(axis=1).sum()) if has_ant.any() else 0.0
        sum_con = float(con_W[has_con].min(axis=1).sum()) if has_con.any() else 0.0
        sum_both = float(all_W[has_both].min(axis=1).sum()) if has_both.any() else 0.0

        support_both = sum_both / N
        support_ant = sum_ant / N
        support_con = sum_con / N

        if support_both < effective_min_support or support_ant == 0 or support_con == 0:
            continue

        confidence = support_both / support_ant
        lift = confidence / support_con
        leverage = support_both - support_ant * support_con
        conviction = float("inf") if confidence >= 1.0 else (1 - support_con) / (1 - confidence)

        results[r_idx] = filter_rule_by_scores_mask(config, lift, leverage, conviction, confidence)

    return results


def _validate_rules(geo_cache, cell_types, neighbor_labels, center_labels, rules_df, config, fn_labels=None, fc_labels=None):
    """
    Validates rules via permutation test.
    Uses high-speed matrix math and pre-calculated labels to avoid bottlenecks.
    """
    if rules_df.empty:
        return rules_df

    def build_fn(shuffled_indices):
        # Slice both using the same shuffled indices
        sh_types = cell_types[shuffled_indices]
        sh_n = neighbor_labels[shuffled_indices]
        sh_c = center_labels[shuffled_indices]
        sh_fn = fn_labels[shuffled_indices] if fn_labels is not None else None
        sh_fc = fc_labels[shuffled_indices] if fc_labels is not None else None
        
        trans, _, _ = _build_transactions_from_cache(geo_cache, sh_types, sh_n, sh_c, sh_fn, sh_fc)
        return trans

    def validation_callback(rdf, transactions):
        return _check_all_rules_vectorized(rdf, transactions, config)

    return run_permutation_test(rules_df, len(cell_types), config["N_PERMUTATIONS"], build_fn,
                                check_fn=None, batch_check_fn=validation_callback)
