import logging
import numpy as np
import pandas as pd
from itertools import combinations

from utils.spatial import MIN_CELLS, is_dominated
from utils.rules import filter_rule_by_scores_mask
from utils.stats import run_permutation_test

logger = logging.getLogger(__name__)


def run(neighborhoods, coords, cell_types, config, method):
    """
    Entry point for Weighted FP-Growth with Gaussian distance-decay.

    Args:
        neighborhoods: Output of utils.spatial.get_neighborhoods()
        coords:        (N, 2) array of cell coordinates
        cell_types:    (N,) array of cell type labels
        config:        Pipeline config dict
        method:        Spatial method name — must be CN or KNN_R

    Returns:
        mined_rules: DataFrame of association rules (may be empty)
        validate_fn: Callable validate_fn(rules_df) -> rules_df with p_value columns
        stats:       Dict with transaction statistics (sizes, cell_counts, orig, kept)
    """
    trans, stats = _build_transactions(neighborhoods, coords, cell_types, method, config)
    mined_rules = _mine(trans, config)

    def validate_fn(rules_df):
        return _validate_rules(neighborhoods, coords, cell_types, rules_df, config, method)

    return mined_rules, validate_fn, stats


# ---------------------------------------------------------------------------
# Phase A: Spatial Preprocessing & Weight Calculation
# ---------------------------------------------------------------------------

def _build_transactions(neighborhoods, coords, cell_types, method, config):
    """
    Builds weighted transactions using Gaussian kernel distance-decay.

    For each neighborhood, computes per-cell-type diffusion weights:
        w_i = exp(-0.5 * (d_i / b)^2)      (Gaussian kernel, b = bandwidth)
        Weight(T) = sum(w_i) for all cells of type T in the neighborhood

    Weights are normalized per transaction to sum to 1, removing density bias
    across heterogeneous FOVs.

    Args:
        b (bandwidth): defaults to config["BANDWIDTH"] or config["RADIUS"].
                       At d=b, Gaussian weight = exp(-0.5) ≈ 0.6.

    Returns:
        transactions: List[Dict[str, float]]  e.g. [{'T_cell': 0.6, 'B_cell': 0.4}]
        stats:        Dict with sizes, cell_counts, orig, kept
    """
    if method not in ("CN", "KNN_R"):
        raise ValueError(
            f"weighted_fpgrowth only supports CN and KNN_R methods (got {method!r}). "
            "Distance-decay requires a well-defined center cell. "
            "Use method='CN' or 'KNN_R', or switch ALGO to 'fpgrowth'."
        )

    b = config.get("BANDWIDTH", config["RADIUS"])

    transactions = []
    sizes = []
    cell_counts = []
    orig_count = len(neighborhoods)

    for item in neighborhoods:
        center_i, idxs = item
        if len(idxs) < MIN_CELLS:
            continue
        raw_types = cell_types[idxs]
        if is_dominated(raw_types):
            continue

        neighbor_idxs = [n for n in idxs if n != center_i]
        if not neighbor_idxs:
            continue

        # Gaussian weight for each neighbor based on distance from center
        dists = np.linalg.norm(coords[neighbor_idxs] - coords[center_i], axis=1)
        weights = np.exp(-0.5 * (dists / b) ** 2)

        trans = {}
        for n_idx, w in zip(neighbor_idxs, weights):
            ct = f"{cell_types[n_idx]}_NEIGHBOR"
            trans[ct] = trans.get(ct, 0.0) + w

        # Center cell at distance 0 → Gaussian weight = 1.0
        trans[f"{cell_types[center_i]}_CENTER"] = 1.0

        if not trans:
            continue

        # Normalize weights to sum to 1 (removes FOV density bias)
        total = sum(trans.values())
        trans = {k: v / total for k, v in trans.items()}

        transactions.append(trans)
        sizes.append(len(trans))
        cell_counts.append(len(idxs))

    stats = {
        "sizes": sizes,
        "cell_counts": cell_counts,
        "orig": orig_count,
        "kept": len(transactions),
    }
    return transactions, stats


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
        self.link = None  # horizontal link to next node with same item in header


class WeightedFPTree:
    """
    FP-Tree that accumulates float weights instead of integer counts.

    Header table: {item: [total_weight, gmaxw, first_node]}
      - total_weight: sum of item's weights across all transactions (weighted support)
      - gmaxw:        max weight of this item in any single transaction (Global Max Weight)
      - first_node:   head of the linked list of nodes for this item
    """

    def __init__(self):
        self.root = _FPNode(None, None)
        self.header = {}

    def first_pass(self, transactions):
        """Scan transactions to compute header weights and GMAXW."""
        for trans in transactions:
            for item, w in trans.items():
                if item not in self.header:
                    self.header[item] = [0.0, 0.0, None]
                self.header[item][0] += w
                if w > self.header[item][1]:
                    self.header[item][1] = w  # update GMAXW

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
                new_node.link = self.header[item][2]
                self.header[item][2] = new_node
            node = node.children[item]


# ---------------------------------------------------------------------------
# Phase C: Mining with GMAXW-safe pruning
# ---------------------------------------------------------------------------

def _mine(transactions, config):
    """Build WeightedFPTree and mine frequent weighted itemsets → rules DataFrame."""
    if not transactions:
        return pd.DataFrame()

    N = len(transactions)
    # MIN_SUPPORT is a fraction [0,1]; scale to raw sum for tree operations
    min_support_raw = config["MIN_SUPPORT"] * N

    tree = WeightedFPTree()
    tree.first_pass(transactions)
    tree.prune(min_support_raw)

    for trans in transactions:
        tree.insert(trans)

    gmaxw = {item: v[1] for item, v in tree.header.items()}
    itemsets_raw = _mine_tree(tree, min_support_raw, [], gmaxw)

    if not itemsets_raw:
        return pd.DataFrame()

    # Normalize support back to fraction to match MIN_SUPPORT semantics
    itemsets = [(fs, sup / N) for fs, sup in itemsets_raw]
    support_map = {fs: sup for fs, sup in itemsets}
    return _itemsets_to_rules(itemsets, support_map, config)


def _mine_tree(tree, min_support, prefix, gmaxw):
    """
    Recursively mine the FP-tree via conditional pattern bases.

    GMAXW pruning (Phase C requirement):
    In conditional contexts, item Y's accumulated weight is bounded by X's
    weights in the prefix paths. We only prune Y if:
        conditional_weight(Y) * gmaxw[Y] < min_support
    This prevents dropping itemsets where Y is low in this context but
    high-weight globally — keeping the algorithm mathematically conservative.
    """
    frequent = []

    for item in sorted(tree.header.keys(), key=lambda x: tree.header[x][0]):
        if tree.header[item][0] < min_support:
            continue

        new_prefix = prefix + [item]
        frequent.append((frozenset(new_prefix), tree.header[item][0]))

        # Collect conditional pattern base (prefix paths to each node of this item)
        cond_patterns = []
        node = tree.header[item][2]
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

        # Build conditional FP-tree with GMAXW-safe pruning
        cond_tree = WeightedFPTree()
        cond_tree.first_pass(cond_patterns)
        cond_tree.header = {
            i: v for i, v in cond_tree.header.items()
            if v[0] * gmaxw.get(i, 1.0) >= min_support
        }

        for pattern in cond_patterns:
            cond_tree.insert(pattern)

        frequent.extend(_mine_tree(cond_tree, min_support, new_prefix, gmaxw))

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

                if s_ant == 0 or s_con == 0 or s_both < config["MIN_SUPPORT"]:
                    continue

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
    rules = rules[(rules["len_ant"] + rules["len_con"]) <= config["MAX_RULE_LENGTH"]]

    # weighted_fpgrowth always uses CN/KNN_R: antecedent must contain center cell
    rules = rules[rules["antecedents"].apply(lambda x: any("_CENTER" in i for i in x))]

    return rules.drop_duplicates().reset_index(drop=True)


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

def _check_rule(rule_row, transactions, config):
    """
    Checks if a rule passes thresholds in weighted transactions.

    Weighted support of an itemset I in transaction T:
        = min(weight of each item in I) if all items present, else 0
    This rewards itemsets where ALL items have high diffusion weight.
    """
    if not transactions:
        return False

    N = len(transactions)
    antecedents = set(rule_row["antecedents"])
    consequents = set(rule_row["consequents"])
    rule_items = antecedents | consequents

    sum_both = sum_ant = sum_con = 0.0

    for trans in transactions:
        has_ant = antecedents.issubset(trans)
        has_con = consequents.issubset(trans)
        if has_ant:
            sum_ant += min(trans[i] for i in antecedents)
        if has_con:
            sum_con += min(trans[i] for i in consequents)
        if has_ant and has_con:
            sum_both += min(trans[i] for i in rule_items)

    support_both = sum_both / N
    support_ant = sum_ant / N
    support_con = sum_con / N

    if support_both < config["MIN_SUPPORT"] or support_ant == 0 or support_con == 0:
        return False

    confidence = support_both / support_ant
    lift = confidence / support_con
    leverage = support_both - support_ant * support_con
    conviction = float("inf") if confidence >= 1.0 else (1 - support_con) / (1 - confidence)

    return filter_rule_by_scores_mask(config, lift, leverage, conviction, confidence)


def _validate_rules(neighborhoods, coords, cell_types, rules_df, config, method):
    """Validates rules via permutation test with weighted transaction rebuilding."""
    if rules_df.empty:
        return rules_df

    def build_fn(shuffled_types):
        trans, _ = _build_transactions(neighborhoods, coords, shuffled_types, method, config)
        return trans

    def check_fn(rule_row, transactions):
        return _check_rule(rule_row, transactions, config)

    return run_permutation_test(rules_df, cell_types, config["N_PERMUTATIONS"], build_fn, check_fn)
