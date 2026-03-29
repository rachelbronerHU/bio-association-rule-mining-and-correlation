import logging
import time
import numpy as np
import pandas as pd
from scipy import sparse
from statsmodels.stats.multitest import multipletests
from utils.rules import filter_rule_by_scores_mask

logger = logging.getLogger(__name__)


def apply_fdr_correction(p_values):
    """Applies Benjamini-Hochberg correction. Returns adjusted p-values."""
    if len(p_values) == 0:
        return []
    _, pvals_corrected, _, _ = multipletests(p_values, method="fdr_bh")
    return pvals_corrected


def calculate_metrics(support_both, support_ant, support_con):
    """Calculates association metrics: confidence, lift, leverage, conviction."""
    if support_ant == 0:
        return 0.0, 0.0, 0.0, float("inf")
    
    confidence = support_both / support_ant
    lift = confidence / support_con if support_con > 0 else 0.0
    leverage = support_both - (support_ant * support_con)
    conviction = float("inf") if confidence >= 1.0 else (1 - support_con) / (1 - confidence)
    
    return confidence, lift, leverage, conviction


def _build_weight_matrix(transactions):
    """
    Internal helper to convert transactions to a dense weight matrix.
    Used during initial mining/rule discovery phase.
    """
    if not transactions:
        return np.array([]), {}
        
    all_items = sorted({item for trans in transactions for item in trans})
    item_idx_map = {item: i for i, item in enumerate(all_items)}
    trans_mat = np.zeros((len(transactions), len(all_items)), dtype=np.float64)
    
    for t_idx, trans in enumerate(transactions):
        if isinstance(trans, set):
            for item in trans:
                trans_mat[t_idx, item_idx_map[item]] = 1.0
        else:
            for item, w in trans.items():
                trans_mat[t_idx, item_idx_map[item]] = w
    return trans_mat, item_idx_map


def calculate_support_vectorized(itemset, trans_mat, item_idx_map):
    """Computes (weighted) support using NumPy matrix operations."""
    if trans_mat.size == 0:
        return 0.0
    items = list(itemset)
    cols = [item_idx_map[i] for i in items if i in item_idx_map]
    if len(cols) < len(items):
        return 0.0
    return float(np.min(trans_mat[:, cols], axis=1).sum()) / trans_mat.shape[0]


def check_rules_batch(rules_df, trans_mat, item_idx_map, config):
    """
    Unified high-performance validation for ALL algorithms.
    Validates all rules at once against a Transaction Matrix via NumPy.
    """
    if rules_df.empty or trans_mat.size == 0:
        return np.zeros(len(rules_df), dtype=bool)

    num_trans = trans_mat.shape[0]
    results = np.zeros(len(rules_df), dtype=bool)
    min_sup = config["MIN_SUPPORT"]

    for r_idx, (_, row) in enumerate(rules_df.iterrows()):
        ant_cols = [item_idx_map[i] for i in row["antecedents"] if i in item_idx_map]
        con_cols = [item_idx_map[i] for i in row["consequents"] if i in item_idx_map]
        
        if len(ant_cols) < len(row["antecedents"]) or len(con_cols) < len(row["consequents"]):
            continue

        ant_weights = trans_mat[:, ant_cols]
        con_weights = trans_mat[:, con_cols]
        rule_weights = trans_mat[:, ant_cols + con_cols]

        has_ant = np.all(ant_weights > 0, axis=1)
        has_con = np.all(con_weights > 0, axis=1)
        has_both = has_ant & has_con

        sup_ant = float(ant_weights[has_ant].min(axis=1).sum()) / num_trans if has_ant.any() else 0.0
        sup_con = float(con_weights[has_con].min(axis=1).sum()) / num_trans if has_con.any() else 0.0
        sup_both = float(rule_weights[has_both].min(axis=1).sum()) / num_trans if has_both.any() else 0.0

        if sup_both < min_sup or sup_ant == 0 or sup_con == 0:
            continue

        confidence, lift, leverage, conviction = calculate_metrics(sup_both, sup_ant, sup_con)
        results[r_idx] = filter_rule_by_scores_mask(config, lift, leverage, conviction, confidence)

    return results


def prepare_validation_matrices(neighborhoods, num_cells, cell_types, functional_subtypes=None, weighted_geo=None):
    """
    Pre-computes fixed Adjacency and Label matrices for the High-Speed simulation.
    
    Args:
        neighborhoods: List of neighborhood indices.
        n_cells:       Total physical cells.
        cell_types:    Original cell labels.
        functional_subtypes: Marker labels.
        weighted_geo: Optional list of (center_i, all_idxs, neighbor_idxs, weights) for weighted_fpgrowth.
        
    Returns:
        A_center:   Sparse matrix (N_trans, N_cells)
        A_neighbor: Sparse matrix (N_trans, N_cells)
        L_base:     Dense matrix (N_cells, N_unique_labels)
        label_names: List of column names for L
    """
    # 1. Label Matrix (Fixed Labels)
    all_labels = sorted(set(cell_types))
    if functional_subtypes is not None:
        marker_set = set()
        for sub in functional_subtypes:
            marker_set.update(sub)
        all_labels = sorted(set(all_labels).union(marker_set))
    
    label_to_idx = {l: i for i, l in enumerate(all_labels)}
    label_mat = np.zeros((num_cells, len(all_labels)), dtype=np.float32)
    for i, ct in enumerate(cell_types):
        label_mat[i, label_to_idx[ct]] = 1.0
        if functional_subtypes is not None:
            for s in functional_subtypes[i]:
                label_mat[i, label_to_idx[s]] = 1.0

    # 2. Adjacency Matrices (Fixed physical structure of the tissue)
    num_trans = len(neighborhoods) if weighted_geo is None else len(weighted_geo)
    row_c, col_c, val_c = [], [], []
    row_n, col_n, val_n = [], [], []

    if weighted_geo is not None:
        # Weighted Case (e.g. distance decay)
        for t_idx, (center_i, _, neighbor_idxs, weights) in enumerate(weighted_geo):
            row_c.append(t_idx); col_c.append(center_i); val_c.append(1.0)
            for n_idx, w in zip(neighbor_idxs, weights):
                row_n.append(t_idx); col_n.append(n_idx); val_n.append(w)
    else:
        # Binary Case (standard connectivity)
        for t_idx, item in enumerate(neighborhoods):
            if isinstance(item, tuple): # CN/KNN_R
                center_i, idxs = item
                row_c.append(t_idx); col_c.append(center_i); val_c.append(1.0)
                for n_idx in idxs:
                    if n_idx != center_i:
                        row_n.append(t_idx); col_n.append(n_idx); val_n.append(1.0)
            else: # BAG/GRID/WINDOW
                for idx in item:
                    row_c.append(t_idx); col_c.append(idx); val_c.append(1.0)

    adj_center = sparse.csr_matrix((val_c, (row_c, col_c)), shape=(num_trans, num_cells))
    adj_neighbor = sparse.csr_matrix((val_n, (row_n, col_n)), shape=(num_trans, num_cells))
    
    return adj_center, adj_neighbor, label_mat, all_labels


def run_matrix_permutation_test(rules_df, adj_center, adj_neighbor, label_mat, label_names, n_perms, config):
    """
    High-Speed Validation using Matrix Multiplication (Adjacency @ Labels).
    """
    rules_df = rules_df.copy()
    if n_perms <= 0 or rules_df.empty:
        rules_df["p_value"] = 1.0; rules_df["p_value_adj"] = 1.0
        return rules_df

    logger.info(f"Starting High-Speed Matrix Validation ({n_perms} perms, {len(rules_df)} rules)...")
    start_total = time.time()
    
    better_counts = np.zeros(len(rules_df))
    num_cells = label_mat.shape[0]
    indices = np.arange(num_cells)
    
    # Pre-build item-to-column mapping
    center_cols = {f"{name}_CENTER": i for i, name in enumerate(label_names)}
    neighbor_cols = {f"{name}_NEIGHBOR": i for i, name in enumerate(label_names)}
    direct_cols = {name: i for i, name in enumerate(label_names)}
    
    item_idx_map = {}
    for name, idx in center_cols.items(): item_idx_map[name] = idx
    for name, idx in neighbor_cols.items(): item_idx_map[name] = idx + len(label_names)
    for name, idx in direct_cols.items():
        if name not in item_idx_map: item_idx_map[name] = idx

    for i in range(n_perms):
        start_perm = time.time()
        
        # 1. SHUFFLE: Randomize labels across physical cell locations
        np.random.shuffle(indices)
        shuffled_labels = label_mat[indices, :]
        
        # 2. MULTIPLY: Rebuild transaction weights via A @ L
        center_mat = adj_center @ shuffled_labels
        neighbor_mat = adj_neighbor @ shuffled_labels
        
        if sparse.issparse(center_mat): center_mat = center_mat.toarray()
        if sparse.issparse(neighbor_mat): neighbor_mat = neighbor_mat.toarray()
        
        center_mat = np.minimum(center_mat, 1.0)
        neighbor_mat = np.minimum(neighbor_mat, 1.0)
        
        # Combined transaction matrix [Center | Neighbor]
        trans_mat = np.hstack([center_mat, neighbor_mat])
        
        # 3. VALIDATE: Update survivor counts
        better_counts += check_rules_batch(rules_df, trans_mat, item_idx_map, config).astype(float)
        
        if (i + 1) % 10 == 0 or i == 0 or i == n_perms - 1:
            logger.info(f"  > Matrix Permutation {i+1}/{n_perms} took {time.time() - start_perm:.4f}s")

    total_duration = time.time() - start_total
    logger.info(f"Matrix Validation finished. Total: {total_duration:.2f}s (avg: {total_duration/n_perms:.4f}s/perm)")

    p_values = (better_counts + 1) / (n_perms + 1)
    rules_df["p_value"] = p_values
    rules_df["p_value_adj"] = apply_fdr_correction(p_values)
    return rules_df
