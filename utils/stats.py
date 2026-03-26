import logging
import numpy as np
from statsmodels.stats.multitest import multipletests

logger = logging.getLogger(__name__)


def apply_fdr_correction(p_values):
    """Applies Benjamini-Hochberg correction. Returns adjusted p-values."""
    if len(p_values) == 0:
        return []
    _, pvals_corrected, _, _ = multipletests(p_values, method="fdr_bh")
    return pvals_corrected


def run_permutation_test(rules_df, n_items, n_perms, build_fn, check_fn=None,
                         batch_check_fn=None):
    """
    Generic permutation test for validating association rules.
    Shuffles an index array of size n_items, rebuilds transactions via build_fn,
    and checks rule survival.

    Args:
        rules_df:        DataFrame of rules to validate (must not be empty)
        n_items:         Number of items (cells) to shuffle
        n_perms:         Number of permutations (0 = skip, assign p_value=1.0)
        build_fn:        build_fn(shuffled_indices) -> transactions
                         Caller is responsible for format (sets, dicts, etc.)
        check_fn:        check_fn(rule_row, transactions) -> bool
                         Per-rule check. Used when batch_check_fn is None.
        batch_check_fn:  Optional batch_check_fn(rules_df, transactions) -> bool ndarray
                         Checks all rules at once per permutation (vectorized path).
                         When provided, check_fn is not called.

    Returns:
        rules_df with added columns: p_value, p_value_adj
    """
    rules_df = rules_df.copy()

    if n_perms <= 0:
        rules_df["p_value"] = 1.0
        rules_df["p_value_adj"] = 1.0
        return rules_df

    better_counts = np.zeros(len(rules_df))
    
    # Pre-allocate index array for shuffling
    indices = np.arange(n_items)

    for _ in range(n_perms):
        # Shuffle indices in-place
        np.random.shuffle(indices)
        perm_trans = build_fn(indices)

        if batch_check_fn is not None:
            better_counts += batch_check_fn(rules_df, perm_trans).astype(float)
        else:
            for idx, (_, row) in enumerate(rules_df.iterrows()):
                if check_fn(row, perm_trans):
                    better_counts[idx] += 1

    p_values = (better_counts + 1) / (n_perms + 1)
    rules_df["p_value"] = p_values
    rules_df["p_value_adj"] = apply_fdr_correction(p_values)

    return rules_df
