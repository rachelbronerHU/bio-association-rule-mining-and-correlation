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


def run_permutation_test(rules_df, cell_types, n_perms, build_fn, check_fn):
    """
    Generic permutation test for validating association rules.

    Args:
        rules_df:   DataFrame of rules to validate (must not be empty)
        cell_types: Array of cell type labels for the sample
        n_perms:    Number of permutations (0 = skip, assign p_value=1.0)
        build_fn:   build_fn(shuffled_cell_types) -> transactions
                    Caller is responsible for format (sets, dicts, etc.)
        check_fn:   check_fn(rule_row, transactions) -> bool
                    Returns True if the rule would be "discovered" in this permutation

    Returns:
        rules_df with added columns: p_value, p_value_adj
    """
    rules_df = rules_df.copy()

    if n_perms <= 0:
        rules_df["p_value"] = 1.0
        rules_df["p_value_adj"] = 1.0
        return rules_df

    better_counts = np.zeros(len(rules_df))

    for _ in range(n_perms):
        shuffled_types = np.random.permutation(cell_types)
        perm_trans = build_fn(shuffled_types)

        for idx, (_, row) in enumerate(rules_df.iterrows()):
            if check_fn(row, perm_trans):
                better_counts[idx] += 1

    p_values = (better_counts + 1) / (n_perms + 1)
    rules_df["p_value"] = p_values
    rules_df["p_value_adj"] = apply_fdr_correction(p_values)

    return rules_df
