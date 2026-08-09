"""
Turning many p-values into an answer you can trust. Nothing here runs on its own —
the caller picks the scope. See README, "Testing many rules at once".

    group_p_value           many samples from one patient -> one answer
    recurrence_p_value      passed in k of n groups -> more than chance?
    false_discovery_rates   many rules tested at once -> corrected p-values
"""

import numpy as np
from scipy.stats import binomtest
from statsmodels.stats.multitest import multipletests


def false_discovery_rates(p_values):
    """Benjamini-Hochberg: of the results you would call significant, what share are wrong."""
    p_values = np.asarray(p_values, dtype=float)
    if p_values.size == 0:
        return p_values
    return multipletests(p_values, method="fdr_bh")[1]


def group_p_value(p_values, attempts):
    """
    One answer for samples that are not independent, such as FOVs from one patient.

    The group's best p-value, charged for the number of attempts it had.
    """
    p_values = np.asarray(p_values, dtype=float)
    if attempts <= 0 or p_values.size == 0:
        return 1.0
    return float(min(1.0, attempts * p_values.min()))


def recurrence_p_value(passed, tested, alpha):
    """
    Passing in `passed` of `tested` groups — more often than chance would give?

    Reads plainly: "held in 40 of 218 groups; chance predicts 11".
    """
    if tested <= 0:
        return 1.0
    return float(binomtest(int(passed), int(tested), alpha, alternative="greater").pvalue)
