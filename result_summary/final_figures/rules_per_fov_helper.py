"""The rules behind the rules per FOV figures: every pairwise rule, and those that pass FDR with their cell types."""
import data_helper as dh
import significance_histograms_vis as sh

P_CUT = 0.05
MAX_FDR = 0.05
INCLUDE_MUSCLE = False


def load(fovs):
    """Every pairwise rule of `fovs`, without Muscle unless INCLUDE_MUSCLE."""
    rules = dh._read_rules(dh.RESULT_CSV_PATH, rule_max_items=dh.PAIRWISE_ITEMS)
    rules = rules[dh.n_items(rules).eq(dh.PAIRWISE_ITEMS) & rules["FOV"].isin(fovs)]
    if not INCLUDE_MUSCLE:
        rules = rules[~(rules["Antecedents"] + rules["Consequents"]).str.contains("Muscle")]
    return rules


def groups(rules):
    """`sh.significance` of each rule at P_CUT and MAX_FDR."""
    return sh.significance(rules, P_CUT, MAX_FDR)


def with_cells(rules):
    """`rules` with their 'center' and 'neighbor' cell types."""
    return rules.assign(
        center=lambda df: df["Antecedents"].map(lambda items: dh.base_items(items)[0]),
        neighbor=lambda df: df["Consequents"].map(lambda items: dh.base_items(items)[0]),
    )


def passed(rules):
    """The rules that pass MAX_FDR, with their 'center' and 'neighbor' cell types."""
    return with_cells(rules[groups(rules).eq(0)])
