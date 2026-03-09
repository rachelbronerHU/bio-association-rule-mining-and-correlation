import logging
import numpy as np
import pandas as pd
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import fpgrowth, association_rules

from utils.spatial import MIN_CELLS, is_dominated
from utils.rules import filter_rule_by_scores_mask
from utils.stats import run_permutation_test

logger = logging.getLogger(__name__)


def run(neighborhoods, coords, cell_types, config, method):
    """
    Entry point for standard binary FP-Growth.

    Args:
        neighborhoods: Output of utils.spatial.get_neighborhoods()
        coords:        (N, 2) array of cell coordinates (unused here, kept for uniform signature)
        cell_types:    (N,) array of cell type labels
        config:        Pipeline config dict
        method:        Spatial method name (BAG, CN, KNN_R, WINDOW, GRID)

    Returns:
        mined_rules: DataFrame of association rules (may be empty)
        validate_fn: Callable validate_fn(rules_df) -> rules_df with p_value columns
        stats:       Dict with transaction statistics (sizes, cell_counts, orig, kept)
    """
    trans, stats = _build_transactions(neighborhoods, cell_types, method, config)
    mined_rules = _mine(trans, config, method)

    def validate_fn(rules_df):
        return _validate_rules(neighborhoods, cell_types, rules_df, config, method)

    return mined_rules, validate_fn, stats


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _build_transactions(neighborhoods, cell_types, method, config):
    """
    Converts neighborhoods into binary transactions (list of item lists).

    Returns:
        transactions: List[List[str]]
        stats:        Dict with sizes, cell_counts, orig, kept
    """
    min_cells = config.get("MIN_CELLS_PER_PATCH", MIN_CELLS)
    transactions = []
    sizes = []
    cell_counts = []
    orig_count = len(neighborhoods)

    for item in neighborhoods:
        if method in ["CN", "KNN_R"]:
            center_i, idxs = item
            if len(idxs) < min_cells:
                continue
            raw_types = cell_types[idxs]
            if is_dominated(raw_types):
                continue
            center = f"{cell_types[center_i]}_CENTER"
            neighbors = [f"{cell_types[n]}_NEIGHBOR" for n in idxs if n != center_i]
            trans = [center] + list(set(neighbors))
        else:
            idxs = item
            if len(idxs) < min_cells:
                continue
            raw_types = cell_types[idxs]
            if is_dominated(raw_types):
                continue
            trans = list(set(raw_types))

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


def _mine(transactions, config, method):
    """Runs mlxtend FP-Growth and returns filtered association rules DataFrame."""
    if not transactions:
        return pd.DataFrame()

    te = TransactionEncoder()
    te_ary = te.fit(transactions).transform(transactions)
    df_trans = pd.DataFrame(te_ary, columns=te.columns_)

    frequent_itemsets = fpgrowth(df_trans, min_support=config["MIN_SUPPORT"], use_colnames=True)
    if frequent_itemsets.empty:
        return pd.DataFrame()

    rules = association_rules(frequent_itemsets, metric="support", min_threshold=config["MIN_SUPPORT"])

    if "conviction" not in rules.columns:
        with np.errstate(divide="ignore", invalid="ignore"):
            rules["conviction"] = (1 - rules["consequent support"]) / (1 - rules["confidence"])
            rules["conviction"] = rules["conviction"].fillna(np.inf)

    is_valid = filter_rule_by_scores_mask(config, rules["lift"], rules["leverage"], rules["conviction"], rules["confidence"])
    rules = rules[is_valid]

    if rules.empty:
        return pd.DataFrame()

    rules["len_ant"] = rules["antecedents"].apply(len)
    rules["len_con"] = rules["consequents"].apply(len)
    rules = rules[(rules["len_ant"] + rules["len_con"]) <= config["MAX_RULE_LENGTH"]]

    if method in ["CN", "KNN_R"]:
        rules["has_center_ant"] = rules["antecedents"].apply(lambda x: any("_CENTER" in item for item in x))
        rules = rules[rules["has_center_ant"]].drop(columns=["has_center_ant"])

    if rules.empty:
        return pd.DataFrame()

    rules["antecedents"] = rules["antecedents"].apply(lambda x: tuple(sorted(list(x))))
    rules["consequents"] = rules["consequents"].apply(lambda x: tuple(sorted(list(x))))

    return rules


def _check_rule(rule_row, transaction_sets, config):
    """
    Checks if a rule passes thresholds in a given set of transactions.

    Args:
        rule_row:         A row from the rules DataFrame
        transaction_sets: List[set] — pre-converted for fast membership checks
        config:           Pipeline config dict
    """
    if not transaction_sets:
        return False

    N = len(transaction_sets)
    antecedents = set(rule_row["antecedents"])
    consequents = set(rule_row["consequents"])

    count_ant = count_cons = count_both = 0
    for t_set in transaction_sets:
        has_ant = antecedents.issubset(t_set)
        has_cons = consequents.issubset(t_set)
        if has_ant:
            count_ant += 1
        if has_cons:
            count_cons += 1
        if has_ant and has_cons:
            count_both += 1

    support_both = count_both / N
    support_ant = count_ant / N
    support_cons = count_cons / N

    if support_both < config["MIN_SUPPORT"]:
        return False
    if support_ant == 0 or support_cons == 0:
        return False

    confidence = support_both / support_ant
    lift = confidence / support_cons
    leverage = support_both - (support_ant * support_cons)
    conviction = float("inf") if confidence >= 1.0 else (1 - support_cons) / (1 - confidence)

    return filter_rule_by_scores_mask(config, lift, leverage, conviction, confidence)


def _validate_rules(neighborhoods, cell_types, rules_df, config, method):
    """
    Validates rules via permutation test.
    Shuffles cell type labels, rebuilds binary transactions, checks if rules survive.
    """
    if rules_df.empty:
        return rules_df

    n_perms = config["N_PERMUTATIONS"]

    def build_fn(shuffled_types):
        trans, _ = _build_transactions(neighborhoods, shuffled_types, method, config)
        return [set(t) for t in trans]

    def check_fn(rule_row, transaction_sets):
        return _check_rule(rule_row, transaction_sets, config)

    return run_permutation_test(rules_df, cell_types, n_perms, build_fn, check_fn)
