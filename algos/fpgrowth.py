import logging
import numpy as np
import pandas as pd
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import fpgrowth, association_rules

from utils.spatial import MIN_CELLS, is_dominated
from utils.rules import filter_rule_by_scores_mask
from utils.validation import prepare_validation_matrices, run_matrix_permutation_test

from typing import Optional

logger = logging.getLogger(__name__)


def run(neighborhoods, coords, cell_types, config, method, functional_subtypes: Optional[np.ndarray] = None):
    """
    Entry point for standard binary FP-Growth.
    """
    trans, stats = _build_transactions(neighborhoods, cell_types, method, config, functional_subtypes)
    mined_rules = _mine(trans, config, method)

    # PRE-COMPUTE ONCE PER FOV:
    # Capture the physical structure and labels before returning the hook
    adj_center, adj_neighbor, label_mat, label_names = prepare_validation_matrices(
        neighborhoods, len(cell_types), cell_types, functional_subtypes
    )
    n_perms = config["N_PERMUTATIONS"]

    def validate_fn(rules_df):
        if rules_df.empty:
            return rules_df
        return run_matrix_permutation_test(
            rules_df, adj_center, adj_neighbor, label_mat, label_names, n_perms, config
        )

    return mined_rules, validate_fn, stats


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _build_transactions(neighborhoods, cell_types, method, config, functional_subtypes: Optional[np.ndarray] = None):
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

            # Center logic - use set to avoid duplicates immediately
            trans = {f"{cell_types[center_i]}_CENTER"}
            if functional_subtypes is not None:
                for s in functional_subtypes[center_i]:
                    trans.add(f"{s}_CENTER")

            # Neighbors logic
            for n in idxs:
                if n == center_i:
                    continue
                n_type = cell_types[n]
                trans.add(f"{n_type}_NEIGHBOR")
                if functional_subtypes is not None:
                    for s in functional_subtypes[n]:
                        trans.add(f"{s}_NEIGHBOR")
            
            trans_list = list(trans)
        else:
            idxs = item
            if len(idxs) < min_cells:
                continue
            raw_types = cell_types[idxs]
            if is_dominated(raw_types):
                continue
            
            trans = set()
            for idx in idxs:
                trans.add(cell_types[idx])
                if functional_subtypes is not None:
                    for s in functional_subtypes[idx]:
                        trans.add(s)
            
            trans_list = list(trans)

        transactions.append(trans_list)
        sizes.append(len(trans_list))
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
    trans_mat = te.fit(transactions).transform(transactions)
    df_trans = pd.DataFrame(trans_mat, columns=te.columns_)

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


# ---------------------------------------------------------------------------
# (End of file)
