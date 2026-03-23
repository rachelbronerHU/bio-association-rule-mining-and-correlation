import logging
import numpy as np
import pandas as pd
from collections import Counter

logger = logging.getLogger(__name__)


def select_top_rules(rules, n=2000):
    if len(rules) <= n:
        return rules

    pos = rules[rules["lift"] >= 1].sort_values("lift", ascending=False)
    neg = rules[rules["lift"] < 1].sort_values("lift", ascending=True)

    n_pos, n_neg = len(pos), len(neg)
    half = n // 2

    if n_pos < half:
        take_pos = n_pos
        take_neg = min(n_neg, n - n_pos)
    elif n_neg < half:
        take_neg = n_neg
        take_pos = min(n_pos, n - n_neg)
    else:
        take_pos = half
        take_neg = n - half

    return pd.concat([pos.head(take_pos), neg.head(take_neg)]).drop_duplicates()


def count_cell_types_in_fov(cell_types):
    """Count actual cells of each type in FOV. Returns dict {cell_type: count}."""
    return dict(Counter(cell_types))


def filter_rules_by_rare_cells(rules, cell_type_counts, config, method):
    """
    Filter out rules where any cell type is rare (below threshold).

    Threshold = max(MIN_CELL_TYPE_FREQUENCY, MIN_CELL_TYPE_PERCENTAGE * total_cells).
    MIN_CELL_TYPE_FREQUENCY: absolute cell count floor (default 5).
    MIN_CELL_TYPE_PERCENTAGE: fraction of all cells (0–1, default 0 = disabled).

    Returns:
        filtered_rules: DataFrame of rules without rare cell types
        n_filtered: Number of rules removed
    """
    if rules.empty:
        return rules, 0

    n_cells = sum(cell_type_counts.values())
    min_absolute = config.get("MIN_CELL_TYPE_FREQUENCY", 5)
    # MIN_CELL_TYPE_PERCENTAGE is a fraction of total cells (0–1), distinct from
    # MIN_SUPPORT which is a fraction of transactions. Defaults to 0 (disabled).
    min_percentage = config.get("MIN_CELL_TYPE_PERCENTAGE", 0)
    threshold = max(min_absolute, int(min_percentage * n_cells))

    def get_cell_types_from_rule(row):
        cell_types = set()
        for item in list(row["antecedents"]) + list(row["consequents"]):
            if method in ["CN", "KNN_R"]:
                cell_type = item.replace("_CENTER", "").replace("_NEIGHBOR", "")
            else:
                cell_type = item
            cell_types.add(cell_type)
        return cell_types

    original_count = len(rules)
    mask = [
        all(cell_type_counts.get(ct, 0) >= threshold for ct in get_cell_types_from_rule(row))
        for _, row in rules.iterrows()
    ]

    filtered_rules = rules[mask].copy()
    n_filtered = original_count - len(filtered_rules)

    if n_filtered > 0:
        logger.info(f"Filtered {n_filtered} rules with rare cell types (threshold={threshold} cells)")

    return filtered_rules, n_filtered


def filter_redundant_rules(rules, config):
    """
    Filters redundant rules based on Occam's Razor.
    A rule is redundant if a simpler rule (subset antecedent) exists with similar or better lift.
    Reference: https://www.bayardo.org/ps/icde99.pdf
    """
    if rules.empty:
        return rules, 0

    rules = rules.copy()
    threshold_factor = config.get("MIN_REDUNDANCY_LIFT_IMPROVEMENT", 1.1)
    indices_to_drop = set()

    rules["ant_frozen"] = rules["antecedents"].apply(frozenset)
    grouped = rules.groupby("consequents")

    for _, group in grouped:
        sorted_group = group.sort_values(by="len_ant", ascending=True)
        rows = list(sorted_group.itertuples())

        for i in range(len(rows)):
            r_complex = rows[i]
            if r_complex.Index in indices_to_drop:
                continue
            for j in range(i):
                r_simple = rows[j]
                if r_simple.ant_frozen < r_complex.ant_frozen:
                    is_neg_simple = r_simple.lift < 1.0
                    is_neg_complex = r_complex.lift < 1.0
                    if is_neg_simple != is_neg_complex:
                        # Positive and negative rules are not comparable
                        continue
                    if is_neg_complex:
                        # Negative rules: lower lift = stronger anti-association.
                        # Complex is redundant if it is not sufficiently more negative.
                        if r_complex.lift >= r_simple.lift / threshold_factor:
                            indices_to_drop.add(r_complex.Index)
                            break
                    else:
                        # Positive rules: higher lift = stronger association.
                        # Complex is redundant if it is not sufficiently higher.
                        if r_complex.lift <= r_simple.lift * threshold_factor:
                            indices_to_drop.add(r_complex.Index)
                            break

    count_removed = len(indices_to_drop)
    filtered_rules = rules.drop(index=list(indices_to_drop)).drop(columns=["ant_frozen"])

    return filtered_rules, count_removed


def filter_rule_by_scores_mask(config, lift, leverage, conviction, confidence):
    is_positive = (lift >= config["MIN_LIFT"]) & (leverage >= config["MIN_LEVERAGE"]) & (conviction >= config["MIN_CONVICTION"]) & (confidence >= config["MIN_CONFIDENCE"])
    is_negative = (lift <= config["MAX_NEGATIVE_LIFT"]) & (leverage <= config["MAX_NEGATIVE_LEVERAGE"])
    
    return is_positive | is_negative
