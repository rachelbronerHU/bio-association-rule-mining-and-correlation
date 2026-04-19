import os

import pandas as pd

from constants import MIBI_GUT_DIR_PATH, RESULTS_DATA_DIR
from visualization.utils.visualization_util import filter_no_self_rules, merge_biopsy_metadata, parse_rule_list


def add_subset_args(parser):
    """Add shared subset-filter arguments used by rule-based visualization scripts."""
    parser.add_argument(
        "--subset_rule_items_eq",
        type=int,
        default=None,
        help="Keep only rules with exactly this many total items (antecedent count + consequent count).",
    )
    parser.add_argument(
        "--subset_min_support",
        type=float,
        default=None,
        help="Keep only rules with Support >= this value.",
    )
    return parser


def get_subset_tag(subset_rule_items_eq=None, subset_min_support=None):
    """Return a stable folder tag for subset runs, or None when no subset is requested."""
    tag_parts = []

    if subset_rule_items_eq is not None:
        tag_parts.append(f"items_{int(subset_rule_items_eq)}")
    if subset_min_support is not None:
        tag_parts.append(f"minsup_{_format_tag_number(subset_min_support)}")
    if not tag_parts:
        return None
    return "subset_" + "_".join(tag_parts)


def load_rule_results(
    method,
    raw=False,
    merge_metadata=False,
    no_self=False,
    organ=None,
    subset_rule_items_eq=None,
    subset_min_support=None,
):
    """
    Load one rule-results table and optionally apply metadata merge, no-self filter, organ split, and subset filters.
    Returns None when the file does not exist.
    """
    file_suffix = "_RAW" if raw else ""
    file_path = os.path.join(RESULTS_DATA_DIR, f"results_{method}{file_suffix}.csv")
    if not os.path.exists(file_path):
        return None

    data_frame = pd.read_csv(file_path)
    if "Rule" not in data_frame.columns and {"Antecedents", "Consequents"}.issubset(data_frame.columns):
        data_frame["Rule"] = data_frame["Antecedents"] + " -> " + data_frame["Consequents"]

    if no_self:
        data_frame = filter_no_self_rules(data_frame)

    if merge_metadata:
        data_frame = merge_biopsy_metadata(data_frame, MIBI_GUT_DIR_PATH)

    if organ is not None and "Organ" in data_frame.columns:
        data_frame = data_frame[data_frame["Organ"] == organ].copy()

    data_frame = _apply_subset_filters(
        data_frame,
        subset_rule_items_eq=subset_rule_items_eq,
        subset_min_support=subset_min_support,
    )
    return data_frame


def _apply_subset_filters(data_frame, subset_rule_items_eq=None, subset_min_support=None):
    """Apply all optional subset filters to a rule-results dataframe."""
    filtered = data_frame

    if subset_rule_items_eq is not None:
        if not {"Antecedents", "Consequents"}.issubset(filtered.columns):
            raise ValueError("Rule item-count filtering needs Antecedents and Consequents columns.")
        item_counts = filtered.apply(
            lambda row: _count_rule_items(row["Antecedents"], row["Consequents"]),
            axis=1,
        )
        filtered = filtered[item_counts == int(subset_rule_items_eq)]

    if subset_min_support is not None:
        if "Support" not in filtered.columns:
            raise ValueError("Support filtering needs a Support column.")
        filtered = filtered[filtered["Support"] >= float(subset_min_support)]

    return filtered.copy()


def _count_rule_items(antecedents, consequents):
    """Count total cleaned rule items as antecedent count + consequent count."""
    antecedent_items = parse_rule_list(antecedents)
    consequent_items = parse_rule_list(consequents)
    return len(antecedent_items) + len(consequent_items)


def _format_tag_number(value):
    """Format float values safely for folder names."""
    return str(value).replace("-", "m").replace(".", "p")
