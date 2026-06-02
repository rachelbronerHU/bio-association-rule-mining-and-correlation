import logging
import numpy as np
import pandas as pd
from collections import Counter

from typing import Optional

logger = logging.getLogger(__name__)


def _normalize_marker_label(base_type: str, subtype_label: str) -> str:
    """
    Extract marker token from a functional subtype label.
    Expected input shape: "<base_type>_<marker>+".
    """
    s = str(subtype_label)
    if not s:
        return ""

    prefix = f"{base_type}_"
    if s.startswith(prefix):
        return s[len(prefix):]

    # Fallback for unexpected formats.
    if "_" in s:
        return s.rsplit("_", 1)[-1]
    return s


def build_cell_item_token(cell_type: str, subtypes: Optional[np.ndarray], suffix: Optional[str] = None) -> str:
    """
    Build one token per physical cell.
    - No markers: "<cell_type>[_<suffix>]"
    - With markers: "<cell_type>_<sorted_unique_markers>[_<suffix>]"
    """
    base = str(cell_type)
    markers = []
    if subtypes is not None:
        for st in subtypes:
            marker = _normalize_marker_label(base, str(st))
            if marker:
                markers.append(marker)

    if markers:
        core = f"{base}_{'_'.join(sorted(set(markers)))}"
    else:
        core = base

    if suffix:
        return f"{core}_{suffix}"
    return core


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


def count_cell_types_in_fov(cell_types, functional_subtypes: Optional[np.ndarray] = None):
    """Count actual cells of each type in FOV. Returns dict {cell_type: count}."""
    counts = Counter(cell_types)
    if functional_subtypes is not None:
        for subtypes in functional_subtypes:
            for s in subtypes:
                counts[s] += 1
    return dict(counts)


def _extract_base_lineage(item):
    """
    Robustly extracts the parent lineage from a spatial item.
    Examples:
        'Neutrophil_CD15_Ki67+_NEIGHBOR' -> 'Neutrophil_CD15'
        'CD8T_Ki67+_CENTER' -> 'CD8T'
        'CD8T_CENTER' -> 'CD8T'
    """
    # 1. Strip spatial suffix
    clean = item.replace("_CENTER", "").replace("_NEIGHBOR", "")
    
    # 2. Handle functional markers (identified by '+').
    # Keep only non-marker segments, so:
    #   CD8T_CD69+_Ki67+        -> CD8T
    #   Neutrophil_CD15_CD103+  -> Neutrophil_CD15
    if "+" in clean:
        parts = clean.split("_")
        base_parts = [p for p in parts if "+" not in p]
        if base_parts:
            return "_".join(base_parts)
    return clean


def filter_rules_by_rare_cells(rules, cell_type_counts, config, method, n_cells_total):
    """
    Filter out rules where any cell type lineage is rare (below threshold).

    Threshold = max(MIN_CELL_TYPE_FREQUENCY, MIN_CELL_TYPE_PERCENTAGE * n_cells_total).
    
    Mathematical Fix: We use n_cells_total (the physical number of cells) rather 
    than summing cell_type_counts, because cell_type_counts contains markers 
    that would cause double-counting.
    """
    if rules.empty:
        return rules, 0

    min_absolute = config.get("MIN_CELL_TYPE_FREQUENCY", 5)
    min_percentage = config.get("MIN_CELL_TYPE_PERCENTAGE", 0)
    threshold = max(min_absolute, int(min_percentage * n_cells_total))

    def get_lineages_from_rule(row):
        lineages = set()
        for item in list(row["antecedents"]) + list(row["consequents"]):
            # Use shared helper for ALL methods. This ensures that even in 
            # BAG/GRID/WINDOW modes, we filter based on the parent lineage count
            # (e.g. CD8T) while allowing rare specific markers to survive.
            lineages.add(_extract_base_lineage(item))
        return lineages

    original_count = len(rules)
    mask = [
        all(cell_type_counts.get(lin, 0) >= threshold for lin in get_lineages_from_rule(row))
        for _, row in rules.iterrows()
    ]

    filtered_rules = rules[mask].copy()
    n_filtered = original_count - len(filtered_rules)

    if n_filtered > 0:
        logger.info(f"Filtered {n_filtered} rules with rare cell lineages (threshold={threshold} cells)")

    return filtered_rules, n_filtered


def _is_hierarchical_redundant(itemset, markers):
    """
    Checks if an itemset contains both a base lineage and its own subtype.
    Uses pre-parsed data to avoid string manipulation in the inner loop.
    """
    items = list(itemset)
    # n^2 check over items in a single rule (usually < 5 items)
    for i_idx in range(len(items)):
        item_i = items[i_idx]
        if item_i not in markers:
            continue
            
        b_name_i, has_plus_i, suffix_i = markers[item_i]
        
        # Only check base types (no plus)
        if has_plus_i:
            continue
            
        for j_idx in range(len(items)):
            if i_idx == j_idx:
                continue
            item_j = items[j_idx]
            if item_j not in markers:
                continue
                
            b_name_j, has_plus_j, suffix_j = markers[item_j]
            
            # REDUNDANCY CONDITION:
            # 1. Same parent lineage (e.g. CD8T)
            # 2. Item J is a subtype (has plus)
            # 3. Same spatial suffix (e.g. both are _CENTER)
            if b_name_i == b_name_j and has_plus_j and suffix_i == suffix_j:
                return True
    return False


# Legacy helper: currently not called in worker pipeline after one-token-per-cell encoding.
def remove_hierarchical_redundancy(rules, config):
    """
    Deletes rules that contain both a base lineage and its own subtype
    on the same side (antecedent or consequent).
    
    Optimization: Pre-parses all unique items in the rule set once into 
    (base_name, has_plus, suffix) tuples to avoid string manipulation 
    during rule iteration.
    """
    from constants import USE_FUNCTIONAL_MARKERS
    if not USE_FUNCTIONAL_MARKERS or rules.empty:
        return rules, 0

    # 1. Pre-parse unique items in the rule set once
    unique_items = set()
    for ant, con in zip(rules["antecedents"], rules["consequents"]):
        unique_items.update(ant)
        unique_items.update(con)
    
    # item -> (base_name, has_plus, suffix)
    markers = {}
    for item in unique_items:
        suffix = "_CENTER" if "_CENTER" in item else "_NEIGHBOR"
        has_plus = "+" in item
        # Use robust shared utility for parent extraction
        clean_base = _extract_base_lineage(item)
        markers[item] = (clean_base, has_plus, suffix)

    # 2. Apply filtering using pre-parsed markers
    # List comprehension over zipped columns is significantly faster than rules.apply(axis=1)
    mask = [
        _is_hierarchical_redundant(ant, markers) or 
        _is_hierarchical_redundant(con, markers)
        for ant, con in zip(rules["antecedents"], rules["consequents"])
    ]
    
    n_removed = int(sum(mask))
    return rules[~np.array(mask)], n_removed


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


def passes_rule_support_policy(config, support, sup_ant, sup_con, confidence, lift, num_transactions):
    """
    Confidence-aware support policy:
    - Positive rules: lower support is allowed only when confidence is high.
    - Negative rules: ignore joint support, ensure base cells meet minimum absolute frequency.
    - Absolute floor (MIN_ABS_SUPPORT) is always enforced for base cells.
    """
    if num_transactions <= 0:
        raise ValueError(f"num_transactions must be > 0, got {num_transactions}")

    standard_min_support = config["MIN_SUPPORT"]
    high_conf_threshold = config.get("HIGH_CONFIDENCE_THRESHOLD", 1.01)
    high_conf_min_support = config.get("HIGH_CONF_MIN_SUPPORT", standard_min_support)
    absolute_min_support = float(config.get("MIN_ABS_SUPPORT", 0)) / float(num_transactions)

    if lift < 1.0:
        # Negative rules represent spatial exclusion, meaning joint support will be inherently low.
        # We drop the joint support requirement entirely, but ensure the base cells (antecedent 
        # and consequent) are frequent enough in the FOV that their failure to co-occur is statistically meaningful.
        return sup_ant >= absolute_min_support and sup_con >= absolute_min_support

    required_support = high_conf_min_support if confidence >= high_conf_threshold else standard_min_support
    return support >= max(required_support, absolute_min_support)
