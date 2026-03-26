import pandas as pd
import numpy as np
import time
import logging
import os
import csv
from typing import Optional, Tuple
from constants import (
    RARE_FILTERING_STATS_DIR, 
    TRANSACTION_DATA_DIR, 
    ALGO, 
    SAVE_RAW_RULES,
    USE_FUNCTIONAL_MARKERS
)
from utils.spatial import get_neighborhoods
from utils.rules import (
    select_top_rules,
    count_cell_types_in_fov,
    filter_rules_by_rare_cells,
    remove_hierarchical_redundancy,
    filter_redundant_rules,
)
from algos import fpgrowth as fpgrowth_algo
from algos import weighted_fpgrowth as wfpg_algo

logger = logging.getLogger("worker")


def _apply_rule_filters(rules: pd.DataFrame, cell_type_counts: dict, config: dict, method: str, n_cells_total: int) -> Tuple[pd.DataFrame, int, int]:
    """
    Shared pipeline for filtering rules:
    1. Rare cell types (calculated against n_cells_total)
    2. Hierarchical redundancy (Apple is a Fruit)
    3. Occam's Razor redundancy
    """
    if rules.empty:
        return rules, 0, 0

    # 1. Rare Cell Filter (Mathematical Fix: use physical total)
    rules, n_rare_filtered = filter_rules_by_rare_cells(rules, cell_type_counts, config, method, n_cells_total)
    if rules.empty:
        return rules, 0, n_rare_filtered

    # 2. Hierarchical Pruning
    rules, n_hier_removed = remove_hierarchical_redundancy(rules, config)
    if rules.empty:
        return rules, n_hier_removed, n_rare_filtered

    # 3. Occam's Razor
    rules, n_occ_removed = filter_redundant_rules(rules, config)
    
    total_removed = n_hier_removed + n_occ_removed
    return rules, total_removed, n_rare_filtered


def process_single_sample(sample_id, df_sample, method, config):
    pid = os.getpid()
    prefix = f"[Worker {pid}] [Sample {sample_id}]"

    coords = df_sample[["x", "y"]].values
    cell_types = df_sample["cell_type"].values
    n_cells_total = len(cell_types) # Actual physical cell count
    
    # Optional Functional Subtypes
    functional_subtypes = None
    if "functional_subtypes" in df_sample.columns:
        functional_subtypes = df_sample["functional_subtypes"].values

    # 1. Get neighborhoods - ONE TIME
    t0 = time.time()
    neighborhoods = get_neighborhoods(coords, method, config)
    t_struct = time.time() - t0

    # 2. Mine (dispatches to the selected algo)
    t0 = time.time()

    if ALGO == "fpgrowth":
        mined_rules, validate_fn, stats = fpgrowth_algo.run(neighborhoods, coords, cell_types, config, method, functional_subtypes)
    elif ALGO == "weighted_fpgrowth":
        mined_rules, validate_fn, stats = wfpg_algo.run(neighborhoods, coords, cell_types, config, method, functional_subtypes)
    else:
        raise ValueError(f"Unknown ALGO in constants.py: {ALGO!r}")

    t_mine = time.time() - t0
    n_mined = len(mined_rules)
    logger.info(f"{prefix} Mined {n_mined} rules in {t_mine:.2f}s")

    _save_transaction_cell_counts(sample_id, method, stats.get("cell_counts", []))

    # 3. Filter & Validate
    t0 = time.time()
    cell_type_counts = count_cell_types_in_fov(cell_types, functional_subtypes)
    
    if SAVE_RAW_RULES:
        rules_v, raw_rules_v, n_rem, n_rare = _process_with_raw_rules(mined_rules, validate_fn, cell_type_counts, config, method, n_cells_total)
    else:
        rules_v, raw_rules_v, n_rem, n_rare = _process_optimized(mined_rules, validate_fn, cell_type_counts, config, method, n_cells_total)

    t_val = time.time() - t0
    n_val = len(rules_v)

    stats["redundant_removed"] = n_rem
    stats["rare_filtered"] = n_rare

    logger.info(f"{prefix} {method}: {stats['kept']} Trans | {n_mined} Mined | {n_rare} RareFiltered | {n_val} Validated (S:{t_struct:.2f}s M:{t_mine:.2f}s V:{t_val:.2f}s)")

    _save_rare_filtering_stats(sample_id, method, n_mined, n_rare, n_val)

    return {
        "Sample": sample_id,
        "Rules": rules_v,
        "RawRules": raw_rules_v,
        "Stats": stats,
    }


def _process_with_raw_rules(mined_rules, validate_fn, cell_type_counts, config, method, n_cells_total):
    """
    SAVE_RAW_RULES = True path:
    1. Select Top N candidates from ALL mined rules
    2. Validate ALL Top N (get p-values for raw rules)
    3. Apply standard filters to validated set
    """
    candidates = select_top_rules(mined_rules, n=config["N_TOP_RULES"])
    raw_rules_v = validate_fn(candidates)
    
    rules_v, n_removed, n_rare_filtered = _apply_rule_filters(raw_rules_v, cell_type_counts, config, method, n_cells_total)

    return rules_v, raw_rules_v, n_removed, n_rare_filtered


def _process_optimized(mined_rules, validate_fn, cell_type_counts, config, method, n_cells_total):
    """
    SAVE_RAW_RULES = False path:
    1. Apply filters to ALL mined rules first (unvalidated)
    2. Select Top N from survivors
    3. Validate survivors only
    """
    filtered_rules, n_removed, n_rare_filtered = _apply_rule_filters(mined_rules, cell_type_counts, config, method, n_cells_total)

    if len(filtered_rules) > config["N_TOP_RULES"]:
        filtered_rules = select_top_rules(filtered_rules, n=config["N_TOP_RULES"])

    rules_v = validate_fn(filtered_rules)

    return rules_v, pd.DataFrame(), n_removed, n_rare_filtered


def _save_rare_filtering_stats(sample_id, method, n_mined, n_rare_filtered, n_validated):
    os.makedirs(RARE_FILTERING_STATS_DIR, exist_ok=True)
    stats_file = os.path.join(RARE_FILTERING_STATS_DIR, f"{method}_rare_filtering.csv")
    file_exists = os.path.exists(stats_file)
    with open(stats_file, "a", newline="") as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(["sample_id", "n_mined", "n_rare_filtered", "n_validated", "pct_filtered"])
        pct_filtered = (n_rare_filtered / n_mined * 100) if n_mined > 0 else 0
        writer.writerow([sample_id, n_mined, n_rare_filtered, n_validated, f"{pct_filtered:.2f}"])


def _save_transaction_cell_counts(sample_id, method, cell_counts):
    if not cell_counts:
        return
    out_dir = os.path.join(TRANSACTION_DATA_DIR, method)
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"{sample_id}.csv")
    with open(out_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["transaction_idx", "num_cells"])
        for idx, count in enumerate(cell_counts):
            writer.writerow([idx, count])
