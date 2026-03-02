import pandas as pd
import time
import logging
import os
import csv
from constants import RARE_FILTERING_STATS_DIR, TRANSACTION_DATA_DIR, ALGO, SAVE_RAW_RULES
from utils.spatial import get_neighborhoods
from utils.rules import (
    select_top_rules,
    count_cell_types_in_fov,
    filter_rules_by_rare_cells,
    filter_redundant_rules,
)
from algos import fpgrowth as fpgrowth_algo

# Setup Worker Logger
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(message)s',
    handlers=[logging.FileHandler(os.path.join(os.getcwd(), "run_association_mining.log"))],
    force=True
)
logger = logging.getLogger("worker")


def process_single_sample(sample_id, df_sample, method, config):
    pid = os.getpid()
    prefix = f"[Worker {pid}] [Sample {sample_id}]"

    coords = df_sample[["x", "y"]].values
    cell_types = df_sample["cell_type"].values

    # 1. Get neighborhoods - ONE TIME
    t0 = time.time()
    neighborhoods = get_neighborhoods(coords, method, config)
    t_struct = time.time() - t0

    # 2. Mine (dispatches to the selected algo)
    t0 = time.time()

    if ALGO == "fpgrowth":
        mined_rules, validate_fn, stats = fpgrowth_algo.run(neighborhoods, coords, cell_types, config, method)
    else:
        raise ValueError(f"Unknown ALGO in constants.py: {ALGO!r}")

    t_mine = time.time() - t0
    n_mined = len(mined_rules)
    logger.info(f"{prefix} Mined {n_mined} rules in {t_mine:.2f}s")

    _save_transaction_cell_counts(sample_id, method, stats.get("cell_counts", []))

    # 3. Filter & Validate
    t0 = time.time()
    if SAVE_RAW_RULES:
        rules_v, raw_rules_v, n_removed, n_rare_filtered = _process_with_raw_rules(mined_rules, validate_fn, cell_types, config, method)
    else:
        rules_v, raw_rules_v, n_removed, n_rare_filtered = _process_optimized(mined_rules, validate_fn, cell_types, config, method)

    t_val = time.time() - t0
    n_val = len(rules_v)

    stats["redundant_removed"] = n_removed
    stats["rare_filtered"] = n_rare_filtered

    logger.info(f"{prefix} {method}: {stats['kept']} Trans | {n_mined} Mined | {n_rare_filtered} RareFiltered | {n_val} Validated (S:{t_struct:.2f}s M:{t_mine:.2f}s V:{t_val:.2f}s)")

    _save_rare_filtering_stats(sample_id, method, n_mined, n_rare_filtered, n_val)

    return {
        "Sample": sample_id,
        "Rules": rules_v,
        "RawRules": raw_rules_v,
        "Stats": stats,
    }


def _process_with_raw_rules(mined_rules, validate_fn, cell_types, config, method):
    """
    SAVE_RAW_RULES = True path:
    1. Filter rare cell types
    2. Select Top N candidates
    3. Validate ALL (get p-values for raw rules)
    4. Filter redundancy from validated set
    """
    n_removed = n_rare_filtered = 0
    empty = pd.DataFrame()

    if mined_rules.empty:
        return empty, empty, n_removed, n_rare_filtered

    cell_type_counts = count_cell_types_in_fov(cell_types)
    mined_rules, n_rare_filtered = filter_rules_by_rare_cells(mined_rules, cell_type_counts, config, method)

    if mined_rules.empty:
        return empty, empty, n_removed, n_rare_filtered

    candidates = select_top_rules(mined_rules, n=config["N_TOP_RULES"])
    raw_rules_v = validate_fn(candidates)
    rules_v, n_removed = filter_redundant_rules(raw_rules_v, config)

    return rules_v, raw_rules_v, n_removed, n_rare_filtered


def _process_optimized(mined_rules, validate_fn, cell_types, config, method):
    """
    SAVE_RAW_RULES = False path:
    1. Filter rare cell types
    2. Filter redundancy (pre-validation)
    3. Select Top N
    4. Validate survivors only
    """
    n_removed = n_rare_filtered = 0
    empty = pd.DataFrame()

    if mined_rules.empty:
        return empty, empty, n_removed, n_rare_filtered

    cell_type_counts = count_cell_types_in_fov(cell_types)
    mined_rules, n_rare_filtered = filter_rules_by_rare_cells(mined_rules, cell_type_counts, config, method)

    if mined_rules.empty:
        return empty, empty, n_removed, n_rare_filtered

    filtered_rules, n_removed = filter_redundant_rules(mined_rules, config)

    if len(filtered_rules) > config["N_TOP_RULES"]:
        filtered_rules = select_top_rules(filtered_rules, n=config["N_TOP_RULES"])

    rules_v = validate_fn(filtered_rules)

    return rules_v, empty, n_removed, n_rare_filtered


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
