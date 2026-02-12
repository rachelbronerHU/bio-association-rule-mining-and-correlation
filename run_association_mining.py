import pandas as pd
import numpy as np
import warnings
import time
import logging
import os
import json
from concurrent.futures import ProcessPoolExecutor
from constants import DEBUG, DEBUG_FOVS_PER_GROUP, MIBI_GUT_DIR_PATH, RESULTS_DATA_DIR, SAVE_RAW_RULES
import worker_task

# Setup Logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger("manager")
warnings.filterwarnings('ignore')

# --- CONFIGURATION ---
GROUP_COL = "Pathological stage"
ID_COL = "fov"
BIOPSY_COL = "Biopsy_ID"

CONFIG = {
    "RADIUS": 30.0,
    "K_NEIGHBORS": 30,
    "GRID_WINDOW_SIZE": 30.0,
    "WINDOW_STEP_FRACTION": 0.5,
    "MIN_SUPPORT": 0.01,
    "MIN_CONFIDENCE": 0.7,
    "MIN_LIFT": 1.2,
    "MIN_LEVERAGE": 0.005,
    "MIN_CONVICTION": 1.0,
    "MIN_REDUNDANCY_LIFT_IMPROVEMENT": 1.1,
    "MAX_NEGATIVE_LIFT": 0.7,
    "MAX_RULE_LENGTH": 4,
    "TARGET_CELLS": 30,
    "MIN_CELLS_PER_PATCH": 2,
    "N_PERMUTATIONS": 5 if DEBUG else 1000,
    "N_TOP_RULES": 100 if DEBUG else 3000
}

# --- 1. DATA LOADING ---
def load_data():
    base_path = os.path.join(os.getcwd(), MIBI_GUT_DIR_PATH)
    logger.info(f"Loading data from {base_path}...")

    df_cells = pd.read_csv(f"{base_path}/cell_table.csv")
    df_fovs = pd.read_csv(f"{base_path}/fovs_metadata.csv")
    df_biopsy = pd.read_csv(f"{base_path}/biopsy_metadata.csv")

    # Check pre-merge count
    n_cells_raw = len(df_cells)

    # Merge Cells with FOVs to get Patient ID (but not Biopsy Data yet)
    df_final = pd.merge(df_cells, df_fovs, left_on="fov", right_on="FOV", how="inner")
    
    n_cells_merged = len(df_final)
    if n_cells_merged < n_cells_raw:
        logger.warning(f"Lost {n_cells_raw - n_cells_merged} cells during FOV metadata merge! (Unknown FOVs in cell table?)")
    
    # Assert no data loss (strict check)
    assert n_cells_merged == n_cells_raw, f"Lost {n_cells_raw - n_cells_merged} cells during FOV merge! Aborting to prevent data corruption."

    df_final = df_final.rename(columns={"centroid_x": "x", "centroid_y": "y", "cell type": "cell_type"})
    
    # We only need basic spatial data for mining
    req_cols = ["fov", "cell_type", "x", "y"]
    df_final = df_final.dropna(subset=req_cols)
    
    logger.info(f"Data loaded: {len(df_final)} cells.")

    df_final = _remove_unknown_biopsies(df_final)
    df_fovs = _remove_unknown_biopsies(df_fovs)

    return df_final, df_biopsy, df_fovs

def get_samples_to_process(df):
    all_samples = df[ID_COL].unique()
    if not DEBUG:
        return all_samples
    
    logger.warning(f"DEBUG MODE ON: Selecting first {DEBUG_FOVS_PER_GROUP} FOVs.")
    return all_samples[:DEBUG_FOVS_PER_GROUP]

# --- 2. OUTPUT UTILS ---

def _enrich_with_metadata(df_flat, df_biopsy, df_fovs):
    # 1. Attach Patient ID from df_fovs
    df_flat = pd.merge(df_flat, df_fovs[["FOV", "Patient"]], on="FOV", how="left")
    df_flat = df_flat.rename(columns={"Patient": "Biopsy_ID"})
    
    # 2. Dynamic Metadata Handling
    df_bio_clean = df_biopsy.copy()
    
    meta_cols = [
        "Biopsy_ID", "Cortico Response", "Survival at follow-up", 
        "GI stage", "Grade GVHD", "liver stage", "skin stage", 
        "Pathological stage", "Clinical score", "Pathological score"
    ]
    
    available_meta = [c for c in meta_cols if c in df_bio_clean.columns]
    
    numeric_cols = []
    categorical_cols = []
    
    # Identify Types and Pre-Shift Numerics in Reference Table
    for col in available_meta:
        if col == "Biopsy_ID": continue
        
        if pd.api.types.is_numeric_dtype(df_bio_clean[col]):
            # Shift 0->1, 1->2... so 0 can be Control
            df_bio_clean[col] = df_bio_clean[col] + 1
            numeric_cols.append(col)
        else:
            categorical_cols.append(col)
            
    # 3. Merge Biopsy Data
    if "Biopsy_ID" not in available_meta and "Biopsy_ID" in df_bio_clean.columns:
        available_meta.append("Biopsy_ID")
        
    df_merged = pd.merge(df_flat, df_bio_clean[available_meta], on="Biopsy_ID", how="left")

    # --- LOGGING MISSING METADATA (CONTROLS) ---
    check_col = numeric_cols[0] if numeric_cols else (categorical_cols[0] if categorical_cols else None)
    
    if check_col:
        n_missing = df_merged[check_col].isna().sum()
        if n_missing > 0:
            logger.info(f"   -> {n_missing} rules (Control/Unknown) will be imputed (Biopsy mismatch).")
            missing_examples = df_merged[df_merged[check_col].isna()]["FOV"].unique()[:5]
            logger.info(f"      Example Control FOVs: {missing_examples}")
            
            # Assertion: Ensure missing metadata ONLY happens for FOVs/BiopsyIDs containing "control"
            missing_rows = df_merged[df_merged[check_col].isna()]
            is_control = missing_rows["Biopsy_ID"].astype(str).str.contains("control", case=False) | \
                         missing_rows["FOV"].astype(str).str.contains("control", case=False)
            
            if not is_control.all():
                invalid_fovs = missing_rows[~is_control]["FOV"].unique()
                raise AssertionError(f"CRITICAL: Metadata missing for FOVs that do NOT appear to be controls (no 'control' in ID): {invalid_fovs[:10]}...")
    
    # 4. Impute Controls (NaNs)
    for col in numeric_cols:
        if col in df_merged.columns:
            df_merged[col] = df_merged[col].fillna(0)
            
    for col in categorical_cols:
        if col in df_merged.columns:
            df_merged[col] = df_merged[col].fillna("Control")
            
    # 5. Restore 'Group' column (Mapped from Pathological stage)
    if "Pathological stage" in df_merged.columns:
        df_merged["Group"] = df_merged["Pathological stage"]
        
    return df_merged

def save_results(results, df_biopsy, df_fovs, suffix, data_key="Rules"):
    logger.info(f"Saving Results ({suffix})...")
    flat_data = []
    for res in results:
        # Check if key exists and isn't empty
        if data_key not in res or res[data_key].empty: continue
        
        for _, row in res[data_key].iterrows():
            # Basic dict
            entry = {
                "FOV": res["Sample"],
                "Antecedents": str(list(row["antecedents"])),
                "Consequents": str(list(row["consequents"])),
                "Lift": row["lift"],
                "Confidence": row["confidence"],
                "Conviction": row["conviction"],
                "Support": row["support"],
            }
            # Optional fields
            if "p_value" in row: entry["P_Value"] = row["p_value"]
            if "p_value_adj" in row: entry["FDR"] = row["p_value_adj"]
            
            flat_data.append(entry)
            
    df_flat = pd.DataFrame(flat_data)
    if df_flat.empty: return
    
    # Enrich with Global Count
    rule_counts = df_flat.groupby(["Antecedents", "Consequents"])["FOV"].nunique()
    rc_df = rule_counts.reset_index(name="Rule_Count_Global")
    df_flat = pd.merge(df_flat, rc_df, on=["Antecedents", "Consequents"], how="left")
    
    # Delegate Metadata Enrichment
    df_merged = _enrich_with_metadata(df_flat, df_biopsy, df_fovs)
    
    # Dir
    out_dir = RESULTS_DATA_DIR
    os.makedirs(out_dir, exist_ok=True)
    
    filename = f"{out_dir}/results_{suffix}.csv"
    df_merged.to_csv(filename, index=False)
    logger.info(f"Saved {filename}")

def run_pipeline():
    start_time = time.time()
    df, df_biopsy, df_fovs = load_data()
    samples = get_samples_to_process(df)
    
    methods = ["BAG", "CN", "KNN_R"]
    
    # Prepare Tasks
    # We group by Method loop to save intermediate results.
    
    # Multiprocessing Pool
    # We can reuse the pool or create one per method. One per method is safer for memory cleanup
    
    for method in methods:
        logger.info(f"=== STARTING METHOD: {method} ===")
        
        # Ensure Raw Rules Dir exists
        if SAVE_RAW_RULES:
            os.makedirs(os.path.join(RESULTS_DATA_DIR, "raw_rules", method), exist_ok=True)
        
        # Prepare Inputs
        tasks = []
        for sample_id in samples:
            df_sample = df[df[ID_COL] == sample_id]
            tasks.append((sample_id, df_sample, method, CONFIG))
            
        results_collection = []
        stats_collection = {"sizes": [], "orig_counts": [], "kept_counts": [], "redundant_removed": []}
        
        with ProcessPoolExecutor() as executor:
            # Map returns iterator in order
            # Note: df_sample pickling might be slow if huge, but here it's small per FOV.
            futures = [executor.submit(worker_task.process_single_sample, *t) for t in tasks]
            
            for future in futures:
                try:
                    res = future.result()
                    if res:
                        results_collection.append(res)
                        if res["Stats"]:
                            stats_collection["sizes"].extend(res["Stats"].get("sizes", []))
                            stats_collection["orig_counts"].append(res["Stats"].get("orig", 0))
                            stats_collection["kept_counts"].append(res["Stats"].get("kept", 0))
                            stats_collection["redundant_removed"].append(res["Stats"].get("redundant_removed", 0))
                except Exception as e:
                    logger.error(f"Task Failed: {e}")

        # Save Batch (Final Rules)
        save_results(results_collection, df_biopsy, df_fovs, suffix=method, data_key="Rules")
        
        # Save Batch (Raw Rules)
        if SAVE_RAW_RULES:
            save_results(results_collection, df_biopsy, df_fovs, suffix=f"{method}_RAW", data_key="RawRules")
        
        # Save Stats
        out_dir = RESULTS_DATA_DIR
        with open(f"{out_dir}/stats_{method}.json", "w") as f:
            json.dump(stats_collection, f)
            
    print(f"Total Suite Time: {time.time() - start_time:.2f}s")

def _remove_unknown_biopsies(df_final, fov_col = "FOV"):
    # Remove "S_*" biopsies because they are not in metadata, and their condition is unknown (likely controls but not confirmed)
    pre_filter_count = len(df_final)
    df_final = df_final[~df_final[fov_col].astype(str).str.startswith("S_")]
    filtered_count = len(df_final)
    logger.info(f"Filtered out {pre_filter_count - filtered_count} cells from 'S_*' biopsies (not in metadata).")   
    return df_final

if __name__ == "__main__":
    run_pipeline()