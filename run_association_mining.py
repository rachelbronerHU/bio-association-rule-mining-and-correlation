import logging
import os
import shutil
import time
import warnings

import numpy as np
import pandas as pd

from constants import (
    DEBUG,
    DEBUG_FOVS_PER_GROUP,
    LABELS_KEPT_FIXED,
    METHOD,
    MIBI_GUT_DIR_PATH,
    MIN_LIFT_GAIN,
    N_SHUFFLES,
    RANDOM_SEED,
    RESULTS_ALGO_DIR,
    RESULTS_DATA_DIR,
    SETTINGS,
    WEIGHTING,
    WORKERS,
)
from fpgrowth_rule_mining import run_samples

logger = logging.getLogger("manager")
warnings.filterwarnings('ignore')

# --- CONFIGURATION ---
ID_COL = "fov"

# --- 1. DATA LOADING ---

def _normalize_coordinates(df):
    """
    Normalizes x/y coordinates to microns using fixed resolution standards.
    Standard: 400um = 1024px, 800um = 2048px.
    """
    if "Size [um]" not in df.columns:
        logger.warning("Size [um] column missing. Skipping coordinate normalization.")
        return df

    # Scale factors based on known resolution:
    # 400um / 1024px = 0.390625 um/px
    # 800um / 2048px = 0.390625 um/px
    
    # Default scale (fallback)
    default_scale = 400.0 / 1024.0

    conditions = [
        df["Size [um]"] == 400,
        df["Size [um]"] == 800
    ]
    
    choices = [
        400.0 / 1024.0,
        800.0 / 2048.0
    ]
    
    scale_factors = np.select(conditions, choices, default=default_scale)
    
    df["x"] = df["x"] * scale_factors
    df["y"] = df["y"] * scale_factors
    
    logger.info("Coordinates normalized to microns using Fixed Resolution Standards (1024px/400um, 2048px/800um).")
    return df

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
        # detailed check
        fovs_cells = set(df_cells["fov"].unique())
        fovs_meta = set(df_fovs["FOV"].unique())
        missing_fovs = fovs_cells - fovs_meta
        logger.warning(f"Lost {n_cells_raw - n_cells_merged} cells during FOV metadata merge!")
        logger.warning(f"The following {len(missing_fovs)} FOVs are in cell_table but missing from fovs_metadata: {sorted(list(missing_fovs))}")
    
    # Assert no data loss (strict check)
    assert n_cells_merged == n_cells_raw, f"Lost {n_cells_raw - n_cells_merged} cells during FOV merge! See log for missing FOVs. Aborting."

    df_final = df_final.rename(columns={"centroid_x": "x", "centroid_y": "y", "cell type": "cell_type"})
    
    df_final = _normalize_coordinates(df_final)

    # We only need basic spatial data for mining
    req_cols = ["fov", "cell_type", "x", "y"]
    df_final = df_final.dropna(subset=req_cols)
    
    logger.info(f"Data loaded: {len(df_final)} cells.")

    return df_final, df_biopsy, df_fovs

def get_samples_to_process(df):
    all_samples = df[ID_COL].unique()
    if not DEBUG:
        return all_samples
    
    logger.warning(f"DEBUG MODE ON: Selecting first {DEBUG_FOVS_PER_GROUP} FOVs.")
    return all_samples[:DEBUG_FOVS_PER_GROUP]

# --- 2. OUTPUT UTILS ---

def _enrich_with_metadata(df_flat, df_biopsy, df_fovs):
    # 1. Attach Patient ID and Cohort from df_fovs
    df_flat = pd.merge(df_flat, df_fovs[["FOV", "Patient", "Cohort"]], on="FOV", how="left")
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
            
            # Assertion: Ensure missing metadata ONLY happens for FOVs/BiopsyIDs/Cohorts containing "control"
            missing_rows = df_merged[df_merged[check_col].isna()]
            is_control = missing_rows["Biopsy_ID"].astype(str).str.contains("control", case=False) | \
                         missing_rows["FOV"].astype(str).str.contains("control", case=False) | \
                         missing_rows["Cohort"].astype(str).str.contains("control", case=False)
            
            if not is_control.all():
                invalid_fovs = missing_rows[~is_control]["FOV"].unique()
                raise AssertionError(f"CRITICAL: Metadata missing for FOVs that do NOT appear to be controls (no 'control' in ID or Cohort): {invalid_fovs[:10]}...")
    
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

def _as_text(items):
    return str([str(item) for item in items])


def save_results(rules, df_biopsy, df_fovs, suffix):
    """Flatten one frame of rules to the results CSV, joined to the biopsy metadata."""
    logger.info(f"Saving Results ({suffix})...")
    if rules.empty:
        return

    df_flat = pd.DataFrame({
        "FOV": rules["sample_id"],
        "Antecedents": rules["antecedents"].apply(_as_text),
        "Consequents": rules["consequents"].apply(_as_text),
        "Kind": rules["kind"],          # "attracts" or "avoids"
        "Lift": rules["lift"],
        "Leverage": rules["leverage"],
        "Confidence": rules["confidence"],
        "Conviction": rules["conviction"],
        "Support": rules["support"],
        "Rule_Type": rules["rule_type"],
        "Complex_Class": rules["complex_class"],
        "Simpler_Rules": rules["simpler_rules"].apply(_as_text),
    })
    if "p_value" in rules.columns:
        df_flat["P_Value"] = rules["p_value"]

    # No count of how many FOVs a rule appeared in: an uncorrected count sitting next
    # to p-values gets read as evidence. dataset_significance_*.csv answers that.

    # Delegate Metadata Enrichment
    df_merged = _enrich_with_metadata(df_flat, df_biopsy, df_fovs)

    filename = f"{RESULTS_DATA_DIR}/results_{suffix}.csv"
    df_merged.to_csv(filename, index=False)
    logger.info(f"Saved {filename}")

def _clear_previous_run(results_dir):
    """A re-run replaces its own directory rather than mixing old and new results."""
    if os.path.exists(results_dir):
        shutil.rmtree(results_dir)
    os.makedirs(results_dir, exist_ok=True)


def run_pipeline():
    _clear_previous_run(RESULTS_ALGO_DIR)
    # Made once, here: a run that finds no rules still has files to write.
    os.makedirs(RESULTS_DATA_DIR, exist_ok=True)
    # Log to console and to a file in the results directory
    logging.basicConfig(
        level=logging.INFO, 
        format="%(asctime)s %(message)s", 
        datefmt="%H:%M:%S",
        handlers=[
            logging.FileHandler(f"{RESULTS_ALGO_DIR}/run.log"),
            logging.StreamHandler()
        ]
    )

    logger.info("===================================================================================")
    logger.info(f"========================= MINING: {WEIGHTING} / {METHOD} =========================")
    logger.info("===================================================================================")

    start_time = time.time()

    df, df_biopsy, df_fovs = load_data()
    chosen = set(get_samples_to_process(df))
    samples = [
        (fov, sub[["x", "y"]].values, sub["cell_type"].values)
        for fov, sub in df.groupby(ID_COL) if fov in chosen
    ]

    report = run_samples(
        samples, SETTINGS,
        n_shuffles=N_SHUFFLES,
        random_seed=RANDOM_SEED,
        labels_kept_fixed=LABELS_KEPT_FIXED,
        min_lift_gain=MIN_LIFT_GAIN,
        workers=WORKERS,
        output_path=RESULTS_ALGO_DIR,
    )

    save_results(report.rules(), df_biopsy, df_fovs, suffix=METHOD)

    # FOVs from one patient are not independent evidence, so they vote together.
    # The library never learns what a patient is: it only compares these values.
    groups = df_fovs.drop_duplicates("FOV").set_index("FOV")["Patient"].to_dict()
    across = report.dataset_significance(groups=groups)
    across.to_csv(f"{RESULTS_DATA_DIR}/dataset_significance_{METHOD}.csv", index=False)
    logger.info(f"Saved dataset_significance_{METHOD}.csv "
                f"({len(across)} rules, {(across['dataset_fdr'] <= 0.05).sum()} at FDR<=0.05)")

    elapsed = time.time() - start_time
    h, rem = divmod(int(elapsed), 3600)
    m, s = divmod(rem, 60)
    logger.info(f"Total Suite Time: {h}h {m}m {s}s")

if __name__ == "__main__":
    run_pipeline()
