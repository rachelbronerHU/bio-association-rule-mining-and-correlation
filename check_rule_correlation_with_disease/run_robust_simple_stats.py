import pandas as pd
import numpy as np
import os
import time
import logging
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.preprocessing import LabelEncoder
from scipy.stats import ttest_ind, f_oneway
from sklearn.neighbors import NearestCentroid
import warnings

warnings.filterwarnings('ignore')

# --- IMPORT CONSTANTS ---
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import constants
from utils.logging_setup import setup_logging
from data_exploration.check_data_bias import load_stratified_biopsies
from check_rule_correlation_with_disease.stratified_utils import (
    CONTROLS_ELIGIBLE, TARGETS, parse_rule_items, load_and_prep_data,
    filter_viable_stratum, compute_per_class_metrics, run_bootstrap_ci
)

# --- CONFIGURATION ---
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
BASE_INPUT_DIR = os.path.join(PROJECT_ROOT, constants.RESULTS_DATA_DIR)

FEATURE_COUNTS = [None, 20, 50, 100]
METHODS = constants.METHODS

logger = logging.getLogger("SimpleStats")

# ── Fold computation ───────────────────────────────────────────────────────────

def _get_top_rules_stats(X_train, y_train, top_n, is_multiclass):
    """
    Selects Top N rules using T-test (Binary) or ANOVA (Multiclass).
    If top_n is None or exceeds available rules, returns all valid rules.
    """
    valid_cols = [c for c in X_train.columns if X_train[c].var() > 0]
    
    if top_n is None or top_n >= len(valid_cols):
        return valid_cols # Return all valid rules if no specific top_n or top_n is too large
        
    scores = []
    
    if is_multiclass:
        # ANOVA F-Test
        groups = [X_train[y_train == c][valid_cols] for c in np.unique(y_train)]
        for col in valid_cols:
            group_vals = [g[col] for g in groups]
            f_stat, p = f_oneway(*group_vals)
            if np.isnan(p): p = 1.0
            scores.append((col, p))
    else:
        # T-Test (Binary)
        classes = np.unique(y_train)
        g0 = X_train[y_train == classes[0]]
        g1 = X_train[y_train == classes[1]]
        for col in valid_cols:
            t, p = ttest_ind(g0[col], g1[col], equal_var=False)
            if np.isnan(p): p = 1.0
            scores.append((col, p))
            
    # Sort by P-Value (Ascending)
    scores.sort(key=lambda x: x[1])
    return [x[0] for x in scores[:top_n]]

def _predict_simple(X_train, y_train, X_test, top_rules, is_multiclass):
    """
    Predicts using simple statistical methods.
    Binary: Voting (High value -> Vote for class with High Mean).
    Multiclass: Nearest Centroid (Assign to class with closest mean profile).
    """
    X_train_sel = X_train[top_rules]
    X_test_sel = X_test[top_rules]
    
    if is_multiclass:
        # Nearest Centroid Classifier (Euclidean distance to class means)
        clf = NearestCentroid()
        clf.fit(X_train_sel, y_train)
        pred = clf.predict(X_test_sel)[0] # Single prediction usually
        # Probability? Centroid doesn't give probs nicely.
        # We'll return dummy prob 1.0 for the predicted class.
        prob = 1.0 
    else:
        # Binary Voting Logic
        classes = np.unique(y_train)
        # Identify "Positive" class (usually the second one in sort)
        pos_class = classes[1]
        
        votes = 0
        for col in top_rules:
            mean0 = X_train_sel[y_train == classes[0]][col].mean()
            mean1 = X_train_sel[y_train == classes[1]][col].mean()
            midpoint = (mean0 + mean1) / 2
            
            val = X_test_sel[col].iloc[0] # Assume single test biopsy (aggregated) or handle per-FOV
            
            # If mean1 > mean0, then High Value => Pos Class
            # If mean0 > mean1, then Low Value => Pos Class
            is_high = val > midpoint
            
            if mean1 > mean0:
                if is_high: votes += 1
            else:
                if not is_high: votes += 1
                
        prob = votes / len(top_rules)
        pred = pos_class if prob >= 0.5 else classes[0]
        
    return pred, prob

def _process_single_fold(X, y_enc, train_idx, test_idx, fold_id, task_label, is_multiclass, k_feat):
    try:
        X_train = X.iloc[train_idx]
        X_test = X.iloc[test_idx]
        
        # Handle Y (Series/Array)
        if hasattr(y_enc, 'iloc'):
            y_train = y_enc.iloc[train_idx]
            y_test_val = y_enc.iloc[test_idx].iloc[0]
        else:
            y_train = y_enc[train_idx]
            y_test_val = y_enc[test_idx][0]
            
        # 1. Selection
        top_rules = _get_top_rules_stats(X_train, y_train, k_feat, is_multiclass)
        
        # 2. Prediction
        pred, prob = _predict_simple(X_train, y_train, X_test, top_rules, is_multiclass)
        
        return {
            "success": True,
            "fold_id": fold_id,
            "true_val": y_test_val,
            "pred_val": pred,
            "prob": prob,
            "selected_rules": top_rules
        }
    except Exception as e:
        logger.error(f"{task_label} Fold {fold_id+1} - Error: {str(e)}")
        return {"success": False, "msg": str(e)}

# ── Save / output helpers ──────────────────────────────────────────────────────

def _save_comparison_summary_simple(leaderboard_data, output_dir):
    """
    Creates a pivot table comparing scores across different feature counts for simple stats.
    """
    if not leaderboard_data: return
    
    df = pd.DataFrame(leaderboard_data)
    df["Config_Label"] = df["Num_Features"].apply(lambda x: "All" if pd.isna(x) or x == "All" else f"Top{x}")
    
    pivot = df.pivot_table(index=["Method", "Organ", "Target"], columns="Config_Label", values="Accuracy")
    
    score_columns = [col for col in pivot.columns if col in ["All"] or col.startswith("Top")]
    numeric_score_pivot = pivot[score_columns]

    pivot["Best_Config"] = numeric_score_pivot.idxmax(axis=1)
    pivot["Best_Score"] = numeric_score_pivot.max(axis=1)
    
    filename = f"{output_dir}/comparison_summary_simple.csv"
    pivot.to_csv(filename)
    logger.info(f"   >>> SAVED COMPARISON SUMMARY (Simple): {filename}")


# ── Orchestration ──────────────────────────────────────────────────────────────

def _load_method_data(method, biopsy_strat, no_self):
    """
    Load and prepare data for a single method:
    - Load rules CSV, filter by FDR, pivot to rule_lift_pivot matrix
    - Attach Organ and Is_Control to fov_metadata
    - Fill NaN target values for eligible control biopsies
    Returns (rule_lift_pivot, fov_metadata, raw_rules) or None on failure.
    """
    input_file = f"{BASE_INPUT_DIR}/results_{method}.csv"
    if not os.path.exists(input_file):
        return None

    rule_lift_pivot, fov_metadata, raw_rules = load_and_prep_data(input_file, no_self=no_self)
    if rule_lift_pivot is None:
        logger.warning(f"   Skipping {method} due to data issues")
        return None

    fov_metadata["Organ"] = fov_metadata["Biopsy_ID"].map(biopsy_strat["Organ"])
    fov_metadata["Is_Control"] = fov_metadata["Biopsy_ID"].map(biopsy_strat["Is_Control"])

    # Fill NaN target values for controls — only for eligible targets
    is_control_fov = fov_metadata["Is_Control"].fillna(False).astype(bool)
    for target in TARGETS:
        if not CONTROLS_ELIGIBLE.get(target, False):
            continue
        if target not in fov_metadata.columns or target not in biopsy_strat.columns:
            continue
        missing_target_for_controls = is_control_fov & fov_metadata[target].isna()
        if missing_target_for_controls.any():
            fov_metadata.loc[missing_target_for_controls, target] = (
                fov_metadata.loc[missing_target_for_controls, "Biopsy_ID"].map(biopsy_strat[target])
            )

    return rule_lift_pivot, fov_metadata, raw_rules

def _run_logo_cv(rule_lift_pivot, fov_metadata, organs, method):
    """
    Run Leave-One-Group-Out CV across all organ × target × top_k configs in parallel.
    Returns (fold_accumulators, bootstrap_data).
    """
    fold_accumulators = {}
    fold_futures = {}
    bootstrap_data = {}  # (organ, target) -> (feat_arr, label_arr, group_arr, is_multiclass)

    with ProcessPoolExecutor() as executor:
        for organ in organs:
            organ_mask = fov_metadata["Organ"] == organ
            organ_rule_features = rule_lift_pivot[organ_mask]
            organ_fov_metadata = fov_metadata[organ_mask]
            n_biopsies = organ_fov_metadata["Biopsy_ID"].nunique()
            logger.info(f"  --- Organ: {organ} ({organ_mask.sum()} FOVs, {n_biopsies} biopsies) ---")

            for target in TARGETS:
                if target not in organ_fov_metadata.columns:
                    continue
                if organ_fov_metadata[target].isna().all():
                    continue

                # Filter to eligible population (GVHD only vs GVHD + controls)
                include_controls = CONTROLS_ELIGIBLE.get(target, True)
                if not include_controls:
                    gvhd_only_mask = ~organ_fov_metadata["Is_Control"].fillna(False).astype(bool)
                    eligible_rule_features = organ_rule_features[gvhd_only_mask]
                    eligible_fov_metadata = organ_fov_metadata[gvhd_only_mask]
                else:
                    eligible_rule_features = organ_rule_features
                    eligible_fov_metadata = organ_fov_metadata

                # Drop rare classes; skip only if still not viable after filtering
                eligible_fov_metadata, note = filter_viable_stratum(eligible_fov_metadata, target)
                if eligible_fov_metadata is None:
                    logger.info(f"  Skipping {target} x {organ}: {note}")
                    continue
                if note:
                    logger.info(f"  {target} x {organ}: {note}")
                eligible_rule_features = eligible_rule_features.loc[eligible_fov_metadata.index]

                # Encode labels
                valid_mask = eligible_fov_metadata[target].notna()
                target_rule_features = eligible_rule_features[valid_mask]
                target_labels_raw = eligible_fov_metadata.loc[valid_mask, target]
                biopsy_ids = eligible_fov_metadata.loc[valid_mask, "Biopsy_ID"]

                label_encoder = LabelEncoder()
                target_labels = pd.Series(
                    label_encoder.fit_transform(target_labels_raw.astype(str)),
                    index=target_labels_raw.index
                )

                if len(np.unique(target_labels)) < 2:
                    logger.warning(f"   Target {target} x {organ} has less than 2 classes. Skipping.")
                    continue
                is_multiclass = len(np.unique(target_labels)) > 2

                # Store full-feature data once per (organ, target) for bootstrap CI
                bootstrap_data[(organ, target)] = (
                    target_rule_features.values,
                    target_labels.values,
                    biopsy_ids.values,
                    is_multiclass
                )

                for top_k in FEATURE_COUNTS:
                    config_key = (organ, target, top_k)
                    fold_accumulators[config_key] = {
                        "biopsy_results": [],
                        "rule_selection_counts": {},
                        "label_encoder": label_encoder
                    }

                    logo = LeaveOneGroupOut()
                    for fold_idx, (train_idx, test_idx) in enumerate(logo.split(target_rule_features, target_labels, biopsy_ids)):
                        features_label = f"Top{top_k}" if top_k else "All"
                        task_label = f"{method}-{organ}-{target}-{features_label}"
                        test_biopsy_id = biopsy_ids.iloc[test_idx[0]]

                        future = executor.submit(
                            _process_single_fold,
                            target_rule_features, target_labels, train_idx, test_idx,
                            fold_idx, task_label, is_multiclass, top_k
                        )
                        fold_futures[future] = (organ, target, top_k, test_biopsy_id, label_encoder)

        # Collect results as they complete
        total_tasks = len(fold_futures)
        logger.info(f"   >>> Processing {total_tasks} tasks (Organ x Target x Config x Folds)...")

        for completed_count, future in enumerate(as_completed(fold_futures), 1):
            if completed_count % 100 == 0:
                logger.info(f"   ... Completed {completed_count}/{total_tasks} tasks")

            organ, target, top_k, biopsy_id, label_encoder = fold_futures[future]
            res = future.result()

            if res["success"]:
                accumulator = fold_accumulators[(organ, target, top_k)]
                true_label = label_encoder.inverse_transform([int(res["true_val"])])[0]
                pred_label = label_encoder.inverse_transform([int(res["pred_val"])])[0]

                accumulator["biopsy_results"].append({
                    "Biopsy_ID": biopsy_id,
                    "True_Value": true_label,
                    "Pred_Label": pred_label,
                    "Prob_Pos": res["prob"]
                })
                for rule in res["selected_rules"]:
                    accumulator["rule_selection_counts"][rule] = accumulator["rule_selection_counts"].get(rule, 0) + 1

    return fold_accumulators, bootstrap_data

def _save_method_results(fold_accumulators, method, output_dir):
    """
    Save prediction logs and rule-selection stats CSVs for all (organ, target, top_k) configs.
    Returns list of leaderboard records for this method.
    """
    logger.info(f"   >>> Saving results for {method}...")
    leaderboard_records = []

    for (organ, target, top_k), accumulator in fold_accumulators.items():
        if not accumulator["biopsy_results"]:
            continue

        df_predictions = pd.DataFrame(accumulator["biopsy_results"])
        config_suffix = f"_Top{top_k}" if top_k else "_All"
        file_label = f"{organ}_{target}{config_suffix}".replace(' ', '_')

        preds_path = f"{output_dir}/preds_SIMPLE_{method}_{file_label}.csv"
        df_predictions.to_csv(preds_path, index=False)

        rule_counts = accumulator["rule_selection_counts"]
        df_rules = pd.DataFrame(list(rule_counts.items()), columns=["Rule", "Count"])
        df_rules["Frequency"] = df_rules["Count"] / len(accumulator["biopsy_results"])
        df_rules = df_rules.sort_values("Count", ascending=False)
        rules_path = f"{output_dir}/scores_SIMPLE_{method}_{file_label}.csv"
        df_rules.to_csv(rules_path, index=False)

        accuracy = (df_predictions["True_Value"] == df_predictions["Pred_Label"]).mean()
        per_class_f1 = compute_per_class_metrics(
            df_predictions["True_Value"].values,
            df_predictions["Pred_Label"].values
        )
        leaderboard_records.append({
            "Method": method,
            "Organ": organ,
            "Target": target,
            "Num_Features": "All" if top_k is None else top_k,
            "Type": "Simple_Stats",
            "Accuracy": accuracy,
            "Per_Class_F1": str(per_class_f1)
        })
        logger.info(f"Saved {organ} x {target} [{config_suffix.strip('_')}]: Acc={accuracy:.2%}")

    return leaderboard_records

def _attach_bootstrap_ci(bootstrap_data, method_leaderboard_records, method):
    """
    Run bootstrap CI in parallel across all (organ, target) combos and attach CI columns
    to method_leaderboard_records in place.
    """
    run_bootstrap_ci(bootstrap_data, method_leaderboard_records)


def run_pipeline(no_self=False):
    if not os.path.exists(BASE_INPUT_DIR):
        logger.error("CRITICAL: Input directory missing.")
        return

    if no_self:
        output_dir = os.path.join(PROJECT_ROOT, constants.RESULTS_SIMPLE_STATS_DATA_DIR_NO_SELF)
        logger.info(f"Running in NO_SELF mode. Output: {output_dir}")
    else:
        output_dir = os.path.join(PROJECT_ROOT, constants.RESULTS_SIMPLE_STATS_DATA_DIR)
        logger.info(f"Running in STANDARD mode. Output: {output_dir}")

    os.makedirs(output_dir, exist_ok=True)

    biopsy_strat = load_stratified_biopsies().set_index("Biopsy_ID")
    organs = sorted([o for o in biopsy_strat["Organ"].unique() if o != "Unknown"])

    all_leaderboard_records = []

    for method in METHODS:
        logger.info(f"=== METHOD: {method} ===")
        loaded = _load_method_data(method, biopsy_strat, no_self)
        if loaded is None:
            continue
        rule_lift_pivot, fov_metadata, _ = loaded

        fold_accumulators, bootstrap_data = _run_logo_cv(rule_lift_pivot, fov_metadata, organs, method)

        method_records = _save_method_results(fold_accumulators, method, output_dir)
        _attach_bootstrap_ci(bootstrap_data, method_records, method)
        all_leaderboard_records.extend(method_records)

    if all_leaderboard_records:
        lb_df = pd.DataFrame(all_leaderboard_records).sort_values(["Organ", "Target", "Accuracy"], ascending=False)
        lb_path = f"{output_dir}/leaderboard_simple.csv"
        lb_df.to_csv(lb_path, index=False)
        logger.info(f"DONE. Leaderboard (Simple) saved to {lb_path}")
        _save_comparison_summary_simple(all_leaderboard_records, output_dir)

    print("\nBenchmark Complete.")


if __name__ == "__main__":
    setup_logging(f"run_robust_simple_stats_{constants.ALGO}")
    parser = argparse.ArgumentParser(description="Run Robust Simple Stats Benchmark")
    parser.add_argument("--no_self", action="store_true", help="Exclude rules with any shared cell types (strict no-self).")
    args = parser.parse_args()

    run_pipeline(no_self=args.no_self)
