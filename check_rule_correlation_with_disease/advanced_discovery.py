import pandas as pd
import numpy as np
import os
import logging
import warnings
import time
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import Lasso, LogisticRegression
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import LeaveOneGroupOut
import xgboost as xgb
from sklearn.feature_selection import SelectKBest, f_classif, f_regression
from sklearn.metrics import f1_score, r2_score
from sklearn.feature_selection import SelectFromModel
# Silence Sklearn Warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

# --- IMPORT CONSTANTS ---
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import constants

from data_exploration.check_data_bias import load_stratified_biopsies
from check_rule_correlation_with_disease.stratified_utils import (
    CONTROLS_ELIGIBLE, TARGETS, parse_rule_items, load_and_prep_data,
    filter_viable_stratum, compute_per_class_metrics, bootstrap_single_iteration
)

# --- CONFIGURATION ---
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
BASE_INPUT_DIR = os.path.join(PROJECT_ROOT, constants.RESULTS_DATA_DIR)

FEATURE_COUNTS = [None, 20, 50, 100]
METHODS_TO_ANALYZE = ["BAG", "CN", "KNN_R"] 
N_BOOTSTRAP = 200


# Setup Logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger("Robust_Discovery")

# ── Data preparation helpers ───────────────────────────────────────────────────

def _sanitize_features(rule_lift_pivot):
    """Rename rule columns to safe names for XGBoost (which rejects special chars)."""
    rule_names = rule_lift_pivot.columns.tolist()
    safe_name_map = {rule: f"Rule_{i}" for i, rule in enumerate(rule_names)}
    rule_features = rule_lift_pivot.rename(columns=safe_name_map)
    return rule_features, rule_names

def prepare_target_data(rule_features, fov_metadata, target):
    """
    Filter to valid rows for a target, encode labels, and return arrays for LOGO CV.
    Returns (rule_features, labels, biopsy_ids, label_encoder, is_classification) or None.
    """
    valid_mask = fov_metadata[target].notna()
    n_valid = valid_mask.sum()
    if n_valid == 0:
        logger.warning(f"   [DEBUG] Target {target}: 0 valid rows (all NaN).")
        return None

    target_rule_features = rule_features[valid_mask]
    target_values = fov_metadata.loc[valid_mask, target]
    biopsy_ids = fov_metadata.loc[valid_mask, "Biopsy_ID"]
    is_classification = (target_values.dtype == 'object') or (len(target_values.unique()) < 10)
    
    logger.info(f"   [DEBUG] Target {target}: {n_valid} valid rows. Unique vals: {target_values.unique()}")

    if is_classification:
        label_encoder = LabelEncoder()
        encoded = label_encoder.fit_transform(target_values.astype(str))
        labels = pd.Series(encoded, index=target_values.index)
        
        if len(np.unique(labels)) < 2: 
            logger.warning(f"   [DEBUG] Target {target} has less than 2 classes ({np.unique(labels)}) after encoding. Skipping.")
            return None
    else:
        label_encoder = None
        labels = target_values

    return target_rule_features, labels, biopsy_ids, label_encoder, is_classification

# ── Fold computation ───────────────────────────────────────────────────────────

def _get_xgb_sample_weights(y_train):
    """Balanced sample weights for XGBoost (mirrors class_weight='balanced' in RF/Lasso)."""
    from sklearn.utils.class_weight import compute_sample_weight
    return compute_sample_weight('balanced', y_train)

def _process_fold_task(rule_features, labels, train_idx, test_idx, fold_id, task_label, is_classification, top_k):

    pid = os.getpid()
    start_t = time.time()
    log_prefix = f"[{task_label}] Fold {fold_id+1} (PID {pid}) K-Feat {top_k}-"

    X_train = rule_features.iloc[train_idx]
    
    if hasattr(labels, 'iloc'):
        y_train = labels.iloc[train_idx]
        y_test = labels.iloc[test_idx]
    else:
        y_train = labels[train_idx]
        y_test = labels[test_idx]
        
    X_test = rule_features.iloc[test_idx]

    if len(np.unique(y_train)) < 2:
        logger.warning(f"{log_prefix} Skipped (<2 classes)")
        return {"success": False, "fold_id": fold_id, "msg": "Skipped (<2 classes)"}
    
    try:
        # --- Feature Selection ---
        n_features_in = X_train.shape[1]
        
        if top_k is not None and top_k < n_features_in:
            if is_classification:
                embed_model = RandomForestClassifier(n_estimators=50, random_state=42, n_jobs=1)
            else:
                embed_model = RandomForestRegressor(n_estimators=50, random_state=42, n_jobs=1)

            selector = SelectFromModel(embed_model, max_features=top_k, threshold=-np.inf)
            selector.fit(X_train, y_train)
            
            selected_rule_indices = selector.get_support(indices=True)
            X_train_selected = selector.transform(X_train)
            X_test_selected = selector.transform(X_test)
        else:
            selected_rule_indices = np.arange(n_features_in)
            X_train_selected = X_train.values
            X_test_selected = X_test.values

        # --- Scaling (critical for Lasso) ---
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train_selected)
        X_test_scaled = scaler.transform(X_test_selected)

        # --- Model Training ---
        if is_classification:
            models = {
                "RF": RandomForestClassifier(n_estimators=50, max_depth=5, class_weight='balanced', random_state=42, n_jobs=1),
                "XGB": xgb.XGBClassifier(eval_metric='logloss', max_depth=3, random_state=42, n_jobs=1),
                "Lasso": LogisticRegression(penalty='l1', solver='saga', C=0.5, class_weight='balanced', random_state=42, max_iter=5000)
                }
        else:
            models = {
                "RF": RandomForestRegressor(n_estimators=50, max_depth=5, random_state=42, n_jobs=1),
                "XGB": xgb.XGBRegressor(max_depth=3, random_state=42, n_jobs=1),
                "Lasso": Lasso(alpha=0.01, random_state=42)
            }
       
        fold_predictions = {}
        fold_scores = {}
        
        for name, model in models.items():
            if name == "XGB":
                model.fit(X_train_scaled, y_train, sample_weight=_get_xgb_sample_weights(y_train))
            else:
                model.fit(X_train_scaled, y_train)
            pred = model.predict(X_test_scaled)
            
            fold_predictions[name] = pred[0]
            
            if is_classification:
                fold_scores[name] = f1_score(y_test, pred, average='macro')
            else:
                fold_scores[name] = r2_score(y_test, pred)

        # --- Feature Importances ---
        rf_importance = models["RF"].feature_importances_
        xgb_importance = models["XGB"].feature_importances_
        lasso_model = models["Lasso"]
        lasso_importance = np.abs(lasso_model.coef_).flatten()
        if lasso_model.coef_.ndim > 1:
            lasso_importance = np.mean(np.abs(lasso_model.coef_), axis=0)
            
        importances = {"RF": rf_importance, "XGB": xgb_importance, "Lasso": lasso_importance}

        # Normalize importances to [0, 1]
        for name in importances:
            arr = importances[name]
            if arr.max() > 0:
                importances[name] = arr / arr.max()
        
        duration = time.time() - start_t
        logger.info(f"{log_prefix} Completed in {duration:.1f}s - Scores: {fold_scores}")

        return {
            "success": True,
            "fold_id": fold_id,
            "scores": fold_scores,
            "predictions": fold_predictions,
            "true_value": y_test.iloc[0],
            "selected_rule_indices": selected_rule_indices,
            "importances": importances 
        }

    except Exception as e:
        logger.error(f"{log_prefix} FAILED: {e}")
        return {"success": False, "fold_id": fold_id, "msg": str(e)}


# ── Save / output helpers ──────────────────────────────────────────────────────

def save_predictions_log(accumulator, method, file_label, label_encoder, is_classification, output_dir):
    """Save per-biopsy prediction details."""
    safe_label = file_label.replace(" ", "_").replace("/", "-")
    pred_df = pd.DataFrame(accumulator["biopsy_predictions"])
    
    if is_classification and label_encoder:
        pred_df["True_Value_Label"] = label_encoder.inverse_transform(pred_df["True_Value"].astype(int))
        
    filename = f"{output_dir}/preds_{method}_{safe_label}.csv"
    pred_df.to_csv(filename, index=False)
    return filename

def compute_and_save_rule_stats(accumulator, n_folds_completed, method, file_label, rule_names, output_dir):
    """
    Calculate rule selection frequency (stability) and mean importance across folds.
    Returns rule_stats DataFrame for further use.
    """
    safe_label = file_label.replace(" ", "_").replace("/", "-")
    
    with np.errstate(divide='ignore', invalid='ignore'):
        avg_importances = {}
        for m in ["RF", "XGB", "Lasso"]:
            avg = accumulator["rule_imp_sums"][m] / accumulator["rule_selection_counts"]
            avg_importances[m] = np.nan_to_num(avg)

    rule_stats = pd.DataFrame({
        "Rule": rule_names,
        "Selection_Frequency": accumulator["rule_selection_counts"],
        "Selection_Percent": (accumulator["rule_selection_counts"] / n_folds_completed) * 100,
        "Mean_Imp_RF": avg_importances["RF"],
        "Mean_Imp_XGB": avg_importances["XGB"],
        "Mean_Imp_Lasso": avg_importances["Lasso"]
    })
    
    rule_stats = rule_stats[rule_stats["Selection_Frequency"] > 0].sort_values("Selection_Frequency", ascending=False)
    
    filename = f"{output_dir}/scores_{method}_{safe_label}.csv"
    rule_stats.to_csv(filename, index=False)
    return rule_stats

def save_filtered_raw_data(raw_rules, rule_stats, method, file_label, output_dir, top_n=100):
    """Save the raw rule rows for the top N most stable rules."""
    top_rules = rule_stats.head(top_n)["Rule"].tolist()
    filtered_df = raw_rules[raw_rules["Rule"].isin(top_rules)].copy()
    safe_label = file_label.replace(" ", "_").replace("/", "-")
    filename = f"{output_dir}/results_{method}_{safe_label}.csv"
    filtered_df.to_csv(filename, index=False)
    logger.info(f"   >>> SAVED FILTERED RAW: {filename}")

def calculate_leaderboard_score(accumulator, method, target, is_classification, predictions_file, top_k):
    """Calculate final aggregated scores for the leaderboard."""
    metric = "F1-Macro" if is_classification else "R2"

    grand_score = (np.mean(accumulator["model_scores"]["RF"]) +
                   np.mean(accumulator["model_scores"]["XGB"]) +
                   np.mean(accumulator["model_scores"]["Lasso"])) / 3

    return {
        "Method": method,
        "Target": target,
        "Num_Features": "All" if top_k is None else top_k,
        "Metric": metric,
        "Grand_Score": grand_score,
        "RF_Mean": np.mean(accumulator["model_scores"]["RF"]),
        "XGB_Mean": np.mean(accumulator["model_scores"]["XGB"]),
        "Lasso_Mean": np.mean(accumulator["model_scores"]["Lasso"]),
        "Log_Preds": os.path.basename(predictions_file)
    }

def save_comparison_summary(leaderboard_records, output_dir):
    """Creates a pivot table comparing scores across different feature counts."""
    if not leaderboard_records: return

    df = pd.DataFrame(leaderboard_records)
    df["Config_Label"] = df["Num_Features"].apply(lambda x: "All" if pd.isna(x) or x == "All" else f"Top{x}")

    pivot = df.pivot_table(index=["Method", "Organ", "Target"], columns="Config_Label", values="Grand_Score")

    score_columns = [col for col in pivot.columns if col in ["All"] or col.startswith("Top")]
    numeric_score_pivot = pivot[score_columns]
    pivot["Best_Config"] = numeric_score_pivot.idxmax(axis=1)
    pivot["Best_Score"] = numeric_score_pivot.max(axis=1)

    filename = f"{output_dir}/comparison_summary.csv"
    pivot.to_csv(filename)
    logger.info(f"   >>> SAVED COMPARISON SUMMARY: {filename}")


# ── Orchestration ──────────────────────────────────────────────────────────────

def _load_method_data(method, biopsy_strat, no_self):
    """
    Load and prepare data for a single method:
    - Load rules CSV, filter by FDR, pivot to rule_features matrix
    - Sanitize feature names for XGBoost
    - Attach Organ and Is_Control to fov_metadata
    - Fill NaN target values for eligible control biopsies
    Returns (rule_features, fov_metadata, raw_rules, rule_names) or None on failure.
    """
    input_file = f"{BASE_INPUT_DIR}/results_{method}.csv"
    if not os.path.exists(input_file):
        return None

    rule_lift_pivot, fov_metadata, raw_rules = load_and_prep_data(input_file, no_self=no_self)
    if rule_lift_pivot is None:
        logger.warning(f"   Skipping {method} due to data issues")
        return None

    rule_features, rule_names = _sanitize_features(rule_lift_pivot)

    # Attach organ and control flag to FOV-level metadata
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

    return rule_features, fov_metadata, raw_rules, rule_names


def _run_logo_cv(rule_features, fov_metadata, organs, method):
    """
    Run Leave-One-Group-Out CV across all organ × target × top_k configs in parallel.
    Returns (fold_accumulators, bootstrap_data).
    """
    fold_accumulators = {}
    fold_futures = {}
    bootstrap_data = {}  # (organ, target) -> (feat_arr, label_arr, group_arr, is_classification)

    with ProcessPoolExecutor() as executor:

        for organ in organs:
            organ_mask = fov_metadata["Organ"] == organ
            organ_rule_features = rule_features[organ_mask]
            organ_fov_metadata = fov_metadata[organ_mask]
            n_biopsies = organ_fov_metadata["Biopsy_ID"].nunique()
            logger.info(f"  --- Organ: {organ} ({organ_mask.sum()} FOVs, {n_biopsies} biopsies) ---")

            for target in TARGETS:
                if target not in organ_fov_metadata.columns:
                    logger.warning(f"   Target {target} not found in metadata. Skipping.")
                    continue
                if organ_fov_metadata[target].isna().all():
                    logger.warning(f"   Target {target} x {organ} has all NaN values. Skipping.")
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

                # Drop rare classes; skip only if not viable after filtering
                eligible_fov_metadata, note = filter_viable_stratum(eligible_fov_metadata, target)
                if eligible_fov_metadata is None:
                    logger.info(f"  Skipping {target} x {organ}: {note}")
                    continue
                if note:
                    logger.info(f"  {target} x {organ}: {note}")
                eligible_rule_features = eligible_rule_features.loc[eligible_fov_metadata.index]

                # Encode labels and prepare arrays for CV
                prep = prepare_target_data(eligible_rule_features, eligible_fov_metadata, target)
                if not prep:
                    continue

                target_rule_features, target_labels, biopsy_ids, label_encoder, is_classification = prep
                n_rules = target_rule_features.shape[1]

                # Store full-feature data once per (organ, target) for bootstrap CI
                bootstrap_data[(organ, target)] = (
                    target_rule_features.values if hasattr(target_rule_features, 'values') else target_rule_features,
                    target_labels.values if hasattr(target_labels, 'values') else np.array(target_labels),
                    biopsy_ids.values if hasattr(biopsy_ids, 'values') else np.array(biopsy_ids),
                    is_classification
                )

                for top_k in FEATURE_COUNTS:
                    config_key = (organ, target, top_k)

                    fold_accumulators[config_key] = {
                        "meta": {"label_encoder": label_encoder, "is_classification": is_classification, "n_folds_completed": 0},
                        "rule_selection_counts": np.zeros(n_rules),
                        "rule_imp_sums": {"RF": np.zeros(n_rules), "XGB": np.zeros(n_rules), "Lasso": np.zeros(n_rules)},
                        "biopsy_predictions": [],
                        "model_scores": {"RF": [], "XGB": [], "Lasso": []}
                    }

                    logo = LeaveOneGroupOut()
                    for fold_idx, (train_idx, test_idx) in enumerate(logo.split(target_rule_features, target_labels, groups=biopsy_ids)):
                        features_label = f"Top{top_k}" if top_k else "All"
                        task_label = f"{method}-{organ}-{target}-{features_label}"
                        test_biopsy_id = biopsy_ids.iloc[test_idx[0]]
                        fold_key = (organ, target, top_k, fold_idx, test_biopsy_id)

                        future = executor.submit(
                            _process_fold_task, target_rule_features, target_labels,
                            train_idx, test_idx, fold_idx, task_label, is_classification, top_k
                        )
                        fold_futures[future] = fold_key

        # --- Collect results as they complete ---
        total_tasks = len(fold_futures)
        logger.info(f"   >>> Processing {total_tasks} tasks (Organ x Target x Config x Folds)...")

        completed_count = 0
        for future in as_completed(fold_futures):
            organ_name, target_name, top_k, fold_id, biopsy_id = fold_futures[future]
            res = future.result()

            completed_count += 1
            if completed_count % 100 == 0:
                logger.info(f"   ... Completed {completed_count}/{total_tasks} tasks")

            if res["success"]:
                accumulator = fold_accumulators[(organ_name, target_name, top_k)]
                accumulator["meta"]["n_folds_completed"] += 1

                for model_name, score in res["scores"].items():
                    accumulator["model_scores"][model_name].append(score)

                selected_rule_indices = res["selected_rule_indices"]
                accumulator["rule_selection_counts"][selected_rule_indices] += 1
                for model_name in ["RF", "XGB", "Lasso"]:
                    accumulator["rule_imp_sums"][model_name][selected_rule_indices] += res["importances"][model_name]

                fold_prediction = {
                    "Biopsy_ID": biopsy_id,
                    "True_Value": res["true_value"],
                    "Pred_RF": res["predictions"]["RF"],
                    "Pred_XGB": res["predictions"]["XGB"],
                    "Pred_Lasso": res["predictions"]["Lasso"]
                }
                accumulator["biopsy_predictions"].append(fold_prediction)

    return fold_accumulators, bootstrap_data


def _save_method_results(fold_accumulators, raw_rules, rule_names, method, output_dir):
    """
    Save artifacts for all (organ, target, top_k) configs of one method.
    Returns list of leaderboard records for this method.
    """
    logger.info(f"   >>> Saving results for {method}...")
    method_leaderboard_records = []

    for (organ, target, top_k), accumulator in fold_accumulators.items():
        n_folds_completed = accumulator["meta"]["n_folds_completed"]
        if n_folds_completed == 0:
            continue

        label_encoder = accumulator["meta"]["label_encoder"]
        is_classification = accumulator["meta"]["is_classification"]

        features_suffix = f"_Top{top_k}" if top_k else "_All"
        file_label = f"{organ}_{target}{features_suffix}"

        predictions_file = save_predictions_log(accumulator, method, file_label, label_encoder, is_classification, output_dir)
        rule_stats = compute_and_save_rule_stats(accumulator, n_folds_completed, method, file_label, rule_names, output_dir)
        save_filtered_raw_data(raw_rules, rule_stats, method, file_label, output_dir)

        record = calculate_leaderboard_score(accumulator, method, target, is_classification, predictions_file, top_k)
        record["Organ"] = organ

        if is_classification and accumulator["biopsy_predictions"]:
            true_labels = np.array([r["True_Value"] for r in accumulator["biopsy_predictions"]], dtype=int)
            record["Per_Class_F1_RF"] = str(compute_per_class_metrics(true_labels, np.array([r["Pred_RF"] for r in accumulator["biopsy_predictions"]], dtype=int), label_encoder))
            record["Per_Class_F1_XGB"] = str(compute_per_class_metrics(true_labels, np.array([r["Pred_XGB"] for r in accumulator["biopsy_predictions"]], dtype=int), label_encoder))
            record["Per_Class_F1_Lasso"] = str(compute_per_class_metrics(true_labels, np.array([r["Pred_Lasso"] for r in accumulator["biopsy_predictions"]], dtype=int), label_encoder))

        method_leaderboard_records.append(record)
        logger.info(f"Saved {organ} x {target} [{features_suffix.strip('_')}]: Score = {record['Grand_Score']:.3f}")

    return method_leaderboard_records


def _attach_bootstrap_ci(bootstrap_data, method_leaderboard_records, method):
    """
    Run bootstrap CI in parallel across all (organ, target) combos and attach CI columns
    to method_leaderboard_records in place.
    """
    if not bootstrap_data:
        return

    bootstrap_tasks = []
    bootstrap_combo_index = []
    for (organ, target), (feat_arr, label_arr, group_arr, is_classification) in bootstrap_data.items():
        unique_biopsies = np.unique(group_arr)
        for seed in range(N_BOOTSTRAP):
            bootstrap_tasks.append((feat_arr, label_arr, group_arr, unique_biopsies, is_classification, seed))
            bootstrap_combo_index.append((organ, target))

    logger.info(f"   >>> Running bootstrap CI ({N_BOOTSTRAP} iters x {len(bootstrap_data)} combos in parallel)...")
    with ProcessPoolExecutor() as executor:
        bootstrap_scores = list(executor.map(bootstrap_single_iteration, bootstrap_tasks))

    ci_scores_per_combo = {}
    for (organ, target), score in zip(bootstrap_combo_index, bootstrap_scores):
        key = (organ, target)
        if key not in ci_scores_per_combo:
            ci_scores_per_combo[key] = []
        if not np.isnan(score):
            ci_scores_per_combo[key].append(score)

    for record in method_leaderboard_records:
        combo_scores = ci_scores_per_combo.get((record["Organ"], record["Target"]), [])
        if len(combo_scores) >= 10:
            record["CI_Mean"] = round(float(np.mean(combo_scores)), 4)
            record["CI_Lower"] = round(float(np.percentile(combo_scores, 2.5)), 4)
            record["CI_Upper"] = round(float(np.percentile(combo_scores, 97.5)), 4)
        else:
            record["CI_Mean"] = record["CI_Lower"] = record["CI_Upper"] = np.nan


def run_pipeline(no_self=False):
    """
    Main orchestrator. Iterates methods, runs LOGO CV per organ×target, saves artifacts and leaderboard.
    """
    if not os.path.exists(os.path.abspath(BASE_INPUT_DIR)):
        logger.error("CRITICAL: Input directory missing.")
        return

    if no_self:
        output_dir = os.path.join(PROJECT_ROOT, constants.RESULTS_ML_DATA_DIR_NO_SELF)
        logger.info(f"Running in NO_SELF mode. Output: {output_dir}")
    else:
        output_dir = os.path.join(PROJECT_ROOT, constants.RESULTS_ML_DATA_DIR)
        logger.info(f"Running in STANDARD mode. Output: {output_dir}")

    os.makedirs(output_dir, exist_ok=True)

    stratification = load_stratified_biopsies()
    organs = sorted([o for o in stratification["Organ"].unique() if o != "Unknown"])
    biopsy_strat = stratification.set_index("Biopsy_ID")

    leaderboard_records = []

    for method in METHODS_TO_ANALYZE:
        logger.info(f"=== METHOD: {method} ===")
        method_data = _load_method_data(method, biopsy_strat, no_self)
        if method_data is None:
            continue

        rule_features, fov_metadata, raw_rules, rule_names = method_data

        fold_accumulators, bootstrap_data = _run_logo_cv(rule_features, fov_metadata, organs, method)
        method_records = _save_method_results(fold_accumulators, raw_rules, rule_names, method, output_dir)
        _attach_bootstrap_ci(bootstrap_data, method_records, method)
        leaderboard_records.extend(method_records)

    if leaderboard_records:
        lb_df = pd.DataFrame(leaderboard_records).sort_values(["Organ", "Target", "Grand_Score"], ascending=False)
        lb_path = f"{output_dir}/final_leaderboard.csv"
        lb_df.to_csv(lb_path, index=False)
        logger.info(f"DONE. Leaderboard saved to {lb_path}")

        save_comparison_summary(leaderboard_records, output_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Advanced Discovery (ML) Benchmark")
    parser.add_argument("--no_self", action="store_true", help="Exclude rules with any shared cell types (strict no-self).")
    args = parser.parse_args()
    
    run_pipeline(no_self=args.no_self)
