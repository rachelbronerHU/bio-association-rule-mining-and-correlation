import pandas as pd
import numpy as np
import os
import logging
import warnings
import time
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

# --- CONFIGURATION ---
BASE_INPUT_DIR = "experiment_results/full_run/data"
OUTPUT_DATA_DIR = "experiment_results/ml_refined_robust_benchmarks/data"
TOP_N_RULES = 50
METHODS_TO_ANALYZE = ["BAG", "CN", "KNN_R"] 
TARGETS = [
    "Pathological stage", 
    "GI stage", 
    # "liver stage", 
    # "skin stage", 
    "Cortico Response", 
    "Grade GVHD",
    "Survival at follow-up",
    "Clinical score",
    "Pathological score"
]


# Setup Logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger("Robust_Discovery")

# --- DATA PREP ---

def load_and_prep_data(input_file):
    if not os.path.exists(input_file):
        logger.warning(f"File not found: {input_file}")
        return None, None, None

    df = pd.read_csv(input_file)
    
    # Filter by P-Value
    initial_len = len(df)
    if "P_Value" in df.columns:
        df = df[df["P_Value"] < 0.05]
        dropped = initial_len - len(df)
        if dropped > 0:
            logger.info(f"   Dropped {dropped} rows with P-Value >= 0.05 (Retained: {len(df)})")
    
    if df.empty:
        logger.warning("   No data left after P-Value filtering.")
        return None, None, None

    if "Rule" not in df.columns:
        df["Rule"] = df["Antecedents"] + " -> " + df["Consequents"]

    # Pivot
    pivot_df = df.pivot_table(index="FOV", columns="Rule", values="Lift", aggfunc="mean").fillna(0)
    
    available_targets = [t for t in TARGETS if t in df.columns]
    meta_cols = ["FOV", "Biopsy_ID"] + available_targets
    meta_df = df[meta_cols].drop_duplicates(subset="FOV").set_index("FOV")
    
    common = pivot_df.index.intersection(meta_df.index)
    return pivot_df.loc[common], meta_df.loc[common], df

def sanitize_features(X):
    original_rules = X.columns.tolist()
    rule_map = {rule: f"Rule_{i}" for i, rule in enumerate(original_rules)}
    reverse_map = {v: k for k, v in rule_map.items()}
    X_renamed = X.rename(columns=rule_map)
    # Return the ordered list of names, which corresponds exactly to Rule_0, Rule_1...
    return X_renamed, original_rules

# --- CORE LOGIC ---

def process_fold_task(X, y_enc, train_idx, test_idx, fold_id, task_label, is_categorical):


    pid = os.getpid()
    start_t = time.time()
    log_prefix = f"[{task_label}] Fold {fold_id+1} (PID {pid}) -"

    X_train = X.iloc[train_idx]
    
    # Handle Series vs Array slicing
    if hasattr(y_enc, 'iloc'):
        y_train = y_enc.iloc[train_idx]
        y_test = y_enc.iloc[test_idx]
    else:
        y_train = y_enc[train_idx]
        y_test = y_enc[test_idx]
        
    X_test = X.iloc[test_idx]

    if len(np.unique(y_train)) < 2:
        logger.warning(f"{log_prefix} Skipped (<2 classes)")
        return {"success": False, "fold_id": fold_id, "msg": "Skipped (<2 classes)"}
    
    try:

        # --- 2. Feature Selection ---
        k_best = min(TOP_N_RULES, X_train.shape[1])
        
        if is_categorical:
            embed_model = RandomForestClassifier(n_estimators=50, random_state=42, n_jobs=1)
        else:
            embed_model = RandomForestClassifier(n_estimators=50, random_state=42, n_jobs=1)

        selector = SelectFromModel(embed_model, max_features=TOP_N_RULES, threshold=-np.inf)
        selector.fit(X_train, y_train)
        
        selected_indices = selector.get_support(indices=True)

        X_train_sel = selector.transform(X_train)
        X_test_sel = selector.transform(X_test)

        # --- Scaling (Critical for Lasso) ---
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train_sel)
        X_test_scaled = scaler.transform(X_test_sel)

        # --- 3. Model Training ---
        if is_categorical:
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
       
        fold_preds = {}
        fold_scores = {}
        
        for name, model in models.items():
            model.fit(X_train_scaled, y_train)       # Training the model with selected features
            pred = model.predict(X_test_scaled)      # Predicting on the test set
            
            fold_preds[name] = pred[0]            # Save prediction (single fov)
            
            # Score Calculation
            if is_categorical:
                fold_scores[name] = f1_score(y_test, pred, average='macro')
            else:
                fold_scores[name] = r2_score(y_test, pred)

        # 1. RF Importance
        imp_rf = models["RF"].feature_importances_
        
        # 2. XGB Importance
        imp_xgb = models["XGB"].feature_importances_
        
        # 3. Lasso Coefficients 
        logit = models["Lasso"]
        imp_lasso = np.abs(logit.coef_).flatten()
        if logit.coef_.ndim > 1:
            imp_lasso = np.mean(np.abs(logit.coef_), axis=0)
            
        
        importances_dict = {"RF": imp_rf, "XGB": imp_xgb, "Lasso": imp_lasso}

        # Normalize Importances
        for name in importances_dict:
            arr = importances_dict[name]
            if arr.max() > 0:
                importances_dict[name] = arr / arr.max()
        
        duration = time.time() - start_t
        logger.info(f"{log_prefix} Completed in {duration:.1f}s - Scores: {fold_scores}")

        return {
            "success": True,
            "fold_id": fold_id,
            "scores": fold_scores,
            "predictions": fold_preds,
            "true_value": y_test.iloc[0],
            "selected_indices": selected_indices,
            "importances": importances_dict 
        }

    except Exception as e:
        logger.error(f"{log_prefix} FAILED: {e}")
        return {"success": False, "fold_id": fold_id, "msg": str(e)}


# =========== HELPER FUNCTIONS ===========

def prepare_target_data(X_safe, y_meta_raw, target):
    """
    Handles data filtering, NaN removal, and Label Encoding for a specific target.
    Returns clean X, y, groups (Biopsy_ID), encoder, and categorical flag.
    """
    valid_mask = y_meta_raw[target].notna()
    n_valid = valid_mask.sum()
    if n_valid == 0:
        logger.warning(f"   [DEBUG] Target {target}: 0 valid rows (all NaN).")
        return None

    # Per FOV
    X_target = X_safe[valid_mask]
    y_target = y_meta_raw.loc[valid_mask, target]

    # Per Biopsy Groups
    groups_target = y_meta_raw.loc[valid_mask, "Biopsy_ID"]
    is_categorical = (y_target.dtype == 'object') or (len(y_target.unique()) < 10)
    
    unique_vals = y_target.unique()
    logger.info(f"   [DEBUG] Target {target}: {n_valid} valid rows. Unique vals: {unique_vals}")

    if is_categorical:
        le = LabelEncoder()
        y_enc = le.fit_transform(y_target.astype(str))
        y_final = pd.Series(y_enc, index=y_target.index)
        
        if len(np.unique(y_final)) < 2: 
            logger.warning(f"   [DEBUG] Target {target} has less than 2 classes ({np.unique(y_final)}) after encoding. Skipping.")
            return None
    else:
        y_final = y_target

    return X_target, y_final, groups_target, le, is_categorical

def save_predictions_log(acc, method, target, le, is_cat):
    """
    Step 1: Save the per-patient prediction details.
    """
    safe_target = target.replace(" ", "_").replace("/", "-")
    pred_df = pd.DataFrame(acc["patient_results"])
    
    # If categorical, map numbers (0,1) back to names (Stage 1, Stage 2)
    if is_cat and le:
        pred_df["True_Value_Label"] = le.inverse_transform(pred_df["True_Value"].astype(int))
        
    filename = f"{OUTPUT_DATA_DIR}/preds_{method}_{safe_target}.csv"
    pred_df.to_csv(filename, index=False)
    return filename

def compute_and_save_rule_stats(acc, count, method, target, original_rule_names):
    """
    Step 2: Calculate Stability (Frequency) and Mean Importance.
    Returns the Rules DataFrame for further use.
    """
    safe_target = target.replace(" ", "_").replace("/", "-")
    
    # Calculate Mean Importance (Sum / Count) safely
    avg_imps = {}
    with np.errstate(divide='ignore', invalid='ignore'):
        for m in ["RF", "XGB", "Lasso"]:
            # We divide sum of importance by the number of times it was SELECTED (not total folds)
            avg = acc["rule_imp_sums"][m] / acc["rule_counts"]
            avg_imps[m] = np.nan_to_num(avg)

    # Create the DataFrame
    rules_df = pd.DataFrame({
        "Rule": original_rule_names,
        "Selection_Frequency": acc["rule_counts"], # The "Stability" Metric (Raw Count)
        "Selection_Percent": (acc["rule_counts"] / count) * 100, # The "Stability" Metric (%)
        "Mean_Imp_RF": avg_imps["RF"],
        "Mean_Imp_XGB": avg_imps["XGB"],
        "Mean_Imp_Lasso": avg_imps["Lasso"]
    })
    
    # Filter: Keep only rules selected at least once
    rules_df = rules_df[rules_df["Selection_Frequency"] > 0].sort_values("Selection_Frequency", ascending=False)
    
    # Save
    filename = f"{OUTPUT_DATA_DIR}/scores_{method}_{safe_target}.csv"
    rules_df.to_csv(filename, index=False)
    
    return rules_df

def save_raw_data_subset(df_raw, rules_df, method, target):
    """
    Step 3: Save the raw data rows for the Top 100 most stable rules.
    """
    # Get the names of the top 100 rules (sorted by stability)
    top_stable_rules = rules_df.head(100)["Rule"].tolist()
    
    # Call the existing helper to filter and save
    save_filtered_raw_data(df_raw, top_stable_rules, method, target)

def save_filtered_raw_data(raw_df, top_rules, method_name, target_name):
    filtered_df = raw_df[raw_df["Rule"].isin(top_rules)].copy()
    safe_target = target_name.replace(" ", "_").replace("/", "-")
    filename = f"{OUTPUT_DATA_DIR}/results_{method_name}_{safe_target}.csv"
    filtered_df.to_csv(filename, index=False)
    logger.info(f"   >>> SAVED FILTERED RAW: {filename}")

def calculate_leaderboard_score(acc, method, target, is_cat, preds_filename):
    """
    Step 4: Calculate final aggregated scores for the Leaderboard.
    """
    metric = "F1-Macro" if is_cat else "R2"
    
    # Grand Score = Average of the 3 models
    grand_score = (np.mean(acc["agg_scores"]["RF"]) + 
                   np.mean(acc["agg_scores"]["XGB"]) + 
                   np.mean(acc["agg_scores"]["Lasso"])) / 3
    
    return {
        "Method": method,
        "Target": target,
        "Metric": metric,
        "Grand_Score": grand_score,
        "RF_Mean": np.mean(acc["agg_scores"]["RF"]),
        "XGB_Mean": np.mean(acc["agg_scores"]["XGB"]),
        "Lasso_Mean": np.mean(acc["agg_scores"]["Lasso"]),
        "Log_Preds": os.path.basename(preds_filename)
    }


# --- ORCHESTRATOR ---

def run_pipeline():
    """
    Main Orchestrator function.
    Iterates over methods and targets, runs the LOGO validation,
    and saves detailed artifacts (Predictions, Scores, Raw Data, Leaderboard).
    """
    abs_base_input = os.path.abspath(BASE_INPUT_DIR)
    logger.info(f"Input Dir: {abs_base_input}")
    if not os.path.exists(abs_base_input):
        logger.error("CRITICAL: Input directory missing.")
        return
    os.makedirs(OUTPUT_DATA_DIR, exist_ok=True)

    leaderboard_data = []

    for method in METHODS_TO_ANALYZE:
        input_file = f"{BASE_INPUT_DIR}/results_{method}.csv"
        if not os.path.exists(input_file): continue
            
        logger.info(f"=== METHOD: {method} === input file: {input_file}")
        
        X_raw, y_meta_raw, df_raw = load_and_prep_data(input_file)
        if X_raw is None: 
            logger.warning(f"   Skipping {method} due to data issues - X_raw is None")
            continue

        X_safe, original_rule_names = sanitize_features(X_raw)

        target_accumulators = {} 
        futures_map = {} # To track which result belongs to which target

        with ProcessPoolExecutor() as executor:
            
            for target in TARGETS:
                if target not in y_meta_raw.columns: 
                    logger.warning(f"   Target {target} not found in metadata. Skipping.")
                    continue

                if y_meta_raw[target].isna().all(): 
                    logger.warning(f"   Target {target} has all NaN values. Skipping.")
                    continue
                
                # --- Step A: Prepare Target Data ---
                # Handles NaN removal, Label Encoding, and alignment
                prep_res = prepare_target_data(X_safe, y_meta_raw, target)
                if not prep_res: continue # Skip if insufficient data
                
                # Unpack prepared data
                X_t, y_t, groups_t, le, is_cat = prep_res

                # --- Step B: Execute Cross Validation ---
                n_features = X_t.shape[1]
                target_accumulators[target] = {
                    "meta": {"le": le, "is_cat": is_cat, "count": 0}, # Metadata needed for saving
                    "rule_counts": np.zeros(n_features),
                    "rule_imp_sums": {"RF": np.zeros(n_features), "XGB": np.zeros(n_features), "Lasso": np.zeros(n_features)},
                    "patient_results": [],
                    "agg_scores": {"RF": [], "XGB": [], "Lasso": []}
                }
                # Submit LOGO Folds
                logo = LeaveOneGroupOut()
                fold_to_patient_map = {} # Local map for this target

                for fold_idx, (train_idx, test_idx) in enumerate(logo.split(X_t, y_t, groups=groups_t)):
                    task_label = f"{method}-{target}"
                    current_biopsy_id = groups_t.iloc[test_idx[0]]
                    
                    # Store Biopsy ID in the future object wrapper or map
                    # We need a complex key for the map: (target, fold_idx, biopsy_id)
                    key_info = (target, fold_idx, current_biopsy_id)
                    
                    future = executor.submit(
                        process_fold_task, X_t, y_t, train_idx, test_idx, fold_idx, task_label, is_cat
                    )
                    futures_map[future] = key_info

            # --- PHASE 2: COLLECT AS COMPLETED (Interleaved) ---
            total_tasks = len(futures_map)
            logger.info(f"   >>> Processing {total_tasks} tasks in parallel...")
            
            completed_count = 0
            for future in as_completed(futures_map):
                target_name, fold_id, biopsy_id = futures_map[future]
                res = future.result()
                
                completed_count += 1
                if completed_count % 50 == 0:
                    logger.info(f"   ... Completed {completed_count}/{total_tasks} tasks")

                if res["success"]:
                    # Get the specific accumulator for this target
                    acc = target_accumulators[target_name]
                    acc["meta"]["count"] += 1
                    
                    # A. Update Scores
                    for m, s in res["scores"].items():
                        acc["agg_scores"][m].append(s)
                    
                    # B. Update Rule Stats
                    idx = res["selected_indices"]
                    acc["rule_counts"][idx] += 1
                    for m in ["RF", "XGB", "Lasso"]:
                        acc["rule_imp_sums"][m][idx] += res["importances"][m]
                    
                    # C. Update Patient Results
                    p_rec = {
                        "Biopsy_ID": biopsy_id,
                        "True_Value": res["true_value"],
                        "Pred_RF": res["predictions"]["RF"],
                        "Pred_XGB": res["predictions"]["XGB"],
                        "Pred_Lasso": res["predictions"]["Lasso"]
                    }
                    acc["patient_results"].append(p_rec)


                # --- PHASE 3: FINALIZE & SAVE ALL TARGETS ---
            logger.info(f"   >>> Saving results for {method}...")
            
            for target, acc in target_accumulators.items():
                count = acc["meta"]["count"]
                if count == 0: continue
                
                le = acc["meta"]["le"]
                is_cat = acc["meta"]["is_cat"]

                # 1. Save Predictions
                preds_file = save_predictions_log(acc, method, target, le, is_cat)
                
                # 2. Compute & Save Rule Stats
                rules_df = compute_and_save_rule_stats(acc, count, method, target, original_rule_names)
                
                # 3. Save Raw Data Subset
                save_raw_data_subset(df_raw, rules_df, method, target)
                
                # 4. Leaderboard
                record = calculate_leaderboard_score(acc, method, target, is_cat, preds_file)
                leaderboard_data.append(record)
                logger.info(f"Saved {target}: Score = {record['Grand_Score']:.3f}")

            logger.info(f"   --- All tasks queued for {method}. Processing... ---")

    # --- Final Step: Save Leaderboard ---
    if leaderboard_data:
        lb_df = pd.DataFrame(leaderboard_data).sort_values("Grand_Score", ascending=False)
        lb_path = f"{OUTPUT_DATA_DIR}/final_leaderboard.csv"
        lb_df.to_csv(lb_path, index=False)
        logger.info(f"DONE. Leaderboard saved to {lb_path}")

if __name__ == "__main__":
    run_pipeline()


# =========================== OLD METHODS =============================


# def process_fold_task(X, y_enc, train_idx, fold_id, task_label):
#     pid = os.getpid()
#     start_t = time.time()
#     log_prefix = f"[{task_label}] Fold {fold_id+1} (PID {pid}) -"
    
#     X_train = X.iloc[train_idx]
#     y_train = y_enc[train_idx]
    
#     if len(np.unique(y_train)) < 2:
#         logger.warning(f"{log_prefix} Skipped (<2 classes)")
#         return {"success": False, "fold_id": fold_id, "msg": "Skipped (<2 classes)"}
        
#     try:
#         imp_rf, imp_xgb, imp_lasso = train_fold_logic(X_train, y_train, log_prefix)
#         duration = time.time() - start_t
        
#         return {
#             "success": True, 
#             "fold_id": fold_id, 
#             "imp_rf": imp_rf,
#             "imp_xgb": imp_xgb,
#             "imp_lasso": imp_lasso,
#             "duration": duration,
#             "pid": pid,
#             "label": task_label
#         }
#     except Exception as e:
#         logger.error(f"{log_prefix} FAILED: {e}")
#         return {"success": False, "fold_id": fold_id, "msg": str(e)}

# def train_fold_logic(X_train, y_train, log_prefix=""):
#     scaler = StandardScaler()
#     X_train_scaled = scaler.fit_transform(X_train)
    
#     # 1. RF
#     logger.info(f"{log_prefix} Training RF")
#     rf = RandomForestClassifier(n_estimators=50, max_depth=5, random_state=42, class_weight='balanced', n_jobs=1)
#     rf.fit(X_train, y_train)
#     imp_rf = rf.feature_importances_
    
#     # 2. XGB
#     logger.info(f"{log_prefix} Training XGB")
#     xgb_model = xgb.XGBClassifier(eval_metric='logloss', max_depth=3, reg_alpha=1, random_state=42, n_jobs=1)
#     xgb_model.fit(X_train, y_train)
#     imp_xgb = xgb_model.feature_importances_
    
#     # 3. Lasso
#     logger.info(f"{log_prefix} Training Lasso")
#     logit = LogisticRegression(penalty='elasticnet', l1_ratio=1.0, solver='saga', C=0.5, 
#                                class_weight='balanced', random_state=42, max_iter=2000)
#     logit.fit(X_train_scaled, y_train)
    
#     imp_lasso = np.abs(logit.coef_).flatten()
#     if logit.coef_.ndim > 1: 
#         imp_lasso = np.mean(np.abs(logit.coef_), axis=0)

#     # Normalize
#     for imp in [imp_rf, imp_xgb, imp_lasso]:
#         if imp.max() > 0: imp /= imp.max()
        
#     return imp_rf, imp_xgb, imp_lasso

# def save_detailed_results(top_rules_df, method_name, target_name):
#     # Save the detailed dataframe directly (Rule, Scores...)
#     safe_target = target_name.replace(" ", "_").replace("/", "-")
#     filename = f"{OUTPUT_DATA_DIR}/scores_{method_name}_{safe_target}.csv"
#     top_rules_df.to_csv(filename, index=False)
#     logger.info(f"   >>> SAVED SCORES: {filename}")
    
#     # Return all rules in this set (Union set)
#     return top_rules_df["Rule"].tolist()

# def execute_cross_validation(executor, X, y, groups, method, target, is_cat):
#     """
#     Manages the Parallel Leave-One-Group-Out Loop.
#     Collects raw results into accumulators.
#     """
#     n_features = X.shape[1]

#     # Accumulators
#     accumulators = {
#         "rule_counts": np.zeros(n_features),
#         "rule_imp_sums": {"RF": np.zeros(n_features), "XGB": np.zeros(n_features), "Lasso": np.zeros(n_features)},
#         "patient_results": [],
#         "agg_scores": {"RF": [], "XGB": [], "Lasso": []}
#     }

#     # Setup LOGO
#     logo = LeaveOneGroupOut()
#     futures = []
#     fold_to_patient_map = {} # Maps fold_id -> Biopsy_ID

#     # Submit Tasks
#     for fold_idx, (train_idx, test_idx) in enumerate(logo.split(X, y, groups=groups)):
#         task_label = f"{method}-{target}"
        
#         # Identify patient for logging
#         current_biopsy_id = groups.iloc[test_idx[0]]
#         fold_to_patient_map[fold_idx] = current_biopsy_id
        
#         futures.append(executor.submit(
#             process_fold_task, X, y, train_idx, test_idx, fold_idx, task_label, is_cat
#         ))

#         logger.info(f"   Queued {task_label} Fold {fold_idx+1} for Biopsy_ID {current_biopsy_id}")

#     # Collect Results
#     completed_count = 0
#     for f in as_completed(futures):
#         res = f.result()
#         if res["success"]:
#             completed_count += 1
            
#             # A. Scores
#             for m, s in res["scores"].items():
#                 accumulators["agg_scores"][m].append(s)
            
#             # B. Rule Stats
#             idx = res["selected_indices"]
#             accumulators["rule_counts"][idx] += 1
#             for m in ["RF", "XGB", "Lasso"]:
#                 accumulators["rule_imp_sums"][m][idx] += res["importances"][m]
            
#             # C. Patient Predictions
#             p_rec = {
#                 "Biopsy_ID": fold_to_patient_map[res["fold_id"]],
#                 "True_Value": res["true_value"],
#                 "Pred_RF": res["predictions"]["RF"],
#                 "Pred_XGB": res["predictions"]["XGB"],
#                 "Pred_Lasso": res["predictions"]["Lasso"]
#             }
#             accumulators["patient_results"].append(p_rec)

#     return accumulators, completed_count

