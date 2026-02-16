import pandas as pd
import numpy as np
import os
import logging
import warnings
import time
import argparse
import ast
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

# --- CONFIGURATION ---
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
BASE_INPUT_DIR = os.path.join(PROJECT_ROOT, constants.RESULTS_DATA_DIR)

FEATURE_COUNTS = [None, 20, 50, 100]
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

def parse_rule_items(item_str):
    """
    Parses string representation of list "['A', 'B']" into a python set.
    """
    try:
        items = ast.literal_eval(item_str)
        # Clean items (remove _CENTER/_NEIGHBOR suffixes)
        return set(i.replace('_CENTER', '').replace('_NEIGHBOR', '') for i in items)
    except:
        return set([str(item_str).strip("[]'\"")])

def load_and_prep_data(input_file, no_self=False):
    if not os.path.exists(input_file):
        logger.warning(f"File not found: {input_file}")
        return None, None, None

    df = pd.read_csv(input_file)
    initial_len = len(df)
    
    # Apply No-Self Filter (Strict: No Shared Items)
    if no_self:
        def has_overlap(row):
            # Parse Antecedents and Consequents
            ant = parse_rule_items(row['Antecedents'])
            con = parse_rule_items(row['Consequents'])
            return not ant.isdisjoint(con)
        
        df = df[~df.apply(has_overlap, axis=1)]
        logger.info(f"   Applied No-Self Filter: {initial_len} -> {len(df)} rules")

    # Filter by FDR
    initial_len = len(df)
    if "FDR" in df.columns:
        df = df[df["FDR"] < 0.05]
        dropped = initial_len - len(df)
        if dropped > 0:
            logger.info(f"   Dropped {dropped} rows with P-Value >= 0.05 (Retained: {len(df)})")
    
    if df.empty:
        logger.warning("   No data left after filtering.")
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

def process_fold_task(X, y_enc, train_idx, test_idx, fold_id, task_label, is_categorical, k_feat):


    pid = os.getpid()
    start_t = time.time()
    log_prefix = f"[{task_label}] Fold {fold_id+1} (PID {pid}) K-Feat {k_feat}-"

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
        n_features_in = X_train.shape[1]
        
        # Determine if we do selection
        if k_feat is not None and k_feat < n_features_in:
            if is_categorical:
                embed_model = RandomForestClassifier(n_estimators=50, random_state=42, n_jobs=1)
            else:
                embed_model = RandomForestRegressor(n_estimators=50, random_state=42, n_jobs=1)

            selector = SelectFromModel(embed_model, max_features=k_feat, threshold=-np.inf)
            selector.fit(X_train, y_train)
            
            selected_indices = selector.get_support(indices=True)
            X_train_sel = selector.transform(X_train)
            X_test_sel = selector.transform(X_test)
        else:
            # Use ALL features
            selected_indices = np.arange(n_features_in)
            X_train_sel = X_train.values
            X_test_sel = X_test.values

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

def save_predictions_log(acc, method, target, le, is_cat, output_dir):
    """
    Step 1: Save the per-patient prediction details.
    """
    safe_target = target.replace(" ", "_").replace("/", "-")
    pred_df = pd.DataFrame(acc["patient_results"])
    
    # If categorical, map numbers (0,1) back to names (Stage 1, Stage 2)
    if is_cat and le:
        pred_df["True_Value_Label"] = le.inverse_transform(pred_df["True_Value"].astype(int))
        
    filename = f"{output_dir}/preds_{method}_{safe_target}.csv"
    pred_df.to_csv(filename, index=False)
    return filename

def compute_and_save_rule_stats(acc, count, method, target, original_rule_names, output_dir):
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
    filename = f"{output_dir}/scores_{method}_{safe_target}.csv"
    rules_df.to_csv(filename, index=False)
    
    return rules_df

def save_raw_data_subset(df_raw, rules_df, method, target, output_dir):
    """
    Step 3: Save the raw data rows for the Top 100 most stable rules.
    """
    # Get the names of the top 100 rules (sorted by stability)
    top_stable_rules = rules_df.head(100)["Rule"].tolist()
    
    # Call the existing helper to filter and save
    save_filtered_raw_data(df_raw, top_stable_rules, method, target, output_dir)

def save_filtered_raw_data(raw_df, top_rules, method_name, target_name, output_dir):
    filtered_df = raw_df[raw_df["Rule"].isin(top_rules)].copy()
    safe_target = target_name.replace(" ", "_").replace("/", "-")
    filename = f"{output_dir}/results_{method_name}_{safe_target}.csv"
    filtered_df.to_csv(filename, index=False)
    logger.info(f"   >>> SAVED FILTERED RAW: {filename}")

def save_comparison_summary(leaderboard_data, output_dir):
    """
    Creates a pivot table comparing scores across different feature counts.
    """
    if not leaderboard_data: return
    
    df = pd.DataFrame(leaderboard_data)
    # Pivot: Index=[Method, Target], Columns=[Num_Features], Values=[Grand_Score]
    # Handle 'All' vs Numbers for column sorting
    df["Config_Label"] = df["Num_Features"].apply(lambda x: "All" if pd.isna(x) or x == "All" else f"Top{x}")
    
    pivot = df.pivot_table(index=["Method", "Target"], columns="Config_Label", values="Grand_Score")
    
    # Ensure idxmax and max are only applied to numerical columns
    # Create a subset of the pivot table containing only numerical score columns
    score_columns = [col for col in pivot.columns if col in ["All"] or col.startswith("Top")]
    numeric_score_pivot = pivot[score_columns]

    pivot["Best_Config"] = numeric_score_pivot.idxmax(axis=1)
    pivot["Best_Score"] = numeric_score_pivot.max(axis=1)
    
    filename = f"{output_dir}/comparison_summary.csv"
    pivot.to_csv(filename)
    logger.info(f"   >>> SAVED COMPARISON SUMMARY: {filename}")

def calculate_leaderboard_score(acc, method, target, is_cat, preds_filename, k_feat):
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
        "Num_Features": "All" if k_feat is None else k_feat,
        "Metric": metric,
        "Grand_Score": grand_score,
        "RF_Mean": np.mean(acc["agg_scores"]["RF"]),
        "XGB_Mean": np.mean(acc["agg_scores"]["XGB"]),
        "Lasso_Mean": np.mean(acc["agg_scores"]["Lasso"]),
        "Log_Preds": os.path.basename(preds_filename)
    }


# --- ORCHESTRATOR ---

def run_pipeline(no_self=False):
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
        
    # Select Output Directory
    if no_self:
        output_dir = os.path.join(PROJECT_ROOT, constants.RESULTS_ML_DATA_DIR_NO_SELF)
        logger.info(f"Running in NO_SELF mode. Output: {output_dir}")
    else:
        output_dir = os.path.join(PROJECT_ROOT, constants.RESULTS_ML_DATA_DIR)
        logger.info(f"Running in STANDARD mode. Output: {output_dir}")

    os.makedirs(output_dir, exist_ok=True)

    leaderboard_data = []

    for method in METHODS_TO_ANALYZE:
        input_file = f"{BASE_INPUT_DIR}/results_{method}.csv"
        if not os.path.exists(input_file): continue
            
        logger.info(f"=== METHOD: {method} === input file: {input_file}")
        
        # Pass no_self flag to loader
        X_raw, y_meta_raw, df_raw = load_and_prep_data(input_file, no_self=no_self)
        if X_raw is None: 
            logger.warning(f"   Skipping {method} due to data issues - X_raw is None")
            continue

        X_safe, original_rule_names = sanitize_features(X_raw)

        # Accumulators Map: Key = (target, k_feat)
        # We need distinct accumulators for each feature configuration
        accumulators_map = {} 
        futures_map = {} 

        with ProcessPoolExecutor() as executor:
            
            for target in TARGETS:
                if target not in y_meta_raw.columns: 
                    logger.warning(f"   Target {target} not found in metadata. Skipping.")
                    continue

                if y_meta_raw[target].isna().all(): 
                    logger.warning(f"   Target {target} has all NaN values. Skipping.")
                    continue
                
                # --- Step A: Prepare Target Data ---
                prep_res = prepare_target_data(X_safe, y_meta_raw, target)
                if not prep_res: continue 
                
                X_t, y_t, groups_t, le, is_cat = prep_res
                n_features_total = X_t.shape[1]

                # --- Iterate over Feature Counts ---
                for k_feat in FEATURE_COUNTS:
                    
                    config_key = (target, k_feat)
                    
                    # Init Accumulator for this specific config
                    accumulators_map[config_key] = {
                        "meta": {"le": le, "is_cat": is_cat, "count": 0}, 
                        "rule_counts": np.zeros(n_features_total),
                        "rule_imp_sums": {"RF": np.zeros(n_features_total), "XGB": np.zeros(n_features_total), "Lasso": np.zeros(n_features_total)},
                        "patient_results": [],
                        "agg_scores": {"RF": [], "XGB": [], "Lasso": []}
                    }

                    # Submit LOGO Folds
                    logo = LeaveOneGroupOut()

                    for fold_idx, (train_idx, test_idx) in enumerate(logo.split(X_t, y_t, groups=groups_t)):
                        
                        feat_label = f"Top{k_feat}" if k_feat else "All"
                        task_label = f"{method}-{target}-{feat_label}"
                        current_biopsy_id = groups_t.iloc[test_idx[0]]
                        
                        # Key info to retrieve correct accumulator later
                        key_info = (target, k_feat, fold_idx, current_biopsy_id)
                        
                        future = executor.submit(
                            process_fold_task, X_t, y_t, train_idx, test_idx, fold_idx, task_label, is_cat, k_feat
                        )
                        futures_map[future] = key_info

            # --- PHASE 2: COLLECT AS COMPLETED ---
            total_tasks = len(futures_map)
            logger.info(f"   >>> Processing {total_tasks} tasks (Combinations of Target x Config x Folds)...")
            
            completed_count = 0
            for future in as_completed(futures_map):
                target_name, k_feat, fold_id, biopsy_id = futures_map[future]
                res = future.result()
                
                completed_count += 1
                if completed_count % 100 == 0:
                    logger.info(f"   ... Completed {completed_count}/{total_tasks} tasks")

                if res["success"]:
                    # Get the specific accumulator
                    acc = accumulators_map[(target_name, k_feat)]
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


            # --- PHASE 3: FINALIZE & SAVE ---
            logger.info(f"   >>> Saving results for {method}...")
            
            # Iterate through the map to save each config separately
            for (target, k_feat), acc in accumulators_map.items():
                count = acc["meta"]["count"]
                if count == 0: continue
                
                le = acc["meta"]["le"]
                is_cat = acc["meta"]["is_cat"]
                
                # Append suffix to target name for file uniqueness
                suffix = f"_Top{k_feat}" if k_feat else "_All"
                target_with_suffix = f"{target}{suffix}"

                # 1. Save Predictions
                preds_file = save_predictions_log(acc, method, target_with_suffix, le, is_cat, output_dir)
                
                # 2. Compute & Save Rule Stats
                rules_df = compute_and_save_rule_stats(acc, count, method, target_with_suffix, original_rule_names, output_dir)
                
                # 3. Save Raw Data Subset (Only for specific Feature Counts if needed, or all)
                # We typically only care about the top rules from the 'Top' configs, 
                # but we can save for all.
                save_raw_data_subset(df_raw, rules_df, method, target_with_suffix, output_dir)
                
                # 4. Leaderboard
                record = calculate_leaderboard_score(acc, method, target, is_cat, preds_file, k_feat)
                leaderboard_data.append(record)
                
                logger.info(f"Saved {target} [{suffix.strip('_')}]: Score = {record['Grand_Score']:.3f}")

    # --- Final Step: Save Leaderboard & Summary ---
    if leaderboard_data:
        # Save Full Leaderboard
        lb_df = pd.DataFrame(leaderboard_data).sort_values(["Target", "Grand_Score"], ascending=False)
        lb_path = f"{output_dir}/final_leaderboard.csv"
        lb_df.to_csv(lb_path, index=False)
        logger.info(f"DONE. Leaderboard saved to {lb_path}")
        
        # Save Pivot Summary
        save_comparison_summary(leaderboard_data, output_dir)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Advanced Discovery (ML) Benchmark")
    parser.add_argument("--no_self", action="store_true", help="Exclude rules with any shared cell types (strict no-self).")
    args = parser.parse_args()
    
    run_pipeline(no_self=args.no_self)
