import pandas as pd
import numpy as np
import os
import time
import logging
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

# --- CONFIGURATION ---
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
BASE_INPUT_DIR = os.path.join(PROJECT_ROOT, constants.RESULTS_DATA_DIR)
OUTPUT_DATA_DIR = os.path.join(PROJECT_ROOT, constants.RESULTS_SIMPLE_STATS_DATA_DIR)
FEATURE_COUNTS = [None, 20, 50, 100]
METHODS = ["BAG", "CN", "KNN_R"] 
TARGETS = [
    "Pathological stage", 
    "GI stage", 
    "Cortico Response", 
    "Grade GVHD",
    "Survival at follow-up",
    "Clinical score",
    "Pathological score"
]

# Setup Logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger("SimpleStats")

def load_and_prep_data(input_file):
    if not os.path.exists(input_file): return None, None, None
    df = pd.read_csv(input_file)
    
    initial_len = len(df)
    if "FDR" in df.columns:
        df = df[df["FDR"] < 0.05]
        dropped = initial_len - len(df)
        if dropped > 0:
            logger.info(f"   Dropped {dropped} rows with FDR >= 0.05 (Retained: {len(df)})")
    
    if df.empty:
        logger.warning("   No data left after FDR filtering.")
        return None, None, None

    if "Rule" not in df.columns:
        df["Rule"] = df["Antecedents"] + " -> " + df["Consequents"]
    pivot = df.pivot_table(index="FOV", columns="Rule", values="Lift", fill_value=0)
    meta_cols = ["FOV", "Biopsy_ID"] + [t for t in TARGETS if t in df.columns]
    meta = df[meta_cols].drop_duplicates(subset="FOV").set_index("FOV")
    common = pivot.index.intersection(meta.index)
    return pivot.loc[common], meta.loc[common], df

def get_top_rules_stats(X_train, y_train, top_n, is_multiclass):
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

def predict_simple(X_train, y_train, X_test, top_rules, is_multiclass):
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

def process_single_fold(X, y_enc, train_idx, test_idx, fold_id, task_label, is_multiclass, k_feat):
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
        top_rules = get_top_rules_stats(X_train, y_train, k_feat, is_multiclass)
        
        # 2. Prediction
        pred, prob = predict_simple(X_train, y_train, X_test, top_rules, is_multiclass)
        
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

def save_comparison_summary_simple(leaderboard_data):
    """
    Creates a pivot table comparing scores across different feature counts for simple stats.
    """
    if not leaderboard_data: return
    
    df = pd.DataFrame(leaderboard_data)
    df["Config_Label"] = df["Num_Features"].apply(lambda x: "All" if pd.isna(x) or x == "All" else f"Top{x}")
    
    pivot = df.pivot_table(index=["Method", "Target"], columns="Config_Label", values="Accuracy")
    
    score_columns = [col for col in pivot.columns if col in ["All"] or col.startswith("Top")]
    numeric_score_pivot = pivot[score_columns]

    pivot["Best_Config"] = numeric_score_pivot.idxmax(axis=1)
    pivot["Best_Score"] = numeric_score_pivot.max(axis=1)
    
    filename = f"{OUTPUT_DATA_DIR}/comparison_summary_simple.csv"
    pivot.to_csv(filename)
    logger.info(f"   >>> SAVED COMPARISON SUMMARY (Simple): {filename}")

def run_pipeline():
    if not os.path.exists(BASE_INPUT_DIR):
        logger.error("CRITICAL: Input directory missing.")
        return
    os.makedirs(OUTPUT_DATA_DIR, exist_ok=True)
    
    leaderboard_data = [] # Changed to leaderboard_data for consistency with advanced_discovery

    for method in METHODS:
        path = f"{BASE_INPUT_DIR}/results_{method}.csv"
        if not os.path.exists(path): continue
        
        logger.info(f"=== METHOD: {method} ===")
        X_raw, y_meta, _ = load_and_prep_data(path)
        if X_raw is None: continue
        
        accumulators_map = {} # Key = (target, k_feat)
        futures_map = {}
        
        with ProcessPoolExecutor() as executor:
            
            for target in TARGETS:
                if target not in y_meta.columns: continue
                if y_meta[target].isna().all(): continue
                
                # Prep Target
                valid = y_meta[target].notna()
                X_t = X_raw[valid]
                y_t = y_meta.loc[valid, target]
                groups = y_meta.loc[valid, "Biopsy_ID"]
                
                le = LabelEncoder()
                y_enc = le.fit_transform(y_t.astype(str))
                y_final = pd.Series(y_enc, index=y_t.index)
                
                classes = np.unique(y_final)
                if len(classes) < 2: 
                    logger.warning(f"   Target {target} has less than 2 classes after encoding. Skipping.")
                    continue
                is_multiclass = len(classes) > 2

                for k_feat in FEATURE_COUNTS:
                    config_key = (target, k_feat)
                    accumulators_map[config_key] = {
                        "patient_results": [],
                        "rule_stats": {} # For tracking stability of selected rules
                    }
                    
                    # Submit Folds
                    logo = LeaveOneGroupOut()
                    
                    for i, (train_idx, test_idx) in enumerate(logo.split(X_t, y_final, groups)):
                        biopsy_id = groups.iloc[test_idx[0]]
                        feat_label = f"Top{k_feat}" if k_feat else "All"
                        task_label = f"{method}-{target}-{feat_label}"
                        
                        fut = executor.submit(process_single_fold, X_t, y_final, train_idx, test_idx, i, task_label, is_multiclass, k_feat)
                        futures_map[fut] = (target, k_feat, biopsy_id, le)
            
            # Collect results
            completed = 0
            total_tasks = len(futures_map)
            logger.info(f"   >>> Processing {total_tasks} tasks (Combinations of Target x Config x Folds)...")

            for fut in as_completed(futures_map):
                completed += 1
                if completed % 100 == 0: 
                    logger.info(f"   ... Completed {completed}/{total_tasks} tasks")
                
                target, k_feat, bio_id, le = futures_map[fut]
                res = fut.result()
                
                if res["success"]:
                    acc = accumulators_map[(target, k_feat)]
                    
                    true_label = le.inverse_transform([int(res["true_val"])])[0]
                    pred_label = le.inverse_transform([int(res["pred_val"])])[0]
                    
                    acc["patient_results"].append({
                        "Biopsy_ID": bio_id,
                        "True_Value": true_label,
                        "Pred_Label": pred_label,
                        "Prob_Pos": res["prob"]
                    })
                    
                    for r in res["selected_rules"]:
                        acc["rule_stats"][r] = acc["rule_stats"].get(r, 0) + 1
            
            # Save all results
            logger.info(f"   >>> Saving results for {method}...")

            for (target, k_feat), acc in accumulators_map.items():
                if not acc["patient_results"]: continue
                
                df_res = pd.DataFrame(acc["patient_results"])
                
                # Suffix for filename
                suffix = f"_Top{k_feat}" if k_feat else "_All"
                target_with_suffix = f"{target}{suffix}"

                fname_preds = f"{OUTPUT_DATA_DIR}/preds_SIMPLE_{method}_{target_with_suffix.replace(' ', '_')}.csv"
                df_res.to_csv(fname_preds, index=False)
                
                r_counts = acc["rule_stats"]
                df_rules = pd.DataFrame(list(r_counts.items()), columns=["Rule", "Count"])
                df_rules["Frequency"] = df_rules["Count"] / len(acc["patient_results"])
                df_rules = df_rules.sort_values("Count", ascending=False)
                fname_scores = f"{OUTPUT_DATA_DIR}/scores_SIMPLE_{method}_{target_with_suffix.replace(' ', '_')}.csv"
                df_rules.to_csv(fname_scores, index=False)
                
                accuracy = (df_res["True_Value"] == df_res["Pred_Label"]).mean()
                leaderboard_data.append({
                    "Method": method,
                    "Target": target,
                    "Num_Features": "All" if k_feat is None else k_feat,
                    "Type": "Simple_Stats",
                    "Accuracy": accuracy
                })
                logger.info(f"Saved {target} [{suffix.strip('_')}]: Acc={accuracy:.2%}")

    # Final Leaderboard
    if leaderboard_data:
        lb_df = pd.DataFrame(leaderboard_data).sort_values(["Target", "Accuracy"], ascending=False)
        lb_path = f"{OUTPUT_DATA_DIR}/leaderboard_simple.csv"
        lb_df.to_csv(lb_path, index=False)
        logger.info(f"DONE. Leaderboard (Simple) saved to {lb_path}")
        
        save_comparison_summary_simple(leaderboard_data)
    print("\nBenchmark Complete.")

if __name__ == "__main__":
    run_pipeline()
