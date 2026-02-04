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

# --- CONFIGURATION ---
BASE_INPUT_DIR = "experiment_results/full_run/data"
OUTPUT_DATA_DIR = "experiment_results/simple_stats_benchmarks/data"
TOP_N_RULES = 20
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
    """
    valid_cols = [c for c in X_train.columns if X_train[c].var() > 0]
    
    scores = []
    
    if is_multiclass:
        # ANOVA F-Test
        # Group samples by class
        groups = [X_train[y_train == c][valid_cols] for c in np.unique(y_train)]
        # f_oneway is vectorized in recent scipy, but usually expects arrays.
        # Simple loop is safer for robustness
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

def process_single_fold(X, y_enc, train_idx, test_idx, fold_id, task_label, is_multiclass):
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
        top_rules = get_top_rules_stats(X_train, y_train, TOP_N_RULES, is_multiclass)
        
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
        return {"success": False, "msg": str(e)}

def run_pipeline():
    if not os.path.exists(BASE_INPUT_DIR):
        print("Input dir missing")
        return
    os.makedirs(OUTPUT_DATA_DIR, exist_ok=True)
    
    leaderboard = []
    
    for method in METHODS:
        path = f"{BASE_INPUT_DIR}/results_{method}.csv"
        if not os.path.exists(path): continue
        
        logger.info(f"--- Method: {method} ---")
        X_raw, y_meta, _ = load_and_prep_data(path)
        if X_raw is None: continue
        
        with ProcessPoolExecutor() as executor:
            futures = {}
            
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
                if len(classes) < 2: continue
                is_multiclass = len(classes) > 2
                
                # Submit Folds
                logo = LeaveOneGroupOut()
                
                for i, (train_idx, test_idx) in enumerate(logo.split(X_t, y_final, groups)):
                    biopsy_id = groups.iloc[test_idx[0]]
                    fut = executor.submit(process_single_fold, X_t, y_final, train_idx, test_idx, i, target, is_multiclass)
                    futures[fut] = (target, biopsy_id, le)

            # Collect
            target_results = {} # target -> list of rows
            rule_stats = {} # target -> rule -> count
            
            completed = 0
            for fut in as_completed(futures):
                completed += 1
                if completed % 50 == 0: print(f"Completed {completed} tasks...", end="\r")
                
                target, bio_id, le = futures[fut]
                res = fut.result()
                
                if res["success"]:
                    if target not in target_results: 
                        target_results[target] = []
                        rule_stats[target] = {}
                        
                    # Decode Labels
                    true_label = le.inverse_transform([int(res["true_val"])])[0]
                    pred_label = le.inverse_transform([int(res["pred_val"])])[0]
                    
                    target_results[target].append({
                        "Biopsy_ID": bio_id,
                        "True_Value": true_label,
                        "Pred_Label": pred_label,
                        "Prob_Pos": res["prob"]
                    })
                    
                    # Track Stability
                    for r in res["selected_rules"]:
                        rule_stats[target][r] = rule_stats[target].get(r, 0) + 1
                        
            # Save Results
            for target, rows in target_results.items():
                # 1. Preds
                df_res = pd.DataFrame(rows)
                fname = f"{OUTPUT_DATA_DIR}/preds_SIMPLE_{method}_{target.replace(' ', '_')}.csv"
                df_res.to_csv(fname, index=False)
                
                # 2. Stability Scores
                r_counts = rule_stats.get(target, {})
                df_rules = pd.DataFrame(list(r_counts.items()), columns=["Rule", "Count"])
                df_rules["Frequency"] = df_rules["Count"] / len(rows) # rows = n_folds
                df_rules = df_rules.sort_values("Count", ascending=False)
                df_rules.to_csv(f"{OUTPUT_DATA_DIR}/scores_SIMPLE_{method}_{target.replace(' ', '_')}.csv", index=False)
                
                # 3. Leaderboard Entry
                acc = (df_res["True_Value"] == df_res["Pred_Label"]).mean()
                leaderboard.append({
                    "Method": method,
                    "Target": target,
                    "Type": "Simple_Stats",
                    "Accuracy": acc
                })
                logger.info(f"Saved {target}: Acc={acc:.2%}")

    # Final Leaderboard
    lb_df = pd.DataFrame(leaderboard).sort_values("Accuracy", ascending=False)
    lb_df.to_csv(f"{OUTPUT_DATA_DIR}/leaderboard_simple.csv", index=False)
    print("\nBenchmark Complete.")

if __name__ == "__main__":
    run_pipeline()
