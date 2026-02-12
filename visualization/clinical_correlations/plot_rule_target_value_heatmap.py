import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
import logging

# --- IMPORT CONSTANTS ---
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))) # Adjust path for constants.py
import constants

# Setup Logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger("ClinicalCorrelationPlotter")

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))

def load_and_prep_data(method, base_input_dir):
    """
    Loads the raw rules data and metadata for a given method.
    """
    input_file = os.path.join(base_input_dir, f"results_{method}.csv")
    if not os.path.exists(input_file):
        logger.error(f"Input file not found: {input_file}")
        return None, None
    
    df = pd.read_csv(input_file)
    if "Rule" not in df.columns:
        df["Rule"] = df["Antecedents"] + " -> " + df["Consequents"]
        
    return df # df is the df_raw from advanced_discovery/simple_stats

def generate_rule_target_value_heatmap(top_n_rules=20, num_best_strategies=1):
    """
    Generates heatmaps of top rules vs. clinical target values, colored by mean Lift,
    for the N best strategies.
    """
    
    os.makedirs(os.path.join(PROJECT_ROOT, constants.RESULTS_CLINICAL_CORRELATION_PLOTS_DIR), exist_ok=True)

    # 1. Load leaderboards to identify the best strategy(ies)
    lb_ml_path = os.path.join(PROJECT_ROOT, constants.RESULTS_ML_DATA_DIR, "final_leaderboard.csv")
    lb_simple_path = os.path.join(PROJECT_ROOT, constants.RESULTS_SIMPLE_STATS_DATA_DIR, "leaderboard_simple.csv")

    all_leaderboard_data = []

    if os.path.exists(lb_ml_path):
        lb_ml = pd.read_csv(lb_ml_path)
        if not lb_ml.empty:
            lb_ml = lb_ml.rename(columns={"Grand_Score": "Score"}).assign(Approach="ML")
            all_leaderboard_data.append(lb_ml[["Method", "Target", "Num_Features", "Score", "Approach"]])
            
    if os.path.exists(lb_simple_path):
        lb_simple = pd.read_csv(lb_simple_path)
        if not lb_simple.empty:
            lb_simple = lb_simple.rename(columns={"Accuracy": "Score"}).assign(Approach="Simple")
            all_leaderboard_data.append(lb_simple[["Method", "Target", "Num_Features", "Score", "Approach"]])

    if not all_leaderboard_data:
        logger.error("No leaderboard files found. Run advanced_discovery.py and run_robust_simple_stats.py first.")
        return

    combined_lb = pd.concat(all_leaderboard_data, ignore_index=True)
    combined_lb = combined_lb.sort_values(by="Score", ascending=False).head(num_best_strategies)

    if combined_lb.empty:
        logger.warning("Could not identify any best strategies from leaderboards.")
        return

    logger.info(f"Identified {len(combined_lb)} Best Strateg(y/ies):")

    for i, (idx, best_strategy) in enumerate(combined_lb.iterrows()):
        logger.info(f"\n--- Strategy {i+1} ({best_strategy['Approach']}) ---")
        logger.info(f"  Method: {best_strategy['Method']}")
        logger.info(f"  Target: {best_strategy['Target']}")
        logger.info(f"  Num_Features: {best_strategy['Num_Features']}")
        logger.info(f"  Score: {best_strategy['Score']:.3f}")

        # 2. Extract Top N Rules for This Best Strategy
        method = best_strategy["Method"]
        target = best_strategy["Target"]
        num_features = best_strategy["Num_Features"] # Can be 'All' or a number
        approach = best_strategy["Approach"]

        score_file_dir = os.path.join(PROJECT_ROOT, constants.RESULTS_ML_DATA_DIR) if approach == "ML" else os.path.join(PROJECT_ROOT, constants.RESULTS_SIMPLE_STATS_DATA_DIR)
        
        config_suffix = ""
        if num_features == "All" or pd.isna(num_features):
            config_suffix = "_All"
        else:
            config_suffix = f"_Top{int(num_features)}"

        # Note: Target name in score files also includes config_suffix
        target_for_filename = target.replace(" ", "_").replace("/", "-")
        score_filename_base = f"scores_{method}_{target_for_filename}{config_suffix}"
        
        # Adjust score filename based on approach (Simple approach uses different prefix)
        if approach == "Simple":
            score_filename_base = f"scores_SIMPLE_{method}_{target_for_filename}{config_suffix}"

        score_file_path = os.path.join(score_file_dir, f"{score_filename_base}.csv")

        if not os.path.exists(score_file_path):
            logger.error(f"Score file for best strategy {i+1} not found: {score_file_path}")
            continue

        rule_scores_df = pd.read_csv(score_file_path)
        
        # Sort rules by importance/frequency and take top N
        if approach == "ML":
            # Average RF, XGB, Lasso importance
            rule_scores_df["Avg_Importance"] = rule_scores_df[["Mean_Imp_RF", "Mean_Imp_XGB", "Mean_Imp_Lasso"]].mean(axis=1)
            top_rules = rule_scores_df.sort_values("Avg_Importance", ascending=False).head(top_n_rules)["Rule"].tolist()
        else: # Simple Stats
            top_rules = rule_scores_df.sort_values("Frequency", ascending=False).head(top_n_rules)["Rule"].tolist()
            
        if not top_rules:
            logger.warning(f"No top rules found for best strategy {i+1}.")
            continue

        logger.info(f"Extracted {len(top_rules)} top rules for strategy {i+1}.")

        # 3. Identify Clinical Target Values (Columns)
        df_raw = load_and_prep_data(method, os.path.join(PROJECT_ROOT, constants.RESULTS_DATA_DIR))
        if df_raw is None:
            logger.error(f"Failed to load raw data for heatmap calculation for strategy {i+1}.")
            continue
        
        # Ensure the target column exists and is not all NaN
        if target not in df_raw.columns or df_raw[target].isna().all():
            logger.error(f"Target '{target}' not found or is all NaN in metadata for heatmap for strategy {i+1}.")
            continue
            
        unique_target_values = sorted(df_raw[target].dropna().unique().tolist())
        if not unique_target_values:
            logger.warning(f"No unique non-NaN target values found for '{target}' for strategy {i+1}. Cannot create heatmap.")
            continue

        # 4. Calculate Cell Values: Mean Lift of FOVs for Each Rule-TargetValue Pair
        heatmap_data = pd.DataFrame(index=top_rules, columns=unique_target_values)
        
        # Filter df_raw by the best strategy's method once for efficiency
        # This is not necessarily the method the df_raw corresponds to.
        # df_raw is loaded based on the method parameter of load_and_prep_data
        # which is the method from the best strategy. So df_raw corresponds to the current method.

        for rule in top_rules:
            for target_value in unique_target_values:
                # Filter df_raw for this rule and target value directly
                rule_data_for_target_value = df_raw[
                    (df_raw["Rule"] == rule) & 
                    (df_raw[target] == target_value)
                ]
                
                if not rule_data_for_target_value.empty and "Lift" in rule_data_for_target_value.columns:
                    mean_lift = rule_data_for_target_value["Lift"].mean()
                    heatmap_data.loc[rule, target_value] = mean_lift
                else:
                    heatmap_data.loc[rule, target_value] = np.nan # Or 0, depending on preference

        heatmap_data = heatmap_data.astype(float) # Ensure all values are numeric

        # Handle NaN values for plotting (e.g., fill with 0 or a low value)
        heatmap_data = heatmap_data.fillna(0) # Or another appropriate value

        # 5. Generate Heatmap
        plt.figure(figsize=(max(10, len(unique_target_values) * 1.5), max(8, len(top_rules) * 0.5)))
        sns.heatmap(heatmap_data, annot=True, cmap="Blues", fmt=".2f", linewidths=.5, cbar_kws={'label': 'Mean Lift'})
        
        plt.title(f"Strategy {i+1} ({approach}): Mean Lift of Top Rules by {target} Value")
        plt.xlabel(f"{target} Value")
        plt.ylabel("Rule")
        plt.tight_layout()
        
        plot_filename = os.path.join(
            PROJECT_ROOT, constants.RESULTS_CLINICAL_CORRELATION_PLOTS_DIR, 
            f"heatmap_rules_by_target_value_strategy_{i+1}_{method}_{target.replace(' ', '_').replace('/', '-')}_{approach}.png"
        )
        plt.savefig(plot_filename, bbox_inches='tight')
        plt.close()
        logger.info(f"Heatmap for strategy {i+1} saved to {plot_filename}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Generate clinical correlation heatmaps.")
    parser.add_argument("--num_strategies", type=int, default=3, help="Number of best strategies to process (default: 3)")
    parser.add_argument("--top_n_rules", type=int, default=20, help="Number of top rules to include in heatmap (default: 20)")
    
    args = parser.parse_args()
    
    # Generate heatmaps with provided arguments
    generate_rule_target_value_heatmap(top_n_rules=args.top_n_rules, num_best_strategies=args.num_strategies)
