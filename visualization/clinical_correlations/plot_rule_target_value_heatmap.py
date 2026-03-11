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

from check_rule_correlation_with_disease.stratified_utils import CONTROLS_ELIGIBLE

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
        
    return df 

def generate_rule_target_value_heatmap(top_n_rules=20, num_best_strategies=1, 
                                       method_filter=None, no_self=False):
    """
    Generates heatmaps of top rules vs. clinical target values.
    If no_self is True, loads data from the pre-filtered NO_SELF directories.
    """
    
    os.makedirs(os.path.join(PROJECT_ROOT, constants.RESULTS_CLINICAL_CORRELATION_PLOTS_DIR), exist_ok=True)

    # 1. Determine Source Directories based on Flag
    if no_self:
        ml_data_dir = os.path.join(PROJECT_ROOT, constants.RESULTS_ML_DATA_DIR_NO_SELF)
        simple_data_dir = os.path.join(PROJECT_ROOT, constants.RESULTS_SIMPLE_STATS_DATA_DIR_NO_SELF)
        logger.info("Running in NO_SELF mode. Loading data from pre-filtered directories.")
    else:
        ml_data_dir = os.path.join(PROJECT_ROOT, constants.RESULTS_ML_DATA_DIR)
        simple_data_dir = os.path.join(PROJECT_ROOT, constants.RESULTS_SIMPLE_STATS_DATA_DIR)
        logger.info("Running in STANDARD mode.")

    # 2. Load Leaderboards
    lb_ml_path = os.path.join(ml_data_dir, "final_leaderboard.csv")
    lb_simple_path = os.path.join(simple_data_dir, "leaderboard_simple.csv")

    all_leaderboard_data = []

    if os.path.exists(lb_ml_path):
        lb_ml = pd.read_csv(lb_ml_path)
        if not lb_ml.empty:
            lb_ml = lb_ml.rename(columns={"Grand_Score": "Score"}).assign(Approach="ML")
            all_leaderboard_data.append(lb_ml[["Method", "Organ", "Target", "Num_Features", "Score", "Approach"]])
            
    if os.path.exists(lb_simple_path):
        lb_simple = pd.read_csv(lb_simple_path)
        if not lb_simple.empty:
            lb_simple = lb_simple.rename(columns={"Accuracy": "Score"}).assign(Approach="Simple")
            all_leaderboard_data.append(lb_simple[["Method", "Organ", "Target", "Num_Features", "Score", "Approach"]])

    if not all_leaderboard_data:
        logger.error(f"No leaderboard files found in:\n  {ml_data_dir}\n  {simple_data_dir}")
        return

    combined_lb = pd.concat(all_leaderboard_data, ignore_index=True)
    
    # Apply Method Filter if requested
    if method_filter:
        combined_lb = combined_lb[combined_lb["Method"] == method_filter]
        if combined_lb.empty:
            logger.warning(f"No strategies found for method '{method_filter}'.")
            return

    # Sort & Deduplicate per (Method, Organ, Target)
    combined_lb = combined_lb.sort_values(by="Score", ascending=False)
    combined_lb = combined_lb.drop_duplicates(subset=["Method", "Organ", "Target"], keep="first")
    
    combined_lb = combined_lb.head(num_best_strategies)

    if combined_lb.empty:
        logger.warning("Could not identify any best strategies from leaderboards.")
        return

    logger.info(f"Identified {len(combined_lb)} Best Strateg(y/ies):")

    for i, (idx, best_strategy) in enumerate(combined_lb.iterrows()):
        logger.info(f"\n--- Strategy {i+1} ({best_strategy['Approach']}) ---")
        logger.info(f"  Method: {best_strategy['Method']}")
        logger.info(f"  Target: {best_strategy['Target']}")
        logger.info(f"  Score: {best_strategy['Score']:.3f}")

        # 3. Extract Top N Rules
        method = best_strategy["Method"]
        target = best_strategy["Target"]
        organ = best_strategy["Organ"]
        num_features = best_strategy["Num_Features"] # Can be 'All' or a number
        approach = best_strategy["Approach"]

        score_file_dir = ml_data_dir if approach == "ML" else simple_data_dir
        
        config_suffix = ""
        if num_features == "All" or pd.isna(num_features):
            config_suffix = "_All"
        else:
            config_suffix = f"_Top{int(num_features)}"

        target_for_filename = target.replace(" ", "_").replace("/", "-")
        organ_for_filename = organ.replace(" ", "_")
        score_filename_base = f"scores_{method}_{organ_for_filename}_{target_for_filename}{config_suffix}"
        
        if approach == "Simple":
            score_filename_base = f"scores_SIMPLE_{method}_{organ_for_filename}_{target_for_filename}{config_suffix}"

        score_file_path = os.path.join(score_file_dir, f"{score_filename_base}.csv")

        if not os.path.exists(score_file_path):
            logger.error(f"Score file for best strategy {i+1} not found: {score_file_path}")
            continue

        rule_scores_df = pd.read_csv(score_file_path)
        
        # Ensure Rule column exists (Standardization)
        if "Rule" not in rule_scores_df.columns:
             if "Antecedents" in rule_scores_df.columns:
                 rule_scores_df["Rule"] = rule_scores_df["Antecedents"] + " -> " + rule_scores_df["Consequents"]
             else:
                 logger.error(f"Score file missing Rule column: {score_file_path}")
                 continue

        # No filtering needed here! Data is already filtered in the source directory.

        # Calculate Ranking Score
        if approach == "ML":
            rule_scores_df["Rank_Score"] = rule_scores_df[["Mean_Imp_RF", "Mean_Imp_XGB", "Mean_Imp_Lasso"]].mean(axis=1)
            score_col_name = "Avg_Imp"
        else: # Simple Stats
            rule_scores_df["Rank_Score"] = rule_scores_df["Frequency"]
            score_col_name = "Freq"

        # Select Top N
        top_rules_df = rule_scores_df.sort_values("Rank_Score", ascending=False).head(top_n_rules)
        top_rules = top_rules_df["Rule"].tolist()
        top_scores = top_rules_df["Rank_Score"].tolist()
            
        if not top_rules:
            logger.warning(f"No top rules found for best strategy {i+1}.")
            continue

        # 4. Load Raw Data (filtered) for Heatmap Values
        # Construct target-specific results filename (includes organ)
        results_filename = f"results_{method}_{organ_for_filename}_{target_for_filename}{config_suffix}.csv"
        
        # Let's try to find the specific subset file first
        target_results_path = os.path.join(score_file_dir, results_filename)
        
        if os.path.exists(target_results_path):
            df_raw = pd.read_csv(target_results_path)
            if "Rule" not in df_raw.columns:
                df_raw["Rule"] = df_raw["Antecedents"] + " -> " + df_raw["Consequents"]
        else:
            logger.warning(f"Target-specific results file not found: {target_results_path}. Trying generic...")
            if no_self:
                logger.error("In NO_SELF mode, target-specific filtered data is required but missing.")
                continue
            
            # Standard fallback
            df_raw = load_and_prep_data(method, os.path.join(PROJECT_ROOT, constants.RESULTS_DATA_DIR))

        if df_raw is None:
            continue
        
        if target not in df_raw.columns or df_raw[target].isna().all():
            continue
            
        unique_target_values = sorted(df_raw[target].dropna().unique().tolist())
        
        # Dynamic Control Filtering
        if not CONTROLS_ELIGIBLE.get(target, True):
            filtered_vals = []
            for v in unique_target_values:
                if str(v).lower() == "control":
                    continue
                try:
                    if float(v) == 0.0:
                        continue
                except ValueError:
                    pass
                filtered_vals.append(v)
            unique_target_values = filtered_vals

        if not unique_target_values:
            continue

        # 5. Calculate Cell Values: Mean Lift
        heatmap_data = pd.DataFrame(index=top_rules, columns=unique_target_values)

        for rule in top_rules:
            for target_value in unique_target_values:
                rule_data_for_target_value = df_raw[
                    (df_raw["Rule"] == rule) & 
                    (df_raw[target] == target_value)
                ]
                
                if not rule_data_for_target_value.empty and "Lift" in rule_data_for_target_value.columns:
                    mean_lift = rule_data_for_target_value["Lift"].mean()
                    heatmap_data.loc[rule, target_value] = mean_lift
                else:
                    heatmap_data.loc[rule, target_value] = np.nan 

        # Cast to float, but leave NaNs intact (no .fillna(0))
        heatmap_data = heatmap_data.astype(float)

        # 6. Generate Layout
        fig_width = max(16, len(unique_target_values) * 1.5 + 6)
        fig_height = max(8, len(top_rules) * 0.5)
        
        fig = plt.figure(figsize=(fig_width, fig_height))
        
        gs = fig.add_gridspec(1, 4, width_ratios=[len(unique_target_values), 1, 2, 0.2], wspace=0.05)
        
        ax_heatmap = fig.add_subplot(gs[0, 0])
        ax_score = fig.add_subplot(gs[0, 1])
        ax_violin = fig.add_subplot(gs[0, 2])
        ax_cbar = fig.add_subplot(gs[0, 3])

        # Plot Heatmap
        sns.heatmap(heatmap_data, annot=True, cmap="RdBu_r", center=1.0, fmt=".2f", 
                    linewidths=.5, linecolor='gray',
                    cbar_ax=ax_cbar, cbar_kws={'label': 'Mean Lift'}, ax=ax_heatmap)
        
        suffix_title = " (NO SELF)" if no_self else ""
        plot_title = f"Strategy {i+1} | {method} | Target: {target} | Score: {best_strategy['Score']:.3f}{suffix_title}"
        ax_heatmap.set_title(plot_title, fontsize=12, pad=10)
        ax_heatmap.set_xlabel(f"{target} Value")
        ax_heatmap.set_ylabel("Rule")

        # Plot Score Column
        ax_score.set_ylim(len(top_rules), 0)
        ax_score.set_xlim(0, 1)
        ax_score.axis('off')
        ax_score.set_title(score_col_name)
        
        for idx, score in enumerate(top_scores):
            ax_score.text(0.5, idx + 0.5, f"{score:.3f}", 
                          ha='center', va='center', fontsize=10)
            ax_score.axhline(idx, color='gray', linewidth=0.5)
            ax_score.axhline(idx + 1, color='gray', linewidth=0.5)

        # Prepare Violin Data
        violin_data = []
        for rule in top_rules:
            lifts = df_raw[df_raw["Rule"] == rule]["Lift"].dropna().values
            violin_data.append(lifts if len(lifts) > 0 else np.array([]))
            
        violin_positions = np.arange(len(top_rules)) + 0.5
        
        # Plot Violins
        parts = ax_violin.violinplot(violin_data, positions=violin_positions, vert=False, 
                                     widths=0.8, showextrema=True, showmedians=True)
        
        for pc in parts['bodies']:
            pc.set_facecolor('#a6cee3')
            pc.set_edgecolor('black')
            pc.set_linewidth(0.5)
            pc.set_alpha(0.8)
            
        for partname in ('cbars', 'cmins', 'cmaxes', 'cmedians'):
            if partname in parts:
                parts[partname].set_edgecolor('black')
                parts[partname].set_linewidth(1)

        for rule_idx, lifts in enumerate(violin_data):
            if len(lifts) > 0:
                y_jitter = (rule_idx + 0.5) + np.random.uniform(-0.1, 0.1, size=len(lifts))
                ax_violin.scatter(lifts, y_jitter, color='black', alpha=0.5, s=3, zorder=3)

        ax_violin.set_ylim(len(top_rules), 0)
        ax_violin.set_yticks([])
        ax_violin.set_yticklabels([])
        ax_violin.spines['left'].set_visible(False)
        ax_violin.spines['right'].set_visible(False)
        ax_violin.spines['top'].set_visible(False)
        ax_violin.spines['bottom'].set_visible(True)
        
        ax_violin.set_title("Distribution of FOV-level Lifts")
        ax_violin.set_xlabel("Lift")
        
        for y in range(len(top_rules) + 1):
            ax_violin.axhline(y, color='gray', linewidth=0.5)

        mode_str = "_NO_SELF" if no_self else ""
        plot_filename = os.path.join(
            PROJECT_ROOT, constants.RESULTS_CLINICAL_CORRELATION_PLOTS_DIR, 
            f"heatmap_rules_{method}_{organ_for_filename}_{target_for_filename}{mode_str}.png"
        )
        plt.savefig(plot_filename, bbox_inches='tight')
        plt.close()
        logger.info(f"Heatmap saved to {plot_filename}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Generate clinical correlation heatmaps.")
    parser.add_argument("--num_strategies", type=int, default=3, help="Number of best strategies to process (default: 3)")
    parser.add_argument("--top_n_rules", type=int, default=20, help="Number of top rules to include in heatmap (default: 20)")
    parser.add_argument("--method", type=str, help="Filter by specific method (e.g., KNN_R, BAG)")
    parser.add_argument("--no_self", action="store_true", help="Load results from 'no_self' directories (strict no shared items).")

    args = parser.parse_args()
    
    generate_rule_target_value_heatmap(
        top_n_rules=args.top_n_rules, 
        num_best_strategies=args.num_strategies,
        method_filter=args.method,
        no_self=args.no_self
    )