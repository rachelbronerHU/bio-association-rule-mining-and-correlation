import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
import logging
import argparse

# --- IMPORT CONSTANTS ---
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))) # Adjust path for constants.py
import constants

from data_exploration.check_data_bias import load_stratified_biopsies
from check_rule_correlation_with_disease.stratified_utils import CONTROLS_ELIGIBLE, filter_viable_stratum

# Setup Logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger("ComparisonPlotter")

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))

def load_and_merge_leaderboards(no_self=False):
    """
    Loads ML and Simple leaderboards from the appropriate directories based on no_self flag.
    """
    if no_self:
        ml_data_dir = os.path.join(PROJECT_ROOT, constants.RESULTS_ML_DATA_DIR_NO_SELF)
        simple_data_dir = os.path.join(PROJECT_ROOT, constants.RESULTS_SIMPLE_STATS_DATA_DIR_NO_SELF)
        logger.info("Loading leaderboards from NO_SELF directories.")
    else:
        ml_data_dir = os.path.join(PROJECT_ROOT, constants.RESULTS_ML_DATA_DIR)
        simple_data_dir = os.path.join(PROJECT_ROOT, constants.RESULTS_SIMPLE_STATS_DATA_DIR)
        logger.info("Loading leaderboards from STANDARD directories.")

    lb_ml_path = os.path.join(ml_data_dir, "final_leaderboard.csv")
    lb_simple_path = os.path.join(simple_data_dir, "leaderboard_simple.csv")

    all_scores = []

    if os.path.exists(lb_ml_path):
        df_ml = pd.read_csv(lb_ml_path)
        if not df_ml.empty:
            # Melt ML individual model scores to long format
            df_ml_melted = df_ml.melt(
                id_vars=["Method", "Organ", "Target", "Num_Features"],
                value_vars=["RF_Mean", "XGB_Mean", "Lasso_Mean"],
                var_name="Model",
                value_name="Score"
            )
            df_ml_melted["Model"] = df_ml_melted["Model"].map({
                "RF_Mean": "Random Forest",
                "XGB_Mean": "XGBoost",
                "Lasso_Mean": "Lasso Regression"
            }) # Use descriptive names
            df_ml_melted["Approach"] = "ML"
            all_scores.append(df_ml_melted)
    else:
        logger.warning(f"ML Leaderboard not found at {lb_ml_path}")

    if os.path.exists(lb_simple_path):
        df_simple = pd.read_csv(lb_simple_path)
        if not df_simple.empty:
            # Simple stats has one score (Accuracy)
            df_simple = df_simple.rename(columns={"Accuracy": "Score"})
            df_simple["Model"] = "Simple"
            df_simple["Approach"] = "Simple"
            all_scores.append(df_simple)
    else:
        logger.warning(f"Simple Leaderboard not found at {lb_simple_path}")

    if not all_scores:
        logger.error("No leaderboard data loaded for comparison.")
        return None

    # Combine all scores
    df_combined = pd.concat(all_scores, ignore_index=True)
    df_combined["Organ_Features"] = df_combined["Organ"].fillna("?") + " | " + df_combined["Num_Features"].apply(
        lambda x: "All" if pd.isna(x) or x == "All" else f"Top{int(x)}"
    )
    df_combined["Num_Features_Str"] = df_combined["Num_Features"].apply(lambda x: "All" if pd.isna(x) or x == "All" else f"Top{int(x)}")
    df_combined["Num_Features_Sort"] = df_combined["Num_Features"].apply(lambda x: -1 if pd.isna(x) or x == "All" else int(x))
    df_combined = df_combined.sort_values(by=["Organ", "Method", "Target", "Num_Features_Sort", "Model"]).drop(columns=["Num_Features_Sort"])

    return df_combined

def generate_comparison_plots(no_self=False):
    """
    Generates grouped bar charts and scatter plots for ML vs Simple performance comparison.
    """
    os.makedirs(os.path.join(PROJECT_ROOT, constants.RESULTS_CLINICAL_CORRELATION_PLOTS_DIR), exist_ok=True)
    
    df_combined = load_and_merge_leaderboards(no_self=no_self)
    if df_combined is None:
        return

    # --- Plot 1: Overall Performance Comparison (Grouped Bar Charts with Models) ---
    logger.info("Generating Grouped Bar Charts for model-wise performance comparison...")

    try:
        df_biopsies = load_stratified_biopsies()
    except Exception as e:
        logger.error(f"Failed to load stratified biopsies: {e}")
        df_biopsies = pd.DataFrame()

    group_counts = {}

    def get_group_count(organ, target):
        if (organ, target) in group_counts:
            return group_counts[(organ, target)]
        
        if df_biopsies.empty:
            return "?"
            
        df_organ = df_biopsies[df_biopsies["Organ"] == organ]
        include_controls = CONTROLS_ELIGIBLE.get(target, True)
        df_t = df_organ if include_controls else df_organ[~df_organ["Is_Control"]]
        df_filtered, note = filter_viable_stratum(df_t, target)
        
        if df_filtered is not None:
            n_groups = df_filtered[df_filtered[target].notna()][target].nunique()
            group_counts[(organ, target)] = n_groups
        else:
            group_counts[(organ, target)] = "?"
        return group_counts[(organ, target)]

    df_combined["Group_Count"] = df_combined.apply(lambda row: get_group_count(row["Organ"], row["Target"]), axis=1)
    df_combined["Organ_Features_Groups"] = df_combined["Organ_Features"] + " (" + df_combined["Group_Count"].astype(str) + " groups)"
    df_combined["Method_Target"] = df_combined["Method"] + " | " + df_combined["Target"]

    # Using catplot for easier faceting and consistent plot styling across facets
    g = sns.catplot(
        data=df_combined, 
        x="Organ_Features_Groups", 
        y="Score", 
        hue="Model",      # Model (Random Forest, XGBoost, Lasso Regression, Simple Stats) as hue
        col="Method_Target", # Combined Method and Target
        col_wrap=1,       # One plot per row
        kind="bar",       # Bar plot
        height=4.0, aspect=3.0,  # Full width per plot
        palette="tab10",  # Color palette for models
        errorbar=None,
        sharey=False,      # Don't share Y-axis
        sharex=False,      # Ensure x-axis labels appear for every subplot
    )
    
    g.set_axis_labels("Organ | Feature Configuration", "Score") # More descriptive X-axis label
    g.set_titles("{col_name}")
    g.set_xticklabels(rotation=45, ha="right")
    
    # Add annotations to bars
    for ax in g.axes.flat:
        # Determine safe Y limit per axes since sharey=False
        y_max = 0
        for container in ax.containers:
            for bar in container:
                y_max = max(y_max, bar.get_height())
        if pd.isna(y_max) or y_max <= 0:
            y_max = 1.0
        ax.set_ylim(0, y_max * 1.15)
        
        for container in ax.containers:
            labels = []
            for bar in container:
                height = bar.get_height()
                if not pd.isna(height) and height > 0: 
                    labels.append(f'{height:.2f}')
                else:
                    labels.append("")
            ax.bar_label(container, labels=labels, label_type='edge', fontsize=8, padding=2)

    suffix_title = "(NO SELF)" if no_self else "(STANDARD)"
    g.fig.suptitle(f"Model-wise Performance Comparison {suffix_title}", y=1.02) 
    
    # Move the auto-generated legend to top-right
    sns.move_legend(g, "upper right", bbox_to_anchor=(1, 1.05))

    plt.tight_layout() 
    
    mode_str = "_NO_SELF" if no_self else ""
    plot_filename_models_bars = os.path.join(PROJECT_ROOT, constants.RESULTS_CLINICAL_CORRELATION_PLOTS_DIR, f"performance_model_wise_bars{mode_str}.png")
    
    plt.savefig(plot_filename_models_bars, bbox_inches='tight')
    plt.close(g.fig) 
    logger.info(f"Model-wise Grouped Bar Charts saved to {plot_filename_models_bars}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate Comparison Plots")
    parser.add_argument("--no_self", action="store_true", help="Load data from NO_SELF directories")
    args = parser.parse_args()
    
    generate_comparison_plots(no_self=args.no_self)