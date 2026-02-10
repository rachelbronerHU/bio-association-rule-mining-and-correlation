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
logger = logging.getLogger("ComparisonPlotter")

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))

def load_and_merge_leaderboards():
    """
    Loads ML and Simple leaderboards, extracts individual model scores, and merges them for comparison.
    """
    lb_ml_path = os.path.join(PROJECT_ROOT, constants.RESULTS_ML_DATA_DIR, "final_leaderboard.csv")
    lb_simple_path = os.path.join(PROJECT_ROOT, constants.RESULTS_SIMPLE_STATS_DATA_DIR, "leaderboard_simple.csv")

    all_scores = []

    if os.path.exists(lb_ml_path):
        df_ml = pd.read_csv(lb_ml_path)
        if not df_ml.empty:
            # Melt ML individual model scores to long format
            df_ml_melted = df_ml.melt(
                id_vars=["Method", "Target", "Num_Features"],
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
    df_combined["Num_Features_Str"] = df_combined["Num_Features"].apply(lambda x: "All" if pd.isna(x) or x == "All" else f"Top{int(x)}")
    df_combined["Num_Features_Sort"] = df_combined["Num_Features"].apply(lambda x: -1 if pd.isna(x) or x == "All" else int(x))
    df_combined = df_combined.sort_values(by=["Method", "Target", "Num_Features_Sort", "Model"]).drop(columns=["Num_Features_Sort"])

    return df_combined

def generate_comparison_plots():
    """
    Generates grouped bar charts and scatter plots for ML vs Simple performance comparison.
    """
    os.makedirs(os.path.join(PROJECT_ROOT, constants.RESULTS_CLINICAL_CORRELATION_PLOTS_DIR), exist_ok=True)
    
    df_combined = load_and_merge_leaderboards()
    if df_combined is None:
        return

    # --- Plot 1: Overall Performance Comparison (Grouped Bar Charts with Models) ---
    logger.info("Generating Grouped Bar Charts for model-wise performance comparison...")

    # Using catplot for easier faceting and consistent plot styling across facets
    g = sns.catplot(
        data=df_combined, 
        x="Num_Features_Str", 
        y="Score", 
        hue="Model",      # Model (Random Forest, XGBoost, Lasso Regression, Simple Stats) as hue
        col="Method",     # Methods as columns
        row="Target",     # Targets as rows
        kind="bar",       # Bar plot
        height=3.5, aspect=1.8,  # Increased figure size for better readability
        palette="tab10",  # Color palette for models
        errorbar=None,
        sharey=True,       # Share Y-axis across plots
        sharex=False,      # Ensure x-axis labels appear for every subplot
        # legend_out=True # Default is True, puts legend outside
    )
    
    g.set_axis_labels("Feature Configuration (Number of Rules Used)", "Score") # More descriptive X-axis label
    g.set_titles("Method: {col_name} | Target: {row_name}")
    g.set_xticklabels(rotation=45, ha="right")
    g.set(ylim=(0, df_combined["Score"].max() * 1.1)) # Adjust y-limit
    
    # Add annotations to bars
    for ax in g.axes.flat:
        for container in ax.containers:
            for bar in container:
                height = bar.get_height()
                if not pd.isna(height) and height > 0: # Only annotate non-NaN/non-zero bars
                    ax.text(bar.get_x() + bar.get_width() / 2, height, 
                            f'{height:.1f}', ha='center', va='bottom', fontsize=7, color='black')

    g.fig.suptitle("Model-wise Performance Comparison (ML vs. Simple)", y=1.02) # Overall title
    
    # Move the auto-generated legend to top-right
    sns.move_legend(g, "upper right", bbox_to_anchor=(1, 1))

    plt.tight_layout(rect=[0, 0, 0.85, 0.98]) # Adjust rect to accommodate layout and legend
    
    plot_filename_models_bars = os.path.join(PROJECT_ROOT, constants.RESULTS_CLINICAL_CORRELATION_PLOTS_DIR, "performance_model_wise_bars.png")
    plt.savefig(plot_filename_models_bars)
    plt.close(g.fig) # Use g.fig to close the figure
    logger.info(f"Model-wise Grouped Bar Charts saved to {plot_filename_models_bars}")


if __name__ == "__main__":
    generate_comparison_plots()
