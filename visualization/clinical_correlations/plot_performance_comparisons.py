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

from data_exploration.check_data_bias import load_stratified_biopsies
from check_rule_correlation_with_disease.stratified_utils import CONTROLS_ELIGIBLE, filter_viable_stratum

# Setup Logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger("ComparisonPlotter")

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))

def _build_organ_rule_counts(no_self=False):
    """
    Returns {(method, organ): n_rules} — FDR-significant rules that actually
    appear in each organ's FOVs (per-organ count, not global pool size).
    """
    try:
        df_fovs = pd.read_csv(os.path.join(PROJECT_ROOT, constants.MIBI_GUT_DIR_PATH, "fovs_metadata.csv"))
        from visualization.utils.visualization_util import add_organ_column
        df_fovs = add_organ_column(df_fovs, os.path.join(PROJECT_ROOT, constants.MIBI_GUT_DIR_PATH))
        fov_organ_map = df_fovs.set_index("FOV")["Organ"].dropna()
    except Exception as e:
        logger.warning(f"Could not build organ rule counts: {e}")
        return {}

    base_input_dir = os.path.join(PROJECT_ROOT, constants.RESULTS_DATA_DIR)
    counts = {}
    for method in constants.METHODS:
        input_file = os.path.join(base_input_dir, f"results_{method}.csv")
        if not os.path.exists(input_file):
            continue
        df_rules = pd.read_csv(input_file)
        if "Rule" not in df_rules.columns:
            df_rules["Rule"] = df_rules["Antecedents"] + " -> " + df_rules["Consequents"]

        if no_self:
            from check_rule_correlation_with_disease.stratified_utils import parse_rule_items
            def _has_overlap(row):
                return not parse_rule_items(row["Antecedents"]).isdisjoint(parse_rule_items(row["Consequents"]))
            df_rules = df_rules[~df_rules.apply(_has_overlap, axis=1)]

        if "FDR" in df_rules.columns:
            df_rules = df_rules[df_rules["FDR"] < 0.05]

        df_rules["Organ"] = df_rules["FOV"].map(fov_organ_map)
        for organ, grp in df_rules.groupby("Organ"):
            if pd.isna(organ) or organ == "Unknown":
                continue
            counts[(method, str(organ))] = int(grp["Rule"].nunique())

    return counts


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

    # Attach per-organ rule counts so the subtitle shows why Top50/All produce identical
    # scores (both use the same N organ-specific rules when fewer than 50 exist).
    # N_Rules = min(top_k, n_organ_rules) — the actual number of features the model could use.
    organ_rule_counts = _build_organ_rule_counts(no_self=no_self)

    def _lookup_n_rules(row):
        n_organ = organ_rule_counts.get((str(row["Method"]), str(row["Organ"])))
        if n_organ is None:
            return None
        top_k = row["Num_Features"]
        if pd.isna(top_k) or top_k == "All":
            return n_organ
        return min(int(top_k), n_organ)

    config_cols = ["Method", "Organ", "Target", "Num_Features", "Approach"]
    unique_configs = df_combined[config_cols].drop_duplicates()
    unique_configs["N_Rules"] = unique_configs.apply(_lookup_n_rules, axis=1)
    df_combined = df_combined.merge(unique_configs, on=config_cols, how="left")

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
    
    def get_target_groups_str(target):
        counts = set(df_combined[df_combined["Target"] == target]["Group_Count"].astype(str))
        counts = {c for c in counts if c != "?"}
        if len(counts) == 1:
            return f"{list(counts)[0]} classes"
        elif len(counts) > 1:
            return f"classes: {','.join(sorted(counts))}"
        return "unknown classes"

    df_combined["Method_Target"] = df_combined.apply(lambda row: f"{row['Method']} | {row['Target']} ({get_target_groups_str(row['Target'])})", axis=1)

    # Using catplot for easier faceting and consistent plot styling across facets
    g = sns.catplot(
        data=df_combined, 
        x="Organ_Features", # Group count removed from x-axis 
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

        # Subtitle: show the actual number of rules used per x-group so it is immediately
        # clear when Top50/Top100/All are identical because fewer rules exist than the cutoff.
        ax_method_target = ax.get_title()
        x_labels = [t.get_text() for t in ax.get_xticklabels() if t.get_text()]
        if x_labels:
            rule_count_parts = []
            for lbl in x_labels:
                # lbl is "Organ | TopK" or "Organ | All" — look it up in df_combined
                match = df_combined[
                    (df_combined["Method_Target"] == ax_method_target) &
                    (df_combined["Organ_Features"] == lbl)
                ]["N_Rules"]
                n = int(match.iloc[0]) if not match.empty and pd.notna(match.iloc[0]) else "?"
                rule_count_parts.append(f"{lbl}: {n}r")
            ax.set_title(
                f"{ax_method_target}\n"
                + "Rules used — " + " | ".join(rule_count_parts),
                fontsize=9
            )

    def get_relevant_rules_subtitle():
        counts = _build_organ_rule_counts(no_self=no_self)
        by_method = {}
        for (method, organ), n in counts.items():
            by_method.setdefault(method, []).append(f"{organ}: {n}")
        parts = [f"{m} [{', '.join(sorted(organs))}]" for m, organs in sorted(by_method.items())]
        return "Relevant Rules (FDR < 0.05): " + " | ".join(parts)

    suffix_title = "(NO SELF)" if no_self else "(STANDARD)"
    rules_subtitle = get_relevant_rules_subtitle()
    g.fig.suptitle(f"Model-wise Performance Comparison {suffix_title}\n{rules_subtitle}", y=1.05, fontsize=11) 
    
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
