import argparse
import ast
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# Add project root to sys.path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(PROJECT_ROOT)

from constants import (
    CONSENSUS_RESULTS_EXPLORATION_DIR,
    METHODS,
    MIBI_GUT_DIR_PATH,
    RESULTS_DATA_DIR,
    RESULTS_PLOTS_DIR,
)
from visualization.utils.heatmap_util import (
    add_shared_row_design,
    create_metadata_figure,
    plot_rule_lift_violin,
    plot_stage_stats_table,
    prepare_stage_heatmap_data,
)
from visualization.utils.visualization_util import filter_no_self_rules, merge_biopsy_metadata

# Input Directory from where we just saved the consensus tables
CONSENSUS_DIR = os.path.join(PROJECT_ROOT, CONSENSUS_RESULTS_EXPLORATION_DIR)
# Output Directory for plots
PLOTS_DIR = os.path.join(PROJECT_ROOT, RESULTS_PLOTS_DIR, "consensus_report")

sns.set_theme(style="whitegrid")


def ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)


def load_data(method, level, organ=None, suffix=""):
    """
    Loads consensus data.
    level: 'global', 'stage', or 'biopsy'
    organ: Optional organ name (e.g., 'Colon', 'Duodenum'). If provided, loads organ-specific file.
    """
    if organ:
        filename = f"{method}_top_consensus_{level}_{organ}{suffix}.csv"
    else:
        filename = f"{method}_top_consensus_{level}{suffix}.csv"

    filepath = os.path.join(CONSENSUS_DIR, filename)
    if os.path.exists(filepath):
        return pd.read_csv(filepath)

    print(f"Warning: File not found {filepath}")
    return None


def load_method_results(method, no_self=False, organ=None):
    """Load raw method results used for metadata panels (violin + mean/std table)."""
    method_file = os.path.join(PROJECT_ROOT, RESULTS_DATA_DIR, f"results_{method}.csv")
    if not os.path.exists(method_file):
        print(f"Warning: File not found {method_file}")
        return None

    method_data = pd.read_csv(method_file)
    method_data = merge_biopsy_metadata(method_data, os.path.join(PROJECT_ROOT, MIBI_GUT_DIR_PATH))

    if no_self:
        method_data = filter_no_self_rules(method_data)

    if organ and "Organ" in method_data.columns:
        method_data = method_data[method_data["Organ"] == organ].copy()

    return method_data


def clean_rule_name(ant, con):
    """
    Cleans rule name for better visualization.
    Removes [' '] and _CENTER/_NEIGHBOR suffixes for display.
    """
    try:
        if isinstance(ant, str) and ant.startswith("["):
            ant = ast.literal_eval(ant)
        if isinstance(con, str) and con.startswith("["):
            con = ast.literal_eval(con)

        ant_clean = [str(value).replace("_CENTER", "").replace("_NEIGHBOR", "") for value in ant]
        con_clean = [str(value).replace("_CENTER", "").replace("_NEIGHBOR", "") for value in con]
        return f"{', '.join(ant_clean)} -> {', '.join(con_clean)}"
    except (ValueError, SyntaxError, TypeError):
        return f"{ant} -> {con}"


def select_consensus_rules(stage_consensus_data, stage_column, rules_per_stage=10):
    """
    Select rules from the stage consensus score matrix.
    We clean rule names, keep the max score per (Rule, Stage), then take top-N
    per stage and use their union so each stage contributes rules.
    """
    if stage_column not in stage_consensus_data.columns:
        return [], pd.DataFrame()

    stage_data = stage_consensus_data.copy()
    stage_data[stage_column] = (
        stage_data[stage_column].fillna(-1).astype(int).astype(str).replace("-1", "Unknown")
    )
    stage_data["Rule"] = stage_data.apply(
        lambda row: clean_rule_name(row["Antecedents"], row["Consequents"]),
        axis=1,
    )

    stage_rule_scores = stage_data.groupby(["Rule", stage_column], as_index=False)["Consensus_Score"].max()
    stage_rule_matrix = (
        stage_rule_scores.pivot(index="Rule", columns=stage_column, values="Consensus_Score").fillna(0)
    )

    selected_rules = []
    for stage in stage_rule_matrix.columns:
        stage_top_rules = stage_rule_matrix.nlargest(rules_per_stage, stage).index.tolist()
        for rule in stage_top_rules:
            if rule not in selected_rules:
                selected_rules.append(rule)

    return selected_rules, stage_rule_matrix


def plot_stage_consensus_heatmap(method, suffix, output_dir, organ=None):
    """
    Plots stage consensus heatmap with metadata panels.
    Heatmap color is Consensus Score (old behavior), while side panels show lift metadata.
    """
    stage_data = load_data(method, "stage", organ=organ, suffix=suffix)
    if stage_data is None or stage_data.empty:
        return

    stage_column = "Pathological stage"
    if stage_column not in stage_data.columns:
        print(f"[{method}] Missing '{stage_column}' column. Skipping.")
        return

    stage_data[stage_column] = stage_data[stage_column].fillna(-1).astype(int).astype(str).replace("-1", "Unknown")
    stage_counts = (
        stage_data[[stage_column, "Total_FOVs_In_Stage"]]
        .drop_duplicates()
        .set_index(stage_column)["Total_FOVs_In_Stage"]
        .to_dict()
    )

    rules_per_stage = 10
    selected_rules, stage_rule_matrix = select_consensus_rules(
        stage_consensus_data=stage_data,
        stage_column=stage_column,
        rules_per_stage=rules_per_stage,
    )
    if not selected_rules:
        print(f"[{method}] No consensus rules were selected. Skipping.")
        return

    stage_rule_matrix = stage_rule_matrix.loc[selected_rules]
    stage_rule_matrix["max_score"] = stage_rule_matrix.max(axis=1)
    stage_rule_matrix = stage_rule_matrix.sort_values("max_score", ascending=False).drop(columns="max_score")

    method_data = load_method_results(method, no_self=(suffix == "_no_self"), organ=organ)
    if method_data is None or method_data.empty:
        print(f"[{method}] No raw result data for metadata panels. Skipping.")
        return
    if stage_column not in method_data.columns:
        print(f"[{method}] Missing '{stage_column}' in raw result data. Skipping.")
        return

    method_data[stage_column] = (
        method_data[stage_column].fillna(-1).astype(int).astype(str).replace("-1", "Unknown")
    )
    method_data["Rule"] = method_data.apply(
        lambda row: clean_rule_name(row["Antecedents"], row["Consequents"]),
        axis=1,
    )

    valid_stages = [str(stage) for stage in stage_rule_matrix.columns]
    method_data = method_data[method_data[stage_column].isin(valid_stages)].copy()
    if method_data.empty:
        print(f"[{method}] No raw rows match selected stages. Skipping.")
        return

    metadata_data = prepare_stage_heatmap_data(
        method_data,
        stage_column,
        valid_stages,
        minimum_stage_count=1,
    )
    rule_stage_counts = metadata_data["rule_stage_counts"]
    rule_stage_mean_lift = metadata_data["rule_stage_mean_lift"]
    rule_stage_lift_std = metadata_data["rule_stage_lift_std"]

    ordered_rules = [rule for rule in stage_rule_matrix.index.tolist() if rule in rule_stage_counts.index]
    if not ordered_rules:
        print(f"[{method}] No selected rules found in raw metadata data. Skipping.")
        return

    plot_matrix = stage_rule_matrix.loc[ordered_rules, valid_stages].copy()
    renamed_columns = {
        stage: f"Stage {stage}\n(N={stage_counts.get(stage, '?')})"
        for stage in valid_stages
    }
    plot_matrix = plot_matrix.rename(columns=renamed_columns)

    figure, heatmap_axis, violin_axis, summary_axis, colorbar_axis = create_metadata_figure(
        valid_stages,
        len(ordered_rules),
    )

    sns.heatmap(
        plot_matrix,
        annot=True,
        fmt=".2f",
        cmap="YlOrRd",
        cbar_ax=colorbar_axis,
        cbar_kws={"label": "Consensus Score"},
        linewidths=0.5,
        linecolor="lightgray",
        ax=heatmap_axis,
    )

    title = f"Stage Consensus Heatmap ({method}{suffix})"
    if organ:
        title += f" - {organ}"
    title += f"\n(Top {rules_per_stage} per Stage, {len(ordered_rules)} unique)"
    if "Stratum_Viable" in stage_data.columns:
        non_viable_count = (~stage_data["Stratum_Viable"]).sum()
        if non_viable_count > 0:
            title += f"\n⚠ Warning: {non_viable_count} stage strata have low counts or high imbalance"

    heatmap_axis.set_title(title, fontsize=14, pad=35)
    heatmap_axis.set_xlabel("")
    heatmap_axis.set_ylabel("Rule")

    plot_rule_lift_violin(violin_axis, method_data, ordered_rules)
    plot_stage_stats_table(
        summary_axis,
        ordered_rules,
        valid_stages,
        rule_stage_counts,
        rule_stage_mean_lift,
        rule_stage_lift_std,
    )
    add_shared_row_design(
        heatmap_axis,
        violin_axis,
        summary_axis,
        row_count=len(ordered_rules),
        stage_count=len(valid_stages),
    )

    organ_suffix = f"_{organ}" if organ else ""
    save_path = os.path.join(output_dir, f"heatmap_stage_consensus_{method}{organ_suffix}{suffix}.png")
    plt.savefig(save_path, bbox_inches="tight")
    plt.close(figure)
    print(f"  Selected {len(ordered_rules)} unique rules ({rules_per_stage} per stage)")
    print(f"Saved {save_path}")


def plot_biopsy_jaccard_similarity(method, suffix, output_dir, organ=None):
    """
    Calculates and plots Jaccard Similarity between Biopsies based on their consensus rules.
    """
    data_frame = load_data(method, "biopsy", organ=organ, suffix=suffix)
    if data_frame is None or data_frame.empty:
        return

    data_frame["Rule"] = data_frame["Antecedents"] + "->" + data_frame["Consequents"]
    biopsy_rules = data_frame.groupby("Biopsy_ID")["Rule"].apply(set).to_dict()

    biopsies = sorted(list(biopsy_rules.keys()))
    biopsy_count = len(biopsies)
    similarity_matrix = np.zeros((biopsy_count, biopsy_count))

    for row_index in range(biopsy_count):
        for col_index in range(biopsy_count):
            first_set = biopsy_rules[biopsies[row_index]]
            second_set = biopsy_rules[biopsies[col_index]]

            if not first_set and not second_set:
                similarity_value = 0
            else:
                overlap_count = len(first_set.intersection(second_set))
                union_count = len(first_set.union(second_set))
                similarity_value = overlap_count / union_count if union_count > 0 else 0
            similarity_matrix[row_index, col_index] = similarity_value

    matrix_data = pd.DataFrame(similarity_matrix, index=biopsies, columns=biopsies)
    if similarity_matrix.sum() == 0:
        print(f"Skipping clustermap for {method}: No similarity found.")
        return

    try:
        cluster_map = sns.clustermap(matrix_data, cmap="viridis", figsize=(12, 12), vmin=0, vmax=1)
        title = f"Biopsy Similarity ClusterMap ({method}{suffix})"
        if organ:
            title += f" - {organ}"
        cluster_map.fig.suptitle(title, fontsize=16, y=1.02)

        organ_suffix = f"_{organ}" if organ else ""
        save_path = os.path.join(
            output_dir,
            f"clustermap_biopsy_similarity_{method}{organ_suffix}{suffix}.png",
        )
        plt.savefig(save_path)
        plt.close()
        print(f"Saved {save_path}")
    except Exception as error:
        print(f"Error plotting clustermap for {method}: {error}")


def plot_global_consensus_bar(method, suffix, output_dir, organ=None):
    """
    Plots top global consensus rules with counts.
    """
    data_frame = load_data(method, "global", organ=organ, suffix=suffix)
    if data_frame is None or data_frame.empty:
        return

    data_frame["Rule"] = data_frame.apply(
        lambda row: clean_rule_name(row["Antecedents"], row["Consequents"]), axis=1
    )

    top_data = data_frame.head(20).copy()
    if "FOV_Count" in top_data.columns and "Total_FOVs_In_Dataset" in top_data.columns:
        total_fovs = top_data["Total_FOVs_In_Dataset"].iloc[0]
    else:
        total_fovs = "?"

    plt.figure(figsize=(14, 8))
    axis = sns.barplot(data=top_data, y="Rule", x="Consensus_Score", palette="Blues_r")

    for bar_index, bar in enumerate(axis.patches):
        if bar_index < len(top_data):
            row = top_data.iloc[bar_index]
            count = row["FOV_Count"]
            axis.text(
                bar.get_width() + 0.01,
                bar.get_y() + bar.get_height() / 2,
                f"n={count}",
                va="center",
                fontsize=10,
                color="black",
            )

    title = f"Top 20 Global Consensus Rules ({method}{suffix})"
    if organ:
        title += f" - {organ}"
    title += f"\nTotal FOVs in Dataset: {total_fovs}"
    plt.title(title, fontsize=14)
    plt.xlabel("Consensus Score (Fraction of FOVs)")
    plt.xlim(0, 1.25)
    plt.tight_layout()

    organ_suffix = f"_{organ}" if organ else ""
    save_path = os.path.join(output_dir, f"bar_global_consensus_{method}{organ_suffix}{suffix}.png")
    plt.savefig(save_path, bbox_inches="tight")
    plt.close()
    print(f"Saved {save_path}")


def main():
    parser = argparse.ArgumentParser(description="Visualize Consensus Tables.")
    parser.add_argument(
        "--no_self",
        action="store_true",
        help="Use tables generated with --no_self flag.",
    )
    parser.add_argument(
        "--organs",
        nargs="+",
        help="Optional: Specific organs to visualize (e.g., --organs Colon Duodenum). If not provided, auto-discovers from files.",
    )
    args = parser.parse_args()

    suffix = "_no_self" if args.no_self else ""

    ensure_dir(PLOTS_DIR)
    print(f"Saving plots to: {PLOTS_DIR}")
    print(f"Using suffix: '{suffix}'")

    import glob

    pattern = os.path.join(CONSENSUS_DIR, f"*_top_consensus_global_*{suffix}.csv")
    files = glob.glob(pattern)

    discovered_organs = set()
    for file_path in files:
        basename = os.path.basename(file_path)
        parts = basename.replace(suffix, "").replace(".csv", "").split("_")
        try:
            global_index = parts.index("global")
            if global_index + 1 < len(parts):
                discovered_organs.add(parts[global_index + 1])
        except ValueError:
            continue

    organs_to_process = args.organs if args.organs else sorted(discovered_organs)

    if not organs_to_process:
        print("No organ-stratified files found. Falling back to legacy (non-stratified) mode.")
        for method in METHODS:
            print(f"\n--- Visualizing {method} (Legacy) ---")
            plot_stage_consensus_heatmap(method, suffix, PLOTS_DIR)
            plot_biopsy_jaccard_similarity(method, suffix, PLOTS_DIR)
            plot_global_consensus_bar(method, suffix, PLOTS_DIR)
    else:
        print(f"\nDiscovered organs: {discovered_organs}")
        print(f"Processing organs: {organs_to_process}")

        for organ in organs_to_process:
            print(f"\n{'=' * 60}")
            print(f"ORGAN: {organ}")
            print(f"{'=' * 60}")

            for method in METHODS:
                print(f"\n--- Visualizing {method} - {organ} ---")
                plot_stage_consensus_heatmap(method, suffix, PLOTS_DIR, organ=organ)
                plot_biopsy_jaccard_similarity(method, suffix, PLOTS_DIR, organ=organ)
                plot_global_consensus_bar(method, suffix, PLOTS_DIR, organ=organ)


if __name__ == "__main__":
    main()
