import argparse
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# Add project root to sys.path to import constants
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from constants import MIBI_GUT_DIR_PATH, METHODS, RESULTS_DATA_DIR, RESULTS_PLOTS_DIR
from visualization.utils.heatmap_util import (
    add_shared_row_design,
    build_rule_annotations,
    create_metadata_figure,
    get_stage_label,
    get_stage_sort_key,
    get_valid_stages,
    plot_rule_lift_violin,
    plot_stage_stats_table,
    prepare_stage_heatmap_data,
)
from visualization.utils.visualization_util import filter_no_self_rules, merge_biopsy_metadata

sns.set_theme(style="whitegrid")


def load_data(no_self=False):
    """Load method results, merge metadata, and split by organ when available."""
    data_by_method = {}
    print(f"Loading data from: {RESULTS_DATA_DIR}")

    for method in METHODS:
        csv_path = os.path.join(RESULTS_DATA_DIR, f"results_{method}.csv")
        if not os.path.exists(csv_path):
            continue

        try:
            method_data = pd.read_csv(csv_path)
            method_data = merge_biopsy_metadata(method_data, MIBI_GUT_DIR_PATH)

            if "Rule" not in method_data.columns:
                method_data["Rule"] = method_data["Antecedents"] + " -> " + method_data["Consequents"]

            if no_self:
                method_data = filter_no_self_rules(method_data)

            if "Organ" in method_data.columns:
                organ_values = [
                    organ
                    for organ in sorted(method_data["Organ"].dropna().unique())
                    if str(organ) != "Unknown"
                ]
                if organ_values:
                    for organ in organ_values:
                        organ_data = method_data[method_data["Organ"] == organ].copy()
                        if organ_data.empty:
                            continue
                        safe_organ = str(organ).replace("/", "_").replace(" ", "_")
                        data_by_method[f"{method}_{safe_organ}"] = organ_data
                    continue

            data_by_method[method] = method_data

        except Exception as error:
            print(f"Error loading {method} CSV: {error}")

    return data_by_method


def select_stage_marker_rules(stage_heatmap_data, valid_stages, rules_per_stage):
    """
    Select marker rules by comparing each stage against all other stages.
    The score combines stage prevalence difference, stage-vs-rest lift ratio, and
    observation depth. For each stage, we keep top positive and top negative rules.
    """
    stage_fov_counts = stage_heatmap_data["stage_fov_counts"]
    rule_stage_counts = stage_heatmap_data["rule_stage_counts"]
    rule_stage_mean_lift = stage_heatmap_data["rule_stage_mean_lift"]
    eligible_rules = stage_heatmap_data["eligible_rules"]

    selected_rules = []
    rules_each_side = max(1, rules_per_stage // 2)

    for stage in valid_stages:
        stage_total_fovs = stage_fov_counts.get(stage, 1)
        stage_prevalence = rule_stage_counts[stage] / stage_total_fovs
        stage_mean_lift = rule_stage_mean_lift[stage]

        other_stages = [other_stage for other_stage in valid_stages if other_stage != stage]
        other_total_fovs = sum(stage_fov_counts.get(other_stage, 0) for other_stage in other_stages)
        other_prevalence = rule_stage_counts[other_stages].sum(axis=1) / max(1, other_total_fovs)
        other_mean_lift = rule_stage_mean_lift[other_stages].mean(axis=1)

        prevalence_gap = stage_prevalence - other_prevalence
        lift_ratio = np.log2((stage_mean_lift + 1e-6) / (other_mean_lift + 1e-6)).clip(-3, 3)
        observation_depth = np.sqrt(
            rule_stage_counts[stage] + rule_stage_counts[other_stages].sum(axis=1)
        )

        selection_score = prevalence_gap * np.abs(lift_ratio) * observation_depth
        selection_score = selection_score.loc[eligible_rules]

        top_positive_rules = selection_score.nlargest(rules_each_side).index.tolist()
        top_negative_rules = selection_score.nsmallest(rules_each_side).index.tolist()

        for rule in top_positive_rules + top_negative_rules:
            if rule not in selected_rules:
                selected_rules.append(rule)

    return selected_rules


def build_interaction_score(rule_stage_mean_lift, rule_stage_counts, valid_stages):
    """Compute stage-marker color values from mean lift with low-count muting."""
    interaction_score = pd.DataFrame(index=rule_stage_counts.index, columns=valid_stages, dtype=float)

    for stage in valid_stages:
        stage_rule_count = rule_stage_counts[stage]
        low_count_weight = np.where(stage_rule_count < 3, stage_rule_count / 3.0, 1.0)
        interaction_score[stage] = np.log2(rule_stage_mean_lift[stage] + 1e-6) * low_count_weight

    return interaction_score


def order_rules_by_peak_stage(plot_score):
    """Order rules by strongest stage signal to improve diagonal readability."""
    peak_stage = plot_score.abs().idxmax(axis=1)
    row_summary = pd.DataFrame(
        {
            "rule": plot_score.index.tolist(),
            "peak_stage": peak_stage,
            "peak_score": [plot_score.loc[rule, peak_stage.loc[rule]] for rule in plot_score.index],
        }
    )
    row_summary["stage_sort_key"] = row_summary["peak_stage"].map(get_stage_sort_key)
    row_summary = row_summary.sort_values(["stage_sort_key", "peak_score"], ascending=[True, False])
    return row_summary["rule"].tolist()


def render_stage_marker_heatmap(
    method_data,
    method_name,
    valid_stages,
    selected_rules,
    stage_heatmap_data,
    save_path,
):
    """Render stage-marker heatmap with metadata panels."""
    available_rules = [
        rule for rule in selected_rules if rule in stage_heatmap_data["rule_stage_counts"].index
    ]
    if not available_rules:
        return False

    stage_fov_counts = stage_heatmap_data["stage_fov_counts"]
    rule_stage_counts = stage_heatmap_data["rule_stage_counts"]
    rule_stage_mean_lift = stage_heatmap_data["rule_stage_mean_lift"]
    rule_stage_lift_std = stage_heatmap_data["rule_stage_lift_std"]

    interaction_score = build_interaction_score(rule_stage_mean_lift, rule_stage_counts, valid_stages)
    plot_score = interaction_score.loc[available_rules, valid_stages].astype(float)
    annotation_data = build_rule_annotations(
        available_rules,
        valid_stages,
        rule_stage_counts,
        stage_fov_counts,
    )
    ordered_rules = order_rules_by_peak_stage(plot_score)

    figure, heatmap_axis, violin_axis, summary_axis, colorbar_axis = create_metadata_figure(
        valid_stages,
        len(ordered_rules),
    )

    max_score = np.nanmax(plot_score.abs().values)
    if max_score < 0.5:
        max_score = 0.5

    zero_count_mask = (rule_stage_counts.loc[ordered_rules, valid_stages] == 0).values

    sns.heatmap(
        plot_score.loc[ordered_rules],
        annot=annotation_data.loc[ordered_rules],
        fmt="",
        cmap="RdBu_r",
        center=0,
        vmin=-max_score,
        vmax=max_score,
        ax=heatmap_axis,
        cbar_ax=colorbar_axis,
        mask=zero_count_mask,
        linewidths=0.5,
        linecolor="lightgray",
        annot_kws={"size": 8},
    )

    heatmap_axis.set_facecolor("whitesmoke")
    heatmap_axis.grid(False)
    heatmap_axis.set_title(
        f"Interaction Strength ({method_name})\n"
        "Red: Lift > 1, Blue: Lift < 1 | Saturation: log2(Lift) | Muted if k < 3",
        fontsize=14,
        pad=35,
    )
    heatmap_axis.set_xticklabels(
        [f"{get_stage_label(stage)}\n(N={stage_fov_counts.get(stage, 0)})" for stage in valid_stages],
        rotation=0,
    )
    heatmap_axis.set_ylabel("Marker Rule")

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

    plt.savefig(save_path, bbox_inches="tight")
    plt.close(figure)
    return True


def plot_stage_marker_rules(data_by_method, output_dir, rules_per_stage=10, no_self=False, min_n_stages=2):
    """Create stage-marker heatmaps using local rendering and marker-specific selection."""
    for method_name, method_data in data_by_method.items():
        if method_data.empty:
            continue

        stage_column = "Pathological stage" if "Pathological stage" in method_data.columns else "Group"
        method_data[stage_column] = (
            method_data[stage_column].fillna(-1).astype(int).astype(str).replace("-1", "Unknown")
        )
        valid_stages = get_valid_stages(method_data, stage_column)
        if len(valid_stages) < 2:
            continue

        stage_heatmap_data = prepare_stage_heatmap_data(
            method_data,
            stage_column,
            valid_stages,
            minimum_stage_count=min_n_stages,
        )
        if not stage_heatmap_data["eligible_rules"]:
            print(f"[{method_name}] No rules meet min_n_stages={min_n_stages}. Skipping.")
            continue

        selected_rules = select_stage_marker_rules(stage_heatmap_data, valid_stages, rules_per_stage)
        if not selected_rules:
            print(f"[{method_name}] No marker rules were selected. Skipping.")
            continue

        safe_stage_column = str(stage_column).replace("/", "_").replace(" ", "_")
        no_self_suffix = "_no_self" if no_self else ""
        save_path = os.path.join(
            output_dir,
            f"heatmap_stage_marker_rules_{safe_stage_column}_{method_name}{no_self_suffix}.png",
        )

        was_saved = render_stage_marker_heatmap(
            method_data=method_data,
            method_name=method_name,
            valid_stages=valid_stages,
            selected_rules=selected_rules,
            stage_heatmap_data=stage_heatmap_data,
            save_path=save_path,
        )
        if was_saved:
            print(f"Saved {save_path}")


def main():
    parser = argparse.ArgumentParser(description="Generate Stage Marker Rules Heatmaps")
    parser.add_argument("--top_n", type=int, default=10, help="Top N rules per stage to display")
    parser.add_argument("--no_self", action="store_true", help="Filter out self-loops")
    parser.add_argument(
        "--min_n_stages",
        type=int,
        default=2,
        help="Minimum number of stages a rule must appear in.",
    )
    args = parser.parse_args()

    output_dir = os.path.join(RESULTS_PLOTS_DIR, "mining_report")
    os.makedirs(output_dir, exist_ok=True)

    data_by_method = load_data(no_self=args.no_self)
    if not data_by_method:
        print("No data found. Exiting.")
        return

    min_n_stages = max(1, args.min_n_stages)
    plot_stage_marker_rules(
        data_by_method,
        output_dir,
        rules_per_stage=args.top_n,
        no_self=args.no_self,
        min_n_stages=min_n_stages,
    )
    print("Visualization Complete.")


if __name__ == "__main__":
    main()
