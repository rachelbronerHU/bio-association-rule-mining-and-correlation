import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def get_stage_sort_key(stage_value):
    """Sort numeric stage values first, then non-numeric values."""
    try:
        return (0, int(stage_value))
    except (TypeError, ValueError):
        return (1, str(stage_value))


def get_stage_label(stage_value):
    """Return a user-friendly stage label."""
    if str(stage_value) == "0":
        return "Control"
    return f"Stage {stage_value}"


def get_valid_stages(data_frame, stage_column):
    """Return sorted stage values, excluding Unknown."""
    stage_values = [value for value in data_frame[stage_column].unique() if value != "Unknown"]
    return sorted(stage_values, key=get_stage_sort_key)


def prepare_stage_heatmap_data(data_frame, stage_column, valid_stages, minimum_stage_count=1):
    """Build shared stage-level metadata matrices used by visualizations."""
    stage_fov_counts = data_frame.drop_duplicates("FOV")[stage_column].value_counts().to_dict()

    rule_stage_counts = data_frame.pivot_table(
        index="Rule",
        columns=stage_column,
        values="FOV",
        aggfunc="nunique",
        fill_value=0,
    )
    # Mean/STD are computed per (Rule, Stage) from rows where the rule is present in an FOV.
    # FOVs where the rule does not appear are not included in these aggregates.
    # Missing (Rule, Stage) cells are filled with 0 after pivot so the plot matrices stay rectangular.
    rule_stage_mean_lift = data_frame.pivot_table(
        index="Rule",
        columns=stage_column,
        values="Lift",
        aggfunc="mean",
        fill_value=0,
    )
    rule_stage_lift_std = data_frame.pivot_table(
        index="Rule",
        columns=stage_column,
        values="Lift",
        aggfunc="std",
        fill_value=0,
    ).fillna(0)

    all_rules = rule_stage_counts.index
    rule_stage_counts = rule_stage_counts.reindex(index=all_rules, columns=valid_stages, fill_value=0)
    rule_stage_mean_lift = rule_stage_mean_lift.reindex(index=all_rules, columns=valid_stages, fill_value=0)
    rule_stage_lift_std = rule_stage_lift_std.reindex(index=all_rules, columns=valid_stages, fill_value=0)

    stage_presence_count = (rule_stage_counts[valid_stages] > 0).sum(axis=1)
    eligible_rules = stage_presence_count[stage_presence_count >= minimum_stage_count].index.tolist()

    return {
        "stage_fov_counts": stage_fov_counts,
        "rule_stage_counts": rule_stage_counts,
        "rule_stage_mean_lift": rule_stage_mean_lift,
        "rule_stage_lift_std": rule_stage_lift_std,
        "eligible_rules": eligible_rules,
    }


def build_rule_annotations(selected_rules, valid_stages, rule_stage_counts, stage_fov_counts):
    """Build k/N text annotations for each rule and stage."""
    annotation_data = pd.DataFrame(index=selected_rules, columns=valid_stages)
    for rule in selected_rules:
        for stage in valid_stages:
            stage_count = int(rule_stage_counts.loc[rule, stage])
            stage_total = stage_fov_counts.get(stage, 0)
            annotation_data.loc[rule, stage] = f"{stage_count}/{stage_total}"
    return annotation_data


def create_metadata_figure(valid_stages, rule_count):
    """Create the shared 4-panel figure layout used by stage heatmaps."""
    figure_height = max(10, rule_count * 0.4)
    figure = plt.figure(figsize=(24, figure_height))
    grid = figure.add_gridspec(
        1,
        4,
        width_ratios=[len(valid_stages) * 1.5, 3, len(valid_stages) * 1.1, 0.2],
        wspace=0.1,
    )
    heatmap_axis = figure.add_subplot(grid[0, 0])
    violin_axis = figure.add_subplot(grid[0, 1])
    summary_axis = figure.add_subplot(grid[0, 2])
    colorbar_axis = figure.add_subplot(grid[0, 3])
    return figure, heatmap_axis, violin_axis, summary_axis, colorbar_axis


def plot_rule_lift_violin(violin_axis, data_frame, ordered_rules):
    """Draw the shared violin panel of FOV-level lift distributions."""
    violin_data = []
    for rule in ordered_rules:
        rule_lifts = data_frame[data_frame["Rule"] == rule]["Lift"].dropna().values
        violin_data.append(rule_lifts if len(rule_lifts) > 0 else np.array([0.0]))

    violin_positions = np.arange(len(ordered_rules)) + 0.5
    violin_parts = violin_axis.violinplot(
        violin_data,
        positions=violin_positions,
        vert=False,
        widths=0.8,
        showextrema=True,
        showmedians=True,
    )
    for body in violin_parts["bodies"]:
        body.set_facecolor("#a6cee3")
        body.set_edgecolor("black")
        body.set_linewidth(0.5)
        body.set_alpha(0.8)
    for part_name in ("cbars", "cmins", "cmaxes", "cmedians"):
        if part_name in violin_parts:
            violin_parts[part_name].set_edgecolor("black")
            violin_parts[part_name].set_linewidth(1)

    for rule_index, rule_lifts in enumerate(violin_data):
        if len(rule_lifts) > 0:
            y_jitter = (rule_index + 0.5) + np.random.uniform(-0.1, 0.1, size=len(rule_lifts))
            violin_axis.scatter(rule_lifts, y_jitter, color="black", alpha=0.35, s=2, zorder=3)

    violin_axis.set_ylim(len(ordered_rules), 0)
    violin_axis.set_yticks([])
    violin_axis.set_yticklabels([])
    violin_axis.spines["left"].set_visible(False)
    violin_axis.spines["right"].set_visible(False)
    violin_axis.spines["top"].set_visible(False)
    violin_axis.set_title("Global Lift Distribution\n ", fontsize=14, pad=35)
    violin_axis.set_xlabel("Lift")
    violin_axis.grid(False)


def plot_stage_stats_table(
    summary_axis,
    ordered_rules,
    valid_stages,
    rule_stage_counts,
    rule_stage_mean_lift,
    rule_stage_lift_std,
):
    """Draw the shared stage statistics table: mean lift ± std."""
    summary_axis.axis("off")
    summary_axis.set_ylim(len(ordered_rules), 0)
    summary_axis.set_xlim(0, len(valid_stages))
    summary_axis.grid(False)

    for stage_index, stage in enumerate(valid_stages):
        summary_axis.text(
            stage_index + 0.5,
            -0.1,
            f"Stage {stage}\nMean ± STD",
            fontweight="bold",
            fontsize=10,
            ha="center",
            va="bottom",
        )

    for row_index, rule in enumerate(ordered_rules):
        for stage_index, stage in enumerate(valid_stages):
            if rule_stage_counts.loc[rule, stage] > 0:
                stage_mean = rule_stage_mean_lift.loc[rule, stage]
                stage_std = rule_stage_lift_std.loc[rule, stage]
                summary_axis.text(
                    stage_index + 0.5,
                    row_index + 0.5,
                    f"{stage_mean:.2f}\n±{stage_std:.2f}",
                    fontsize=8,
                    va="center",
                    ha="center",
                )

    summary_axis.set_title("Stage Statistics\nMean Lift ± STD", fontsize=14, pad=35)


def add_shared_row_design(heatmap_axis, violin_axis, summary_axis, row_count, stage_count):
    """Draw shared zebra striping and row guide lines across all three panels."""
    for row_index in range(row_count):
        if row_index % 2 == 1:
            heatmap_axis.add_patch(
                plt.Rectangle(
                    (0, row_index),
                    stage_count,
                    1,
                    facecolor="whitesmoke",
                    alpha=0.3,
                    zorder=-1,
                )
            )
            violin_axis.axhspan(row_index, row_index + 1, facecolor="whitesmoke", alpha=0.3, zorder=-1)
            summary_axis.axhspan(row_index, row_index + 1, facecolor="whitesmoke", alpha=0.3, zorder=-1)

    for y_value in range(row_count + 1):
        heatmap_axis.axhline(y_value, color="lightgray", linewidth=0.5, alpha=0.5)
        violin_axis.axhline(y_value, color="lightgray", linewidth=0.5, alpha=0.5)
        summary_axis.axhline(y_value, color="lightgray", linewidth=0.5, alpha=0.5)
