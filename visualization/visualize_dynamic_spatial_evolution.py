import sys
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import argparse
from matplotlib.lines import Line2D

sys.path.append(os.getcwd())
import constants
from visualization.utils.visualization_util import (
    merge_biopsy_metadata,
    filter_no_self_rules,
    parse_rule_list,
    format_rule_for_display,
    normalize_cell_table,
    select_representative_fov,
    get_sorted_stage_values,
    get_baseline_stage,
    get_stage_label,
    get_cell_type_palette,
    get_rule_highlight_labels,
    get_rule_item_palette,
    plot_fov_cells,
    format_spatial_axis
)

# --- Configuration ---
TOP_N_RULES = 3
STAGE_COL = 'GI stage'  # Unified 0-4 scale
TIME_COL = 'Days after Transplant grouped'
TIME_ORDER = ['<30', '30-100', '>100']
MIN_DELTA = 0.01
PSEUDO_COUNT = 0.001
# Minimum stage coverage per rule for stage-evolution selection
MIN_FOVS_PER_STAGE = 2
# Minimum time-bin coverage per rule for temporal-evolution selection
MIN_FOVS_PER_TIME_BIN = 2

def _filter_rules_by_stage_coverage(df, stage_col, stage_order, min_n_stages):
    """
    Keep rules that appear in at least `min_n_stages`, with at least
    MIN_FOVS_PER_STAGE unique FOVs in each counted stage.
    """
    if min_n_stages <= 0:
        return None
    counts = df.groupby(['Rule_ID', stage_col])['FOV'].nunique().unstack(fill_value=0)
    counts = counts.reindex(columns=stage_order, fill_value=0)
    eligible = (counts >= MIN_FOVS_PER_STAGE).sum(axis=1) >= min_n_stages
    return set(counts.index[eligible])

def _filter_rules_by_time_coverage(df, time_col, time_order, min_n_stages):
    """
    Keep rules that appear in at least `min_n_stages` time bins, with at least
    MIN_FOVS_PER_TIME_BIN unique FOVs in each counted bin.
    """
    if min_n_stages <= 0:
        return None
    counts = df.groupby(['Rule_ID', time_col])['FOV'].nunique().unstack(fill_value=0)
    counts = counts.reindex(columns=time_order, fill_value=0)
    eligible = (counts >= MIN_FOVS_PER_TIME_BIN).sum(axis=1) >= min_n_stages
    return set(counts.index[eligible])

def _top_rules_from_support_dynamics(stats_df, high_col, low_col, n):
    """Apply delta gate then fold-change ranking and return top rule IDs."""
    working = stats_df.copy()
    working['Delta'] = working[high_col] - working[low_col]
    working = working[working['Delta'] >= MIN_DELTA]
    if working.empty:
        return []
    working['Fold_Change'] = (
        (working[high_col] + PSEUDO_COUNT) /
        (working[low_col] + PSEUDO_COUNT)
    )
    return working.sort_values('Fold_Change', ascending=False).head(n).index.tolist()

def get_dynamic_rules_by_stage(df, stage_col, stage_order, baseline_stage, n=3, min_n_stages=2):
    """Identifies rules with highest support fold-change vs the baseline stage."""
    if not stage_order or baseline_stage is None:
        return []
    diseased_stages = [s for s in stage_order if s != baseline_stage]
    if not diseased_stages:
        return []

    df = df.copy()
    df['Rule_ID'] = df['Antecedents'] + " -> " + df['Consequents']
    df = df[df[stage_col].isin(stage_order)]
    eligible_rules = _filter_rules_by_stage_coverage(df, stage_col, stage_order, min_n_stages=min_n_stages)
    if eligible_rules is not None:
        df = df[df['Rule_ID'].isin(eligible_rules)]
    if df.empty:
        return []

    stage_stats = df.groupby(['Rule_ID', stage_col])['Support'].mean().unstack(fill_value=0)
    stage_stats = stage_stats.reindex(columns=stage_order, fill_value=0)
    stage_stats['Max_Diseased_Support'] = stage_stats[diseased_stages].max(axis=1)
    stage_stats['Baseline_Support'] = stage_stats[baseline_stage]
    return _top_rules_from_support_dynamics(
        stage_stats,
        high_col='Max_Diseased_Support',
        low_col='Baseline_Support',
        n=n,
    )

def get_dynamic_rules_by_time(df, time_col, time_order, n=3, min_n_stages=2):
    """Identifies temporal rules by max-min delta gate then fold-change ranking."""
    if not time_order or len(time_order) < 2:
        return []

    df = df.copy()
    df['Rule_ID'] = df['Antecedents'] + " -> " + df['Consequents']
    df = df[df[time_col].isin(time_order)]
    eligible_rules = _filter_rules_by_time_coverage(df, time_col, time_order, min_n_stages=min_n_stages)
    if eligible_rules is not None:
        df = df[df['Rule_ID'].isin(eligible_rules)]
    if df.empty:
        return []
    time_stats = df.groupby(['Rule_ID', time_col])['Support'].mean().unstack(fill_value=0)
    time_stats = time_stats.reindex(columns=time_order, fill_value=0)
    time_stats['Max_Support'] = time_stats.max(axis=1)
    time_stats['Min_Support'] = time_stats.min(axis=1)
    return _top_rules_from_support_dynamics(
        time_stats,
        high_col='Max_Support',
        low_col='Min_Support',
        n=n,
    )

def _time_bin_label(time_bin):
    if time_bin == "<30":
        return "<30 Days"
    if time_bin == "30-100":
        return "30-100 Days"
    if time_bin == ">100":
        return ">100 Days"
    return str(time_bin)

def _format_panel_stats(stats):
    """Compact stats text for panel headers."""
    if not stats:
        return "Relevant FOVs=0/0"
    n_fov = int(stats.get('n_fov', 0))
    n_fov_total = int(stats.get('n_fov_total', 0))
    parts = [f"Relevant FOVs={n_fov}/{n_fov_total}"]
    for key, label in [
        ("mean_lift", "Mean Lift"),
        ("mean_support", "Mean Support"),
        ("mean_confidence", "Mean Confidence"),
        ("mean_conviction", "Mean Conviction"),
    ]:
        val = stats.get(key, np.nan)
        if pd.notna(val):
            if np.isfinite(val):
                parts.append(f"{label}={val:.3f}")
            else:
                parts.append(f"{label}=inf")
    if len(parts) <= 3:
        return " | ".join(parts)
    return " | ".join(parts[:3]) + "\n" + " | ".join(parts[3:])

def _compute_rule_group_stats(df, rule_id, group_col, group_order):
    """Compute per-group summary stats for subplot headers."""
    sub = df[df['Rule_ID'] == rule_id].copy()
    if sub.empty:
        return {g: {"n_fov": 0} for g in group_order}

    stats_by_group = {}
    total_fov_by_group = (
        df.dropna(subset=['FOV', group_col]).groupby(group_col)['FOV'].nunique().to_dict()
        if 'FOV' in df.columns else {}
    )
    for g in group_order:
        gdf = sub[sub[group_col] == g]
        total_in_group = int(total_fov_by_group.get(g, 0))
        if gdf.empty:
            stats_by_group[g] = {"n_fov": 0, "n_fov_total": total_in_group}
            continue
        stats_by_group[g] = {
            "n_fov": int(gdf['FOV'].nunique()) if 'FOV' in gdf.columns else 0,
            "n_fov_total": total_in_group,
            "mean_lift": gdf['Lift'].mean() if 'Lift' in gdf.columns else np.nan,
            "mean_support": gdf['Support'].mean() if 'Support' in gdf.columns else np.nan,
            "mean_confidence": gdf['Confidence'].mean() if 'Confidence' in gdf.columns else np.nan,
            "mean_conviction": gdf['Conviction'].mean() if 'Conviction' in gdf.columns else np.nan,
        }
    return stats_by_group

def plot_evolution_grid(rule_id, representative_fovs, group_order, cell_df, output_path, mode, group_stats=None):
    """Renders a grid of spatial tissue maps for representative samples."""
    n_groups = len(group_order)
    fig, axes = plt.subplots(1, n_groups, figsize=(5 * n_groups, 5))
    if n_groups == 1: axes = [axes]

    ant_con = rule_id.split(" -> ")
    ant_items = parse_rule_list(ant_con[0])
    con_items = parse_rule_list(ant_con[1])
    rule_palette = get_rule_item_palette(ant_items + con_items)

    for i, group_val in enumerate(group_order):
        ax = axes[i]
        fov_id = representative_fovs.get(group_val)

        if mode == "stage":
            subplot_title = f"{get_stage_label(group_val)} (FOV: {fov_id if fov_id is not None else 'N/A'})"
        else:
            subplot_title = f"{_time_bin_label(group_val)} (FOV: {fov_id if fov_id is not None else 'N/A'})"
        stats_text = _format_panel_stats((group_stats or {}).get(group_val))
        subplot_title = f"{subplot_title}\n{stats_text}"

        if fov_id is None:
            ax.text(0.5, 0.5, "No Data", transform=ax.transAxes, ha='center')
            format_spatial_axis(ax, subplot_title)
            continue

        df_fov = cell_df[cell_df['fov'] == fov_id].copy()

        color_map = get_cell_type_palette(df_fov['cell_type'].unique())
        highlight_labels = get_rule_highlight_labels(df_fov, ant_items + con_items)
        highlight_mask = highlight_labels.notna()
        plot_fov_cells(
            ax, fig, df_fov, color_map,
            highlighted_mask=highlight_mask,
            highlighted_labels=highlight_labels,
            highlighted_color_map=rule_palette,
            constant_size=None,
            show_grid=False
        )

        format_spatial_axis(ax, subplot_title)

    if mode == "stage":
        main_title = f"Spatial Evolution of Rule: {format_rule_for_display(ant_con[0], ant_con[1])} Across Disease Stages"
    else:
        main_title = f"Temporal Kinetics of Rule: {format_rule_for_display(ant_con[0], ant_con[1])} Post-Transplantation"

    plt.suptitle(main_title, fontsize=14, y=0.98)

    if rule_palette:
        handles = [
            Line2D([0], [0], marker='o', color='none', markerfacecolor=rule_palette[label], markersize=7, label=label)
            for label in sorted(rule_palette.keys())
        ]
        fig.legend(
            handles=handles,
            loc='lower center',
            bbox_to_anchor=(0.5, 0.01),
            ncol=min(4, len(handles)),
            frameon=True,
            title="Rule colors"
        )
        fig.tight_layout(rect=[0, 0.12, 1, 0.95])
    else:
        fig.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(output_path, dpi=300)
    plt.close()

def run_visualizations(top_n=TOP_N_RULES, method="CN", stage_col=STAGE_COL, time_col=TIME_COL, min_n_stages=2):
    # Use constants.py for paths
    results_path = os.path.join(constants.RESULTS_DATA_DIR, f"results_{method}.csv")
    cell_table_path = os.path.join(constants.MIBI_GUT_DIR_PATH, "cell_table.csv")
    output_dir = os.path.join(constants.RESULTS_PLOTS_DIR, "evolution_plots")
    os.makedirs(output_dir, exist_ok=True)

    if not os.path.exists(results_path):
        print(f"Error: Results file not found at {results_path}")
        return

    print("Loading Rule Results and Cell Data...")
    results_df = pd.read_csv(results_path)
    if 'Biopsy_ID' not in results_df.columns and 'Patient' in results_df.columns:
        results_df = results_df.rename(columns={'Patient': 'Biopsy_ID'})
    results_df = merge_biopsy_metadata(results_df, constants.MIBI_GUT_DIR_PATH)
    results_df = filter_no_self_rules(results_df)
    results_df['Rule_ID'] = results_df['Antecedents'] + " -> " + results_df['Consequents']
    results_df = results_df.dropna(subset=['Rule_Count_Global'])
    if 'Organ' not in results_df.columns:
        print("Warning: Organ column missing after metadata merge. Running in pooled mode.")
        results_df['Organ'] = "All"

    has_stage_col = stage_col in results_df.columns
    has_time_col = time_col in results_df.columns
    if not has_stage_col and not has_time_col:
        print(f"Error: Missing both requested columns: stage_col='{stage_col}', time_col='{time_col}'")
        return
    if has_stage_col:
        results_df[stage_col] = pd.to_numeric(results_df[stage_col], errors='coerce')
    else:
        print(f"Warning: Stage column '{stage_col}' not found. Skipping stage evolution plots.")
    if not has_time_col:
        print(f"Warning: Time column '{time_col}' not found. Skipping temporal evolution plots.")

    cell_df = normalize_cell_table(pd.read_csv(cell_table_path))
    organ_values = [
        organ for organ in sorted(results_df['Organ'].dropna().unique())
        if str(organ) != "Unknown"
    ]
    if not organ_values:
        organ_values = ["All"]

    for organ in organ_values:
        if organ == "All":
            organ_df = results_df.copy()
            print("\nProcessing pooled data (no organ split available)...")
        else:
            organ_df = results_df[results_df['Organ'] == organ].copy()
            print(f"\nProcessing organ: {organ}")
        if organ_df.empty:
            continue
        safe_organ = str(organ).replace('/', '_').replace(' ', '_')
        safe_stage_col = str(stage_col).replace('/', '_').replace(' ', '_')

        # 1. Stage Evolution
        if has_stage_col:
            print(f"  Processing Stage Evolution (using {stage_col})...")
            stage_df = organ_df.dropna(subset=[stage_col]).copy()
            stage_order = get_sorted_stage_values(stage_df, stage_col)
            baseline_stage = get_baseline_stage(stage_order)

            if not stage_order or baseline_stage is None or len(stage_order) < 2:
                print(f"  Warning: Need at least two numeric stages in '{stage_col}'. Skipping stage evolution plots.")
            else:
                print(f"    Detected stage order: {stage_order} | baseline stage: {baseline_stage}")
                top_stage_rules = get_dynamic_rules_by_stage(
                    stage_df,
                    stage_col,
                    stage_order=stage_order,
                    baseline_stage=baseline_stage,
                    n=top_n,
                    min_n_stages=min_n_stages
                )
                if not top_stage_rules:
                    print(
                        f"    No stage rules passed filters "
                        f"(MIN_DELTA={MIN_DELTA}, min_n_stages={min_n_stages}, "
                        f"MIN_FOVS_PER_STAGE={MIN_FOVS_PER_STAGE})."
                    )

                for rule in top_stage_rules:
                    print(f"    Generating stage plot for {rule}...")
                    rep_fovs = {s: select_representative_fov(stage_df, rule, stage_col, s) for s in stage_order}
                    stage_stats_by_group = _compute_rule_group_stats(stage_df, rule, stage_col, stage_order)
                    safe_name = rule.replace('/', '_').replace(' ', '_').replace('->', 'to').replace('[', '').replace(']', '').replace("'", "")
                    plot_evolution_grid(
                        rule, rep_fovs, stage_order, cell_df,
                        os.path.join(output_dir, f"stage_evolution_{safe_organ}_{safe_stage_col}_{safe_name}.png"),
                        mode="stage",
                        group_stats=stage_stats_by_group
                    )

        # 2. Temporal Evolution
        if has_time_col:
            print(f"  Processing Temporal Evolution (using {time_col})...")
            time_df = organ_df.dropna(subset=[time_col]).copy()
            time_order = [b for b in TIME_ORDER if b in set(time_df[time_col].unique())]
            if not time_order or len(time_order) < 2:
                print(f"  Warning: Need at least two time bins in '{time_col}'. Skipping temporal evolution plots.")
            else:
                print(f"    Detected time order: {time_order}")
                top_time_rules = get_dynamic_rules_by_time(
                    time_df,
                    time_col,
                    time_order=time_order,
                    n=top_n,
                    min_n_stages=min_n_stages,
                )
                if not top_time_rules:
                    print(
                        f"    No temporal rules passed filters "
                        f"(MIN_DELTA={MIN_DELTA}, min_n_stages={min_n_stages}, "
                        f"MIN_FOVS_PER_TIME_BIN={MIN_FOVS_PER_TIME_BIN})."
                    )

                for rule in top_time_rules:
                    print(f"    Generating temporal plot for {rule}...")
                    rep_fovs = {t: select_representative_fov(time_df, rule, time_col, t) for t in time_order}
                    time_stats_by_group = _compute_rule_group_stats(time_df, rule, time_col, time_order)
                    safe_name = rule.replace('/', '_').replace(' ', '_').replace('->', 'to').replace('[', '').replace(']', '').replace("'", "")
                    plot_evolution_grid(
                        rule, rep_fovs, time_order, cell_df,
                        os.path.join(output_dir, f"temporal_evolution_{safe_organ}_{safe_name}.png"),
                        mode="time",
                        group_stats=time_stats_by_group
                    )

def main():
    parser = argparse.ArgumentParser(description="Generate Stage/Temporal evolution representative FOV plots.")
    parser.add_argument("--top_n", type=int, default=TOP_N_RULES)
    parser.add_argument("--method", type=str, default="CN")
    parser.add_argument("--stage_col", type=str, default=STAGE_COL)
    parser.add_argument("--time_col", type=str, default=TIME_COL)
    parser.add_argument("--min_n_stages", type=int, default=2, help="Minimum number of stages a rule must appear in.")
    args = parser.parse_args()
    run_visualizations(
        top_n=args.top_n,
        method=args.method,
        stage_col=args.stage_col,
        time_col=args.time_col,
        min_n_stages=args.min_n_stages,
    )

if __name__ == "__main__":
    main()
