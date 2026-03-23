import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import numpy as np
import sys
import argparse
import ast

# Add project root to sys.path to import constants
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from constants import RESULTS_DATA_DIR, RESULTS_PLOTS_DIR, METHODS, MIBI_GUT_DIR_PATH

sns.set_theme(style="whitegrid")

def load_data(no_self=False):
    """Loads results CSVs."""
    dfs = {}
    print(f"Loading data from: {RESULTS_DATA_DIR}")
    for m in METHODS:
        csv_path = os.path.join(RESULTS_DATA_DIR, f"results_{m}.csv")
        if os.path.exists(csv_path):
            try:
                df = pd.read_csv(csv_path)
                if "Rule" not in df.columns:
                    df["Rule"] = df["Antecedents"] + " -> " + df["Consequents"]
                if no_self:
                    from visualization.utils.visualization_util import filter_no_self_rules
                    df = filter_no_self_rules(df)
                dfs[m] = df
            except Exception as e:
                print(f"Error loading {m} CSV: {e}")
    return dfs

def plot_stage_marker_rules(dfs, output_dir, rules_per_stage=10, no_self=False):
    """
    Plots Stage-Specific Markers (Positive & Negative).
    Color = Marker Score (Red = Enriched in stage, Blue = Depleted in stage).
    Text = k/N (FOVs with rule / Total FOVs).
    """
    for m, df in dfs.items():
        if df.empty: continue
        
        stage_col = "Pathological stage" if "Pathological stage" in df.columns else "Group"
        df[stage_col] = df[stage_col].fillna(-1).astype(int).astype(str).replace('-1', 'Unknown')
        
        def _stage_sort_key(s):
            try: return (0, int(s))
            except: return (1, str(s))
            
        valid_stages = sorted([s for s in df[stage_col].unique() if s != 'Unknown'], key=_stage_sort_key)
        if len(valid_stages) < 2: continue

        # 1) Aggregate Statistics
        stage_fov_counts = df.drop_duplicates("FOV")[stage_col].value_counts().to_dict()
        
        # Pre-calculate matrix of mean lift and prevalence
        pivot_k = df.pivot_table(index="Rule", columns=stage_col, values="FOV", aggfunc="nunique", fill_value=0)
        pivot_mu = df.pivot_table(index="Rule", columns=stage_col, values="Lift", aggfunc="mean", fill_value=0)
        pivot_std = df.pivot_table(index="Rule", columns=stage_col, values="Lift", aggfunc="std", fill_value=0).fillna(0)
        
        # Ensure indices match exactly
        all_method_rules = pivot_k.index
        pivot_mu = pivot_mu.reindex(index=all_method_rules, columns=valid_stages, fill_value=0)
        pivot_k = pivot_k.reindex(index=all_method_rules, columns=valid_stages, fill_value=0)
        pivot_std = pivot_std.reindex(index=all_method_rules, columns=valid_stages, fill_value=0)
        
        # 2) Calculate Interaction Strength Score
        # Color (Red/Blue) determined by Lift > 1 or < 1
        # Saturation determined by abs(log2(Lift)) * sqrt(Prevalence)
        score_df = pd.DataFrame(index=all_method_rules, columns=valid_stages)
        marker_rules = []

        for stage in valid_stages:
            n_stage = stage_fov_counts.get(stage, 1)
            # Prevalence (k/N)
            p_target = pivot_k[stage] / n_stage
            # Mean Lift in stage
            mu_target = pivot_mu[stage]
            
            # Comparison for rule selection (still use relative markers to pick interesting rules)
            others = [s for s in valid_stages if s != stage]
            n_others = sum(stage_fov_counts.get(s, 0) for s in others)
            p_other = pivot_k[others].sum(axis=1) / max(1, n_others)
            mu_other = pivot_mu[others].mean(axis=1)
            
            # Score for SELECTION (not for color)
            eps = 1e-6
            prev_diff = p_target - p_other
            lift_ratio = np.log2((mu_target + eps) / (mu_other + eps)).clip(-3, 3)
            reliability = np.sqrt(pivot_k[stage] + pivot_k[others].sum(axis=1))
            selection_score = prev_diff * np.abs(lift_ratio) * reliability

            # Pick Top N/2 Positive and Top N/2 Negative per stage
            n_each = max(1, rules_per_stage // 2)
            top_pos = selection_score.nlargest(n_each).index.tolist()
            top_neg = selection_score.nsmallest(n_each).index.tolist()
            
            for r in top_pos + top_neg:
                if r not in marker_rules: marker_rules.append(r)

        # Calculate actual Plotting Score (Interaction Strength)
        # Color (Hue): Red for Lift > 1, Blue for Lift < 1
        # Saturation (Intensity): Only penalize if k < 3 FOVs
        for stage in valid_stages:
            n_stage = stage_fov_counts.get(stage, 1)
            p_target = pivot_k[stage] / n_stage
            mu_target = pivot_mu[stage]
            k_target = pivot_k[stage]
            
            # Simple piecewise logic:
            # < 3 FOVs: Lower saturation (k/3) to hide noise.
            # >= 3 FOVs: Normal saturation (weight = 1.0).
            weight = np.where(k_target < 3, k_target / 3.0, 1.0)
            
            # Core score is log2(Lift)
            score_df[stage] = np.log2(mu_target + 1e-6) * weight

        if not marker_rules: continue

        # 3) Prep Plotting Data
        plot_score = score_df.loc[marker_rules, valid_stages].astype(float)
        
        annot_data = pd.DataFrame(index=marker_rules, columns=valid_stages)
        for r in marker_rules:
            for s in valid_stages:
                annot_data.loc[r, s] = f"{int(pivot_k.loc[r, s])}/{stage_fov_counts.get(s, 0)}"

        # Sort rules for diagonal visualization: group by peak stage magnitude
        peak_stage = plot_score.abs().idxmax(axis=1)
        row_meta = pd.DataFrame({
            "rule": marker_rules, 
            "peak_stage": peak_stage, 
            "peak_val": [plot_score.loc[r, peak_stage.loc[r]] for r in marker_rules]
        })
        row_meta["stage_key"] = row_meta["peak_stage"].map(_stage_sort_key)
        # Sort by stage, then score (positives first, then negatives)
        row_meta = row_meta.sort_values(["stage_key", "peak_val"], ascending=[True, False])
        ordered_rules = row_meta["rule"].tolist()

        # 4) Plotting
        fig_height = max(10, len(ordered_rules) * 0.4)
        fig = plt.figure(figsize=(24, fig_height))
        # 4 columns: Heatmap, Violins, Stage-wise Summary, Colorbar
        gs = fig.add_gridspec(1, 4, width_ratios=[len(valid_stages)*1.5, 3, len(valid_stages)*1.1, 0.2], wspace=0.1)
        ax_hm = fig.add_subplot(gs[0, 0])
        ax_violin = fig.add_subplot(gs[0, 1])
        ax_sum = fig.add_subplot(gs[0, 2])
        ax_cb = fig.add_subplot(gs[0, 3])

        # Heatmap Normalization
        # Use the actual maximum in the plotting data as the scale ceiling
        # This ensures the gradient covers the full range from min lift to max lift.
        vmax = np.nanmax(plot_score.abs().values)
        if vmax < 0.5: vmax = 0.5
        
        # Mask out cells where FOV count is 0
        zero_mask = (pivot_k.loc[ordered_rules, valid_stages] == 0).values
        
        sns.heatmap(plot_score.loc[ordered_rules], annot=annot_data.loc[ordered_rules], fmt="", 
                    cmap="RdBu_r", center=0, vmin=-vmax, vmax=vmax, ax=ax_hm, cbar_ax=ax_cb,
                    mask=zero_mask, linewidths=0.5, linecolor="lightgray", annot_kws={"size": 8})
        
        # Color the background grey so masked cells show as grey
        ax_hm.set_facecolor("whitesmoke")
        ax_hm.grid(False)
        
        ax_hm.set_title(f"Interaction Strength ({m})\nRed: Lift > 1, Blue: Lift < 1 | Saturation: log2(Lift) | Muted if k < 3", fontsize=14, pad=35)
        ax_hm.set_xticklabels([f"Stage {s}\n(N={stage_fov_counts.get(s, 0)})" for s in valid_stages], rotation=0)
        ax_hm.set_ylabel("Marker Rule")

        # Violin panel (global lift distribution)
        violin_data = []
        for rule in ordered_rules:
            lifts = df[df["Rule"] == rule]["Lift"].dropna().values
            violin_data.append(lifts if len(lifts) > 0 else np.array([0.0]))

        violin_positions = np.arange(len(ordered_rules)) + 0.5
        parts = ax_violin.violinplot(violin_data, positions=violin_positions, vert=False, 
                                     widths=0.8, showextrema=True, showmedians=True)
        for pc in parts['bodies']:
            pc.set_facecolor('#a6cee3'); pc.set_edgecolor('black'); pc.set_linewidth(0.5); pc.set_alpha(0.8)
        for partname in ('cbars', 'cmins', 'cmaxes', 'cmedians'):
            if partname in parts: parts[partname].set_edgecolor('black'); parts[partname].set_linewidth(1)

        for rule_idx, lifts in enumerate(violin_data):
            if len(lifts) > 0:
                y_jitter = (rule_idx + 0.5) + np.random.uniform(-0.1, 0.1, size=len(lifts))
                ax_violin.scatter(lifts, y_jitter, color="black", alpha=0.35, s=2, zorder=3)

        ax_violin.set_ylim(len(ordered_rules), 0)
        ax_violin.set_yticks([]); ax_violin.set_yticklabels([])
        ax_violin.spines["left"].set_visible(False); ax_violin.spines["right"].set_visible(False); ax_violin.spines["top"].set_visible(False)
        ax_violin.set_title("Global Lift Distribution\n ", fontsize=14, pad=35)
        ax_violin.set_xlabel("Lift")
        ax_violin.grid(False)

        # Summary Panel (Stage Statistics)
        ax_sum.axis("off")
        ax_sum.set_ylim(len(ordered_rules), 0)
        ax_sum.set_xlim(0, len(valid_stages))
        ax_sum.grid(False)
        
        # Headers for Summary Panel columns
        for j, s in enumerate(valid_stages):
            ax_sum.text(j + 0.5, -0.1, f"Stage {s}\nMean \u00b1 STD", 
                        fontweight="bold", fontsize=10, ha="center", va="bottom")

        for i, r in enumerate(ordered_rules):
            for j, s in enumerate(valid_stages):
                if pivot_k.loc[r, s] > 0:
                    txt_stats = f"{pivot_mu.loc[r, s]:.2f}\n\u00b1{pivot_std.loc[r, s]:.2f}"
                    ax_sum.text(j + 0.5, i + 0.5, txt_stats, fontsize=8, va="center", ha="center")

        ax_sum.set_title("Stage Statistics\nMean Lift \u00b1 STD", fontsize=14, pad=35)

        # Zebra striping and unified horizontal lines
        for i in range(len(ordered_rules)):
            if i % 2 == 1:
                # Add light background for every second row to improve horizontal tracking
                bg_rect = plt.Rectangle((0, i), len(valid_stages), 1, facecolor="whitesmoke", alpha=0.3, zorder=-1)
                ax_hm.add_patch(plt.Rectangle((0, i), len(valid_stages), 1, facecolor="whitesmoke", alpha=0.3, zorder=-1))
                ax_violin.axhspan(i, i+1, facecolor="whitesmoke", alpha=0.3, zorder=-1)
                ax_sum.axhspan(i, i+1, facecolor="whitesmoke", alpha=0.3, zorder=-1)

        for y in range(len(ordered_rules) + 1):
            ax_hm.axhline(y, color="lightgray", linewidth=0.5, alpha=0.5)
            ax_violin.axhline(y, color="lightgray", linewidth=0.5, alpha=0.5)
            ax_sum.axhline(y, color="lightgray", linewidth=0.5, alpha=0.5)

        no_self_suffix = "_no_self" if no_self else ""
        save_path = os.path.join(output_dir, f"heatmap_stage_marker_rules_{m}{no_self_suffix}.png")
        plt.savefig(save_path, bbox_inches='tight')
        plt.close(fig)
        print(f"Saved {save_path}")

def main():
    parser = argparse.ArgumentParser(description="Generate Stage Marker Rules Heatmaps")
    parser.add_argument("--top_n", type=int, default=10, help="Top N rules per stage to display")
    parser.add_argument("--no_self", action="store_true", help="Filter out self-loops")
    args = parser.parse_args()
    
    out_dir = os.path.join(RESULTS_PLOTS_DIR, "mining_report")
    os.makedirs(out_dir, exist_ok=True)
    
    dfs = load_data(no_self=args.no_self)
    if not dfs:
        print("No data found. Exiting.")
        return

    plot_stage_marker_rules(dfs, out_dir, rules_per_stage=args.top_n, no_self=args.no_self)
    print("Visualization Complete.")

if __name__ == "__main__":
    main()
