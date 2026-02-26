import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os
import numpy as np
import sys
import argparse
import ast

# Add project root to sys.path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(PROJECT_ROOT)
from constants import RESULTS_PLOTS_DIR, CONSENSUS_RESULTS_EXPLORATION_DIR

# Input Directory from where we just saved the consensus tables
CONSENSUS_DIR = os.path.join(PROJECT_ROOT, CONSENSUS_RESULTS_EXPLORATION_DIR)
# Output Directory for plots
PLOTS_DIR = os.path.join(PROJECT_ROOT, RESULTS_PLOTS_DIR, "consensus_report")

METHODS = ["BAG", "CN", "KNN_R"]

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
    else:
        print(f"Warning: File not found {filepath}")
        return None

def clean_rule_name(ant, con):
    """
    Cleans rule name for better visualization.
    Removes [' '] and _CENTER/_NEIGHBOR suffixes for display.
    """
    try:
        # Parse if stringified list
        if isinstance(ant, str) and ant.startswith("["):
            ant = ast.literal_eval(ant)
        if isinstance(con, str) and con.startswith("["):
            con = ast.literal_eval(con)
            
        # Clean items
        ant_clean = [str(x).replace("_CENTER", "").replace("_NEIGHBOR", "") for x in ant]
        con_clean = [str(x).replace("_CENTER", "").replace("_NEIGHBOR", "") for x in con]
        
        ant_str = ", ".join(ant_clean)
        con_str = ", ".join(con_clean)
        
        return f"{ant_str} -> {con_str}"
    except:
        return f"{ant} -> {con}"

def plot_stage_consensus_heatmap(method, suffix, output_dir, organ=None):
    """
    Plots a heatmap of Rules vs Pathological Stages.
    Color = Consensus Score.
    X-Axis labels include (N=...) count of FOVs.
    """
    df = load_data(method, "stage", organ=organ, suffix=suffix)
    if df is None or df.empty:
        return

    # Extract Stage Counts map: {Stage: Count}
    # df has columns: 'Pathological stage', 'Total_FOVs_In_Stage'
    # We must handle the column name carefully if it's 'Pathological stage' or just 'Stage'
    stage_col = "Pathological stage"
    
    stage_counts = df[[stage_col, "Total_FOVs_In_Stage"]].drop_duplicates().set_index(stage_col)["Total_FOVs_In_Stage"].to_dict()

    # Create Clean Rule ID
    df["Rule"] = df.apply(lambda row: clean_rule_name(row["Antecedents"], row["Consequents"]), axis=1)

    # Handle duplicates created by name cleaning (take max score)
    df = df.groupby(["Rule", stage_col], as_index=False)["Consensus_Score"].max()

    # Pivot: Index=Rule, Columns=Stage, Values=Score
    pivot_df = df.pivot(index="Rule", columns=stage_col, values="Consensus_Score")
    
    # Fill NaN with 0 (Rule not present in that stage)
    pivot_df = pivot_df.fillna(0)
    
    # NEW LOGIC: Select top N rules per stage, then take union
    # This ensures each stage is represented equally
    rules_per_stage = 10  # Top 10 rules per stage
    selected_rules = set()
    
    for stage in pivot_df.columns:
        # Get top N rules for this specific stage
        stage_top = pivot_df.nlargest(rules_per_stage, stage).index.tolist()
        selected_rules.update(stage_top)
    
    # Filter to selected rules only
    pivot_df = pivot_df.loc[list(selected_rules)]
    
    # Sort rows by max score across stages for visual clarity
    pivot_df["max_score"] = pivot_df.max(axis=1)
    pivot_df = pivot_df.sort_values("max_score", ascending=False).drop(columns="max_score")
    
    print(f"  Selected {len(pivot_df)} unique rules ({rules_per_stage} per stage)")

    # Rename Columns to include N
    new_cols = {col: f"Stage {col}\n(N={stage_counts.get(col, '?')})" for col in pivot_df.columns}
    pivot_df = pivot_df.rename(columns=new_cols)

    plt.figure(figsize=(14, 12))
    sns.heatmap(pivot_df, annot=True, fmt=".2f", cmap="YlOrRd", cbar_kws={'label': 'Consensus Score'})
    
    # Add warnings for weak strata if available
    title = f"Stage Consensus Heatmap ({method}{suffix})"
    if organ:
        title += f" - {organ}"
    title += f"\n(Top {rules_per_stage} per Stage, {len(pivot_df)} unique)"
    
    # Check for stratum warnings in the data
    if "Stratum_Viable" in df.columns:
        non_viable = (~df["Stratum_Viable"]).sum()
        if non_viable > 0:
            title += f"\n⚠ Warning: {non_viable} stage strata have low counts or high imbalance"
    
    plt.title(title, fontsize=14)
    plt.xlabel("") # Hide generic label since columns have it
    plt.ylabel("Rule")
    plt.tight_layout()
    
    organ_suffix = f"_{organ}" if organ else ""
    save_path = os.path.join(output_dir, f"heatmap_stage_consensus_{method}{organ_suffix}{suffix}.png")
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()
    print(f"Saved {save_path}")

def plot_biopsy_jaccard_similarity(method, suffix, output_dir, organ=None):
    """
    Calculates and plots Jaccard Similarity between Biopsies based on their consensus rules.
    """
    df = load_data(method, "biopsy", organ=organ, suffix=suffix)
    if df is None or df.empty:
        return

    # Create Rule ID
    df["Rule"] = df["Antecedents"] + "->" + df["Consequents"]
    
    # Get set of rules per biopsy
    biopsy_rules = df.groupby("Biopsy_ID")["Rule"].apply(set).to_dict()
    
    biopsies = sorted(list(biopsy_rules.keys()))
    n = len(biopsies)
    matrix = np.zeros((n, n))
    
    for i in range(n):
        for j in range(n):
            set_i = biopsy_rules[biopsies[i]]
            set_j = biopsy_rules[biopsies[j]]
            
            if not set_i and not set_j:
                val = 0
            else:
                intersection = len(set_i.intersection(set_j))
                union = len(set_i.union(set_j))
                val = intersection / union if union > 0 else 0
            matrix[i, j] = val
            
    # Plot using ClusterMap to group similar biopsies
    matrix_df = pd.DataFrame(matrix, index=biopsies, columns=biopsies)
    
    # If matrix is all zeros, clustermap might fail
    if matrix.sum() == 0:
        print(f"Skipping clustermap for {method}: No similarity found.")
        return

    # Use a safe clustermap call
    try:
        g = sns.clustermap(matrix_df, cmap="viridis", figsize=(12, 12), vmin=0, vmax=1)
        title = f"Biopsy Similarity ClusterMap ({method}{suffix})"
        if organ:
            title += f" - {organ}"
        g.fig.suptitle(title, fontsize=16, y=1.02)
        
        organ_suffix = f"_{organ}" if organ else ""
        save_path = os.path.join(output_dir, f"clustermap_biopsy_similarity_{method}{organ_suffix}{suffix}.png")
        plt.savefig(save_path)
        plt.close()
        print(f"Saved {save_path}")
    except Exception as e:
        print(f"Error plotting clustermap for {method}: {e}")

def plot_global_consensus_bar(method, suffix, output_dir, organ=None):
    """
    Plots top global consensus rules with counts.
    """
    df = load_data(method, "global", organ=organ, suffix=suffix)
    if df is None or df.empty:
        return

    # Create Clean Rule ID
    df["Rule"] = df.apply(lambda row: clean_rule_name(row["Antecedents"], row["Consequents"]), axis=1)
    
    # Top 20
    top_df = df.head(20).copy()
    
    # Add labels for bars
    # Assuming 'FOV_Count' and 'Total_FOVs_In_Dataset' exist
    if "FOV_Count" in top_df.columns and "Total_FOVs_In_Dataset" in top_df.columns:
        total_fovs = top_df["Total_FOVs_In_Dataset"].iloc[0]
    else:
        total_fovs = "?"

    plt.figure(figsize=(14, 8))
    ax = sns.barplot(data=top_df, y="Rule", x="Consensus_Score", palette="Blues_r")
    
    # Annotate bars
    for i, p in enumerate(ax.patches):
        if i < len(top_df):
            row = top_df.iloc[i]
            count = row["FOV_Count"]
            ax.text(p.get_width() + 0.01, p.get_y() + p.get_height()/2, 
                    f"n={count}", 
                    va='center', fontsize=10, color='black')

    title = f"Top 20 Global Consensus Rules ({method}{suffix})"
    if organ:
        title += f" - {organ}"
    title += f"\nTotal FOVs in Dataset: {total_fovs}"
    plt.title(title, fontsize=14)
    plt.xlabel("Consensus Score (Fraction of FOVs)")
    plt.xlim(0, 1.25) # Add space for labels
    plt.tight_layout()
    
    organ_suffix = f"_{organ}" if organ else ""
    save_path = os.path.join(output_dir, f"bar_global_consensus_{method}{organ_suffix}{suffix}.png")
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()
    print(f"Saved {save_path}")

def main():
    parser = argparse.ArgumentParser(description="Visualize Consensus Tables.")
    parser.add_argument("--no_self", action="store_true", help="Use tables generated with --no_self flag.")
    parser.add_argument("--organs", nargs='+', help="Optional: Specific organs to visualize (e.g., --organs Colon Duodenum). If not provided, auto-discovers from files.")
    args = parser.parse_args()
    
    suffix = "_no_self" if args.no_self else ""
    
    ensure_dir(PLOTS_DIR)
    print(f"Saving plots to: {PLOTS_DIR}")
    print(f"Using suffix: '{suffix}'")
    
    # Auto-discover available organs from consensus files
    import glob
    pattern = os.path.join(CONSENSUS_DIR, f"*_top_consensus_global_*{suffix}.csv")
    files = glob.glob(pattern)
    
    discovered_organs = set()
    for f in files:
        # Extract organ from filename like "KNN_R_top_consensus_global_Colon.csv"
        basename = os.path.basename(f)
        parts = basename.replace(suffix, "").replace(".csv", "").split("_")
        # Find the organ name (after 'global_')
        try:
            idx = parts.index("global")
            if idx + 1 < len(parts):
                organ = parts[idx + 1]
                discovered_organs.add(organ)
        except ValueError:
            continue
    
    organs_to_process = args.organs if args.organs else sorted(discovered_organs)
    
    if not organs_to_process:
        print("No organ-stratified files found. Falling back to legacy (non-stratified) mode.")
        # Legacy mode for backward compatibility
        for method in METHODS:
            print(f"\n--- Visualizing {method} (Legacy) ---")
            plot_stage_consensus_heatmap(method, suffix, PLOTS_DIR)
            plot_biopsy_jaccard_similarity(method, suffix, PLOTS_DIR)
            plot_global_consensus_bar(method, suffix, PLOTS_DIR)
    else:
        print(f"\nDiscovered organs: {discovered_organs}")
        print(f"Processing organs: {organs_to_process}")
        
        for organ in organs_to_process:
            print(f"\n{'='*60}")
            print(f"ORGAN: {organ}")
            print(f"{'='*60}")
            
            for method in METHODS:
                print(f"\n--- Visualizing {method} - {organ} ---")
                plot_stage_consensus_heatmap(method, suffix, PLOTS_DIR, organ=organ)
                plot_biopsy_jaccard_similarity(method, suffix, PLOTS_DIR, organ=organ)
                plot_global_consensus_bar(method, suffix, PLOTS_DIR, organ=organ)

if __name__ == "__main__":
    main()
