import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import glob
import os
import numpy as np
import json
import sys

import ast

# Add project root to sys.path to import constants
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from constants import RESULTS_DATA_DIR, RESULTS_PLOTS_DIR

sns.set_theme(style="whitegrid")

METHODS = ["BAG", "CN", "KNN_R"]

def load_data():
    """Loads results CSVs and Stats JSONs."""
    dfs = {}
    stats = {}
    
    print(f"Loading data from: {RESULTS_DATA_DIR}")
    
    for m in METHODS:
        # Load CSV
        csv_path = os.path.join(RESULTS_DATA_DIR, f"results_{m}.csv")
        if os.path.exists(csv_path):
            try:
                df = pd.read_csv(csv_path)
                # Create Rule ID if missing
                if "Rule" not in df.columns:
                    df["Rule"] = df["Antecedents"] + " -> " + df["Consequents"]
                
                # Rule Length (Total Items)
                # Safely parse string representation of list/tuple
                def get_len(x):
                    try:
                        return len(ast.literal_eval(str(x)))
                    except:
                        return 1 # Fallback
                        
                df["Rule_Len"] = df["Antecedents"].apply(get_len) + df["Consequents"].apply(get_len)
                
                dfs[m] = df
            except Exception as e:
                print(f"Error loading {m} CSV: {e}")
        else:
            print(f"Warning: {csv_path} not found.")

        # Load Stats
        json_path = os.path.join(RESULTS_DATA_DIR, f"stats_{m}.json")
        if os.path.exists(json_path):
            try:
                with open(json_path) as f:
                    stats[m] = json.load(f)
            except Exception as e:
                print(f"Error loading {m} JSON: {e}")
                
    return dfs, stats

def plot_qc_stats(dfs, stats, output_dir):
    """Plots Process Quality Control metrics."""
    print("Generating QC Plots...")
    
    # 1. Redundancy Removal
    data_red = []
    for m, stat in stats.items():
        if "redundant_removed" in stat:
            data_red.append({"Method": m, "Removed": sum(stat["redundant_removed"])})
            
    if data_red:
        plt.figure(figsize=(8, 5))
        sns.barplot(data=pd.DataFrame(data_red), x="Method", y="Removed", palette="viridis")
        plt.title("Total Redundant Rules Removed per Method")
        plt.ylabel("Count of Removed Rules")
        plt.tight_layout()
        plt.savefig(f"{output_dir}/qc_redundancy_removed.png")
        plt.close()

    # 2. Yield (Rules vs Transactions)
    data_yield = []
    for m, df in dfs.items():
        avg_rules = len(df) / df["FOV"].nunique() if not df.empty else 0
        avg_trans = 0
        if m in stats and "sizes" in stats[m]:
             avg_trans = np.mean(stats[m]["sizes"]) if stats[m]["sizes"] else 0
             
        data_yield.append({"Method": m, "Metric": "Avg Rules/FOV", "Value": avg_rules})
        # Scaling down transactions to fit on same chart conceptually, or just plotting separately
        # Let's plot separately or side-by-side
    
    # Actually, simpler: Rules Found Count
    counts = {m: len(df) for m, df in dfs.items()}
    plt.figure(figsize=(8, 5))
    sns.barplot(x=list(counts.keys()), y=list(counts.values()), palette="magma")
    plt.title("Total Valid Rules Found (After Filtering)")
    plt.savefig(f"{output_dir}/qc_total_rules.png")
    plt.close()

def plot_distributions(dfs, output_dir):
    """Plots metric distributions (Lift, Conviction, FDR)."""
    print("Generating Distribution Plots...")
    
    # Combine data
    combined = []
    for m, df in dfs.items():
        cols = ["Lift", "Confidence", "Conviction", "FDR"]
        if "Pathological stage" in df.columns:
            cols.append("Pathological stage")
            
        temp = df[cols].copy()
        # Handle Inf in Conviction
        temp["Conviction"] = temp["Conviction"].replace([np.inf, -np.inf], 100) # Cap for viz
        temp["Method"] = m
        combined.append(temp)
        
    if not combined: return
    full_df = pd.concat(combined)
    
    # Lift Violin
    plt.figure(figsize=(10, 6))
    sns.violinplot(data=full_df, x="Method", y="Lift", palette="muted")
    plt.title("Lift Distribution by Method")
    plt.savefig(f"{output_dir}/dist_lift.png")
    plt.close()
    
    # Conviction (Log scale often better, but capped here)
    plt.figure(figsize=(10, 6))
    sns.kdeplot(data=full_df, x="Conviction", hue="Method", fill=True, clip=(0, 100))
    plt.title("Conviction Distribution (Capped at 100)")
    plt.savefig(f"{output_dir}/dist_conviction.png")
    plt.close()
    
    # Lift vs Confidence (Colored by Stage if possible)
    hue_col = "Method"
    if "Pathological stage" in full_df.columns:
        hue_col = "Pathological stage"
        
    g = sns.FacetGrid(full_df, col="Method", col_wrap=3, height=4)
    g.map_dataframe(sns.scatterplot, x="Confidence", y="Lift", hue=hue_col, alpha=0.6, s=20, palette="viridis")
    g.add_legend(title=hue_col)
    plt.savefig(f"{output_dir}/scatter_lift_vs_conf.png")
    plt.close()

def plot_volcano(dfs, output_dir):
    """Volcano Plot Grid: Lift vs -log10(FDR), colored by Stage."""
    print("Generating Volcano Grid...")
    
    valid_methods = [m for m in dfs.keys() if "FDR" in dfs[m].columns]
    if not valid_methods: return

    # Setup Grid
    n_plots = len(valid_methods)
    cols = 3
    rows = (n_plots // cols) + (1 if n_plots % cols > 0 else 0)
    
    fig, axes = plt.subplots(rows, cols, figsize=(15, 5 * rows), constrained_layout=True)
    axes = axes.flatten()
    
    for i, m in enumerate(valid_methods):
        ax = axes[i]
        df = dfs[m].copy()
        
        # Log FDR
        df["log_fdr"] = -np.log10(df["FDR"] + 1e-10)
        
        # Color
        hue_col = "Pathological stage" if "Pathological stage" in df.columns else "Group"
        
        sns.scatterplot(
            data=df, 
            x="Lift", 
            y="log_fdr", 
            hue=hue_col, 
            palette="Spectral", 
            alpha=0.7,
            s=30,
            ax=ax
        )
        
        ax.axhline(-np.log10(0.05), color='red', linestyle='--', label="FDR=0.05")
        ax.axvline(1.0, color='gray', linestyle='--', label="Lift=1.0")
        ax.set_title(f"Method: {m}")
        ax.set_ylabel("-log10(FDR)")
        
        # Handle Legend (only distinct if needed, but here we keep per plot or dedupe later)
        # For simplicity in grid, we keep them or move outside. 
        # Seaborn adds legend to ax automatically.
        if i == 0:
            ax.legend(loc='lower right', title="Pathological Stage", fontsize='small')
        else:
            ax.get_legend().remove()

    # Hide empty axes
    for j in range(i + 1, len(axes)):
        axes[j].axis('off')
        
    plt.suptitle("Volcano Plots: Rule Significance vs Strength", fontsize=16)
    plt.savefig(f"{output_dir}/volcano_grid.png")
    plt.close()

def plot_complexity(dfs, output_dir):
    """Heatmap of Rule Lengths."""
    print("Generating Complexity Plots...")
    
    data = []
    for m, df in dfs.items():
        counts = df["Rule_Len"].value_counts(normalize=True).sort_index()
        for length, prop in counts.items():
            data.append({"Method": m, "Length": length, "Proportion": prop})
            
    if not data: return
    
    heatmap_df = pd.DataFrame(data).pivot(index="Method", columns="Length", values="Proportion").fillna(0)
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(heatmap_df, annot=True, fmt=".1%", cmap="Blues")
    plt.title("Proportion of Rules by Item Count (Complexity)")
    plt.savefig(f"{output_dir}/heatmap_complexity.png")
    plt.close()

def plot_overlap(dfs, output_dir):
    """Jaccard Similarity of Unique Rules between Methods."""
    print("Generating Overlap Plots...")
    
    unique_rules = {m: set(df["Rule"].unique()) for m, df in dfs.items()}
    methods = list(unique_rules.keys())
    n = len(methods)
    matrix = np.zeros((n, n))
    
    for i in range(n):
        for j in range(n):
            set_i = unique_rules[methods[i]]
            set_j = unique_rules[methods[j]]
            if not set_i and not set_j:
                val = 0
            else:
                val = len(set_i.intersection(set_j)) / len(set_i.union(set_j))
            matrix[i, j] = val
            
    plt.figure(figsize=(8, 6))
    sns.heatmap(matrix, xticklabels=methods, yticklabels=methods, annot=True, fmt=".2f", cmap="Greens")
    plt.title("Jaccard Similarity of Rule Sets")
    plt.tight_layout()
    plt.savefig(f"{output_dir}/heatmap_overlap.png")
    plt.close()

def main():
    # Setup Output Dir
    out_dir = os.path.join(RESULTS_PLOTS_DIR, "mining_report")
    os.makedirs(out_dir, exist_ok=True)
    print(f"Saving plots to: {out_dir}")
    
    dfs, stats = load_data()
    
    if not dfs:
        print("No data found. Exiting.")
        return

    plot_qc_stats(dfs, stats, out_dir)
    plot_distributions(dfs, out_dir)
    plot_volcano(dfs, out_dir)
    plot_complexity(dfs, out_dir)
    plot_overlap(dfs, out_dir)
    
    print("Visualization Complete.")

if __name__ == "__main__":
    main()