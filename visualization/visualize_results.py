import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import glob
import os
import numpy as np
import json
import argparse
import umap
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from scipy.spatial.distance import pdist, squareform

from constants import RESULTS_ML_DATA_DIR, RESULTS_ML_PLOTS_DIR

sns.set_theme(style="whitegrid")

# Metadata columns to analyze
STAGE_COLS = [
    "Pathological stage", 
    "GI stage", 
    "liver stage", 
    "skin stage", 
    "Cortico Response", 
    "Grade GVHD",
    "Survival at follow-up",
    "Clinical score",
    "Pathological score"
]

def build_fov_rule_matrix(df, value_col="Lift", fill_value=0.0):
    """
    Builds a matrix where Index=FOV, Columns=Rule, Values=Lift (or other metric).
    """
    if "Rule" not in df.columns:
        df["Rule"] = df["Antecedents"] + " -> " + df["Consequents"]
        
    pivot = df.pivot_table(index="FOV", columns="Rule", values=value_col, aggfunc="max")
    pivot = pivot.fillna(fill_value)
    return pivot

import sys
sys.setrecursionlimit(20000)

def filter_pivot_for_plotting(pivot_df, max_cols=1000):
    """
    If pivot has too many columns (Rules), keep only the top N most variable ones.
    """
    if pivot_df.shape[1] > max_cols:
        print(f"Filtering {pivot_df.shape[1]} rules down to top {max_cols} by variance for plotting...")
        variances = pivot_df.var()
        top_cols = variances.nlargest(max_cols).index
        return pivot_df[top_cols]
    return pivot_df

def plot_pca_analysis(pivot_df, meta_df, output_dir, method_name):
    if pivot_df.empty: return
    
    pivot_df = filter_pivot_for_plotting(pivot_df)
    X = StandardScaler().fit_transform(pivot_df.values)
    pca = PCA(n_components=2)
    components = pca.fit_transform(X)
    
    pca_df = pd.DataFrame(data=components, columns=['PC1', 'PC2'], index=pivot_df.index)
    pca_df = pca_df.merge(meta_df, left_index=True, right_on="FOV", how="left")
    var = pca.explained_variance_ratio_
    
    for col in STAGE_COLS:
        if col not in pca_df.columns: continue
        if pca_df[col].isna().all(): continue

        plt.figure(figsize=(10, 8))
        sns.scatterplot(data=pca_df, x="PC1", y="PC2", hue=col, palette="viridis", s=100, alpha=0.8)
        
        plt.suptitle(f"PCA of Rules ({method_name}) - Colored by {col}", fontsize=14)
        plt.title(f"Each dot is a Patient Biopsy. Closer dots have similar spatial patterns.\nPC1: {var[0]:.1%} | PC2: {var[1]:.1%} Variance Explained", fontsize=10)
        plt.xlabel(f"PC1 (Dominant Pattern of Variation)")
        plt.ylabel(f"PC2 (Secondary Pattern of Variation)")
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', title=col)
        plt.tight_layout()
        
        safe_col = col.replace(" ", "_").replace("/", "-")
        plt.savefig(f"{output_dir}/pca_{method_name}_{safe_col}.png")
        plt.close()

def plot_umap_analysis(pivot_df, meta_df, output_dir, method_name):
    """
    Performs UMAP on the FOV x Rule matrix and plots UMAP1 vs UMAP2.
    """
    if pivot_df.empty: return
    
    pivot_df = filter_pivot_for_plotting(pivot_df)

    # Standardize
    X = StandardScaler().fit_transform(pivot_df.values)
    
    # UMAP
    # n_neighbors=30 is a good default for ~100-1000 samples to preserve global structure
    reducer = umap.UMAP(n_neighbors=30, min_dist=0.1, n_components=2, random_state=42)
    try:
        embedding = reducer.fit_transform(X)
    except Exception as e:
        print(f"UMAP failed for {method_name} (likely too few samples): {e}")
        return
    
    umap_df = pd.DataFrame(data=embedding, columns=['UMAP1', 'UMAP2'], index=pivot_df.index)
    
    # Merge metadata
    umap_df = umap_df.merge(meta_df, left_index=True, right_on="FOV", how="left")
    
    # Plot for each stage column
    for col in STAGE_COLS:
        if col not in umap_df.columns: continue
        if umap_df[col].isna().all(): continue

        plt.figure(figsize=(10, 8))
        sns.scatterplot(data=umap_df, x="UMAP1", y="UMAP2", hue=col, palette="viridis", s=100, alpha=0.8)
        
        plt.suptitle(f"UMAP of Rules ({method_name}) - Colored by {col}", fontsize=14)
        plt.title(f"Non-linear projection of patient spatial patterns.\nCloser dots = More similar rule profiles.", fontsize=10)
        plt.xlabel(f"UMAP 1")
        plt.ylabel(f"UMAP 2")
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', title=col)
        plt.tight_layout()
        
        safe_col = col.replace(" ", "_").replace("/", "-")
        plt.savefig(f"{output_dir}/umap_{method_name}_{safe_col}.png")
        plt.close()

def plot_hierarchical_clustering(pivot_df, meta_df, output_dir, method_name):
    if pivot_df.shape[0] < 2 or pivot_df.shape[1] < 2: return

    row_colors_df = pd.DataFrame(index=pivot_df.index)
    meta_mapped = meta_df.set_index("FOV")
    common_fovs = pivot_df.index.intersection(meta_mapped.index)
    
    if len(common_fovs) == 0: return
    
    pivot_subset = pivot_df.loc[common_fovs]
    pivot_subset = filter_pivot_for_plotting(pivot_subset)
    meta_subset = meta_mapped.loc[common_fovs]

    colors_list = []

    for col in STAGE_COLS:
        if col not in meta_subset.columns: continue
        if meta_subset[col].isna().all(): continue
        
        unique_vals = meta_subset[col].unique()
        if "stage" in col.lower() or "score" in col.lower():
             palette = sns.color_palette("Reds", n_colors=len(unique_vals))
        elif len(unique_vals) > 20:
             palette = sns.cubehelix_palette(len(unique_vals))
        else:
             palette = sns.color_palette("tab10", n_colors=len(unique_vals))
             
        lut = dict(zip(unique_vals, palette))
        row_colors_df[col] = meta_subset[col].map(lut)
        colors_list.append(row_colors_df[col])

    if not colors_list:
        row_colors = None
    else:
        row_colors = pd.concat(colors_list, axis=1)

    try:
        g = sns.clustermap(
            pivot_subset,
            method="average",
            metric="euclidean",
            z_score=None, 
            standard_scale=1, 
            row_colors=row_colors,
            cmap="vlag",
            center=0.5,
            figsize=(14, 12),
            xticklabels=False, 
            yticklabels=True,
            cbar_kws={'label': 'Relative Rule Strength (0-1 Scaled)'}
        )
        
        g.fig.suptitle(f"Hierarchical Clustering of Patients ({method_name})", fontsize=16, y=1.02)
        g.ax_heatmap.set_title("Rows = Patients (FOVs) | Columns = Top Variable Rules\nGroups of patients with similar spatial biology will cluster together.", fontsize=10)
        g.ax_heatmap.set_xlabel("Spatial Rules (Top 1000 Variable)")
        g.ax_heatmap.set_ylabel("Patient Biopsy ID")
        
        plt.savefig(f"{output_dir}/clustering_{method_name}.png", bbox_inches='tight')
        plt.close()
    except Exception as e:
        print(f"Clustering failed for {method_name}: {e}")

def plot_jaccard_heatmap(pivot_df, output_dir, method_name):
    pivot_df = filter_pivot_for_plotting(pivot_df)
    binary_matrix = (pivot_df > 0).astype(int)
    
    if binary_matrix.shape[0] < 2: return

    dists = pdist(binary_matrix.values, metric='jaccard')
    dist_matrix = squareform(dists)
    dist_df = pd.DataFrame(dist_matrix, index=binary_matrix.index, columns=binary_matrix.index)

    plt.figure(figsize=(10, 8))
    sns.heatmap(dist_df, cmap="viridis_r", square=True, cbar_kws={'label': 'Jaccard Distance\n(0=Identical Rules, 1=No Overlap)'})
    plt.suptitle(f"Patient Similarity Matrix ({method_name})", fontsize=14)
    plt.title("How similar are the rules found in each patient?\nBlue/Dark = Very Similar. Yellow/Bright = Very Different.", fontsize=10)
    plt.xlabel("Patient Biopsy ID")
    plt.ylabel("Patient Biopsy ID")
    plt.tight_layout()
    plt.savefig(f"{output_dir}/jaccard_heatmap_{method_name}.png")
    plt.close()

def plot_multi_stage_heatmaps(df, output_dir, method_name):
    if "Rule" not in df.columns:
        df["Rule"] = df["Antecedents"] + " -> " + df["Consequents"]

    for col in STAGE_COLS:
        if col not in df.columns: continue
        if df[col].isna().all(): continue
        
        pivot = df.pivot_table(index="Rule", columns=col, values="Lift", aggfunc="mean")
        if pivot.empty: continue

        if len(pivot) > 25:
            pivot["var"] = pivot.var(axis=1)
            top_rules = pivot.sort_values("var", ascending=False).head(25).index
            pivot = pivot.loc[top_rules].drop(columns=["var"])
        elif "var" in pivot.columns:
            pivot = pivot.drop(columns=["var"])

        plt.figure(figsize=(12, 10))
        sns.heatmap(pivot, cmap="vlag", center=1.0, annot=True, fmt=".2f", cbar_kws={'label': 'Mean Lift (Association Strength)'})
        plt.suptitle(f"Top Distinctive Rules by {col} ({method_name})", fontsize=16)
        plt.title("Which rules are strongest in each clinical group?\nRed = Strong Positive Association (>1.0). Blue = Negative/Weak.", fontsize=10)
        plt.xlabel(f"Clinical Group ({col})")
        plt.ylabel("Spatial Rule (Antecedent -> Consequent)")
        plt.tight_layout()
        
        safe_col = col.replace(" ", "_").replace("/", "-")
        plt.savefig(f"{output_dir}/heatmap_rules_{method_name}_{safe_col}.png")
        plt.close()

def plot_model_agreement(results_df, output_dir, method_name, original_filename, data_dir):
    """
    Plots a heatmap of the ML scores (RF vs XGB vs Lasso) for the rules in the results file.
    Requires finding the matching 'scores_' file.
    """
    # Construct scores filename from results filename
    # e.g., results_BAG_Target.csv -> scores_BAG_Target.csv
    scores_fname = original_filename.replace("results_", "scores_")
    scores_path = os.path.join(data_dir, scores_fname)
    
    if not os.path.exists(scores_path):
        print(f"  Skipping model agreement: Scores file not found ({scores_fname})")
        return

    try:
        scores_df = pd.read_csv(scores_path)
    except Exception as e:
        print(f"  Error loading scores file: {e}")
        return

    # Filter scores to only include rules present in the results_df (The Union Set)
    if "Rule" not in results_df.columns:
        results_df["Rule"] = results_df["Antecedents"] + " -> " + results_df["Consequents"]
        
    kept_rules = results_df["Rule"].unique()
    filtered_scores = scores_df[scores_df["Rule"].isin(kept_rules)].copy()
    
    if filtered_scores.empty: return

    # Prepare for Heatmap
    # We want columns: Score_RF, Score_XGB, Score_Lasso, Score_Mean
    score_cols = ["Score_RF", "Score_XGB", "Score_Lasso", "Score_Mean"]
    # Check if cols exist
    existing_cols = [c for c in score_cols if c in filtered_scores.columns]
    
    if not existing_cols: return
    
    # Sort by Mean Score descending
    if "Score_Mean" in existing_cols:
        filtered_scores = filtered_scores.sort_values("Score_Mean", ascending=False)
        
    heatmap_data = filtered_scores.set_index("Rule")[existing_cols]
    
    # Plot
    plt.figure(figsize=(10, len(heatmap_data) * 0.25 + 2)) # Dynamic height
    sns.heatmap(heatmap_data, cmap="viridis", annot=True, fmt=".2f", linewidths=.5)
    plt.title(f"ML Model Agreement ({method_name})", fontsize=14)
    plt.ylabel("Rule")
    plt.xlabel("Model Importance Score")
    plt.tight_layout()
    plt.savefig(f"{output_dir}/agreement_{method_name}.png")
    plt.close()

def plot_rule_counts(dfs, methods, plot_dir):
    plt.figure(figsize=(10, 6))
    counts = []
    labels = []
    annotations = []

    for m in methods:
        if m not in dfs: continue
        df = dfs[m]
        if "Rule" not in df.columns:
            df["Rule"] = df["Antecedents"] + " -> " + df["Consequents"]
            
        unique_rules = df.groupby("Rule").agg({"Lift": "mean", "P_Value": "mean"}).reset_index()
        count = len(unique_rules)
        counts.append(count)
        labels.append(m)
        
        if count > 0:
            top_100 = unique_rules.sort_values("Lift", ascending=False).head(100)
            lift_thresh = top_100["Lift"].min()
            p_thresh = top_100["P_Value"].max()
            txt = f"Top 100:\nLift >= {lift_thresh:.2f}\nP <= {p_thresh:.1e}"
            annotations.append(txt)
        else:
            annotations.append("")

    bars = plt.bar(labels, counts, color=sns.color_palette("pastel"))
    
    for bar, txt in zip(bars, annotations):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2, height + (max(counts)*0.02), txt, ha='center', va='bottom', fontsize=9, color='black')

    plt.title("Unique Rule Definitions Found per Method")
    plt.ylim(0, max(counts) * 1.25 if counts else 1)
    plt.savefig(f"{plot_dir}/rule_counts.png")
    plt.close()

def plot_lift_violin(dfs, methods, plot_dir):
    combined_data = []
    for name in methods:
        if name in dfs:
            temp = dfs[name][["Lift"]].copy()
            temp["Method"] = name
            combined_data.append(temp)
            
    if combined_data:
        full_df = pd.concat(combined_data)
        plt.figure(figsize=(10, 6))
        sns.violinplot(data=full_df, x="Method", y="Lift", palette="muted")
        plt.suptitle("Lift Distribution by Method", fontsize=14)
        plt.title("Lift measures rule strength (Lift > 1 = Positive Association).", fontsize=10)
        plt.xlabel("Spatial Neighborhood Method")
        plt.ylabel("Lift Value")
        plt.savefig(f"{plot_dir}/lift_violin.png")
        plt.close()

def plot_volcano_grid(dfs, methods, plot_dir):
    cols = 3
    valid_methods = [m for m in methods if m in dfs]
    if not valid_methods: return
    rows = int(np.ceil(len(valid_methods) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(18, 5*rows))
    axes = axes.flatten() if hasattr(axes, 'flatten') else [axes]
    
    plt.suptitle("Volcano Plots: Rule Significance vs Strength", fontsize=16, y=0.98)
    
    for i, name in enumerate(valid_methods):
        ax = axes[i]
        df = dfs[name]
        df["log_p"] = -np.log10(df["P_Value"] + 1e-5)
        hue_col = "Pathological stage" if "Pathological stage" in df.columns else "Group"
        if hue_col not in df.columns: hue_col = None
        
        sns.scatterplot(data=df, x="Lift", y="log_p", hue=hue_col, alpha=0.6, ax=ax, palette="viridis", legend=False)
        ax.set_title(f"Method: {name}", fontsize=12)
        ax.set_xlabel("Lift (Strength)")
        ax.set_ylabel("-log10(P-Value) (Significance)")
        ax.axvline(1.0, color='gray', linestyle='--')
        
    plt.tight_layout()
    plt.savefig(f"{plot_dir}/volcano_grid.png")
    plt.close()

def plot_stats(stats, methods, plot_dir):
    # Sizes
    plt.figure(figsize=(10, 6))
    valid_data = False
    for m in methods:
        if m in stats and "sizes" in stats[m] and len(stats[m]["sizes"]) > 1:
            sns.kdeplot(stats[m]["sizes"], label=m, fill=True, alpha=0.3)
            valid_data = True
    if valid_data:
        plt.title("Transaction Batch Size Distribution")
        plt.legend()
        plt.savefig(f"{plot_dir}/batch_sizes.png")
    plt.close()
    
    # Filter Impact
    ratios = {}
    for m in methods:
        if m in stats:
            orig = sum(stats[m].get("orig_counts", [0]))
            kept = sum(stats[m].get("kept_counts", [0]))
            ratios[m] = kept / orig if orig > 0 else 0
            
    if ratios:
        plt.figure(figsize=(8, 5))
        plt.bar(ratios.keys(), ratios.values(), color=sns.color_palette("pastel"))
        plt.title("Retention Rate after Dominance Filter")
        plt.savefig(f"{plot_dir}/filter_impact.png")
        plt.close()

def main():
    parser = argparse.ArgumentParser(description="Generate visualizations for rule mining results.")
    parser.add_argument("--mode", choices=["full", "debug", "robust"], default="full", help="Data source mode (full, debug, or robust).")
    parser.add_argument("--plots", nargs="+", default=["all"], 
                            choices=["all", "counts", "lift", "volcano", "pca", "umap", "clustering", "jaccard", "heatmaps", "stats", "model_agreement"],
                            help="List of plots to generate. Default is 'all'.")   
    
    args = parser.parse_args()

    print(f"Generating visualizations for mode: {args.mode}")
    print(f"Selected plots: {args.plots}")
    
    dfs = {}
    stats = {}
    methods = []

    if args.mode == "robust":
        # Robust Mode: Auto-discover files in ml_refined_robust
        data_dir = RESULTS_ML_DATA_DIR
        plot_dir = RESULTS_ML_PLOTS_DIR
        
        if not os.path.exists(data_dir):
            print(f"Error: Directory {data_dir} does not exist. Run advanced_discovery.py first.")
            return

        csv_files = glob.glob(os.path.join(data_dir, "results_*.csv"))
        for fpath in csv_files:
            fname = os.path.basename(fpath)
            # Use filename as method name (e.g. "BAG_Pathological_stage")
            name = fname.replace(".csv", "").replace("results_", "")
            methods.append(name)
            
            try:
                df = pd.read_csv(fpath)
                if "Rule" not in df.columns:
                    df["Rule"] = df["Antecedents"] + " -> " + df["Consequents"]
                dfs[name] = df
            except Exception as e:
                print(f"Error loading {fname}: {e}")
                
    else:
        # Standard Mode (full/debug)
        data_dir = RESULTS_ML_DATA_DIR
        plot_dir = RESULTS_ML_PLOTS_DIR
        
        methods = ["BAG", "CN", "KNN_R", "WINDOW", "GRID"]
        
        # Load Results
        for m in methods:
            f = f"{data_dir}/results_{m}.csv"
            if os.path.exists(f):
                try:
                    dfs[m] = pd.read_csv(f)
                    if "Rule" not in dfs[m].columns:
                        dfs[m]["Rule"] = dfs[m]["Antecedents"] + " -> " + dfs[m]["Consequents"]
                except Exception as e:
                    print(f"Error loading {m}: {e}")

        # Load Stats
        for m in methods:
            f = f"{data_dir}/stats_{m}.json"
            if os.path.exists(f):
                try:
                    with open(f) as json_file:
                        stats[m] = json.load(json_file)
                except Exception as e:
                    print(f"Error loading stats for {m}: {e}")

    os.makedirs(plot_dir, exist_ok=True)

    if not dfs:
        print("No results to visualize.")
        return

    # Helper to check if a plot should be run
    def should_run(name):
        return "all" in args.plots or name in args.plots

    # --- Aggregate Plots ---
    if should_run("counts"):
        print("Plotting Rule Counts...")
        plot_rule_counts(dfs, methods, plot_dir)
        
    if should_run("lift"):
        print("Plotting Lift Violins...")
        plot_lift_violin(dfs, methods, plot_dir)
        
    if should_run("volcano"):
        print("Plotting Volcano Grids...")
        plot_volcano_grid(dfs, methods, plot_dir)
        
    if should_run("stats") and stats:
        print("Plotting Stats...")
        plot_stats(stats, methods, plot_dir)

    # --- Per-Method Plots ---
    advanced_plots = ["pca", "umap", "clustering", "jaccard", "heatmaps", "model_agreement"]
    if any(should_run(p) for p in advanced_plots):
        for m, df in dfs.items():
            print(f"Processing advanced plots for {m}...")
            
            # Extract Metadata
            meta_cols_present = [c for c in STAGE_COLS if c in df.columns]
            if not meta_cols_present:
                print(f"  Skipping metadata-dependent plots for {m} (missing columns)")
                continue
                
            meta_df = df[["FOV"] + meta_cols_present].drop_duplicates(subset=["FOV"])
            pivot = build_fov_rule_matrix(df, value_col="Lift", fill_value=0.0)
            
            if should_run("pca"):
                plot_pca_analysis(pivot, meta_df, plot_dir, m)
                
            if should_run("umap"):
                plot_umap_analysis(pivot, meta_df, plot_dir, m)
            
            if should_run("clustering"):
                plot_hierarchical_clustering(pivot, meta_df, plot_dir, m)
                
            if should_run("jaccard"):
                plot_jaccard_heatmap(pivot, plot_dir, m)
                
            if should_run("heatmaps"):
                plot_multi_stage_heatmaps(df, plot_dir, m)
                
            if should_run("model_agreement"):
                fname = f"results_{m}.csv"
                plot_model_agreement(df, plot_dir, m, fname, data_dir)

    print(f"Visualizations saved to {plot_dir}/")

if __name__ == "__main__":
    main()