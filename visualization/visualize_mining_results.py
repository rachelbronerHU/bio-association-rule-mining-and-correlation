import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import glob
import os
import numpy as np
import json
import sys

import ast
from sklearn.decomposition import PCA

# Add project root to sys.path to import constants
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from constants import RESULTS_DATA_DIR, RESULTS_PLOTS_DIR, METHODS, MIBI_GUT_DIR_PATH
from visualization.utils.visualization_util import add_organ_column

sns.set_theme(style="whitegrid")

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
        full_df[hue_col] = full_df[hue_col].fillna(-1).astype(int)
        
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
    
    # Fixed Color Map for Stages
    stage_colors = {
        0: "forestgreen", # Control
        1: "gold",
        2: "orange",
        3: "darkorange",
        4: "firebrick"
    }
    
    # Legend Labels
    stage_labels = ["Control", "Stage 1", "Stage 2", "Stage 3", "Stage 4"]
    
    for i, m in enumerate(valid_methods):
        ax = axes[i]
        df = dfs[m].copy()
        
        # Log FDR
        df["log_fdr"] = -np.log10(df["FDR"] + 1e-10)
        
        # Color Logic
        hue_col = "Pathological stage" if "Pathological stage" in df.columns else "Group"
        palette = stage_colors if hue_col == "Pathological stage" else "Spectral"
        
        if hue_col == "Pathological stage":
             df[hue_col] = df[hue_col].fillna(-1).astype(int)

        sns.scatterplot(
            data=df, 
            x="Lift", 
            y="log_fdr", 
            hue=hue_col, 
            palette=palette, 
            alpha=0.6,
            s=25,
            ax=ax,
            legend='brief' # Ensure legend is generated for all to extract handles later
        )
        
        ax.axhline(-np.log10(0.05), color='red', linestyle='--', label="FDR=0.05", linewidth=1)
        ax.axvline(1.0, color='gray', linestyle='--', label="Lift=1.0", linewidth=1)
        
        # Calculate Statistics
        n_total = len(df)
        p_vals = df["FDR"]
        
        stats_subtitle = ""
        if not p_vals.empty:
            pcts = np.percentile(p_vals, [10, 25, 50, 75, 90])
            stats_subtitle = (
                f"N={n_total} | FDR P-Values: "
                f"10%:{pcts[0]:.1e}, 50%:{pcts[2]:.1e}, 90%:{pcts[4]:.1e}"
            )
            
        ax.set_title(f"Method: {m}\n{stats_subtitle}", fontsize=10)
        ax.set_ylabel("-log10(FDR)")
        ax.set_xlabel("Lift")
        
        # Remove individual legends to create a shared one later
        if ax.get_legend():
            handles, labels = ax.get_legend_handles_labels()
            # Store handles from the first plot for the shared legend
            if i == 0:
                shared_handles = handles
                shared_labels = labels
            ax.get_legend().remove()

    # Hide empty axes
    for j in range(i + 1, len(axes)):
        axes[j].axis('off')
        
    # Add Shared Legend to the Figure
    # Filter handles to just show Stages (0,1,2,3) if present
    if 'shared_handles' in locals():
        fig.legend(shared_handles, shared_labels, loc='center right', title="Stage/Group")
        # Adjust layout to make room for legend
        plt.subplots_adjust(right=0.85)

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

def plot_rules_per_fov_per_method(dfs, output_dir):
    """Plots the number of rules per FOV for each method separately."""
    print("Generating Rules per FOV Plots (per method)...")
    
    for m, df in dfs.items():
        if "FOV" not in df.columns:
            print(f"Skipping {m}: 'FOV' column missing.")
            continue
            
        # Count rules per FOV
        counts = df["FOV"].value_counts().reset_index()
        counts.columns = ["FOV", "Rule_Count"]
        
        # Sort for better visualization
        counts = counts.sort_values("Rule_Count", ascending=False)
        
        plt.figure(figsize=(12, 6))
        sns.barplot(data=counts, x="FOV", y="Rule_Count", color="steelblue")
        plt.title(f"Number of Rules per FOV - Method: {m}")
        # Remove x-tick labels as they are too messy
        plt.xticks([])
        plt.xlabel("FOVs (Sorted by Count)")
        plt.tight_layout()
        plt.savefig(f"{output_dir}/rules_per_fov_{m}.png")
        plt.close()

def plot_pca_rules(dfs, output_dir):
    """PCA of FOV x Rules matrix based on Lift."""
    print("Generating PCA Plots...")
    
    for m, df in dfs.items():
        if df.empty or "FOV" not in df.columns:
            continue
            
        # Extract necessary columns (now using standard helper)
        if "Organ" not in df.columns:
            df = add_organ_column(df, MIBI_GUT_DIR_PATH)
        
        hue_stage = "Pathological stage" if "Pathological stage" in df.columns else "Group"
        hue_organ = "Organ"
        
        # Create Pivot Table (FOV x Rule), values=Lift, fillna=0
        pivot_df = df.pivot_table(index="FOV", columns="Rule", values="Lift", fill_value=0)
        
        if len(pivot_df) < 3 or len(pivot_df.columns) < 2:
            print(f"Not enough data for PCA in method {m}.")
            continue
            
        # Extract FOV metadata mapping
        meta_cols = ["FOV", hue_stage, hue_organ]
        meta_cols = [c for c in meta_cols if c in df.columns]
        meta_df = df[meta_cols].drop_duplicates(subset=["FOV"]).set_index("FOV")
        
        # Align meta_df with pivot_df index
        meta_df = meta_df.loc[pivot_df.index]
        
        # Run PCA
        pca = PCA(n_components=2)
        pcs = pca.fit_transform(pivot_df)
        
        # Create Plotting DataFrame
        plot_df = pd.DataFrame(data=pcs, columns=['PC1', 'PC2'], index=pivot_df.index)
        plot_df = plot_df.join(meta_df)
        
        # Ensure categorical variables for hue
        if hue_stage in plot_df.columns:
            plot_df[hue_stage] = plot_df[hue_stage].fillna(-1).astype(int).astype(str)
            plot_df[hue_stage] = plot_df[hue_stage].replace('-1', 'Unknown')
        
        # 1. Plot by Organ/Location
        plt.figure(figsize=(10, 8))
        if hue_organ in plot_df.columns:
            sns.scatterplot(data=plot_df, x="PC1", y="PC2", hue=hue_organ, palette="Set1", s=100, alpha=0.7)
            plt.legend(title="Organ/Location", bbox_to_anchor=(1.05, 1), loc='upper left')
        else:
            sns.scatterplot(data=plot_df, x="PC1", y="PC2", s=100, alpha=0.7)
            
        plt.title(f"PCA of Rules x FOV (Method: {m}) - Colored by Location")
        plt.xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.2%} variance)")
        plt.ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.2%} variance)")
        plt.tight_layout()
        plt.savefig(f"{output_dir}/pca_organ_{m}.png")
        plt.close()
        
        # 2. Plot by Stage (Global)
        plt.figure(figsize=(10, 8))
        if hue_stage in plot_df.columns:
            sns.scatterplot(data=plot_df, x="PC1", y="PC2", hue=hue_stage, palette="viridis", s=100, alpha=0.7)
            plt.legend(title="Stage", bbox_to_anchor=(1.05, 1), loc='upper left')
        else:
            sns.scatterplot(data=plot_df, x="PC1", y="PC2", s=100, alpha=0.7)
            
        plt.title(f"PCA of Rules x FOV (Method: {m}) - Colored by Stage")
        plt.xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.2%} variance)")
        plt.ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.2%} variance)")
        plt.tight_layout()
        plt.savefig(f"{output_dir}/pca_stage_{m}.png")
        plt.close()
        
        # 3. Plot by Stage (Organ-Specific)
        if hue_organ in plot_df.columns and hue_stage in plot_df.columns:
            for organ_val in plot_df[hue_organ].dropna().unique():
                if str(organ_val) == 'Unknown': continue
                
                organ_df = df[df[hue_organ] == organ_val]
                if organ_df.empty: continue
                
                organ_pivot = organ_df.pivot_table(index="FOV", columns="Rule", values="Lift", fill_value=0)
                if len(organ_pivot) < 3 or len(organ_pivot.columns) < 2: continue
                
                organ_meta = organ_df[["FOV", hue_stage]].drop_duplicates(subset=["FOV"]).set_index("FOV")
                organ_meta = organ_meta.loc[organ_pivot.index]
                
                # We need to run a fresh PCA just for this organ's variance!
                pca_organ = PCA(n_components=2)
                pcs_organ = pca_organ.fit_transform(organ_pivot)
                
                plot_organ_df = pd.DataFrame(data=pcs_organ, columns=['PC1', 'PC2'], index=organ_pivot.index)
                plot_organ_df = plot_organ_df.join(organ_meta)
                
                plot_organ_df[hue_stage] = plot_organ_df[hue_stage].fillna(-1).astype(int).astype(str)
                plot_organ_df[hue_stage] = plot_organ_df[hue_stage].replace('-1', 'Unknown')
                
                plt.figure(figsize=(10, 8))
                sns.scatterplot(data=plot_organ_df, x="PC1", y="PC2", hue=hue_stage, palette="viridis", s=100, alpha=0.7)
                plt.legend(title="Stage", bbox_to_anchor=(1.05, 1), loc='upper left')
                plt.title(f"PCA of Rules x FOV ({organ_val} Only) (Method: {m}) - Colored by Stage")
                plt.xlabel(f"PC1 ({pca_organ.explained_variance_ratio_[0]:.2%} variance)")
                plt.ylabel(f"PC2 ({pca_organ.explained_variance_ratio_[1]:.2%} variance)")
                plt.tight_layout()
                plt.savefig(f"{output_dir}/pca_stage_{organ_val}_{m}.png")
                plt.close()

def _build_rule_items_matrix(df):
    """Builds a binary matrix of Rules x Cell Types."""
    rule_items = {}
    for _, row in df.drop_duplicates(subset=["Rule"]).iterrows():
        items = []
        try:
            ant = ast.literal_eval(row["Antecedents"])
            con = ast.literal_eval(row["Consequents"])
            items.extend(ant)
            items.extend(con)
        except:
            pass
        rule_items[row["Rule"]] = items
        
    if not rule_items:
        return None, None
        
    all_items = sorted(list(set(item for items in rule_items.values() for item in items)))
    
    matrix = []
    rules_list = list(rule_items.keys())
    for rule in rules_list:
        row_data = {item: 1 if item in rule_items[rule] else 0 for item in all_items}
        matrix.append(row_data)
        
    rule_item_df = pd.DataFrame(matrix, index=rules_list)
    rule_item_df = rule_item_df.loc[:, (rule_item_df != rule_item_df.iloc[0]).any()] 
    
    return rule_item_df, rules_list

def _calculate_rule_metadata(df, rules_list, stage_col):
    """Calculates Average Stage and Pct Colon for each rule."""
    rule_meta = []
    for rule in rules_list:
        rule_df = df[df["Rule"] == rule]
        
        avg_stage = np.nan
        if stage_col in rule_df.columns:
            stages = pd.to_numeric(rule_df[stage_col], errors='coerce').dropna()
            if not stages.empty:
                avg_stage = stages.mean()
                
        pct_colon = np.nan
        if "Organ" in rule_df.columns:
            organs = rule_df["Organ"].dropna()
            colon_count = (organs == "Colon").sum()
            total = len(organs)
            if total > 0:
                pct_colon = colon_count / total
                
        rule_meta.append({
            "Rule": rule,
            "Avg_Stage": avg_stage,
            "Pct_Colon": pct_colon
        })
        
    return pd.DataFrame(rule_meta).set_index("Rule")

def _plot_rule_family_scatter(plot_df, hue_col, palette, title, pca, output_dir, filename, label_context):
    plt.figure(figsize=(10, 8))
    scatter = sns.scatterplot(data=plot_df, x="PC1", y="PC2", hue=hue_col, palette=palette, s=80, alpha=0.8, edgecolor="w")
    plt.title(title)
    plt.xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.2%} variance)")
    plt.ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.2%} variance)")
    
    if scatter.get_legend():
        scatter.get_legend().remove()
    
    valid_vals = plot_df[hue_col].dropna()
    if not valid_vals.empty:
        if hue_col == "Pct_Colon":
            norm = plt.Normalize(0, 1)
            lbl = "Organ Enrichment (0 = Duodenum, 1 = Colon)"
        else:
            norm = plt.Normalize(valid_vals.min(), valid_vals.max())
            lbl = f"Average {label_context} (0=Control, 4=Severe)"
            
        sm = plt.cm.ScalarMappable(cmap=palette, norm=norm)
        sm.set_array([])
        plt.colorbar(sm, ax=plt.gca(), label=lbl)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, filename))
    plt.close()

def _build_rule_items_matrix(df):
    """Builds a binary matrix of Rules x Cell Types."""
    import ast
    rule_items = {}
    for _, row in df.drop_duplicates(subset=["Rule"]).iterrows():
        items = []
        try:
            ant = ast.literal_eval(row["Antecedents"])
            con = ast.literal_eval(row["Consequents"])
            items.extend(ant)
            items.extend(con)
        except:
            pass
        rule_items[row["Rule"]] = items
        
    if not rule_items:
        return None, None
        
    all_items = sorted(list(set(item for items in rule_items.values() for item in items)))
    
    matrix = []
    rules_list = list(rule_items.keys())
    for rule in rules_list:
        row_data = {item: 1 if item in rule_items[rule] else 0 for item in all_items}
        matrix.append(row_data)
        
    import pandas as pd
    rule_item_df = pd.DataFrame(matrix, index=rules_list)
    rule_item_df = rule_item_df.loc[:, (rule_item_df != rule_item_df.iloc[0]).any()] 
    
    return rule_item_df, rules_list

def _calculate_rule_metadata(df, rules_list, stage_col):
    """Calculates Average Stage and Pct Colon for each rule."""
    import numpy as np
    import pandas as pd
    rule_meta = []
    for rule in rules_list:
        rule_df = df[df["Rule"] == rule]
        
        avg_stage = np.nan
        if stage_col in rule_df.columns:
            stages = pd.to_numeric(rule_df[stage_col], errors='coerce').dropna()
            if not stages.empty:
                avg_stage = stages.mean()
                
        pct_colon = np.nan
        if "Organ" in rule_df.columns:
            organs = rule_df["Organ"].dropna()
            colon_count = (organs == "Colon").sum()
            total = len(organs)
            if total > 0:
                pct_colon = colon_count / total
                
        rule_meta.append({
            "Rule": rule,
            "Avg_Stage": avg_stage,
            "Pct_Colon": pct_colon
        })
        
    return pd.DataFrame(rule_meta).set_index("Rule")

def _plot_rule_family_scatter(plot_df, hue_col, palette, title, pca, output_dir, filename, label_context):
    import matplotlib.pyplot as plt
    import seaborn as sns
    import os
    plt.figure(figsize=(10, 8))
    scatter = sns.scatterplot(data=plot_df, x="PC1", y="PC2", hue=hue_col, palette=palette, s=80, alpha=0.8, edgecolor="w")
    plt.title(title)
    plt.xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.2%} variance)")
    plt.ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.2%} variance)")
    
    if scatter.get_legend():
        scatter.get_legend().remove()
    
    valid_vals = plot_df[hue_col].dropna()
    if not valid_vals.empty:
        if hue_col == "Pct_Colon":
            norm = plt.Normalize(0, 1)
            lbl = "Organ Enrichment (0 = Duodenum, 1 = Colon)"
        else:
            norm = plt.Normalize(valid_vals.min(), valid_vals.max())
            lbl = f"Average {label_context} (0=Control, 4=Severe)"
            
        sm = plt.cm.ScalarMappable(cmap=palette, norm=norm)
        sm.set_array([])
        plt.colorbar(sm, ax=plt.gca(), label=lbl)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, filename))
    plt.close()

def plot_pca_rule_families(dfs, output_dir):
    """PCA of Rules x Cell Types matrix to find rule families, colored by clinical metadata."""
    from sklearn.decomposition import PCA
    import pandas as pd
    from visualization.utils.visualization_util import add_organ_column
    from constants import MIBI_GUT_DIR_PATH
    print("Generating Rule Families PCA Plots (Global and Per Organ)...")
    
    for m, df in dfs.items():
        if df.empty or "Rule" not in df.columns:
            continue
            
        if "Organ" not in df.columns:
            df = add_organ_column(df, MIBI_GUT_DIR_PATH)
            
        stage_col = "Pathological stage" if "Pathological stage" in df.columns else "Group"
        
        # --- GLOBAL PCA ---
        rule_item_df, rules_list = _build_rule_items_matrix(df)
        if rule_item_df is None or len(rule_item_df) < 3 or len(rule_item_df.columns) < 2:
            print(f"Not enough variance in rules for PCA in method {m}.")
            continue
            
        pca = PCA(n_components=2)
        pcs = pca.fit_transform(rule_item_df)
        plot_df = pd.DataFrame(data=pcs, columns=['PC1', 'PC2'], index=rules_list)
        
        meta_df = _calculate_rule_metadata(df, rules_list, stage_col)
        plot_df = plot_df.join(meta_df)
        
        # Global Stage Plot
        _plot_rule_family_scatter(plot_df, "Avg_Stage", "viridis", 
                                  f"Rule Families PCA ({m}) - Colored by Average Stage", 
                                  pca, output_dir, f"pca_rule_families_stage_{m}.png", stage_col)
                                  
        # Global Organ Plot
        _plot_rule_family_scatter(plot_df, "Pct_Colon", "coolwarm", 
                                  f"Rule Families PCA ({m}) - Colored by Organ Enrichment", 
                                  pca, output_dir, f"pca_rule_families_organ_{m}.png", "Organ Enrichment")
                                  
        # --- PER ORGAN PCA ---
        for organ_val in df["Organ"].dropna().unique():
            if str(organ_val) == 'Unknown': continue
            
            organ_df = df[df["Organ"] == organ_val]
            if organ_df.empty: continue
            
            organ_rule_item_df, organ_rules_list = _build_rule_items_matrix(organ_df)
            if organ_rule_item_df is None or len(organ_rule_item_df) < 3 or len(organ_rule_item_df.columns) < 2:
                continue
                
            pca_organ = PCA(n_components=2)
            pcs_organ = pca_organ.fit_transform(organ_rule_item_df)
            plot_organ_df = pd.DataFrame(data=pcs_organ, columns=['PC1', 'PC2'], index=organ_rules_list)
            
            organ_meta_df = _calculate_rule_metadata(organ_df, organ_rules_list, stage_col)
            plot_organ_df = plot_organ_df.join(organ_meta_df)
            
            _plot_rule_family_scatter(plot_organ_df, "Avg_Stage", "viridis", 
                                      f"Rule Families PCA ({organ_val} Only) ({m}) - Avg Stage", 
                                      pca_organ, output_dir, f"pca_rule_families_stage_{organ_val}_{m}.png", stage_col)

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
    plot_rules_per_fov_per_method(dfs, out_dir)
    plot_pca_rules(dfs, out_dir)
    plot_pca_rule_families(dfs, out_dir)
    
    print("Visualization Complete.")

if __name__ == "__main__":
    main()