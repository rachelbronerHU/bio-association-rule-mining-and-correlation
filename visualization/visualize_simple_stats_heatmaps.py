import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os
from scipy.stats import ttest_ind
import numpy as np

from constants import RESULTS_DATA_DIR, RESULTS_SIMPLE_STATS_PLOTS_DIR

# Configuration
METHODS = ["BAG", "CN", "KNN_R"]
BASE_DIR = RESULTS_DATA_DIR
OUTPUT_DIR = RESULTS_SIMPLE_STATS_PLOTS_DIR
TARGET_COL = "Cortico Response"
GROUP_COL = "Biopsy_ID"
TOP_N = 20

def ensure_dir(d):
    if not os.path.exists(d):
        os.makedirs(d)

def load_data(method):
    path = f"{BASE_DIR}/results_{method}.csv"
    if not os.path.exists(path): return None, None
    
    df = pd.read_csv(path)
    if "Rule" not in df.columns:
        df["Rule"] = df["Antecedents"] + " -> " + df["Consequents"]
        
    pivot = df.pivot_table(index="FOV", columns="Rule", values="Lift", fill_value=0)
    meta = df[["FOV", TARGET_COL]].drop_duplicates(subset="FOV").set_index("FOV")
    
    common = pivot.index.intersection(meta.index)
    X = pivot.loc[common]
    y = meta.loc[common, TARGET_COL]
    
    return X[y.notna()], y[y.notna()]

def plot_heatmap(X, y, method):
    # 1. Identify Top Rules
    classes = y.unique()
    if len(classes) != 2: return
    
    group0 = X[y == classes[0]]
    group1 = X[y == classes[1]]
    
    stats = []
    valid_cols = [c for c in X.columns if X[c].var() > 0]
    
    for col in valid_cols:
        t, p = ttest_ind(group0[col], group1[col], equal_var=False)
        if np.isnan(p): p = 1.0
        stats.append({"Rule": col, "P": p, "Mean0": group0[col].mean(), "Mean1": group1[col].mean()})
        
    stats_df = pd.DataFrame(stats).sort_values("P").head(TOP_N)
    
    # 2. Prepare Plot Data
    plot_data = stats_df.set_index("Rule")[["Mean0", "Mean1"]]
    plot_data.columns = [classes[0], classes[1]]
    
    # Sort by difference for visual clarity
    plot_data["diff"] = plot_data[classes[0]] - plot_data[classes[1]]
    plot_data = plot_data.sort_values("diff", ascending=False).drop(columns=["diff"])
    
    # 3. Plot
    plt.figure(figsize=(10, 12))
    sns.heatmap(plot_data, annot=True, cmap="vlag", center=1.0, fmt=".2f", linewidths=.5)
    plt.title(f"Top 20 Simple Stats Rules: {method} -> {TARGET_COL}\n(Values = Mean Lift)", fontsize=14)
    plt.xlabel("Patient Group")
    plt.ylabel("Spatial Rule")
    plt.tight_layout()
    
    save_path = f"{OUTPUT_DIR}/heatmap_SIMPLE_{method}_{TARGET_COL.replace(' ', '_')}.png"
    plt.savefig(save_path)
    print(f"Saved: {save_path}")

def main():
    print("Generating Simple Stats Heatmaps...")
    ensure_dir(OUTPUT_DIR)
    
    for method in METHODS:
        X, y = load_data(method)
        if X is not None:
            plot_heatmap(X, y, method)

if __name__ == "__main__":
    main()
