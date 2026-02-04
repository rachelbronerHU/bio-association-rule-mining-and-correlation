import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os

# Configuration
METHODS = ["BAG", "CN", "KNN_R"]
BASE_DIR = "experiment_results/ml_refined_robust_benchmarks/data"
OUTPUT_DIR = "experiment_results/ml_refined_robust_benchmarks/plots"
TARGET_COL = "Cortico Response"

def ensure_dir(d):
    if not os.path.exists(d):
        os.makedirs(d)

def plot_heatmap(method):
    scores_file = f"{BASE_DIR}/scores_{method}_{TARGET_COL.replace(' ', '_')}.csv"
    results_file = f"{BASE_DIR}/results_{method}_{TARGET_COL.replace(' ', '_')}.csv"
    
    if not os.path.exists(scores_file) or not os.path.exists(results_file):
        print(f"Missing files for {method}")
        return

    # 1. Get Top Rules (ML Stability)
    scores = pd.read_csv(scores_file)
    top_rules = scores.sort_values("Selection_Frequency", ascending=False).head(20)["Rule"].tolist()
    
    # 2. Get Data
    df = pd.read_csv(results_file)
    if "Rule" not in df.columns:
        df["Rule"] = df["Antecedents"] + " -> " + df["Consequents"]
        
    df_subset = df[df["Rule"].isin(top_rules)]
    
    # 3. Pivot
    pivot = df_subset.pivot_table(index="Rule", columns=TARGET_COL, values="Lift", aggfunc="mean")
    
    # Sort
    if pivot.shape[1] == 2:
        cols = pivot.columns
        pivot["diff"] = pivot[cols[0]] - pivot[cols[1]]
        pivot = pivot.sort_values("diff", ascending=False).drop(columns=["diff"])
        
    # 4. Plot
    plt.figure(figsize=(10, 12))
    sns.heatmap(pivot, annot=True, cmap="vlag", center=1.0, fmt=".2f", linewidths=.5)
    plt.title(f"Top 20 ML Rules: {method} -> {TARGET_COL}\n(Selection Frequency > Stability)", fontsize=14)
    plt.xlabel("Patient Group")
    plt.ylabel("Spatial Rule")
    plt.tight_layout()
    
    save_path = f"{OUTPUT_DIR}/heatmap_ML_{method}_{TARGET_COL.replace(' ', '_')}.png"
    plt.savefig(save_path)
    print(f"Saved: {save_path}")

def main():
    print("Generating ML Heatmaps for all methods...")
    ensure_dir(OUTPUT_DIR)
    for m in METHODS:
        plot_heatmap(m)

if __name__ == "__main__":
    main()
