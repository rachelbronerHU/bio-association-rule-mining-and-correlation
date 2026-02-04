import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os

from constants import RESULTS_DIR, RESULTS_ML_DATA_DIR, RESULTS_SIMPLE_STATS_DATA_DIR

# Configuration
ML_LEADERBOARD = RESULTS_ML_DATA_DIR + "final_leaderboard.csv"
SIMPLE_LEADERBOARD = RESULTS_SIMPLE_STATS_DATA_DIR + "leaderboard_simple.csv"
OUTPUT_DIR = RESULTS_DIR + "benchmark_comparison_plots"

# We might need to reload the raw ML scores if leaderboard doesn't have detailed RF/XGB/Lasso per row for ALL targets.
# The leaderboard currently has summaries.
# Let's rely on the leaderboard structure.

def ensure_dir(d):
    if not os.path.exists(d):
        os.makedirs(d)

def main():
    print("Generating Comprehensive Model Comparison Plots...")
    ensure_dir(OUTPUT_DIR)
    
    # 1. Load Data
    if not os.path.exists(ML_LEADERBOARD):
        print("ML Leaderboard missing.")
        return
    if not os.path.exists(SIMPLE_LEADERBOARD):
        print("Simple Stats Leaderboard missing.")
        return
        
    df_ml = pd.read_csv(ML_LEADERBOARD)
    df_simple = pd.read_csv(SIMPLE_LEADERBOARD)
    
    # 2. Reshape ML Data to Long Format
    # Current: Method, Target, Grand_Score, RF_Mean, XGB_Mean, Lasso_Mean
    # Target: Method, Target, Model_Type, Score
    
    ml_long = []
    for _, row in df_ml.iterrows():
        ml_long.append({
            "Method": row["Method"],
            "Target": row["Target"],
            "Model_Type": "ML - Random Forest",
            "Score": row["RF_Mean"]
        })
        ml_long.append({
            "Method": row["Method"],
            "Target": row["Target"],
            "Model_Type": "ML - XGBoost",
            "Score": row["XGB_Mean"]
        })
        ml_long.append({
            "Method": row["Method"],
            "Target": row["Target"],
            "Model_Type": "ML - Lasso",
            "Score": row["Lasso_Mean"]
        })
        
    df_ml_long = pd.DataFrame(ml_long)
    
    # 3. Reshape Simple Stats Data
    # Current: Method, Target, Type, Accuracy
    simple_long = []
    for _, row in df_simple.iterrows():
        simple_long.append({
            "Method": row["Method"],
            "Target": row["Target"],
            "Model_Type": "Simple Stats (Voting)",
            "Score": row["Accuracy"]
        })
        
    df_simple_long = pd.DataFrame(simple_long)
    
    # 4. Merge
    df_final = pd.concat([df_ml_long, df_simple_long], ignore_index=True)
    
    # Filter to interesting targets if too many
    # For now, keep all
    
    # 5. Plotting
    methods = df_final["Method"].unique()
    
    sns.set_theme(style="whitegrid")
    
    for method in methods:
        data_subset = df_final[df_final["Method"] == method]
        
        if data_subset.empty: continue
        
        plt.figure(figsize=(14, 8))
        
        # Sort targets by max score to make it pretty
        target_order = data_subset.groupby("Target")["Score"].max().sort_values(ascending=False).index
        
        g = sns.barplot(
            data=data_subset,
            x="Target",
            y="Score",
            hue="Model_Type",
            order=target_order,
            palette="viridis",
            edgecolor="black"
        )
        
        plt.title(f"Model Comparison: {method} Method\n(Simple Stats vs. Machine Learning)", fontsize=16)
        plt.ylim(0, 1.0)
        plt.ylabel("Accuracy / F1 Score", fontsize=12)
        plt.xlabel("Clinical Target", fontsize=12)
        plt.xticks(rotation=45, ha='right')
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', title="Model Logic")
        plt.axhline(0.5, color='red', linestyle='--', alpha=0.5, label="Random Guess (approx)")
        
        plt.tight_layout()
        save_path = f"{OUTPUT_DIR}/comparison_{method}.png"
        plt.savefig(save_path)
        print(f"Saved plot: {save_path}")

if __name__ == "__main__":
    main()
