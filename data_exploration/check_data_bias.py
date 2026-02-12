import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns

import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from constants import MIBI_GUT_DIR_PATH, RESULTS_PLOTS_DIR

# Configuration
# We need to load the BIOPSY metadata, as that's the "Unit of Analysis".
# The cell table is too granular.
DATA_PATH = MIBI_GUT_DIR_PATH + "biopsy_metadata.csv"
OUTPUT_DIR = RESULTS_PLOTS_DIR + "data_bias_report/"

TARGETS = [
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

def ensure_dir(d):
    if not os.path.exists(d):
        os.makedirs(d)

def analyze_targets():
    ensure_dir(OUTPUT_DIR)
    
    if not os.path.exists(DATA_PATH):
        print(f"Error: Metadata file not found at {DATA_PATH}")
        return

    df = pd.read_csv(DATA_PATH)
    print(f"Loaded {len(df)} Biopsies.")
    
    report = []
    
    for target in TARGETS:
        if target not in df.columns:
            print(f"Skipping {target} (Not found in CSV)")
            continue
            
        # Drop NaNs
        valid_df = df[df[target].notna()]
        n = len(valid_df)
        
        if n == 0:
            print(f"Skipping {target} (All NaN)")
            continue
            
        # 1. Class Balance
        counts = valid_df[target].value_counts()
        props = valid_df[target].value_counts(normalize=True)
        
        # Calculate "Imbalance Score" (Gini Impurity-like or just Max Prop)
        # 0.5 = Perfectly balanced (2 classes). 1.0 = All one class.
        majority_class_prop = props.max()
        
        # 2. Print Stats
        print(f"\n--- {target} (N={n}) ---")
        for cls, count in counts.items():
            print(f"   {cls}: {count} ({props[cls]:.1%})")
            
        # 3. Store for Plot
        report.append({
            "Target": target,
            "N": n,
            "Majority_Class_Prop": majority_class_prop,
            "Majority_Class": counts.index[0],
            "Minority_Class": counts.index[-1],
            "Minority_Count": counts.values[-1]
        })
        
        # 4. Plot
        plt.figure(figsize=(6, 4))
        sns.countplot(data=valid_df, x=target, palette="viridis")
        plt.title(f"Distribution: {target}\n(N={n})")
        plt.ylabel("Number of Biopsies")
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(f"{OUTPUT_DIR}/dist_{target.replace(' ', '_').replace('/', '-')}.png")
        plt.close()

    # Summary Table
    summary_df = pd.DataFrame(report)
    summary_df = summary_df.sort_values("Majority_Class_Prop", ascending=False)
    
    print("\n=== BIAS SUMMARY (Sorted by Imbalance) ===")
    print(summary_df[["Target", "N", "Majority_Class_Prop", "Minority_Count"]].to_string(index=False))
    
    summary_df.to_csv(f"{OUTPUT_DIR}/bias_summary.csv", index=False)
    print(f"\nReport saved to {OUTPUT_DIR}/bias_summary.csv")

if __name__ == "__main__":
    analyze_targets()
