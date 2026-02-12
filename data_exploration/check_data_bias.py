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

    df_biopsy = pd.read_csv(DATA_PATH)
    print(f"Loaded {len(df_biopsy)} Biopsies from metadata.")

    # --- Load FOVs and Merge Controls (Logic from run_association_mining.py) ---
    fovs_path = MIBI_GUT_DIR_PATH + "fovs_metadata.csv"
    
    if os.path.exists(fovs_path):
        df_fovs = pd.read_csv(fovs_path)
        
        # Filter unknown 'S_*' (same as _remove_unknown_biopsies)
        df_fovs = df_fovs[~df_fovs["FOV"].astype(str).str.startswith("S_")]
        
        # Get unique Biopsies from FOVs
        # "Patient" in fovs_metadata maps to "Biopsy_ID"
        unique_biopsies = df_fovs[["Patient"]].drop_duplicates().rename(columns={"Patient": "Biopsy_ID"})
        print(f"Total Biopsies from FOVs (excluding S_*): {len(unique_biopsies)}")

        # Pre-process Biopsy Metadata (Shift Numerics) so 0 can be Control
        numeric_targets = []
        categorical_targets = []
        
        for col in TARGETS:
            if col in df_biopsy.columns:
                 if pd.api.types.is_numeric_dtype(df_biopsy[col]):
                    # Shift 0->1, 1->2... so 0 can be Control
                    df_biopsy[col] = df_biopsy[col] + 1
                    numeric_targets.append(col)
                 else:
                    categorical_targets.append(col)

        # Merge unique_biopsies (All) with df_biopsy (Metadata)
        df = pd.merge(unique_biopsies, df_biopsy, on="Biopsy_ID", how="left")
        
        # Fill NaNs (Controls)
        for col in numeric_targets:
            df[col] = df[col].fillna(0)
        
        for col in categorical_targets:
            df[col] = df[col].fillna("Control")
            
        print(f"Final Merged Biopsies for Analysis: {len(df)}")
    else:
        print(f"Warning: FOVs metadata not found at {fovs_path}. Using only Biopsy metadata.")
        df = df_biopsy
    
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

        # Sort classes for Display and Plotting
        sorted_classes = sorted(counts.index)
        
        # 2. Print Stats
        print(f"\n--- {target} (N={n}) ---")
        for cls in sorted_classes:
            count = counts[cls]
            prop = props[cls]
            print(f"   {cls}: {count} ({prop:.1%})")
            
        # 3. Store for Plot
        report.append({
            "Target": target,
            "N": n,
            "Majority_Class_Prop": majority_class_prop,
            "Majority_Class": counts.index[0], # Keep these based on frequency (counts)
            "Minority_Class": counts.index[-1],
            "Minority_Count": counts.values[-1]
        })
        
        # 4. Plot
        plt.figure(figsize=(6, 4))
        sns.countplot(data=valid_df, x=target, palette="viridis", order=sorted_classes)
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
