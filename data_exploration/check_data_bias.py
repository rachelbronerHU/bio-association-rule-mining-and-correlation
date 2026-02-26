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

def load_stratified_biopsies():
    """
    Load biopsy data with organ stratification.
    Returns DataFrame with Biopsy_ID, Organ, and all target columns.
    Can be imported by other scripts.
    """
    biopsy_path = DATA_PATH
    fovs_path = MIBI_GUT_DIR_PATH + "fovs_metadata.csv"
    
    if not os.path.exists(biopsy_path):
        raise FileNotFoundError(f"Metadata file not found at {biopsy_path}")
    
    df_biopsy = pd.read_csv(biopsy_path)
    
    # Load FOVs to get all biopsies (including controls)
    if not os.path.exists(fovs_path):
        raise FileNotFoundError(f"FOVs metadata not found at {fovs_path}")
    
    df_fovs = pd.read_csv(fovs_path)
    df_fovs = df_fovs[~df_fovs["FOV"].astype(str).str.startswith("S_")]
    
    # Get unique biopsies and their cohorts
    unique_biopsies = df_fovs[["Patient", "Cohort"]].drop_duplicates().rename(
        columns={"Patient": "Biopsy_ID"}
    )
    
    # Pre-process Biopsy Metadata (Shift Numerics) so 0 can be Control
    numeric_targets = []
    categorical_targets = []
    
    for col in TARGETS:
        if col in df_biopsy.columns:
            if pd.api.types.is_numeric_dtype(df_biopsy[col]):
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
    
    # Derive Organ from Cohort and Localization
    def get_organ(row):
        if pd.notna(row.get("Localization")):
            return row["Localization"]  # GVHD samples have Localization
        # Control samples: derive from Cohort name
        cohort = row["Cohort"]
        if "Colon" in cohort:
            return "Colon"
        elif "Duodenum" in cohort:
            return "Duodenum"
        return "Unknown"
    
    df["Organ"] = df.apply(get_organ, axis=1)
    df["Is_Control"] = df["Cohort"].apply(lambda x: "GVHD" not in x)
    
    return df

def analyze_targets_by_organ(df=None, min_biopsies=3, max_majority_pct=0.80):
    """
    Analyze class balance for each target×organ combination.
    
    Args:
        df: DataFrame from load_stratified_biopsies(). If None, will load it.
        min_biopsies: Minimum biopsies per class for viability
        max_majority_pct: Maximum proportion for majority class
    
    Returns:
        DataFrame with analysis results (can be used by other scripts)
    """
    if df is None:
        df = load_stratified_biopsies()
    
    results = []
    
    for target in TARGETS:
        if target not in df.columns:
            continue
        
        for organ in sorted(df["Organ"].unique()):
            if organ == "Unknown":
                continue
            
            df_stratum = df[df["Organ"] == organ].copy()
            valid_df = df_stratum[df_stratum[target].notna()]
            
            if len(valid_df) == 0:
                continue
            
            counts = valid_df[target].value_counts()
            props = valid_df[target].value_counts(normalize=True)
            
            # Check viability
            viable = True
            reason = "OK"
            
            if counts.min() < min_biopsies:
                viable = False
                reason = f"Class {counts.idxmin()} has only {counts.min()} biopsies (need ≥{min_biopsies})"
            elif props.max() > max_majority_pct:
                viable = False
                reason = f"Majority class ({counts.index[0]}) is {props.max():.1%} (need ≤{max_majority_pct:.0%})"
            
            results.append({
                "Target": target,
                "Organ": organ,
                "N_Total": len(valid_df),
                "N_Classes": len(counts),
                "Minority_Count": counts.values[-1],
                "Majority_Pct": props.max(),
                "Majority_Pct_Str": f"{props.max():.1%}",
                "Viable": "✓" if viable else "✗",
                "Reason": reason,
                "Class_Counts": dict(counts)
            })
    
    return pd.DataFrame(results)

def analyze_targets():
    """Original function - now calls new stratified analysis"""
    ensure_dir(OUTPUT_DIR)
    
    # Load stratified data
    df = load_stratified_biopsies()
    print(f"Loaded {len(df)} Biopsies")
    print(f"Organs: {sorted(df['Organ'].unique())}")
    print(f"Controls: {df['Is_Control'].sum()}, GVHD: {(~df['Is_Control']).sum()}")
    print()
    
    # --- ORGAN-STRATIFIED ANALYSIS ---
    print("=" * 80)
    print("ORGAN-STRATIFIED BIAS ANALYSIS")
    print("=" * 80)
    
    results_df = analyze_targets_by_organ(df, min_biopsies=3, max_majority_pct=0.80)
    
    # Print summary
    print("\nSummary Table:")
    print(results_df[["Target", "Organ", "N_Total", "N_Classes", "Minority_Count", "Majority_Pct_Str", "Viable", "Reason"]].to_string(index=False))
    
    # Save stratified results
    stratified_csv = f"{OUTPUT_DIR}/bias_summary_by_organ.csv"
    results_df.to_csv(stratified_csv, index=False)
    print(f"\n✓ Stratified report saved to {stratified_csv}")
    
    # Print viability summary
    viable_count = (results_df["Viable"] == "✓").sum()
    total_count = len(results_df)
    print(f"\nViable strata: {viable_count}/{total_count} ({viable_count/total_count:.1%})")
    
    non_viable = results_df[results_df["Viable"] == "✗"]
    if not non_viable.empty:
        print(f"\nNON-VIABLE STRATA (will be skipped in correlation analysis):")
        for _, row in non_viable.iterrows():
            print(f"  • {row['Target']} × {row['Organ']}: {row['Reason']}")
    
    # --- ORGAN-STRATIFIED PLOTS ---
    print("\n--- Generating Organ-Stratified Plots ---")
    for target in TARGETS:
        if target not in df.columns:
            continue
        
        # Check if we have data for multiple organs
        organs_with_data = []
        for organ in sorted(df["Organ"].unique()):
            if organ == "Unknown":
                continue
            df_organ = df[(df["Organ"] == organ) & (df[target].notna())]
            if len(df_organ) > 0:
                organs_with_data.append(organ)
        
        if len(organs_with_data) == 0:
            continue
        
        # Create side-by-side plots for each organ
        fig, axes = plt.subplots(1, len(organs_with_data), figsize=(6*len(organs_with_data), 4))
        if len(organs_with_data) == 1:
            axes = [axes]  # Make it iterable
        
        for idx, organ in enumerate(organs_with_data):
            df_organ = df[(df["Organ"] == organ) & (df[target].notna())]
            n = len(df_organ)
            
            # Get sorted classes
            counts = df_organ[target].value_counts()
            sorted_classes = sorted(counts.index)
            
            # Get viability info for subtitle
            viable_info = results_df[
                (results_df["Target"] == target) & 
                (results_df["Organ"] == organ)
            ]
            viable_marker = "✓" if not viable_info.empty and viable_info.iloc[0]["Viable"] == "✓" else "✗"
            
            ax = axes[idx]
            sns.countplot(data=df_organ, x=target, palette="viridis", order=sorted_classes, ax=ax)
            ax.set_title(f"{organ} {viable_marker}\n(N={n})")
            ax.set_ylabel("Number of Biopsies")
            ax.set_xlabel(target if idx == len(organs_with_data)//2 else "")
            ax.tick_params(axis='x', rotation=45)
        
        plt.suptitle(f"Distribution by Organ: {target}", fontsize=14, y=1.02)
        plt.tight_layout()
        plt.savefig(f"{OUTPUT_DIR}/dist_by_organ_{target.replace(' ', '_').replace('/', '-')}.png", bbox_inches='tight')
        plt.close()
        print(f"  Saved: dist_by_organ_{target.replace(' ', '_').replace('/', '-')}.png")
    
    print("\n" + "=" * 80)
    print("GLOBAL (POOLED) BIAS ANALYSIS")
    print("=" * 80)
    
    # --- ORIGINAL GLOBAL ANALYSIS (kept for backward compatibility) ---
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
