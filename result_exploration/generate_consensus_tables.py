import pandas as pd
import os
import glob
import sys
import ast
import argparse

# Add parent directory to path to allow importing constants
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from constants import CONSENSUS_RESULTS_EXPLORATION_DIR, RESULTS_DATA_DIR, METHODS

# Import organ stratification loader
from data_exploration.check_data_bias import load_stratified_biopsies

def load_bias_flags():
    """
    Load bias summary by organ to flag weak strata.
    Returns dict mapping (target, organ) -> viability info.
    """
    from constants import RESULTS_PLOTS_DIR
    bias_file = os.path.join(PROJECT_ROOT, RESULTS_PLOTS_DIR, "data_bias_report", "bias_summary_by_organ.csv")
    
    if not os.path.exists(bias_file):
        print(f"Warning: Bias summary file not found at {bias_file}")
        print("Run 'python data_exploration/check_data_bias.py' first to generate bias flags.")
        return {}
    
    bias_df = pd.read_csv(bias_file)
    
    # Create lookup dict
    bias_map = {}
    for _, row in bias_df.iterrows():
        key = (row["Target"], row["Organ"])
        bias_map[key] = {
            "viable": row["Viable"] == "✓",
            "reason": row["Reason"],
            "n_total": row["N_Total"],
            "n_classes": row["N_Classes"],
            "minority_count": row["Minority_Count"],
            "majority_pct": row["Majority_Pct"]
        }
    
    return bias_map

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
INPUT_DIR = RESULTS_DATA_DIR
OUTPUT_DIR = CONSENSUS_RESULTS_EXPLORATION_DIR
TOP_N_PRINT = 5

def load_data(method):
    """
    Load results CSV and merge with organ stratification data.
    Returns DataFrame with Organ column added.
    """
    # Construct absolute path to ensure it works from any CWD
    file_path = os.path.join(PROJECT_ROOT, INPUT_DIR, f"results_{method}.csv")
    if not os.path.exists(file_path):
        print(f"Warning: File not found {file_path}")
        return None
    
    df = pd.read_csv(file_path)
    
    # Load organ stratification data
    biopsy_df = load_stratified_biopsies()
    
    # Merge Organ column from biopsy data
    # Keep only Biopsy_ID and Organ to avoid duplicating other columns
    organ_map = biopsy_df[['Biopsy_ID', 'Organ']].drop_duplicates()
    df = df.merge(organ_map, on='Biopsy_ID', how='left')
    
    # Check for unmapped rows
    unmapped = df['Organ'].isna().sum()
    if unmapped > 0:
        print(f"Warning: {unmapped} rows have no organ mapping (will be excluded from stratified analysis)")
        # Show which biopsies are unmapped
        unmapped_biopsies = df[df['Organ'].isna()]['Biopsy_ID'].unique()
        print(f"Unmapped biopsies: {list(unmapped_biopsies)[:5]}...")
    
    return df

def filter_no_self(df):
    """
    Removes rules where Antecedents and Consequents have overlapping items.
    """
    from visualization.utils.visualization_util import filter_no_self_rules
    return filter_no_self_rules(df)

def calculate_biopsy_consensus(df):
    """
    Level 1: Biopsy Consensus (Intra-Patient Stability)
    Calculation: (Count of FOVs in Biopsy with Rule) / (Total FOVs in Biopsy)
    """
    # Total FOVs per Biopsy
    fovs_per_biopsy = df.groupby("Biopsy_ID")["FOV"].nunique()
    
    # Count FOVs per Rule per Biopsy
    # We group by Biopsy, Antecedents, Consequents
    rule_counts = df.groupby(["Biopsy_ID", "Antecedents", "Consequents"])["FOV"].nunique().reset_index(name="FOV_Count")
    
    # Calculate Score
    rule_counts["Total_FOVs_In_Biopsy"] = rule_counts["Biopsy_ID"].map(fovs_per_biopsy)
    rule_counts["Consensus_Score"] = rule_counts["FOV_Count"] / rule_counts["Total_FOVs_In_Biopsy"]
    
    # Sort
    rule_counts = rule_counts.sort_values("Consensus_Score", ascending=False)
    
    return rule_counts

def calculate_stage_consensus(df):
    """
    Level 2: Stage Consensus (Group Stability)
    Calculation: (Count of FOVs in Stage with Rule) / (Total FOVs in Stage)
    """
    # Total FOVs per Stage
    # We need unique FOVs per stage.
    # Note: A single FOV belongs to only one stage, so simple nunique works on the full df.
    fovs_per_stage = df.groupby("Pathological stage")["FOV"].nunique()
    
    # Count FOVs per Rule per Stage
    rule_counts = df.groupby(["Pathological stage", "Antecedents", "Consequents"])["FOV"].nunique().reset_index(name="FOV_Count")
    
    # Calculate Score
    rule_counts["Total_FOVs_In_Stage"] = rule_counts["Pathological stage"].map(fovs_per_stage)
    rule_counts["Consensus_Score"] = rule_counts["FOV_Count"] / rule_counts["Total_FOVs_In_Stage"]
    
    # Sort
    rule_counts = rule_counts.sort_values("Consensus_Score", ascending=False)
    
    return rule_counts

def calculate_global_consensus(df):
    """
    Level 3: Global Consensus (Dataset Stability)
    Calculation: (Count of Total FOVs with Rule) / (Total FOVs in Dataset)
    """
    # Total FOVs in Dataset
    total_fovs = df["FOV"].nunique()
    
    # Count FOVs with Rule
    rule_counts = df.groupby(["Antecedents", "Consequents"])["FOV"].nunique().reset_index(name="FOV_Count")
    
    # Calculate Score
    rule_counts["Total_FOVs_In_Dataset"] = total_fovs
    rule_counts["Consensus_Score"] = rule_counts["FOV_Count"] / total_fovs
    
    # Sort
    rule_counts = rule_counts.sort_values("Consensus_Score", ascending=False)
    
    return rule_counts

def main():
    parser = argparse.ArgumentParser(description="Generate Consensus Tables based on FOV counts.")
    parser.add_argument("--top_n", type=int, default=None, help="Optional: Number of top rules to save per table. If not provided, saves ALL rules.")
    parser.add_argument("--no_self", action="store_true", help="Exclude rules where Antecedents contain Consequents (or overlap).")
    parser.add_argument("--organs", nargs='+', help="Optional: Specific organs to process (e.g., --organs Colon Duodenum). If not provided, auto-discovers all organs.")
    
    args = parser.parse_args()
    
    suffix = "_no_self" if args.no_self else ""
    top_n = args.top_n
    
    abs_output_dir = os.path.join(PROJECT_ROOT, OUTPUT_DIR)
    if not os.path.exists(abs_output_dir):
        os.makedirs(abs_output_dir)
    
    top_n_desc = f"Top {top_n}" if top_n else "ALL"
    print(f"Configuration: Top N={top_n_desc}, No Self={args.no_self}, Organs Filter={args.organs}")
    print(f"Output Directory: {abs_output_dir}")
    
    # Load bias flags for reliability context
    bias_map = load_bias_flags()

    for method in METHODS:
        print(f"\n{'='*20} PROCESSING METHOD: {method} {'='*20}")
        df = load_data(method)
        if df is None:
            continue
            
        # Ensure we have the necessary columns
        req_cols = ["Biopsy_ID", "FOV", "Pathological stage", "Antecedents", "Consequents", "Organ"]
        if not all(c in df.columns for c in req_cols):
            print(f"Error: Missing columns in {method} data. Found: {df.columns}")
            continue

        # Filter No Self if requested
        if args.no_self:
            print("Applying 'No Self' filter...")
            orig_len = len(df)
            df = filter_no_self(df)
            print(f"Filtered {orig_len - len(df)} occurrences. Remaining: {len(df)}")

        # Auto-discover organs from data (exclude Unknown)
        available_organs = sorted([o for o in df["Organ"].dropna().unique() if o != "Unknown"])
        
        # Use user-specified organs if provided, otherwise use all discovered organs
        organs_to_process = args.organs if args.organs else available_organs
        
        print(f"\nDiscovered organs in data: {available_organs}")
        print(f"Processing organs: {organs_to_process}")
        
        # --- ORGAN-STRATIFIED CONSENSUS ---
        for organ in organs_to_process:
            if organ not in available_organs:
                print(f"\nWarning: Organ '{organ}' not found in data. Skipping.")
                continue
                
            print(f"\n{'='*60}")
            print(f"ORGAN: {organ}")
            print(f"{'='*60}")
            
            # Filter to this organ
            df_organ = df[df["Organ"] == organ].copy()
            n_fovs = df_organ["FOV"].nunique()
            n_biopsies = df_organ["Biopsy_ID"].nunique()
            
            print(f"Data: {n_biopsies} biopsies, {n_fovs} FOVs")
            
            if n_fovs == 0:
                print(f"No data for organ {organ}. Skipping.")
                continue

            # --- GLOBAL CONSENSUS (per organ) ---
            print("\n--- Level 3: Global Consensus (Dataset Stability) ---")
            print("Calculation: (Count of Total FOVs with Rule) / (Total FOVs in Dataset)")
            df_global = calculate_global_consensus(df_organ)
            save_path = os.path.join(abs_output_dir, f"{method}_top_consensus_global_{organ}{suffix}.csv")
            
            # Save with optional limit
            if top_n:
                df_global.head(top_n).to_csv(save_path, index=False)
                print(df_global.head(TOP_N_PRINT).to_string(index=False))
                print(f"Saved top {top_n} to {save_path}")
            else:
                df_global.to_csv(save_path, index=False)
                print(df_global.head(TOP_N_PRINT).to_string(index=False))
                print(f"Saved ALL {len(df_global)} rules to {save_path}")

            # --- STAGE CONSENSUS (per organ) ---
            print("\n--- Level 2: Stage Consensus (Group Stability) ---")
            print("Calculation: (Count of FOVs in Stage with Rule) / (Total FOVs in Stage)")
            df_stage = calculate_stage_consensus(df_organ)
            
            # Add reliability metadata (bias flags) for each stage
            if bias_map:
                df_stage["Stratum_Viable"] = df_stage["Pathological stage"].apply(
                    lambda stage: bias_map.get(("Pathological stage", organ), {}).get("viable", True)
                )
                df_stage["Stratum_Warning"] = df_stage["Pathological stage"].apply(
                    lambda stage: bias_map.get(("Pathological stage", organ), {}).get("reason", "")
                )
            
            save_path = os.path.join(abs_output_dir, f"{method}_top_consensus_stage_{organ}{suffix}.csv")
            
            # Save with optional limit
            if top_n:
                df_stage.head(top_n).to_csv(save_path, index=False)
                print(df_stage.head(TOP_N_PRINT).to_string(index=False))
                print(f"Saved top {top_n} to {save_path}")
            else:
                df_stage.to_csv(save_path, index=False)
                print(df_stage.head(TOP_N_PRINT).to_string(index=False))
                print(f"Saved ALL {len(df_stage)} rules to {save_path}")
            
            # --- BIOPSY CONSENSUS (per organ) ---
            print("\n--- Level 1: Biopsy Consensus (Intra-Patient Stability) ---")
            print("Calculation: (Count of FOVs in Biopsy with Rule) / (Total FOVs in Biopsy)")
            df_biopsy = calculate_biopsy_consensus(df_organ)
            save_path = os.path.join(abs_output_dir, f"{method}_top_consensus_biopsy_{organ}{suffix}.csv")
            
            # Save with optional limit
            if top_n:
                df_biopsy.head(top_n).to_csv(save_path, index=False)
                print(df_biopsy.head(TOP_N_PRINT).to_string(index=False))
                print(f"Saved top {top_n} to {save_path}")
            else:
                df_biopsy.to_csv(save_path, index=False)
                print(df_biopsy.head(TOP_N_PRINT).to_string(index=False))
                print(f"Saved ALL {len(df_biopsy)} rules to {save_path}")


if __name__ == "__main__":
    main()
