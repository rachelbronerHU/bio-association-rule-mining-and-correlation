import pandas as pd
import os
import glob
import sys
import ast
import argparse

# Add parent directory to path to allow importing constants
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from constants import CONSENSUS_RESULTS_EXPLORATION_DIR, RESULTS_DATA_DIR

METHODS = ["BAG", "CN", "KNN_R"]
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
INPUT_DIR = RESULTS_DATA_DIR
OUTPUT_DIR = CONSENSUS_RESULTS_EXPLORATION_DIR
TOP_N_PRINT = 5

def load_data(method):
    # Construct absolute path to ensure it works from any CWD
    file_path = os.path.join(PROJECT_ROOT, INPUT_DIR, f"results_{method}.csv")
    if not os.path.exists(file_path):
        print(f"Warning: File not found {file_path}")
        return None
    return pd.read_csv(file_path)

def filter_no_self(df):
    """
    Removes rules where Antecedents and Consequents have overlapping items.
    Example: ['A', 'B'] -> ['B'] should be removed.
    Also handles suffixes like _CENTER and _NEIGHBOR.
    """
    def clean_item(item):
        return item.replace("_CENTER", "").replace("_NEIGHBOR", "")

    def has_overlap(row):
        try:
            ant_raw = ast.literal_eval(row["Antecedents"])
            con_raw = ast.literal_eval(row["Consequents"])
            
            ant_clean = {clean_item(x) for x in ant_raw}
            con_clean = {clean_item(x) for x in con_raw}
            
            return not ant_clean.isdisjoint(con_clean)
        except:
            return False # Keep if parse fails (safety)

    # Filter
    mask = df.apply(has_overlap, axis=1)
    filtered_df = df[~mask]
    return filtered_df

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
    parser.add_argument("--top_n", type=int, default=50, help="Number of top rules to save per table.")
    parser.add_argument("--no_self", action="store_true", help="Exclude rules where Antecedents contain Consequents (or overlap).")
    
    args = parser.parse_args()
    
    suffix = "_no_self" if args.no_self else ""
    top_n = args.top_n
    
    abs_output_dir = os.path.join(PROJECT_ROOT, OUTPUT_DIR)
    if not os.path.exists(abs_output_dir):
        os.makedirs(abs_output_dir)
        
    print(f"Configuration: Top N={top_n}, No Self={args.no_self}")
    print(f"Output Directory: {abs_output_dir}")

    for method in METHODS:
        print(f"\n{'='*20} PROCESSING METHOD: {method} {'='*20}")
        df = load_data(method)
        if df is None:
            continue
            
        # Ensure we have the necessary columns
        req_cols = ["Biopsy_ID", "FOV", "Pathological stage", "Antecedents", "Consequents"]
        if not all(c in df.columns for c in req_cols):
            print(f"Error: Missing columns in {method} data. Found: {df.columns}")
            continue

        # Filter No Self if requested
        if args.no_self:
            print("Applying 'No Self' filter...")
            orig_len = len(df)
            df = filter_no_self(df)
            print(f"Filtered {orig_len - len(df)} occurrences. Remaining: {len(df)}")

        # --- GLOBAL CONSENSUS ---
        print("\n--- Level 3: Global Consensus (Dataset Stability) ---")
        print("Calculation: (Count of Total FOVs with Rule) / (Total FOVs in Dataset)")
        df_global = calculate_global_consensus(df)
        save_path = os.path.join(abs_output_dir, f"{method}_top_consensus_global{suffix}.csv")
        df_global.head(top_n).to_csv(save_path, index=False)
        print(df_global.head(TOP_N_PRINT).to_string(index=False))
        print(f"Saved top {top_n} to {save_path}")

        # --- STAGE CONSENSUS ---
        print("\n--- Level 2: Stage Consensus (Group Stability) ---")
        print("Calculation: (Count of FOVs in Stage with Rule) / (Total FOVs in Stage)")
        df_stage = calculate_stage_consensus(df)
        save_path = os.path.join(abs_output_dir, f"{method}_top_consensus_stage{suffix}.csv")
        df_stage.head(top_n).to_csv(save_path, index=False)
        print(df_stage.head(TOP_N_PRINT).to_string(index=False))
        print(f"Saved top {top_n} to {save_path}")
        
        # --- BIOPSY CONSENSUS ---
        print("\n--- Level 1: Biopsy Consensus (Intra-Patient Stability) ---")
        print("Calculation: (Count of FOVs in Biopsy with Rule) / (Total FOVs in Biopsy)")
        df_biopsy = calculate_biopsy_consensus(df)
        save_path = os.path.join(abs_output_dir, f"{method}_top_consensus_biopsy{suffix}.csv")
        df_biopsy.head(top_n).to_csv(save_path, index=False)
        print(df_biopsy.head(TOP_N_PRINT).to_string(index=False))
        print(f"Saved top {top_n} to {save_path}")

if __name__ == "__main__":
    main()