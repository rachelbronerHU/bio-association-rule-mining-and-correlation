import pandas as pd
import os

DATA_DIR = "results/full_run/data"
METHODS = ["BAG", "CN", "KNN_R", "WINDOW", "GRID"]

def show_top_rules():
    print(f"{'METHOD':<10} | {'RULE':<60} | {'LIFT':<8} | {'CONF':<8} | {'CONV':<8} | {'SUP':<8} | {'FDR':<8}")
    print("-" * 130)
    
    for method in METHODS:
        file_path = os.path.join(DATA_DIR, f"results_{method}.csv")
        if not os.path.exists(file_path):
            print(f"{method:<10} | File not found")
            continue
            
        try:
            df = pd.read_csv(file_path)
            if df.empty:
                print(f"{method:<10} | No rules found")
                continue
                
            # Check if Conviction exists
            if "Conviction" not in df.columns:
                print(f"{method:<10} | 'Conviction' column missing")
                continue

            # Create Rule ID for deduplication
            df["Rule_Str"] = df["Antecedents"].astype(str) + " -> " + df["Consequents"].astype(str)

            # Sort by Lift descending
            df_sorted = df.sort_values(by="Lift", ascending=False)
            
            # Keep top entry for each unique rule string
            unique_rules = df_sorted.drop_duplicates(subset=["Rule_Str"], keep="first").head(5)
            
            for _, row in unique_rules.iterrows():
                # Format Rule String
                ant = str(row['Antecedents']).replace("'", "").replace("[", "").replace("]", "")
                con = str(row['Consequents']).replace("'", "").replace("[", "").replace("]", "")
                rule_str = f"{ant} -> {con}"
                
                # Truncate rule string if too long
                if len(rule_str) > 58:
                    rule_str = rule_str[:55] + "..."
                
                # Get FDR if exists, else 'N/A'
                fdr = f"{row['FDR']:.4f}" if 'FDR' in row else "N/A"
                
                print(f"{method:<10} | {rule_str:<60} | {row['Lift']:<8.4f} | {row['Confidence']:<8.4f} | {row['Conviction']:<8.4f} | {row['Support']:<8.4f} | {fdr:<8}")
            
            print("-" * 130)
            
        except Exception as e:
            print(f"Error reading {method}: {e}")

if __name__ == "__main__":
    show_top_rules()