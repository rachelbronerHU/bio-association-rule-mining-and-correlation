import pandas as pd
import os
import glob

from constants import RESULTS_DATA_DIR


RESULTS_DIR = RESULTS_DATA_DIR
METHODS = ["BAG", "CN", "KNN_R", "WINDOW", "GRID"]

def analyze_results():
    print(f"{'METHOD':<10} | {'RULE':<60} | {'LIFT':<8} | {'CONF':<8} | {'CONV':<8} | {'SUP':<8}")
    print("-" * 120)
    
    for method in METHODS:
        file_path = os.path.join(RESULTS_DIR, f"results_{method}.csv")
        if not os.path.exists(file_path):
            print(f"{method:<10} | File not found")
            continue
            
        try:
            df = pd.read_csv(file_path)
            if df.empty:
                print(f"{method:<10} | No rules found")
                continue
                
            # Check if Conviction exists (it should)
            if "Conviction" not in df.columns:
                print(f"{method:<10} | 'Conviction' column missing")
                continue

            # Sort by Lift descending
            top_rules = df.sort_values(by="Lift", ascending=False).head(3)
            
            for _, row in top_rules.iterrows():
                # Format Rule String
                ant = row['Antecedents'].replace("'", "").replace("[", "").replace("]", "")
                con = row['Consequents'].replace("'", "").replace("[", "").replace("]", "")
                rule_str = f"{ant} -> {con}"
                
                # Truncate rule string if too long
                if len(rule_str) > 58:
                    rule_str = rule_str[:55] + "..."
                
                print(f"{method:<10} | {rule_str:<60} | {row['Lift']:<8.4f} | {row['Confidence']:<8.4f} | {row['Conviction']:<8.4f} | {row['Support']:<8.4f}")
            
            print("-" * 120)
            
        except Exception as e:
            print(f"Error reading {method}: {e}")

if __name__ == "__main__":
    analyze_results()