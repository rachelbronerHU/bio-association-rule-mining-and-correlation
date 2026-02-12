import pandas as pd
import os
import sys
import re

DATA_DIR = "results/full_run/data"
METHODS = ["BAG", "CN", "KNN_R"]
README_PATH = "README.md"

def get_top_rules_markdown(style="blockquote"):
    """Generates Markdown table for Top Rules (Pos & Neg) for README."""
    prefix = "> " if style == "blockquote" else ""
    
    md = f"{prefix}| METHOD | RULE | LIFT | CONF | CONV | SUP | FDR |\n"
    md += f"{prefix}| :--- | :--- | :--- | :--- | :--- | :--- | :--- |\n"
    
    for method in METHODS:
        file_path = os.path.join(DATA_DIR, f"results_{method}.csv")
        if not os.path.exists(file_path): continue
        
        try:
            df = pd.read_csv(file_path)
            if df.empty: continue
            
            # Create Rule ID
            df["Rule_Str"] = df["Antecedents"].astype(str) + " -> " + df["Consequents"].astype(str)
            if "Conviction" not in df.columns: df["Conviction"] = 0
            
            # --- Select Top Rules ---
            # 1. Top 2 Positive (Highest Lift)
            top_pos = df[df["Lift"] >= 1.0].sort_values("Lift", ascending=False).drop_duplicates("Rule_Str").head(2)
            
            # 2. Top 2 Negative (Lowest Lift)
            top_neg = df[df["Lift"] < 1.0].sort_values("Lift", ascending=True).drop_duplicates("Rule_Str").head(2)
            
            # Combine
            # We want to show them grouped by method.
            # Convert to list of rows to iterate cleanly
            rows_to_print = pd.concat([top_pos, top_neg])
            
            for _, row in rows_to_print.iterrows():
                ant = str(row['Antecedents']).replace("'", "").replace("[", "").replace("]", "")
                con = str(row['Consequents']).replace("'", "").replace("[", "").replace("]", "")
                rule = f"{ant} -> {con}"
                
                fdr = f"{row['FDR']:.4f}" if 'FDR' in row else "N/A"
                # Handle inf conviction
                conv = f"{row['Conviction']:.2f}" if row['Conviction'] != float('inf') else "inf"
                
                md += f"{prefix}| **{method}** | {rule} | {row['Lift']:.2f} | {row['Confidence']:.2f} | {conv} | {row['Support']:.3f} | {fdr} |\n"
            
            # Separator (Visual only, using standard markdown table syntax for empty row or just skip)
            # README used `| --- |` which renders as a separator line in some viewers or just text.
            # To match README style:
            md += f"{prefix}| --- | --- | --- | --- | --- | --- | --- |\n"
            
        except Exception as e:
            print(f"Error processing {method}: {e}")
            
    return md

def update_readme():
    if not os.path.exists(README_PATH):
        print(f"Error: {README_PATH} not found.")
        return

    with open(README_PATH, "r", encoding="utf-8") as f:
        content = f.read()

    # Target Section
    section_marker = "## 3. Top Ranked Rules by Method"
    
    start_idx = content.find(section_marker)
    if start_idx == -1:
        print("Error: Target section not found in README.")
        return
        
    # Find start of table
    table_header_sig = "| METHOD | RULE |"
    
    table_start = content.find(table_header_sig, start_idx)
    if table_start == -1:
        print("Error: Old table header not found.")
        return
        
    # Find the beginning of that line (to check for >)
    line_start = content.rfind("\n", start_idx, table_start) + 1
    
    # Determine current style
    current_prefix = content[line_start:table_start]
    style = "blockquote" if ">" in current_prefix else "standard"
    
    # Find end of table
    # Scan for next double newline
    table_end = table_start
    for match in re.finditer(r"\n\s*\n", content[table_start:]):
        table_end = table_start + match.start()
        break
    else:
        table_end = len(content)

    # Generate new table
    new_table = get_top_rules_markdown(style)
    
    # Replace
    new_content = content[:line_start] + new_table + content[table_end:]
    
    with open(README_PATH, "w", encoding="utf-8") as f:
        f.write(new_content)
    
    print(f"Successfully updated README table under '{section_marker}'.")

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
                
            if "Conviction" not in df.columns: df["Conviction"] = 0
            df["Rule_Str"] = df["Antecedents"].astype(str) + " -> " + df["Consequents"].astype(str)

            # --- Helper to print rows ---
            def print_rows(rows, label):
                if rows.empty: return
                print(f"--- {label} ---")
                for _, row in rows.iterrows():
                    ant = str(row['Antecedents']).replace("'", "").replace("[", "").replace("]", "")
                    con = str(row['Consequents']).replace("'", "").replace("[", "").replace("]", "")
                    rule_str = f"{ant} -> {con}"
                    if len(rule_str) > 58: rule_str = rule_str[:55] + "..."
                    fdr = f"{row['FDR']:.4f}" if 'FDR' in row else "N/A"
                    print(f"{method:<10} | {rule_str:<60} | {row['Lift']:<8.4f} | {row['Confidence']:<8.4f} | {row['Conviction']:<8.4f} | {row['Support']:<8.4f} | {fdr:<8}")

            # 1. Positive Rules (High Lift)
            df_pos = df[df["Lift"] >= 1.0].sort_values(by="Lift", ascending=False)
            top_pos = df_pos.drop_duplicates(subset=["Rule_Str"], keep="first").head(5)
            print_rows(top_pos, "TOP POSITIVE (Lift >= 1.0)")

            # 2. Negative Rules (Low Lift)
            df_neg = df[df["Lift"] < 1.0].sort_values(by="Lift", ascending=True)
            top_neg = df_neg.drop_duplicates(subset=["Rule_Str"], keep="first").head(5)
            print_rows(top_neg, "TOP NEGATIVE (Lift < 1.0)")
            
            print("-" * 130)
            
        except Exception as e:
            print(f"Error reading {method}: {e}")

if __name__ == "__main__":
    show_top_rules()
    if "--update-readme" in sys.argv:
        update_readme()
