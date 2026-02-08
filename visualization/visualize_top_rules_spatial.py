import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import ast
import sys
import numpy as np
import argparse

# Add project root to sys.path to import constants
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import constants

# --- Configuration ---
CELL_TABLE_PATH = os.path.join(constants.MIBI_GUT_DIR_PATH, 'cell_table.csv')
RESULTS_DIR = constants.RESULTS_DATA_DIR
OUTPUT_DIR = os.path.join(constants.RESULTS_PLOTS_DIR, 'top_rules_spatial')
METHODS = ["BAG", "CN", "KNN_R"]

# --- Helper Functions ---

def load_cell_data(path):
    print(f"Loading cell data from {path}...")
    df = pd.read_csv(path)
    # Ensure consistent column naming
    if 'centroid_x' in df.columns:
        df = df.rename(columns={'centroid_x': 'x', 'centroid_y': 'y'})
    return df

def clean_rule_item(item_str):
    """
    Cleans rule items like 'Goblet_CENTER' or 'Epithelial_NEIGHBOR' 
    to match cell types in the cell table (e.g., 'Goblet').
    """
    return item_str.replace('_CENTER', '').replace('_NEIGHBOR', '')

def parse_rule_list(rule_str):
    """
    Parses string representation of list "['A', 'B']" into a python list.
    """
    try:
        items = ast.literal_eval(rule_str)
        return [clean_rule_item(i) for i in items]
    except:
        return [clean_rule_item(str(rule_str).strip("[]'\""))]

def get_color_map(cell_types):
    """Generates a consistent color map for all cell types."""
    # Use 'husl' for vibrant, distinct colors (avoiding greys)
    palette = sns.color_palette("husl", n_colors=len(cell_types))
    return dict(zip(sorted(cell_types), palette))

def plot_fov_full(ax, df_fov, color_map):
    """Plots the full FOV with all cell types colored."""
    df_shuffled = df_fov.sample(frac=1, random_state=42)
    
    for cell_type, group in df_shuffled.groupby('cell type'):
        ax.scatter(
            group['x'],
            group['y'],
            label=cell_type,
            color=color_map.get(cell_type, 'gray'),
            s=12,
            alpha=1.0, # Increased alpha for stronger colors
            edgecolors='none'
        )
    ax.set_title("Full Tissue Map", fontsize=10)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.invert_yaxis() 
    # Legend only on the full plot
    ax.legend(bbox_to_anchor=(-0.15, 1), loc='upper right', fontsize='x-small', markerscale=2, title="Cell Types")

def plot_fov_rule_highlight(ax, df_fov, antecedents, consequents, color_map):
    """
    Plots the FOV highlighting only the rule's cells using their original colors.
    """
    COLOR_OTHER = "lightgray"

    def assign_category(row_type):
        if row_type in antecedents:
            return 2 
        elif row_type in consequents:
            return 2 
        else:
            return 0 

    df_fov = df_fov.copy()
    df_fov['plot_order'] = df_fov['cell type'].apply(assign_category)
    df_fov = df_fov.sort_values('plot_order')

    # Background
    mask_other = df_fov['plot_order'] == 0
    ax.scatter(df_fov[mask_other]['x'], df_fov[mask_other]['y'], c=COLOR_OTHER, s=5, alpha=0.2)
    
    # Active Rule Cells (Antecedents & Consequents)
    mask_active = df_fov['plot_order'] == 2
    
    if mask_active.any():
        active_cells = df_fov[mask_active]
        # Plot each type individually to get correct color from map
        for cell_type, group in active_cells.groupby('cell type'):
             ax.scatter(
                group['x'], 
                group['y'], 
                c=[color_map.get(cell_type, 'black')], 
                s=15, 
                alpha=0.9, 
                edgecolors='none'
            )

    ax.set_title("Rule Interaction Highlight", fontsize=10)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.invert_yaxis()
    # No legend on this side as requested

def visualize_method(method_name, cell_df, color_map, top_n, exclude_self_rules=False, exclude_containing_self_rules=False):
    results_path = os.path.join(RESULTS_DIR, f"results_{method_name}.csv")
    if not os.path.exists(results_path):
        print(f"Skipping {method_name}: File not found.")
        return

    print(f"Processing {method_name} (Top {top_n})...")
    rules_df = pd.read_csv(results_path)
    
    if 'Lift' not in rules_df.columns:
        print(f"Skipping {method_name}: 'Lift' column missing.")
        return

    # Create a unique rule identifier for deduplication
    rules_df['Rule_Str'] = rules_df['Antecedents'] + "->" + rules_df['Consequents']

    # 1. Filter: Exclude Containing Self (Stricter)
    if exclude_containing_self_rules:
        # Check intersection: if any item is in both antecedent and consequent
        def has_overlap(row):
            ants = parse_rule_list(row['Antecedents'])
            cons = parse_rule_list(row['Consequents'])
            return not set(ants).isdisjoint(set(cons))
        
        rules_df = rules_df[~rules_df.apply(has_overlap, axis=1)]
        
    # 2. Filter: Exclude Exact Self Rules (Simpler)
    # Only run if strict filter wasn't applied (since strict covers simple)
    elif exclude_self_rules:
        def is_self_rule(row):
            ants = parse_rule_list(row['Antecedents'])
            cons = parse_rule_list(row['Consequents'])
            return set(ants) == set(cons)
        
        rules_df = rules_df[~rules_df.apply(is_self_rule, axis=1)]

    # Sort by Lift descending
    rules_df = rules_df.sort_values(by="Lift", ascending=False)
    
    # Drop duplicates to keep only the best instance of each unique rule
    unique_rules = rules_df.drop_duplicates(subset=['Rule_Str'], keep='first')
    
    # Take top N
    top_rules = unique_rules.head(top_n)
    
    if top_rules.empty:
        print(f"No rules found for {method_name}.")
        return

    # Layout: Rows = top_n, Cols = 2
    ROW_HEIGHT = 6
    total_fig_height = ROW_HEIGHT * top_n
    fig, axes = plt.subplots(nrows=top_n, ncols=2, figsize=(18, total_fig_height))
    if top_n == 1: axes = np.array([axes])

    plt.subplots_adjust(top=0.92, hspace=0.8, wspace=0.2)
    
    for idx, (_, row) in enumerate(top_rules.iterrows()):
        fov_id = row['FOV']
        
        # Parse Rule
        raw_ant_str = row['Antecedents']
        raw_con_str = row['Consequents']
        ants = parse_rule_list(raw_ant_str)
        cons = parse_rule_list(raw_con_str)
        
        # Metrics
        lift_val = row.get('Lift', np.nan)
        conf_val = row.get('Confidence', np.nan)
        supp_val = row.get('Support', np.nan)
        conv_val = row.get('Conviction', np.nan)
        fdr_val = row.get('FDR', np.nan)
        pval_val = row.get('P_Value', np.nan)

        df_fov = cell_df[cell_df['fov'] == fov_id]
        
        ax_full = axes[idx][0]
        ax_rule = axes[idx][1]

        if df_fov.empty:
            ax_full.text(0.5, 0.5, f"FOV {fov_id} not found", ha='center')
            ax_full.axis('off')
            ax_rule.axis('off')
            continue

        # Plotting
        plot_fov_full(ax_full, df_fov, color_map)
        plot_fov_rule_highlight(ax_rule, df_fov, ants, cons, color_map)
        
        # --- Row Title with Metrics ---
        # Formatting metrics nicely
        # Use .4f for FDR/P-Val to avoid scientific notation unless very small
        def fmt_p(val):
            return f"{val:.4f}" if val > 0.0001 else f"{val:.2e}"

        metrics_str = (
            f"Lift: {lift_val:.2f} | Conf: {conf_val:.2f} | Supp: {supp_val:.3f} | "
            f"Conv: {conv_val:.2f} | FDR: {fmt_p(fdr_val)} | P-Val: {fmt_p(pval_val)}"
        )
        
        title_str = (f"Rank #{idx+1} | FOV: {fov_id}\n"
                     f"Rule: {raw_ant_str} -> {raw_con_str}\n"
                     f"{metrics_str}")
        
        # Centering title over the pair of plots
        box_full = ax_full.get_position()
        box_rule = ax_rule.get_position()
        center_x = (box_full.x0 + box_rule.x1) / 2
        
        # Calculate dynamic offset to keep title specific distance (in inches) above plot
        TITLE_OFFSET_INCHES = 0.25
        offset_frac = TITLE_OFFSET_INCHES / total_fig_height
        
        top_y = box_full.y1 + offset_frac
        
        fig.text(center_x, top_y, title_str, ha='center', va='bottom', 
                 fontsize=12, fontweight='bold', bbox=dict(facecolor='white', alpha=0.9, edgecolor='lightgray'))

    # Determine Output Filename
    suffix = ""
    if exclude_containing_self_rules:
        suffix = "_no_containing_self"
    elif exclude_self_rules:
        suffix = "_no_self"

    out_path = os.path.join(OUTPUT_DIR, f"{method_name}_top_rules{suffix}.png")
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved visualization to {out_path}")

def main():
    parser = argparse.ArgumentParser(description="Visualize Top Spatial Rules")
    parser.add_argument("--top_n", type=int, default=3, help="Number of top rules to visualize per method")
    parser.add_argument("--exclude_self_rules", action="store_true", help="Filter out rules where antecedent set EQUALS consequent set.")
    parser.add_argument("--exclude_containing_self_rules", action="store_true", help="Filter out rules where ANY cell type appears in both antecedent and consequent.")
    args = parser.parse_args()

    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR, exist_ok=True)

    if not os.path.exists(CELL_TABLE_PATH):
        print(f"Error: Cell table not found at {CELL_TABLE_PATH}")
        return

    cell_df = load_cell_data(CELL_TABLE_PATH)
    all_cell_types = cell_df['cell type'].dropna().unique()
    color_map = get_color_map(all_cell_types)

    for method in METHODS:
        visualize_method(method, cell_df, color_map, 
                         top_n=args.top_n, 
                         exclude_self_rules=args.exclude_self_rules,
                         exclude_containing_self_rules=args.exclude_containing_self_rules)

if __name__ == "__main__":
    main()