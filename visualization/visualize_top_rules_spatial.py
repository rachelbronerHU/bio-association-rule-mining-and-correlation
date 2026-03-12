import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import ast
import sys
import numpy as np
import argparse
from scipy.spatial import ConvexHull
from matplotlib.patches import Polygon

# Add project root to sys.path to import constants
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import constants

# --- Configuration ---
CELL_TABLE_PATH = os.path.join(constants.MIBI_GUT_DIR_PATH, 'cell_table.csv')
RESULTS_DIR = constants.RESULTS_DATA_DIR
OUTPUT_DIR = os.path.join(constants.RESULTS_PLOTS_DIR, 'top_rules_spatial')
METHODS = constants.METHODS

# Boolean columns for tissue compartments
TISSUE_BOOLEAN_COLS = [
    'in_CryptVilli', 'in_BrunnerGland', 'in_SMV', 'in_Muscle',
    'in_LP', 'in_Submucosa', 'in_Follicle', 'in_Lumen'
]

# Tissue region color map (pastel colors, distinct from cell types)
TISSUE_COLOR_MAP = {
    'in_CryptVilli': '#C8E6C9',    # Light green
    'in_BrunnerGland': '#FFECB3',  # Light yellow
    'in_SMV': '#BBDEFB',           # Light blue
    'in_Muscle': '#FFCCBC',        # Light coral
    'in_LP': '#E1BEE7',            # Light purple
    'in_Submucosa': '#F8BBD0',     # Light pink
    'in_Follicle': '#DCEDC8',      # Lime
    'in_Lumen': '#CFD8DC'          # Light gray-blue
}

TISSUE_DISPLAY_NAMES = {
    'in_CryptVilli': 'Crypt/Villi',
    'in_BrunnerGland': 'Brunner Gland',
    'in_SMV': 'SMV',
    'in_Muscle': 'Muscle',
    'in_LP': 'Lamina Propria',
    'in_Submucosa': 'Submucosa',
    'in_Follicle': 'Follicle',
    'in_Lumen': 'Lumen'
}

# --- Helper Functions ---

def load_cell_data(path):
    print(f"Loading cell data from {path}...")
    df = pd.read_csv(path)
    # Ensure consistent column naming
    if 'centroid_x' in df.columns:
        df = df.rename(columns={'centroid_x': 'x', 'centroid_y': 'y'})
    
    # Ensure boolean columns exist
    for col in TISSUE_BOOLEAN_COLS:
        if col not in df.columns:
            df[col] = False
        else:
            # Convert string TRUE/FALSE to boolean if needed
            if df[col].dtype == 'object':
                df[col] = df[col].map({'TRUE': True, 'FALSE': False, True: True, False: False})
    
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
    """Generates a consistent color map for cell types in the FOV."""
    # Use 'husl' for vibrant, distinct colors (avoiding greys)
    palette = sns.color_palette("husl", n_colors=len(cell_types))
    return dict(zip(sorted(cell_types), palette))

def plot_tissue_backgrounds(ax, df_fov, alpha=0.15):
    """
    Plots tissue compartment backgrounds as convex hulls.
    
    Args:
        ax: matplotlib axis
        df_fov: DataFrame with x, y, and boolean tissue columns
        alpha: Transparency for background regions
    """
    from matplotlib.patches import Polygon
    from scipy.spatial import ConvexHull
    
    for tissue_col in TISSUE_BOOLEAN_COLS:
        if tissue_col not in df_fov.columns:
            continue
        
        # Get cells in this tissue region
        tissue_cells = df_fov[df_fov[tissue_col] == True]
        
        if len(tissue_cells) < 3:  # Need at least 3 points for convex hull
            continue
        
        coords = tissue_cells[['x', 'y']].values
        
        try:
            # Compute convex hull
            hull = ConvexHull(coords)
            hull_points = coords[hull.vertices]
            
            # Create polygon
            polygon = Polygon(hull_points, 
                            facecolor=TISSUE_COLOR_MAP.get(tissue_col, 'lightgray'),
                            edgecolor='black',  # Added black border
                            linewidth=2,  # Border width
                            alpha=alpha,
                            zorder=0)  # Draw behind cells
            ax.add_patch(polygon)
        except Exception as e:
            # Skip if convex hull fails (e.g., collinear points)
            continue

def get_constant_scatter_size(ax, fig, cell_diam_um=10.0, fov_size_um=400.0, fov_size_px=1024.0):
    """
    Calculates the exact Matplotlib 's' parameter to render a cell of constant biological diameter.
    """
    fig.canvas.draw()
    bbox = ax.get_window_extent()
    ax_height_display_px = bbox.height
    
    # 1. Biological size to image pixels
    px_per_um = fov_size_px / fov_size_um
    cell_diam_px = cell_diam_um * px_per_um
    
    # 2. Image pixels to display points
    y_min, y_max = ax.get_ylim()
    data_height = abs(y_max - y_min)
    if data_height == 0: data_height = fov_size_px
        
    display_px_per_data_px = ax_height_display_px / data_height
    pts_per_data_px = display_px_per_data_px * (72.0 / fig.dpi)
    
    cell_diam_pts = cell_diam_px * pts_per_data_px
    
    # 3. Return Area in points^2
    return cell_diam_pts ** 2


def plot_fov_full(ax, fig, df_fov, color_map, show_tissue_background=False):
    """Plots the full FOV with all cell types colored."""
    
    if show_tissue_background:
        plot_tissue_backgrounds(ax, df_fov, alpha=0.5)
    
    # Determine FOV size from data limits to establish scale
    fov_max_x = df_fov['x'].max() if df_fov['x'].max() > 0 else 1024.0
    fov_max_y = df_fov['y'].max() if df_fov['y'].max() > 0 else 1024.0
    ax.set_xlim(0, fov_max_x)
    ax.set_ylim(fov_max_y, 0) # Inverted Y-axis
    ax.set_aspect('equal', adjustable='box')
    
    # Calculate a single, constant cell size for a 10um cell
    # (Assuming standard 1024px = 400um ratio, scales automatically for 2048px/800um)
    fov_size_um = 800.0 if fov_max_x > 1500 else 400.0
    constant_size = get_constant_scatter_size(ax, fig, cell_diam_um=10.0, fov_size_um=fov_size_um, fov_size_px=fov_max_x)
    
    df_shuffled = df_fov.sample(frac=1, random_state=42)
    
    for cell_type, group in df_shuffled.groupby('cell type'):
        ax.scatter(
            group['x'], group['y'],
            label=cell_type,
            color=color_map.get(cell_type, 'gray'),
            s=constant_size,
            alpha=1.0,
            edgecolors='none',
            zorder=2
        )
    
    ax.set_title("Full Tissue Map", fontsize=10)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    
    from matplotlib.ticker import MultipleLocator
    radius_um = constants.CONFIG.get('RADIUS', 25)
    pixel_per_um = fov_max_x / fov_size_um
    grid_spacing = radius_um * pixel_per_um
    
    ax.xaxis.set_minor_locator(MultipleLocator(grid_spacing))
    ax.yaxis.set_minor_locator(MultipleLocator(grid_spacing))
    ax.grid(which='minor', color='gray', linestyle=':', linewidth=0.5, alpha=0.5)
    
    handles, labels = ax.get_legend_handles_labels()
    for handle in handles:
        if hasattr(handle, 'set_sizes'):
            handle.set_sizes([30])
    
    if show_tissue_background:
        from matplotlib.patches import Patch
        tissue_handles = []
        tissue_labels = []
        for tissue_col in TISSUE_BOOLEAN_COLS:
            if tissue_col in df_fov.columns and df_fov[tissue_col].any():
                tissue_handles.append(Patch(facecolor=TISSUE_COLOR_MAP[tissue_col], edgecolor='black', linewidth=1, alpha=0.7))
                tissue_labels.append(TISSUE_DISPLAY_NAMES[tissue_col])
        if tissue_handles:
            handles = tissue_handles + handles
            labels = tissue_labels + labels
    
    ax.legend(handles, labels, bbox_to_anchor=(-0.15, 1), loc='upper right', fontsize='x-small', title="Regions & Cell Types")
    return constant_size

def plot_fov_rule_highlight(ax, fig, df_fov, antecedents, consequents, color_map, constant_size):
    """
    Plots the FOV highlighting only the rule's cells.
    Receives exact constant_size from left plot to guarantee perfectly identical rendering.
    """
    COLOR_OTHER = "lightgray"

    fov_max_x = df_fov['x'].max() if df_fov['x'].max() > 0 else 1024.0
    fov_max_y = df_fov['y'].max() if df_fov['y'].max() > 0 else 1024.0
    ax.set_xlim(0, fov_max_x)
    ax.set_ylim(fov_max_y, 0)
    ax.set_aspect('equal', adjustable='box')

    def assign_category(row_type):
        if row_type in antecedents: return 2 
        elif row_type in consequents: return 2 
        else: return 0 

    df_fov = df_fov.copy()
    df_fov['plot_order'] = df_fov['cell type'].apply(assign_category)
    df_fov = df_fov.sort_values('plot_order')

    mask_other = df_fov['plot_order'] == 0
    if mask_other.any():
        other_cells = df_fov[mask_other]
        ax.scatter(other_cells['x'], other_cells['y'], c=COLOR_OTHER, s=constant_size, alpha=0.2, label='Other')
    
    mask_active = df_fov['plot_order'] == 2
    rule_types = []
    if mask_active.any():
        active_cells = df_fov[mask_active]
        for cell_type, group in active_cells.groupby('cell type'):
            ax.scatter(
                group['x'], group['y'], 
                c=[color_map.get(cell_type, 'black')], 
                s=constant_size, alpha=0.9, edgecolors='none', label=cell_type
            )
            rule_types.append(cell_type)

    ax.set_title("Rule Interaction Highlight", fontsize=10)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    
    from matplotlib.ticker import MultipleLocator
    fov_size_um = 800.0 if fov_max_x > 1500 else 400.0
    radius_um = constants.CONFIG.get('RADIUS', 25)
    pixel_per_um = fov_max_x / fov_size_um
    grid_spacing = radius_um * pixel_per_um
    
    ax.xaxis.set_minor_locator(MultipleLocator(grid_spacing))
    ax.yaxis.set_minor_locator(MultipleLocator(grid_spacing))
    ax.grid(which='minor', color='gray', linestyle=':', linewidth=0.5, alpha=0.5)
    
    if rule_types:
        handles, labels = ax.get_legend_handles_labels()
        filtered_handles = []
        filtered_labels = []
        for h, l in zip(handles, labels):
            if l != 'Other':
                if hasattr(h, 'set_sizes'): h.set_sizes([40])
                filtered_handles.append(h)
                filtered_labels.append(l)
        ax.legend(filtered_handles, filtered_labels, bbox_to_anchor=(1.15, 1), loc='upper left', fontsize='small', title="Rule Types")

def _save_individual_plot(method_name, idx, df_fov, ants, cons, color_map, title_str, show_tissue_background=False):
    """Saves a separate plot for a single rule (for README/Reports)."""
    readme_dir = os.path.join(constants.RESULTS_PLOTS_DIR, 'readme_assets')
    os.makedirs(readme_dir, exist_ok=True)
    
    fig_single, axes_single = plt.subplots(1, 2, figsize=(24, 8))  # Increased from (18, 6)
    
    constant_size = plot_fov_full(axes_single[0], fig_single, df_fov, color_map, show_tissue_background)
    plot_fov_rule_highlight(axes_single[1], fig_single, df_fov, ants, cons, color_map, constant_size)
    
    fig_single.suptitle(title_str, fontsize=14, fontweight='bold', y=0.98)  # Increased font from 12 to 14
    
    single_out_path = os.path.join(readme_dir, f"{method_name}_rule_{idx+1}.png")
    plt.savefig(single_out_path, dpi=150, bbox_inches='tight')
    plt.close(fig_single)
    print(f"   -> Saved individual asset: {single_out_path}")

def visualize_method(method_name, cell_df, top_n, exclude_self_loops=False, exclude_shared_items=False, show_tissue_background=False):
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

    # 1. Filter: Exclude Shared Items (Stricter)
    if exclude_shared_items:
        # Check intersection: if any item is in both antecedent and consequent
        def has_overlap(row):
            ants = parse_rule_list(row['Antecedents'])
            cons = parse_rule_list(row['Consequents'])
            return not set(ants).isdisjoint(set(cons))
        
        rules_df = rules_df[~rules_df.apply(has_overlap, axis=1)]
        
    # 2. Filter: Exclude Exact Self Loops (Simpler)
    # Only run if strict filter wasn't applied (since strict covers simple)
    elif exclude_self_loops:
        def is_self_rule(row):
            ants = parse_rule_list(row['Antecedents'])
            cons = parse_rule_list(row['Consequents'])
            return set(ants) == set(cons)
        
        rules_df = rules_df[~rules_df.apply(is_self_rule, axis=1)]

    # 3. Filter by Significance (FDR or P-Value < 0.05)
    if 'FDR' in rules_df.columns:
        rules_df = rules_df[rules_df['FDR'] < 0.05]
    elif 'P_Value' in rules_df.columns:
        rules_df = rules_df[rules_df['P_Value'] < 0.05]

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
    ROW_HEIGHT = 8  # Increased from 6 to 8
    total_fig_height = ROW_HEIGHT * top_n
    fig, axes = plt.subplots(nrows=top_n, ncols=2, figsize=(24, total_fig_height))  # Increased width from 18 to 24
    if top_n == 1: axes = np.array([axes])

    plt.subplots_adjust(top=0.92, hspace=0.4, wspace=0.15)  # Reduced hspace from 0.8 to 0.4, wspace from 0.3 to 0.15
    
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

        df_fov = cell_df[cell_df['fov'] == fov_id].copy() # Copy to avoid SettingWithCopyWarning
        
        ax_full = axes[idx][0]
        ax_rule = axes[idx][1]

        if df_fov.empty:
            ax_full.text(0.5, 0.5, f"FOV {fov_id} not found", ha='center')
            ax_full.axis('off')
            ax_rule.axis('off')
            continue

        # Ensure x and y columns exist (sometimes it's centroid_x, centroid_y depending on load_cell_data)
        if 'x' not in df_fov.columns and 'centroid_x' in df_fov.columns:
            df_fov['x'] = df_fov['centroid_x']
            df_fov['y'] = df_fov['centroid_y']

        # Create FOV-specific color map
        fov_cell_types = df_fov['cell type'].unique()
        color_map = get_color_map(fov_cell_types)

        # Plotting
        constant_size = plot_fov_full(ax_full, fig, df_fov, color_map, show_tissue_background)
        plot_fov_rule_highlight(ax_rule, fig, df_fov, ants, cons, color_map, constant_size)
        
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

        # Save individual plots for top 3 rules of specific methods
        if method_name in ["BAG", "KNN_R"] and idx < 3:
            _save_individual_plot(method_name, idx, df_fov, ants, cons, color_map, title_str, show_tissue_background)

    # Determine Output Filename
    suffix = ""
    if show_tissue_background:
        suffix += "_with_tissue"
    if exclude_shared_items:
        suffix += "_no_shared_items"
    elif exclude_self_loops:
        suffix += "_no_self_loops"

    out_path = os.path.join(OUTPUT_DIR, f"{method_name}_top_rules{suffix}.png")
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved visualization to {out_path}")

def main():
    parser = argparse.ArgumentParser(description="Visualize Top Spatial Rules")
    parser.add_argument("--top_n", type=int, default=3, help="Number of top rules to visualize per method")
    parser.add_argument("--exclude_self_loops", action="store_true", help="Filter out rules where antecedent set EQUALS consequent set.")
    parser.add_argument("--exclude_shared_items", action="store_true", help="Filter out rules where ANY cell type appears in both antecedent and consequent.")
    parser.add_argument("--show_tissue_background", action="store_true", help="Show tissue compartment backgrounds (convex hulls)")
    args = parser.parse_args()

    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR, exist_ok=True)

    if not os.path.exists(CELL_TABLE_PATH):
        print(f"Error: Cell table not found at {CELL_TABLE_PATH}")
        return

    cell_df = load_cell_data(CELL_TABLE_PATH)

    for method in METHODS:
        visualize_method(method, cell_df, 
                         top_n=args.top_n, 
                         exclude_self_loops=args.exclude_self_loops,
                         exclude_shared_items=args.exclude_shared_items,
                         show_tissue_background=args.show_tissue_background)

if __name__ == "__main__":
    main()