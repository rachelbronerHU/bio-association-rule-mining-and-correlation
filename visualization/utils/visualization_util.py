import os
import sys
import pandas as pd
import ast
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import colorsys
from scipy.spatial import cKDTree, ConvexHull
from matplotlib.patches import Polygon

# Ensure project root imports are stable when scripts are run directly
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)
UTILS_DIR = os.path.join(REPO_ROOT, "utils")
if UTILS_DIR not in sys.path:
    sys.path.insert(0, UTILS_DIR)
import constants
from rules import _extract_base_lineage

# Shared spatial/tissue schema
TISSUE_BOOLEAN_COLS = [
    'in_CryptVilli', 'in_BrunnerGland', 'in_SMV', 'in_Muscle',
    'in_LP', 'in_Submucosa', 'in_Follicle', 'in_Lumen'
]

TISSUE_COLOR_MAP = {
    'in_CryptVilli': '#e1f5fe',
    'in_BrunnerGland': '#fff3e0',
    'in_SMV': '#e3f2fd',
    'in_Muscle': '#ffebee',
    'in_LP': '#e8f5e9',
    'in_Submucosa': '#f3e5f5',
    'in_Follicle': '#fff3e0',
    'in_Lumen': '#eceff1'
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

STAGE_LABELS = {
    0: "Control",
    1: "Stage 1",
    2: "Stage 2",
    3: "Stage 3",
    4: "Stage 4",
}

# ==============================================================================
# SECTION 1: DATA & RULES (Wrangling & Logic)
# ==============================================================================

def load_shifted_biopsy_metadata(mibi_gut_dir_path):
    """
    Standardized loader for biopsy metadata.
    1. Loads biopsy_metadata.csv and fovs_metadata.csv.
    2. Identifies Control biopsies (those in fovs_metadata but not biopsy_metadata).
    3. Shifts numeric stages by +1 and assigns Stage 0 to Controls.
    """
    biopsy_path = os.path.join(mibi_gut_dir_path, "biopsy_metadata.csv")
    fovs_path = os.path.join(mibi_gut_dir_path, "fovs_metadata.csv")
    
    if not os.path.exists(biopsy_path) or not os.path.exists(fovs_path):
        return None

    df_biopsy = pd.read_csv(biopsy_path)
    df_fovs = pd.read_csv(fovs_path)
    
    # Filter out secondary/special FOVs (starts with S_)
    df_fovs = df_fovs[~df_fovs["FOV"].astype(str).str.startswith("S_")]
    
    # Get all unique biopsies from FOV list
    unique_biopsies = df_fovs[["Patient", "Cohort"]].drop_duplicates().rename(
        columns={"Patient": "Biopsy_ID"}
    )
    
    # Targets that should be shifted to allow 0 for Control
    targets = [
        "Pathological stage", "GI stage", "liver stage", "skin stage", 
        "Grade GVHD", "Clinical score", "Pathological score"
    ]
    
    numeric_targets = []
    categorical_targets = []
    
    for col in targets:
        if col in df_biopsy.columns:
            if pd.api.types.is_numeric_dtype(df_biopsy[col]):
                df_biopsy[col] = df_biopsy[col] + 1
                numeric_targets.append(col)
            else:
                categorical_targets.append(col)
                
    # Merge all biopsies with available metadata
    df = pd.merge(unique_biopsies, df_biopsy, on="Biopsy_ID", how="left")
    
    # Fill NaNs for Controls
    for col in numeric_targets:
        df[col] = df[col].fillna(0)
    for col in categorical_targets:
        df[col] = df[col].fillna("Control")
        
    # Derive Organ and Control Flag
    df["Is_Control"] = df["Cohort"].apply(lambda x: "GVHD" not in str(x))
    
    def get_organ(row):
        if pd.notna(row.get("Localization")):
            return row["Localization"]
        cohort = str(row.get("Cohort", ""))
        if "Colon" in cohort: return "Colon"
        if "Duodenum" in cohort: return "Duodenum"
        return "Unknown"
        
    df["Organ"] = df.apply(get_organ, axis=1)
    
    return df

def merge_biopsy_metadata(df, mibi_gut_dir_path):
    """
    Merges a dataframe with standardized, shifted biopsy metadata.
    Drops existing staging columns in df to ensure "Source of Truth" from metadata.
    """
    strat_df = load_shifted_biopsy_metadata(mibi_gut_dir_path)
    if strat_df is None:
        return df
        
    # Drop existing overlapping columns (except Biopsy_ID) to avoid .x / .y suffixes
    cols_to_drop = [c for c in strat_df.columns if c in df.columns and c != 'Biopsy_ID']
    df_clean = df.drop(columns=cols_to_drop)
    
    return pd.merge(df_clean, strat_df, on='Biopsy_ID', how='left')

def add_organ_column(df, mibi_gut_dir_path):
    """Standardized helper to add an 'Organ' column to a dataframe."""
    df_merged = merge_biopsy_metadata(df, mibi_gut_dir_path)
    return df_merged

def filter_no_self_rules(df):
    """Removes rules where Antecedents and Consequents have overlapping items."""
    def clean_item(item):
        return item.replace("_CENTER", "").replace("_NEIGHBOR", "")

    def has_overlap(row):
        try:
            ant = parse_rule_list(row["Antecedents"])
            con = parse_rule_list(row["Consequents"])
            ant_clean = {clean_item(x) for x in ant}
            con_clean = {clean_item(x) for x in con}
            return not ant_clean.isdisjoint(con_clean)
        except Exception:
            return False 

    if 'Antecedents' not in df.columns or 'Consequents' not in df.columns:
        return df

    mask = df.apply(has_overlap, axis=1)
    return df[~mask].copy()

def parse_rule_list(rule_str):
    """Parses string representation of list into a python list and cleans suffixes."""
    def clean_item(item_str):
        return item_str.replace('_CENTER', '').replace('_NEIGHBOR', '')
    try:
        if isinstance(rule_str, list):
            items = rule_str
        else:
            items = ast.literal_eval(rule_str)
        return [clean_item(i) for i in items]
    except (ValueError, SyntaxError, TypeError):
        return [clean_item(str(rule_str).strip("[]'\""))]

def format_rule_for_display(antecedents, consequents):
    """Returns a clean display string for a rule title."""
    ant_items = parse_rule_list(antecedents)
    con_items = parse_rule_list(consequents)
    ant_display = "[" + ", ".join(ant_items) + "]"
    con_display = "[" + ", ".join(con_items) + "]"
    return f"{ant_display} -> {con_display}"

def normalize_cell_table(cell_df, ensure_tissue_columns=False):
    """
    Normalizes cell-table schema to shared names used across visualizations.
    Output guarantees: cell_type, x, y (when source columns exist).
    """
    df = cell_df.copy()
    rename_map = {}
    if 'cell type' in df.columns:
        rename_map['cell type'] = 'cell_type'
    if 'centroid_x' in df.columns:
        rename_map['centroid_x'] = 'x'
    if 'centroid_y' in df.columns:
        rename_map['centroid_y'] = 'y'
    if rename_map:
        df = df.rename(columns=rename_map)

    if ensure_tissue_columns:
        for col in TISSUE_BOOLEAN_COLS:
            if col not in df.columns:
                df[col] = False
            elif df[col].dtype == 'object':
                df[col] = df[col].map({'TRUE': True, 'FALSE': False, True: True, False: False}).fillna(False)
            else:
                df[col] = df[col].fillna(False)
    return df

def get_subtype_mask(df, subtype_str):
    """
    Vectorized mask to identify functional subtypes based on constants.py thresholds.
    Handles both base types ('CD8T') and functional subtypes ('CD8T_GZMB+').
    """
    subtype_str = str(subtype_str).replace("_CENTER", "").replace("_NEIGHBOR", "")
    base = _extract_base_lineage(subtype_str)
    mask = (df['cell_type'] == base)

    # Marker-aware subtype only when explicitly encoded as "<base>_<marker1+>_<marker2+>..."
    marker_tokens = [part for part in subtype_str.split('_') if part.endswith('+')]
    for marker_plus in marker_tokens:
        marker = marker_plus.rstrip('+')
        if marker not in df.columns:
            return pd.Series(False, index=df.index)
        threshold = constants.CELLTYPE_MARKER_THRESHOLDS.get(base, {}).get(marker, 0)
        mask &= (df[marker] > threshold)
    return mask

def get_rule_cells_mask(df, rule_items):
    """Subtype-aware boolean mask for cells participating in any rule item."""
    return get_rule_highlight_labels(df, rule_items).notna()

def get_rule_highlight_labels(df, rule_items):
    """
    Subtype-aware highlight labels per cell for rule rendering.
    More-specific items (marker subtypes) are assigned before base lineages.
    """
    labels = pd.Series(pd.NA, index=df.index, dtype="object")
    if not rule_items:
        return labels

    normalized = [str(i).replace("_CENTER", "").replace("_NEIGHBOR", "") for i in rule_items]
    unique_items = sorted(set(normalized), key=lambda x: ("+" not in x, -len(x), x))

    for item in unique_items:
        mask = get_subtype_mask(df, item)
        labels.loc[mask & labels.isna()] = item
    return labels

# ==============================================================================
# SECTION 2: SPATIAL BIOLOGY
# ==============================================================================

def calculate_distance_to_muscle(cell_df):
    """Calculates min distance to Muscle for each cell per FOV."""
    normalized_df = normalize_cell_table(cell_df)
    results = []
    for fov, group in normalized_df.groupby('fov'):
        muscle_cells = group[group['cell_type'] == 'Muscle']
        if muscle_cells.empty: continue

        muscle_coords = muscle_cells[['x', 'y']].values
        tree = cKDTree(muscle_coords)
        all_coords = group[['x', 'y']].values
        distances, _ = tree.query(all_coords)

        group = group.copy()
        group['distance_to_muscle'] = distances
        results.append(group)
    return pd.concat(results) if results else normalized_df

def select_representative_fov(df, rule_id, group_col, group_val):
    """Picks FOV closest to the 90th percentile of Rule_Count_Global for a group."""
    group_df = df[(df['Rule_ID'] == rule_id) & (df[group_col] == group_val)]
    if group_df.empty: return None
    p90 = np.percentile(group_df['Rule_Count_Global'], 90)
    idx = (group_df['Rule_Count_Global'] - p90).abs().idxmin()
    return group_df.loc[idx, 'FOV']

def get_sorted_stage_values(df, stage_col):
    """Returns sorted numeric stage values present in the dataframe."""
    if stage_col not in df.columns:
        return []
    numeric = pd.to_numeric(df[stage_col], errors='coerce').dropna()
    if numeric.empty:
        return []
    return sorted(numeric.astype(int).unique().tolist())

def get_baseline_stage(stage_values):
    """Returns the smallest stage value to use as baseline."""
    if not stage_values:
        return None
    return min(stage_values)

# ==============================================================================
# SECTION 3: VISUAL IDENTITY
# ==============================================================================

def get_stage_palette(n_stages=5):
    """Returns the standardized 'flare' palette for disease stages (0-4)."""
    return sns.color_palette("flare", n_colors=n_stages)

def get_stage_label(stage_value):
    """Returns canonical display label for a stage value."""
    try:
        key = int(stage_value)
        return STAGE_LABELS.get(key, f"Stage {key}")
    except Exception:
        return str(stage_value)

def get_cell_type_palette(cell_types):
    """Returns a consistent 'husl' palette for cell types."""
    palette = sns.color_palette("husl", n_colors=len(cell_types))
    return dict(zip(sorted(cell_types), palette))

def _adjust_lightness(rgb_color, factor):
    h, l, s = colorsys.rgb_to_hls(*rgb_color)
    l = max(0.2, min(0.85, l * factor))
    return colorsys.hls_to_rgb(h, l, s)

def get_rule_item_palette(rule_labels):
    """
    Palette for rule-highlight labels.
    Labels from the same lineage get close shades from the same color family.
    """
    labels = [str(x) for x in rule_labels if pd.notna(x)]
    if not labels:
        return {}

    lineage_to_labels = {}
    for lbl in sorted(set(labels)):
        lineage = _extract_base_lineage(lbl)
        lineage_to_labels.setdefault(lineage, []).append(lbl)

    base_lineages = sorted(lineage_to_labels.keys())
    base_colors = sns.color_palette("tab20", n_colors=max(1, len(base_lineages)))

    palette = {}
    for idx, lineage in enumerate(base_lineages):
        members = sorted(lineage_to_labels[lineage], key=lambda x: (x != lineage, x))
        base_rgb = base_colors[idx]
        if len(members) == 1:
            palette[members[0]] = base_rgb
            continue
        factors = np.linspace(0.85, 1.15, len(members))
        for m, f in zip(members, factors):
            palette[m] = _adjust_lightness(base_rgb, float(f))
    return palette

# ==============================================================================
# SECTION 4: CANVAS HELPERS
# ==============================================================================

def get_constant_scatter_size(ax, fig, cell_diam_um=10.0, fov_size_um=400.0, fov_size_px=1024.0):
    """Calculates Matplotlib 's' for constant biological diameter."""
    fig.canvas.draw()
    bbox = ax.get_window_extent()
    ax_height_display_px = bbox.height
    px_per_um = fov_size_px / fov_size_um
    cell_diam_px = cell_diam_um * px_per_um
    y_min, y_max = ax.get_ylim()
    data_height = abs(y_max - y_min)
    if data_height == 0: data_height = fov_size_px
    points_per_px = 72.0 / fig.dpi
    cell_diam_pts = (cell_diam_px * (ax_height_display_px / data_height)) * points_per_px
    return cell_diam_pts ** 2

def plot_tissue_backgrounds(ax, df_fov, alpha=0.15):
    """Plots tissue compartment backgrounds as convex hulls."""
    for col in TISSUE_BOOLEAN_COLS:
        if col not in df_fov.columns: continue
        cells = df_fov[df_fov[col] == True]
        if len(cells) < 3: continue
        coords = cells[['x', 'y']].values
        try:
            hull = ConvexHull(coords)
            ax.add_patch(Polygon(coords[hull.vertices], facecolor=TISSUE_COLOR_MAP.get(col, 'gray'), alpha=alpha, zorder=0))
        except: continue

def format_spatial_axis(ax, title=None):
    """Standard axis cleaning for spatial maps."""
    ax.set_aspect('equal')
    ax.set_xticks([]); ax.set_yticks([])
    if title: ax.set_title(title)
    for spine in ax.spines.values(): spine.set_visible(False)

def plot_fov_cells(
    ax,
    fig,
    df_fov,
    color_map,
    color_labels=None,
    highlighted_mask=None,
    highlighted_labels=None,
    highlighted_color_map=None,
    constant_size=None,
    show_grid=True
):
    """
    Shared FOV scatter renderer.
    - highlighted_mask=None: color all cells by color_labels/cell_type.
    - highlighted_mask provided: gray non-highlighted cells and color highlighted cells.
    - highlighted_labels can provide subtype-aware labels for highlighted cells.
    """
    fov_max_x = df_fov['x'].max() if df_fov['x'].max() > 0 else 1024.0
    fov_max_y = df_fov['y'].max() if df_fov['y'].max() > 0 else 1024.0
    ax.set_xlim(0, fov_max_x)
    ax.set_ylim(fov_max_y, 0)
    ax.set_aspect('equal', adjustable='box')

    if constant_size is None:
        fov_size_um = 800.0 if fov_max_x > 1500 else 400.0
        constant_size = get_constant_scatter_size(
            ax, fig, cell_diam_um=10.0, fov_size_um=fov_size_um, fov_size_px=fov_max_x
        )

    if highlighted_mask is None:
        if color_labels is not None:
            color_labels = color_labels.reindex(df_fov.index).astype("object")
            draw_df = df_fov.copy()
            draw_df["__color_label"] = color_labels.fillna(draw_df["cell_type"])
        else:
            draw_df = df_fov.copy()
            draw_df["__color_label"] = draw_df["cell_type"]

        for label, group in draw_df.groupby('__color_label'):
            ax.scatter(
                group['x'], group['y'],
                c=[color_map.get(label, 'gray')],
                s=constant_size, alpha=1.0, edgecolors='none', zorder=2, label=label
            )
        highlighted_types = sorted(draw_df['__color_label'].dropna().unique().tolist())
    else:
        highlighted_mask = highlighted_mask.reindex(df_fov.index, fill_value=False)
        other_cells = df_fov[~highlighted_mask]
        if not other_cells.empty:
            ax.scatter(
                other_cells['x'], other_cells['y'],
                c='lightgray', s=constant_size, alpha=0.2, edgecolors='none', zorder=1, label='Other'
            )

        active_cells = df_fov[highlighted_mask].copy()
        highlighted_types = []
        if highlighted_labels is not None:
            active_cells['__rule_label'] = highlighted_labels.reindex(active_cells.index)
            active_cells = active_cells[active_cells['__rule_label'].notna()]
            cmap = highlighted_color_map or {}
            for lbl, group in active_cells.groupby('__rule_label'):
                ax.scatter(
                    group['x'], group['y'],
                    c=[cmap.get(lbl, 'black')],
                    s=constant_size, alpha=0.9, edgecolors='none', zorder=3, label=lbl
                )
                highlighted_types.append(lbl)
        else:
            for cell_type, group in active_cells.groupby('cell_type'):
                ax.scatter(
                    group['x'], group['y'],
                    c=[color_map.get(cell_type, 'black')],
                    s=constant_size, alpha=0.9, edgecolors='none', zorder=3, label=cell_type
                )
                highlighted_types.append(cell_type)

    if show_grid:
        from matplotlib.ticker import MultipleLocator
        fov_size_um = 800.0 if fov_max_x > 1500 else 400.0
        radius_um = constants.CONFIG.get('RADIUS', 25)
        pixel_per_um = fov_max_x / fov_size_um
        grid_spacing = radius_um * pixel_per_um
        ax.xaxis.set_minor_locator(MultipleLocator(grid_spacing))
        ax.yaxis.set_minor_locator(MultipleLocator(grid_spacing))
        ax.grid(which='minor', color='gray', linestyle=':', linewidth=0.5, alpha=0.5)

    return constant_size, highlighted_types
