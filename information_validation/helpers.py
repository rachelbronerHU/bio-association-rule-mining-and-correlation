"""Helper functions for the information-validation pipeline (notebook 13).

All the underlying machinery -- null models, information theory, rule selection,
the simulated-annealing engine, scoring and plots -- lives here so the notebook
stays a short, readable flow. Import with:

    import information_validation.helpers as H
    from information_validation.helpers import *
    H.GRID_SIZE = 150.0            # share the macro/micro bin size

Module state read inside the helpers (set from the notebook / loaders):
    GRID_SIZE       macro/micro bin size (um)
    CELL_COLOR_MAP  cell_type -> color, populated by make_cell_palette
"""
import os
import ast
import math
import random
import textwrap
import itertools
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
from matplotlib.ticker import MultipleLocator
from sklearn.neighbors import NearestNeighbors

warnings.filterwarnings('ignore')

# ---- module state (set by the notebook config / loaders) ----
GRID_SIZE = 150.0                 # macro/micro bin size; notebook sets H.GRID_SIZE
CELL_COLOR_MAP = {}               # cell_type -> color; filled by make_cell_palette
OTHER_COLOR = (0.5, 0.5, 0.5)

# Which mining algorithm's neighbor logic the simulation reproduces:
#   'fpgrowth'          -> binary presence (a neighbor type is there or not)
#   'weighted_fpgrowth' -> min(sum of Gaussian distance weights, 1.0), bandwidth BANDWIDTH,
#                          exactly matching algos/weighted_fpgrowth.py.
# Binary is the special case where every edge weight is 1.0, so both share one code path.
ALGO = "weighted_fpgrowth"
BANDWIDTH = 15.0                  # Gaussian bandwidth (um) for 'weighted_fpgrowth'



# ==========================================================================
# Data loading & tissue setup
# ==========================================================================

def get_data_dir():
    current_dir = os.path.abspath(os.getcwd())
    while current_dir != os.path.dirname(current_dir):
        potential_data = os.path.join(current_dir, 'data', 'MIBIGutCsv')
        if os.path.exists(potential_data):
            return potential_data
        current_dir = os.path.dirname(current_dir)
    return '../../data/MIBIGutCsv'

def load_tissue(data_dir):
    """Load cell + FOV tables; add micron coords (x_um, y_um) and a clean cell_type."""
    df_cells = pd.read_csv(os.path.join(data_dir, 'cell_table.csv'))
    df_fovs  = pd.read_csv(os.path.join(data_dir, 'fovs_metadata.csv'))
    fov_to_size = df_fovs.set_index('FOV')['Size [um]'].to_dict()
    def to_um(row):                                    # pixels -> microns, per-FOV scale
        size = fov_to_size.get(row['fov'], 800)
        res  = 1024 if size == 400 else 2048
        return row['centroid_x'] * (size / res), row['centroid_y'] * (size / res)
    df_cells[['x_um', 'y_um']] = df_cells.apply(lambda r: pd.Series(to_um(r)), axis=1)
    if 'cell type' in df_cells.columns:                # normalize the column name
        df_cells['cell_type'] = df_cells['cell type']
    else:
        df_cells['cell type'] = df_cells['cell_type']
    return df_cells, df_fovs

def make_cell_palette(df_cells):
    """Build the shared cell_type -> color map used by every tissue plot (module state)."""
    global CELL_COLOR_MAP, OTHER_COLOR
    types = sorted(df_cells['cell type'].dropna().astype(str).unique())
    try:
        pal = plt.colormaps.get_cmap('tab20b').resampled(max(len(types), 1))
    except AttributeError:
        pal = plt.cm.get_cmap('tab20b', max(len(types), 1))
    CELL_COLOR_MAP = {ct: pal(i) for i, ct in enumerate(types)}
    OTHER_COLOR = (0.5, 0.5, 0.5)
    return CELL_COLOR_MAP

def locate_results_dir(result_folder_name):
    """Find the results/.../data folder that holds the mined rule CSVs."""
    d = os.path.join(os.path.dirname(os.path.dirname(get_data_dir())),
                     'results', 'full_run', result_folder_name, 'data')
    if not os.path.exists(d):
        d = f'../../results/full_run/{result_folder_name}/data'
    if not os.path.exists(d):                           # last resort: walk and find it
        for root, _, files in os.walk('../../results/'):
            if 'results_CN.csv' in files and 'weighted_fpgrowth' in root:
                d = root; break
    return d

def load_rule_sets(results_dir):
    """Load the unfiltered pool ('full') and the significance-filtered set ('full filtered')."""
    def _load(name, label):
        p = os.path.join(results_dir, name)
        if os.path.exists(p):
            df = pd.read_csv(p)
            print(f"Loaded {len(df):,} rules from {name}  ({label}).")
            return df
        print(f"File not found: {p}")
        return pd.DataFrame()
    results_orig        = _load('results_CN.csv',          "UNFILTERED pool -> 'full'")
    results_sigfiltered = _load('results_CN_filtered.csv', "significance-filtered -> 'full filtered'")
    return results_orig, results_sigfiltered



# ==========================================================================
# Neighborhoods & patch states
# ==========================================================================

def precompute_fov_neighbors(df_fov, radius=30.0):
    """Builds a spatial neighborhood graph for a single FOV exactly once."""
    coords = df_fov[['x_um', 'y_um']].values
    nn = NearestNeighbors(radius=radius)
    nn.fit(coords)

    # Get neighbors for ALL cells in the FOV at once
    all_neighbors_indices = nn.radius_neighbors(coords, return_distance=False, sort_results=False)

    neighborhood_graph = {}

    for i, cell_idx in enumerate(df_fov.index):
        # Convert integer positions back to actual dataframe indices
        neighbor_idx = df_fov.index[all_neighbors_indices[i]]

        # A cell is always within its own search radius (distance 0), so this
        # lookup lists it as its own neighbor. The real mining pipeline that
        # produced our rules explicitly excludes this (see algos/fpgrowth.py
        # and algos/weighted_fpgrowth.py), so we match that here: a same-type
        # rule (e.g. "Muscle center -> Muscle neighbor") should require a
        # genuinely different nearby cell, not just the center counting itself.
        neighbor_idx = neighbor_idx[neighbor_idx != cell_idx]

        neighborhood_graph[cell_idx] = neighbor_idx

    return neighborhood_graph

def precompute_edge_weights(df_fov, neighborhood_graph):
    """Per-edge Gaussian weights exp(-0.5*(d/BANDWIDTH)^2), aligned with neighborhood_graph[c].

    Matches the mining transaction weights in algos/weighted_fpgrowth.py. Returns None for
    ALGO='fpgrowth' (binary presence == every edge weight 1.0). Assumes df_fov is reset_index'd,
    so neighbor indices are row positions into the coordinate array (as precompute_fov_neighbors
    produces for the notebook's df_fov)."""
    if ALGO == "fpgrowth":
        return None
    coords = df_fov[['x_um', 'y_um']].values
    edge_weights = {}
    for c, neigh in neighborhood_graph.items():
        if len(neigh) == 0:
            edge_weights[c] = np.array([])
            continue
        d = np.linalg.norm(coords[neigh] - coords[c], axis=1)
        edge_weights[c] = np.exp(-0.5 * (d / BANDWIDTH) ** 2)
    return edge_weights

def extract_patch_states(df_fov, neighborhood_graph, center_type, neighbor_types):
    """Extracts the boolean state of specified neighbor cell types around each center cell,"""
    # Isolate only the center cells
    centers = df_fov[df_fov['cell_type'] == center_type]
    
    if centers.empty:
        return pd.DataFrame(columns=neighbor_types)
    
    states_list = []
    
    # Loop only over the center cells we care about
    for center_idx in centers.index:
        # O(1) lookup to get neighbor indices
        neighbor_indices = neighborhood_graph[center_idx]
        
        # Get the actual neighbor cells
        neighborhood_cells = df_fov.loc[neighbor_indices]
        
        present_types = set(neighborhood_cells['cell_type'].unique())
        state_vector = {ntype: int(ntype in present_types) for ntype in neighbor_types}
        states_list.append(state_vector)
        
    return pd.DataFrame(states_list)

def calculate_smoothed_probabilities(df_states, alpha=0.01):
    """Calculates the empirical probability distribution of states"""
    N = len(df_states)
    if N == 0:
        return {}
        
    num_neighbors = len(df_states.columns)
    
    # Calculate the size of the state space (|M|)
    # For k binary neighbor columns, there are exactly 2^k possible combinations
    M_size = 2 ** num_neighbors
    
    # Convert each row to a tuple (e.g., (1,0)) to make it hashable, 
    # then count exactly how many times each spatial motif appeared in the tissue
    state_counts = Counter(tuple(row) for row in df_states.itertuples(index=False, name=None))
    
    # Generate the full theoretical state space (including motifs never observed)
    # Cartesian product of [0, 1] repeated 'num_neighbors' times
    all_possible_states = list(itertools.product([0, 1], repeat=num_neighbors))
    
    # Denominator for Laplace Smoothing: N + alpha * |M|
    denominator = N + (alpha * M_size)
    p_dist = {}
    
    for state in all_possible_states:
        # Retrieve the empirical count, defaulting to 0 if the motif never appeared
        count_m = state_counts.get(state, 0)
        
        # Apply the smoothing formula to ensure strict positivity
        p_dist[state] = (count_m + alpha) / denominator
        
    return p_dist



# ==========================================================================
# Null models (P_global anatomy-free, P_macro anatomy-preserving)
# ==========================================================================

def create_macro_null_model(df_fov, grid_size=None):
    """Creates the P_macro null model by locally shuffling cell types within spatial grid bins."""
    grid_size = GRID_SIZE if grid_size is None else grid_size
    df_shuffled = df_fov.copy()
    
    # 1. Map continuous coordinates into discrete bins (e.g., 100um blocks)
    bin_x = (df_shuffled['x_um'] // grid_size).astype(int)
    bin_y = (df_shuffled['y_um'] // grid_size).astype(int)
    
    # 2. Create a unified string identifier for each spatial block (e.g., "3_5")
    df_shuffled['grid_bin'] = bin_x.astype(str) + "_" + bin_y.astype(str)
    
    # 3. Group by the spatial block and randomly shuffle ONLY the 'cell_type' column
    df_shuffled['cell_type'] = df_shuffled.groupby('grid_bin')['cell_type'].transform(np.random.permutation)
    
    # 4. Remove the temporary grouping column to perfectly match the original dataframe structure
    df_shuffled = df_shuffled.drop(columns=['grid_bin'])
    
    return df_shuffled

def create_global_null_model(df_fov):
    """Creates the P_global baseline by randomly shuffling ALL cell types"""
    df_shuffled = df_fov.copy()

    # Randomly reassign every cell type across the whole FOV
    df_shuffled['cell_type'] = np.random.permutation(df_shuffled['cell_type'].values)

    return df_shuffled

def create_stable_global_model(df_fov, neighborhood_graph, center_type, neighbor_types,
                                n_shuffles=50, alpha=0.01):
    """Creates a stable P_global by averaging the probability distribution"""
    all_distributions = []

    for _ in range(n_shuffles):
        df_shuffled = create_global_null_model(df_fov)
        df_states = extract_patch_states(df_shuffled, neighborhood_graph, center_type, neighbor_types)
        if df_states.empty:
            continue
        p_dist = calculate_smoothed_probabilities(df_states, alpha=alpha)
        all_distributions.append(p_dist)

    if not all_distributions:
        return {}

    all_states = all_distributions[0].keys()
    return {
        state: np.mean([dist[state] for dist in all_distributions])
        for state in all_states
    }

def create_stable_macro_model(df_fov, neighborhood_graph, center_type, neighbor_types,
                               n_shuffles=50, alpha=0.01):
    """Creates a stable P_macro by averaging the probability distribution"""
    all_distributions = []

    for _ in range(n_shuffles):
        # Create one locally shuffled version of the tissue
        df_shuffled = create_macro_null_model(df_fov)

        # Extract patch states from the shuffled tissue
        # (cell positions are unchanged ג€” only the type labels were shuffled)
        df_states = extract_patch_states(df_shuffled, neighborhood_graph, center_type, neighbor_types)

        if df_states.empty:
            continue

        p_dist = calculate_smoothed_probabilities(df_states, alpha=alpha)
        all_distributions.append(p_dist)

    if not all_distributions:
        return {}

    # Average the probability of each state across all N shuffles
    all_states = all_distributions[0].keys()
    p_macro_stable = {
        state: np.mean([dist[state] for dist in all_distributions])
        for state in all_states
    }

    return p_macro_stable



# ==========================================================================
# Information theory (KL, IPF, Information Gain)
# ==========================================================================

def calculate_kl_divergence(p_real, p_model):
    """Calculates the Kullback-Leibler Divergence D_KL(P_real || P_model) in bits."""
    kl_div = 0.0
    for state in p_real:
        # We only calculate for states that actually occur in reality
        if p_real[state] > 0 and p_model[state] > 0:
            kl_div += p_real[state] * math.log2(p_real[state] / p_model[state])
    
    return kl_div

def apply_ipf_for_rule(p_baseline, p_real, rule_states, max_iters=100, tol=1e-5):
    """Adjusts a baseline distribution (P_global for IG; P_macro for the macro-preservation"""
    # Start with the baseline distribution
    p_k = p_baseline.copy()
    
    # Calculate the exact target probability of this rule in reality
    target_prob = sum(p_real[state] for state in rule_states)
    
    for i in range(max_iters):
        # Calculate the current probability of the rule in the running model
        current_prob = sum(p_k[state] for state in rule_states)
        
        # Prevent division by zero if the rule is completely absent
        if current_prob == 0:
            break
            
        # The IPF update ratio: Target / Current
        ratio = target_prob / current_prob
        
        # Apply the update ONLY to the states that satisfy the rule
        for state in rule_states:
            p_k[state] *= ratio
            
        # Normalize the entire distribution so the sum is exactly 1.0 again
        total_sum = sum(p_k.values())
        for state in p_k:
            p_k[state] /= total_sum
            
        # Check for convergence: if the model matches reality, stop the loop
        new_prob = sum(p_k[state] for state in rule_states)
        if abs(new_prob - target_prob) < tol:
            break
            
    return p_k

def calculate_information_gain(p_real, p_baseline, p_k):
    """Calculates the Information Gain (IG) of a specific spatial rule."""
    # Step 1: Baseline Information Loss (Reality vs. baseline; P_global for IG)
    i_baseline = calculate_kl_divergence(p_real, p_baseline)
    
    # Step 2: Remaining Information Loss (Reality vs. IPF micro-rule model)
    i_loss = calculate_kl_divergence(p_real, p_k)
    
    # Step 3: Information Gain (Micro information successfully explained by the rule)
    ig_k = i_baseline - i_loss
    
    return ig_k

def build_integrated_model(p_baseline, p_real, list_of_rules, max_iters=100, tol=1e-5):
    """Builds the integrated model (P_integrated) by applying IPF to multiple rules simultaneously."""
    p_integrated = p_baseline.copy()
    
    # Pre-calculate the target empirical probabilities for all rules from reality
    target_probs = [sum(p_real[state] for state in rule_states) for rule_states in list_of_rules]
    
    for i in range(max_iters):
        max_diff = 0.0
        
        # Sequentially update the model for each rule in the list
        for rule_idx, rule_states in enumerate(list_of_rules):
            target_prob = target_probs[rule_idx]
            current_prob = sum(p_integrated[state] for state in rule_states)
            
            # Avoid division by zero if a rule's probability drops to absolute zero
            if current_prob == 0:
                continue
                
            # The IPF update ratio: Target / Current
            ratio = target_prob / current_prob
            
            # Apply the update only to the states that satisfy the current rule
            for state in rule_states:
                p_integrated[state] *= ratio
                
            # Normalize the entire distribution so the sum is exactly 1.0
            total_sum = sum(p_integrated.values())
            for state in p_integrated:
                p_integrated[state] /= total_sum
                
            # Track the largest deviation to check for global convergence
            new_prob = sum(p_integrated[state] for state in rule_states)
            diff = abs(new_prob - target_prob)
            if diff > max_diff:
                max_diff = diff
                
        # If the largest deviation across ALL rules is below the tolerance, we converged
        if max_diff < tol:
            break
            
    return p_integrated



# ==========================================================================
# Rule parsing & IG components (macro-aligned vs beyond-anatomy)
# ==========================================================================

def parse_rule_components(row):
    """Extracts the center type, neighbor types, and rule states from a single rule row."""
    antecedents = ast.literal_eval(row['Antecedents'])
    consequents  = ast.literal_eval(row['Consequents'])

    center_type    = None
    neighbor_types = []

    for item in antecedents + consequents:
        if '_CENTER' in item:
            center_type = item.replace('_CENTER', '')
        else:
            neighbor_types.append(item.replace('_NEIGHBOR', ''))

    # Sort for a consistent, reproducible ordering across all calls
    neighbor_types = sorted(set(neighbor_types))

    # The rule fires when ALL neighbor types are present — always exactly one state
    rule_states = [tuple(1 for _ in neighbor_types)]

    return center_type, neighbor_types, rule_states

def has_duplicate_items(ant_list, con_list):
    """True if the rule repeats a type within the SAME side (e.g. two identical"""
    ant_clean = [i.replace('_CENTER', '').replace('_NEIGHBOR', '') for i in ant_list]
    con_clean = [i.replace('_CENTER', '').replace('_NEIGHBOR', '') for i in con_list]
    if len(ant_clean) != len(set(ant_clean)):
        return True
    if len(con_clean) != len(set(con_clean)):
        return True
    return False

def is_same_type_rule(ant_list, con_list):
    """True if the rule's center type also appears among its neighbor types"""
    ant_clean = [i.replace('_CENTER', '').replace('_NEIGHBOR', '') for i in ant_list]
    con_clean = [i.replace('_CENTER', '').replace('_NEIGHBOR', '') for i in con_list]
    return not set(ant_clean).isdisjoint(set(con_clean))

def classify_tier(row):
    """Classify a rule into a tier by different/same-type x simple/complex."""
    _, neighbor_types, _ = parse_rule_components(row)
    is_simple = len(neighbor_types) == 1
    is_same_type = is_same_type_rule(
        ast.literal_eval(row['Antecedents']), ast.literal_eval(row['Consequents'])
    )
    if not is_same_type:
        return 1 if is_simple else 2
    return 3 if is_simple else 4

def compute_all_individual_igs(rules_df, df_fov, neighborhood_graph, n_shuffles=50, alpha=0.01):
    """Batch-computes individual IG for all rules, caching P_real and P_macro per group."""
    try:
        from tqdm import tqdm
        iterator = tqdm(rules_df.iterrows(), total=len(rules_df), desc='Computing IGs')
    except ImportError:
        iterator = rules_df.iterrows()

    group_cache = {}
    igs = []

    for _, row in iterator:
        center_type, neighbor_types, rule_states = parse_rule_components(row)
        group_key = (center_type, tuple(neighbor_types))

        if group_key not in group_cache:
            df_states = extract_patch_states(df_fov, neighborhood_graph, center_type, neighbor_types)
            if df_states.empty:
                group_cache[group_key] = (None, None)
            else:
                p_real  = calculate_smoothed_probabilities(df_states, alpha=alpha)
                p_global = create_stable_global_model(
                    df_fov, neighborhood_graph, center_type, neighbor_types,
                    n_shuffles=n_shuffles, alpha=alpha
                )
                group_cache[group_key] = (p_real, p_global)

        p_real, p_global = group_cache[group_key]
        if p_real is None or not p_global:
            igs.append(0.0)
            continue

        p_k = apply_ipf_for_rule(p_global, p_real, rule_states)
        igs.append(calculate_information_gain(p_real, p_global, p_k))

    return igs

def compute_rule_components(rules_df, df_fov, neighborhood_graph, n_shuffles=50, alpha=0.01):
    """Split each rule's IG into ig_micro (beyond anatomy) and ig_macro (anatomy-aligned)."""
    try:
        from tqdm import tqdm
        iterator = tqdm(rules_df.iterrows(), total=len(rules_df), desc='IG components')
    except ImportError:
        iterator = rules_df.iterrows()

    group_cache = {}
    ig_global_l, ig_micro_l, ig_macro_l = [], [], []

    for _, row in iterator:
        center_type, neighbor_types, rule_states = parse_rule_components(row)
        key = (center_type, tuple(neighbor_types))

        if key not in group_cache:
            df_states = extract_patch_states(df_fov, neighborhood_graph, center_type, neighbor_types)
            if df_states.empty:
                group_cache[key] = (None, None, None)
            else:
                p_real   = calculate_smoothed_probabilities(df_states, alpha=alpha)
                p_global = create_stable_global_model(
                    df_fov, neighborhood_graph, center_type, neighbor_types,
                    n_shuffles=n_shuffles, alpha=alpha)
                p_macro  = create_stable_macro_model(
                    df_fov, neighborhood_graph, center_type, neighbor_types,
                    n_shuffles=n_shuffles, alpha=alpha)
                group_cache[key] = (p_real, p_global, p_macro)

        p_real, p_global, p_macro = group_cache[key]
        if p_real is None or not p_global or not p_macro:
            ig_global_l.append(0.0); ig_micro_l.append(0.0); ig_macro_l.append(0.0)
            continue

        pk_g = apply_ipf_for_rule(p_global, p_real, rule_states)
        pk_m = apply_ipf_for_rule(p_macro,  p_real, rule_states)
        ig_g = calculate_information_gain(p_real, p_global, pk_g)
        ig_m = calculate_information_gain(p_real, p_macro,  pk_m)
        ig_global_l.append(ig_g)
        ig_micro_l.append(ig_m)
        ig_macro_l.append(ig_g - ig_m)

    return ig_global_l, ig_micro_l, ig_macro_l



# ==========================================================================
# Rule ranking & elbow selection
# ==========================================================================

def evaluate_cumulative_information(p_baseline, p_real, ranked_rules, step=1):
    """Evaluates the cumulative Information Gain as ranked rules are incrementally added."""
    x_values = []
    y_values = []

    # Calculate the baseline information loss once (Reality vs. baseline)
    baseline_loss = calculate_kl_divergence(p_real, p_baseline)

    total_rules = len(ranked_rules)

    # Incrementally add rules based on the defined step size
    for i in range(1, total_rules + 1, step):
        current_rules    = ranked_rules[:i]
        p_integrated     = build_integrated_model(p_baseline, p_real, current_rules)
        current_loss     = calculate_kl_divergence(p_real, p_integrated)
        cumulative_ig    = baseline_loss - current_loss
        x_values.append(i)
        y_values.append(cumulative_ig)

    # Ensure the final point is always included
    if len(x_values) == 0 or x_values[-1] != total_rules:
        p_integrated  = build_integrated_model(p_baseline, p_real, ranked_rules)
        current_loss  = calculate_kl_divergence(p_real, p_integrated)
        cumulative_ig = baseline_loss - current_loss
        x_values.append(total_rules)
        y_values.append(cumulative_ig)

    return x_values, y_values

def build_unified_elbow_curve(ranked_rules_df, df_fov, neighborhood_graph,
                               n_shuffles=50, alpha=0.01, log_steps=True):
    """Builds a unified cumulative Information Gain curve across all rules."""
    total_rules = len(ranked_rules_df)

    # Determine which steps to actually evaluate at
    if log_steps and total_rules > 50:
        indices  = np.unique(np.logspace(0, np.log10(total_rules), 80).astype(int) - 1)
        indices  = np.clip(indices, 0, total_rules - 1)
        eval_set = set(indices)
        eval_set.add(total_rules - 1)   # always include the last point
    else:
        eval_set = set(range(total_rules))

    # Per-group state: each group is one (center_type, neighbor_types) combination
    group_p_real        = {}   # group_key -> p_real distribution
    group_p_global       = {}   # group_key -> stable p_global distribution
    group_rules         = {}   # group_key -> list of rule_states added so far
    group_baseline_loss = {}   # group_key -> I_baseline, fixed once per group
    group_current_loss  = {}   # group_key -> current I_loss after latest IPF

    x_values = []
    y_values = []

    for step_idx, (_, row) in enumerate(ranked_rules_df.iterrows()):
        center_type, neighbor_types, rule_states = parse_rule_components(row)
        group_key = (center_type, tuple(neighbor_types))

        # --- Initialize this group the first time we encounter it ---
        if group_key not in group_p_real:
            df_states = extract_patch_states(df_fov, neighborhood_graph, center_type, neighbor_types)
            if df_states.empty:
                continue

            p_real  = calculate_smoothed_probabilities(df_states, alpha=alpha)
            p_global = create_stable_global_model(df_fov, neighborhood_graph, center_type,
                                                   neighbor_types, n_shuffles=n_shuffles, alpha=alpha)
            if not p_global:
                continue

            group_p_real[group_key]        = p_real
            group_p_global[group_key]       = p_global
            group_rules[group_key]         = []
            group_baseline_loss[group_key] = calculate_kl_divergence(p_real, p_global)
            group_current_loss[group_key]  = group_baseline_loss[group_key]

        # --- Add rule to its group and re-run IPF only for that group ---
        group_rules[group_key].append(rule_states)

        p_integrated = build_integrated_model(
            group_p_global[group_key],
            group_p_real[group_key],
            group_rules[group_key]
        )
        group_current_loss[group_key] = calculate_kl_divergence(group_p_real[group_key], p_integrated)

        # --- Record total IG across all groups at this checkpoint ---
        if step_idx in eval_set:
            total_ig = sum(
                group_baseline_loss[k] - group_current_loss[k]
                for k in group_baseline_loss
            )
            x_values.append(step_idx + 1)
            y_values.append(total_ig)

    total_baseline = sum(group_baseline_loss.values())
    return x_values, y_values, total_baseline, group_p_real

def build_greedy_mdl_curve(candidate_rules_df, df_fov, neighborhood_graph,
                           n_shuffles=50, alpha=0.01):
    """Order rules GREEDILY by marginal Information Gain, and return the cumulative"""
    # --- precompute per-group distributions once ---
    groups = {}                 # group_key -> dict(p_real, p_global, baseline_loss, current_loss, selected)
    rule_group = {}             # df index -> group_key
    rule_states_by_idx = {}     # df index -> rule_states

    for idx, row in candidate_rules_df.iterrows():
        center_type, neighbor_types, rule_states = parse_rule_components(row)
        gkey = (center_type, tuple(neighbor_types))
        rule_group[idx] = gkey
        rule_states_by_idx[idx] = rule_states
        if gkey not in groups:
            df_states = extract_patch_states(df_fov, neighborhood_graph, center_type, neighbor_types)
            if df_states.empty:
                groups[gkey] = None
                continue
            p_real = calculate_smoothed_probabilities(df_states, alpha=alpha)
            p_global = create_stable_global_model(df_fov, neighborhood_graph, center_type,
                                                  neighbor_types, n_shuffles=n_shuffles, alpha=alpha)
            if not p_global:
                groups[gkey] = None
                continue
            bl = calculate_kl_divergence(p_real, p_global)
            groups[gkey] = dict(p_real=p_real, p_global=p_global,
                                baseline_loss=bl, current_loss=bl, selected=[])

    pending = [idx for idx in candidate_rules_df.index if groups.get(rule_group[idx])]

    def marginal_gain(idx):
        g = groups[rule_group[idx]]
        p_int = build_integrated_model(g['p_global'], g['p_real'], g['selected'] + [rule_states_by_idx[idx]])
        return g['current_loss'] - calculate_kl_divergence(g['p_real'], p_int)

    gain_cache = {idx: marginal_gain(idx) for idx in pending}

    order, x_values, y_values = [], [], []
    cum_ig = 0.0
    while pending:
        best = max(pending, key=lambda idx: gain_cache[idx])
        g = groups[rule_group[best]]

        # commit the pick: add to its group, refresh that group's current loss
        g['selected'].append(rule_states_by_idx[best])
        p_int = build_integrated_model(g['p_global'], g['p_real'], g['selected'])
        g['current_loss'] = calculate_kl_divergence(g['p_real'], p_int)

        cum_ig += gain_cache[best]
        order.append(best)
        pending.remove(best)
        x_values.append(len(order))
        y_values.append(cum_ig)

        # only the group that changed needs its remaining gains recomputed
        for idx in pending:
            if rule_group[idx] == rule_group[best]:
                gain_cache[idx] = marginal_gain(idx)

    total_baseline = sum(g['baseline_loss'] for g in groups.values() if g)
    ordered_rules_df = candidate_rules_df.loc[order].reset_index(drop=True)
    return ordered_rules_df, x_values, y_values, total_baseline

def elbow_k(x_vals, y_vals):
    """Kneedle: rule count at max perpendicular distance from the first-to-last chord."""
    x = np.asarray(x_vals, float); y = np.asarray(y_vals, float)
    if len(x) < 3:
        return int(x[-1]) if len(x) else 0
    xn = (x - x.min()) / (x.max() - x.min() + 1e-9)
    yn = (y - y.min()) / (y.max() - y.min() + 1e-9)
    d = np.abs((yn[-1]-yn[0])*xn - (xn[-1]-xn[0])*yn + xn[-1]*yn[0] - yn[-1]*xn[0]) / \
        np.sqrt((yn[-1]-yn[0])**2 + (xn[-1]-xn[0])**2 + 1e-9)
    return int(x[int(np.argmax(d))])

def elbow_cut(df, col):
    """Elbow on the cumulative (positive) contribution of `col`, descending."""
    order = df[col].clip(lower=0).sort_values(ascending=False).values
    cum = np.cumsum(order)
    k = elbow_k(range(1, len(cum) + 1), cum) if len(cum) >= 3 else len(cum)
    return max(int(k), 1), cum



# ==========================================================================
# Macro-preservation test (Test 1)
# ==========================================================================

def compute_macro_preservation_test(golden_rules_df, df_fov, neighborhood_graph, df_global_null,
                                    n_shuffles=50, alpha=0.01):
    """Information metrics for a rule set, summed over its groups (Test 1 quantities, bits)."""
    groups = {}
    for _, row in golden_rules_df.iterrows():
        center_type, neighbor_types, rule_states = parse_rule_components(row)
        groups.setdefault((center_type, tuple(neighbor_types)), []).append(rule_states)

    m = dict(i_macro=0.0, i_macro_model=0.0, i_loss=0.0, i_baseline=0.0, i_loss_macro=0.0)
    for (center_type, ntup), list_of_rule_states in groups.items():
        neighbor_types = list(ntup)
        s_real = extract_patch_states(df_fov, neighborhood_graph, center_type, neighbor_types)
        if s_real.empty:
            continue
        p_real = calculate_smoothed_probabilities(s_real, alpha=alpha)
        p_macro = create_stable_macro_model(df_fov, neighborhood_graph, center_type, neighbor_types,
                                             n_shuffles=n_shuffles, alpha=alpha)
        if not p_macro:
            continue
        s_glob = extract_patch_states(df_global_null, neighborhood_graph, center_type, neighbor_types)
        if s_glob.empty:
            continue
        p_global = calculate_smoothed_probabilities(s_glob, alpha=alpha)
        p_k = build_integrated_model(p_macro, p_real, list_of_rule_states)

        m['i_macro']       += calculate_kl_divergence(p_macro, p_global)
        m['i_macro_model'] += calculate_kl_divergence(p_k, p_global)
        m['i_loss']        += calculate_kl_divergence(p_real, p_k)
        m['i_baseline']    += calculate_kl_divergence(p_real, p_global)
        m['i_loss_macro']  += calculate_kl_divergence(p_real, p_macro)
    return m



# ==========================================================================
# Simulated-annealing engine (synthetic tissue)
# ==========================================================================

def parse_rule_for_annealing(row):
    """Break a rule into center type, extra antecedent types, and consequent types."""
    antecedents = ast.literal_eval(row["Antecedents"])
    consequents = ast.literal_eval(row["Consequents"])

    center_type = None
    antecedent_extra_types = []
    for item in antecedents:
        if "_CENTER" in item:
            center_type = item.replace("_CENTER", "")
        else:
            antecedent_extra_types.append(item.replace("_NEIGHBOR", ""))

    consequent_types = [item.replace("_NEIGHBOR", "") for item in consequents]

    return center_type, antecedent_extra_types, consequent_types

def _type_weight(type_array, neighbor_ids, edge_w, t):
    """Weight of neighbor type t around one cell.
    fpgrowth: 1.0 if present else 0.0.  weighted_fpgrowth: min(sum of Gaussian weights, 1.0)."""
    if edge_w is None:                                  # binary presence
        return 1.0 if any(type_array[n] == t for n in neighbor_ids) else 0.0
    s = 0.0                                             # capped Gaussian sum
    for n, w in zip(neighbor_ids, edge_w):
        if type_array[n] == t:
            s += w
    return s if s < 1.0 else 1.0

def _min_type_weight(type_array, neighbor_ids, edge_w, types):
    """min over the given neighbor types (1.0 if none) -- the min-based support of that set."""
    if len(types) == 0:
        return 1.0
    return min(_type_weight(type_array, neighbor_ids, edge_w, t) for t in types)

def compute_rule_confidence(rule, type_array, neighborhood_graph, edge_weights=None):
    """Rule confidence = min-based weighted support ratio, exactly as the mining computes it
    (utils/validation.py: support = mean over neighborhoods of the min item weight). Each
    center-type cell is one neighborhood/transaction and the center item has weight 1.0.
    With binary presence weights this reduces to the previous hits / population."""
    center = rule["center_type"]
    sum_ant = sum_both = 0.0
    for cell, cell_type in enumerate(type_array):
        if cell_type != center:
            continue
        neigh = neighborhood_graph[cell]
        edge_w = None if edge_weights is None else edge_weights[cell]
        w_ant = min(1.0, _min_type_weight(type_array, neigh, edge_w, rule["antecedent_extra_types"]))
        sum_ant += w_ant
        w_both = min(w_ant, _min_type_weight(type_array, neigh, edge_w, rule["consequent_types"]))
        sum_both += w_both
    return sum_both / sum_ant if sum_ant > 0 else 0.0

def build_rule_targets(rules_df):
    """Prepare a rule set for the simulation: target = mined Confidence, weight = IG."""
    rules = []
    for _, row in rules_df.iterrows():
        center_type, antecedent_extra_types, consequent_types = parse_rule_for_annealing(row)
        if center_type is None or not consequent_types:
            continue

        rules.append({
            "center_type": center_type,
            "antecedent_extra_types": antecedent_extra_types,
            "consequent_types": consequent_types,
            "target_confidence": row["Confidence"],
            "weight": row["individual_ig"],
        })

    return rules

def total_energy(rules, type_array, neighborhood_graph, edge_weights=None):
    """Scores how well the current tissue matches the rule set's confidence targets."""
    energy = 0.0
    for rule in rules:
        current_confidence = compute_rule_confidence(rule, type_array, neighborhood_graph, edge_weights)
        error = current_confidence - rule["target_confidence"]
        energy += -rule["weight"] * (error ** 2)
    return energy

def propose_swap(type_array):
    """Picks two random cells to swap."""
    n_cells = len(type_array)
    while True:
        i, j = random.sample(range(n_cells), 2)
        if type_array[i] != type_array[j]:
            return i, j

def find_affected_rules(rules, type_a, type_b):
    """Finds only the rules that could possibly change after a swap."""
    affected = []
    for rule in rules:
        types_in_rule = {rule["center_type"], *rule["antecedent_extra_types"], *rule["consequent_types"]}
        if type_a in types_in_rule or type_b in types_in_rule:
            affected.append(rule)
    return affected

def metropolis_accept(delta_energy, temperature):
    """Decides whether to keep a proposed swap."""
    if delta_energy >= 0:
        return True
    if temperature <= 0:
        return False
    probability_of_accepting = math.exp(delta_energy / temperature)
    return random.random() < probability_of_accepting

def _build_csr_graph(neighborhood_graph, edge_weights, n_cells):
    """Flatten the neighbour graph into CSR arrays (positional 0..N-1) plus a parallel array of
    edge weights: all 1.0 for binary presence (fpgrowth), Gaussian for weighted_fpgrowth."""
    ptr = np.zeros(n_cells + 1, dtype=np.int64)
    for c in range(n_cells):
        ptr[c + 1] = ptr[c] + len(neighborhood_graph[c])
    flat = np.empty(int(ptr[-1]), dtype=np.int64)
    flat_w = np.ones(int(ptr[-1]), dtype=np.float64)
    for c in range(n_cells):
        flat[ptr[c]:ptr[c + 1]] = np.asarray(neighborhood_graph[c], dtype=np.int64)
        if edge_weights is not None:
            flat_w[ptr[c]:ptr[c + 1]] = edge_weights[c]
    return flat, flat_w, ptr

def _capped(x):
    """min(x, 1.0) -- the per-type item weight (a count/weight sum capped at 1, as in mining)."""
    return x if x < 1.0 else 1.0

def _cell_ant_weight(state, ri, c):
    """Antecedent support cell c contributes to rule ri: min(center=1.0, antecedent-type weights).
    0.0 unless c is the center type. For binary presence this is 1.0 iff c qualifies, else 0.0."""
    if state["types"][c] != state["r_center"][ri]:
        return 0.0
    nw = state["nbr_w"][c]
    a = 1.0
    for t in state["r_ante"][ri]:
        wt = _capped(nw[t])
        if wt < a:
            a = wt
    return a

def _cell_both_weight(state, ri, c):
    """Antecedent+consequent support cell c contributes: min(ant weight, consequent-type weights)."""
    a = _cell_ant_weight(state, ri, c)
    if a == 0.0:
        return 0.0
    nw = state["nbr_w"][c]
    b = a
    for t in state["r_cons"][ri]:
        wt = _capped(nw[t])
        if wt < b:
            b = wt
    return b

def _rule_sums_fullscan(state, ri):
    """One-time exact (sum_ant, sum_both) for rule ri by scanning its center-type cells."""
    center = state["r_center"][ri]
    cand = np.nonzero(state["types"] == center)[0]
    a = b = 0.0
    for c in cand:
        ca = _cell_ant_weight(state, ri, c)
        if ca == 0.0:
            continue
        a += ca
        b += _cell_both_weight(state, ri, c)
    return a, b

def build_annealing_state(rules, type_array, neighborhood_graph, edge_weights=None):
    """Precompute the numeric structures the incremental engine needs."""
    n_cells = len(type_array)

    # type <-> integer code (include every type any rule mentions, even if absent)
    type_set = set(type_array)
    for r in rules:
        type_set.add(r["center_type"])
        type_set.update(r["antecedent_extra_types"])
        type_set.update(r["consequent_types"])
    type_names = sorted(type_set)
    code = {t: i for i, t in enumerate(type_names)}
    n_types = len(type_names)

    types = np.array([code[t] for t in type_array], dtype=np.int64)
    flat, flat_w, ptr = _build_csr_graph(neighborhood_graph, edge_weights, n_cells)

    # per-cell neighbour-type weight sums (edge weights are all 1.0 in binary mode -> plain counts)
    nbr_w = np.zeros((n_cells, n_types), dtype=np.float64)
    for c in range(n_cells):
        neigh = flat[ptr[c]:ptr[c + 1]]
        if len(neigh):
            np.add.at(nbr_w[c], types[neigh], flat_w[ptr[c]:ptr[c + 1]])

    # numeric rules
    r_center = np.array([code[r["center_type"]] for r in rules], dtype=np.int64)
    r_ante = [np.array([code[t] for t in r["antecedent_extra_types"]], dtype=np.int64) for r in rules]
    r_cons = [np.array([code[t] for t in r["consequent_types"]], dtype=np.int64) for r in rules]
    r_weight = np.array([r["weight"] for r in rules], dtype=np.float64)
    r_target = np.array([r["target_confidence"] for r in rules], dtype=np.float64)
    n_rules = len(rules)

    # which rules involve each type (fast affected-rule lookup on a swap)
    touching = [[] for _ in range(n_types)]
    for ri in range(n_rules):
        involved = {int(r_center[ri]), *r_ante[ri].tolist(), *r_cons[ri].tolist()}
        for t in involved:
            touching[t].append(ri)
    rules_touching = [np.array(x, dtype=np.int64) for x in touching]

    state = dict(
        n_cells=n_cells, n_types=n_types, type_names=type_names, code=code,
        types=types, flat=flat, flat_w=flat_w, ptr=ptr, nbr_w=nbr_w,
        r_center=r_center, r_ante=r_ante, r_cons=r_cons,
        r_weight=r_weight, r_target=r_target, n_rules=n_rules,
        rules_touching=rules_touching,
    )

    sum_ant = np.zeros(n_rules, dtype=np.float64)
    sum_both = np.zeros(n_rules, dtype=np.float64)
    for ri in range(n_rules):
        sum_ant[ri], sum_both[ri] = _rule_sums_fullscan(state, ri)
    state["sum_ant"] = sum_ant
    state["sum_both"] = sum_both
    return state

def _energy_from_sums(state):
    """Objective E = -sum_r weight_r * (conf_r - target_r)^2, from running support sums."""
    a = state["sum_ant"]
    b = state["sum_both"]
    conf = np.where(a > 0, b / np.where(a > 0, a, 1.0), 0.0)
    return float(-np.sum(state["r_weight"] * (conf - state["r_target"]) ** 2))

def _affected_cells(state, i, j):
    """Cells whose rule membership can change when i and j swap: {i, j} and their neighbours."""
    ptr, flat = state["ptr"], state["flat"]
    S = {i, j}
    S.update(flat[ptr[i]:ptr[i + 1]].tolist())
    S.update(flat[ptr[j]:ptr[j + 1]].tolist())
    return np.array(sorted(S), dtype=np.int64)

def _apply_swap(state, i, j):
    """Mutate global types + nbr_w to swap cells i and j in place (weights are symmetric)."""
    types, nbr_w = state["types"], state["nbr_w"]
    ptr, flat, flat_w = state["ptr"], state["flat"], state["flat_w"]
    ti, tj = int(types[i]), int(types[j])
    if ti == tj:
        return
    Ni, Wi = flat[ptr[i]:ptr[i + 1]], flat_w[ptr[i]:ptr[i + 1]]
    Nj, Wj = flat[ptr[j]:ptr[j + 1]], flat_w[ptr[j]:ptr[j + 1]]
    # neighbours of i lose i's ti-weight, gain tj; neighbours of j lose tj, gain ti.
    # (mutual neighbours i in N(j) / j in N(i) are handled naturally via the other cell's list;
    #  a shared neighbour keeps each edge's own weight, so unequal distances do not cancel.)
    nbr_w[Ni, ti] -= Wi
    nbr_w[Ni, tj] += Wi
    nbr_w[Nj, tj] -= Wj
    nbr_w[Nj, ti] += Wj
    types[i], types[j] = tj, ti

def _s_contribution(state, aff, S):
    """For each affected rule, the support sums (ant, both) contributed by the cells in S."""
    ant_S, both_S = {}, {}
    for r in aff:
        r = int(r)
        a = b = 0.0
        for c in S:
            c = int(c)
            ca = _cell_ant_weight(state, r, c)
            if ca == 0.0:
                continue
            a += ca
            b += _cell_both_weight(state, r, c)
        ant_S[r], both_S[r] = a, b
    return ant_S, both_S

def _swap_energy_delta(state, i, j, aff):
    """Apply the swap globally and return (delta_energy, dant, dboth) over the affected rules."""
    S = _affected_cells(state, i, j)
    old_ant_S, old_both_S = _s_contribution(state, aff, S)
    _apply_swap(state, i, j)
    new_ant_S, new_both_S = _s_contribution(state, aff, S)

    w, tgt = state["r_weight"], state["r_target"]
    sum_ant, sum_both = state["sum_ant"], state["sum_both"]
    delta = 0.0
    dant, dboth = {}, {}
    for r in aff:
        r = int(r)
        da = new_ant_S[r] - old_ant_S[r]
        db = new_both_S[r] - old_both_S[r]
        dant[r], dboth[r] = da, db
        a0, b0 = float(sum_ant[r]), float(sum_both[r])
        a1, b1 = a0 + da, b0 + db
        c0 = (b0 / a0) if a0 > 0 else 0.0
        c1 = (b1 / a1) if a1 > 0 else 0.0
        # contribution is -w*(c-tgt)^2, so delta = w*((c0-tgt)^2 - (c1-tgt)^2)
        delta += w[r] * ((c0 - tgt[r]) ** 2 - (c1 - tgt[r]) ** 2)
    return delta, dant, dboth

def validate_fast_engine(rules, type_array, neighborhood_graph, edge_weights=None, n_checks=150, seed=0):
    """Correctness gate: assert the fast engine matches the slow oracle (total_energy)."""
    rng = random.Random(seed)
    tol = 1e-6

    ref_E = total_energy(rules, list(type_array), neighborhood_graph, edge_weights)
    state = build_annealing_state(rules, type_array, neighborhood_graph, edge_weights)
    fast_E = _energy_from_sums(state)
    assert abs(ref_E - fast_E) < tol, f"initial energy mismatch: ref={ref_E} fast={fast_E}"

    cur = list(type_array)
    n = len(cur)
    max_err = 0.0
    for _ in range(n_checks):
        i, j = rng.randrange(n), rng.randrange(n)
        while cur[i] == cur[j]:
            i, j = rng.randrange(n), rng.randrange(n)
        ti, tj = int(state["types"][i]), int(state["types"][j])
        aff = np.union1d(state["rules_touching"][ti], state["rules_touching"][tj])

        e_before = total_energy(rules, cur, neighborhood_graph, edge_weights)
        cur[i], cur[j] = cur[j], cur[i]
        ref_delta = total_energy(rules, cur, neighborhood_graph, edge_weights) - e_before

        delta, dant, dboth = _swap_energy_delta(state, i, j, aff)   # leaves state swapped
        for r in aff:                                               # commit to stay in sync with cur
            r = int(r)
            state["sum_ant"][r] += dant[r]
            state["sum_both"][r] += dboth[r]

        max_err = max(max_err, abs(delta - ref_delta))
        assert abs(delta - ref_delta) < tol, f"delta mismatch: fast={delta} ref={ref_delta}"

    final_ref = total_energy(rules, cur, neighborhood_graph, edge_weights)
    final_fast = _energy_from_sums(state)
    assert abs(final_ref - final_fast) < tol, f"final energy mismatch: ref={final_ref} fast={final_fast}"
    print(f"validate_fast_engine: {n_checks} swaps OK  |  init err={abs(ref_E - fast_E):.2e}, "
          f"max delta err={max_err:.2e}, final err={abs(final_ref - final_fast):.2e}")
    return True

def compute_bin_ids(df_fov, grid_size=None):
    """Integer macro-bin id per cell (x//grid, y//grid). Intra-bin swaps leave macro fixed."""
    g = GRID_SIZE if grid_size is None else grid_size
    bx = (df_fov['x_um'] // g).astype(int).values
    by = (df_fov['y_um'] // g).astype(int).values
    return bx * (by.max() + 1) + by

def _propose_swap(types, n_cells, bin_ids, bins_to_cells):
    """Two different-type cells to swap. bin_ids=None -> anywhere; else within the same macro bin."""
    if bin_ids is None:
        i, j = random.randrange(n_cells), random.randrange(n_cells)
        while types[i] == types[j]:
            i, j = random.randrange(n_cells), random.randrange(n_cells)
        return i, j
    for _ in range(64):                        # intra-bin swap keeps each bin's composition fixed
        i = random.randrange(n_cells)
        cells = bins_to_cells[int(bin_ids[i])]
        if len(cells) < 2:
            continue
        j = random.choice(cells)
        if types[i] != types[j]:
            return i, j
    return None, None                          # no different-type partner in the bin this step

def run_simulated_annealing(rules, type_array, neighborhood_graph, edge_weights=None,
                            n_iterations=20000, start_temperature=1.0, cooling_rate=0.9998,
                            bin_ids=None, verbose=False):
    """Runs the full simulated annealing search using the fast incremental engine.
    bin_ids given -> restrict swaps to intra-bin (macro-preserving) moves."""
    state = build_annealing_state(rules, type_array, neighborhood_graph, edge_weights)
    types = state["types"]
    rules_touching = state["rules_touching"]
    n_cells = state["n_cells"]

    bins_to_cells = None                        # cells grouped by macro bin, for intra-bin swaps
    if bin_ids is not None:
        bins_to_cells = {}
        for c, b in enumerate(bin_ids):
            bins_to_cells.setdefault(int(b), []).append(c)

    def _as_strings():
        names = state["type_names"]
        return [names[t] for t in state["types"]]

    current_energy = _energy_from_sums(state)
    energy_history = [current_energy]
    delta_energy_history = []
    checkpoint_tissues = []
    checkpoint_every = max(1, n_iterations // 10)

    for step in range(n_iterations):
        i, j = _propose_swap(types, n_cells, bin_ids, bins_to_cells)   # any swap, or intra-bin
        if i is None:                          # no valid intra-bin partner this step
            energy_history.append(current_energy); continue
        ti, tj = int(types[i]), int(types[j])

        aff = np.union1d(rules_touching[ti], rules_touching[tj])
        if aff.size == 0:
            # swap touches no golden rule -- nothing can change, skip it
            energy_history.append(current_energy)
            continue

        delta, dant, dboth = _swap_energy_delta(state, i, j, aff)  # state now swapped
        delta_energy_history.append(delta)

        temperature = start_temperature * (cooling_rate ** step)
        if metropolis_accept(delta, temperature):
            for r in aff:                      # commit the running support sums
                r = int(r)
                state["sum_ant"][r] += dant[r]
                state["sum_both"][r] += dboth[r]
            current_energy += delta
        else:
            _apply_swap(state, i, j)           # undo the swap; pop/hit never committed

        energy_history.append(current_energy)

        if verbose and (step + 1) % checkpoint_every == 0 and len(checkpoint_tissues) < 10:
            checkpoint_tissues.append(_as_strings())
            print(f"  Checkpoint {len(checkpoint_tissues)}/10 (iteration {step + 1}/{n_iterations}): "
                  f"objective = {current_energy:.4f}")

    if verbose:
        # guarantee exactly 10 checkpoints, the last one being the true final tissue
        if len(checkpoint_tissues) < 10:
            checkpoint_tissues.append(_as_strings())
        checkpoint_tissues[-1] = _as_strings()

    return _as_strings(), energy_history, delta_energy_history, checkpoint_tissues

def staged_anneal(macro_rules_df, joint_rules_df, type_array, neighborhood_graph, bin_ids,
                  edge_weights=None, n_iterations=100000, start_temperature=1.0,
                  cooling_rate=0.9998, phase2_temp_frac=0.2):
    """Two-phase run: (1) build MACRO from macro rules with any swaps; (2) refine with ALL rules
    using intra-bin swaps only (macro preserved exactly). Returns (final_tissue, energy_history)."""
    macro_ann = build_rule_targets(macro_rules_df)
    joint_ann = build_rule_targets(joint_rules_df)
    mid, e1, *_ = run_simulated_annealing(                       # phase 1: build the scaffold
        macro_ann, type_array, neighborhood_graph, edge_weights=edge_weights,
        n_iterations=n_iterations, start_temperature=start_temperature,
        cooling_rate=cooling_rate, bin_ids=None)
    final, e2, *_ = run_simulated_annealing(                     # phase 2: refine, macro-preserving
        joint_ann, mid, neighborhood_graph, edge_weights=edge_weights,
        n_iterations=n_iterations, start_temperature=start_temperature * phase2_temp_frac,
        cooling_rate=cooling_rate, bin_ids=bin_ids)
    return final, e1 + e2



# ==========================================================================
# Synthetic-tissue scoring (emergent macro / micro / eye-aligned)
# ==========================================================================

def build_synthetic_dataframe(df_fov, type_array):
    """Packages the final simulated cell types back into a tissue table, in the"""
    df_synthetic = df_fov.copy()
    df_synthetic["cell_type"] = type_array
    return df_synthetic

def _grid_morans_i(df, cell_type, grid_size):
    """Spatial autocorrelation (Moran's I, rook adjacency) of one cell type's"""
    if df.empty:
        return float('nan')
    bx_all = (df['x_um'] // grid_size).astype(int)
    by_all = (df['y_um'] // grid_size).astype(int)
    nx, ny = int(bx_all.max()) + 1, int(by_all.max()) + 1
    grid = np.zeros((nx, ny))
    sub = df[df['cell_type'] == cell_type]
    for x, y in zip((sub['x_um'] // grid_size).astype(int), (sub['y_um'] // grid_size).astype(int)):
        grid[int(x), int(y)] += 1
    z = grid - grid.mean()
    denom = (z ** 2).sum()
    if denom == 0:
        return float('nan')
    num, W = 0.0, 0
    for i in range(nx):
        for j in range(ny):
            for di, dj in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                ii, jj = i + di, j + dj
                if 0 <= ii < nx and 0 <= jj < ny:
                    num += z[i, j] * z[ii, jj]; W += 1
    if W == 0:
        return float('nan')
    return (nx * ny / W) * (num / denom)

def score_synthetic_tissue(df_real, df_synth, df_random, neighborhood_graph,
                           grid_size=None, alpha=0.01):
    """Quantify how well a synthetic (annealed) tissue reproduces the real tissue,"""
    grid_size = GRID_SIZE if grid_size is None else grid_size
    types = sorted(df_real['cell_type'].dropna().unique())

    # --- MICRO: local neighbourhood motifs over all cell-type pairs ---
    micro_res = micro_base = 0.0
    for c in types:
        for n in types:
            s_real = extract_patch_states(df_real, neighborhood_graph, c, [n])
            if s_real.empty:
                continue
            s_syn = extract_patch_states(df_synth,  neighborhood_graph, c, [n])
            s_rnd = extract_patch_states(df_random, neighborhood_graph, c, [n])
            if s_syn.empty or s_rnd.empty:
                continue
            p_real = calculate_smoothed_probabilities(s_real, alpha=alpha)
            p_syn  = calculate_smoothed_probabilities(s_syn,  alpha=alpha)
            p_rnd  = calculate_smoothed_probabilities(s_rnd,  alpha=alpha)
            micro_res  += calculate_kl_divergence(p_real, p_syn)
            micro_base += calculate_kl_divergence(p_real, p_rnd)
    micro_reproduced = 1 - micro_res / micro_base if micro_base > 0 else float('nan')

    # --- MACRO: density gradients via coarse (bin, cell-type) composition ---
    def bin_counts(df):
        bx = (df['x_um'] // grid_size).astype(int).astype(str)
        by = (df['y_um'] // grid_size).astype(int).astype(str)
        return Counter(bx + '_' + by + '|' + df['cell_type'].astype(str))

    cr, cs, cn = bin_counts(df_real), bin_counts(df_synth), bin_counts(df_random)
    support = set(cr) | set(cs) | set(cn)

    def to_prob(counts):
        denom = sum(counts.values()) + alpha * len(support)
        return {k: (counts.get(k, 0) + alpha) / denom for k in support}

    P_real, P_syn, P_rnd = to_prob(cr), to_prob(cs), to_prob(cn)
    macro_res  = calculate_kl_divergence(P_real, P_syn)
    macro_base = calculate_kl_divergence(P_real, P_rnd)
    macro_reproduced = 1 - macro_res / macro_base if macro_base > 0 else float('nan')

    # --- MACRO cross-checks (eye-aligned) ---
    comp_keys = sorted(set(cr) | set(cs))
    ra = np.array([cr.get(k, 0) for k in comp_keys], float)
    sa = np.array([cs.get(k, 0) for k in comp_keys], float)
    comp_corr = float(np.corrcoef(ra, sa)[0, 1]) if ra.std() > 0 and sa.std() > 0 else float('nan')

    mi_real, mi_syn = [], []
    for t in types:
        ir = _grid_morans_i(df_real, t, grid_size)
        isy = _grid_morans_i(df_synth, t, grid_size)
        if not (np.isnan(ir) or np.isnan(isy)):
            mi_real.append(ir); mi_syn.append(isy)
    morans_corr = (float(np.corrcoef(mi_real, mi_syn)[0, 1])
                   if len(mi_real) >= 2 and np.std(mi_real) > 0 and np.std(mi_syn) > 0
                   else float('nan'))

    return dict(micro_reproduced=micro_reproduced, macro_reproduced=macro_reproduced,
                micro_residual=micro_res, micro_baseline=micro_base,
                macro_residual=macro_res, macro_baseline=macro_base,
                comp_corr=comp_corr, morans_corr=morans_corr)



# ==========================================================================
# Rule-set comparison by simulation
# ==========================================================================

def _score_rep(df_fov, final, energy, df_global_null, neighborhood_graph):
    """Score one annealed tissue into a repeat dict (macro/micro/comp/Moran + final + energy)."""
    sc = score_synthetic_tissue(df_fov, build_synthetic_dataframe(df_fov, final),
                                df_global_null, neighborhood_graph)
    return dict(macro=100 * sc['macro_reproduced'], micro=100 * sc['micro_reproduced'],
                comp=sc['comp_corr'], moran=sc['morans_corr'], final=final, energy=energy)

def _print_row(row):
    mac = [x['macro'] for x in row['reps']]
    print(f"  {row['name']:16s} {row['k']:4d} rules |  MACRO {np.mean(mac):7.1f}% +/-{np.std(mac):4.1f}  "
          f"(micro {np.mean([x['micro'] for x in row['reps']]):5.1f}%)")

def compare_rule_sets(named_sets, df_fov, neighborhood_graph, df_global_null, df_fovs_meta,
                      fov_id, edge_weights=None, n_iterations=100000, start_temperature=1.0,
                      cooling_rate=0.9998, seed=0, n_repeats=1, staged=None, show_tissue=True,
                      swap_bin_ids=None):
    """Anneal each rule set n_repeats times (paired seeds) and score emergent MACRO + cross-checks.
    staged = (name, macro_rules_df, joint_rules_df, bin_ids) adds a two-phase macro->micro group.
    swap_bin_ids restricts every anneal to intra-bin swaps, holding bin composition fixed."""
    start_types = df_global_null['cell_type'].tolist()
    rows = []
    for name, rules_df in named_sets:
        ann = build_rule_targets(rules_df)
        reps = []
        for r in range(n_repeats):
            random.seed(seed + r); np.random.seed(seed + r)   # same seed per repeat across sets
            final, energy, *_ = run_simulated_annealing(ann, start_types, neighborhood_graph,
                                    edge_weights=edge_weights, n_iterations=n_iterations,
                                    start_temperature=start_temperature, cooling_rate=cooling_rate,
                                    bin_ids=swap_bin_ids)
            reps.append(_score_rep(df_fov, final, energy, df_global_null, neighborhood_graph))
        rows.append(dict(name=name, k=len(rules_df), reps=reps))
        _print_row(rows[-1])

    if staged is not None:                                    # two-phase macro->micro group
        s_name, macro_df, joint_df, bin_ids = staged
        reps = []
        for r in range(n_repeats):
            random.seed(seed + r); np.random.seed(seed + r)
            final, energy = staged_anneal(macro_df, joint_df, start_types, neighborhood_graph,
                                bin_ids, edge_weights=edge_weights, n_iterations=n_iterations,
                                start_temperature=start_temperature, cooling_rate=cooling_rate)
            reps.append(_score_rep(df_fov, final, energy, df_global_null, neighborhood_graph))
        rows.append(dict(name=s_name, k=len(joint_df), reps=reps))
        _print_row(rows[-1])

    _plot_ruleset_comparison(rows, fov_id)
    _plot_ruleset_convergence(rows, fov_id)
    if show_tissue:
        _plot_repeats_grid(rows, df_fov, start_types, fov_id, df_fovs_meta)
    return rows

def _agg(reps, key):
    """Mean and std of one metric across a set's repeats."""
    v = [x[key] for x in reps]
    return float(np.mean(v)), float(np.std(v))

def _plot_ruleset_comparison(rows, fov_id):
    """Scatter (emergent macro vs micro) + all-metrics bars; points are mean +/- std over repeats."""
    fig, (axS, axB) = plt.subplots(1, 2, figsize=(14, 5.5), facecolor='white')
    for r in rows:                                      # left: macro (rank by this, y) vs micro (x)
        mmac, smac = _agg(r['reps'], 'macro'); mmic, smic = _agg(r['reps'], 'micro')
        axS.errorbar(mmic, mmac, xerr=smic, yerr=smac, fmt='o', ms=10, capsize=3, zorder=3)
        axS.annotate(f"{r['name']}\n({r['k']} rules)", (mmic, mmac),
                     fontsize=8, textcoords='offset points', xytext=(8, 6))
    axS.axhline(0, color='#bbb', lw=0.9); axS.axvline(0, color='#bbb', lw=0.9)
    axS.set_xlabel('MICRO reproduced (%)  -  local motifs (sanity)')
    axS.set_ylabel('MACRO reproduced (%)  -  emergent anatomy')
    axS.set_title('Emergent macro vs micro   (0 = random start, 100 = real)')
    axS.set_facecolor('white')
    metrics = [('macro', 'MACRO %', 1), ('micro', 'micro %', 1),
               ('comp', 'comp-r x100', 100), ('moran', 'Moran-r x100', 100)]
    x = np.arange(len(metrics)); w = 0.8 / max(len(rows), 1)
    for i, r in enumerate(rows):                         # right: all four metrics, mean +/- std
        means = [_agg(r['reps'], k)[0] * s for k, _, s in metrics]
        errs  = [_agg(r['reps'], k)[1] * s for k, _, s in metrics]
        axB.bar(x + i * w, means, width=w, yerr=errs, capsize=2, label=f"{r['name']} ({r['k']})")
    axB.axhline(0, color='#bbb', lw=0.9)
    axB.set_xticks(x + w * (len(rows) - 1) / 2); axB.set_xticklabels([m[1] for m in metrics], fontsize=9)
    axB.set_ylabel('score'); axB.set_title('All metrics per rule set')
    axB.legend(fontsize=8, frameon=False); axB.set_facecolor('white')
    fig.suptitle(f'FOV {fov_id} - rule sets: rank by MACRO (y); micro = sanity (x)', y=1.02, fontsize=13)
    plt.tight_layout(); plt.show()

def _plot_ruleset_convergence(rows, fov_id):
    """Overlay every anneal's objective curve, one color per rule set (all repeats shown)."""
    fig, ax = plt.subplots(figsize=(8, 5), facecolor='white')
    colors = plt.cm.tab10(np.linspace(0, 1, max(len(rows), 1)))
    for r, c in zip(rows, colors):
        for j, rep in enumerate(r['reps']):
            ax.plot(rep['energy'], color=c, lw=1, alpha=0.7, label=r['name'] if j == 0 else None)
    ax.set_xlabel('iteration'); ax.set_ylabel('objective (E)')
    ax.set_title(f'FOV {fov_id} - annealing convergence by rule set')
    ax.legend(fontsize=8, frameon=False); ax.set_facecolor('white')
    plt.tight_layout(); plt.show()

def _plot_repeats_grid(rows, df_fov, start_types, fov_id, df_fovs_meta):
    """Tissue grid: one row per rule set; columns = Real, Random start, then each repeat."""
    n_rep = len(rows[0]['reps']) if rows else 0
    ncols = 2 + n_rep
    fig, axes = plt.subplots(len(rows), ncols, figsize=(9 * ncols, 9 * len(rows)),
                             facecolor='white', squeeze=False)   # 9in panels, like the progression plot
    for i, r in enumerate(rows):
        plot_fov_tissue(df_fov, fov_id, df_fovs_meta, ax=axes[i][0], title=f'{r["name"]}: Real')
        plot_fov_tissue(build_synthetic_dataframe(df_fov, start_types), fov_id, df_fovs_meta,
                        ax=axes[i][1], title='Random start')
        for j, rep in enumerate(r['reps']):
            plot_fov_tissue(build_synthetic_dataframe(df_fov, rep['final']), fov_id, df_fovs_meta,
                            ax=axes[i][2 + j], title=f'repeat {j + 1}  (macro {rep["macro"]:.0f}%)')
    for ax in axes.ravel():                              # drop per-panel legends to keep it clean
        leg = ax.get_legend()
        if leg:
            leg.remove()
    fig.suptitle(f'FOV {fov_id} - synthetic tissue per rule set x repeat', fontsize=14, y=1.005)
    plt.tight_layout(); plt.show()



# ==========================================================================
# Plots
# ==========================================================================

def plot_ig_distribution(rules_df, threshold=None):
    """Histogram of individual IG values. Helps choose the filtering threshold."""
    igs = rules_df['individual_ig']
    positive = igs[igs > 0]

    fig, ax = plt.subplots(figsize=(8, 4), facecolor='white')
    ax.hist(positive, bins=60, log=True, color='#457B9D', edgecolor='white', linewidth=0.3)
    if threshold is not None:
        ax.axvline(threshold, color='#E63946', linestyle='--', linewidth=1.5,
                   label=f'Threshold = {threshold} bits')
        ax.legend(frameon=False)
    ax.set_xlabel('Individual Information Gain (bits)', fontsize=11)
    ax.set_ylabel('Rule Count (log scale)', fontsize=11)
    ax.set_title('Individual IG Distribution', fontsize=12)
    ax.set_facecolor('white')
    plt.tight_layout()
    plt.show()

def plot_top_rules(rules_df, n=20):
    """Horizontal bar chart of the top N rules by individual IG."""
    top = rules_df.nlargest(n, 'individual_ig').copy()

    def rule_label(row):
        ants = [item.replace('_CENTER', '').replace('_NEIGHBOR', '')
                for item in ast.literal_eval(row['Antecedents'])]
        cons = [item.replace('_CENTER', '').replace('_NEIGHBOR', '')
                for item in ast.literal_eval(row['Consequents'])]
        return ' + '.join(ants) + ' → ' + ' + '.join(cons)

    top['label'] = top.apply(rule_label, axis=1)
    top['label'] = top['label'].apply(lambda s: textwrap.shorten(s, width=55, placeholder='...'))

    tier_colors = {1: '#457B9D', 2: '#E63946', 3: '#F4A300', 4: '#8E44AD'}
    colors = [tier_colors.get(t, '#9E9E9E') for t in top['tier']]

    fig, ax = plt.subplots(figsize=(10, max(4, n * 0.35)), facecolor='white')
    ax.barh(range(len(top)), top['individual_ig'].values, color=colors, edgecolor='none')
    ax.set_yticks(range(len(top)))
    ax.set_yticklabels(top['label'].values, fontsize=8)
    ax.invert_yaxis()
    ax.set_xlabel('Individual Information Gain (bits)', fontsize=11)
    ax.set_title(f'Top {n} Rules by Individual IG', fontsize=12)
    ax.set_facecolor('white')

    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(color=tier_colors[1], label='Tier 1 (different-type, simple)'),
                       Patch(color=tier_colors[2], label='Tier 2 (different-type, complex)'),
                       Patch(color=tier_colors[3], label='Tier 3 (same-type, simple)'),
                       Patch(color=tier_colors[4], label='Tier 4 (same-type, complex)')],
              frameon=False, fontsize=9)
    plt.tight_layout()
    plt.show()

def plot_fidelity_selection(x_vals, cum_ig, total_baseline, achievable, fidelity, k_star, fov_id,
                            elbow_k_val=None):
    """Two clean panels for rule selection by information-capture fidelity."""
    x   = np.array(x_vals)
    cum = np.array(cum_ig)
    pct = 100 * cum / total_baseline if total_baseline > 0 else np.zeros_like(cum)
    ceiling_pct = 100 * achievable / total_baseline if total_baseline > 0 else 0.0
    target_pct  = fidelity * ceiling_pct
    marginal = np.diff(cum, prepend=0.0)
    k_idx = list(x).index(k_star)
    kept = x <= k_star

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 4.8), facecolor='white')

    # Left: information captured, with reachable ceiling, target, k*, and elbow
    ax1.fill_between(x, 0, pct, color='#457B9D', alpha=0.15)
    ax1.plot(x, pct, color='#457B9D', linewidth=2, marker='o', markersize=3)
    ax1.axhline(ceiling_pct, color='#9E9E9E', linestyle=':', linewidth=1.3,
                label=f'rule-set ceiling ({ceiling_pct:.0f}%)')
    ax1.axhline(target_pct, color='#E63946', linestyle='--', linewidth=1.3,
                label=f'{fidelity:.0%} of ceiling')
    if elbow_k_val is not None and elbow_k_val in list(x):
        e_idx = list(x).index(elbow_k_val)
        ax1.axvline(elbow_k_val, color='#F4A300', linewidth=1.5)
        ax1.scatter([elbow_k_val], [pct[e_idx]], color='#F4A300', s=70, marker='D', zorder=5,
                    label=f'elbow = {elbow_k_val} rules  ({pct[e_idx]:.0f}% of total)')
    ax1.axvline(k_star, color='#2DC653', linewidth=1.5)
    ax1.scatter([k_star], [pct[k_idx]], color='#2DC653', s=70, zorder=5,
                label=f'k* = {k_star} rules  ({pct[k_idx]:.0f}% of total)')
    ax1.set_xlabel('Number of rules', fontsize=11)
    ax1.set_ylabel('Tissue information captured (%)', fontsize=11)
    ax1.set_title('Information captured', fontsize=12)
    ax1.set_ylim(0, 105)
    ax1.legend(frameon=False, fontsize=9, loc='lower right')
    ax1.set_facecolor('white')

    # Right: per-rule marginal IG (diminishing returns)
    ax2.bar(x[kept], marginal[kept], color='#2DC653', alpha=0.85, label='kept (<= k*)')
    if (~kept).any():
        ax2.bar(x[~kept], marginal[~kept], color='#CCCCCC', label='beyond k*')
    if elbow_k_val is not None and elbow_k_val in list(x):
        ax2.axvline(elbow_k_val, color='#F4A300', linewidth=1.5, label='elbow')
    ax2.axvline(k_star, color='#2DC653', linewidth=1.5)
    ax2.set_xlabel('Number of rules', fontsize=11)
    ax2.set_ylabel('Marginal information gain (bits)', fontsize=11)
    ax2.set_title('Per-rule contribution', fontsize=12)
    ax2.legend(frameon=False, fontsize=9)
    ax2.set_facecolor('white')

    fig.suptitle(f'Rule selection - FOV {fov_id}', fontsize=13)
    plt.tight_layout()
    plt.show()

def _add_scale_bar(ax, x_max, y_max, bar_um=50.0):
    x_end   = x_max - 25
    x_start = x_end - bar_um
    y_line  = y_max - 25
    ax.plot([x_start, x_end], [y_line, y_line],
            color='black', linewidth=4, solid_capstyle='butt')
    ax.text((x_start + x_end) / 2, y_line - 12, '50 µm',
            color='black', ha='center', va='bottom', fontsize=9)

def plot_fov_tissue(df_fov, fov_id, df_fovs_meta, ax=None, title=None):
    """Plot a single FOV. Exact nb06 visual style. Pass ax to use as subplot."""
    show = ax is None
    if show:
        fig, ax = plt.subplots(figsize=(10, 10), facecolor='#ffffff')

    row_meta = df_fovs_meta[df_fovs_meta['FOV'] == fov_id]
    size_um  = row_meta['Size [um]'].values[0] if not row_meta.empty else 400
    cell_s   = 90 if size_um == 400 else 45

    ax.set_facecolor('#eaeaeaff')
    for ct, group in df_fov.groupby('cell_type'):
        ax.scatter(group['x_um'], group['y_um'], s=cell_s,
                   c=[CELL_COLOR_MAP.get(ct, OTHER_COLOR)], alpha=0.9, linewidths=0)

    x_max, y_max = df_fov['x_um'].max(), df_fov['y_um'].max()
    ax.set_axisbelow(True)
    ax.xaxis.set_major_locator(MultipleLocator(100.0))
    ax.yaxis.set_major_locator(MultipleLocator(100.0))
    ax.xaxis.set_minor_locator(MultipleLocator(25.0))
    ax.yaxis.set_minor_locator(MultipleLocator(25.0))
    ax.grid(which='minor', color='white', linestyle='-', linewidth=0.8, alpha=0.7)
    ax.grid(which='major', color='lightgray', linestyle='-', linewidth=0.8, alpha=0.8)
    _add_scale_bar(ax, x_max, y_max)
    ax.set_xlabel('x (µm)', color='black')
    ax.set_ylabel('y (µm)', color='black')
    ax.set_aspect('equal', adjustable='box')
    ax.invert_yaxis()
    ax.tick_params(axis='both', colors='black', labelsize=9)
    for spine in ax.spines.values():
        spine.set_visible(False)

    ax.set_title(title or f'Full FOV: {fov_id}', color='black', fontsize=14)

    present = sorted(df_fov['cell_type'].dropna().unique())
    handles = [plt.Line2D([0], [0], marker='o', color='w',
               markerfacecolor=CELL_COLOR_MAP.get(ct, 'black'),
               markeredgecolor='none', markersize=8, label=ct)
               for ct in present]
    ax.legend(handles=handles, title='Cell type',
              bbox_to_anchor=(1.0, 1), loc='upper left', fontsize=7, frameon=False)

    if show:
        plt.tight_layout()
        plt.show()

def plot_three_panel_fov(df_fov, fov_id, df_fovs_meta):
    """Three-panel comparison: real tissue / macro shuffle / global shuffle."""
    df_macro  = create_macro_null_model(df_fov)
    df_global = create_global_null_model(df_fov)

    fig, axes = plt.subplots(1, 3, figsize=(30, 10), facecolor='#ffffff')
    titles  = ['Real Tissue',
               f'Macro Shuffle (grid={GRID_SIZE:.0f}µm)',
               'Global Shuffle']
    tissues = [df_fov, df_macro, df_global]

    for ax, df, title in zip(axes, tissues, titles):
        plot_fov_tissue(df, fov_id, df_fovs_meta, ax=ax, title=title)
        leg = ax.get_legend()
        if leg:
            leg.remove()  # remove per-panel legends; one shared legend below

    # Single shared legend on the last panel
    present = sorted(df_fov['cell_type'].dropna().unique())
    handles = [plt.Line2D([0], [0], marker='o', color='w',
               markerfacecolor=CELL_COLOR_MAP.get(ct, 'black'),
               markeredgecolor='none', markersize=8, label=ct)
               for ct in present]
    axes[-1].legend(handles=handles, title='Cell type',
                    bbox_to_anchor=(1.02, 1), loc='upper left',
                    fontsize=7, frameon=False)

    fig.suptitle(f'FOV {fov_id} — Null Model Comparison', fontsize=14, y=1.01)
    plt.tight_layout()
    plt.show()

def compute_power_spectra(df_fov, pixel_size_um=10.0):
    """Rasterize each cell type to a density grid, apply 2D FFT,"""
    x_max = df_fov['x_um'].max() + pixel_size_um
    y_max = df_fov['y_um'].max() + pixel_size_um
    nx    = max(4, int(np.ceil(x_max / pixel_size_um)))
    ny    = max(4, int(np.ceil(y_max / pixel_size_um)))

    x_edges = np.linspace(0, x_max, nx + 1)
    y_edges = np.linspace(0, y_max, ny + 1)

    results = {}
    for ct in sorted(df_fov['cell_type'].dropna().unique()):
        cells_ct = df_fov[df_fov['cell_type'] == ct]
        density, _, _ = np.histogram2d(cells_ct['x_um'], cells_ct['y_um'],
                                       bins=[x_edges, y_edges])
        fft2  = np.fft.fft2(density - density.mean())
        power = np.fft.fftshift(np.abs(fft2) ** 2)

        cy, cx  = ny // 2, nx // 2
        yi, xi  = np.ogrid[:ny, :nx]
        r       = np.round(np.sqrt((xi - cx)**2 + (yi - cy)**2)).astype(int)
        max_r   = min(cx, cy)

        radial = np.array([power[r == ri].mean() if (r == ri).any() else 0
                           for ri in range(1, max_r + 1)])
        scales = (max(nx, ny) * pixel_size_um) / np.arange(1, max_r + 1)

        if radial.max() > 0:
            radial = radial / radial.max()
        results[ct] = (scales, radial)

    return results

def plot_power_spectra(df_fov, fov_id, pixel_size_um=10.0):
    """Power spectrum per cell type (radially averaged 2D FFT of density maps)."""
    spectra = compute_power_spectra(df_fov, pixel_size_um=pixel_size_um)

    fig, ax = plt.subplots(figsize=(10, 5), facecolor='white')
    for ct, (scales, power) in spectra.items():
        mask = scales <= 400
        ax.plot(scales[mask], power[mask],
                color=CELL_COLOR_MAP.get(ct, OTHER_COLOR),
                linewidth=1.5, alpha=0.85, label=ct)

    ax.axvline(GRID_SIZE, color='black', linestyle='--', linewidth=1.8,
               label=f'GRID_SIZE = {GRID_SIZE:.0f} µm')
    ax.set_xlabel('Spatial Scale (µm)', fontsize=11)
    ax.set_ylabel('Normalized Power', fontsize=11)
    ax.set_title(f'Cell Type Power Spectra — FOV {fov_id}', fontsize=12)
    ax.set_xscale('log')
    ax.invert_xaxis()  # large scales on left (macro), small on right (micro)
    ax.legend(fontsize=7, ncol=2, frameon=False)
    ax.set_facecolor('white')
    plt.tight_layout()
    plt.show()
    print(f'Inspect the curve to calibrate GRID_SIZE (currently {GRID_SIZE} µm).')
    print('If the macro/micro transition falls at a different scale, '
          'update GRID_SIZE in cell 2 and re-run.')

def plot_real_sim_random(df_fov, final_tissue, initial_tissue, fov_id, df_fovs_meta, set_name='rules'):
    """Three tissue panels side by side: real, simulated (from set_name), and random start."""
    df_sim    = build_synthetic_dataframe(df_fov, final_tissue)
    df_random = build_synthetic_dataframe(df_fov, initial_tissue)

    fig, axes = plt.subplots(1, 3, figsize=(30, 10), facecolor='#ffffff')
    panels = [(df_fov,    'A. Real Tissue'),
              (df_sim,    f'B. Simulated ({set_name})'),
              (df_random, 'C. Random (start)')]
    for ax, (df, title) in zip(axes, panels):
        plot_fov_tissue(df, fov_id, df_fovs_meta, ax=ax, title=title)
        leg = ax.get_legend()
        if leg:
            leg.remove()

    present = sorted(df_fov['cell_type'].dropna().unique())
    handles = [plt.Line2D([0], [0], marker='o', color='w',
               markerfacecolor=CELL_COLOR_MAP.get(ct, 'black'),
               markeredgecolor='none', markersize=8, label=ct)
               for ct in present]
    axes[-1].legend(handles=handles, title='Cell type',
                    bbox_to_anchor=(1.02, 1), loc='upper left',
                    fontsize=7, frameon=False)

    fig.suptitle(f'FOV {fov_id} - Real vs Simulated vs Random', fontsize=14, y=1.01)
    plt.tight_layout()
    plt.show()

def plot_annealing_progression(df_fov, initial_tissue, checkpoint_tissues, fov_id, df_fovs_meta):
    """One row of panels: the real tissue, the fully-random starting point, then a"""
    n_panels = 2 + len(checkpoint_tissues)
    fig, axes = plt.subplots(1, n_panels, figsize=(9 * n_panels, 9), facecolor='#ffffff')

    plot_fov_tissue(df_fov, fov_id, df_fovs_meta, ax=axes[0], title='Real Tissue')

    df_random = build_synthetic_dataframe(df_fov, initial_tissue)
    plot_fov_tissue(df_random, fov_id, df_fovs_meta, ax=axes[1], title='Totally Random (start)')

    for k, (ax, tissue) in enumerate(zip(axes[2:], checkpoint_tissues), start=1):
        pct = int(round(100 * k / len(checkpoint_tissues)))
        df_stage = build_synthetic_dataframe(df_fov, tissue)
        plot_fov_tissue(df_stage, fov_id, df_fovs_meta, ax=ax, title=f'{pct}% of run')

    for ax in axes:
        leg = ax.get_legend()
        if leg:
            leg.remove()

    present = sorted(df_fov['cell_type'].dropna().unique())
    handles = [plt.Line2D([0], [0], marker='o', color='w',
               markerfacecolor=CELL_COLOR_MAP.get(ct, 'black'),
               markeredgecolor='none', markersize=8, label=ct)
               for ct in present]
    axes[-1].legend(handles=handles, title='Cell type',
                    bbox_to_anchor=(1.02, 1), loc='upper left',
                    fontsize=7, frameon=False)

    fig.suptitle(f'FOV {fov_id} - Simulated Annealing Progression', fontsize=14, y=1.02)
    plt.tight_layout()
    plt.show()

def plot_macro_micro_selection(macro_cum, k_macro, micro_cum, k_micro,
                               ranked_rules, in_macro, in_micro, fov_id):
    fig, (axL, axR) = plt.subplots(1, 2, figsize=(13.5, 5), facecolor='white')

    # Panel A: cumulative component curves + elbow cuts
    axL.plot(np.arange(1, len(macro_cum) + 1), macro_cum, color='#C1440E', lw=2,
             label='anatomy-aligned (macro)')
    axL.plot(np.arange(1, len(micro_cum) + 1), micro_cum, color='#1F6FB2', lw=2,
             label='beyond-anatomy (micro)')
    axL.axvline(k_macro, color='#C1440E', ls='--', lw=1.3)
    axL.axvline(k_micro, color='#1F6FB2', ls='--', lw=1.3)
    axL.scatter([k_macro], [macro_cum[k_macro - 1]], color='#C1440E', s=70, zorder=5,
                label=f'macro elbow = {k_macro}')
    axL.scatter([k_micro], [micro_cum[k_micro - 1]], color='#1F6FB2', s=70, zorder=5,
                label=f'micro elbow = {k_micro}')
    axL.set_xlabel('rules added (ranked by that component)')
    axL.set_ylabel('cumulative information (bits)')
    axL.set_title('Rule selection by the information split')
    axL.legend(fontsize=8, frameon=False); axL.set_facecolor('white')

    # Panel B: per-rule scatter, colored by membership
    x = ranked_rules['ig_micro'].values
    y = ranked_rules['ig_macro'].values
    cat = np.where(in_macro & in_micro, 'both',
          np.where(in_macro, 'macro', np.where(in_micro, 'micro', 'neither')))
    colors = {'both': '#7B3294', 'macro': '#C1440E', 'micro': '#1F6FB2', 'neither': '#c9c9c9'}
    for c in ['neither', 'micro', 'macro', 'both']:
        m = cat == c
        axR.scatter(x[m], y[m], s=45, color=colors[c], edgecolor='white', linewidth=0.4,
                    zorder=3, label=f'{c} ({int(m.sum())})')
    axR.axhline(0, color='#ddd', lw=0.8); axR.axvline(0, color='#ddd', lw=0.8)
    axR.set_xlabel('ig_micro  (beyond-anatomy, bits)')
    axR.set_ylabel('ig_macro  (anatomy-aligned, bits)')
    axR.set_title('Where each rule sits in the split')
    axR.legend(fontsize=8, frameon=False); axR.set_facecolor('white')

    fig.suptitle(f'FOV {fov_id} - macro / micro rule selection', y=1.02, fontsize=13)
    plt.tight_layout(); plt.show()

def plot_annealing_convergence(energy_history):
    """Plots the objective score over the course of the simulation."""
    fig, ax = plt.subplots(figsize=(10, 4.5), facecolor="white")
    ax.plot(energy_history, color="#457B9D", linewidth=1.2)
    ax.set_xlabel("Iteration", fontsize=11)
    ax.set_ylabel("Objective (E)", fontsize=11)
    ax.set_title("Simulated Annealing Convergence", fontsize=12)
    ax.set_facecolor("white")
    plt.tight_layout()
    plt.show()

def plot_delta_energy_distribution(delta_energy_history):
    """Histogram of every proposed swap's raw delta_energy (before accept/reject)."""
    fig, ax = plt.subplots(figsize=(9, 4), facecolor="white")
    ax.hist(delta_energy_history, bins=80, color="#457B9D", edgecolor="white", linewidth=0.3)
    ax.axvline(0, color="#E63946", linestyle="--", linewidth=1.2)
    ax.set_xlabel("Delta Energy per proposed swap", fontsize=11)
    ax.set_ylabel("Count", fontsize=11)
    ax.set_title("Distribution of Proposed Swap Sizes", fontsize=12)
    ax.set_facecolor("white")
    plt.tight_layout()
    plt.show()

