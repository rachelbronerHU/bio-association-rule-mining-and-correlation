import logging
import numpy as np
from sklearn.neighbors import NearestNeighbors
from collections import Counter

logger = logging.getLogger("transactions")

# Defaults
MIN_CELLS = 2
DOMINANCE_THRESHOLD = 0.9

def is_dominated(cells, threshold=DOMINANCE_THRESHOLD):
    """ Returns True if any single cell type makes up > threshold of the batch """
    if len(cells) == 0: return False
    # If cells are strings (types)
    counts = Counter(cells)
    most_common = counts.most_common(1)[0][1]
    return (most_common / len(cells)) > threshold

def get_neighborhoods(coords, method, config):
    """
    Calculates the structural neighborhoods (indices) based on the method.
    Returns a list of 'patches'. 
    For BAG, WINDOW, GRID: List of lists of indices [ [idx1, idx2], ... ]
    For CN, KNN_R: List of tuples (center_idx, [neighbor_indices])
    """
    if len(coords) == 0: return []
    
    neighborhoods = []
    
    if method == "BAG":
        radius = config["RADIUS"]
        nn = NearestNeighbors(radius=radius, n_jobs=-1).fit(coords)
        neighbors_idx = nn.radius_neighbors(coords, return_distance=False)
        
        # Log Stats
        lens = [len(i) for i in neighbors_idx]
        if lens:
            min_nbrs = np.min(lens)
            median_nbrs = np.median(lens)
            avg_nbrs = np.mean(lens)
            max_nbrs = np.max(lens)
            logger.info(f"[BAG] Radius {radius}: Neighbors Min {min_nbrs}, Median {median_nbrs:.1f}, Avg {avg_nbrs:.1f}, Max {max_nbrs}")

        for idxs in neighbors_idx:
            neighborhoods.append(idxs)
            
    elif method == "CN":
        radius = config["RADIUS"]
        nn = NearestNeighbors(radius=radius, n_jobs=-1).fit(coords)
        neighbors_idx = nn.radius_neighbors(coords, return_distance=False)
        for center_i, idxs in enumerate(neighbors_idx):
            neighborhoods.append((center_i, idxs))
            
    elif method == "KNN_R":
        k = config["K_NEIGHBORS"]
        radius = config["RADIUS"]
        nn = NearestNeighbors(n_neighbors=k+1, n_jobs=-1).fit(coords)
        dists, indices = nn.kneighbors(coords)
        
        for center_i, (nbr_dists, nbr_idxs) in enumerate(zip(dists, indices)):
            valid = nbr_dists <= radius
            valid_idxs = nbr_idxs[valid]
            neighborhoods.append((center_i, valid_idxs))
            
    elif method == "WINDOW":
        # Dynamic Logic
        target_cells = config.get("TARGET_CELLS", 25)
        step_fraction = config.get("WINDOW_STEP_FRACTION", 0.5)
        
        x_min, x_max = coords[:, 0].min(), coords[:, 0].max()
        y_min, y_max = coords[:, 1].min(), coords[:, 1].max()

        num_cells = len(coords)

        area = (x_max - x_min) * (y_max - y_min)
        density = num_cells / area if area > 0 else 1
        window_size = np.sqrt(target_cells / density)
        step_size = window_size * step_fraction
        
        # Ensure at least one step
        if x_max - window_size + step_size < x_min:
             x_starts = [x_min]
        else:
             x_starts = np.arange(x_min, x_max - window_size + step_size, step_size)
             
        if y_max - window_size + step_size < y_min:
             y_starts = [y_min]
        else:
             y_starts = np.arange(y_min, y_max - window_size + step_size, step_size)
        
        logger.info(f"WINDOW Config: target_cells={target_cells}, density={density:.2f}, window_size={window_size:.1f}, step_size={step_size:.1f}, x_steps={len(x_starts)}, y_steps={len(y_starts)}")
        for x_s in x_starts:
            for y_s in y_starts:
                x_e, y_e = x_s + window_size, y_s + window_size
                mask = (coords[:,0] >= x_s) & (coords[:,0] < x_e) & \
                       (coords[:,1] >= y_s) & (coords[:,1] < y_e)
                indices = np.where(mask)[0]
                if len(indices) > 0:
                    neighborhoods.append(indices)

    elif method == "GRID":
        size = config["GRID_WINDOW_SIZE"]
        step = size # Non-overlapping
        
        x_max, y_max = coords[:,0].max(), coords[:,1].max()
        x_steps = int(x_max / step) + 1
        y_steps = int(y_max / step) + 1
        
        for i in range(x_steps):
            for j in range(y_steps):
                x_s, y_s = i * step, j * step
                x_e, y_e = x_s + size, y_s + size
                mask = (coords[:,0] >= x_s) & (coords[:,0] < x_e) & \
                       (coords[:,1] >= y_s) & (coords[:,1] < y_e)
                indices = np.where(mask)[0]
                if len(indices) > 0:
                    neighborhoods.append(indices)
                    
    return neighborhoods

def build_transactions_from_neighborhoods(neighborhoods, cell_types, method, config=None):
    """
    Builds transactions from neighborhoods, optionally filtering rare cell types.
    
    Args:
        neighborhoods: List of neighborhoods (structure depends on method)
        cell_types: Array of cell type labels
        method: Mining method (BAG, CN, KNN_R, etc.)
        config: Optional config dict with MIN_CELL_TYPE_FREQUENCY and MIN_SUPPORT
    
    Returns:
        transactions: List of transactions (each is a list of items)
        stats: Dict with statistics about the transactions
    """
    # Calculate rare cell type threshold if config provided
    rare_cell_types = set()
    if config is not None and config.get("MIN_CELL_TYPE_FREQUENCY", 0) > 0:
        from collections import Counter
        cell_type_counts = Counter(cell_types)
        n_cells = len(cell_types)
        
        # Adaptive threshold: max(absolute minimum, percentage * total cells)
        min_absolute = config.get("MIN_CELL_TYPE_FREQUENCY", 5)
        min_percentage = config.get("MIN_SUPPORT", 0.01)
        min_threshold = max(min_absolute, int(min_percentage * n_cells))
        
        # Identify rare cell types
        rare_cell_types = {ct for ct, count in cell_type_counts.items() if count < min_threshold}
        
        if rare_cell_types:
            logger.info(f"Filtering {len(rare_cell_types)} rare cell types (threshold={min_threshold}): {sorted(rare_cell_types)[:5]}...")
    
    transactions = []
    sizes = []
    cell_counts = []
    orig_count = len(neighborhoods)
    filtered_rare_count = 0
    
    for item in neighborhoods:
        # Resolve indices based on method structure
        if method in ["CN", "KNN_R"]:
            center_i, idxs = item
            if len(idxs) < MIN_CELLS: continue
            
            # Check dominance
            raw_types = cell_types[idxs]
            if is_dominated(raw_types): continue
            
            # Filter out rare cell types from this transaction
            center_type = cell_types[center_i]
            if center_type in rare_cell_types:
                filtered_rare_count += 1
                continue  # Skip if center cell is rare
            
            # Build transaction, excluding rare neighbors
            center = f"{center_type}_CENTER"
            neighbors = [f"{cell_types[n]}_NEIGHBOR" for n in idxs 
                        if n != center_i and cell_types[n] not in rare_cell_types]
            
            if not neighbors:  # If all neighbors were rare
                filtered_rare_count += 1
                continue
            
            trans = [center] + list(set(neighbors))
            transactions.append(trans)
            sizes.append(len(trans))
            cell_counts.append(len(idxs))
        else:
            # BAG, WINDOW, GRID
            idxs = item
            if len(idxs) < MIN_CELLS: continue
            
            raw_types = cell_types[idxs]
            if is_dominated(raw_types): continue
            
            # Filter out rare cell types from this transaction
            filtered_types = [ct for ct in raw_types if ct not in rare_cell_types]
            
            if not filtered_types:  # If all types were rare
                filtered_rare_count += 1
                continue
            
            trans = list(set(filtered_types))
            transactions.append(trans)
            sizes.append(len(trans))
            cell_counts.append(len(idxs))
    
    stats = {
        "sizes": sizes, 
        "cell_counts": cell_counts, 
        "orig": orig_count, 
        "kept": len(transactions),
        "filtered_rare": filtered_rare_count
    }
    
    return transactions, stats

# Wrappers for backward compatibility (used by worker_task if not refactored immediately)
def get_bag(coords, cell_types, radius):
    config = {"RADIUS": radius}
    nb = get_neighborhoods(coords, "BAG", config)
    return build_transactions_from_neighborhoods(nb, cell_types, "BAG", config)

def get_cn(coords, cell_types, radius):
    config = {"RADIUS": radius}
    nb = get_neighborhoods(coords, "CN", config)
    return build_transactions_from_neighborhoods(nb, cell_types, "CN", config)

def get_knn_r(coords, cell_types, k, radius):
    config = {"K_NEIGHBORS": k, "RADIUS": radius}
    nb = get_neighborhoods(coords, "KNN_R", config)
    return build_transactions_from_neighborhoods(nb, cell_types, "KNN_R", config)

def get_window(coords, cell_types, target_cells, step_fraction):
    config = {"TARGET_CELLS": target_cells, "STEP_FRACTION": step_fraction}
    nb = get_neighborhoods(coords, "WINDOW", config)
    return build_transactions_from_neighborhoods(nb, cell_types, "WINDOW", config)

def get_grid(coords, cell_types, size):
    config = {"GRID_WINDOW_SIZE": size}
    nb = get_neighborhoods(coords, "GRID", config)
    return build_transactions_from_neighborhoods(nb, cell_types, "GRID", config)