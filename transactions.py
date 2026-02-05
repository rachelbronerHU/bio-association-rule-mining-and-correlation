import numpy as np
from sklearn.neighbors import NearestNeighbors
from collections import Counter

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
        target_cells = config.get("TARGET_CELLS", 30)
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

def build_transactions_from_neighborhoods(neighborhoods, cell_types, method):
    transactions = []
    sizes = []
    orig_count = len(neighborhoods)
    
    for item in neighborhoods:
        # Resolve indices based on method structure
        if method in ["CN", "KNN_R"]:
            center_i, idxs = item
            if len(idxs) < MIN_CELLS: continue
            
            # Check dominance
            raw_types = cell_types[idxs]
            if is_dominated(raw_types): continue
            
            center = f"{cell_types[center_i]}_CENTER"
            neighbors = [f"{cell_types[n]}_NEIGHBOR" for n in idxs if n != center_i]
            trans = [center] + list(set(neighbors))
            transactions.append(trans)
            sizes.append(len(trans))
        else:
            # BAG, WINDOW, GRID
            idxs = item
            if len(idxs) < MIN_CELLS: continue
            
            raw_types = cell_types[idxs]
            if is_dominated(raw_types): continue
            
            trans = list(set(raw_types))
            transactions.append(trans)
            sizes.append(len(trans))
            
    return transactions, {"sizes": sizes, "orig": orig_count, "kept": len(transactions)}

# Wrappers for backward compatibility (used by worker_task if not refactored immediately)
def get_bag(coords, cell_types, radius):
    config = {"RADIUS": radius}
    nb = get_neighborhoods(coords, "BAG", config)
    return build_transactions_from_neighborhoods(nb, cell_types, "BAG")

def get_cn(coords, cell_types, radius):
    config = {"RADIUS": radius}
    nb = get_neighborhoods(coords, "CN", config)
    return build_transactions_from_neighborhoods(nb, cell_types, "CN")

def get_knn_r(coords, cell_types, k, radius):
    config = {"K_NEIGHBORS": k, "RADIUS": radius}
    nb = get_neighborhoods(coords, "KNN_R", config)
    return build_transactions_from_neighborhoods(nb, cell_types, "KNN_R")

def get_window(coords, cell_types, target_cells, step_fraction):
    config = {"TARGET_CELLS": target_cells, "STEP_FRACTION": step_fraction}
    nb = get_neighborhoods(coords, "WINDOW", config)
    return build_transactions_from_neighborhoods(nb, cell_types, "WINDOW")

def get_grid(coords, cell_types, size):
    config = {"GRID_WINDOW_SIZE": size}
    nb = get_neighborhoods(coords, "GRID", config)
    return build_transactions_from_neighborhoods(nb, cell_types, "GRID")