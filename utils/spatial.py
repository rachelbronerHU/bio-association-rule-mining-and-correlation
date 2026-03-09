import logging
import numpy as np
from sklearn.neighbors import NearestNeighbors
from collections import Counter

logger = logging.getLogger(__name__)

MIN_CELLS = 2
DOMINANCE_THRESHOLD = 0.9


def is_dominated(cells, threshold=DOMINANCE_THRESHOLD):
    """Returns True if any single cell type makes up > threshold of the batch."""
    if len(cells) == 0:
        return False
    counts = Counter(cells)
    most_common = counts.most_common(1)[0][1]
    return (most_common / len(cells)) > threshold


def get_neighborhoods(coords, method, config):
    """
    Calculates structural neighborhoods (indices) based on the method.

    Returns a list of 'patches':
      BAG, WINDOW, GRID : List of index arrays [ [idx1, idx2], ... ]
      CN, KNN_R         : List of tuples (center_idx, neighbor_indices)
    """
    if len(coords) == 0:
        return []

    neighborhoods = []

    if method == "BAG":
        radius = config["RADIUS"]
        nn = NearestNeighbors(radius=radius, n_jobs=-1).fit(coords)
        neighbors_idx = nn.radius_neighbors(coords, return_distance=False)

        lens = [len(i) for i in neighbors_idx]
        if lens:
            logger.info(f"[BAG] Radius {radius}: Neighbors Min {np.min(lens)}, Median {np.median(lens):.1f}, Avg {np.mean(lens):.1f}, Max {np.max(lens)}")

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
        # Clamp to available points — crashes if n_neighbors > n_samples
        nn = NearestNeighbors(n_neighbors=min(k + 1, len(coords)), n_jobs=-1).fit(coords)
        dists, indices = nn.kneighbors(coords)

        for center_i, (nbr_dists, nbr_idxs) in enumerate(zip(dists, indices)):
            valid = nbr_dists <= radius
            valid_idxs = nbr_idxs[valid]
            neighborhoods.append((center_i, valid_idxs))

    elif method == "WINDOW":
        target_cells = config.get("TARGET_CELLS", 25)
        step_fraction = config.get("WINDOW_STEP_FRACTION", 0.5)

        x_min, x_max = coords[:, 0].min(), coords[:, 0].max()
        y_min, y_max = coords[:, 1].min(), coords[:, 1].max()

        area = (x_max - x_min) * (y_max - y_min)
        density = len(coords) / area if area > 0 else 1
        window_size = np.sqrt(target_cells / density)
        step_size = window_size * step_fraction

        x_starts = np.arange(x_min, x_max - window_size + step_size, step_size) if x_max - window_size + step_size >= x_min else [x_min]
        y_starts = np.arange(y_min, y_max - window_size + step_size, step_size) if y_max - window_size + step_size >= y_min else [y_min]

        logger.info(f"WINDOW Config: target_cells={target_cells}, density={density:.2f}, window_size={window_size:.1f}, step_size={step_size:.1f}, x_steps={len(x_starts)}, y_steps={len(y_starts)}")

        for x_s in x_starts:
            for y_s in y_starts:
                x_e, y_e = x_s + window_size, y_s + window_size
                mask = (coords[:, 0] >= x_s) & (coords[:, 0] < x_e) & \
                       (coords[:, 1] >= y_s) & (coords[:, 1] < y_e)
                indices = np.where(mask)[0]
                if len(indices) > 0:
                    neighborhoods.append(indices)

    elif method == "GRID":
        size = config["GRID_WINDOW_SIZE"]

        x_min, y_min = coords[:, 0].min(), coords[:, 1].min()
        x_max, y_max = coords[:, 0].max(), coords[:, 1].max()
        # Start grid at x_min/y_min, not 0 — handles negative coordinates and
        # avoids iterating empty tiles when coordinates are far from the origin.
        x_steps = int((x_max - x_min) / size) + 1
        y_steps = int((y_max - y_min) / size) + 1

        for i in range(x_steps):
            for j in range(y_steps):
                x_s, y_s = x_min + i * size, y_min + j * size
                x_e, y_e = x_s + size, y_s + size
                mask = (coords[:, 0] >= x_s) & (coords[:, 0] < x_e) & \
                       (coords[:, 1] >= y_s) & (coords[:, 1] < y_e)
                indices = np.where(mask)[0]
                if len(indices) > 0:
                    neighborhoods.append(indices)

    return neighborhoods
