"""How each cell of an FOV counts for a pairwise rule, from the mining's own patches and weights."""
import json
from pathlib import Path

import numpy as np
import pandas as pd
from spatial_association_rules import Method, Settings, Weighting
from spatial_association_rules import transactions as tr

import data_helper as dh


def run_settings():
    """The mining settings of the run the results come from, read from its run_config.json."""
    saved = json.loads((Path(dh.RESULT_CSV_PATH).parent / "run_config.json").read_text())["settings"]
    saved.pop("decay_distance")
    return Settings(**saved | {"weighting": Weighting(saved["weighting"]), "method": Method(saved["method"])})


SETTINGS = run_settings()


def rule_scores(cells, rule):
    """For one FOV's `cells` and `rule` (centre, neighbour), with the mining's patches and weights:
    per centre cell, how much of the neighbour type its patch holds (0 to 1, 0 when the mining drops
    its patch); per neighbour cell, its weight in the closest centre cell's patch, 0 when in none;
    NaN for every other cell."""
    center, neighbor = rule
    coords, labels = cells[["x_um", "y_um"]].to_numpy(), cells["cell type"].astype(str).to_numpy()
    patches = [patch for patch in tr.measure_patches(tr.find_patches(coords, SETTINGS), coords, SETTINGS)
               if not tr.is_crowded_by_one_type(labels[patch.members], SETTINGS.max_one_type_share)]
    found, _ = tr.build_transactions(patches, labels, SETTINGS)
    held = np.array([transaction.get(tr.item_of(neighbor, tr.NEIGHBOR), 0.0) for transaction in found])
    centers = np.array([patch.center for patch in patches])
    of_center = labels[centers] == center
    score = np.where(np.isin(labels, rule), 0.0, np.nan)
    score[centers[of_center]] = held[of_center]
    for patch in (patch for patch, mine in zip(patches, of_center) if mine):
        near = labels[patch.neighbors] == neighbor
        score[patch.neighbors[near]] = np.fmax(score[patch.neighbors[near]], patch.weights[near])
    return pd.Series(score, index=cells.index)
