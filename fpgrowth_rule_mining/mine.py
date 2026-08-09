"""The entry point: coordinates and labels in, rules out."""

from dataclasses import dataclass, field
from typing import List

import numpy as np
import pandas as pd

from .attraction import mine_attraction
from .avoidance import mine_avoidance
from .rules import drop_rare_labels, empty_rules, labels_with_enough_cells, weight_matrix
from .settings import Settings
from .validation.significance import p_values_for
from .transactions import Patch, build_transactions, find_patches, measure_patches


def mine_rules(transactions, settings: Settings) -> pd.DataFrame:
    """Transactions in, rules out: both searches, one frame."""
    if not transactions:
        return empty_rules()

    matrix, item_index = weight_matrix(transactions)
    found = [mine_attraction(transactions, matrix, item_index, settings)]
    if settings.include_avoidance_rules:
        found.append(mine_avoidance(matrix, item_index, settings))

    found = [frame for frame in found if not frame.empty]
    return pd.concat(found, ignore_index=True) if found else empty_rules()


@dataclass
class Result:
    """What one run produced, plus what it needs to test those rules later."""

    rules: pd.DataFrame
    stats: dict
    patches: List[Patch] = field(repr=False)
    labels: np.ndarray = field(repr=False)
    settings: Settings = field(repr=False)

    def add_p_values(self, n_shuffles, rules=None, random_seed=None, labels_kept_fixed=()):
        """
        Test rules against shuffled labels and attach a raw p_value.

        Defaults to this run's rules, but takes any subset. Nothing is corrected.
        """
        rules = (self.rules if rules is None else rules).copy()
        rules["p_value"] = p_values_for(
            rules, self.patches, self.labels, self.settings,
            n_shuffles, random_seed, labels_kept_fixed,
        )
        return rules


def mine(coords, labels, settings: Settings) -> Result:
    """
    Mine spatial association rules.

    coords:  (n_cells, 2) positions
    labels:  (n_cells,) one label per cell, one label per cell type

    No significance testing here — call result.add_p_values() for that.
    """
    coords = np.asarray(coords, dtype=float)
    labels = np.asarray(labels, dtype=object)
    if len(coords) != len(labels):
        raise ValueError(f"coords has {len(coords)} rows but labels has {len(labels)}")

    patches = find_patches(coords, settings)
    measured = measure_patches(patches, coords, settings)
    transactions, stats = build_transactions(measured, labels, settings)
    stats["patches_found"] = len(patches)

    # Rare labels go first: the shuffle test after them is what the run pays for.
    rules = drop_rare_labels(mine_rules(transactions, settings), labels, settings)
    # Lets a later step tell "the cell type was not here" from "tested and failed".
    stats["labels_with_enough_cells"] = labels_with_enough_cells(labels, settings)

    return Result(rules=rules, stats=stats, patches=measured, labels=labels, settings=settings)
