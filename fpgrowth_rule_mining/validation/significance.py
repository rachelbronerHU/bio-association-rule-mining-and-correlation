"""
Is a rule more than an accident of how common its cell types are?

Hold the tissue still, shuffle the labels between cells, and see how often the rule
still passes. See README, "add_p_values".
"""

import logging
import time
import zlib
from typing import NamedTuple

import numpy as np
from scipy import sparse

from ..attraction import attracts
from ..avoidance import avoids
from ..rules import AVOIDS, metrics, support_of_many
from ..transactions import CENTER, NEIGHBOR, item_of

logger = logging.getLogger(__name__)


def seed_for(base_seed, sample_id):
    """
    A seed per sample: re-runs match, and samples stay independent.

    crc32, not hash(), which Python randomises per process.
    """
    if base_seed is None:
        return None
    return base_seed + zlib.crc32(str(sample_id).encode())


def p_values_for(rules, patches, labels, settings, n_shuffles, random_seed=None, labels_kept_fixed=()):
    """The raw p-value for each rule, in order. Nothing is corrected here."""
    if rules.empty or n_shuffles <= 0:
        return np.ones(len(rules))
    if "kind" not in rules.columns:
        raise ValueError("rules need a 'kind' column: a rule is tested against the "
                         "thresholds of the search that found it, and this cannot guess which")
    if (rules["kind"] == AVOIDS).any() and settings.avoidance_max_lift is None:
        raise ValueError("these rules include avoidance, but the settings have no "
                         "avoidance_max_lift to test them against. Testing them by a "
                         "different threshold than the one that found them would make "
                         "the p-values answer a different question")

    labels = np.asarray(labels, dtype=object)
    names = sorted({str(label) for label in labels})
    column_of = {name: i for i, name in enumerate(names)}

    # One row per cell, marking its label. Shuffling permutes these rows.
    # float64 to match the mining matrix, so both judge a borderline rule the same.
    cell_labels = np.zeros((len(labels), len(names)), dtype=float)
    for cell, label in enumerate(labels):
        cell_labels[cell, column_of[str(label)]] = 1.0

    centers, neighbors, membership, patch_sizes = _adjacency(patches, len(labels))
    # A transaction column per item: centres first, then neighbours.
    item_index = {}
    for name, column in column_of.items():
        item_index[item_of(name, CENTER)] = column
        item_index[item_of(name, NEIGHBOR)] = column + len(names)

    movable = np.arange(len(labels))[~_held_fixed(labels, labels_kept_fixed)]
    _check_enough_moves(movable.size, len(labels), labels_kept_fixed)
    rng = np.random.default_rng(random_seed)
    layout = _rule_columns(rules, item_index)   # identical every shuffle

    logger.info(f"Shuffling labels {n_shuffles} times against {len(rules)} rules...")
    started = time.time()

    survived = np.zeros(len(rules))
    for i in range(n_shuffles):
        order = np.arange(len(labels))
        if movable.size > 1:
            order[movable] = rng.permutation(movable)
        shuffled = cell_labels[order, :]

        # Rebuild the transaction weights, capped as a real transaction is.
        transactions = np.hstack([
            np.minimum(_dense(centers @ shuffled), 1.0),
            np.minimum(_dense(neighbors @ shuffled), 1.0),
        ])
        # Drop crowded patches as the real run did, judged on the shuffled labels:
        # the null repeats the procedure, not the outcome.
        transactions = transactions[not_crowded(membership, patch_sizes, shuffled,
                                                settings.max_one_type_share)]
        survived += survives_shuffle(layout, transactions, settings)

        if i == 0 or (i + 1) % 100 == 0 or i == n_shuffles - 1:
            logger.info(f"  shuffle {i + 1}/{n_shuffles}")

    logger.info(f"Shuffling took {time.time() - started:.2f}s")

    # +1 top and bottom: never surviving is not proof, just "under 1 in n".
    p_values = (survived + 1) / (n_shuffles + 1)
    # A rule naming a cell type this sample lacks was never tested. No evidence is 1.0.
    p_values[~layout.usable] = 1.0
    return p_values


def _adjacency(patches, n_cells):
    """
    Who is where: one row per patch, one column per cell.

    centers/neighbors carry the weights. membership is plain 1s, for counting labels.
    """
    center_rows, center_cols = [], []
    neighbor_rows, neighbor_cols, neighbor_weights = [], [], []
    member_rows, member_cols = [], []

    for row, patch in enumerate(patches):
        center_rows.append(row)
        center_cols.append(patch.center)
        for neighbor, weight in zip(patch.neighbors, patch.weights):
            neighbor_rows.append(row)
            neighbor_cols.append(neighbor)
            neighbor_weights.append(weight)
        for member in patch.members:
            member_rows.append(row)
            member_cols.append(member)

    shape = (len(patches), n_cells)
    membership = sparse.csr_matrix((np.ones(len(member_rows)), (member_rows, member_cols)), shape=shape)
    return (
        sparse.csr_matrix((np.ones(len(center_rows)), (center_rows, center_cols)), shape=shape),
        sparse.csr_matrix((neighbor_weights, (neighbor_rows, neighbor_cols)), shape=shape),
        membership,
        np.asarray([len(p.members) for p in patches], dtype=float),
    )


def not_crowded(membership, patch_sizes, cell_labels, max_one_type_share):
    """
    Which patches are mixed enough to keep. Same rule as build_transactions, in bulk.
    """
    counts = _dense(membership @ cell_labels)
    return counts.max(axis=1) <= max_one_type_share * patch_sizes


def _held_fixed(labels, patterns):
    """
    Cells whose label never moves.

    'Name' matches exactly. 'Name*' matches any label starting with Name.
    """
    fixed = np.zeros(len(labels), dtype=bool)
    for pattern in patterns:
        pattern = str(pattern).strip()
        if pattern == "*":
            raise ValueError("labels_kept_fixed of '*' would hold every label still, leaving nothing to shuffle")
        if not pattern:
            continue
        for cell, label in enumerate(labels):
            label = str(label)
            if label.startswith(pattern[:-1]) if pattern.endswith("*") else label == pattern:
                fixed[cell] = True

    if patterns and not fixed.any():
        logger.warning(f"No cell matched labels_kept_fixed={tuple(patterns)}")
    return fixed


ENOUGH_MOVABLE = 0.1     # below this share still free to move, warn


def _check_enough_moves(movable, n_cells, patterns):
    """Pin too many labels and the shuffled tissue is the real one, so refuse."""
    if movable <= 1:
        raise ValueError(
            f"labels_kept_fixed={tuple(patterns)} leaves {movable} of {n_cells} cells free "
            f"to move, so no shuffle changes anything and every p-value would be 1.0. "
            f"Pin fewer labels, or pass n_shuffles=0 if you meant not to test"
        )
    if movable < ENOUGH_MOVABLE * n_cells:
        logger.warning(
            f"labels_kept_fixed={tuple(patterns)} leaves only {movable} of {n_cells} cells "
            f"({movable / n_cells:.1%}) free to move. The shuffled tissue barely differs "
            f"from the real one, so the p-values will be very conservative"
        )


class _Layout(NamedTuple):
    """Where each rule finds its supports, and which judge it answers to."""

    sized: list          # (positions, columns) per group size, measured in one array
    n_groups: int
    ant_at: np.ndarray   # index into the measured supports, per rule
    con_at: np.ndarray
    joint_at: np.ndarray
    usable: np.ndarray   # False for a rule naming an item this sample does not have
    avoiding: np.ndarray


def _rule_columns(rules, item_index):
    """
    The distinct column groups to measure, and where each rule reads its own.

    Rules overlap heavily, so each group is measured once per shuffle. The layout is
    the same every shuffle, so it is worked out once.
    """
    known, groups = {}, []            # column tuple -> its place in the answers

    def place(columns):
        key = tuple(sorted(columns))
        if key not in known:
            known[key] = len(known)
            groups.append(key)
        return known[key]

    ant_at, con_at, joint_at, usable = [], [], [], []
    for rule in rules.itertuples():
        ant = [item_index.get(item) for item in rule.antecedents]
        con = [item_index.get(item) for item in rule.consequents]
        if None in ant or None in con:
            ant_at.append(0), con_at.append(0), joint_at.append(0), usable.append(False)
            continue
        ant_at.append(place(ant))
        con_at.append(place(con))
        joint_at.append(place(ant + con))
        usable.append(True)

    # Groups of the same size can be measured together in one array.
    by_size = {}
    for position, columns in enumerate(groups):
        by_size.setdefault(len(columns), []).append((position, columns))
    sized = [(np.asarray([p for p, _ in members]), np.asarray([c for _, c in members]))
             for members in by_size.values()]

    return _Layout(sized, len(groups), np.asarray(ant_at), np.asarray(con_at),
                   np.asarray(joint_at), np.asarray(usable),
                   (rules["kind"] == AVOIDS).to_numpy())


def _supports(layout, transactions):
    """Support of every distinct column group, then read off per rule."""
    measured = np.zeros(layout.n_groups)
    for positions, group_columns in layout.sized:
        measured[positions] = support_of_many(transactions, group_columns)

    return measured[layout.ant_at], measured[layout.con_at], measured[layout.joint_at]


def survives_shuffle(layout, transactions, settings):
    """Which rules still pass their own thresholds here. Each judged by its own kind."""
    held = np.zeros(len(layout.usable))
    # Nothing usable means nothing was measured, so nothing to read off.
    if transactions.size == 0 or not layout.usable.any():
        return held

    n = transactions.shape[0]
    ant_sup, con_sup, joint = _supports(layout, transactions)

    # Both sides must exist before the thresholds mean anything.
    testable = layout.usable & (ant_sup > 0) & (con_sup > 0)
    if not testable.any():
        return held

    # Both judges run over every rule; each rule takes its own answer.
    measures = metrics(joint, ant_sup, con_sup)
    held[testable] = np.where(layout.avoiding,
                              avoids(settings, joint, ant_sup, con_sup, measures, n),
                              attracts(settings, joint, ant_sup, con_sup, measures, n))[testable]
    return held



def _dense(matrix):
    return matrix.toarray() if sparse.issparse(matrix) else matrix
