"""
Search for cell types that keep apart. See README, "Attraction and avoidance".
"""

import logging
from itertools import combinations
from math import comb

import numpy as np

from .rules import AVOIDS, frame_of, rules_from, splits_of, support_of_many
from .transactions import is_center

logger = logging.getLogger(__name__)

MOST_COMBINATIONS = 2_000_000    # refuse to start above this many
BATCH = 100_000                  # combinations held as tuples at once


def enough_to_judge_avoidance(settings, ant_support, con_support, n_transactions):
    """Enough patches to measure a rate on, and enough expected meetings to miss."""
    return ((ant_support * n_transactions >= settings.min_patches)
            & (ant_support * con_support * n_transactions >= settings.avoidance_min_expected_meetings))


def is_weak_enough(settings, lift, leverage):
    """The avoidance thresholds."""
    weak = lift <= settings.avoidance_max_lift
    if settings.avoidance_max_leverage is not None:
        weak = weak & (leverage <= settings.avoidance_max_leverage)
    return weak


def avoids(settings, support, ant_support, con_support, measures, n_transactions):
    """Do these cell types clearly keep apart? Same signature as attracts()."""
    _, lift, leverage, _ = measures
    return ((lift < 1.0)
            & enough_to_judge_avoidance(settings, ant_support, con_support, n_transactions)
            & is_weak_enough(settings, lift, leverage))


def items_worth_combining(matrix, item_index, settings, n_transactions):
    """
    Items worth combining, and the support of each. A speed filter only.

    The bar is the lower of the two requirements, since a consequent item needs no
    patches of its own — a common antecedent supplies the expected meetings.
    """
    if matrix.size == 0:
        return [], {}

    bar = min(settings.min_patches, settings.avoidance_min_expected_meetings) / n_transactions
    supports = matrix.sum(axis=0) / n_transactions
    kept = sorted(item for item, column in item_index.items() if supports[column] >= bar)
    return kept, {item: float(supports[item_index[item]]) for item in kept}


def by_role(items):
    """The items split into centres and neighbours."""
    return ([item for item in items if is_center(item)],
            [item for item in items if not is_center(item)])


def candidate_sets(items, max_items):
    """Every combination that could be read as a rule: exactly one centre."""
    centres, neighbours = by_role(items)
    for size in range(2, max_items + 1):
        for centre in centres:
            for combo in combinations(neighbours, size - 1):
                yield frozenset((centre,) + combo)


def combinations_to_measure(items, size):
    """Every combination of this size a rule could look up: at most one centre."""
    centres, neighbours = by_role(items)
    yield from combinations(neighbours, size)
    for centre in centres:
        for combo in combinations(neighbours, size - 1):
            yield (centre,) + combo


def how_many_to_measure(n_centres, n_neighbours, max_items):
    """How many combinations_to_measure will produce, without building them."""
    return sum(comb(n_neighbours, size) + n_centres * comb(n_neighbours, size - 1)
               for size in range(1, max_items + 1))


def subset_supports(items, single_supports, matrix, item_index, max_items):
    """Support of every combination a rule could look up. Single items are given."""
    supports = {frozenset([item]): support for item, support in single_supports.items()}

    def measure(batch):
        columns = np.asarray([[item_index[item] for item in combo] for combo in batch])
        for combo, support in zip(batch, support_of_many(matrix, columns)):
            supports[frozenset(combo)] = float(support)

    for size in range(2, max_items + 1):
        batch = []
        for combo in combinations_to_measure(items, size):
            batch.append(combo)
            if len(batch) >= BATCH:
                measure(batch)
                batch = []
        if batch:
            measure(batch)
    return supports


def mine_avoidance(matrix, item_index, settings):
    """Every combination worth trying, split into rules, keeping what keeps apart."""
    n = matrix.shape[0]
    if n == 0:
        return frame_of([])

    items, single_supports = items_worth_combining(matrix, item_index, settings, n)
    if len(items) < 2:
        return frame_of([])

    max_items = min(settings.max_items_per_rule, len(items))
    centres, neighbours = by_role(items)
    planned = how_many_to_measure(len(centres), len(neighbours), max_items)
    if planned > MOST_COMBINATIONS:
        raise ValueError(
            f"the avoidance search would measure {planned:,} combinations of "
            f"{len(items)} items at max_items_per_rule={max_items}, over the "
            f"{MOST_COMBINATIONS:,} it will attempt. The count grows with both, so "
            f"lower max_items_per_rule, or raise avoidance_min_expected_meetings to "
            f"leave fewer items worth combining"
        )

    supports = subset_supports(items, single_supports, matrix, item_index, max_items)

    splits = []

    logger.info(f"Mine_avoidance - about to test {planned} itemsets!")

    for itemset in candidate_sets(items, max_items):
        support = supports[itemset]
        for antecedent, consequent in splits_of(itemset):
            ant_support, con_support = supports[antecedent], supports[consequent]
            if ant_support <= 0 or con_support <= 0:
                continue
            splits.append((antecedent, consequent, support, ant_support, con_support))

    rules = rules_from(splits, settings, n, avoids, AVOIDS)
    logger.debug(f"Avoidance: {len(items)} of {len(item_index)} items worth combining, "
                 f"{len(supports)} combinations measured, {len(rules)} rules")
    return rules
