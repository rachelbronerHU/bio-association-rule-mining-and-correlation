"""Shared rule-state tables and permutation tests for differential-rule notebooks."""

from itertools import combinations

import numpy as np
import pandas as pd
from statsmodels.stats.multitest import multipletests

import data_helper as dh


def state_tables(rules, cells, fovs, min_cells=20):
    """Return all rule states, eligibility, and states masked by eligibility."""
    states = (
        rules.pivot(index="Clean_Rule", columns="FOV", values="state")
        .reindex(columns=fovs, fill_value=0)
        .fillna(0)
        .astype(int)
    )
    definitions = rules.drop_duplicates("Clean_Rule").set_index("Clean_Rule")
    rule_cells = {
        rule: dh.base_items(row["Antecedents"]) + dh.base_items(row["Consequents"])
        for rule, row in definitions.iterrows()
    }
    eligible = dh.eligible_fovs(rule_cells, cells, min_cells).reindex(
        index=states.index, columns=states.columns, fill_value=False
    )
    eligible.attrs["min_cells"] = min_cells
    return states, eligible, states.astype(float).where(eligible)


def units_for(metadata, organ, group_col, group, unit_col="FOV"):
    """Unique units from one organ and one group."""
    keep = (metadata["Organ"] == organ) & (metadata[group_col] == group)
    return metadata.loc[keep, unit_col].drop_duplicates().tolist()


def gap_pvalues(values, n_a, observed, n_permutations=5_000, seed=0):
    """Two-sided permutation p-values for a difference between two means."""
    rng = np.random.default_rng(seed)
    labels = np.r_[np.ones(n_a), np.zeros(values.shape[1] - n_a)]
    in_a = np.column_stack([rng.permutation(labels) for _ in range(n_permutations)])
    in_b = 1 - in_a
    valid = np.isfinite(values).astype(float)
    numbers = np.nan_to_num(values)
    count_a, count_b = valid @ in_a, valid @ in_b
    mean_a = np.divide(numbers @ in_a, count_a, out=np.full_like(count_a, np.nan), where=count_a > 0)
    mean_b = np.divide(numbers @ in_b, count_b, out=np.full_like(count_b, np.nan), where=count_b > 0)
    null = mean_a - mean_b
    extreme = np.abs(null) >= np.abs(observed)[:, None] - 1e-12
    return (np.nansum(extreme, axis=1) + 1) / (np.isfinite(null).sum(axis=1) + 1)


def compare(states, units_a, units_b, label_a, label_b, min_eligible=3,
            min_present=3, n_permutations=5_000):
    """Compare two groups using each rule's eligible units."""
    a = states.reindex(columns=units_a).to_numpy(float)
    b = states.reindex(columns=units_b).to_numpy(float)
    n_a, n_b = np.isfinite(a).sum(1), np.isfinite(b).sum(1)
    present = np.sum(np.isfinite(a) & (a != 0), axis=1)
    present += np.sum(np.isfinite(b) & (b != 0), axis=1)
    keep = (n_a >= min_eligible) & (n_b >= min_eligible) & (present >= min_present)
    a, b = a[keep], b[keep]
    names, n_a, n_b = states.index[keep], n_a[keep], n_b[keep]
    net_a, net_b = np.nanmean(a, axis=1), np.nanmean(b, axis=1)
    effect = net_a - net_b
    p_value = gap_pvalues(np.hstack([a, b]), len(units_a), effect, n_permutations)
    return pd.DataFrame({
        f"net_{label_a}": net_a,
        f"net_{label_b}": net_b,
        "effect_size": effect,
        f"n_eligible_{label_a}": n_a,
        f"n_eligible_{label_b}": n_b,
        f"n_attract_{label_a}": np.nansum(a == 1, axis=1),
        f"n_avoid_{label_a}": np.nansum(a == -1, axis=1),
        f"n_attract_{label_b}": np.nansum(b == 1, axis=1),
        f"n_avoid_{label_b}": np.nansum(b == -1, axis=1),
        "p_value": p_value,
    }, index=names).sort_values("effect_size", key=abs, ascending=False)


def correct_together(results):
    """Add FDR values using one correction across all supplied result tables."""
    if not results:
        return results
    p_values = np.concatenate([result["p_value"] for result in results.values()])
    adjusted = multipletests(p_values, method="fdr_bh")[1]
    start = 0
    for result in results.values():
        stop = start + len(result)
        result["fdr"] = adjusted[start:stop]
        start = stop
    return results


def _slopes(values, ranks):
    valid = np.isfinite(values).astype(float)
    numbers = np.nan_to_num(values)
    n = valid.sum(axis=1)
    sx, sx2 = valid @ ranks, valid @ (ranks ** 2)
    sy, sxy = numbers.sum(axis=1), numbers @ ranks
    numerator = sxy - sx * sy / n
    denominator = sx2 - sx ** 2 / n
    return np.divide(numerator, denominator, out=np.full_like(numerator, np.nan), where=denominator > 0)


def _slope_pvalues(values, ranks, observed, n_permutations=5_000, seed=0):
    rng = np.random.default_rng(seed)
    shuffled = np.column_stack([rng.permutation(ranks) for _ in range(n_permutations)])
    valid = np.isfinite(values).astype(float)
    numbers = np.nan_to_num(values)
    n, sy = valid.sum(axis=1)[:, None], numbers.sum(axis=1)[:, None]
    sx, sx2 = valid @ shuffled, valid @ (shuffled ** 2)
    denominator = sx2 - sx ** 2 / n
    null = np.divide(
        numbers @ shuffled - sx * sy / n,
        denominator,
        out=np.full_like(denominator, np.nan),
        where=denominator > 0,
    )
    extreme = np.abs(null) >= np.abs(observed)[:, None] - 1e-12
    return (np.nansum(extreme, axis=1) + 1) / (np.isfinite(null).sum(axis=1) + 1)


def trend(states, groups, stages, min_eligible=3, min_present=3,
          n_permutations=5_000):
    """Test an ordered trend while retaining rule-specific eligible denominators."""
    ordered = [unit for stage in stages for unit in groups[stage]]
    values = states.reindex(columns=ordered).to_numpy(float)
    eligible_n = {
        stage: np.isfinite(states.reindex(columns=groups[stage])).sum(axis=1)
        for stage in stages
    }
    keep = np.sum(np.isfinite(values) & (values != 0), axis=1) >= min_present
    for count in eligible_n.values():
        keep &= count >= min_eligible
    values = values[keep]
    names = states.index[keep]
    ranks = np.concatenate([
        np.full(len(groups[stage]), rank, dtype=float)
        for rank, stage in enumerate(stages)
    ])
    slope = _slopes(values, ranks)
    p_value = _slope_pvalues(values, ranks, slope, n_permutations)
    result = pd.DataFrame(index=names)
    start = 0
    for stage in stages:
        stop = start + len(groups[stage])
        result[f"net_{stage}"] = np.nanmean(values[:, start:stop], axis=1)
        result[f"n_eligible_{stage}"] = eligible_n[stage][keep]
        start = stop
    steps = np.diff(result[[f"net_{stage}" for stage in stages]].to_numpy(), axis=1)
    result["slope"] = slope
    result["steady"] = (steps >= 0).all(axis=1) | (steps <= 0).all(axis=1)
    result["p_value"] = p_value
    result["fdr"] = multipletests(p_value, method="fdr_bh")[1]
    return result.sort_values("slope", key=abs, ascending=False)


def severity_screen(states, metadata, organ, group_col, stages, min_eligible=3,
                    min_present=3, n_permutations=5_000, unit_col="FOV"):
    """Ordered trend and all pairwise stage comparisons for one organ and score."""
    groups = {
        stage: units_for(metadata, organ, group_col, stage, unit_col)
        for stage in stages
    }
    ordered = trend(states, groups, stages, min_eligible, min_present, n_permutations)
    pairs = {
        pair: compare(
            states, groups[pair[0]], groups[pair[1]], *pair,
            min_eligible=min_eligible, min_present=min_present,
            n_permutations=n_permutations,
        )
        for pair in combinations(stages, 2)
    }
    correct_together(pairs)
    return ordered, pairs
