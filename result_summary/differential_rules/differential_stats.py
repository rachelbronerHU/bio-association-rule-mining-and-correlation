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


_METRIC_COLUMNS = {"lift": "Lift", "conf": "Confidence", "conv": "Conviction",
                   "sup": "Support", "lev": "Leverage"}


def rule_metrics(item):
    """The mining metrics of one rule occurrence, under short names."""
    return {short: pd.to_numeric(item.get(column, np.nan), errors="coerce")
            for short, column in _METRIC_COLUMNS.items()}


def representative_fovs(rules, eligible, metadata, rule, organ, group_col,
                        groups, metric="Lift", state=None, include_no_rule=True,
                        include_all_states=True):
    """Choose typical attraction, avoidance and eligible no-rule FOVs per group.

    ``state`` restricts the rule-bearing example to one direction. Otherwise both
    directions are returned when they occur anywhere in the displayed groups. An
    eligible no-rule example is also returned by default. This makes the tissue maps
    an honest companion to the prevalence plot, rather than showing hits only.
    """
    info = metadata.drop_duplicates("FOV").set_index("FOV")
    allowed = eligible.columns[eligible.loc[rule]]
    rows = rules[
        (rules["Clean_Rule"] == rule) & rules["FOV"].isin(allowed)
    ].copy()
    rows["group"] = rows["FOV"].map(info[group_col])
    rows = rows[rows["FOV"].map(info["Organ"]) == organ]

    definitions = rules[rules["Clean_Rule"] == rule]
    if definitions.empty:
        return pd.DataFrame()
    definition = definitions.iloc[0]
    ant = dh.base_items(definition["Antecedents"])
    con = dh.base_items(definition["Consequents"])
    items = {"antecedent_items": definition["Antecedents"],
             "consequent_items": definition["Consequents"],
             "kind": definition.get("Kind", "attracts")}
    available_states = set(rows["state"].astype(int))
    if state is not None:
        shown_states = [state]
    elif include_all_states:
        shown_states = [value for value in (1, -1) if value in available_states]
    else:
        shown_states = []

    examples = []
    eligible_n = {}
    for group in groups:
        group_fovs = info.index[(info["Organ"] == organ) & (info[group_col] == group)]
        group_eligible = eligible.loc[rule].reindex(group_fovs, fill_value=False)
        eligible_fovs = group_eligible.index[group_eligible]
        eligible_n[group] = int(group_eligible.sum())
        current = rows[rows["group"] == group]
        local_states = shown_states
        if not local_states and not include_all_states and state is None and not current.empty:
            local_states = [int(current["state"].value_counts().index[0])]
        for chosen_state in local_states:
            candidates = current[current["state"] == chosen_state]
            if candidates.empty:
                continue
            values = pd.to_numeric(candidates[metric], errors="coerce")
            index = ((values - values.median()).abs().idxmin()
                     if values.notna().any() else candidates.index[0])
            item = candidates.loc[index]
            examples.append({
                "rule": rule, "stage": group, "FOV": item["FOV"],
                "state": int(item["state"]), "metric": item[metric],
                "fdr": item.get("Individual_FDR", np.nan),
                "metrics": rule_metrics(item),
                "antecedent_cells": ant, "consequent_cells": con, **items,
            })

        if include_no_rule:
            hit_fovs = set(current["FOV"])
            no_rule = [fov for fov in eligible_fovs if fov not in hit_fovs]
            if no_rule:
                # Median total cell count gives a typical, deterministic field.
                cell_counts = metadata.set_index("FOV").reindex(no_rule)
                count_col = next((column for column in ("Cell count", "cell_count", "n_cells")
                                  if column in cell_counts), None)
                fov = (cell_counts[count_col] - cell_counts[count_col].median()).abs().idxmin() \
                    if count_col else sorted(no_rule, key=str)[len(no_rule) // 2]
                examples.append({
                    "rule": rule, "stage": group, "FOV": fov,
                    "state": 0, "metric": np.nan, "fdr": np.nan, "metrics": {},
                    "antecedent_cells": ant, "consequent_cells": con, **items,
                })
    result = pd.DataFrame(examples)
    result.attrs["eligible_n"] = eligible_n
    return result


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
    result = pd.DataFrame({
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
    }, index=names)
    result["fdr"] = (
        multipletests(p_value, method="fdr_bh")[1]
        if len(p_value) else np.array([], dtype=float)
    )
    return result.sort_values("effect_size", key=abs, ascending=False)


def correct_together(results):
    """Add FDR values using one correction across all supplied result tables."""
    if not results:
        return results
    p_values = np.concatenate([result["p_value"] for result in results.values()])
    if not len(p_values):
        for result in results.values():
            result["fdr"] = np.array([], dtype=float)
        return results
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
    result["fdr"] = (
        multipletests(p_value, method="fdr_bh")[1]
        if len(p_value) else np.array([], dtype=float)
    )
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


def aggregate_fovs(values, metadata, unit_col="Biopsy"):
    """Average a rule-by-FOV matrix within each independent biological unit."""
    fov_to_unit = (
        metadata[["FOV", unit_col]].drop_duplicates("FOV").set_index("FOV")[unit_col]
    )
    columns = values.columns.intersection(fov_to_unit.index, sort=False)
    transposed = values.loc[:, columns].T.copy()
    transposed[unit_col] = fov_to_unit.reindex(columns).to_numpy()
    transposed = transposed.dropna(subset=[unit_col])
    return transposed.groupby(unit_col, sort=False).mean().T


def direction_matrix(states, state):
    """Binary occurrence matrix for one rule direction, preserving ineligibility."""
    if state not in (-1, 1):
        raise ValueError("state must be 1 (attraction) or -1 (avoidance)")
    return (states == state).astype(float).where(states.notna())


def direction_metric_matrix(rules, states, eligibility, metric="Lift", state=1):
    """Rule metric values where one direction fires; other eligible FOVs are NaN."""
    if state not in (-1, 1):
        raise ValueError("state must be 1 (attraction) or -1 (avoidance)")
    rows = rules[rules["state"] == state]
    values = rows.pivot_table(
        index="Clean_Rule", columns="FOV", values=metric, aggfunc="median"
    ).reindex(index=states.index, columns=states.columns)
    return values.where(eligibility)


def control_severe_prevalence(states, eligibility, metadata, organ, group_col,
                              control="Control", severe="Severe", min_fovs=10):
    """Visible Control--Severe attraction and avoidance gaps, kept separately.

    This table is descriptive: it ranks what is most evident in eligible FOVs and
    does not use FDR as a display or selection gate.
    """
    groups = {}
    for label in (control, severe):
        fovs = metadata.loc[
            (metadata["Organ"] == organ) & (metadata[group_col] == label), "FOV"
        ].drop_duplicates()
        groups[label] = fovs

    rows = []
    for rule in states.index:
        summaries = {}
        for label, fovs in groups.items():
            can_test = eligibility.reindex(columns=fovs).loc[rule]
            observed = states.reindex(columns=fovs).loc[rule, can_test.index[can_test]]
            summaries[label] = {
                "eligible": len(observed),
                "attraction": int((observed == 1).sum()),
                "avoidance": int((observed == -1).sum()),
            }
        if min(summaries[label]["eligible"] for label in (control, severe)) < min_fovs:
            continue
        for state, direction in ((1, "attraction"), (-1, "avoidance")):
            key = direction
            first = summaries[control][key] / summaries[control]["eligible"]
            last = summaries[severe][key] / summaries[severe]["eligible"]
            rows.append({
                "rule": rule, "direction": direction,
                "control_count": summaries[control][key],
                "control_eligible": summaries[control]["eligible"],
                "severe_count": summaries[severe][key],
                "severe_eligible": summaries[severe]["eligible"],
                "control_share": first, "severe_share": last,
                "effect": last - first, "abs_effect": abs(last - first),
            })
    return pd.DataFrame(rows).sort_values(
        ["direction", "abs_effect"], ascending=[True, False]
    ).reset_index(drop=True)


def prevalence_range(states, eligibility, metadata, organ, group_col, groups,
                     min_fovs=10):
    """Largest prevalence gap across named groups for attraction and avoidance."""
    rows = []
    for rule in states.index:
        shares, counts, totals = {}, {}, {}
        enough = True
        for group in groups:
            fovs = metadata.loc[
                (metadata["Organ"] == organ) & (metadata[group_col] == group), "FOV"
            ].drop_duplicates()
            can_test = eligibility.reindex(columns=fovs).loc[rule]
            observed = states.reindex(columns=fovs).loc[rule, can_test.index[can_test]]
            totals[group] = len(observed)
            enough &= len(observed) >= min_fovs
            for value, direction in ((1, "attraction"), (-1, "avoidance")):
                counts[(group, direction)] = int((observed == value).sum())
                shares[(group, direction)] = (
                    counts[(group, direction)] / len(observed) if len(observed) else np.nan
                )
        if not enough:
            continue
        for direction in ("attraction", "avoidance"):
            values = pd.Series({group: shares[(group, direction)] for group in groups})
            low, high = values.idxmin(), values.idxmax()
            row = {
                "rule": rule, "direction": direction, "lowest": low, "highest": high,
                "effect": values[high] - values[low], "abs_effect": values[high] - values[low],
            }
            for group in groups:
                row[f"{group}_count"] = counts[(group, direction)]
                row[f"{group}_eligible"] = totals[group]
                row[f"{group}_share"] = values[group]
            rows.append(row)
    return pd.DataFrame(rows).sort_values(
        ["direction", "abs_effect"], ascending=[True, False]
    ).reset_index(drop=True)


def evidence_table(ordered=None, pairs=None, stages=None):
    """Put ordered and pairwise screen results into one comparable long table.

    ``effect`` is the visible end-to-end difference for an ordered screen and the
    observed difference for a pairwise screen. Each row therefore keeps its effect,
    FDR and denominator from the same statistical comparison.
    """
    records = []
    if ordered is not None and not ordered.empty:
        net_columns = [f"net_{stage}" for stage in (stages or [])]
        n_columns = [f"n_eligible_{stage}" for stage in (stages or [])]
        for rule, row in ordered.iterrows():
            effect = (
                row[net_columns[-1]] - row[net_columns[0]]
                if len(net_columns) >= 2 and all(column in row for column in net_columns)
                else row["slope"]
            )
            records.append({
                "rule": rule,
                "test": "ordered",
                "effect": float(effect),
                "abs_effect": abs(float(effect)),
                "p_value": float(row["p_value"]),
                "fdr": float(row["fdr"]),
                "min_n": int(row[n_columns].min()) if n_columns else np.nan,
                "steady": bool(row.get("steady", False)),
            })
    for comparison, result in (pairs or {}).items():
        label_a, label_b = comparison
        n_columns = [f"n_eligible_{label_a}", f"n_eligible_{label_b}"]
        for rule, row in result.iterrows():
            effect = float(row["effect_size"])
            records.append({
                "rule": rule,
                "test": f"{label_a} vs {label_b}",
                "effect": effect,
                "abs_effect": abs(effect),
                "p_value": float(row["p_value"]),
                "fdr": float(row["fdr"]),
                "min_n": int(row[n_columns].min()),
                "steady": np.nan,
            })
    columns = ["rule", "test", "effect", "abs_effect", "p_value", "fdr",
               "min_n", "steady"]
    return pd.DataFrame.from_records(records, columns=columns)


def select_candidates(statistical_evidence, visual_evidence=None, fdr_cutoff=0.05,
                      statistical_min_effect=0.20, visual_min_effect=0.30,
                      visual_min_n=10, max_statistical=15, max_visual=15):
    """Union independent-unit significance and strong FOV-visible effects.

    Pass one evidence table to use the same unit for both streams, or provide a
    separate ``visual_evidence`` table (normally FOV-level) while statistical
    evidence is calculated at biopsy level.
    """
    if visual_evidence is None:
        visual_evidence = statistical_evidence
    if statistical_evidence.empty and visual_evidence.empty:
        return pd.DataFrame(columns=[
            "rule", "selection", "best_fdr", "statistical_effect",
            "visual_effect", "visual_min_n", "best_biopsy_test",
            "biopsy_min_n", "statistical_test", "visual_test",
        ])

    significant = statistical_evidence[
        (statistical_evidence["fdr"] <= fdr_cutoff)
        & (statistical_evidence["abs_effect"] >= statistical_min_effect)
    ].sort_values(["fdr", "abs_effect"], ascending=[True, False])
    significant = significant.drop_duplicates("rule").head(max_statistical)

    visual = visual_evidence[
        (visual_evidence["abs_effect"] >= visual_min_effect)
        & (visual_evidence["min_n"] >= visual_min_n)
    ].sort_values(["abs_effect", "fdr"], ascending=[False, True])
    visual = visual.drop_duplicates("rule").head(max_visual)

    rules = list(dict.fromkeys(significant["rule"].tolist() + visual["rule"].tolist()))
    rows = []
    for rule in rules:
        stat = significant[significant["rule"] == rule]
        seen = visual[visual["rule"] == rule]
        biopsy_rows = statistical_evidence[statistical_evidence["rule"] == rule]
        best_biopsy = None
        if not biopsy_rows.empty:
            best_biopsy = biopsy_rows.sort_values(
                ["fdr", "abs_effect"], ascending=[True, False]
            ).iloc[0]
        reasons = []
        if not stat.empty:
            reasons.append("statistical")
        if not seen.empty:
            reasons.append("strong visible effect")
        rows.append({
            "rule": rule,
            "selection": " + ".join(reasons),
            "best_fdr": (
                stat["fdr"].iloc[0]
                if not stat.empty
                else (best_biopsy["fdr"] if best_biopsy is not None else np.nan)
            ),
            "statistical_effect": stat["effect"].iloc[0] if not stat.empty else np.nan,
            "visual_effect": seen["effect"].iloc[0] if not seen.empty else np.nan,
            "visual_min_n": seen["min_n"].iloc[0] if not seen.empty else np.nan,
            "best_biopsy_test": best_biopsy["test"] if best_biopsy is not None else "",
            "biopsy_min_n": best_biopsy["min_n"] if best_biopsy is not None else np.nan,
            "statistical_test": stat["test"].iloc[0] if not stat.empty else "",
            "visual_test": seen["test"].iloc[0] if not seen.empty else "",
        })
    return pd.DataFrame(rows)
