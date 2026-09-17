"""Shared preparation and selection for the complex-rule stage notebooks."""

import ast

import numpy as np
import pandas as pd

from data_helper import base_items, check_rule_overlap, clean_items, eligible_fovs


COMPLEX_TYPES = ["ant-complex", "con-complex", "both-complex"]


def _cell_types(items):
    return tuple(sorted(set(base_items(items))))


def _stored_rule(antecedents, consequents):
    side = lambda value: " + ".join(sorted(ast.literal_eval(value)))
    return f"{side(antecedents)} -> {side(consequents)}"


def prepare_rules(results, fovs, no_self=True):
    """Add readable rule identifiers and keep FOVs present in the atlas."""
    rules = results[results["FOV"].isin(fovs["FOV"])].copy()
    if no_self:
        overlap = [
            check_rule_overlap(base_items(a), base_items(c))
            for a, c in zip(rules["Antecedents"], rules["Consequents"])
        ]
        rules = rules[~pd.Series(overlap, index=rules.index)].copy()
    rules["ant_t"] = rules["Antecedents"].map(_cell_types)
    rules["con_t"] = rules["Consequents"].map(_cell_types)
    rules["name"] = (
        rules["Antecedents"].map(clean_items)
        + " -> "
        + rules["Consequents"].map(clean_items)
    )
    rules["stored_rule"] = [
        _stored_rule(a, c)
        for a, c in zip(rules["Antecedents"], rules["Consequents"])
    ]
    return rules


def metric_strength(rules, metric="Lift"):
    """Use larger values for stronger attraction and stronger avoidance."""
    values = pd.to_numeric(rules[metric], errors="coerce")
    if metric == "Leverage":
        return values.abs()
    return values.where(rules["Kind"] == "attracts", 1 / values.clip(lower=1e-6))


def metric_values(rules, metric="Lift"):
    """The requested metric as stored, for direct display."""
    return pd.to_numeric(rules[metric], errors="coerce")


def select_complex_rules(rules, max_fdr, new_only=False, min_strength=1.3,
                         metric="Lift"):
    """Select significant informative complex rules for one notebook."""
    selected = rules[
        rules["Rule_Type"].isin(COMPLEX_TYPES)
        & rules["Adds_Information"]
        & (rules["Individual_FDR"] <= max_fdr)
    ].copy()
    if new_only:
        selected = selected[
            (selected["Complex_Class"] == "new")
            & (metric_strength(selected, metric) >= min_strength)
        ]
    else:
        selected = selected[selected["Complex_Class"] != "new"]
    return selected


def rule_eligibility(complex_rules, cells, min_cells):
    definitions = complex_rules.drop_duplicates("name").set_index("name")
    rule_cells = {
        name: sorted(set(row.ant_t + row.con_t))
        for name, row in definitions.iterrows()
    }
    return eligible_fovs(rule_cells, cells, min_cells)


def _parent_names(rows):
    names = set()
    for value in rows["Simpler_Rules"].dropna():
        names.update(ast.literal_eval(value))
    return names


def _scope_fovs(fovs, organ, stage, stage_column):
    return set(fovs.loc[
        (fovs["Organ"] == organ) & (fovs[stage_column] == stage), "FOV"
    ])


def stage_prevalence(complex_rules, all_rules, eligibility, fovs, organ, stages,
                     stage_column, max_fdr, use_eligibility):
    """Count complex and significant simpler-rule occurrence in each stage."""
    rows = []
    significant_parent_fovs = (
        all_rules[all_rules["Individual_FDR"] <= max_fdr]
        .groupby("stored_rule")["FOV"].agg(set).to_dict()
    )
    for name, occurrences in complex_rules.groupby("name", sort=False):
        parent_names = _parent_names(occurrences)
        parent_fovs = set().union(*(
            significant_parent_fovs.get(parent, set()) for parent in parent_names
        )) if parent_names else set()
        for stage in stages:
            available = _scope_fovs(fovs, organ, stage, stage_column)
            if use_eligibility:
                available &= set(eligibility.columns[eligibility.loc[name]])
            complex_fovs = available & set(occurrences["FOV"])
            simpler_fovs = available & parent_fovs
            denominator = len(available)
            rows.append({
                "name": name,
                "stage": stage,
                "n_fovs": denominator,
                "complex": len(complex_fovs),
                "simpler": len(simpler_fovs),
                "complex_share": len(complex_fovs) / denominator if denominator else np.nan,
                "simpler_share": len(simpler_fovs) / denominator if denominator else np.nan,
            })
    return pd.DataFrame(rows)


def availability_breakdown(names, complex_rules, all_rules, eligibility, fovs,
                           organ, stages, stage_column, max_fdr):
    """Split every stage FOV into unavailable, empty, parent-only, or complex."""
    significant_parent_fovs = (
        all_rules[all_rules["Individual_FDR"] <= max_fdr]
        .groupby("stored_rule")["FOV"].agg(set).to_dict()
    )
    rows = []
    for name in names:
        occurrences = complex_rules[complex_rules["name"] == name]
        parent_names = _parent_names(occurrences)
        parent_fovs = set().union(*(
            significant_parent_fovs.get(parent, set()) for parent in parent_names
        )) if parent_names else set()
        eligible_fovs_for_rule = set(eligibility.columns[eligibility.loc[name]])
        complex_fovs = set(occurrences["FOV"])
        for stage in stages:
            total = _scope_fovs(fovs, organ, stage, stage_column)
            eligible = total & eligible_fovs_for_rule
            complex_here = eligible & complex_fovs
            parents_here = eligible & parent_fovs
            rows.append({
                "name": name,
                "stage": stage,
                "n_total": len(total),
                "ineligible": len(total - eligible),
                "nothing": len(eligible - complex_here - parents_here),
                "covered": len(parents_here - complex_here),
                "informative": len(complex_here),
            })
    return pd.DataFrame(rows)


def rank_rules(prevalence, stages, min_rule_fovs, top_n):
    """Keep rules occurring in enough FOVs in at least one stage, then rank movement."""
    if prevalence.empty:
        return prevalence, []
    counts = prevalence.pivot(index="name", columns="stage", values="complex")
    shares = prevalence.pivot(index="name", columns="stage", values="complex_share")
    eligible = counts.max(axis=1) >= min_rule_fovs
    spread = (shares.max(axis=1) - shares.min(axis=1)).where(eligible).dropna()
    order = sorted(
        spread.index,
        key=lambda name: (spread[name], counts.loc[name].max(), shares.loc[name].max()),
        reverse=True,
    )[:top_n]
    selected = prevalence[prevalence["name"].isin(order)].copy()
    selected["name"] = pd.Categorical(selected["name"], order, ordered=True)
    return selected.sort_values(["name", "stage"]), order


def opposite_direction_rules(prevalence, stages, min_rule_fovs, min_gap=0.10):
    """Rules whose complex and simpler prevalence move opposite ways, control to severe.

    Both moves must clear `min_gap`, so a rule only qualifies when each side really
    moves rather than wobbling around its starting value. Returns one row per rule,
    largest combined movement first.
    """
    if prevalence.empty or len(stages) < 2:
        return pd.DataFrame()
    first, last = stages[0], stages[-1]
    counts = prevalence.pivot(index="name", columns="stage", values="complex")
    complex_share = prevalence.pivot(index="name", columns="stage", values="complex_share")
    simpler_share = prevalence.pivot(index="name", columns="stage", values="simpler_share")
    if first not in complex_share or last not in complex_share:
        return pd.DataFrame()

    complex_move = complex_share[last] - complex_share[first]
    simpler_move = simpler_share[last] - simpler_share[first]
    enough_fovs = counts.max(axis=1) >= min_rule_fovs
    opposed = (complex_move * simpler_move < 0)
    both_move = (complex_move.abs() >= min_gap) & (simpler_move.abs() >= min_gap)

    rows = pd.DataFrame({
        "name": complex_share.index,
        "complex_first": complex_share[first].to_numpy(),
        "complex_last": complex_share[last].to_numpy(),
        "simpler_first": simpler_share[first].to_numpy(),
        "simpler_last": simpler_share[last].to_numpy(),
        "complex_move": complex_move.to_numpy(),
        "simpler_move": simpler_move.to_numpy(),
        "fovs": counts.max(axis=1).to_numpy(),
    })
    keep = (enough_fovs & opposed & both_move).to_numpy()
    rows = rows[keep].copy()
    if rows.empty:
        return rows
    rows["movement"] = rows["complex_move"].abs() + rows["simpler_move"].abs()
    return rows.sort_values("movement", ascending=False).reset_index(drop=True)


def opposite_table(rows):
    """The opposite-direction screen as plain percentages."""
    if rows is None or rows.empty:
        return pd.DataFrame({"note": ["No rule moves both ways by the required margin."]})
    percent = lambda values: (values * 100).round(0).astype(int).astype(str) + "%"
    return pd.DataFrame({
        "rule": rows["name"].astype(str),
        "complex": percent(rows["complex_first"]) + " -> " + percent(rows["complex_last"]),
        "simpler": percent(rows["simpler_first"]) + " -> " + percent(rows["simpler_last"]),
        "FOVs": rows["fovs"],
    })


def detail_rules(complex_rules, ranked_names, n=2, new_only=False):
    """Choose reproducible examples, covering both non-new success modes when possible."""
    if new_only:
        return list(ranked_names[:n])
    chosen = []
    for verdict in ["stronger_effect", "simpler_are_noise"]:
        available = set(complex_rules.loc[
            complex_rules["Complex_Class"] == verdict, "name"
        ])
        chosen.extend(name for name in ranked_names if name in available and name not in chosen)
        if len(chosen) >= n:
            return chosen[:n]
    chosen.extend(name for name in ranked_names if name not in chosen)
    return chosen[:n]


def _best_row(rows, metric):
    if rows.empty:
        return None
    strength = metric_strength(rows, metric)
    return rows.loc[strength.idxmax()]


def _patient_medians(rows, fovs, stages, stage_column, series):
    if not rows:
        return pd.DataFrame(columns=["stage", "PatientID", "series", "value"])
    values = pd.DataFrame(rows)
    metadata = fovs.set_index("FOV")[["PatientID", stage_column]]
    values = values.join(metadata, on="FOV").rename(columns={stage_column: "stage"})
    values = values[values["stage"].isin(stages)]
    values["series"] = series
    return values.groupby(["stage", "PatientID", "series"], as_index=False)["value"].median()


def patient_strengths(name, complex_rules, all_rules, eligibility, fovs, organ,
                      stages, stage_column, metric, use_eligibility):
    """Patient medians for the complex rule and its best available simpler rule."""
    scoped_fovs = set(fovs.loc[fovs["Organ"] == organ, "FOV"])
    if use_eligibility:
        scoped_fovs &= set(eligibility.columns[eligibility.loc[name]])
    complex_rows = complex_rules[
        (complex_rules["name"] == name) & complex_rules["FOV"].isin(scoped_fovs)
    ]
    complex_values = [
        {"FOV": row["FOV"], "value": float(metric_values(row.to_frame().T, metric).iloc[0])}
        for _, row in complex_rows.iterrows()
    ]
    result = [_patient_medians(
        complex_values, fovs, stages, stage_column, "complex"
    )]

    parent_names = _parent_names(complex_rows)
    parent_rows = all_rules[
        all_rules["stored_rule"].isin(parent_names)
        & all_rules["FOV"].isin(scoped_fovs)
    ]
    simpler_values = []
    for fov, rows in parent_rows.groupby("FOV"):
        best = _best_row(rows, metric)
        simpler_values.append({
            "FOV": fov,
            "value": float(metric_values(best.to_frame().T, metric).iloc[0]),
        })
    result.append(_patient_medians(
        simpler_values, fovs, stages, stage_column, "simpler"
    ))
    return pd.concat(result, ignore_index=True)


def _metrics(row):
    return {
        name: row.get(name, np.nan)
        for name in ["Lift", "Confidence", "Leverage", "Support", "Conviction",
                     "Individual_FDR"]
    }


def cell_abundance(name, complex_rules, cells, eligibility, fovs, organ, stages,
                   stage_column, use_eligibility):
    """How common each of the rule's cell types is in every FOV of each stage.

    Abundance is the share of that FOV's cells, so a 400 and an 800 micron field
    can sit on the same axis. One row per FOV per cell type.
    """
    definition = complex_rules[complex_rules["name"] == name]
    if definition.empty:
        return pd.DataFrame(columns=["stage", "FOV", "cell type", "share"])
    row = definition.iloc[0]
    rule_cells = sorted(set(row["ant_t"] + row["con_t"]))

    totals = cells.groupby("fov").size()
    counts = (
        cells[cells["cell type"].isin(rule_cells)]
        .groupby(["fov", "cell type"]).size()
    )
    rows = []
    for stage in stages:
        stage_fovs = _scope_fovs(fovs, organ, stage, stage_column)
        if use_eligibility:
            stage_fovs &= set(eligibility.columns[eligibility.loc[name]])
        for fov in sorted(stage_fovs):
            total = int(totals.get(fov, 0))
            if not total:
                continue
            for cell_type in rule_cells:
                rows.append({
                    "stage": stage,
                    "FOV": fov,
                    "cell type": cell_type,
                    "share": int(counts.get((fov, cell_type), 0)) / total,
                })
    return pd.DataFrame(rows)


def fov_examples(name, complex_rules, all_rules, eligibility, fovs, organ, stages,
                 stage_column, metric, use_eligibility, new_only, max_fdr=0.05):
    """One representative FOV per stage for a rule, and where the rule is absent
    but one of its simpler rules still holds, a FOV showing that instead."""
    scoped = complex_rules[
        (complex_rules["name"] == name)
        & complex_rules["FOV"].isin(fovs.loc[fovs["Organ"] == organ, "FOV"])
    ]
    eligible_fovs = set(eligibility.columns[eligibility.loc[name]])
    if use_eligibility:
        scoped = scoped[scoped["FOV"].isin(eligible_fovs)]
    metadata = fovs.set_index("FOV")
    parent_names = _parent_names(scoped)
    examples = []
    for stage in stages:
        rows = scoped[scoped["FOV"].map(metadata[stage_column]) == stage]
        if rows.empty:
            fallback = _simpler_only_example(
                name, stage, parent_names, all_rules, fovs, eligible_fovs, organ,
                stage_column, metric, use_eligibility, new_only, max_fdr,
            )
            if fallback is not None:
                examples.append(fallback)
            continue
        strengths = metric_strength(rows, metric)
        middle = strengths.median()
        row = rows.loc[(strengths - middle).abs().idxmin()]
        item = {
            "rule": name,
            "stage": stage,
            "FOV": row["FOV"],
            "verdict": row["Complex_Class"],
            "rule_present": True,
            "antecedent_cells": list(row["ant_t"]),
            "consequent_cells": list(row["con_t"]),
            "complex_metrics": _metrics(row),
        }
        if not new_only:
            parents = all_rules[
                (all_rules["FOV"] == row["FOV"])
                & all_rules["stored_rule"].isin(ast.literal_eval(row["Simpler_Rules"]))
            ]
            parent = _best_row(parents, metric)
            if parent is None:
                continue
            item.update(
                simpler_rule=parent["stored_rule"],
                simpler_metrics=_metrics(parent),
            )
        examples.append(item)
    return pd.DataFrame(examples)


def _simpler_only_example(name, stage, parent_names, all_rules, fovs, eligible_fovs,
                          organ, stage_column, metric, use_eligibility, new_only,
                          max_fdr):
    """A representative FOV of this stage where a simpler rule holds but the
    complex rule does not. Nothing to show for a new rule, which has no parents."""
    if new_only or not parent_names:
        return None
    stage_fovs = _scope_fovs(fovs, organ, stage, stage_column)
    if use_eligibility:
        stage_fovs &= eligible_fovs
    rows = all_rules[
        all_rules["stored_rule"].isin(parent_names)
        & all_rules["FOV"].isin(stage_fovs)
        & (all_rules["Individual_FDR"] <= max_fdr)
    ]
    if rows.empty:
        return None
    best_per_fov = [_best_row(group, metric) for _, group in rows.groupby("FOV")]
    best_per_fov = [row for row in best_per_fov if row is not None]
    if not best_per_fov:
        return None
    frame = pd.DataFrame(best_per_fov)
    strengths = metric_strength(frame, metric)
    middle = strengths.median()
    row = frame.loc[(strengths - middle).abs().idxmin()]
    ant, con = _rule_cells(row["stored_rule"])
    return {
        "rule": name,
        "stage": stage,
        "FOV": row["FOV"],
        "verdict": "absent",
        "rule_present": False,
        "antecedent_cells": ant,
        "consequent_cells": con,
        "complex_metrics": {},
        "simpler_rule": row["stored_rule"],
        "simpler_metrics": _metrics(row),
    }


def _rule_cells(stored_rule):
    """The two sides of a stored rule name, as plain cell-type lists."""
    left, right = str(stored_rule).split(" -> ", 1)
    clean = lambda side: [
        item.replace("_CENTER", "").replace("_NEIGHBOR", "")
        for item in side.split(" + ")
    ]
    return clean(left), clean(right)


def result_table(prevalence, new_only=False):
    """Compact counts and percentages for the ranked rules."""
    if prevalence.empty:
        return prevalence
    table = prevalence.copy()
    table["complex prevalence"] = table.apply(
        lambda row: f"{row['complex']} / {row['n_fovs']} ({row['complex_share']:.0%})",
        axis=1,
    )
    table["simpler prevalence"] = table.apply(
        lambda row: f"{row['simpler']} / {row['n_fovs']} ({row['simpler_share']:.0%})",
        axis=1,
    )
    columns = ["name", "stage", "complex prevalence"]
    if not new_only:
        columns.append("simpler prevalence")
    return table[columns]
