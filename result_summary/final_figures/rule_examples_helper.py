"""Example FOVs for the rule trends: which FOVs to show for a rule, and how each of their cells counts
for it, from the mining's own patches and weights."""
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
from spatial_association_rules import Method, Settings, Weighting
from spatial_association_rules import transactions as tr

import data_helper as dh
import rule_trends_helper as rt
import rules_per_fov_helper as rf


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


def load():
    """Everything the examples need: cells, FOVs, the rule trend results and has-rule tables, every
    pairwise rule with its cell types, those that pass, and the cells per FOV and type."""
    cells, fovs, _, has_rule = rt.load()
    all_rules = rf.with_cells(rf.load(fovs["FOV"]))
    return SimpleNamespace(cells=cells, fovs=fovs, results=rt.all_results(fovs, has_rule), has_rule=has_rule,
                           all_rules=all_rules, passed=rf.passed(all_rules),
                           counts=pd.crosstab(cells["fov"], cells["cell type"].astype(str)))


def pick_fov(data, has, rule, fovs, found):
    """Of `fovs`, the one where `has` (1 has the rule, 0 testable without it) is `found`
    and the rarer cell type of `rule` has the most cells; None when there is none."""
    fovs = has[fovs].index[has[fovs].eq(found)]
    return data.counts.loc[fovs, list(rule)].min(axis=1).idxmax() if len(fovs) else None


def mined(data, rule, fov=None):
    """The mined rows of `rule` (centre, neighbour), of every kind and significance; in `fov` only when given."""
    center, neighbor = rule
    rows = data.all_rules[data.all_rules["center"].eq(center) & data.all_rules["neighbor"].eq(neighbor)]
    return rows if fov is None else rows[rows["FOV"].eq(fov)]


def unmined(data, fovs, kind, rule):
    """Of `fovs`, those where the mining did not keep `rule` of `kind` by its metrics."""
    rows = mined(data, rule)
    return fovs[~fovs.isin(rows.loc[rows["Kind"].eq(kind), "FOV"])]


def metrics(data, fov, rule, columns):
    """`fov` and the mined `columns` of `rule` there; or that it was not mined there."""
    found = mined(data, rule, fov)
    return f"{fov}\n" + (found[columns].round(3).to_string(index=False) if len(found)
                                   else "not mined in this FOV")


def pick_examples(data, organ, trend, stable, only_400, colors):
    """For each changing rule of the organ by `trend` (each stable one with `stable`), as in the rule
    trends at `rt.LEVEL`, the first `rt.MAX_RULES` per kind, two FOVs from the trend's first and last groups, first group first as in the heatmap, each
    picked by `pick_fov`: for a changing rule, the FOV with it from the end where it is more common and
    the FOV without it from the other, among the `unmined` when any is; for a stable rule, a FOV with it
    from each end; only 400 um FOVs with `only_400`. Each scored by `rule_scores`.

    Returns a table of the picks, {name: (rule, kind, [(FOV, score, its group's colour of `colors`
    (first, last)), ...])} per rule, and {group: colour} of the two ends.
    """
    fovs = data.fovs[data.fovs["Organ"].eq(organ)]
    if only_400:
        fovs = fovs[fovs["Size [um]"].eq(400)]
    groups = fovs.set_index("FOV")[trend]
    picks, examples, end_colors = [], {}, {}
    for at, by, kind, result, chosen in rt.picked(data.results, rt.LEVEL, rt.stable if stable else rt.changed):
        if (at, by) != (organ, trend):
            continue
        end_colors = dict(zip(result["found"].columns[[0, -1]], colors))
        share = (result["found"] / result["tested"])[list(end_colors)]
        for name in chosen[:rt.MAX_RULES]:
            rule, has = tuple(name.split(" -> ")), data.has_rule[kind].loc[name]
            more, sides = share.loc[name].idxmax(), []
            for group in end_colors:
                found = int(stable or group == more)
                in_group = groups.index[groups.eq(group)]
                fov = pick_fov(data, has, rule, in_group, found)
                if not found:
                    fov = pick_fov(data, has, rule, unmined(data, in_group, kind, rule), 0) or fov
                picks.append({"kind": kind, "rule": name, "group": group, "has rule": bool(found), "FOV": fov,
                              "rarer cell type's cells": data.counts.loc[fov, list(rule)].min() if fov else None})
                if fov:
                    sides.append((fov, rule_scores(data.cells[data.cells["fov"].eq(fov)], rule), end_colors[group]))
            examples[f"{kind}_{'_to_'.join(rule)}"] = (rule, kind, sides)
    return pd.DataFrame(picks), examples, end_colors
