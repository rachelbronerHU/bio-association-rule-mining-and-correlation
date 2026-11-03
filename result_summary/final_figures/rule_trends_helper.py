"""Rules that change, or stay, across a score's groups, per organ and kind: the tests behind the rule trend figures."""
import pandas as pd

import data_helper as dh
import differential_stats as ds
import rule_metrics as rm

LEVEL = "FOV" # FOV / patient
MAX_RULES = 4
MIN_SHARE = 0.1
MIN_TESTABLE = {"FOV": 10, "patient": 3}
MAX_FDR = 0.05
STABLE_SHARE = 0.5
MAX_SPREAD = 0.1
INCLUDE_MUSCLE = False
KINDS = (rm.ATTRACTS, rm.AVOIDS)
ORGANS = ("Colon", "Duodenum")
TRENDS = {  # groups in order, and whether all are tested at once rather than the first against the last
    "Pathological score": (["Control", "Mild", "Severe"], False),
    "Clinical score": (["Control", "Mild", "Severe"], False),
    "Pathological stage": (["Control", 0, 1, 2, 3], False),
    "GI stage": (["Control", 0, 1, 2, 3], False),
    "Cortico Response": (["Responder", "Non-responder"], False),
    "Days after Transplant grouped": (["Control", "<30", "30-100", ">100"], True),
}
TITLES = {"Days after Transplant grouped": "Days after transplantation"}


def load():
    """Cells, FOVs, the pairwise rules, and per kind a rule x FOV table: 1 has the rule,
    0 testable without it, NaN not testable."""
    cells, fovs, _ = dh.load_spatial_data()
    rules = dh.prepare_rules(dh.load_results(rule_max_items=dh.PAIRWISE_ITEMS, kind=None))
    rules = rules[rules["FOV"].isin(fovs["FOV"])]
    if not INCLUDE_MUSCLE:
        rules = rules[~rules["Clean_Rule"].str.contains("Muscle")]
    has_rule = {kind: ds.direction_matrix(ds.state_tables(rules, cells, fovs["FOV"], kinds=(kind,))[2],
                                          1 if kind == rm.ATTRACTS else -1)
                for kind in KINDS}
    return cells, fovs, rules, has_rule


def trend_result(fovs, has_rule, organ, trend, level):
    """`ds.trend_test` on the organ's FOVs, or on its patients: one per patient and group,
    with the rule when any of their FOVs has it; with the units per group as 'units'."""
    order, all_groups = TRENDS[trend]
    fovs = fovs[fovs["Organ"].eq(organ)]
    values, groups = has_rule[fovs["FOV"]], fovs.set_index("FOV")[trend]
    if level == "patient":
        fovs = fovs.assign(patient=fovs["PatientID"].astype(str) + " · " + fovs[trend].astype(str))
        values = ds.aggregate_fovs(values, fovs, "patient", how="max")
        groups = fovs.groupby("patient")[trend].first()
    result = ds.trend_test(values, groups, order, all_groups, MIN_TESTABLE[level], MIN_SHARE)
    return result and result | {"units": groups.value_counts()}


def all_results(fovs, has_rule):
    """`trend_result` at LEVEL per organ, trend and kind, those with a test only."""
    results = {(LEVEL, organ, trend, kind): trend_result(fovs, has_rule[kind], organ, trend, LEVEL)
               for organ in ORGANS for trend in TRENDS for kind in KINDS}
    return {key: result for key, result in results.items() if result is not None}


def summary(results):
    """Rules tested and passing MAX_FDR, per organ, trend and kind, a column pair per level."""
    table = pd.DataFrame({key: {"tested": len(result["p"]), "passed": result["fdr"].le(MAX_FDR).sum()}
                          for key, result in results.items()}).T
    table.index.names = ["level", "organ", "trend", "kind"]
    return table.unstack("level")


def changed(result, level):
    """Every rule that passes MAX_FDR, lowest p first."""
    return result["p"][result["fdr"].le(MAX_FDR)].sort_values().index


def stable(result, level):
    """Every rule found in STABLE_SHARE of the testable units of every group, each group with
    MIN_TESTABLE of them, whose share moves at most MAX_SPREAD across the groups; least first."""
    share = result["found"] / result["tested"]
    common = share.ge(STABLE_SHARE).all(axis=1) & result["tested"].ge(MIN_TESTABLE[level]).all(axis=1)
    spread = (share.max(axis=1) - share.min(axis=1))[common]
    return spread[spread.le(MAX_SPREAD)].sort_values().index


def picked(results, level, pick):
    """Per organ, trend and kind of `level`: the organ, trend, kind, its result and the rules `pick`
    chooses; only when it chooses any."""
    for (at, organ, trend, kind), result in results.items():
        chosen = pick(result, level) if at == level else []
        if len(chosen):
            yield organ, trend, kind, result, chosen


def panels_for(results, level, pick, with_fdr, max_rules=None, cells=None):
    """Per organ and trend, titled 'organ | trend' (its TITLES name when it has one), a block per kind with
    the first `max_rules` (all when None) rules `pick` chooses; with `cells`, each rule's class."""
    panels = {}
    for organ, trend, kind, result, chosen in picked(results, level, pick):
        chosen = chosen[:max_rules]
        block = {"found": result["found"].loc[chosen], "tested": result["tested"].loc[chosen], "kind": kind,
                 "unit": level, "units": result["units"]}
        if cells is not None:
            block["classes"] = dh.rule_class(chosen, cells)
        panels.setdefault(f"{organ} | {TITLES.get(trend, trend)}", []).append(
            block | ({"fdr": result["fdr"].loc[chosen]} if with_fdr else {}))
    return panels
