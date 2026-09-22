"""One rule's metrics in one field, recomputed by the mining code itself.

A field that carries no rule tells you nothing on its own: the rule may have been
measured and then dropped at a gate, or never built at all. This asks the mining
package the same question it asks during a run, for a rule and a field of your
choosing, so a "no rule" panel can say which number fell short.

Nothing here reimplements the mining. Patches, transactions, support, the metrics
and the verdict all come from `spatial_association_rules`, so a change there changes
this too. Recomputing stored rules reproduces their numbers to 1e-16.
"""
import ast
import json
from dataclasses import fields as dataclass_fields
from functools import lru_cache
from pathlib import Path
from typing import NamedTuple

import numpy as np
import pandas as pd

from spatial_association_rules import rules as mining_rules
from spatial_association_rules import transactions as mining_transactions
from spatial_association_rules.attraction import attracts
from spatial_association_rules.avoidance import avoids
from spatial_association_rules.settings import Method, Settings, Weighting

ATTRACTS = "attracts"


def settings_of_run(run_dir):
    """The Settings a finished run recorded, so a recomputation matches that run."""
    config = json.loads((Path(run_dir) / "run_config.json").read_text())
    known = {field.name for field in dataclass_fields(Settings)}
    values = {name: value for name, value in config["settings"].items() if name in known}
    values["weighting"] = Weighting(values["weighting"])
    values["method"] = Method(values["method"])
    return Settings(**values)


@lru_cache(None)
def _settings_of(run_dir):
    return settings_of_run(run_dir)


def current_settings():
    """The Settings of the run the notebooks are reading, or None when unrecorded."""
    import data_helper

    try:
        return _settings_of(str(Path(data_helper.RESULT_CSV_PATH).parent))
    except (OSError, KeyError, ValueError) as problem:
        print(f"No usable run_config.json for this run ({problem}); "
              "fields without the rule cannot be explained.")
        return None


_STORED_METRICS = {"lift": "Lift", "conf": "Confidence", "conv": "Conviction",
                   "sup": "Support", "lev": "Leverage"}


@lru_cache(None)
def _mined_rows(run_dir, max_items=4):
    """Every rule the run measured in a field, before the FDR and redundancy filters."""
    import data_helper

    rules = data_helper._read_rules(run_dir, max_items)
    if rules.empty:
        return {}
    found = {}
    for row in rules.to_dict("records"):
        found.setdefault(
            (str(row["Antecedents"]), str(row["Consequents"]), row["FOV"]), row)
    return found


def stored_metrics(row):
    """The numbers the run already wrote for this rule in this field."""
    return {short: float(row[column]) for short, column in _STORED_METRICS.items()
            if pd.notna(row.get(column))}


def items_of(stored):
    """A stored side of a rule as its role-carrying items, e.g. ['Goblet_CENTER']."""
    if isinstance(stored, str):
        return tuple(ast.literal_eval(stored))
    return tuple(stored)


def transactions_of(cells, fov, settings, label_col="cell type",
                    coord_cols=("x_um", "y_um"), fov_col="fov"):
    """The transactions the mining would build for this one field."""
    block = cells[cells[fov_col] == fov]
    if block.empty:
        return []
    coords = block[list(coord_cols)].to_numpy(dtype=float)
    labels = block[label_col].to_numpy(dtype=object)
    patches = mining_transactions.find_patches(coords, settings)
    measured = mining_transactions.measure_patches(patches, coords, settings)
    built, _ = mining_transactions.build_transactions(measured, labels, settings)
    return built


class Counted(NamedTuple):
    """The cells a rule was counted on in one field, as positions in that field."""

    centers: tuple          # the patch centers that carry the whole rule
    neighbors: tuple        # the neighbors that complete it, centers left out
    antecedent: tuple       # the same cells, by the side of the arrow they fill
    consequent: tuple

    @property
    def total(self):
        return len(self.centers) + len(self.neighbors)


def counted_cells(antecedent_items, consequent_items, cells, fov, settings=None,
                  label_col="cell type", coord_cols=("x_um", "y_um"), fov_col="fov"):
    """The cells that make a rule hold in one field.

    A patch counts when its center carries the rule's center item and every other
    item is on one of its neighbors, which is the condition the mining counts
    support by. The cells of those patches come back both ways: by the role they
    play, and by the side of the arrow they fill.
    """
    settings = current_settings() if settings is None else settings
    block = cells[cells[fov_col] == fov]
    if settings is None or block.empty:
        return Counted((), (), (), ())

    sides = {"antecedent": items_of(antecedent_items),
             "consequent": items_of(consequent_items)}
    middle = {side: {mining_transactions.strip_role(item) for item in items
                     if mining_transactions.is_center(item)}
              for side, items in sides.items()}
    around = {side: {mining_transactions.strip_role(item) for item in items
                     if not mining_transactions.is_center(item)}
              for side, items in sides.items()}

    coords = block[list(coord_cols)].to_numpy(dtype=float)
    labels = block[label_col].to_numpy(dtype=object)
    patches = mining_transactions.measure_patches(
        mining_transactions.find_patches(coords, settings), coords, settings)

    center_types = middle["antecedent"] | middle["consequent"]
    neighbor_types = around["antecedent"] | around["consequent"]

    centers, neighbors = set(), set()
    taking = {side: set() for side in sides}
    for patch in patches:
        if not center_types <= {labels[patch.center]}:
            continue
        if mining_transactions.is_crowded_by_one_type(labels[patch.members],
                                                      settings.max_one_type_share):
            continue
        if not neighbor_types <= set(labels[patch.neighbors]):
            continue
        centers.add(patch.center)
        neighbors.update(patch.neighbors[np.isin(labels[patch.neighbors],
                                                 list(neighbor_types))])
        for side in sides:
            if labels[patch.center] in middle[side]:
                taking[side].add(patch.center)
            taking[side].update(patch.neighbors[np.isin(labels[patch.neighbors],
                                                        list(around[side]))])
    return Counted(tuple(sorted(centers)), tuple(sorted(neighbors - centers)),
                   tuple(sorted(taking["antecedent"])),
                   tuple(sorted(taking["consequent"])))


def _measure(matrix, index, antecedents, consequents):
    """One rule against one field's weight matrix."""
    missing = [item for item in antecedents + consequents if item not in index]
    joint = mining_rules.support_of(frozenset(antecedents + consequents), matrix, index)
    ant_support = mining_rules.support_of(frozenset(antecedents), matrix, index)
    con_support = mining_rules.support_of(frozenset(consequents), matrix, index)
    confidence, lift, leverage, conviction = mining_rules.metrics(
        joint, ant_support, con_support)

    return {
        "support": joint,
        "antecedent_support": ant_support,
        "consequent_support": con_support,
        "confidence": confidence,
        "lift": lift,
        "leverage": leverage,
        "conviction": conviction,
        "patches": matrix.shape[0],
        "missing_items": tuple(missing),
    }


class Fields:
    """One field's patches, built once and reused for every rule asked about it.

    Building a field's transactions costs far more than measuring a rule in them
    (~68 ms against ~0.02 ms), so anything comparing rules over many fields should
    go through one of these rather than call `metrics_in_fov` per rule.
    """

    def __init__(self, cells, settings=None, **columns):
        self.cells = cells
        self.settings = current_settings() if settings is None else settings
        self._columns = columns
        self._built = {}

    def _matrix(self, fov):
        if fov not in self._built:
            built = transactions_of(self.cells, fov, self.settings, **self._columns)
            self._built[fov] = (mining_rules.weight_matrix(built) if built else None)
        return self._built[fov]

    def metrics(self, antecedents, consequents, fov):
        """The same dict as `metrics_in_fov`, off the shared matrix."""
        if self.settings is None:
            return None
        built = self._matrix(fov)
        if built is None:
            return None
        return _measure(*built, items_of(antecedents), items_of(consequents))

    def lift(self, antecedents, consequents, fov):
        """Just the lift, for a figure that plots it."""
        measured = self.metrics(antecedents, consequents, fov)
        return np.nan if measured is None or measured["missing_items"] else measured["lift"]


def metrics_in_fov(antecedents, consequents, cells, fov, settings, **columns):
    """Every descriptive metric for one rule in one field, or None when unmeasurable.

    The numbers are the mining's own: same patches, same support, same formulas.
    No p-value is returned: it needs the whole shuffled run for that field, and an
    FDR needs every rule tested there, so neither is defined for one rule alone.
    """
    built = transactions_of(cells, fov, settings, **columns)
    if not built:
        return None
    return _measure(*mining_rules.weight_matrix(built),
                    items_of(antecedents), items_of(consequents))


def would_be_mined(measured, settings, kind=ATTRACTS):
    """The mining's own verdict on these numbers, from attracts() or avoids()."""
    if measured is None or measured["patches"] == 0:
        return False
    judge = attracts if kind == ATTRACTS else avoids
    measures = (measured["confidence"], measured["lift"],
                measured["leverage"], measured["conviction"])
    return bool(judge(settings, measured["support"], measured["antecedent_support"],
                      measured["consequent_support"], measures, measured["patches"]))


def _shortfall(value, limit):
    """value and limit, with just enough digits that they cannot print the same."""
    for digits in (3, 4, 5, 6):
        shown, bar = f"{value:.{digits}g}", f"{limit:.{digits}g}"
        if shown != bar:
            return shown, bar
    return f"{value:.6g}", f"{limit:.6g}"


def _attraction_checks(measured, settings):
    n = measured["patches"]
    strong = (settings.strong_confidence is not None
              and measured["confidence"] >= settings.strong_confidence)
    needed = (settings.min_support_when_strong if strong else settings.min_support)
    floor = max(needed, settings.min_patches / n)
    counted = settings.min_patches / n > needed
    support_reason = (
        f"only {measured['support'] * n:.0f} patches, needs {settings.min_patches}" if counted
        else "sup {} < {}".format(*_shortfall(measured["support"], floor)))
    return [
        (measured["support"] >= floor, support_reason),
        (measured["lift"] >= settings.min_lift,
         "lift {} < {}".format(*_shortfall(measured["lift"], settings.min_lift))),
        (settings.min_confidence is None or measured["confidence"] >= settings.min_confidence,
         "conf {} < {}".format(*_shortfall(measured["confidence"], settings.min_confidence or 0))),
        (settings.min_leverage is None or measured["leverage"] >= settings.min_leverage,
         "lev {} < {}".format(*_shortfall(measured["leverage"], settings.min_leverage or 0))),
        (settings.min_conviction is None or measured["conviction"] >= settings.min_conviction,
         "conv {} < {}".format(*_shortfall(measured["conviction"], settings.min_conviction or 0))),
    ]


def _avoidance_checks(measured, settings):
    n = measured["patches"]
    meetings = measured["antecedent_support"] * measured["consequent_support"] * n
    return [
        (measured["antecedent_support"] * n >= settings.min_patches,
         f"only {measured['antecedent_support'] * n:.0f} patches, needs {settings.min_patches}"),
        (meetings >= settings.avoidance_min_expected_meetings,
         f"only {meetings:.0f} expected meetings, needs {settings.avoidance_min_expected_meetings}"),
        (settings.avoidance_max_lift is None or measured["lift"] <= settings.avoidance_max_lift,
         "lift {} > {}".format(*_shortfall(measured["lift"], settings.avoidance_max_lift or 0))),
        (settings.avoidance_max_leverage is None
         or measured["leverage"] <= settings.avoidance_max_leverage,
         f"lev {measured['leverage']:.3g} > {settings.avoidance_max_leverage}"),
    ]


def why_not_mined(measured, settings, kind=ATTRACTS):
    """Short reason this rule is not here, naming the threshold it misses."""
    if measured is None or measured["patches"] == 0:
        return "no patches in this field"
    if measured["missing_items"]:
        names = dict.fromkeys(mining_transactions.strip_role(item)
                              for item in measured["missing_items"])
        return "absent here: " + ", ".join(names)

    checks = (_attraction_checks(measured, settings) if kind == ATTRACTS
              else _avoidance_checks(measured, settings))
    missed = [reason for ok, reason in checks if not ok]
    if missed:
        # Two is all a panel caption can carry; they are in the order the run applies them.
        shown = "; ".join(missed[:2])
        return "below gate: " + (shown + ", and more" if len(missed) > 2 else shown)
    if would_be_mined(measured, settings, kind):
        return "clears every gate; dropped later, by FDR or as redundant"
    return "does not pass the mining test"


def caption_metrics(measured):
    """The measured numbers under the short names the FOV figures use.

    A rule naming a cell type this field does not have has no numbers worth
    printing: every one of them is zero by construction, and the reason says so.
    """
    if measured is None or measured["missing_items"]:
        return {}
    return {"lift": measured["lift"], "conf": measured["confidence"],
            "conv": measured["conviction"], "sup": measured["support"],
            "lev": measured["leverage"]}


def explain_missing(examples, cells, settings=None, fields=None, max_fdr=None, **columns):
    """Fill the metrics and the missed gate into every field that carries no rule.

    A field the run already measured keeps the run's own numbers and its FDR; only
    a field the run never wrote a row for is measured again here.

    Pass a shared `Fields` when several rules are explained over the same fields;
    each field is then built once rather than once per rule.
    """
    import data_helper

    fields = Fields(cells, settings, **columns) if fields is None else fields
    if fields.settings is None or examples.empty or "state" not in examples.columns:
        return examples
    mined = _mined_rows(str(data_helper.RESULT_CSV_PATH))
    examples = examples.copy()
    if "why" not in examples.columns:
        examples["why"] = None
    for position in examples.index[examples["state"].eq(0)]:
        row = examples.loc[position]
        written = mined.get((str(row["antecedent_items"]), str(row["consequent_items"]),
                             row["FOV"]))
        if written is not None:
            fdr = pd.to_numeric(written.get("Individual_FDR"), errors="coerce")
            examples.at[position, "metrics"] = stored_metrics(written)
            examples.at[position, "fdr"] = fdr
            examples.at[position, "why"] = (
                "measured here, not significant"
                if max_fdr is not None and pd.notna(fdr) and fdr > max_fdr
                else "measured here, dropped by a later filter")
            continue
        measured = fields.metrics(row["antecedent_items"], row["consequent_items"],
                                  row["FOV"])
        examples.at[position, "metrics"] = caption_metrics(measured)
        examples.at[position, "why"] = why_not_mined(
            measured, fields.settings, row.get("kind", ATTRACTS))
    return examples
