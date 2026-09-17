"""Paired literature checks: mined occurrence and independent spatial diagnostics.

The coordinate diagnostic uses every cell center, not only mined patches. Its
random-neighbor-label expectation is deliberately distinct from the mining null.
"""
import ast
import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.spatial import cKDTree
from scipy.stats import binomtest, hypergeom
from statsmodels.stats.multitest import multipletests

import data_helper as dh
import differential_vis as dv
import compare_rules as cr


def load_pairs():
    return json.loads(Path(__file__).with_name("literature_pairs.json").read_text())


def pair_specs(pairs):
    return [dict(pair, organ=organ, pair=f"{pair['a']} / {pair['b']}")
            for pair in pairs for organ in pair["organs"]]


def prepare_directional_rules(raw, cells, metadata, pairs, min_cells=20):
    """Keep CENTER -> NEIGHBOR explicitly; preserve unmined reverse directions."""
    ant = raw["Antecedents"].map(ast.literal_eval)
    con = raw["Consequents"].map(ast.literal_eval)
    keep = ant.map(lambda x: len(x) == 1 and x[0].endswith("_CENTER"))
    keep &= con.map(lambda x: len(x) == 1 and x[0].endswith("_NEIGHBOR"))
    rules = dh.prepare_rules(raw.loc[keep])
    definitions = {}
    for pair in pairs:
        a, b = pair["a"], pair["b"]
        definitions[f"{a} -> {b}"] = [a, b]
        definitions[f"{b} -> {a}"] = [a, b]
    rules = rules[rules["Clean_Rule"].isin(definitions)
                  & rules["FOV"].isin(metadata["FOV"])].copy()
    if rules.duplicated(["Clean_Rule", "FOV"]).any():
        raise ValueError("Multiple states for one centered rule/FOV: inspect the run.")
    states = rules.pivot(index="Clean_Rule", columns="FOV", values="state")
    states = states.reindex(index=list(definitions), columns=metadata["FOV"]).fillna(0)
    eligible = dh.eligible_fovs(definitions, cells, min_cells).reindex(
        index=states.index, columns=states.columns, fill_value=False)
    eligible.attrs["min_cells"] = min_cells
    return rules, states, eligible, int((~keep).sum())


def neighbor_rates(coords, labels, a, b, radius):
    """Observed and exact finite-population expected probability of >=1 neighbor.

    Conditional on a center of type A and its degree k, place the observed n_B
    labels uniformly among the other N-1 cells: P(hit) = hypergeom.sf(0,N-1,n_B,k).
    Self is excluded. Isolated centers contribute observed=expected=0.
    """
    labels = np.asarray(labels)
    coords = np.asarray(coords, dtype=float)
    if not len(coords) or not (labels == a).any() or not (labels == b).any():
        return None
    tree = cKDTree(coords)
    degree = tree.query_ball_point(coords, radius, return_length=True) - 1
    values = {}
    for direction, center, neighbor in (("ab", a, b), ("ba", b, a)):
        targets = labels == neighbor
        centers = labels == center
        hits = cKDTree(coords[targets]).query_ball_point(
            coords[centers], radius, return_length=True)
        observed = float((hits > 0).mean())
        expected = float(hypergeom.sf(
            0, len(labels) - 1, int(targets.sum()), degree[centers]).mean())
        values.update({f"observed_{direction}": observed,
                       f"expected_{direction}": expected,
                       f"excess_{direction}": observed - expected,
                       f"centers_{direction}": int(centers.sum())})
    values["difference"] = values["excess_ab"] - values["excess_ba"]
    return values


def spatial_diagnostics(cells, metadata, pairs, eligible, radius=25, score="Clinical score", states=None):
    records = []
    info = metadata.set_index("FOV")
    for fov, block in cells.groupby("fov", sort=False):
        if fov not in info.index:
            continue
        organ = info.at[fov, "Organ"]
        for pair in pairs:
            a, b = pair["a"], pair["b"]
            if organ not in pair["organs"] or not eligible.at[f"{a} -> {b}", fov]:
                continue
            values = neighbor_rates(block[["x_um", "y_um"]].to_numpy(),
                                    block["cell type"].to_numpy(), a, b, radius)
            if states is not None:
                values["mined_difference"] = (
                    int(states.at[f"{a} -> {b}", fov] == 1)
                    - int(states.at[f"{b} -> {a}", fov] == 1))
            records.append(dict(FOV=fov, pair=f"{a} / {b}", **values))
    columns = ["FOV", "pair", "observed_ab", "expected_ab", "excess_ab",
               "centers_ab", "observed_ba", "expected_ba", "excess_ba",
               "centers_ba", "difference"]
    if states is not None:
        columns.append("mined_difference")
    return pd.DataFrame(records, columns=columns).merge(
        metadata[["FOV", "Biopsy", "PatientID", "Organ", score]],
        on="FOV", validate="many_to_one")


def patient_values(diagnostics, stages, score="Clinical score"):
    """Equal biopsy means within patients; pooled and within-stage estimates."""
    metrics = ["observed_ab", "expected_ab", "excess_ab", "observed_ba",
               "expected_ba", "excess_ba", "difference"]
    if "mined_difference" in diagnostics:
        metrics.append("mined_difference")
    records = []
    for stage in ["All", *stages]:
        part = diagnostics if stage == "All" else diagnostics[diagnostics[score] == stage]
        biopsy = part.groupby(["pair", "Organ", "PatientID", "Biopsy"])[metrics].mean()
        patient = biopsy.groupby(["pair", "Organ", "PatientID"])[metrics].mean().reset_index()
        records.append(patient.assign(stage=stage))
    return pd.concat(records, ignore_index=True)


def sign_evidence(patients, min_patients=5):
    """Two-sided patient sign tests; one BH family for all pairs/organs/stages/endpoints."""
    rows = []
    for (pair, organ, stage), part in patients.groupby(["pair", "Organ", "stage"]):
        endpoints = ["excess_ab", "excess_ba", "difference"]
        if "mined_difference" in part:
            endpoints.append("mined_difference")
        for endpoint in endpoints:
            x = part[endpoint].dropna().to_numpy()
            nonzero = x[~np.isclose(x, 0, atol=1e-12)]
            n_pos = int((nonzero > 0).sum())
            tested = len(nonzero) >= min_patients
            p = binomtest(n_pos, len(nonzero), 0.5).pvalue if tested else np.nan
            rows.append(dict(pair=pair, Organ=organ, stage=stage, endpoint=endpoint,
                             patients=len(x), nonzero_patients=len(nonzero),
                             positive=n_pos, negative=len(nonzero)-n_pos,
                             mean_pp=100*x.mean(), median_pp=100*np.median(x), p_value=p))
    result = pd.DataFrame(rows)
    if result.empty:
        return pd.DataFrame(columns=["pair", "Organ", "stage", "endpoint", "patients",
                                     "nonzero_patients", "positive", "negative", "mean_pp",
                                     "median_pp", "p_value", "fdr"])
    tested = result["p_value"].notna()
    result["fdr"] = np.nan
    if tested.any():
        result.loc[tested, "fdr"] = multipletests(result.loc[tested, "p_value"], method="fdr_bh")[1]
    return result


def occurrence_table(states, eligible, metadata, specs, stages, score):
    rows = []
    for spec in specs:
        a, b, organ = spec["a"], spec["b"], spec["organ"]
        for stage in ["All", *stages]:
            meta = metadata[metadata["Organ"] == organ]
            if stage != "All":
                meta = meta[meta[score] == stage]
            for left, right in ((a, b), (b, a)):
                rule = f"{left} -> {right}"
                allowed = meta[eligible.loc[rule].reindex(meta["FOV"]).to_numpy()]
                x = states.loc[rule, allowed["FOV"]]
                hits = allowed[allowed["FOV"].isin(x.index[x == 1])]
                n = len(allowed)
                rows.append(dict(pair=spec["pair"], organ=organ, stage=stage, rule=rule,
                                 eligible=n, ineligible=len(meta)-n,
                                 biopsies=allowed["Biopsy"].nunique(),
                                 patients=allowed["PatientID"].nunique(),
                                 attraction=int((x == 1).sum()), avoidance=int((x == -1).sum()),
                                 no_rule=int((x == 0).sum()),
                                 attraction_patients=hits["PatientID"].nunique(),
                                 attraction_pct=100*(x == 1).mean() if n else np.nan))
    return pd.DataFrame(rows)


def directional_specs(specs):
    """One index per direction, keeping each literature pair adjacent."""
    return [dict(spec, rule=f"{left} -> {right}")
            for spec in specs
            for left, right in ((spec["a"], spec["b"]), (spec["b"], spec["a"]))]


def plot_known_rule(spec, rules, states, eligible, metadata, evidence, stages,
                    score="Clinical score", save=None):
    """Familiar prevalence, Lift and state panels, pooled and by stage."""
    groups = ["All", *stages]
    # Duplicate metadata only for display; pooled values still count each FOV once.
    shown = pd.concat([metadata.assign(**{score: "All"}), metadata], ignore_index=True)
    plot_spec = dict(spec, score=score)
    with plt.rc_context(dv._PANEL_FONTS):
        fig, axes = plt.subplots(3, 1, figsize=(dv._TEXT_WIDTH, 6.6),
                                 gridspec_kw={"height_ratios": [1, 1.1, 1], "hspace": 0.70})
        dv._prevalence_panel(axes[0], states, eligible, shown, plot_spec, groups, bars=True)
        dv._metric_panel(axes[1], rules, eligible, shown, plot_spec, groups, connect=False)
        counts = dv._stage_state_counts(states, eligible, shown, spec["rule"],
                                        spec["organ"], score, groups)
        dv._draw_state_bars(axes[2], *counts, groups, show_net=False)
        for ax, title in zip(axes, ("how common is the rule?", "how strong is it? (Lift)",
                                   "every FOV, including ineligible fields")):
            ax.set_title(title, loc="left", color="#5F5D58", pad=6)
            ax.axvline(0.5, color="#B7B4AE", lw=0.8, ls=":")
        axes[1].set_xlabel("rule-bearing / eligible FOVs · dots = FOVs · diamonds = medians",
                          fontsize=7.5, labelpad=3)
        handles = [dv.Line2D([0], [0], marker="s", linestyle="none", markersize=6,
                             color=dv._STATE_COLORS[name], label=name)
                   for name in dv._STATE_ORDER]
        axes[2].legend(handles=handles, ncol=1, frameon=False, loc="center left",
                       bbox_to_anchor=(1.01, 0.5), fontsize=7)
        fig.suptitle(f"{spec['organ']} · {spec['rule'].replace(' -> ', ' → ')}",
                     fontsize=11.5, y=0.995)
        threshold = eligible.attrs.get("min_cells", 20)
        fig.text(0.5, 0.959,
                 f"All = pooled organ · {score} · ≥{threshold} cells/type",
                 ha="center", va="top", fontsize=7.2, color="#706E68")
        fig.align_ylabels(axes)
        fig.subplots_adjust(top=0.88, bottom=0.10, left=0.14, right=0.76)
        dv._finish(fig, save)


def show_selected(index, specs, rules, states, eligible, cells, metadata, evidence,
                  occurrences, stages, score):
    from IPython.display import Markdown, display
    import differential_stats as ds

    if not 0 <= index < len(specs):
        print(f"No rule at index {index}; {len(specs)} directional checks.")
        return
    spec = specs[index]
    display(Markdown(f"**[{index}] {spec['organ']} · {spec['rule']}** — {spec['expectation']}\n\n"
                     f"{spec['signaling']} [Source]({spec['source']}) "
                     f"{spec['limitation']} Spatial arrows do not establish signaling direction."))
    rows = occurrences[(occurrences.rule == spec["rule"]) & (occurrences.organ == spec["organ"])]
    display(rows[["stage", "eligible", "ineligible", "patients", "attraction", "avoidance",
                  "attraction_patients"]].rename(columns={"patients": "eligible_patients"}))
    stem = f"biology_{spec['organ']}_{spec['rule'].replace(' -> ', '_to_')}"
    plot_known_rule(spec, rules, states, eligible, metadata, evidence, stages, score, stem + "_summary.pdf")
    examples = ds.representative_fovs(rules, eligible, metadata, spec["rule"], spec["organ"],
                                      score, stages, "Lift")
    cr.plot_rule_fovs(examples, stages, cells, metadata, spec["organ"], score,
                      min_cells=eligible.attrs.get("min_cells", 20),
                      save=stem + "_fovs.pdf")
    return examples
