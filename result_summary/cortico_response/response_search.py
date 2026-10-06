"""Can the rules tell steroid responders from non-responders?

Every setting in the grid below is one run, done twice: one row per patient, and one row
per FOV. Either way each patient is left out in turn and predicted by a model trained on
all the other patients. Each unit gets its own runs_<unit>.csv and bar plot in output/.

    python response_search.py          # run the grid, then plot
    python response_search.py --plot   # plot the saved runs again without rerunning
"""
import os
import sys
import itertools

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegressionCV
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import LeaveOneGroupOut, StratifiedKFold, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
import data_helper as dh
import vis_helper as vh

ORGAN = "Duodenum"
OUT_DIR = os.path.join(HERE, "output")
LABEL = "Cortico Response"
RESPONDER, NON_RESPONDER = "Responder", "Non-responder"
UNITS = {"patient": "Biopsy", "FOV": "FOV"}

# name: (kinds of rule, pairwise only)
RULE_SETS = {
    "Attraction, pairwise": (["attracts"], True),
    "Attraction, all sizes": (["attracts"], False),
    "Attraction + avoidance, pairwise": (["attracts", "avoids"], True),
    "Attraction + avoidance, all sizes": (["attracts", "avoids"], False),
}
COMPOSITION = "Cell composition"
FEATURES = {"share of FOVs": None, "mean lift": "Lift", "mean conviction": "Conviction"}
MIN_PATIENTS = [0, 0.1, 0.2, 0.3]

def logistic(l1_ratio):
    """Logistic regression, its penalty strength tuned on the training rows.
    l1_ratio 0 = ridge (every rule, small weights), 1 = lasso (most weights exactly 0)."""
    return lambda: make_pipeline(
        StandardScaler(),
        LogisticRegressionCV(Cs=10, cv=StratifiedKFold(5), scoring="balanced_accuracy",
                             l1_ratios=(l1_ratio,), solver="liblinear",
                             class_weight="balanced", max_iter=5000,
                             use_legacy_attributes=False))


MODELS = {
    "Ridge logistic regression": logistic(0),
    "Lasso logistic regression": logistic(1),
    "Random forest": lambda: RandomForestClassifier(
        n_estimators=500, class_weight="balanced", random_state=0, n_jobs=-1),
}
MODEL_COLORS = {"Ridge logistic regression": "#33658A", "Lasso logistic regression": "#86BBD8",
                "Random forest": "#E0A33B"}


# ---------------------------------------------------------------------------
# Tables: one row per patient (Biopsy) or per FOV
# ---------------------------------------------------------------------------

def rule_table(rules, fovs, unit, feature, min_patients):
    """Unit x rule. 'share of FOVs': the share of the unit's FOVs where the rule was
    mined (0 or 1 for a single FOV). 'mean lift' / 'mean conviction': that metric
    averaged over the unit's FOVs, 1 where the rule was not mined. Keeps rules mined
    in at least min_patients of the patients."""
    n_fovs = fovs.groupby(unit)["FOV"].nunique()
    by_fov = fovs.set_index("FOV", drop=False)
    metric = FEATURES[feature]
    rules = rules.assign(unit=rules["FOV"].map(by_fov[unit]),
                         patient=rules["FOV"].map(by_fov["Biopsy"]),
                         value=1.0 if metric is None else rules[metric] - 1)

    patients_with = rules.groupby("Rule")["patient"].nunique() / fovs["Biopsy"].nunique()
    kept = patients_with.index[patients_with >= min_patients]
    rules = rules[rules["Rule"].isin(kept)]

    table = (rules.pivot_table(index="unit", columns="Rule", values="value", aggfunc="sum")
             .reindex(n_fovs.index).fillna(0).div(n_fovs, axis=0))
    return table if metric is None else table + 1


def composition_table(cells, fovs, unit):
    """Unit x cell type: the share of each cell type among the unit's cells."""
    cells = cells[cells["fov"].isin(fovs["FOV"]) & ~cells["cell type"].isin(dh.DROP_CELLS)]
    counts = pd.crosstab(cells["fov"].map(fovs.set_index("FOV", drop=False)[unit]), cells["cell type"])
    return counts.div(counts.sum(axis=1), axis=0)


def all_tables(rules, cells, fovs, unit):
    """(rule set, feature, min patients, table) for every setting in the grid."""
    yield COMPOSITION, "", np.nan, composition_table(cells, fovs, unit)
    for (name, (kinds, pairwise)), feature, share in itertools.product(
            RULE_SETS.items(), FEATURES, MIN_PATIENTS):
        chosen = rules["Kind"].isin(kinds)
        if pairwise:
            chosen &= rules["Rule_Type"] == "pairwise"
        yield name, feature, share, rule_table(rules[chosen], fovs, unit, feature, share)


# ---------------------------------------------------------------------------
# Runs
# ---------------------------------------------------------------------------

def score(table, is_responder, patient_of, model):
    """Predict each patient's rows from all the other patients, and count how many were
    right."""
    y = is_responder.loc[table.index].to_numpy()
    predicted = cross_val_predict(MODELS[model](), table.to_numpy(), y, cv=LeaveOneGroupOut(),
                                  groups=patient_of.loc[table.index].to_numpy())
    return {
        "balanced_accuracy": 100 * balanced_accuracy_score(y, predicted),
        "responders_right": 100 * (predicted[y == 1] == 1).mean(),
        "non_responders_right": 100 * (predicted[y == 0] == 0).mean(),
        "n_responders": int(y.sum()),
        "n_non_responders": int((y == 0).sum()),
    }


def load():
    """Rules, cells and FOVs of this organ's patients with a known response."""
    cells, fovs, _ = dh.load_spatial_data()
    fovs = fovs[(fovs["Organ"] == ORGAN) & fovs[LABEL].isin([RESPONDER, NON_RESPONDER])]
    rules = dh.load_results(rule_max_items=4, kind=None)
    rules = rules[rules["FOV"].isin(fovs["FOV"])]
    rules["Rule"] = rules["Antecedents"] + " -> " + rules["Consequents"] + " " + rules["Kind"]
    finite = rules["Conviction"].replace(np.inf, np.nan)
    rules["Conviction"] = finite.fillna(finite.max())
    return rules, cells, fovs


def runs_csv(unit):
    return os.path.join(OUT_DIR, f"runs_{unit}.csv")


def weights_of(table, is_responder, model):
    """The model's weight for each rule, fitted once on all rows."""
    fitted = MODELS[model]().fit(table.to_numpy(), is_responder.loc[table.index].to_numpy())
    return pd.Series(fitted[-1].coef_[0], index=table.columns, name="weight")


def run_unit(rules, cells, fovs, unit):
    by_unit = fovs.groupby(UNITS[unit])
    is_responder = by_unit[LABEL].first().eq(RESPONDER).astype(int)
    patient_of = by_unit["Biopsy"].first()
    rows, weights = [], {}
    for name, feature, share, table in all_tables(rules, cells, fovs, UNITS[unit]):
        for model in MODELS:
            row = {"organ": ORGAN, "unit": unit, "rule_set": name, "feature": feature,
                   "min_patients": share, "model": model, "n_features": table.shape[1],
                   **score(table, is_responder, patient_of, model)}
            if "logistic" in model:
                weights[len(rows)] = weights_of(table, is_responder, model)
            print(f"{unit:8} {name:34} {feature:16} {share:>4} {model:26} "
                  f"{row['n_features']:>6} features  {row['balanced_accuracy']:5.1f}%")
            rows.append(row)
    runs = pd.DataFrame(rows).round(1)
    os.makedirs(OUT_DIR, exist_ok=True)
    runs.to_csv(runs_csv(unit), index=False)
    print(f"saved {runs_csv(unit)}")
    save_best_weights(runs, weights, unit)
    return runs


def save_best_weights(runs, weights, unit):
    """The weights of the best-scoring logistic run, biggest pull first, zeros left out."""
    best = runs.loc[list(weights)]["balanced_accuracy"].idxmax()
    chosen = weights[best]
    chosen = chosen[chosen != 0].sort_values(key=abs, ascending=False)
    path = os.path.join(OUT_DIR, f"weights_{unit}.csv")
    setting = runs.loc[best, ["rule_set", "feature", "min_patients", "model", "balanced_accuracy"]]
    chosen.rename_axis("rule").reset_index().assign(**setting).to_csv(path, index=False)
    print(f"saved {path}  ({', '.join(map(str, setting))})")


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------

def setting_label(row):
    if row["rule_set"] == COMPOSITION:
        return row["rule_set"]
    return f"{row['rule_set']}  ·  {row['feature']}  ·  in ≥{row['min_patients']:.0%} of patients"


def plot_runs(runs):
    """One row per setting, one bar per model: % predicted right, 50% = guessing."""
    runs = runs.assign(setting=[setting_label(r) for _, r in runs.iterrows()])
    settings = list(dict.fromkeys(runs["setting"]))
    first = runs.iloc[0]
    unit = first["unit"]
    height = 1.2 + 0.42 * len(settings)

    with plt.rc_context(vh.PANEL_FONTS):
        fig, ax = plt.subplots(figsize=(vh.TEXT_WIDTH, height))
        bar = 0.8 / len(MODELS)
        for i, model in enumerate(MODELS):
            values = runs[runs["model"] == model].set_index("setting")["balanced_accuracy"]
            y = np.arange(len(settings)) + (i - (len(MODELS) - 1) / 2) * bar
            ax.barh(y, values.reindex(settings), height=bar * 0.92,
                    color=MODEL_COLORS[model], label=model)
            for yy, v in zip(y, values.reindex(settings)):
                ax.text(v + 0.8, yy, f"{v:.0f}", va="center", fontsize=6.5, color=vh.INK)

        ax.axvline(50, color=vh.ZERO, lw=0.9, ls="--")
        ax.text(50, -0.75, "guessing", ha="center", va="bottom", fontsize=6.5, color=vh.ZERO)
        ax.set_yticks(range(len(settings)), settings)
        ax.invert_yaxis()
        ax.set_xlim(0, 100)
        ax.set_xlabel(f"{unit[0].upper() + unit[1:]}s predicted right (%, average of responders and non-responders)")
        vh.tidy_axes(ax, grid="x")
        ax.legend(loc="lower right", frameon=False)
        vh.figure_titles(
            fig, f"Predicting steroid response, one row per {unit}", organ=ORGAN,
            subtitle=(f"{first['n_responders']} responder {unit}s, {first['n_non_responders']} "
                      f"non-responder {unit}s; each patient left out in turn"),
        )
        vh.save_figure(fig, f"response_all_runs_{unit}", figure_dir=OUT_DIR)
        plt.close(fig)


if __name__ == "__main__":
    data = None if "--plot" in sys.argv else load()
    for unit in UNITS:
        plot_runs(pd.read_csv(runs_csv(unit)) if data is None else run_unit(*data, unit))
