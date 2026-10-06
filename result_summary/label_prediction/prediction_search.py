"""Can the rules predict a biopsy label, such as steroid response?

The pieces run_predictions.py puts together: the tables (one row per patient or per FOV),
the models, one leave-one-out run, and the saved runs, weights and plot.

A prediction is a dict:
    target   : the biopsy column to predict
    value    : None for the column's two values; a value for it vs everything else
    positive : with value None, the "+" side (weights above 0 push toward it)
    only_patients_with_muscle_and_epithelium : Azulay et al.'s filter for their NRM model
"""
import os
import re
import sys
import itertools

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegressionCV
from sklearn.metrics import balanced_accuracy_score, roc_auc_score
from sklearn.model_selection import LeaveOneGroupOut, StratifiedKFold, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
import data_helper as dh
import vis_helper as vh

ORGAN = "Duodenum"
OUT_DIR = os.path.join(HERE, "output")
MIN_EPITHELIAL_CELLS = 50     # with the muscle filter on, a patient needs more than this many
UNIT_COLUMNS = {"patient": "Biopsy", "FOV": "FOV"}

# name: (kinds of rule, pairwise only)
RULE_SETS = {
    "Attraction, pairwise": (["attracts"], True),
    "Attraction, all sizes": (["attracts"], False),
    "Attraction + avoidance, pairwise": (["attracts", "avoids"], True),
    "Attraction + avoidance, all sizes": (["attracts", "avoids"], False),
}
COMPOSITION = "Cell composition"
SEVERITY = ["Pathological score", "Clinical score"]
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


AZULAY = "Azulay: 2 PCA + linear SVM"
MODELS = {
    "Ridge logistic regression": logistic(0),
    "Lasso logistic regression": logistic(1),
    "Random forest": lambda: RandomForestClassifier(
        n_estimators=500, class_weight="balanced", random_state=0, n_jobs=1),
    AZULAY: lambda: make_pipeline(
        StandardScaler(), PCA(n_components=2), SVC(kernel="linear", class_weight="balanced")),
}
MODEL_COLORS = {"Ridge logistic regression": "#33658A", "Lasso logistic regression": "#86BBD8",
                "Random forest": "#E0A33B", AZULAY: "#8E5BB5"}


# ---------------------------------------------------------------------------
# Data and patients
# ---------------------------------------------------------------------------

def load_data():
    """Cells, FOVs and rules of this organ's transplanted patients, each FOV with every
    biopsy column. The slow part: once."""
    cells, fovs, biopsies = dh.load_spatial_data()
    fovs = fovs[(fovs["Organ"] == ORGAN) & fovs["Biopsy_ID"].notna()]
    missing = [column for column in biopsies if column not in fovs]
    fovs = fovs.merge(biopsies[["Biopsy_ID"] + missing], on="Biopsy_ID", how="left")
    rules = dh.load_results(rule_max_items=4, kind=None)
    rules = rules[rules["FOV"].isin(fovs["FOV"])]
    rules["Rule"] = rules["Antecedents"] + " -> " + rules["Consequents"] + " " + rules["Kind"]
    finite = rules["Conviction"].replace(np.inf, np.nan)
    rules["Conviction"] = finite.fillna(finite.max())
    return cells, fovs, rules


def sides(prediction, fovs):
    """The names of the "+" and "-" sides, and each FOV's side: 1, 0, or empty when its
    patient has no answer. With a value set, an empty answer counts as "not value"."""
    target, value, positive = prediction["target"], prediction["value"], prediction["positive"]
    column = fovs[target]
    if value is not None:
        return value, f"not {value}", column.eq(value).astype(int)
    values = sorted(column.dropna().unique())
    if len(values) != 2 or positive not in values:
        raise ValueError(f"{target} has the values {values}: set value to one of them, "
                         f"or positive to one of two")
    negative = next(v for v in values if v != positive)
    return positive, negative, column.eq(positive).astype(int).where(column.notna())


def with_muscle_and_epithelium(cells, fovs):
    """The patients with muscle in at least one FOV and more than MIN_EPITHELIAL_CELLS
    epithelial cells over all their FOVs."""
    cells = cells[cells["fov"].isin(fovs["FOV"])]
    patient = cells["fov"].map(fovs.set_index("FOV")["Biopsy"])
    per_patient = (cells.assign(muscle=cells["in_Muscle"].astype(bool),
                                epithelial=cells["population"].eq("Epithel"))
                   .groupby(patient).agg(muscle=("muscle", "any"), epithelial=("epithelial", "sum")))
    kept = per_patient["muscle"] & (per_patient["epithelial"] > MIN_EPITHELIAL_CELLS)
    return per_patient.index[kept]


def patients_of(prediction, cells, fovs):
    """The FOVs of the patients this prediction uses, each with its side."""
    positive, negative, side = sides(prediction, fovs)
    fovs = fovs.assign(side=side, positive=positive, negative=negative).dropna(subset=["side"])
    if prediction["only_patients_with_muscle_and_epithelium"]:
        kept = with_muscle_and_epithelium(cells, fovs)
        print(f"{name_of(prediction)}: kept {len(kept)} of {fovs['Biopsy'].nunique()} patients "
              f"with muscle and more than {MIN_EPITHELIAL_CELLS} epithelial cells.")
        fovs = fovs[fovs["Biopsy"].isin(kept)]
    return fovs


def labels(fovs, unit):
    """Per row of the unit: its side (1 or 0) and its patient."""
    by_unit = fovs.groupby(UNIT_COLUMNS[unit])
    return by_unit["side"].first().astype(int), by_unit["Biopsy"].first()


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


def severity_table(fovs, unit, column):
    """Unit x one column: 1 when the biopsy's score is Severe, 0 when Mild."""
    return fovs.groupby(unit)[column].first().eq("Severe").astype(int).to_frame(column)


def all_tables(rules, cells, fovs, unit):
    """((rule set, feature, min patients), table) for every setting in the grid. The first
    are references with no rules: cell composition and each severity score alone."""
    unit = UNIT_COLUMNS[unit]
    rules = rules[rules["FOV"].isin(fovs["FOV"])]
    yield (COMPOSITION, "", np.nan), composition_table(cells, fovs, unit)
    for column in SEVERITY:
        yield (f"{column} only", "", np.nan), severity_table(fovs, unit, column)
    for (name, (kinds, pairwise)), feature, share in itertools.product(
            RULE_SETS.items(), FEATURES, MIN_PATIENTS):
        chosen = rules["Kind"].isin(kinds)
        if pairwise:
            chosen &= rules["Rule_Type"] == "pairwise"
        yield (name, feature, share), rule_table(rules[chosen], fovs, unit, feature, share)


# ---------------------------------------------------------------------------
# One run
# ---------------------------------------------------------------------------

def models_for(table):
    """Every model, but Azulay's only when the table has at least 2 columns to reduce to 2."""
    return [model for model in MODELS if model != AZULAY or table.shape[1] >= 2]


def score(table, is_positive, patient_of, model):
    """Predict each patient's rows from all the other patients, and count how many were
    right.

    Each row gets a lean: above 0 means "+". It is the chance of "+" minus 0.5, or for
    the SVM its signed distance from the line."""
    y = is_positive.loc[table.index].to_numpy()
    estimator = MODELS[model]()
    method = "predict_proba" if hasattr(estimator, "predict_proba") else "decision_function"
    found = cross_val_predict(estimator, table.to_numpy(), y, cv=LeaveOneGroupOut(),
                              groups=patient_of.loc[table.index].to_numpy(), method=method)
    lean = found[:, 1] - 0.5 if method == "predict_proba" else found
    predicted = (lean > 0).astype(int)
    return {
        "balanced_accuracy": 100 * balanced_accuracy_score(y, predicted),
        "auc": 100 * roc_auc_score(y, lean),
        "positive_right": 100 * (predicted[y == 1] == 1).mean(),
        "negative_right": 100 * (predicted[y == 0] == 0).mean(),
        "n_positive": int(y.sum()),
        "n_negative": int((y == 0).sum()),
    }


def weights_of(table, is_positive, model):
    """The model's weight for each rule, fitted once on all rows."""
    fitted = MODELS[model]().fit(table.to_numpy(), is_positive.loc[table.index].to_numpy())
    return pd.Series(fitted[-1].coef_[0], index=table.columns, name="weight")


def run_one(table, is_positive, patient_of, model):
    """One run: its scores, and for a logistic model its weights (else None)."""
    weights = weights_of(table, is_positive, model) if "logistic" in model else None
    return score(table, is_positive, patient_of, model), weights


# ---------------------------------------------------------------------------
# Saving
# ---------------------------------------------------------------------------

def name_of(prediction):
    """e.g. cortico_response, or cause_of_death_nrm_muscle with a value and the filter.
    Safe for a file name: '<30' becomes 'under30', '>100' becomes 'over100'."""
    muscle = "muscle" if prediction["only_patients_with_muscle_and_epithelium"] else None
    parts = (prediction["target"], prediction["value"], muscle)
    name = "_".join(str(part) for part in parts if part is not None).lower()
    name = name.replace("<", "under").replace(">", "over")
    return re.sub(r"[^a-z0-9]+", "_", name).strip("_")


def output_path(prediction, kind, unit, ext="csv"):
    """output/<kind>_<prediction>_<unit>.<ext>, e.g. runs_cortico_response_patient.csv."""
    return os.path.join(OUT_DIR, f"{kind}_{name_of(prediction)}_{unit}.{ext}")


def save(prediction, unit, runs, weights):
    """The runs CSV, the best logistic run's weights, and the plot."""
    os.makedirs(OUT_DIR, exist_ok=True)
    runs.to_csv(output_path(prediction, "runs", unit), index=False)
    print(f"saved {output_path(prediction, 'runs', unit)}")
    save_best_weights(prediction, unit, runs, weights)
    plot_runs(prediction, unit, runs)


def save_best_weights(prediction, unit, runs, weights):
    """The weights of the best-scoring logistic run, biggest pull first, zeros left out."""
    best = runs.loc[list(weights)]["balanced_accuracy"].idxmax()
    chosen = weights[best]
    chosen = chosen[chosen != 0].sort_values(key=abs, ascending=False)
    path = output_path(prediction, "weights", unit)
    setting = runs.loc[best, ["rule_set", "feature", "min_patients", "model", "balanced_accuracy", "auc"]]
    chosen.rename_axis("rule").reset_index().assign(**setting).to_csv(path, index=False)
    print(f"saved {path}  ({', '.join(map(str, setting))})")


# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------

def setting_label(row):
    if pd.isna(row["min_patients"]):
        return row["rule_set"]
    return f"{row['rule_set']}  ·  {row['feature']}  ·  in ≥{row['min_patients']:.0%} of patients"


def plot_runs(prediction, unit, runs):
    """One row per setting, one bar per model: % predicted right, 50% = guessing."""
    runs = runs.assign(setting=[setting_label(r) for _, r in runs.iterrows()])
    settings = list(dict.fromkeys(runs["setting"]))
    first = runs.iloc[0]
    height = 1.8 + 0.42 * len(settings)

    with plt.rc_context(vh.PANEL_FONTS):
        fig, ax = plt.subplots(figsize=(vh.TEXT_WIDTH, height))
        bar = 0.8 / len(MODELS)
        for i, model in enumerate(MODELS):
            values = runs[runs["model"] == model].set_index("setting")["balanced_accuracy"]
            y = np.arange(len(settings)) + (i - (len(MODELS) - 1) / 2) * bar
            ax.barh(y, values.reindex(settings), height=bar * 0.92,
                    color=MODEL_COLORS[model], label=model)
            for yy, v in zip(y, values.reindex(settings)):
                if pd.notna(v):
                    ax.text(v + 0.8, yy, f"{v:.0f}", va="center", fontsize=6.5, color=vh.INK)

        ax.axvline(50, color=vh.ZERO, lw=0.9, ls="--")
        ax.text(50, -0.75, "guessing", ha="center", va="bottom", fontsize=6.5, color=vh.ZERO)
        ax.set_yticks(range(len(settings)), settings)
        ax.invert_yaxis()
        ax.set_xlim(0, 100)
        ax.set_xlabel(f"{unit[0].upper() + unit[1:]}s predicted right "
                      f"(%, average of {first['positive']} and {first['negative']})")
        vh.tidy_axes(ax, grid="x")
        fig.subplots_adjust(bottom=0.95 / height)
        fig.legend(*ax.get_legend_handles_labels(), loc="lower center", ncol=2, frameon=False)
        vh.figure_titles(
            fig, f"Predicting {first['target']}, one row per {unit}", organ=ORGAN,
            subtitle=(f"{first['n_positive']} {first['positive']} {unit}s, {first['n_negative']} "
                      f"{first['negative']} {unit}s; each patient left out in turn"),
        )
        vh.save_figure(fig, output_path(prediction, "plot", unit, "pdf"), figure_dir=OUT_DIR)
        plt.close(fig)
