"""Can the rules predict a biopsy label, such as steroid response (TARGET below)?

Every setting in the grid below is one run, done twice: one row per patient, and one row
per FOV. Either way each patient is left out in turn and predicted by a model trained on
all the other patients. Each target and unit gets its own runs, weights and plot files in
output/.

    python response_search.py          # run the grid, then plot
    python response_search.py --plot   # plot the saved runs again without rerunning
"""
import os
import sys
import time
import itertools

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegressionCV
from sklearn.metrics import balanced_accuracy_score, roc_auc_score
from sklearn.model_selection import LeaveOneGroupOut, StratifiedKFold, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
import data_helper as dh
import vis_helper as vh

ORGAN = "Duodenum"
OUT_DIR = os.path.join(HERE, "output")
TARGET = "Cortico Response"   # the biopsy column to predict
VALUE = None                  # None: the column's two values. A value: it vs everything else.
POSITIVE = "Responder"        # with VALUE None, the "+" side (weights above 0 push toward it)
ONLY_PATIENTS_WITH_MUSCLE_AND_EPITHELIUM = False   # Azulay et al.'s filter for their NRM model
MIN_EPITHELIAL_CELLS = 50     # with the filter on, a patient needs more than this many
UNITS = {"patient": "Biopsy", "FOV": "FOV"}

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


MODELS = {
    "Ridge logistic regression": logistic(0),
    "Lasso logistic regression": logistic(1),
    "Random forest": lambda: RandomForestClassifier(
        n_estimators=500, class_weight="balanced", random_state=0, n_jobs=1),
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


def severity_table(fovs, unit, column):
    """Unit x one column: 1 when the biopsy's score is Severe, 0 when Mild."""
    return fovs.groupby(unit)[column].first().eq("Severe").astype(int).to_frame(column)


def all_tables(rules, cells, fovs, unit):
    """(rule set, feature, min patients, table) for every setting in the grid. The first
    rows are references with no rules: cell composition and each severity score alone."""
    yield COMPOSITION, "", np.nan, composition_table(cells, fovs, unit)
    for column in SEVERITY:
        yield f"{column} only", "", np.nan, severity_table(fovs, unit, column)
    for (name, (kinds, pairwise)), feature, share in itertools.product(
            RULE_SETS.items(), FEATURES, MIN_PATIENTS):
        chosen = rules["Kind"].isin(kinds)
        if pairwise:
            chosen &= rules["Rule_Type"] == "pairwise"
        yield name, feature, share, rule_table(rules[chosen], fovs, unit, feature, share)


# ---------------------------------------------------------------------------
# Runs
# ---------------------------------------------------------------------------

def score(table, is_positive, patient_of, model):
    """Predict each patient's rows from all the other patients, and count how many were
    right. The left-out patients run side by side, one per CPU core."""
    y = is_positive.loc[table.index].to_numpy()
    chance = cross_val_predict(MODELS[model](), table.to_numpy(), y, cv=LeaveOneGroupOut(),
                               groups=patient_of.loc[table.index].to_numpy(),
                               method="predict_proba", n_jobs=-1)
    predicted = chance.argmax(axis=1)
    return {
        "balanced_accuracy": 100 * balanced_accuracy_score(y, predicted),
        "auc": 100 * roc_auc_score(y, chance[:, 1]),
        "positive_right": 100 * (predicted[y == 1] == 1).mean(),
        "negative_right": 100 * (predicted[y == 0] == 0).mean(),
        "n_positive": int(y.sum()),
        "n_negative": int((y == 0).sum()),
    }


def sides(fovs):
    """The names of the "+" and "-" sides, and each FOV's side: 1, 0, or empty when its
    patient has no answer. With VALUE set, an empty answer counts as "not VALUE"."""
    column = fovs[TARGET]
    if VALUE is not None:
        return VALUE, f"not {VALUE}", column.eq(VALUE).astype(int)
    values = sorted(column.dropna().unique())
    if len(values) != 2 or POSITIVE not in values:
        raise ValueError(f"{TARGET} has the values {values}: set VALUE to one of them, "
                         f"or POSITIVE to one of two")
    negative = next(v for v in values if v != POSITIVE)
    return POSITIVE, negative, column.eq(POSITIVE).astype(int).where(column.notna())


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


def load():
    """Rules, cells and FOVs of this organ's transplanted patients that have an answer."""
    cells, fovs, _ = dh.load_spatial_data()
    fovs = fovs[(fovs["Organ"] == ORGAN) & fovs["Biopsy_ID"].notna()]
    positive, negative, side = sides(fovs)
    fovs = fovs.assign(side=side, positive=positive, negative=negative).dropna(subset=["side"])
    if ONLY_PATIENTS_WITH_MUSCLE_AND_EPITHELIUM:
        kept = with_muscle_and_epithelium(cells, fovs)
        print(f"Kept {len(kept)} of {fovs['Biopsy'].nunique()} patients with muscle and more "
              f"than {MIN_EPITHELIAL_CELLS} epithelial cells.")
        fovs = fovs[fovs["Biopsy"].isin(kept)]
    rules = dh.load_results(rule_max_items=4, kind=None)
    rules = rules[rules["FOV"].isin(fovs["FOV"])]
    rules["Rule"] = rules["Antecedents"] + " -> " + rules["Consequents"] + " " + rules["Kind"]
    finite = rules["Conviction"].replace(np.inf, np.nan)
    rules["Conviction"] = finite.fillna(finite.max())
    return rules, cells, fovs


def output_path(kind, unit, ext="csv"):
    """output/<kind>_<target>_<unit>.<ext>, e.g. runs_cortico_response_patient.csv.
    With the muscle filter on, the target ends in _muscle."""
    muscle = "muscle" if ONLY_PATIENTS_WITH_MUSCLE_AND_EPITHELIUM else None
    target = "_".join(str(part) for part in (TARGET, VALUE, muscle) if part is not None)
    return os.path.join(OUT_DIR, f"{kind}_{target.lower().replace(' ', '_')}_{unit}.{ext}")


def weights_of(table, is_positive, model):
    """The model's weight for each rule, fitted once on all rows."""
    fitted = MODELS[model]().fit(table.to_numpy(), is_positive.loc[table.index].to_numpy())
    return pd.Series(fitted[-1].coef_[0], index=table.columns, name="weight")


def run_unit(rules, cells, fovs, unit):
    by_unit = fovs.groupby(UNITS[unit])
    is_positive = by_unit["side"].first().astype(int)
    patient_of = by_unit["Biopsy"].first()
    positive, negative = fovs["positive"].iloc[0], fovs["negative"].iloc[0]
    rows, weights = [], {}
    total = (1 + len(SEVERITY) + len(RULE_SETS) * len(FEATURES) * len(MIN_PATIENTS)) * len(MODELS)
    start = time.time()
    for name, feature, share, table in all_tables(rules, cells, fovs, UNITS[unit]):
        for model in MODELS:
            row = {"organ": ORGAN, "target": TARGET, "positive": positive, "negative": negative,
                   "unit": unit, "rule_set": name, "feature": feature,
                   "min_patients": share, "model": model, "n_features": table.shape[1],
                   **score(table, is_positive, patient_of, model)}
            if "logistic" in model:
                weights[len(rows)] = weights_of(table, is_positive, model)
            rows.append(row)
            minutes, seconds = divmod(int(time.time() - start), 60)
            print(f"[{len(rows):>3}/{total}  {minutes:>3}:{seconds:02}]  {unit:8} {name:34} "
                  f"{feature:16} {share:>4} {model:26} {row['n_features']:>6} features  "
                  f"{row['balanced_accuracy']:5.1f}%  ({positive} {row['positive_right']:3.0f}%, "
                  f"{negative} {row['negative_right']:3.0f}%)  AUC {row['auc']:3.0f}")
    runs = pd.DataFrame(rows).round(1)
    os.makedirs(OUT_DIR, exist_ok=True)
    runs.to_csv(output_path("runs", unit), index=False)
    print(f"saved {output_path('runs', unit)}")
    save_best_weights(runs, weights, unit)
    return runs


def save_best_weights(runs, weights, unit):
    """The weights of the best-scoring logistic run, biggest pull first, zeros left out."""
    best = runs.loc[list(weights)]["balanced_accuracy"].idxmax()
    chosen = weights[best]
    chosen = chosen[chosen != 0].sort_values(key=abs, ascending=False)
    path = output_path("weights", unit)
    setting = runs.loc[best, ["rule_set", "feature", "min_patients", "model", "balanced_accuracy", "auc"]]
    chosen.rename_axis("rule").reset_index().assign(**setting).to_csv(path, index=False)
    print(f"saved {path}  ({', '.join(map(str, setting))})")


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------

def setting_label(row):
    if pd.isna(row["min_patients"]):
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
        ax.set_xlabel(f"{unit[0].upper() + unit[1:]}s predicted right "
                      f"(%, average of {first['positive']} and {first['negative']})")
        vh.tidy_axes(ax, grid="x")
        ax.legend(loc="lower right", frameon=False)
        vh.figure_titles(
            fig, f"Predicting {first['target']}, one row per {unit}", organ=ORGAN,
            subtitle=(f"{first['n_positive']} {first['positive']} {unit}s, {first['n_negative']} "
                      f"{first['negative']} {unit}s; each patient left out in turn"),
        )
        vh.save_figure(fig, output_path("plot", unit, "pdf"), figure_dir=OUT_DIR)
        plt.close(fig)


if __name__ == "__main__":
    data = None if "--plot" in sys.argv else load()
    for unit in UNITS:
        plot_runs(pd.read_csv(output_path("runs", unit)) if data is None
                  else run_unit(*data, unit))
