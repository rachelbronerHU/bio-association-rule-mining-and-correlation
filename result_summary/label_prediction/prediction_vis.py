"""Plots of the label predictions. Reads the saved runs; never reruns anything."""
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

import prediction_search as ps

sys.path.insert(0, os.path.dirname(ps.HERE))
import vis_helper as vh

MODEL_COLORS = {"Ridge logistic regression": "#33658A", "Lasso logistic regression": "#86BBD8",
                "Random forest": "#E0A33B", ps.AZULAY: "#8E5BB5"}

RULES_COLOR = "#2B5C9E"
RUN_COLOR = "#A9B6C6"
REFERENCES = {                     # reference row in the runs: (name in the legend, color)
    ps.COMPOSITION: ("Cell composition", "#3E9C7A"),
    "Pathological score only": ("Pathological score", "#E3A04F"),
    "Clinical score only": ("Clinical score", "#A8571F"),
}
QUESTIONS = {                      # prediction name: how the figure says it
    "cortico_response": "Steroid response",
    "cause_of_death_nrm_muscle": "Death not from relapse (NRM)",
    "survival_at_follow_up": "Survival",
    "days_after_transplant_grouped_under30": "Early biopsy (<30 days)",
    "days_after_transplant_grouped_over100": "Late biopsy (>100 days)",
    "diagnosis": "Diagnosis, AML vs MDS (control)",
}


# ---------------------------------------------------------------------------
# Every run of one prediction and unit
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
        bar = 0.8 / len(MODEL_COLORS)
        for i, model in enumerate(MODEL_COLORS):
            values = runs[runs["model"] == model].set_index("setting")["balanced_accuracy"]
            y = np.arange(len(settings)) + (i - (len(MODEL_COLORS) - 1) / 2) * bar
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
            fig, f"Predicting {first['target']}, one row per {unit}", organ=ps.ORGAN,
            subtitle=(f"{first['n_positive']} {first['positive']} {unit}s, {first['n_negative']} "
                      f"{first['negative']} {unit}s; each patient left out in turn"),
        )
        vh.save_figure(fig, ps.output_path(prediction, "plot", unit, "pdf"), figure_dir=ps.OUT_DIR)
        plt.close(fig)


# ---------------------------------------------------------------------------
# The best of each prediction, against the references
# ---------------------------------------------------------------------------

def best_per_prediction(runs):
    """One row per prediction and unit: the best rule run and its setting, the median of
    all rule runs, and the best score of each reference (any model)."""
    rows = []
    for (prediction, unit), group in runs.groupby(["prediction", "unit"], sort=False):
        rules = group[group["min_patients"].notna()]
        best = rules.loc[rules["balanced_accuracy"].idxmax()]
        references = group[group["min_patients"].isna()].groupby("rule_set")["balanced_accuracy"].max()
        rows.append({"prediction": prediction, "unit": unit,
                     "best_rules": best["balanced_accuracy"],
                     "rules_median": rules["balanced_accuracy"].median(),
                     **references.to_dict(),
                     "best_reference": references.max(),
                     "best_setting": setting_label(best), "best_model": best["model"],
                     "best_auc": best["auc"]})
    return pd.DataFrame(rows)


def plot_best(runs, units=("patient", "FOV"), save=None):
    """One panel per unit, one row per prediction: a small dot per rule run, their median,
    the best rule run, and cell composition and the severity scores. Rows sorted by how
    far the best rule run beats the best reference, per patient."""
    best = best_per_prediction(runs)
    first = best[best["unit"] == units[0]].set_index("prediction")
    order = (first["best_rules"] - first["best_reference"]).sort_values(ascending=False).index
    height = 1.9 + 0.45 * len(order)
    spread = np.random.default_rng(0)

    with plt.rc_context(vh.PANEL_FONTS):
        fig, axes = plt.subplots(1, len(units), sharey=True, figsize=(vh.TEXT_WIDTH, height))
        for ax, unit in zip(np.atleast_1d(axes), units):
            rows = best[best["unit"] == unit].set_index("prediction").loc[order]
            y = np.arange(len(rows))
            for yy, prediction in zip(y, order):
                scores = runs[(runs["prediction"] == prediction) & (runs["unit"] == unit)
                              & runs["min_patients"].notna()]["balanced_accuracy"]
                ax.scatter(scores, yy + spread.uniform(-0.22, 0.22, len(scores)), s=5,
                           color=RUN_COLOR, alpha=0.55, lw=0, zorder=1)
            ax.scatter(rows["rules_median"], y, s=28, facecolor="white", edgecolor=RULES_COLOR,
                       lw=1.2, zorder=2)
            for column, (_, color) in REFERENCES.items():
                ax.scatter(rows[column], y, s=34, color=color, edgecolor="white", lw=0.8,
                           zorder=3)
            ax.scatter(rows["best_rules"], y, s=62, color=RULES_COLOR, edgecolor="white",
                       lw=0.8, zorder=4)
            for yy, value in zip(y, rows["best_rules"]):
                ax.text(value, yy - 0.3, f"{value:.0f}", ha="center", va="bottom",
                        fontsize=6.5, color=vh.INK)

            ax.axvline(50, color=vh.ZERO, lw=0.9, ls="--", zorder=0.6)
            ax.set_xlim(0, 100)
            ax.set_xticks(range(0, 101, 25))
            ax.set_ylim(len(rows) - 0.5, -0.8)
            ax.set_title(f"One row per {unit}", color=vh.INK)
            ax.set_xlabel("Predicted right (%)")
            vh.tidy_axes(ax, grid="x", hide=("top", "right", "left"))
            ax.tick_params(axis="y", length=0)
        axes[0].set_yticks(y, [QUESTIONS.get(name, name) for name in order])

        legend = [Line2D([], [], marker="o", ls="", ms=7, color=RULES_COLOR, mec="white",
                         label="Best rule run"),
                  Line2D([], [], marker="o", ls="", ms=5, mfc="white", mec=RULES_COLOR, mew=1.2,
                         label="Typical rule run (median)"),
                  Line2D([], [], marker="o", ls="", ms=2.5, color=RUN_COLOR,
                         label="Each rule run"),
                  *[Line2D([], [], marker="o", ls="", ms=5.5, color=color, mec="white",
                           label=label) for label, color in REFERENCES.values()]]
        fig.subplots_adjust(bottom=1.0 / height, wspace=0.12)
        fig.legend(handles=legend, loc="lower center", ncol=3, frameon=False)
        vh.figure_titles(
            fig, "How well the rules predict each question", organ=ps.ORGAN,
            subtitle="Every rule run against cell composition and the severity scores",
            params="% right = average of both groups; 50% = guessing; "
                   "each patient left out in turn",
        )
        vh.save_figure(fig, save, figure_dir=ps.OUT_DIR)
        plt.show()
