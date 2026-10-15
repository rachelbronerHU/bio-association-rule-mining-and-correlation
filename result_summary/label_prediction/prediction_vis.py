"""Plots of the label predictions. Reads the saved runs; never reruns anything."""
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

import prediction_search as ps

sys.path.insert(0, os.path.dirname(ps.HERE))
import data_helper as dh
import vis_helper as vh

MODEL_COLORS = {"Ridge logistic regression": "#33658A", "Lasso logistic regression": "#86BBD8",
                "Random forest": "#E0A33B", ps.AZULAY: "#8E5BB5"}

RULES_COLOR = "#2B5C9E"
RUN_COLOR = "#A9B6C6"
REFERENCES = {                     # reference row in the runs: (name in the legend, color, marker)
    ps.COMPOSITION: ("Cell composition", "#3E9C7A", "o"),
    "Pathological score only": ("Pathological score", "#E3A04F", "o"),
    "Clinical score only": ("Clinical score", "#A8571F", "o"),
    ps.PANETH: ("Paneth share (Azulay et al.)", "#E07A9B", "D"),
}
QUESTIONS = {                      # prediction name: how the figure says it
    "cortico_response": "Steroid response",
    "cause_of_death_nrm_muscle": "Death not from relapse (NRM)",
    "survival_at_follow_up": "Survival",
    "days_after_transplant_grouped_under30": "Early biopsy (<30 days)",
    "days_after_transplant_grouped_over100": "Late biopsy (>100 days)",
    "diagnosis": "Diagnosis, AML vs MDS (control)",
}
def reference_handles(size):
    """Legend entries for the references, each in its color and marker."""
    return [Line2D([], [], marker=marker, ls="", ms=size, color=color, mec="white", label=label)
            for label, color, marker in REFERENCES.values()]


SHORT = {                          # prediction name: a column header
    "cortico_response": "Steroid",
    "cause_of_death_nrm_muscle": "NRM",
    "survival_at_follow_up": "Survival",
    "days_after_transplant_grouped_under30": "Early",
    "days_after_transplant_grouped_over100": "Late",
    "diagnosis": "Diagnosis",
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
                     "best_positive_right": best["positive_right"],
                     "best_negative_right": best["negative_right"],
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
            for column, (_, color, marker) in REFERENCES.items():
                if column in rows:
                    ax.scatter(rows[column], y, s=34, color=color, marker=marker,
                               edgecolor="white", lw=0.8, zorder=3)
            ax.scatter(rows["best_rules"], y, s=62, color=RULES_COLOR, edgecolor="white",
                       lw=0.8, zorder=4)
            for yy, (_, row) in zip(y, rows.iterrows()):
                ax.text(row["best_rules"], yy - 0.3,
                        f"{row['best_rules']:.0f} ({row['best_positive_right']:.0f}/"
                        f"{row['best_negative_right']:.0f})",
                        ha="center", va="bottom", fontsize=6.5, color=vh.INK)

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
                  *reference_handles(5.5)]
        fig.subplots_adjust(bottom=1.0 / height, wspace=0.12)
        fig.legend(handles=legend, loc="lower center", ncol=3, frameon=False)
        vh.figure_titles(
            fig, "How well the rules predict each question", organ=ps.ORGAN,
            subtitle="Every rule run against cell composition and the severity scores",
            params="% right = average of both groups (in brackets); 50% = guessing; "
                   "each patient left out in turn",
        )
        vh.save_figure(fig, save, figure_dir=ps.OUT_DIR)
        plt.show()


# ---------------------------------------------------------------------------
# Which kind of rule carries the signal
# ---------------------------------------------------------------------------

def plot_rule_kinds(runs, units=("patient", "FOV"), save=None):
    """One panel per unit: a row per rule set and feature, a column per question. Each
    cell is the best of its runs (every filter and model); 50% or less stays white."""
    rules = runs[runs["min_patients"].notna()]
    rules = rules.assign(setting=rules["rule_set"] + "  ·  " + rules["feature"])
    questions = list(dict.fromkeys(runs["prediction"]))
    settings = list(dict.fromkeys(rules["setting"]))
    height = 2.0 + 0.24 * len(settings)

    with plt.rc_context(vh.PANEL_FONTS):
        fig, axes = plt.subplots(1, len(units), sharey=True, figsize=(vh.TEXT_WIDTH, height))
        for ax, unit in zip(np.atleast_1d(axes), units):
            grid = (rules[rules["unit"] == unit]
                    .pivot_table(index="setting", columns="prediction",
                                 values="balanced_accuracy", aggfunc="max")
                    .reindex(index=settings, columns=questions))
            ax.imshow(grid, cmap="Blues", vmin=50, vmax=95, aspect="auto")
            for (i, j), value in np.ndenumerate(grid.to_numpy()):
                ax.text(j, i, f"{value:.0f}", ha="center", va="center", fontsize=6.3,
                        color="white" if value > 78 else vh.INK)
            ax.set_xticks(range(len(questions)), [SHORT.get(q, q) for q in questions],
                          rotation=40, ha="right")
            ax.set_title(f"One row per {unit}", color=vh.INK)
            ax.tick_params(length=0)
            ax.spines[:].set_visible(False)
        axes[0].set_yticks(range(len(settings)), settings)
        vh.figure_titles(
            fig, "Which kind of rule predicts which question", organ=ps.ORGAN,
            subtitle="Best % right of each rule set and feature (over every filter and model)",
            params="A whole bright row is a steady signal; a single bright cell may be luck",
        )
        vh.save_figure(fig, save, figure_dir=ps.OUT_DIR)
        plt.show()


# ---------------------------------------------------------------------------
# Both groups right, run by run
# ---------------------------------------------------------------------------

def plot_sides(runs, unit="patient", save=None):
    """One panel per question: each run's % right of one group against the other. On the
    dashed line the score is 50% (guessing); runs on an edge call nearly everyone one group."""
    runs = runs[runs["unit"] == unit]
    questions = list(dict.fromkeys(runs["prediction"]))
    columns = 3
    rows_count = -(-len(questions) // columns)

    with plt.rc_context(vh.PANEL_FONTS):
        fig, axes = plt.subplots(rows_count, columns,
                                 figsize=(vh.TEXT_WIDTH, 1.0 + 2.05 * rows_count))
        for ax, prediction in zip(axes.flat, questions):
            group = runs[runs["prediction"] == prediction]
            rules = group[group["min_patients"].notna()]
            best = rules.loc[rules["balanced_accuracy"].idxmax()]
            ax.plot([0, 100], [100, 0], color=vh.ZERO, lw=0.8, ls="--", zorder=0)
            ax.scatter(rules["positive_right"], rules["negative_right"], s=6, color=RUN_COLOR,
                       alpha=0.6, lw=0, zorder=1)
            for column, (_, color, marker) in REFERENCES.items():
                reference = group[group["rule_set"] == column]
                ax.scatter(reference["positive_right"], reference["negative_right"], s=24,
                           color=color, marker=marker, edgecolor="white", lw=0.7, zorder=2)
            ax.scatter(best["positive_right"], best["negative_right"], s=48, color=RULES_COLOR,
                       edgecolor="white", lw=0.8, zorder=3)
            ax.set(xlim=(-3, 103), ylim=(-3, 103), xticks=[0, 50, 100], yticks=[0, 50, 100])
            ax.set_aspect("equal")
            ax.set_title(QUESTIONS.get(prediction, prediction), color=vh.INK, fontsize=7.6)
            ax.set_xlabel(f"{best['positive']} right (%)", fontsize=7)
            ax.set_ylabel(f"{best['negative']} right (%)", fontsize=7)
            right = {side: f"{round(best[f'{side}_right'] * best[f'n_{side}'] / 100)}/"
                           f"{best[f'n_{side}']}" for side in ("positive", "negative")}
            ax.annotate(right["positive"], xy=(1, 0.5), xycoords=ax.xaxis.label,
                        xytext=(3, 0), textcoords="offset points", ha="left", va="center",
                        fontsize=6.5, color=vh.ZERO)
            ax.annotate(right["negative"], xy=(0.5, 1), xycoords=ax.yaxis.label,
                        xytext=(0, 3), textcoords="offset points", ha="center", va="bottom",
                        rotation=90, fontsize=6.5, color=vh.ZERO)
            vh.tidy_axes(ax, grid="both")
        for ax in list(axes.flat)[len(questions):]:
            ax.set_visible(False)

        legend = [Line2D([], [], marker="o", ls="", ms=6.5, color=RULES_COLOR, mec="white",
                         label="Best rule run"),
                  Line2D([], [], marker="o", ls="", ms=2.5, color=RUN_COLOR, label="Each rule run"),
                  *reference_handles(5)]
        fig.subplots_adjust(hspace=0.45, wspace=0.5, bottom=0.15)
        fig.legend(handles=legend, loc="lower center", ncol=3, frameon=False)
        vh.figure_titles(
            fig, f"Are both groups predicted right? One row per {unit}", organ=ps.ORGAN,
            subtitle="Top right = both right; dashed line = guessing; "
                     "on an edge = one group for everyone",
        )
        vh.save_figure(fig, save, figure_dir=ps.OUT_DIR)
        plt.show()


def smaller_group(runs):
    """Each run's smaller group, its size and % right, keeping only the runs that get the
    larger group more than half right."""
    positive_smaller = runs["n_positive"] < runs["n_negative"]
    larger_right = runs["negative_right"].where(positive_smaller, runs["positive_right"])
    return runs.assign(
        smaller=runs["positive"].where(positive_smaller, runs["negative"]),
        smaller_n=runs[["n_positive", "n_negative"]].min(axis=1),
        smaller_right=runs["positive_right"].where(positive_smaller, runs["negative_right"]),
        larger_right=larger_right,
    )[larger_right > 50]


def plot_smaller_group(runs, units=("patient", "FOV"), save=None):
    """One panel per unit, one row per question: the best % right of the smaller group,
    over the runs that get the larger group more than half right."""
    runs = smaller_group(runs)
    questions = list(dict.fromkeys(runs["prediction"]))
    height = 1.9 + 0.45 * len(questions)
    spread = np.random.default_rng(0)

    with plt.rc_context(vh.PANEL_FONTS):
        fig, axes = plt.subplots(1, len(units), sharey=True, figsize=(vh.TEXT_WIDTH, height))
        for ax, unit in zip(axes, units):
            for y, prediction in enumerate(questions):
                group = runs[(runs["prediction"] == prediction) & (runs["unit"] == unit)]
                rules = group[group["min_patients"].notna()]
                ax.scatter(rules["smaller_right"], y + spread.uniform(-0.22, 0.22, len(rules)),
                           s=5, color=RUN_COLOR, alpha=0.55, lw=0, zorder=1)
                for column, (_, color, marker) in REFERENCES.items():
                    reference = group[group["rule_set"] == column]["smaller_right"]
                    ax.scatter(reference.max(), y, s=34, color=color, marker=marker,
                               edgecolor="white", lw=0.8, zorder=2)
                best = rules.sort_values(["smaller_right", "balanced_accuracy"]).iloc[-1]
                ax.scatter(best["smaller_right"], y, s=62, color=RULES_COLOR, edgecolor="white",
                           lw=0.8, zorder=3)
                ax.text(best["smaller_right"], y - 0.3,
                        f"{best['balanced_accuracy']:.0f} ({best['smaller_right']:.0f}/"
                        f"{best['larger_right']:.0f})",
                        ha="center", va="bottom", fontsize=6.5, color=vh.INK)
                ax.text(106, y, f"n={group['smaller_n'].iloc[0]}", va="center", fontsize=6.5,
                        color=vh.ZERO)
            ax.axvline(50, color=vh.ZERO, lw=0.9, ls="--", zorder=0.6)
            ax.set_xlim(-3, 103)
            ax.set_xticks(range(0, 101, 25))
            ax.set_ylim(len(questions) - 0.5, -0.8)
            ax.set_title(f"One row per {unit}", color=vh.INK)
            ax.set_xlabel("Smaller group right (%)")
            vh.tidy_axes(ax, grid="x", hide=("top", "right", "left"))
            ax.tick_params(axis="y", length=0)
        names = runs.drop_duplicates("prediction").set_index("prediction")["smaller"]
        axes[0].set_yticks(range(len(questions)),
                           [f"{QUESTIONS.get(q, q)}\n{names[q]}" for q in questions])

        legend = [Line2D([], [], marker="o", ls="", ms=7, color=RULES_COLOR, mec="white",
                         label="Best rule run"),
                  Line2D([], [], marker="o", ls="", ms=2.5, color=RUN_COLOR,
                         label="Each rule run"),
                  *reference_handles(5.5)]
        fig.subplots_adjust(bottom=1.0 / height, wspace=0.2)
        fig.legend(handles=legend, loc="lower center", ncol=3, frameon=False)
        vh.figure_titles(
            fig, "How well the smaller group is predicted", organ=ps.ORGAN,
            subtitle="Best % right of the smaller group, over runs that get the larger group "
                     "more than half right",
            params="% right = average of both groups (in brackets: smaller / larger group)",
        )
        vh.save_figure(fig, save, figure_dir=ps.OUT_DIR)
        plt.show()


# ---------------------------------------------------------------------------
# The rules behind the best run
# ---------------------------------------------------------------------------

def rule_name(rule):
    """A saved rule as (clean name, kind): ('A → B', 'attracts')."""
    sides, kind = rule.rsplit(" ", 1)
    antecedents, consequents = sides.split(" -> ")
    return f"{dh.clean_items(antecedents)} → {dh.clean_items(consequents)}", kind


def plot_weights(prediction, unit="patient", top=100, save=None):
    """The biggest weights of the best logistic run on rules: to the right pushes toward
    the "+" group, to the left toward the other."""
    weights = pd.read_csv(ps.output_path(prediction, "weights", unit))
    first = weights.iloc[0]
    if pd.isna(first["min_patients"]):
        print(f"{ps.name_of(prediction)}: the saved weights are of '{first['rule_set']}', "
              f"not rules. Rerun this prediction to save the best rule run's weights.")
        return
    sides = pd.read_csv(ps.output_path(prediction, "runs", unit)).iloc[0]
    shown = weights.head(top).iloc[::-1]
    names, kinds = zip(*shown["rule"].map(rule_name))
    colors = [vh.ATTRACTION if kind == "attracts" else vh.AVOIDANCE for kind in kinds]
    height = 1.8 + 0.22 * len(shown)

    with plt.rc_context(vh.PANEL_FONTS):
        fig, ax = plt.subplots(figsize=(vh.TEXT_WIDTH, height))
        ax.barh(range(len(shown)), shown["weight"], color=colors, height=0.7)
        ax.axvline(0, color=vh.ZERO, lw=0.8)
        ax.set_yticks(range(len(shown)), names)
        ax.set_xlabel(f"← toward {sides['negative']}          weight          "
                      f"toward {sides['positive']} →")
        vh.tidy_axes(ax, grid="x", hide=("top", "right", "left"))
        ax.tick_params(axis="y", length=0)
        ax.legend(handles=[Line2D([], [], color=vh.ATTRACTION, lw=6, label="Attraction"),
                           Line2D([], [], color=vh.AVOIDANCE, lw=6, label="Avoidance")],
                  loc="lower right", bbox_to_anchor=(1, 1), ncol=2, frameon=False)
        vh.figure_titles(
            fig, f"{QUESTIONS.get(ps.name_of(prediction), prediction['target'])}: "
                 f"the rules the best run leans on", organ=ps.ORGAN,
            subtitle=(f"{first['rule_set']} · {first['feature']} · in ≥{first['min_patients']:.0%} "
                      f"of patients · {first['model']} · {first['balanced_accuracy']:.0f}% right"),
            params=f"Top {len(shown)} of {len(weights)} rules by size of weight; one row per {unit}",
        )
        vh.save_figure(fig, save, figure_dir=ps.OUT_DIR)
        plt.show()


# ---------------------------------------------------------------------------
# Against Azulay et al.
# ---------------------------------------------------------------------------

PAPER_AUC = {"Paper: 10 imaging features": 91, "Paper: Paneth share": 76}


def against_paper(runs, prediction="cause_of_death_nrm_muscle", unit="patient"):
    """Our AUCs for the paper's question, next to theirs: each reference and the best rule
    run, with their model and with any model."""
    runs = runs[(runs["prediction"] == prediction) & (runs["unit"] == unit)]
    theirs = runs["model"] == ps.AZULAY
    rules = runs["min_patients"].notna()
    rows = [(f"{label}, their model", runs[theirs & (runs["rule_set"] == column)]["auc"].max(),
             color, marker)
            for column, (label, color, marker) in REFERENCES.items()]
    rows += [("Best rule run, their model", runs[theirs & rules]["auc"].max(), RULES_COLOR, "o"),
             ("Best rule run, any model", runs[rules]["auc"].max(), RULES_COLOR, "o")]
    return pd.DataFrame(rows, columns=["what", "auc", "color", "marker"])


def plot_paper(runs, prediction="cause_of_death_nrm_muscle", unit="patient", save=None):
    """Our AUCs for NRM against the paper's two: their 10-feature model and their Paneth
    share alone, both with their model and their patient filter."""
    ours = against_paper(runs, prediction, unit).dropna(subset=["auc"])
    first = runs[(runs["prediction"] == prediction) & (runs["unit"] == unit)].iloc[0]
    height = 1.9 + 0.3 * len(ours)

    with plt.rc_context(vh.PANEL_FONTS):
        fig, ax = plt.subplots(figsize=(vh.TEXT_WIDTH, height))
        y = np.arange(len(ours))
        ax.hlines(y, 40, ours["auc"], color=vh.GRID, lw=0.8, zorder=0)
        for yy, (_, row) in zip(y, ours.iterrows()):
            ax.scatter(row["auc"], yy, s=48, color=row["color"], marker=row["marker"],
                       edgecolor="white", lw=0.8, zorder=2)
            ax.text(row["auc"] + 1.2, yy, f"{row['auc']:.0f}", va="center", fontsize=6.8,
                    color=vh.INK)
        for (label, value), shift in zip(PAPER_AUC.items(), (-0.9, -0.55)):
            ax.axvline(value, color=vh.INK, lw=0.9, ls=":", zorder=1)
            ax.text(value, shift, f"{label}: {value}", ha="center", va="bottom", fontsize=6.6,
                    color=vh.INK)
        ax.axvline(50, color=vh.ZERO, lw=0.9, ls="--", zorder=0.6)
        ax.set_yticks(y, ours["what"])
        ax.set_ylim(len(ours) - 0.5, -1.2)
        ax.set_xlim(40, 100)
        ax.set_xlabel("AUC (%), 50 = guessing")
        vh.tidy_axes(ax, grid="x", hide=("top", "right", "left"))
        ax.tick_params(axis="y", length=0)
        vh.figure_titles(
            fig, "Death not from relapse (NRM): ours against Azulay et al.", organ=ps.ORGAN,
            subtitle=(f"{first['n_positive']} NRM, {first['n_negative']} other patients "
                      "(their filter: muscle and more than 50 epithelial cells)"),
            params="Each patient left out in turn; best of many runs looks better than it is",
        )
        vh.save_figure(fig, save, figure_dir=ps.OUT_DIR)
        plt.show()


# ---------------------------------------------------------------------------
# Permutation test
# ---------------------------------------------------------------------------

def plot_permutation(prediction, unit="patient", save=None):
    """The best score of each search on shuffled labels, against the best on the real ones."""
    results = pd.read_csv(ps.output_path(prediction, "permutation", unit))
    shuffled = results[~results["real"]]
    metrics = {"balanced_accuracy": "Best % right", "auc": "Best AUC (%)"}

    with plt.rc_context(vh.PANEL_FONTS):
        fig, axes = plt.subplots(1, len(metrics), sharey=True, figsize=(vh.TEXT_WIDTH, 2.6))
        for ax, (column, label) in zip(axes, metrics.items()):
            real = results.loc[results["real"], column].iloc[0]
            ax.hist(shuffled[column], bins=np.arange(0, 101, 2.5), color=RUN_COLOR,
                    edgecolor="white", label="Shuffled labels")
            ax.axvline(real, color=RULES_COLOR, lw=1.6, label="Real labels")
            ax.text(real, ax.get_ylim()[1], f" {real:.0f}\n p = {ps.p_value(results, column):.2f}",
                    va="top", fontsize=6.8, color=vh.INK)
            ax.set_xlim(30, 100)
            ax.set_xlabel(label)
            vh.tidy_axes(ax, grid="y")
        axes[0].set_ylabel("Searches")
        axes[0].legend(loc="upper center", bbox_to_anchor=(1.1, -0.3), ncol=2, frameon=False)
        vh.figure_titles(
            fig, f"{QUESTIONS.get(ps.name_of(prediction), ps.name_of(prediction))}: "
                 "real or luck?", organ=ps.ORGAN,
            subtitle=(f"Best of every rule run, on the real labels and on {len(shuffled)} "
                      f"shufflings of them; one row per {unit}"),
            params="p = share of searches at least as good as the real one",
        )
        vh.save_figure(fig, save or ps.output_path(prediction, "permutation", unit, "pdf"),
                       figure_dir=ps.OUT_DIR)
        plt.show()
