"""Drawing for the stage analysis: where a longer rule lives as the disease worsens.

Only plotting lives here; the counting stays in the notebook.
Colours come from `complex_vis`, so a verdict means the same thing in every figure.
"""
from collections import defaultdict

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.patches import Patch, Rectangle
from matplotlib.transforms import blended_transform_factory, offset_copy

from vis_helper import (
    figure_titles, tidy_axes, spread_labels,
    plot_fov, set_cell_colors, INK,
)
from complex_vis import (
    _finish, clear_exports, CLASS_COLORS, text_on,
    _plain_rule, _rule_parts, _rule_metric_text,
)

# What one FOV can say about one rule. Four fine-grained states, and the two coarse
# ones each pair collapses into. 'nothing' is neither the rule nor its parts, and is
# drawn as empty space rather than a colour.
STATE_COLORS = {
    "informative": CLASS_COLORS["stronger_effect"],   # adds + new
    "adds":        CLASS_COLORS["stronger_effect"],
    "new":         CLASS_COLORS["new"],
    "covered":     "#d8d4c8",                         # redundant + pairs_only
    "redundant":   CLASS_COLORS["consequent_driven"],
    "pairs_only":  CLASS_COLORS["redundant_by_simpler"],
    "nothing":     "#f4f3ef",
    "ineligible":  "#9a9994",
}
STATE_LABELS = {
    "informative": "the rule is here and adds something",
    "adds":        "here, and beat a real shorter rule",
    "new":         "here, and nothing shorter was mined",
    "covered":     "the pairs cover it, the rule adds nothing",
    "redundant":   "here, but a shorter rule already said it",
    "pairs_only":  "not here at all, but its pairs are",
    "nothing":     "eligible, but neither the rule nor its pairs are here",
    "ineligible":  "too few cells to measure this rule",
    # Not a state but a share of one: of the FOVs where the rule or ANY ONE of its
    # pairwise parts turned up, how often the rule was the better description.
    "earns": "how often the rule beats its parts, where either turns up",
}

_RAMP = LinearSegmentedColormap.from_list(
    "sand_teal", ["#f8f6f1", "#dfe7e0", "#a8c4b8", "#4c8496", "#22596b"])
# For 'earns', 0.5 is a real middle - below it the parts usually suffice.
_DIVERGING = LinearSegmentedColormap.from_list(
    "sand_split_teal", ["#b9ab8c", "#ddd5c3", "#f6f4ef", "#a8c4b8", "#2e7387"])

# Thin evidence is outlined, never lightened: the diverging ramp is already palest at
# 50%, so a lighter cell has to keep meaning "near a coin flip" and nothing else.
_THIN_EDGE = (0, (3, 2))

# All in points, so they stay the same size whatever the figure ends up being:
# the gap between the name column and its panel, the size of the names and of the
# axis numbers, and the width one panel gets.
_NAME_GAP = 26
_NAME_SIZE = 8
_TICK_SIZE = 9
_PANEL_WIDTH = 383
_EDGE = 10


def plot_stage_rule_fovs(examples, stages, df_cells, df_fovs, organ=None,
                         subtitle=None, params=None, save=None, cell_size=14):
    """Full and highlighted FOVs for a complex rule and its best simpler rule."""
    examples = pd.DataFrame(examples)
    if examples.empty:
        print("No stage FOV examples to plot.")
        return []
    cell_colors = set_cell_colors(df_cells)

    figures = []
    for rule, rows in examples.groupby("rule", sort=False):
        rows = rows.set_index("stage")
        has_parent = "simpler_rule" in rows and rows["simpler_rule"].notna().any()
        nrows = 3 if has_parent else 2
        fig, axes = plt.subplots(
            nrows, len(stages), figsize=(3.75 * len(stages), 3.8 * nrows),
            squeeze=False, facecolor="white",
            gridspec_kw={"wspace": 0.18, "hspace": 0.46},
        )
        legend_types = []
        for column, stage in enumerate(stages):
            if stage not in rows.index:
                for row_number in range(nrows):
                    axes[row_number, column].set_visible(False)
                axes[0, column].set_visible(True)
                axes[0, column].axis("off")
                axes[0, column].set_title(stage, fontsize=10)
                axes[0, column].text(
                    0.5, 0.5, "No matching FOV", ha="center", va="center",
                    color="#898781", fontsize=9,
                )
                continue

            item = rows.loc[stage]
            if isinstance(item, pd.DataFrame):
                item = item.iloc[0]
            fov = item["FOV"]
            ant = list(item["antecedent_cells"])
            con = list(item["consequent_cells"])
            legend_types.extend(ant + con)

            plot_fov(
                fov, "", df_cells, df_fovs, ax=axes[0, column],
                show_legend=False, cell_size=cell_size,
            )
            axes[0, column].set_title(f"{stage} · {fov}\nFull FOV", fontsize=9.5)

            plot_fov(
                fov, "", df_cells, df_fovs, target_ant_cells=ant,
                target_cons_cells=con, ax=axes[1, column],
                show_legend=False, cell_size=cell_size,
            )
            metrics = _rule_metric_text(item.get("complex_metrics", {}))
            axes[1, column].set_title(
                "Higher-order rule" + (f"\n{metrics}" if metrics else ""),
                fontsize=9.2,
            )

            if has_parent:
                parent_ant, parent_con = _rule_parts(item["simpler_rule"])
                legend_types.extend(parent_ant + parent_con)
                plot_fov(
                    fov, "", df_cells, df_fovs,
                    target_ant_cells=parent_ant, target_cons_cells=parent_con,
                    ax=axes[2, column], show_legend=False, cell_size=cell_size,
                )
                parent_metrics = _rule_metric_text(
                    item.get("simpler_metrics", {}),
                    include_fdr=True, wrap=True,
                )
                axes[2, column].set_title(
                    f"Best simpler\n{_plain_rule(item['simpler_rule'])}"
                    + (f"\n{parent_metrics}" if parent_metrics else ""),
                    fontsize=8.8,
                )

        legend_types = list(dict.fromkeys(legend_types))
        handles = [
            plt.Line2D(
                [0], [0], marker="o", linestyle="none",
                markerfacecolor=cell_colors.get(cell, "black"),
                markeredgecolor="none", markersize=6.5, label=cell,
            )
            for cell in legend_types
        ]
        if handles:
            fig.legend(
                handles=handles, title="Cell type", frameon=False,
                ncol=len(handles), loc="lower center",
                bbox_to_anchor=(0.5, 0.005), fontsize=8, title_fontsize=9,
            )
        figure_titles(
            fig, str(rule).replace(" -> ", " → "), organ=organ,
            subtitle=subtitle, params=params,
        )
        fig.subplots_adjust(
            bottom=0.12 if nrows == 2 else 0.07,
            wspace=0.18, hspace=0.46,
        )
        _finish(fig, save)
        figures.append(fig)
    return figures


def states(split_adds=False, split_covered=False):
    """The columns a figure draws, left to right, at the detail asked for.

    'redundant' and 'pairs_only' merge by default: both say the pairwise level already
    covers this FOV, so two greys for one answer is one grey too many.
    """
    adds = ["adds", "new"] if split_adds else ["informative"]
    covered = ["redundant", "pairs_only"] if split_covered else ["covered"]
    return adds + covered


def _keys(columns):
    return [Patch(facecolor=STATE_COLORS[c], edgecolor="white", label=STATE_LABELS[c])
            for c in columns]


def _wide(counts, stages, column):
    return counts.pivot(index="name", columns="stage", values=column)[stages]


def _stage_ticks(counts, stages):
    """Each stage's name over its total FOV count."""
    column = "n_total" if "n_total" in counts else "n_fovs"
    sizes = counts.drop_duplicates("stage").set_index("stage")[column]
    eligible = "eligibility_on" in counts and bool(counts["eligibility_on"].any())
    label = "total n" if eligible else "n"
    return [f"{s}\n{label}={int(sizes[s])}" for s in stages]


def plot_stage_bars(counts, stages, organ=None, subtitle=None, params=None,
                    split_adds=False, split_covered=False, save=None):
    """One bar per rule per stage: how much of the stage the rule accounts for.

    Every bar runs over EVERY FOV of its stage, so the empty space on the right is real -
    it is the FOVs where neither the rule nor any of its pairs turned up. Bars share one
    axis across the stages, so a rule that fades reads as a bar getting shorter.
    """
    if counts.empty:
        print("No rules to plot.")
        return

    columns = states(split_adds, split_covered)
    wide = {c: _wide(counts, stages, f"{c}_share") for c in columns}
    names = list(wide[columns[0]].index)
    reach = float(sum(np.nan_to_num(w.to_numpy()) for w in wide.values()).max()) * 1.06
    reach = max(reach, 0.05)
    y = np.arange(len(names))

    fig, axes = plt.subplots(1, len(stages), figsize=(12.5, len(names) * 0.34 + 2.2),
                             sharey=True, gridspec_kw={"wspace": 0.05})
    axes = np.atleast_1d(axes)

    for ax, stage, tick in zip(axes, stages, _stage_ticks(counts, stages)):
        left = np.zeros(len(names))
        for column in columns:
            width = np.nan_to_num(wide[column][stage].to_numpy())
            ax.barh(y, width, left=left, height=0.64, color=STATE_COLORS[column],
                    edgecolor="white", linewidth=0.8, zorder=3)
            left = left + width

        ax.set_xlim(0, reach)
        # No tick on the right edge, or neighbouring panels print their last and first
        # label on top of each other at the seam.
        ax.set_xticks([t for t in np.arange(0, reach, 0.2) if t < reach - 0.06])
        ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0%}"))
        ax.set_title(tick, fontsize=10.5, color=INK, pad=8)
        tidy_axes(ax, grid="x", hide=("top", "right", "left"))
        ax.tick_params(labelsize=9)

    axes[0].set_yticks(y, names, fontsize=8.5)
    axes[0].set_ylim(len(names) - 0.4, -0.6)
    axes[0].tick_params(axis="y", length=0)
    eligible = "eligibility_on" in counts and bool(counts["eligibility_on"].any())
    denominator = "eligible FOVs" if eligible else "the stage's FOVs"
    axes[len(stages) // 2].set_xlabel(
        f"share of {denominator}      (empty = neither the rule nor its pairs)", fontsize=10)

    axes[-1].legend(handles=_keys(columns), fontsize=8.5, frameon=False,
                    loc="upper left", bbox_to_anchor=(1.03, 1.0),
                    handlelength=1.1, borderpad=0, labelspacing=0.7)
    figure_titles(fig, "Where each rule lives, stage by stage", organ=organ,
                  subtitle=subtitle, params=params)
    _finish(fig, save)


def plot_availability_breakdown(counts, stages, names, organ=None,
                                subtitle=None, params=None, save=None):
    """Separate cell availability from spatial organization for one organ."""
    columns = ["ineligible", "nothing", "covered", "informative"]
    fig, axes = plt.subplots(
        1, len(stages), figsize=(3.5 * len(stages) + 4.2, 0.42 * len(names) + 3.5),
        sharex=True, sharey=True, squeeze=False, gridspec_kw={"wspace": 0.06},
    )
    axes = axes[0]
    y = np.arange(len(names))
    indexed = counts.set_index(["name", "stage"])
    for column, stage in enumerate(stages):
        ax = axes[column]
        left = np.zeros(len(names))
        for state in columns:
            values = []
            for name in names:
                key = (name, stage)
                if key not in indexed.index:
                    values.append(0.0)
                    continue
                item = indexed.loc[key]
                values.append(float(item[state]) / max(float(item["n_total"]), 1.0))
            values = np.asarray(values)
            ax.barh(y, values, left=left, height=0.66, color=STATE_COLORS[state],
                    edgecolor="white", linewidth=0.7)
            for row, (start, width) in enumerate(zip(left, values)):
                if width >= 0.07:
                    count = int(indexed.loc[(names[row], stage)][state])
                    ax.text(start + width / 2, y[row], f"{count} ({width:.0%})",
                            ha="center", va="center", fontsize=7.8,
                            color=text_on(STATE_COLORS[state]))
            left += values

        total = (counts.drop_duplicates("stage").set_index("stage").at[stage, "n_total"])
        ax.set_title(f"{stage}\nn={int(total)}", fontsize=9.5)
        ax.set_xlim(0, 1)
        labels = ["0%", "50%", "100%"]
        if column > 0:
            labels[0] = ""
        ax.set_xticks([0, 0.5, 1], labels)
        tidy_axes(ax, grid="x", hide=("top", "right", "left"))
        ax.tick_params(axis="both", length=0, labelsize=8.5)

    one_rule = len(names) == 1
    axes[0].set_yticks(y, [""] if one_rule else names, fontsize=8.2)
    axes[0].set_ylim(len(names) - 0.4, -0.6)
    axes[len(stages) // 2].set_xlabel("share of all FOVs", fontsize=10)
    handles = [Patch(facecolor=STATE_COLORS[state], edgecolor="white",
                     label=STATE_LABELS[state]) for state in columns]
    fig.legend(handles=handles, frameon=False, fontsize=8.5, ncol=2,
               loc="lower center", bbox_to_anchor=(0.58, 0.005))
    title = names[0].replace(" -> ", " → ") if one_rule else (
        "Cell availability versus higher-order organization"
    )
    subtitle = subtitle or (
        "Cell availability versus higher-order organization" if one_rule else None
    )
    figure_titles(fig, title, organ=organ, subtitle=subtitle, params=params)
    fig.subplots_adjust(bottom=0.28)
    _finish(fig, save)


def plot_strength_movement(values, stages, metric="Lift", organ=None,
                           params=None, save=None):
    """Show patient-level rule strength, with parent comparison when available."""
    if values.empty:
        print("No rule strengths to plot.")
        return []

    compare = {"simpler", "gain"}.issubset(values.columns)
    colors = {"complex": CLASS_COLORS["stronger_effect"], "simpler": "#9a8d70"}
    figures = []
    for rule, rows in values.groupby("rule", sort=False):
        if compare:
            fig, axes = plt.subplots(1, 2, figsize=(10.8, 4.45),
                                     gridspec_kw={"wspace": 0.28})
            raw_ax, gain_ax = axes
        else:
            fig, raw_ax = plt.subplots(figsize=(6.4, 4.45))
            axes = [raw_ax]
        x = np.arange(len(stages), dtype=float)
        counts = []

        series = [
            ("complex", -0.09, "o", "higher-order rule"),
            ("simpler", 0.09, "s", "best simpler rule"),
        ] if compare else [("complex", 0, "o", "new higher-order rule")]
        for key, offset, marker, label in series:
            medians = []
            for position, stage in enumerate(stages):
                observed = rows.loc[rows["stage"] == stage, key].to_numpy(float)
                observed = observed[np.isfinite(observed)]
                if key == "complex":
                    counts.append(len(observed))
                if len(observed):
                    jitter = (np.linspace(-0.045, 0.045, len(observed))
                              if len(observed) > 1 else np.zeros(1))
                    raw_ax.scatter(position + offset + jitter, observed, s=31,
                                   marker=marker, color=colors[key], alpha=0.68,
                                   edgecolor="white", linewidth=0.55, zorder=3)
                medians.append(np.median(observed) if len(observed) else np.nan)
            raw_ax.plot(x + offset, medians, color=colors[key], lw=1.5, alpha=0.85)
            raw_ax.scatter(x + offset, medians, s=68, marker="D", color=colors[key],
                           edgecolor="white", linewidth=0.8, zorder=4, label=label)

        if compare:
            gain_medians = []
            for position, stage in enumerate(stages):
                observed = rows.loc[rows["stage"] == stage, "gain"].to_numpy(float)
                observed = observed[np.isfinite(observed)]
                if len(observed):
                    jitter = (np.linspace(-0.06, 0.06, len(observed))
                              if len(observed) > 1 else np.zeros(1))
                    gain_ax.scatter(position + jitter, observed, s=34,
                                    color=CLASS_COLORS["stronger_effect"], alpha=0.68,
                                    edgecolor="white", linewidth=0.55, zorder=3)
                gain_medians.append(np.median(observed) if len(observed) else np.nan)
            gain_ax.plot(x, gain_medians, color="#777570", lw=1.5)
            gain_ax.scatter(x, gain_medians, marker="D", s=68,
                            color=["#7fcdbb", "#f2b35d", "#e46c68"][:len(stages)],
                            edgecolor="white", linewidth=0.8, zorder=4)

        reference = 0 if metric == "Leverage" else 1
        raw_ax.axhline(reference, color="#bab8b1", lw=1.0)
        labels = [f"{stage}\npatient n={count}" for stage, count in zip(stages, counts)]
        raw_ax.set_xticks(x, labels)
        raw_ax.set_ylabel(metric)
        raw_ax.set_title(
            "Rule and strongest simpler rule" if compare else "New rule strength",
            fontsize=10.5,
        )
        raw_ax.legend(frameon=False, fontsize=8.3, loc="best")
        if compare:
            gain_ax.axhline(1, color="#bab8b1", lw=1.0)
            gain_ax.set_xticks(x, labels)
            gain_ax.set_ylabel(f"complex / simpler {metric.lower()}")
            gain_ax.set_title("Additional strength from the complex rule", fontsize=10.5)
        for ax in axes:
            tidy_axes(ax, grid="y", hide=("top", "right"))
            ax.tick_params(length=0)
            if "lift" in metric.lower() or metric == "Conviction":
                ax.set_yscale("log")

        figure_titles(
            fig, str(rule).replace(" -> ", " → "), organ=organ,
            subtitle=f"{metric} movement across disease stages",
            params=" · ".join(value for value in [
                params, "dots are patient medians", "diamonds are stage medians",
            ] if value),
        )
        fig.subplots_adjust(bottom=0.18, wspace=0.28 if compare else 0.2)
        _finish(fig, save)
        figures.append(fig)
    return figures


def plot_prevalence_overview(prevalence, stages, organ=None, params=None, save=None):
    """Ranked complex-rule prevalence with the FOV counts visible."""
    if prevalence.empty:
        print("No rules pass the FOV threshold.")
        return
    share = prevalence.pivot(index="name", columns="stage", values="complex_share")
    count = prevalence.pivot(index="name", columns="stage", values="complex")
    total = prevalence.pivot(index="name", columns="stage", values="n_fovs")
    names = list(share.index)
    grid = share.to_numpy(float) * 100
    norm = Normalize(0, max(float(np.nanmax(grid)), 1))
    rgba = _RAMP(norm(np.nan_to_num(grid)))

    fig, ax = plt.subplots(figsize=(2.35 * len(stages) + 4.8,
                                    0.43 * len(names) + 2.2))
    ax.imshow(rgba, aspect="auto")
    for row in range(len(names)):
        for column in range(len(stages)):
            value = grid[row, column]
            if not np.isfinite(value):
                continue
            ink = text_on(rgba[row, column, :3])
            ax.text(column, row - 0.08, f"{value:.0f}%", ha="center", va="center",
                    fontsize=9.5, color=ink)
            ax.text(column, row + 0.19,
                    f"{int(count.iloc[row, column])}/{int(total.iloc[row, column])} FOVs",
                    ha="center", va="center", fontsize=7.7, color=ink)
    ax.set_xticks(range(len(stages)), stages, fontsize=10)
    ax.set_yticks(range(len(names)), names, fontsize=8.5)
    ax.tick_params(length=0)
    ax.spines[:].set_visible(False)
    ax.set_xticks(np.arange(len(stages) + 1) - 0.5, minor=True)
    ax.set_yticks(np.arange(len(names) + 1) - 0.5, minor=True)
    ax.grid(which="minor", color="white", lw=2.5)
    bar = fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=_RAMP), ax=ax,
                       pad=0.02, fraction=0.04)
    bar.set_label("complex-rule prevalence", fontsize=9)
    bar.ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0f}%"))
    bar.outline.set_visible(False)
    figure_titles(
        fig, "Complex-rule prevalence", organ=organ,
        subtitle="Top stage-changing rules that pass the FOV threshold",
        params=params, align="left",
    )
    _finish(fig, save)


def plot_prevalence_and_strength(prevalence, strengths, stages, metric="Lift",
                                 organ=None, params=None, new_only=False, save=None):
    """For one rule, compare stage prevalence and patient-level strength."""
    if prevalence.empty:
        print("No rule values to plot.")
        return
    rule = str(prevalence["name"].iloc[0])
    fig, (prevalence_ax, strength_ax) = plt.subplots(
        1, 2, figsize=(10.7, 4.55), gridspec_kw={"wspace": 0.30}
    )
    x = np.arange(len(stages), dtype=float)
    colors = {"complex": CLASS_COLORS["stronger_effect"], "simpler": "#9a8d70"}
    labels = {"complex": "higher-order rule", "simpler": "strongest simpler rule"}
    series = ["complex"] if new_only else ["complex", "simpler"]

    indexed = prevalence.set_index("stage")
    for key, offset in zip(series, [-0.06, 0.06] if len(series) == 2 else [0]):
        shares, annotations = [], []
        for stage in stages:
            if stage not in indexed.index:
                shares.append(np.nan)
                annotations.append("")
                continue
            row = indexed.loc[stage]
            shares.append(float(row[f"{key}_share"]))
            annotations.append(f"{int(row[key])}/{int(row['n_fovs'])}")
        prevalence_ax.plot(x + offset, shares, color=colors[key], lw=1.6,
                           marker="o", ms=6, label=labels[key])
        text_x = -4 if key == "complex" and len(series) == 2 else 4
        text_align = "right" if text_x < 0 else "left"
        for position, value, annotation in zip(x + offset, shares, annotations):
            if np.isfinite(value):
                prevalence_ax.annotate(annotation, (position, value), xytext=(text_x, 7),
                                       textcoords="offset points", ha=text_align,
                                       fontsize=7.6, color=colors[key])

    patient_counts = defaultdict(list)
    for key, offset in zip(series, [-0.07, 0.07] if len(series) == 2 else [0]):
        medians = []
        for position, stage in enumerate(stages):
            values = strengths.loc[
                (strengths["stage"] == stage) & (strengths["series"] == key), "value"
            ].dropna().to_numpy(float)
            patient_counts[key].append(len(values))
            jitter = np.linspace(-0.04, 0.04, len(values)) if len(values) > 1 else np.zeros(len(values))
            strength_ax.scatter(position + offset + jitter, values, s=28,
                                color=colors[key], alpha=0.65, edgecolor="white",
                                linewidth=0.5, zorder=3)
            medians.append(np.median(values) if len(values) else np.nan)
        strength_ax.plot(x + offset, medians, color=colors[key], lw=1.5, alpha=0.85)
        strength_ax.scatter(x + offset, medians, s=65, marker="D", color=colors[key],
                            edgecolor="white", linewidth=0.8, zorder=4,
                            label=labels[key])

    prevalence_ax.set_xticks(x, [
        f"{stage}\nFOV n={int(indexed.loc[stage, 'n_fovs'])}" if stage in indexed.index else stage
        for stage in stages
    ])
    prevalence_ax.set_ylim(-0.03, 1.10)
    prevalence_ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0%}"))
    prevalence_ax.set_ylabel("prevalence")
    prevalence_ax.set_title("Prevalence across FOVs", fontsize=10.7)

    patient_labels = []
    for position, stage in enumerate(stages):
        counts = [patient_counts[key][position] for key in series]
        count_text = str(counts[0]) if new_only else f"{counts[0]} | {counts[1]}"
        patient_labels.append(f"{stage}\npatient n={count_text}")
    strength_ax.set_xticks(x, patient_labels)
    strength_ax.set_ylabel(metric)
    strength_ax.set_title("Strength in rule-bearing FOVs", fontsize=10.7)
    reference = 0 if metric == "Leverage" else 1
    strength_ax.axhline(reference, color="#bab8b1", lw=1)
    for ax in (prevalence_ax, strength_ax):
        tidy_axes(ax, grid="y", hide=("top", "right"))
        ax.tick_params(length=0)
        ax.legend(frameon=False, fontsize=8.2, loc="best")
    figure_titles(
        fig, rule.replace(" -> ", " → "), organ=organ,
        subtitle=f"Prevalence and {metric} across disease stages",
        params=params,
    )
    fig.subplots_adjust(bottom=0.19, wspace=0.30)
    _finish(fig, save)


# What each figure measures, in the words the reader sees: the title over it, the label
# on its colour scale, and the note under the title, one sentence per line.
_MEASURES = {
    "earns": {
        "title": "Better complex rules by disease stage",
        "subtitle": "Share among FOVs containing the rule or one of its simpler parts",
        "scale": "% - better complex rule out of the FOVs with the rule or its parts",
        "note": [
            "Each cell: the share, then the two counts it divides — FOVs with "
            "informative complex rule | simpler rule only / better.",
            "Dashed outline: fewer than {min_n} FOVs behind it.",
        ],
    },
    "informative": {
        "title": "Informative complex rules by disease stage",
        "subtitle": "Share of eligible FOVs containing an informative complex rule",
        "scale": "% - informative complex rule out of all the FOVs in the stage",
        "note": [
            "Each cell: the share of informative complex rule's FOVs, then the counts: "
            "adds something | simpler only / better.",
            "The stage's other FOVs have neither.",
            "Dashed outline: fewer than {min_n} FOVs behind it.",
        ],
    },
}
_OTHER_NOTE = [
    "Each cell: the share of the stage, then the counts: adds something | "
    "simpler only / better.",
    "Dashed outline: fewer than {min_n} FOVs behind it.",
]


def _says(value, part):
    """One line of a figure's wording. A state without wording of its own gets a plain
    one, built from its label."""
    plain = {"title": STATE_LABELS.get(value, value).capitalize(),
             "subtitle": "Share of FOVs in each disease stage",
             "scale": f"% - {STATE_LABELS.get(value, value)}",
             "note": _OTHER_NOTE}
    return _MEASURES.get(value, {}).get(part, plain[part])


def plot_stage_heatmap(counts, stages, value="earns", organ=None, params=None,
                       min_n=5, save=None):
    """One number per rule per stage, as colour.

    `value='earns'` is the conditional one: of the FOVs where the rule OR at least one of
    its pairwise parts turned up, how often the rule was the better description. It takes
    abundance out of the picture - a rule that is rare but always earns its place scores
    high, one that is everywhere and always redundant scores low. Any state name works
    too ('informative', 'covered', 'adds', ...), and is then a share of the whole stage.

    Each cell carries the two counts behind it, and a cell resting on fewer than `min_n`
    FOVs is outlined rather than dropped.
    """
    if counts.empty:
        print("No rules to plot.")
        return

    earns = value == "earns"
    wide = _wide(counts, stages, f"{value}_share")
    grid = wide.to_numpy(dtype=float) * 100
    names = list(wide.index)
    eligible = "eligibility_on" in counts and bool(counts["eligibility_on"].any())
    behind_column = "n_present" if earns else "n_fovs"
    behind = _wide(counts, stages, behind_column).reindex(names)
    adds = _wide(counts, stages, "informative").reindex(names).to_numpy()
    covered = _wide(counts, stages, "covered").reindex(names).to_numpy()
    eligible_n = _wide(counts, stages, "n_fovs").reindex(names).to_numpy()
    thin = behind.to_numpy() < min_n

    ramp = _DIVERGING if earns else _RAMP
    norm = Normalize(0, 100) if earns else Normalize(0, np.nanmax(grid))

    rgba = ramp(norm(np.nan_to_num(grid)))

    fig, ax = plt.subplots(figsize=(2.5 * len(stages) + 5.0, len(names) * 0.42 + 2.0))
    ax.imshow(rgba, aspect="auto")

    for row, column in zip(*np.nonzero(thin)):
        ax.add_patch(Rectangle((column - 0.42, row - 0.42), 0.84, 0.84, fill=False,
                               edgecolor=INK, linestyle=_THIN_EDGE, linewidth=1.2,
                               zorder=5))

    for row in range(grid.shape[0]):
        for column in range(grid.shape[1]):
            share = grid[row, column]
            if not np.isfinite(share):
                continue
            # Read the contrast off the colour actually painted, not off the number -
            # the scale's top is whatever the data reaches, so no fixed cutoff works.
            ink = text_on(rgba[row, column, :3])
            ax.annotate(f"{share:.0f}%", xy=(column - 0.04, row), ha="right",
                        va="center", fontsize=10, color=ink)
            counts_text = f"{int(adds[row, column])} | {int(covered[row, column])}"
            if eligible and not earns:
                counts_text += f" / e={int(eligible_n[row, column])}"
            ax.annotate(counts_text,
                        xy=(column + 0.04, row), ha="left", va="center", fontsize=8,
                        color=ink)

    ax.set_xticks(range(len(stages)), _stage_ticks(counts, stages), fontsize=10.5)
    ax.set_yticks(range(len(names)), names, fontsize=8.5)
    ax.tick_params(length=0)
    ax.spines[:].set_visible(False)
    ax.set_xticks(np.arange(len(stages) + 1) - 0.5, minor=True)
    ax.set_yticks(np.arange(len(names) + 1) - 0.5, minor=True)
    ax.grid(which="minor", color="white", lw=2.5)

    scale = _says(value, "scale")
    if eligible and not earns:
        scale = scale.replace("all the FOVs in the stage", "eligible FOVs in the stage")
    bar = fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=ramp), ax=ax,
                       pad=0.02, fraction=0.04)
    bar.set_label(scale, fontsize=9)
    bar.outline.set_visible(False)

    counts_note = ("Cells: share · informative | pair-covered"
                   + (" / eligible" if eligible and not earns else "")
                   + f" · dashed = fewer than {min_n} supporting FOVs")
    figure_titles(fig, _says(value, "title"), organ=organ,
                  subtitle=_says(value, "subtitle"), params=params,
                  note=counts_note, align="left")
    _finish(fig, save)


# One hue per rule, far enough apart to follow a line across three stages.
_LINE_COLORS = ["#22596b", "#c4622d", "#4c8496", "#8a5a78", "#5f8f5a", "#b8912f",
                "#7a6ba8", "#a34a4a", "#3f7d6e", "#96739b"]


def _one_way(change, falling, top_n):
    """One direction's rules - all of them, and the `top_n` that moved most."""
    moved = change[change < 0] if falling else change[change > 0]
    return moved, list(moved.abs().sort_values(ascending=False).head(top_n).index)


def _widest(fig, labels, fontsize):
    """How wide the longest of these labels prints, in points."""
    if not labels:
        return 0.0
    try:
        renderer = fig.canvas.get_renderer()
    except AttributeError:
        renderer = fig._get_renderer()
    probe = fig.text(0, 0, "", fontsize=fontsize)
    widest = 0.0
    for label in labels:
        probe.set_text(label)
        widest = max(widest, probe.get_window_extent(renderer).width)
    probe.remove()
    return widest * 72 / fig.dpi


def _room_for_names(fig, left_names, right_names):
    """Give the names, the panels and the axis numbers a lane each, then size the figure
    to whatever that adds up to.

    The names are drawn outside the panels, where `tight_layout` cannot see them, so the
    space is measured here instead. Both panels put their numbers on the right, so the
    lane a name column sits in is free of them.
    """
    numbers = _widest(fig, ["100"], _TICK_SIZE) + 10     # a panel's scale, on its right
    left_lane = _EDGE + _widest(fig, left_names, _NAME_SIZE) + _NAME_GAP
    middle_lane = _widest(fig, right_names, _NAME_SIZE) + _NAME_GAP + numbers
    width = left_lane + middle_lane + numbers + _EDGE + 2 * _PANEL_WIDTH

    fig.set_size_inches(width / 72, fig.get_figheight())
    fig.subplots_adjust(left=left_lane / width, right=1 - (numbers + _EDGE) / width,
                        top=0.835, bottom=0.10, wspace=middle_lane / _PANEL_WIDTH)


def plot_stage_trend(counts, stages, value="earns", organ=None, params=None,
                     top_n=8, min_n=5, save=None):
    """A line per rule across the stages, the ones that fade apart from the ones that
    arrive, so neither half is drawn over the other.

    Laid out like `differential_vis.plot_trend`: every name in a column down the left of
    its own panel with a short arrow to where its line starts, and the change written
    beside it. The % scale sits on the right of each panel, clear of those arrows. Each
    panel keeps its own scale - the two rarely cover the same range, and one shared axis
    leaves whichever panel is smaller crushed into a corner.

    `value` is as for the heatmap. Dot size is how many FOVs are behind the point, and a
    point resting on fewer than `min_n` is drawn hollow.
    """
    if counts.empty:
        print("No rules to plot.")
        return

    wide = _wide(counts, stages, f"{value}_share") * 100
    behind = _wide(counts, stages, "n_present").reindex(wide.index)
    change = wide[stages[-1]] - wide[stages[0]]
    x = np.arange(len(stages))
    ticks = _stage_ticks(counts, stages)

    sides = []
    for falling, title in ((True, "fading as it worsens"),
                           (False, "arriving as it worsens")):
        moved, picked = _one_way(change, falling, top_n)
        sides.append((title, moved, picked))
    # Measured before anything is drawn: the figure is sized around these.
    labels = [[f"{name}   {change[name]:+.0f}" for name in picked]
              for _, _, picked in sides]

    fig, axes = plt.subplots(1, 2, figsize=(18, 5.8))
    _room_for_names(fig, labels[0], labels[1])

    for ax, (title, moved, picked), names in zip(axes, sides, labels):
        ax.set_title(f"{title}   ({len(moved)} rules)", fontsize=10.5, color=INK,
                     loc="left", pad=22)
        ax.annotate(_says(value, "scale"), xy=(0, 1.012),
                    xycoords="axes fraction", fontsize=8, color="#898781")
        ax.set_xticks(x, ticks, fontsize=10)
        ax.set_xlim(-0.08, len(stages) - 0.92)
        tidy_axes(ax, grid="y", hide=("top", "right", "bottom", "left"))
        ax.yaxis.tick_right()                    # the arrows come in from the left
        ax.tick_params(length=0, labelsize=9, colors="#898781")

        if not picked:
            ax.annotate("none", xy=(0.5, 0.5), xycoords="axes fraction", ha="center",
                        fontsize=9, color="#96958f")
            ax.set_yticks([])                    # nothing drawn, so no scale to read
            continue

        # Only this panel's own lines set its scale.
        block, counted = wide.loc[picked], behind.loc[picked]
        low, high = float(block.to_numpy().min()), float(block.to_numpy().max())
        pad = max((high - low) * 0.10, 2.0)
        ax.set_ylim(low - pad, high + pad)
        # x from the axes' own left edge, y in data units, then shifted a fixed number
        # of points further left.
        column_at = offset_copy(
            blended_transform_factory(ax.transAxes, ax.transData),
            fig=fig, x=-_NAME_GAP, y=0, units="points")

        rows = spread_labels(block[stages[0]].to_numpy(),
                             (high - low + 2 * pad) * 0.075,
                             low - pad * 0.6, high + pad * 0.6)

        for (name, line), label, row, color in zip(block.iterrows(), names, rows,
                                                   _LINE_COLORS):
            values = line[stages].to_numpy(dtype=float)
            n = counted.loc[name, stages].to_numpy(dtype=float)
            solid = n >= min_n

            ax.plot(x, values, color=color, lw=1.8, alpha=0.9, zorder=3,
                    solid_capstyle="round")
            ax.scatter(x[solid], values[solid], s=(14 + 5 * np.sqrt(n))[solid],
                       color=color, edgecolor="white", linewidth=1.1, zorder=4)
            ax.scatter(x[~solid], values[~solid], s=(14 + 5 * np.sqrt(n))[~solid],
                       facecolors="white", edgecolors=color, linewidth=1.2, zorder=4)
            # Right-aligned in a column a fixed distance left of the panel, with an
            # arrow in to the line's own starting point.
            ax.annotate(label, xy=(x[0], values[0]),
                        xytext=(0, row), textcoords=column_at,
                        annotation_clip=False, fontsize=_NAME_SIZE, va="center",
                        ha="right", color=color, zorder=5,
                        arrowprops=dict(arrowstyle="->", color=color, alpha=0.5, lw=0.8,
                                        shrinkA=4, shrinkB=4))

    figure_titles(
        fig, "Rules that fade or arrive with disease severity", organ=organ,
        subtitle=f"Top {top_n} changes each way · {_says(value, 'subtitle')}",
        params=(params or "") + f" · hollow dot = fewer than {min_n} supporting FOVs",
    )
    _finish(fig, save)
