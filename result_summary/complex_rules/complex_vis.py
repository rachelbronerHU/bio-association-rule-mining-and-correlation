"""Drawing for the higher-order rule analysis: what the longer rules add.

Only plotting lives here; the counting stays in the notebook.
The shared pieces - saving, titles, ink, axis chrome - come from `vis_helper.py`.
"""
import textwrap
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import to_rgb
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

from vis_helper import (
    save_figure, figure_titles, tidy_axes, _plain_log_ticks,
    plot_fov, set_cell_colors, resolve_cell_colors, INK, ZERO, NEUTRAL,
)


_FIGURE_DIR = Path(__file__).resolve().parent / "summary_downloads"


def _finish(fig, save=None, dpi=200):
    """Save complex-rule figures beside their notebooks, then show them."""
    save_figure(fig, save, dpi=dpi, figure_dir=_FIGURE_DIR)
    plt.show()


def clear_exports(names):
    """Remove known report figures before rebuilding them."""
    for name in filter(None, names):
        (_FIGURE_DIR / name).unlink(missing_ok=True)

# A ramp, not a set of categories: the verdicts are ordered, nothing -> something.
# Plum sits off the ramp on purpose - a rule whose parent was never measured is a
# different kind of statement, not a stronger one.
CLASS_COLORS = {
    "redundant_by_simpler": "#e6e0d4",
    "consequent_driven":    "#c2b8a3",
    "simpler_are_noise":    "#bcd2c6",
    "consequent_is_noise":  "#4a7d70",
    "stronger_effect":      "#2e7387",
    "new":                  "#8a5a78",
}
_BAND = "#f1efe9"         # the zone where a longer rule is no better than its parts

# The verdicts that say the rule was already accounted for. Drawn first, so where they
# end is the share of the shape worth reading.
ADDS_NOTHING = ("redundant_by_simpler", "consequent_driven")

_BOUND = "new"            # nothing shorter was mined, so its gain is a floor

_GROUP_GAP = 1.0          # blank rows between two groups, where the group's name goes
_MIN_LABEL = 0.045        # a segment narrower than this has no room for its number
_GAIN_TICKS = [0.5, 0.7, 1, 1.2, 1.5, 2, 2.5, 3, 4, 5, 7, 10, 15, 20, 30, 50]


def text_on(color):
    """Ink or white, whichever stays readable on this colour. Shared with complex_stages_vis."""
    r, g, b = to_rgb(color)
    return "white" if (0.299 * r + 0.587 * g + 0.114 * b) < 0.55 else INK


def _rule_parts(rule):
    """Cell types on each side of one stored rule name."""
    ant, con = str(rule).split(" -> ", 1)
    clean = lambda side: [
        item.replace("_CENTER", "").replace("_NEIGHBOR", "")
        for item in side.split(" + ")
    ]
    return clean(ant), clean(con)


def _plain_rule(rule):
    ant, con = _rule_parts(rule)
    return f"{', '.join(sorted(ant))} → {', '.join(sorted(con))}"


# Short labels, because five metrics plus FDR have to fit over one small panel.
METRIC_LABELS = (
    ("Lift", "lift"),
    ("Confidence", "conf"),
    ("Leverage", "lev"),
    ("Support", "supp"),
    ("Conviction", "conv"),
)
METRICS_PER_ROW = 3


def _rule_metric_text(values, include_fdr=False, wrap=False):
    """Compact metric line used above rule-highlighted FOVs."""
    if not isinstance(values, dict):
        return ""
    parts = []
    for key, label in METRIC_LABELS:
        value = values.get(key, np.nan)
        if pd.notna(value):
            parts.append(f"{label} {value:.3g}")
    fdr = values.get("Individual_FDR", np.nan)
    if include_fdr and pd.notna(fdr):
        parts.append(f"FDR {fdr:.3g}")
    if not parts:
        return ""
    if wrap or len(parts) > METRICS_PER_ROW:
        rows = [parts[start:start + METRICS_PER_ROW]
                for start in range(0, len(parts), METRICS_PER_ROW)]
        return "\n".join(" · ".join(row) for row in rows)
    return " · ".join(parts)


def plot_rule_fov_pairs(examples, df_cells, df_fovs, save=None, cell_size=16):
    """One figure per rule: full FOV, higher-order rule, then its mined parents."""
    if examples is None or len(examples) == 0:
        print("No FOV examples to plot.")
        return []
    set_cell_colors(df_cells)

    figures = []
    for _, example in pd.DataFrame(examples).reset_index(drop=True).iterrows():
        fov = example["FOV"]
        detail = str(example.get("detail", "")).strip()
        simpler = example.get("simpler_rules", [])
        simpler = simpler if isinstance(simpler, (list, tuple)) else []
        shown = list(example["antecedent_cells"]) + list(example["consequent_cells"])
        for name in simpler:
            ant, con = _rule_parts(name)
            shown += ant + con
        cell_colors = resolve_cell_colors(shown)
        panels = [(
            "Higher-order rule", example["rule"],
            list(example["antecedent_cells"]), list(example["consequent_cells"]),
            example.get("complex_metrics", {}),
        )]
        simpler_metrics = example.get("simpler_metrics", [])
        for number, rule in enumerate(simpler, 1):
            ant, con = _rule_parts(rule)
            metrics = (
                simpler_metrics[number - 1]
                if number <= len(simpler_metrics) else {}
            )
            label = (
                "Strongest simpler rule"
                if len(simpler) == 1 else f"Simpler rule {number}"
            )
            panels.append((label, rule, ant, con, metrics))

        panel_count = 1 + len(panels)
        nrows = int(np.ceil(panel_count / 4))
        ncols = int(np.ceil(panel_count / nrows))
        fig, axes = plt.subplots(
            nrows, ncols, figsize=(4.0 * ncols, 4.25 * nrows + 0.9),
            squeeze=False, facecolor="white",
            gridspec_kw={"wspace": 0.18, "hspace": 0.30},
        )
        axes = axes.ravel()
        plot_fov(
            fov, "", df_cells, df_fovs, ax=axes[0], show_legend=False,
            cell_size=cell_size, colors=cell_colors,
        )
        axes[0].set_title("Full FOV", fontsize=10)

        target_types = []
        for ax, (label, rule, ant, con, metrics) in zip(axes[1:], panels):
            target_types.extend(ant + con)
            plot_fov(
                fov, "", df_cells, df_fovs, target_ant_cells=ant,
                target_cons_cells=con, ax=ax, show_legend=False,
                cell_size=cell_size, colors=cell_colors,
            )
            is_simpler = label != "Higher-order rule"
            name = (
                _plain_rule(rule)
                if is_simpler else str(rule).replace(" -> ", " → ")
            )
            name = "\n".join(textwrap.wrap(
                name, width=34, break_long_words=False, break_on_hyphens=False,
            ))
            metrics = _rule_metric_text(
                metrics, include_fdr=is_simpler, wrap=is_simpler,
            )
            ax.set_title(
                f"{label}\n{name}" + (f"\n{metrics}" if metrics else ""),
                fontsize=8.8,
            )
        for ax in axes[panel_count:]:
            ax.set_visible(False)

        target_types = list(dict.fromkeys(target_types))
        handles = [
            plt.Line2D(
                [0], [0], marker="o", linestyle="none",
                markerfacecolor=cell_colors.get(cell, "black"),
                markeredgecolor="none", markersize=7, label=cell,
            )
            for cell in target_types
        ]
        # No legend title: every handle is already labelled with its cell type, and
        # the extra line collided with the bottom row's x-axis label.
        fig.legend(
            handles=handles, frameon=False,
            ncol=max(len(handles), 1), loc="lower center",
            bbox_to_anchor=(0.5, 0.004), fontsize=8,
        )
        meta = df_fovs[df_fovs["FOV"] == fov]
        organ = meta["Organ"].iat[0] if not meta.empty and "Organ" in meta else None
        axes_top = figure_titles(
            fig, str(example["rule"]).replace(" -> ", " → "),
            organ=organ, subtitle=f"FOV {fov}", params=detail,
        )
        axes_top -= 34 / (fig.get_figheight() * 72)
        fig.subplots_adjust(
            bottom=0.14 if nrows > 1 else 0.18,
            top=axes_top, wspace=0.18, hspace=0.30,
        )
        _finish(fig, save)
        figures.append(fig)
    return figures


def _key(handles, ax, at):
    """One column of keys down the right, clear of whatever the plot writes there."""
    ax.legend(handles=handles, fontsize=8.5, frameon=False, ncol=1,
              loc="upper left", bbox_to_anchor=(at, 1.0), handlelength=1.1,
              borderpad=0, labelspacing=0.7)


_LEGEND_FLOOR = 0.005     # a class thinner than this sends the eye hunting for nothing


def _class_keys(verdicts, class_labels):
    """A swatch per verdict, in the palette's own order.

    A class holding under half a percent is left out: 20 rows in 14,000 cannot be found
    on the page, and a key for them is a search with no answer.
    """
    share = pd.Series(list(verdicts)).value_counts(normalize=True)
    return [Patch(facecolor=CLASS_COLORS[v], edgecolor="white",
                  label=(class_labels or {}).get(v, v))
            for v in CLASS_COLORS
            if v in share.index and share[v] >= _LEGEND_FLOOR]


def _blocks(counts):
    """y for each bar, and the (group name, rows) blocks a MultiIndex splits them into."""
    names = (list(counts.index.get_level_values(0)) if counts.index.nlevels > 1
             else [None] * len(counts))
    starts = [i for i in range(len(names)) if i == 0 or names[i] != names[i - 1]]

    y = np.arange(len(names), dtype=float)
    for start in starts[1:]:
        y[start:] += _GROUP_GAP

    edges = starts + [len(names)]
    return y, [(names[a], range(a, b)) for a, b in zip(starts, edges[1:])]


def plot_class_split(counts, labels=None, class_labels=None, organ=None,
                     subtitle=None, params=None,
                     title="What the longer rules add", unit_label="rules", save=None):
    """How the library's verdicts split, one bar per rule shape.

    `counts` is rows x classes, already in the order to draw: the index is the rule
    shape, or a (group, shape) MultiIndex for one block per group; the columns are the
    classes, worst first.

    Bars are shares of their own row, so shapes with very different totals still line up -
    the total itself is written on the right. The line joining them marks where the greys
    end, drawn per group and never across two, since the groups are not a sequence.
    """
    if counts.empty:
        print("No rules to plot.")
        return

    totals = counts.sum(axis=1)
    shares = counts.div(totals, axis=0)
    y, blocks = _blocks(counts)
    grouped = blocks[0][0] is not None

    fig, ax = plt.subplots(figsize=(10, y.max() * 0.5 + 1.7))

    start = np.zeros(len(counts))
    for column in counts.columns:
        width = shares[column].to_numpy()
        color = CLASS_COLORS.get(column, NEUTRAL)
        ax.barh(y, width, left=start, height=0.62, color=color,
                edgecolor="white", linewidth=1.1, zorder=3)
        for row in np.flatnonzero(width >= _MIN_LABEL):
            ax.annotate(f"{width[row] * 100:.0f}", zorder=4, fontsize=8.5,
                        xy=(start[row] + width[row] / 2, y[row]),
                        ha="center", va="center", color=text_on(color))
        start = start + width

    edge = shares[[c for c in counts.columns if c in ADDS_NOTHING]].sum(axis=1).to_numpy()
    for name, rows in blocks:
        ax.plot(edge[rows], y[rows], color=ZERO, lw=1.3, marker="o", markersize=4.5,
                markerfacecolor="white", markeredgewidth=1.3, zorder=5)
        if name:
            ax.annotate(str(name), xy=(0, y[rows[0]] - 0.62), va="bottom",
                        xycoords=("axes fraction", "data"),
                        fontsize=10.5, fontweight="bold", color=INK)

    for row, total in enumerate(totals):
        ax.annotate(f"{int(total):,}", xy=(1.0, y[row]), xytext=(7, 0),
                    textcoords="offset points", ha="left", va="center",
                    fontsize=8.5, color=INK, annotation_clip=False)

    ax.set_yticks(y, [(labels or {}).get(s, s)
                      for s in counts.index.get_level_values(-1)], fontsize=10)
    ax.set_ylim(y.max() + 0.7, -1.1 if grouped else -0.7)
    ax.set_xlim(0, 1)
    ax.set_xticks(np.linspace(0, 1, 5), [f"{v:.0%}" for v in np.linspace(0, 1, 5)])
    ax.set_xlabel(f"share within each shape      (number of {unit_label}, on the right)",
                  fontsize=10)
    ax.tick_params(axis="y", length=0)

    _key(_class_keys(counts.columns, class_labels), ax, 1.10)
    tidy_axes(ax, grid="x", hide=("top", "right", "left"))
    figure_titles(fig, title, organ=organ, subtitle=subtitle, params=params,
                  align="left")
    _finish(fig, save)


# ---------------------------------------------------------------------------
# Plot B - how much stronger the longer rule is than its best shorter part
# ---------------------------------------------------------------------------

def _dot_sizes(fovs):
    """Bigger dot = more FOVs behind the row. Square-rooted, so 40 FOVs reads as more
    than 5 without swamping the row it sits on."""
    return 26 + 9.0 * np.sqrt(np.asarray(fovs, dtype=float))


def _no_change_band(ax, min_gain):
    """Everything at or under the bar, shaded: this zone is not an improvement.

    A band instead of two labelled lines - the labels sat on top of each other, and the
    zone is the point, not its number.
    """
    bottom = ax.get_ylim()[0]          # the data's own floor, not the span's
    ax.axhspan(bottom, min_gain, color=_BAND, zorder=0)
    ax.axhline(1.0, color=ZERO, lw=1.2, zorder=1)
    ax.set_ylim(bottom=bottom)


def plot_gain_scatter(table, class_labels=None, organ=None, subtitle=None,
                      params=None, metric="Lift", min_gain=1.1, save=None):
    """Every mined instance of a surviving longer rule: how strong, and how much it won by.

    One dot per rule per FOV, so nothing is averaged: each dot's strength, parent and gain
    are all measured in the same FOV. Across is the rule's own strength, so a weak rule
    and a strong one never look alike. Up is the gain over the strongest shorter rule
    inside it - on its own axis, so the spread of the dots *is* the distribution of gains.
    The panel on the right is that same spread as a histogram, split by verdict.

    Hollow dots had nothing shorter mined in that FOV: their parent is pinned at the bar
    it must have failed, the most it could have been, so their gain is a floor.
    """
    if table.empty:
        print("No rules to plot.")
        return

    bound = (table["verdict"] == _BOUND).to_numpy()
    x = table["strength"].to_numpy(dtype=float)
    y = table["gain"].to_numpy(dtype=float)
    colors = np.array([CLASS_COLORS.get(v, NEUTRAL) for v in table["verdict"]])

    fig, (ax, side) = plt.subplots(1, 2, figsize=(11.5, 6.2), sharey=True,
                                   gridspec_kw={"width_ratios": [4.4, 1], "wspace": 0.03})

    ax.scatter(x[~bound], y[~bound], s=13, c=colors[~bound], linewidth=0,
               alpha=0.45, zorder=4)
    ax.scatter(x[bound], y[bound], s=15, facecolors="none", edgecolors=colors[bound],
               linewidth=0.7, alpha=0.45, zorder=3)

    ax.set_xscale("log")
    ax.set_yscale("log")
    _no_change_band(ax, min_gain)
    _plain_log_ticks(ax, "x", _GAIN_TICKS)
    _plain_log_ticks(ax, "y", _GAIN_TICKS)
    ax.set_xlabel(f"how strong the rule is      ({metric.lower()}, 1 = chance)", fontsize=10)
    ax.set_ylabel(f"how much it beats its best shorter rule      (x {metric.lower()})",
                  fontsize=10)
    tidy_axes(ax, grid="both")

    # The same gains again as a distribution, so the spread is a shape and not a guess.
    present = [v for v in CLASS_COLORS if v in set(table["verdict"])]
    side.hist([table.loc[table["verdict"] == v, "gain"] for v in present],
              bins=np.geomspace(y.min(), y.max(), 36), orientation="horizontal",
              stacked=True, color=[CLASS_COLORS[v] for v in present],
              edgecolor="white", linewidth=0.4)
    _no_change_band(side, min_gain)
    side.set_xlabel("rules", fontsize=9)
    tidy_axes(side, grid="y", hide=("top", "right", "left"))
    side.tick_params(axis="y", length=0, labelleft=False)

    keys = _class_keys(table["verdict"], class_labels)
    keys.append(Line2D([0], [0], marker="o", linestyle="none", markersize=8,
                       markerfacecolor="none", markeredgecolor=INK, markeredgewidth=1.3,
                       label="hollow: no parent measured"))
    _key(keys, side, 1.25)
    figure_titles(fig, "What the longer rules gain over their parts", organ=organ,
                  subtitle=subtitle, params=params, align="left")
    _finish(fig, save)


def plot_gain_dumbbell(table, class_labels=None, organ=None, subtitle=None,
                       params=None, metric="Lift", top_n=15, save=None):
    """The biggest gains, by name: where the best shorter rule sits, and where this one does.

    One row per rule, hollow dot the shorter rule, filled dot the longer one, and the bar
    between them is the gain. Each is named on its own side - the shorter rule down the
    left, the longer one down the right - so no row has to be read across twice. The axis is the strength itself, so a pair sitting at 1.3 and
    a pair sitting at 8 never read the same however equal their ratio.

    A dotted bar means nothing shorter was mined, so the gain is a floor.
    """
    if table.empty:
        print("No rules to plot.")
        return

    top = table.nlargest(top_n, "gain").sort_values("gain")
    y = np.arange(len(top))
    parent = top["parent"].to_numpy(dtype=float)
    strength = top["strength"].to_numpy(dtype=float)
    bound = (top["verdict"] == _BOUND).to_numpy()
    colors = [CLASS_COLORS.get(v, NEUTRAL) for v in top["verdict"]]

    fig, ax = plt.subplots(figsize=(13, max(3.0, len(top) * 0.34 + 1.6)))
    for row in range(len(top)):
        ax.plot([parent[row], strength[row]], [y[row], y[row]], color=colors[row],
                lw=2.6, alpha=0.55, zorder=2,
                linestyle=(0, (2, 2)) if bound[row] else "-")
    sizes = _dot_sizes(top["fovs"])
    ax.scatter(parent, y, s=sizes, facecolors="white", edgecolors=colors,
               linewidth=1.6, zorder=3)
    ax.scatter(strength, y, s=sizes, color=colors, edgecolor="white",
               linewidth=1.2, zorder=4)

    ax.set_yticks(y, top["parent_name"], fontsize=8.5)
    ax.set_ylim(-0.9, len(top) - 0.1)
    ax.set_xscale("log")
    _plain_log_ticks(ax, "x", _GAIN_TICKS)
    ax.set_xlabel(f"how strong the rule is      ({metric.lower()}, 1 = chance)", fontsize=10)
    ax.tick_params(axis="y", length=0)

    # Each name sits beside its own dot: the shorter rule on the left, the longer one
    # on the right, then by how much it won and how many FOVs it was seen in.
    for row, (name, gain, fovs, is_bound) in enumerate(
            zip(top["name"], top["gain"], top["fovs"], bound)):
        ax.annotate(str(name), xy=(1.02, y[row]), xycoords=("axes fraction", "data"),
                    ha="left", va="center", fontsize=7.5, color=INK,
                    annotation_clip=False)
        ax.annotate(f"{'>=' if is_bound else ''}{gain:.1f}x", xy=(1.36, y[row]),
                    xycoords=("axes fraction", "data"), ha="left", va="center",
                    fontsize=8.5, color=INK, annotation_clip=False)
        ax.annotate(f"{int(fovs)}", xy=(1.44, y[row]),
                    xycoords=("axes fraction", "data"), ha="left", va="center",
                    fontsize=8.5, color=INK, annotation_clip=False)
    for at, header in ((1.02, "the longer rule"), (1.36, "gain"), (1.44, "FOVs")):
        ax.annotate(header, xy=(at, len(top) - 0.35), xycoords=("axes fraction", "data"),
                    ha="left", va="center", fontsize=8, color=INK, annotation_clip=False)
    ax.annotate("the shorter rule it beat", xy=(-0.01, len(top) - 0.35),
                xycoords=("axes fraction", "data"), ha="right", va="center",
                fontsize=8, color=INK, annotation_clip=False)

    keys = [Line2D([0], [0], marker="o", linestyle="none", markersize=8,
                   markerfacecolor="white", markeredgecolor=INK, label="shorter rule"),
            Line2D([0], [0], marker="o", linestyle="none", markersize=8,
                   markerfacecolor=INK, markeredgecolor="white", label="the longer rule")]
    keys += _class_keys(top["verdict"], class_labels)
    _key(keys, ax, 1.53)
    tidy_axes(ax, grid="x", hide=("top", "right", "left"))
    figure_titles(fig, "The biggest gains, named", organ=organ,
                  subtitle=subtitle, params=params, align="left")
    _finish(fig, save)


def plot_rule_occurrences(rows, rule, metric="Lift", stage_column="Pathological score",
                          organ_order=None, stage_order=None, threshold=None, save=None):
    """Show every FOV carrying one recurring rule, grouped by organ and stage."""
    current = rows[rows["name"] == rule].copy()
    if current.empty:
        print(f"No occurrences to plot for {rule}.")
        return
    organ_order = organ_order or list(current["Organ"].dropna().unique())
    stage_order = stage_order or list(current[stage_column].dropna().unique())
    groups = [(organ, stage) for organ in organ_order for stage in stage_order]
    present = [(organ, stage) for organ, stage in groups if not current[
        (current["Organ"] == organ) & (current[stage_column] == stage)
    ].empty]

    fig, ax = plt.subplots(figsize=(max(7.2, 1.8 * len(present)), 4.6))
    labels = []
    for position, (organ, stage) in enumerate(present):
        block = current[
            (current["Organ"] == organ) & (current[stage_column] == stage)
        ]
        values = block[metric].astype(float).to_numpy()
        jitter = np.linspace(-0.08, 0.08, len(values)) if len(values) > 1 else [0]
        ax.scatter(position + np.asarray(jitter), values, s=34, color=CLASS_COLORS["new"],
                   alpha=0.65, edgecolor="white", linewidth=0.6, zorder=3)
        ax.scatter(position, np.median(values), s=78, marker="D",
                   color=CLASS_COLORS["new"], edgecolor="white", linewidth=0.8, zorder=4)
        patients = block["PatientID"].nunique()
        fov_label = "FOV" if len(block) == 1 else "FOVs"
        patient_label = "patient" if patients == 1 else "patients"
        labels.append(
            f"{organ}\n{stage}\n{len(block)} {fov_label}\n{patients} {patient_label}"
        )

    ax.axhline(1, color="#bab8b1", lw=1.0)
    if threshold is not None:
        ax.axhline(threshold, color=CLASS_COLORS["new"], lw=1.0, ls=(0, (3, 2)),
                   label=f"selection threshold = {threshold:g}")
        ax.legend(frameon=False, fontsize=8.3)
    ax.set_xticks(range(len(present)), labels)
    ax.set_ylabel(metric)
    if metric in {"Lift", "Conviction"}:
        ax.set_yscale("log")
        _plain_log_ticks(ax, "y")
    tidy_axes(ax, grid="y", hide=("top", "right"))
    ax.tick_params(length=0)
    figure_titles(
        fig, rule.replace(" -> ", " → "),
        subtitle="Reproducibility of a recurring new rule",
        params=(f"stage column: {stage_column} · all rule-bearing FOVs are shown · "
                "dots are FOVs · diamonds are group medians"),
    )
    fig.subplots_adjust(bottom=0.28)
    _finish(fig, save)


def clear_all_exports():
    """Empty the figure folder, so a rename can never leave a stale file behind."""
    _FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    removed = 0
    for path in _FIGURE_DIR.glob("*.png"):
        path.unlink()
        removed += 1
    print(f"cleared {removed} figures from {_FIGURE_DIR.name}/")
