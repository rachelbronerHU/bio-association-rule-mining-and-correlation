"""Drawing for the differential-rule analysis: which rules tell two groups apart.

Only plotting lives here; the tests and the tables stay in the notebook.
The shared pieces - saving, titles, group colours - come from `vis_helper.py`.
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from matplotlib.colors import to_rgba
from matplotlib.lines import Line2D

from vis_helper import (save_figure, _titled, _category_colors, tidy_axes, spread_labels,
                        plot_fov, set_cell_colors,
                        NEUTRAL as _NEUTRAL, HAIRLINE as _HAIRLINE,
                        ZERO as _ZERO, INK as _INK)


_FIGURE_DIR = Path(__file__).resolve().parent.parent / "summary_downloads"

# A4 minus the summary's 1.8 cm margins: the width a full-width figure is given.
_TEXT_WIDTH = 6.85
_PANEL_FONTS = {
    "font.size": 8.2, "axes.titlesize": 8.4, "axes.labelsize": 8.0,
    "xtick.labelsize": 7.8, "ytick.labelsize": 7.8, "legend.fontsize": 7.8,
}


def _finish(fig, save=None, dpi=200):
    """Save differential-rule figures beside their notebooks, then show them."""
    save_figure(fig, save, dpi=dpi, figure_dir=_FIGURE_DIR)
    plt.show()


def _group_colors(label_a, label_b):
    """One colour per group. Stages keep the colours they have everywhere else."""
    colors, _ = _category_colors(pd.Series([label_a, label_b]))
    return to_rgba(colors[label_a]), to_rgba(colors[label_b])


def _legend(ax, label_a, label_b, color_a, color_b, show_hollow=False, over_label=None):
    """Two dots naming the groups, so colour is never the only clue.

    Above the axes, where it can never land on a dot or a rule name. `show_hollow`
    adds the key for the rules that did not pass the cutoffs, `over_label` the one
    for the rules drawn on the top edge.
    """
    marks = [Line2D([0], [0], marker="o", linestyle="none", markersize=8,
                    markerfacecolor=c, markeredgecolor="white", label=l)
             for l, c in ((label_a, color_a), (label_b, color_b))]
    if show_hollow:
        marks.append(Line2D([0], [0], marker="o", linestyle="none", markersize=8,
                            markerfacecolor="white", markeredgecolor="#96958f",
                            markeredgewidth=1.6, label="did not pass"))
    if over_label:
        marks.append(Line2D([0], [0], marker="^", linestyle="none", markersize=8,
                            markerfacecolor="#96958f", markeredgecolor="white",
                            label=over_label))
    ax.legend(handles=marks, fontsize=9, frameon=False, ncol=len(marks),
              loc="lower right", bbox_to_anchor=(1.0, 1.0))


def _flip_marks(comparison, label_a, label_b):
    """'±' in front of a rule whose two groups sit on opposite sides of zero.

    The cell types attract in one group and avoid in the other - the direction itself
    changed, not only how often. Everything else gets nothing, so the mark is only ever
    there where it means something.
    """
    a = comparison[f"net_{label_a}"].to_numpy(dtype=float)
    b = comparison[f"net_{label_b}"].to_numpy(dtype=float)
    return np.where(((a > 0) & (b < 0)) | ((a < 0) & (b > 0)), "±  ", "")


def _title(ax, text, scope):
    """Short figure title followed by a separate scope subtitle."""
    fig = ax.figure
    fig.suptitle(text, fontsize=13.5, y=0.985)
    if scope:
        fig.text(0.5, 0.943, str(scope), ha="center", va="top",
                 fontsize=8.5, color="#706E68")


# Below this FDR the exact number stops meaning anything - the rule is simply certain -
# so the axis stops here and the surer rules sit on the top edge instead.
_CEILING = -np.log10(1e-6)

# FDR values worth a tick. A short axis can afford the in-between values; a long one
# only has room for the decades, or the labels pile up on each other.
_TICKS_NEAR = [1, 0.5, 0.2, 0.1, 0.05, 0.01, 1e-3]
_TICKS_FAR = [1, 0.1, 0.05, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6]


def _fdr_ticks(ax, top):
    """Label the y axis with FDR values instead of -log10 of them."""
    wanted = _TICKS_NEAR if top <= 3.4 else _TICKS_FAR
    shown = [v for v in wanted if -np.log10(v) <= top]
    ax.set_yticks([-np.log10(v) for v in shown])
    ax.set_yticklabels([f"{v:g}" for v in shown])


def _place_names(ax, spots, top):
    """One name per row above the dots, each with a thin leader down to its dot.

    The surest rules all land on the same line - the permutation test cannot go below
    1 / (shuffles + 1) - and rule names are long, so there is no room sideways. A fixed
    ladder of rows cannot overlap, however crowded the plot gets.
    """
    if not spots:
        return
    floor = max(y for _, y, _ in spots) + 0.10 * top
    rows = np.linspace(0.97 * top, floor, len(spots))
    left, right = ax.get_xlim()
    for (x, y, text), row in zip(sorted(spots), rows):
        # Names near an edge lean inwards, so a long one never runs off the plot.
        side = "left" if x < 0.5 * left else "right" if x > 0.5 * right else "center"
        ax.annotate(text, xy=(x, y), xytext=(x, row), fontsize=7, color=_INK,
                    ha=side, va="center", linespacing=1.35, zorder=5,
                    arrowprops=dict(arrowstyle="-", color="0.6", lw=0.6,
                                    shrinkA=3, shrinkB=4))


def passing(comparison, fdr_threshold=0.05, min_gap=None):
    """True for the rules a plot should colour in: sure enough, and big enough.

    `min_gap` is in percentage points, and is a judgement call - there is no number
    the statistics can hand you. Left out, only the FDR cutoff applies.
    """
    ok = comparison["fdr"] <= fdr_threshold
    if min_gap is not None:
        ok &= comparison["effect_size"].abs() * 100 >= min_gap
    return ok


def plot_volcano(comparison, label_a, label_b, scope=None, fdr_threshold=0.05,
                 min_gap=None, name_top=5, save=None):
    """Every tested rule at once: how big the gap is, and how sure we are.

    Right of zero = more common in `label_a`, left = more common in `label_b`.
    Coloured dots passed the cutoffs; grey ones did not.
    """
    if comparison.empty:
        print("No data to plot.")
        return

    gap = comparison["effect_size"].to_numpy(dtype=float) * 100
    fdr = comparison["fdr"].to_numpy(dtype=float)
    sure = -np.log10(np.clip(fdr, 1e-300, None))
    passed = passing(comparison, fdr_threshold, min_gap).to_numpy()
    cutoff = -np.log10(fdr_threshold)

    # A single rule can be surer than the rest by ten orders of magnitude, which would
    # squash every other dot onto the bottom line. Past the ceiling they are all just
    # 'certain', so those ride the top edge as triangles and the axis keeps the range
    # people actually read.
    ceiling = _CEILING if float(sure.max()) > _CEILING else float(sure.max())
    over = sure > ceiling
    height = np.minimum(sure, ceiling)

    color_a, color_b = _group_colors(label_a, label_b)
    colors = np.array([color_a if (p and g > 0) else color_b if (p and g < 0)
                       else to_rgba(_NEUTRAL) for p, g in zip(passed, gap)])

    fig, ax = plt.subplots(figsize=(9.5, 5.6))
    ax.axvline(0, color=_ZERO, lw=1.4, zorder=1)
    ax.axhline(cutoff, color=_HAIRLINE, lw=1.0, zorder=1)
    if min_gap is not None:                    # the 'big enough to care' line, both ways
        for side in (-min_gap, min_gap):
            ax.axvline(side, color=_HAIRLINE, lw=1.0, ls=(0, (5, 4)), zorder=1)
    ax.scatter(gap[~over], height[~over], s=54, c=colors[~over], alpha=0.9,
               edgecolor="white", linewidth=0.8, zorder=3)
    if over.any():                             # surer than the axis goes
        ax.scatter(gap[over], height[over], s=78, marker="^", c=colors[over], alpha=0.9,
                   edgecolor="white", linewidth=0.8, zorder=4, clip_on=False)

    # Both sides of zero get the same reach, so a gap one way looks like a gap the
    # other way. The limits are set before the names are placed - the names are
    # pushed apart inside these limits, so changing them afterwards would undo it.
    limit = float(np.abs(gap).max()) * 1.15 or 1.0
    ax.set_xlim(-limit, limit)
    # Always tall enough to show the cutoff line, even when nothing came close to it,
    # and with a margin below zero so a dot sitting at 0 is not cut in half by the axis.
    # The extra room on top is for the names: many rules can land on the same FDR - the
    # permutation test cannot go below 1 / (shuffles + 1) - so their labels need a band
    # above the dots to spread into.
    top = max(ceiling, cutoff) * 1.4
    ax.set_ylim(-0.04 * top, top)
    _fdr_ticks(ax, top)

    # Name only the few strongest, and only among the ones that passed. Two rules with
    # the same numbers - a pair like 'A -> B' and 'B -> A' that fires in the same FOVs -
    # sit on one dot, so they share one label instead of two arrows to the same place.
    named = (pd.Series(np.abs(gap), index=comparison.index).where(passed)
             .dropna().sort_values(ascending=False).head(name_top).index)
    flips = _flip_marks(comparison, label_a, label_b)
    together = {}
    for rule in named:
        i = comparison.index.get_loc(rule)
        together.setdefault((round(gap[i], 6), round(height[i], 6)), []).append(
            f"{flips[i]}{rule}")

    _place_names(ax, [(gx, gy, "\n".join(names)) for (gx, gy), names in together.items()],
                 top)

    ax.annotate(f"cutoff {fdr_threshold:g}", xy=(-limit, cutoff), xytext=(3, 3),
                textcoords="offset points", fontsize=8, color=_INK)
    if min_gap is not None:
        ax.annotate(f"{min_gap:g} pp", xy=(min_gap, ax.get_ylim()[1]), xytext=(3, -10),
                    textcoords="offset points", fontsize=8, color=_INK)
    # Which side means what, under the axis - never on the plot, where a dot may sit.
    ax.set_xlabel(f"gap in net attraction (% of FOVs)\n"
                  f"←  more in {label_b}          more in {label_a}  →", fontsize=10)
    ax.set_ylabel("FDR  (lower = surer)", fontsize=10)
    _title(ax, f"{label_a} vs {label_b}: which rules differ", scope)
    _legend(ax, label_a, label_b, color_a, color_b,
            over_label=f"surer than {10 ** -ceiling:g}" if over.any() else None)
    tidy_axes(ax, grid="both")
    fig.tight_layout(rect=(0, 0, 1, 0.88))
    _finish(fig, save)


def plot_dumbbell(comparison, label_a, label_b, scope=None, fdr_threshold=0.05,
                  min_gap=None, top_n=15, value_label=None,
                  what="how often a rule fires", save=None):
    """The rules that differ most, by name: where each group sits between avoid and attract.

    Any table `compare_groups` returns can be drawn here, not only rules - give
    `value_label` and `what` to say what is being measured (e.g. cell-type shares),
    so a reader can tell the two kinds of plot apart at a glance.

    One row per rule, a dot per group, the line between them is the gap, and the gap
    written again on the right. A dot left of zero means the cell types mostly avoid each
    other in that group, right of zero they mostly attract. Biggest gap on top. Rules that
    did not pass the cutoffs are still drawn, hollow and faint, so a comparison that found
    nothing still shows how close it came.
    """
    if comparison.empty:
        print("No data to plot.")
        return

    # Always the widest gaps, whether or not they passed - a comparison where nothing
    # passed is still worth looking at, to see how far off it was.
    ok = passing(comparison, fdr_threshold, min_gap)
    top = (comparison.reindex(comparison["effect_size"].abs().sort_values().index)
           .tail(top_n))                                   # biggest gap ends up on top
    won = ok.reindex(top.index).to_numpy()
    a = top[f"net_{label_a}"].to_numpy(dtype=float) * 100
    b = top[f"net_{label_b}"].to_numpy(dtype=float) * 100
    y = np.arange(len(top))

    color_a, color_b = _group_colors(label_a, label_b)
    # Wide enough that the rule names on the left still leave the title and the
    # legend room to sit side by side.
    fig, ax = plt.subplots(figsize=(10, max(3.2, len(top) * 0.34 + 1.4)))

    # A solid bar is a rule that passed; a faint one did not, so the two never read
    # the same even though both are drawn.
    lo, hi = np.minimum(a, b), np.maximum(a, b)
    ax.hlines(y[won], lo[won], hi[won], color="#c8c7c0", lw=2.6, zorder=1)
    ax.hlines(y[~won], lo[~won], hi[~won], color="#ebeae4", lw=2.6, zorder=1)
    for vals, color in ((b, color_b), (a, color_a)):
        ax.scatter(vals[won], y[won], s=64, color=color,
                   edgecolor="white", linewidth=1.4, zorder=3)
        ax.scatter(vals[~won], y[~won], s=52, facecolors="white", edgecolor=color,
                   linewidth=1.6, alpha=0.85, zorder=3)

    # '±' marks a rule that changed direction: attracting in one group, avoiding in
    # the other - the two dots then sit on either side of the zero line.
    names = [f"{m}{n}" for m, n in zip(_flip_marks(top, label_a, label_b),
                                       top.index.astype(str))]

    ax.set_yticks(y)
    ax.set_yticklabels(names, fontsize=8.5)
    for tick, is_won in zip(ax.get_yticklabels(), won):
        tick.set_color("#0b0b0b" if is_won else "#96958f")
    ax.set_ylim(-0.7, len(top) - 0.3)

    # Zero is a real place now: left of it the cell types avoid each other, right of it
    # they attract, so the axis has to show both sides.
    ax.axvline(0, color=_ZERO, lw=1.4, zorder=1)

    # The gap itself, written down the right - the number MIN_GAP is compared against.
    reach = max(float(np.abs(np.concatenate([a, b])).max()), 1.0)
    ax.set_xlim(-reach * 1.12, reach * 1.45)
    for yi, (g, is_won) in enumerate(zip(top["effect_size"] * 100, won)):
        ax.annotate(f"{abs(g):.0f}", xy=(ax.get_xlim()[1], yi), xytext=(-4, 0),
                    textcoords="offset points", ha="right", va="center", fontsize=8,
                    color="#0b0b0b" if is_won else "#96958f")
    ax.annotate("gap (pp)", xy=(ax.get_xlim()[1], len(top) - 0.35), xytext=(-4, 0),
                textcoords="offset points", ha="right", va="center", fontsize=8,
                color=_INK)
    if min_gap is not None:
        ax.annotate(f"cutoff {min_gap:g}", xy=(ax.get_xlim()[1], -0.55), xytext=(-4, 0),
                    textcoords="offset points", ha="right", va="center", fontsize=8,
                    color=_INK)
    ax.set_xlabel(value_label or "net: attracts minus avoids (% of the group's FOVs)\n"
                                 "←  avoid          attract  →", fontsize=10)
    _title(ax, f"{label_a} vs {label_b}: biggest gaps in {what}", scope)
    _legend(ax, label_a, label_b, color_a, color_b, show_hollow=bool((~won).any()))
    tidy_axes(ax, grid="x", hide=("top", "right", "left"))
    ax.tick_params(axis="y", length=0)
    fig.tight_layout(rect=(0, 0, 1, 0.88))
    _finish(fig, save)


# One hue per rule, so a line can be followed across the stages and found again in the
# names. Kept to eight - past that neighbouring hues stop being told apart - and ordered
# so the ones drawn next to each other are the furthest apart in colour.
_LINE_COLORS = ["#2a78d6", "#eb6834", "#199e70", "#c98500",
                "#d55181", "#4a3aa7", "#008300", "#e34948"]


def _cell_types_in(rules):
    """Every cell type named in these rules, in the order they first come up."""
    seen = []
    for rule in rules:
        for part in str(rule).replace("->", ",").split(","):
            name = part.strip()
            if name and name not in seen:
                seen.append(name)
    return seen


def _trend_panel(ax, nets, names, slopes, steady, title, stages, unit="", colors=None):
    """One direction's rules: a line per rule, named outside on the left with an arrow in."""
    ax.set_title(title, fontsize=10.5, color=_INK, loc="left", pad=14)
    if not len(nets):
        ax.annotate("none", xy=(0.5, 0.5), xycoords="axes fraction", ha="center",
                    va="center", fontsize=9, color="#96958f")
        ax.set_xticks([])
        ax.set_yticks([])
        for side in ax.spines.values():
            side.set_visible(False)
        ax.grid(False)
        return

    # The unit rides under the title instead of on a rotated y label, which would have to
    # share the left margin with the names.
    notes = [unit] + (["dashed: went back at one step"] if not steady.all() else [])
    ax.annotate("      ".join(n for n in notes if n), xy=(0, 1.012),
                xycoords="axes fraction", fontsize=8, color="#898781")

    # Only this panel's own lines set the scale - an empty half-panel just to match the
    # other side helps nobody.
    low, high = float(nets.min()), float(max(nets.max(), 0.0))
    pad = max((high - low) * 0.10, 2.0)
    ax.set_ylim(low - pad, high + pad)

    x = np.arange(len(stages))
    ax.set_xlim(-0.08, len(stages) - 0.92)
    rows = spread_labels(nets[:, 0], (high - low + 2 * pad) * 0.075, low - pad * 0.6, high + pad * 0.6)

    for i, (row, name, slope, is_steady, at) in enumerate(
            zip(nets, names, slopes, steady, rows)):
        color = (colors or {}).get(name) or _LINE_COLORS[i % len(_LINE_COLORS)]
        # Dashed means the rule went back at one step: the drift is real, but not gradual.
        ax.plot(x, row, color=color, lw=1.8, alpha=0.9, solid_capstyle="round", zorder=3,
                linestyle="-" if is_steady else (0, (4.5, 2.5)))
        ax.scatter(x, row, s=24, color=color, edgecolor="white", linewidth=1.1, zorder=4)
        # A slope that rounds to nothing is written '0', never '-0'.
        moved = f"{slope:+.0f}" if abs(slope) >= 0.5 else "0"
        # Outside the panel, right-aligned so the names end in a straight column, with an
        # arrow in to the line's own starting point - the only way to tell crowded starts apart.
        ax.annotate(f"{name}   {moved}", xy=(x[0], row[0]), xytext=(-0.15, at),
                    textcoords=("axes fraction", "data"), annotation_clip=False,
                    fontsize=8, va="center", ha="right", color=color, zorder=5,
                    arrowprops=dict(arrowstyle="->", color=color, alpha=0.5, lw=0.8,
                                    shrinkA=4, shrinkB=4))

    ax.set_xticks(x, stages, fontsize=10)


def _movers(trend, stages, fdr_threshold, top_n, rising=None):
    """The rows a panel draws: the steepest few that passed, optionally one direction only."""
    nets = trend[[f"net_{stage}" for stage in stages]].to_numpy(dtype=float) * 100
    slope = trend["slope"].to_numpy(dtype=float) * 100
    wanted = (trend["fdr"] <= fdr_threshold).to_numpy(copy=True)
    if rising is not None:
        wanted &= slope > 0 if rising else slope < 0

    order = np.argsort(-np.abs(np.where(wanted, slope, 0)))[:int(wanted.sum())][:top_n]
    return (nets[order], trend.index[order].astype(str), slope[order],
            trend["steady"].to_numpy()[order], int(wanted.sum()))


def _plain(ax):
    """The chrome every trend panel shares: a zero line, hairline grid, no frame."""
    ax.axhline(0, color=_ZERO, lw=1.5, zorder=2)
    tidy_axes(ax, grid="y", hide=("top", "right", "bottom", "left"))
    ax.tick_params(length=0, labelsize=9, colors="#898781")


_NET_LABEL = "net: attracts minus avoids (% of the stage's FOVs)"
_SHARE_LABEL = "share of the cells (%)"


def _room_for_names(fig, top=0.845):
    """Margins the names can live in: they sit outside the panels, where tight_layout
    cannot see them, so the space is set by hand."""
    fig.subplots_adjust(left=0.205, right=0.99, top=top, bottom=0.09, wspace=0.63)


def plot_rules_with_cells(trend, cell_trend, stages, rising=True, scope=None,
                          fdr_threshold=0.05, top_n=8, cell_colors=None, save=None):
    """One direction's rules beside the cell types those rules are built from.

    Same three stages on both sides, each panel on its own scale - a rule's net is a
    share of FOVs and a cell type's share is a share of cells, so one axis for both would
    invent a relationship. Read across: if the rules move and their cell types do not,
    the change is in how the cells are arranged, not in how many of them there are.
    """
    nets, names, slopes, steady, found = _movers(trend, stages, fdr_threshold, top_n, rising)
    way = "Climbing" if rising else "Fading"
    if not len(nets):
        print(f"No {way.lower()} rules to plot.")
        return

    # Only the cell types those rules are made of, steepest first so the busiest ones
    # are the ones that get a colour of their own.
    types = [t for t in _cell_types_in(names) if t in cell_trend.index]
    shown = cell_trend.loc[types].reindex(cell_trend.loc[types]["slope"].abs()
                                          .sort_values(ascending=False).index)
    if cell_colors is None:
        shown = shown.head(len(_LINE_COLORS))

    fig, axes = plt.subplots(1, 2, figsize=(14.5, 5.4))
    _trend_panel(axes[0], nets, names, slopes, steady,
                 f"the rules   ({found} {way.lower()})", stages, unit=_NET_LABEL)
    _trend_panel(axes[1], shown[[f"net_{s}" for s in stages]].to_numpy(dtype=float) * 100,
                 shown.index.astype(str), shown["slope"].to_numpy(dtype=float) * 100,
                 shown["steady"].to_numpy(), "how common those cell types are", stages,
                 unit=_SHARE_LABEL, colors=cell_colors)
    for ax in axes:
        _plain(ax)

    fig.suptitle(f"{way} rules and the cells behind them (top {top_n})",
                 fontsize=13, y=0.985)
    if scope:
        fig.text(0.5, 0.943, str(scope), ha="center", va="top",
                 fontsize=8.5, color="#706E68")
    _room_for_names(fig)
    _finish(fig, save)


def plot_trend(trend, stages, scope=None, fdr_threshold=0.05, top_n=8,
               what="Rules", value_label=None, save=None):
    """Rules whose net climbs or fades as the disease worsens, one line per rule.

    One point per stage, so the line itself is the trend the test measured, and the
    number beside each name is its slope. Climbing and fading are drawn apart,
    otherwise the lines cross and neither is readable. Any table `compare_stages`
    returns can be drawn - give `what` and `value_label` for cell types instead of rules.
    """
    thing = what.rstrip("s").lower()             # 'Cell types' -> 'cell type'

    movers = {
        rising: _movers(trend, stages, fdr_threshold, top_n, rising)
        for rising in (True, False)
    }
    # Nothing passing is a result worth stating, but it does not need a full-height
    # panel of empty axes to state it.
    height = 5.4 if any(len(m[0]) for m in movers.values()) else 1.7

    # Each panel keeps its own scale: the two directions rarely cover the same range, and
    # a shared one leaves whichever panel is smaller as mostly empty space.
    fig, axes = plt.subplots(1, 2, figsize=(14.5, height))
    for ax, rising, title in ((axes[0], True, "climbing with severity"),
                              (axes[1], False, "fading with severity")):
        nets, names, slopes, steady, found = movers[rising]
        _trend_panel(ax, nets, names, slopes, steady,
                     f"{title}   ({found} {thing}{'s' if found != 1 else ''})", stages,
                     unit=value_label or _NET_LABEL)
        _plain(ax)

    title_y = 0.985 if height > 3 else 0.93
    fig.suptitle(f"{what} that move with severity (top {top_n} each way)",
                 fontsize=13, y=title_y)
    if scope:
        fig.text(0.5, title_y - 0.042 * (5.4 / height), str(scope), ha="center",
                 va="top", fontsize=8.5, color="#706E68")
    _room_for_names(fig, top=0.845 if height > 3 else 0.52)
    _finish(fig, save)


def plot_state_breakdown(states, selected, fovs_by_stage, stages, scope=None, save=None):
    """For selected rules, split each stage into attract, avoid, and no-rule FOVs."""
    if selected.empty:
        print("No eligible trends passed the cutoff.")
        return

    colors = {"Attract": "#2a78d6", "No rule": "#dddcd5", "Avoid": "#eb6834"}
    ncols = 2 if len(selected) > 1 else 1
    nrows = int(np.ceil(len(selected) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(11, 2.65 * nrows + 1),
                             squeeze=False)

    for ax, rule in zip(axes.flat, selected.index):
        shares, totals = [], []
        for stage in stages:
            values = states.reindex(columns=fovs_by_stage[stage]).loc[rule].dropna()
            totals.append(len(values))
            shares.append([100 * (values == value).mean() for value in (1, 0, -1)])

        shares = np.asarray(shares)
        bottom = np.zeros(len(stages))
        for column, label in enumerate(("Attract", "No rule", "Avoid")):
            bars = ax.bar(np.arange(len(stages)), shares[:, column], bottom=bottom,
                          color=colors[label], width=0.68, label=label)
            for bar, percent in zip(bars, shares[:, column]):
                if percent >= 10:
                    ax.text(bar.get_x() + bar.get_width() / 2,
                            bar.get_y() + bar.get_height() / 2, f"{percent:.0f}%",
                            ha="center", va="center", fontsize=8,
                            color="white" if label != "No rule" else _INK)
            bottom += shares[:, column]

        slope = selected.loc[rule, "slope"] * 100
        arrow = "increasing" if slope > 0 else "decreasing"
        ax.set_title(f"{rule}\n{arrow} {abs(slope):.0f} pp/stage · "
                     f"FDR {selected.loc[rule, 'fdr']:.3g}", fontsize=9.5, loc="left")
        ax.set_xticks(np.arange(len(stages)),
                      [f"{stage}\nn={total}" for stage, total in zip(stages, totals)])
        ax.set_ylim(0, 100)
        ax.set_ylabel("eligible FOVs (%)", fontsize=8)
        tidy_axes(ax, grid="y", hide=("top", "right", "left"))
        ax.tick_params(length=0, labelsize=8)

    for ax in axes.flat[len(selected):]:
        ax.set_visible(False)

    handles = [Line2D([0], [0], marker="s", linestyle="none", markersize=8,
                      markerfacecolor=colors[label], markeredgecolor="none", label=label)
               for label in ("Attract", "No rule", "Avoid")]
    fig.legend(handles=handles, loc="upper center", ncol=3, frameon=False,
               bbox_to_anchor=(0.5, 0.89))
    fig.suptitle("What changes inside the strongest eligible trends",
                 fontsize=13, y=0.99)
    if scope:
        fig.text(0.5, 0.947, str(scope), ha="center", va="top",
                 fontsize=8.5, color="#706E68")
    fig.subplots_adjust(top=0.78, hspace=0.72, wspace=0.32)
    _finish(fig, save)


# ---------------------------------------------------------------------------
# Focused evidence views used by the paper-alignment and time notebooks
# ---------------------------------------------------------------------------

_STATE_COLORS = {
    "Attraction": "#2878D0",
    "No rule": "#E8E7E2",
    "Avoidance": "#E66A4E",
    "Insufficient cells": "#B8B4C7",
}

_STAGE_COLORS = {"Control": "#67B58A", "Mild": "#E7A33E", "Severe": "#D85D62"}
_STATE_ORDER = ["Attraction", "No rule", "Avoidance", "Insufficient cells"]


def _rule_test_text(spec, trend_results=None, pair_results=None, test_label=None):
    """Compact corrected-test summary for one rule panel."""
    rule, organ, group_col = spec["rule"], spec["organ"], spec["score"]
    scope_key = (organ, group_col)
    trend = (trend_results or {}).get(scope_key, pd.DataFrame())
    pair = (pair_results or {}).get(scope_key, pd.DataFrame())
    tests = []
    prefix = f"{test_label} " if test_label else ""
    if rule in trend.index:
        tests.append(f"ordered {prefix}FDR {trend.at[rule, 'fdr']:.3g}")
    if isinstance(pair, dict):
        labels = {
            ("Control", "Mild"): "C–M",
            ("Mild", "Severe"): "M–S",
            ("Control", "Severe"): "C–S",
        }
        for comparison, result in pair.items():
            if rule in result.index:
                label = labels.get(comparison, "–".join(comparison))
                tests.append(f"{label} {prefix}FDR {result.at[rule, 'fdr']:.3g}")
    elif rule in pair.index:
        tests.append(f"C–S {prefix}FDR {pair.at[rule, 'fdr']:.3g}")
    return " · ".join(tests) if tests else spec.get("test_text", "")


def _stage_state_counts(states, eligibility, metadata, rule, organ, group_col,
                        stages, unit_col="FOV"):
    """Counts and net scores for one rule in each displayed stage."""
    stage_counts, eligible_counts, nets = [], [], []
    for stage in stages:
        mask = (metadata["Organ"] == organ) & (metadata[group_col] == stage)
        units = metadata.loc[mask, unit_col].drop_duplicates()
        state = states.reindex(columns=units).loc[rule]
        can_test = eligibility.reindex(columns=units).loc[rule]
        eligible_n = int(can_test.sum())
        counts = {
            "Attraction": int((can_test & (state > 0)).sum()),
            "No rule": int((can_test & (state == 0)).sum()),
            "Avoidance": int((can_test & (state < 0)).sum()),
            "Insufficient cells": int((~can_test).sum()),
        }
        stage_counts.append(counts)
        eligible_counts.append((eligible_n, len(units)))
        nets.append(
            100 * (counts["Attraction"] - counts["Avoidance"]) / eligible_n
            if eligible_n else np.nan
        )
    return stage_counts, eligible_counts, nets


def _draw_state_bars(ax, stage_counts, eligible_counts, nets, stages,
                     unit_label="FOVs", label_threshold=8, compact=False, show_net=True):
    """Draw the shared all-unit rule-state bars."""
    x = np.arange(len(stages))
    bottom = np.zeros(len(stages))
    ax.set_ylim(0, 114)          # set first: the labels below size themselves against it
    for label in _STATE_ORDER:
        percentages = np.array([
            100 * counts[label] / total if total else 0
            for counts, (_, total) in zip(stage_counts, eligible_counts)
        ])
        bars = ax.bar(
            x, percentages, bottom=bottom, width=0.64,
            color=_STATE_COLORS[label], edgecolor="white", linewidth=0.8,
        )
        # A band only gets the count over the percentage when it is tall enough to
        # hold two lines; a middling band gets them side by side, a thin one nothing.
        axes_points = ax.get_window_extent().height * 72 / ax.figure.dpi
        span = np.diff(ax.get_ylim())[0] or 114
        for bar, percent, counts in zip(bars, percentages, stage_counts):
            band_points = percent / span * axes_points
            if percent < label_threshold or band_points < 9:
                continue
            ink = _INK if label == "No rule" else "white"
            two_lines = band_points >= 20
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_y() + bar.get_height() / 2,
                f"{counts[label]}\n{percent:.0f}%" if two_lines
                else f"{counts[label]} · {percent:.0f}%",
                ha="center", va="center", fontsize=8, color=ink,
            )
        bottom += percentages

    for position, net in enumerate(nets):
        if not show_net:
            continue
        prefix = "" if compact else "net "
        text = f"{prefix}n/a" if np.isnan(net) else f"{prefix}{net:+.0f}%"
        ax.text(position, 103, text, ha="center", va="bottom",
                fontsize=7 if compact else 8.5,
                color="#454440", fontweight="bold")
    ax.set_xticks(x, [
        f"{stage}\n{eligible_n}/{total}" if compact
        else f"{stage}\n{eligible_n}/{total} eligible"
        for stage, (eligible_n, total) in zip(stages, eligible_counts)
    ])
    ax.set_ylabel(f"all {unit_label} (%)")
    tidy_axes(ax, grid="y", hide=("top", "right", "left", "bottom"))
    ax.tick_params(length=0)


def plot_rule_states(states, eligibility, metadata, specs, stages, heading,
                     trend_results=None, pair_results=None, unit_col="FOV",
                     unit_label="FOVs", test_label=None, save=None):
    """Show attraction, no rule, avoidance and insufficient cells for every unit.

    Each item in ``specs`` names a rule, organ and grouping column. The full bar uses
    every unit in that group, so loss of a cell type remains visible instead of being
    hidden by the eligibility filter.
    """
    n_panels = len(specs)
    figure_height = 3.7 * n_panels + 1.8
    fig, axes = plt.subplots(
        n_panels, 1, figsize=(11.8, figure_height), squeeze=False,
    )
    axes = axes[:, 0]
    threshold = eligibility.attrs.get("min_cells")
    eligibility_text = (
        f"eligibility ≥{threshold} cells/type" if threshold is not None
        else "eligibility-controlled"
    )
    groupings = list(dict.fromkeys(
        spec["score"].replace(" score", "").lower() for spec in specs
    ))
    grouping_scope = (
        f"{groupings[0]} score" if len(groupings) == 1
        else " + ".join(f"{grouping} score" for grouping in groupings)
    )
    figure_subtitle = (
        f"Analysis: pairwise rule states · Unit: FOV · Score: {grouping_scope} · "
        f"Stages: {' / '.join(stages)} · Eligibility: {eligibility_text.replace('eligibility ', '')} · "
        "Bars: all FOVs"
    )
    if any(isinstance(value, dict) for value in (pair_results or {}).values()):
        figure_subtitle += (
            "\nFDR correction: all rules and C–M / M–S / C–S comparisons"
        )

    for ax, spec in zip(axes, specs):
        rule, organ, group_col = spec["rule"], spec["organ"], spec["score"]
        summary = _stage_state_counts(
            states, eligibility, metadata, rule, organ, group_col, stages, unit_col,
        )
        _draw_state_bars(ax, *summary, stages, unit_label)

        panel = spec.get("panel", rule.replace(" -> ", " → "))
        tests = _rule_test_text(spec, trend_results, pair_results, test_label)
        detail = group_col.lower()
        if tests:
            detail += f" · {tests}"
        ax.set_title(panel, fontsize=10.8, pad=31, loc="center")
        ax.text(0.5, 1.02, detail, transform=ax.transAxes, ha="center",
                va="bottom", fontsize=8.6, color="#66645F")

    handles = [
        Line2D([0], [0], marker="s", linestyle="none", markersize=9,
               markerfacecolor=_STATE_COLORS[label], markeredgecolor="none", label=label)
        for label in _STATE_ORDER
    ]
    # Keep the header a fixed physical size. Fixed fractional positions create
    # several inches of empty space when a figure contains many rule panels.
    title_y = 1 - 0.15 / figure_height
    subtitle_y = 1 - 0.63 / figure_height
    legend_y = 1 - 1.05 / figure_height
    axes_top = 1 - 1.55 / figure_height
    fig.legend(handles=handles, ncol=4, frameon=False, loc="upper center",
               bbox_to_anchor=(0.5, legend_y))
    fig.suptitle(heading, fontsize=14.5, x=0.5, ha="center", y=title_y)
    fig.text(0.5, subtitle_y, figure_subtitle, ha="center", va="top", fontsize=8.6,
             color="#66645F", linespacing=1.5)
    fig.subplots_adjust(left=0.12, right=0.985, top=axes_top, bottom=0.07, hspace=0.9)
    _finish(fig, save)


def _metric_panel(ax, metrics, eligibility, metadata, spec, stages, metric="Lift",
                  reference=1, unit_col="FOV", state=None, connect=True):
    """One axes of raw metric values, drawn only where an eligible FOV has the rule."""
    rule, organ, group_col = spec["rule"], spec["organ"], spec["score"]
    columns = [unit_col, metric] + (["state"] if state is not None else [])
    rows = metrics.loc[metrics["Clean_Rule"] == rule, columns].copy()
    if state is not None:
        rows = rows[rows["state"] == state]
    values_by_fov = rows.dropna(subset=[metric]).groupby(unit_col)[metric].mean()
    plot_rows, labels, medians = [], [], []

    for position, stage in enumerate(stages):
        fovs = metadata.loc[
            (metadata["Organ"] == organ) & (metadata[group_col] == stage), unit_col
        ].drop_duplicates()
        can_test = eligibility.reindex(columns=fovs).loc[rule]
        eligible_fovs = can_test.index[can_test]
        values = values_by_fov.reindex(eligible_fovs).dropna()
        plot_rows.extend(
            {"stage": stage, "position": position, metric: value}
            for value in values
        )
        labels.append(f"{stage}\n{len(values)}/{len(eligible_fovs)}")
        medians.append(values.median() if len(values) else np.nan)

    data = pd.DataFrame(plot_rows)
    if not data.empty:
        colors = {stage: _STAGE_COLORS.get(stage, "#777570") for stage in stages}
        sns.stripplot(
            data=data, x="stage", y=metric, hue="stage", legend=False,
            order=stages, palette=colors,
            jitter=0.18, size=5, alpha=0.72, edgecolor="white", linewidth=0.6, ax=ax,
        )
        if connect:
            ax.plot(range(len(stages)), medians, color="#777570", lw=1.5, zorder=4)
        ax.scatter(range(len(stages)), medians, marker="D", s=58,
                   color=[colors[stage] for stage in stages],
                   edgecolor="white", linewidth=0.8, zorder=5)
    if reference is not None:
        ax.axhline(reference, color=_ZERO, lw=1.1)
    ax.set_xticks(range(len(stages)), labels)
    ax.set_xlabel("")
    ax.set_xlim(-0.5, len(stages) - 0.5)
    direction = {1: "attraction", -1: "avoidance"}.get(state)
    ax.set_ylabel(f"{direction.title() + ' ' if direction else ''}{metric} where the rule fires")
    tidy_axes(ax, grid="y", hide=("top", "right"))
    ax.tick_params(length=0)


def plot_rule_metric(metrics, eligibility, metadata, spec, stages, metric="Lift",
                     reference=1, heading=None, unit_col="FOV", state=None,
                     trend_results=None, pair_results=None, save=None):
    """The metric panel on its own, titled and captioned."""
    rule, organ, group_col = spec["rule"], spec["organ"], spec["score"]
    fig, ax = plt.subplots(figsize=(8.8, 4.4))
    _metric_panel(ax, metrics, eligibility, metadata, spec, stages, metric,
                  reference, unit_col, state)
    direction = {1: "attraction", -1: "avoidance"}.get(state)
    label = f"{direction.title()} {metric}" if direction else metric
    tests = _rule_test_text(
        spec, trend_results, pair_results,
        f"{direction} {metric}" if direction else metric,
    )
    threshold = eligibility.attrs.get("min_cells")
    detail = (
        f"Analysis: pairwise rules · Unit: FOV · Organ: {organ} · "
        f"Score: {group_col.replace(' score', '').lower()} · Stages: {' / '.join(stages)} · "
        f"Eligibility: {f'≥{threshold} cells/type' if threshold else 'controlled'} · "
        f"State: {direction or 'either'} · Metric: {metric}"
        "\nDots: rule-bearing FOVs · Diamonds: stage medians"
    )
    if tests:
        detail += f"\nTests: {tests}"
    fig.suptitle(heading or f"{rule.replace(' -> ', ' → ')} — {label}",
                 fontsize=13.5, y=0.99)
    fig.text(0.5, 0.935, detail, ha="center", va="top", fontsize=8.3,
             color="#706E68", linespacing=1.35)
    fig.subplots_adjust(top=0.72, bottom=0.19, left=0.12, right=0.98)
    _finish(fig, save)


def plot_cell_counts(cells, metadata, cell_type, organ, group_col, stages,
                     threshold, heading, save=None, stage_colors=None):
    """Per-unit cell counts with the rule-eligibility threshold shown explicitly."""
    counts = cells.loc[cells["cell type"] == cell_type].groupby("fov").size()
    data = metadata.loc[metadata["Organ"] == organ, ["FOV", group_col]].copy()
    data["count"] = data["FOV"].map(counts).fillna(0)
    data = data[data[group_col].isin(stages)]
    colors = stage_colors or _STAGE_COLORS

    fig, ax = plt.subplots(figsize=(9.8, 4.9))
    sns.boxplot(data=data, x=group_col, y="count", order=stages, color="#F4F2EC",
                width=0.48, fliersize=0, linewidth=1.1, ax=ax)
    sns.stripplot(data=data, x=group_col, y="count", order=stages,
                  palette=colors, size=5.2, alpha=0.72, edgecolor="white",
                  linewidth=0.7, jitter=0.22, ax=ax)
    ax.axhline(threshold, color="#665C9A", lw=1.5, linestyle=(0, (5, 3)))
    ax.annotate(f"eligibility threshold = {threshold}", xy=(0.01, threshold),
                xycoords=("axes fraction", "data"), xytext=(0, 6),
                textcoords="offset points", ha="left", va="bottom", fontsize=9,
                color="#574F83")

    upper = max(float(data["count"].max()), threshold) * 1.16 + 1
    for position, stage in enumerate(stages):
        values = data.loc[data[group_col] == stage, "count"]
        enough = int((values >= threshold).sum())
        ax.text(position, upper * 0.97, f"{enough}/{len(values)} reach threshold",
                ha="center", va="top", fontsize=8.5, fontweight="bold")

    ax.set_ylim(0, upper)
    ax.set_xlabel("")
    ax.set_ylabel(f"{cell_type} cells per FOV")
    ax.set_title(heading, fontsize=14, pad=31)
    ax.text(0.5, 1.02,
            (f"Unit: FOV · Organ: {organ} · Score: {group_col.replace(' score', '').lower()} · "
             f"Stages: {' / '.join(stages)} · Cell type: {cell_type} · "
             f"Eligibility threshold: ≥{threshold} cells"),
            transform=ax.transAxes, ha="center", va="bottom", fontsize=8.6,
            color="#66645F")
    tidy_axes(ax, grid="y", hide=("top", "right"))
    _finish(fig, save)


def plot_time_coverage(metadata, day_col, group_col, groups, save=None):
    """One dot per biopsy, showing where the temporal evidence is concentrated."""
    data = metadata.dropna(subset=[day_col]).drop_duplicates("Biopsy").copy()
    colors = {"<30": "#58A6A6", "30-100": "#E3A33D", ">100": "#806FB3"}
    fig, ax = plt.subplots(figsize=(10.5, 3.5))
    organs = [organ for organ in ("Colon", "Duodenum") if organ in set(data["Organ"])]
    y_by_organ = {organ: i for i, organ in enumerate(organs)}
    rng = np.random.default_rng(4)
    for group in groups:
        part = data[data[group_col] == group]
        y = np.array([y_by_organ[o] for o in part["Organ"]], float)
        y += rng.uniform(-0.09, 0.09, len(part))
        ax.scatter(part[day_col], y, s=54, color=colors.get(group, _NEUTRAL),
                   edgecolor="white", linewidth=0.8, alpha=0.85, label=group)
    ax.set_yticks(range(len(organs)), organs)
    ax.set_xlabel("days after transplantation")
    ax.set_ylabel("")
    fig.suptitle("Biopsy coverage across time", fontsize=13, y=0.99)
    fig.text(
        0.5, 0.935,
        f"Unit: biopsy · Time variable: {day_col.lower()} · Groups: {' / '.join(groups)} · Color: time group",
        ha="center", va="top", fontsize=8.5, color="#706E68",
    )
    ax.legend(title="post-transplant window", frameon=False, ncol=len(groups),
              loc="lower center", bbox_to_anchor=(0.5, 1.08))
    tidy_axes(ax, grid="x", hide=("top", "right", "left"))
    ax.tick_params(axis="y", length=0)
    fig.subplots_adjust(top=0.67)
    _finish(fig, save)


def plot_temporal_screen(result, scope=None, fdr_threshold=0.05, name_top=8, save=None):
    """All tested rules: largest between-bin change against temporal FDR."""
    if result.empty:
        print("No temporal rules could be tested.")
        return
    x = result["range"].to_numpy(float) * 100
    fdr = result["fdr"].to_numpy(float)
    y = -np.log10(np.clip(fdr, 1e-300, None))
    passed = fdr <= fdr_threshold
    fig, (ax, key) = plt.subplots(
        1, 2, figsize=(12, 5.4), gridspec_kw={"width_ratios": [4.2, 1.7]},
    )
    ax.scatter(x[~passed], y[~passed], s=28, color=_NEUTRAL, alpha=0.55,
               edgecolor="none")
    ax.scatter(x[passed], y[passed], s=48, color="#6A5AA8", alpha=0.9,
               edgecolor="white", linewidth=0.7)
    ax.axhline(-np.log10(fdr_threshold), color=_HAIRLINE, lw=1.1)
    names = result.sort_values(["fdr", "range"], ascending=[True, False]).head(name_top)
    colors = sns.color_palette("colorblind", n_colors=max(len(names), 1))
    for color, (rule, row) in zip(colors, names.iterrows()):
        px = 100 * row["range"]
        py = -np.log10(max(row["fdr"], 1e-300))
        ax.scatter(px, py, s=62, facecolor=color, edgecolor="white",
                   linewidth=0.8, zorder=4)

    key.axis("off")
    status = ("FDR-significant signals" if passed.any()
              else "Largest exploratory signals\n(none passes FDR 0.05)")
    key.set_title(status, fontsize=10.5, loc="left", pad=14)
    for rank, (color, (rule, row)) in enumerate(zip(colors, names.iterrows()), start=1):
        ypos = 1 - (rank - 0.4) / max(name_top, len(names))
        key.scatter(0.02, ypos - 0.005, s=48, color=color, edgecolor="white",
                    linewidth=0.7, transform=key.transAxes)
        key.text(0.08, ypos, rule.replace(" -> ", " → "),
                 transform=key.transAxes, fontsize=8.5, fontweight="bold",
                 va="top", color=_INK)
        key.text(0.08, ypos - 0.055,
                 f"gap {100 * row['range']:.0f} pp · FDR {row['fdr']:.3g}",
                 transform=key.transAxes, fontsize=7.8, va="top", color="#706E68")
    top = max(-np.log10(fdr_threshold) + 0.15, np.nanmax(y) * 1.08)
    ax.set_ylim(-0.05, top)
    _fdr_ticks(ax, top)
    ax.set_xlabel("largest gap between any two post-transplant windows (percentage points)")
    ax.set_ylabel("FDR across all tested rules  (smaller = stronger)")
    ax.set_title(_titled("Temporal pair-rule screen", scope), fontsize=13, pad=38)
    # Two lines: one long one runs out of the axes and under the key's heading.
    ax.text(
        0, 1.015,
        ("Analysis: pairwise rules · Unit: biopsy · Groups: <30 / 30–100 / >100 days\n"
         f"Eligibility: cell-count controlled · Tested rules: {len(result)}"),
        transform=ax.transAxes, ha="left", va="bottom", fontsize=8.5,
        color="#706E68", linespacing=1.4,
    )
    tidy_axes(ax, grid="both", hide=("top", "right"))
    fig.subplots_adjust(wspace=0.08)
    _finish(fig, save)


def plot_time_profiles(values, metadata, rules, result, groups, group_col,
                       scope=None, save=None):
    """Biopsy-level rule scores across Control and post-transplant time bins."""
    rules = [rule for rule in rules if rule in values.index]
    if not rules:
        print("No requested temporal rules are available.")
        return
    ncols = 2 if len(rules) > 1 else 1
    nrows = int(np.ceil(len(rules) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(12, 3.7 * nrows + 0.8),
                             squeeze=False)
    colors = {"Control": "#67B58A", "<30": "#58A6A6",
              "30-100": "#E3A33D", ">100": "#806FB3"}
    rng = np.random.default_rng(7)

    for ax, rule in zip(axes.flat, rules):
        means = []
        for position, group in enumerate(groups):
            units = metadata.loc[metadata[group_col] == group, "Biopsy"].drop_duplicates()
            observed = values.reindex(columns=units).loc[rule].dropna() * 100
            jitter = rng.uniform(-0.12, 0.12, len(observed))
            ax.scatter(position + jitter, observed, s=33, color=colors.get(group, _NEUTRAL),
                       alpha=0.62, edgecolor="white", linewidth=0.6)
            mean = observed.mean() if len(observed) else np.nan
            means.append(mean)
            if len(observed):
                ax.scatter(position, mean, s=92, marker="D", color=colors.get(group, _INK),
                           edgecolor="white", linewidth=1.2, zorder=5)
            ax.text(position, -108, f"eligible n={len(observed)}", ha="center", va="top",
                    fontsize=8, color="#66645F")
        ax.plot(range(len(groups)), means, color="#4A4844", lw=1.4, alpha=0.7,
                zorder=3)
        ax.axhline(0, color=_ZERO, lw=1.2)
        ax.set_xticks(range(len(groups)), groups)
        ax.set_ylim(-118, 108)
        ax.set_ylabel("biopsy rule balance (pp)\nattraction FOVs − avoidance FOVs")
        evidence = f"temporal FDR {result.at[rule, 'fdr']:.3g}" if rule in result.index else "context rule"
        ax.set_title(f"{rule.replace(' -> ', ' → ')}\n{evidence}", fontsize=10, pad=8)
        tidy_axes(ax, grid="y", hide=("top", "right"))
        ax.tick_params(length=0)

    for ax in axes.flat[len(rules):]:
        ax.set_visible(False)
    parts = str(scope or "").split(" — ", 1)
    organ = parts[0] if parts and parts[0] else ""
    context = parts[1] if len(parts) > 1 else ""
    title = _titled("Biopsy-level temporal pair-rule profiles", organ)
    subtitle = (f"Analysis: pairwise rules · Unit: biopsy · Groups: {' / '.join(groups)} · "
                "FOVs: eligible only")
    if context:
        subtitle = f"Context: {context} · {subtitle}"
    fig.suptitle(title, fontsize=14, y=0.99)
    fig.text(0.5, 0.947, subtitle, ha="center", va="top", fontsize=8.5,
             color="#706E68")
    fig.subplots_adjust(top=0.81, hspace=0.62, wspace=0.28)
    _finish(fig, save)


def plot_pooled_time_contrasts(results, contrasts, organ, fdr_threshold=0.05,
                               name_top=6, save=None):
    """Two pooled biopsy-level time contrasts with one joint FDR family."""
    shown = [(key, contrasts[key], results[key]) for key in contrasts if key in results]
    if not shown:
        print("No pooled temporal contrasts could be tested.")
        return

    total_tests = sum(len(result) for _, _, result in shown)
    fig, axes = plt.subplots(
        len(shown), 2, figsize=(12, 4.25 * len(shown) + 1.4),
        gridspec_kw={"width_ratios": [4.8, 1.7]}, squeeze=False,
    )

    for row, (key_name, spec, result) in enumerate(shown):
        ax, key = axes[row]
        gap = result["effect_size"].to_numpy(float) * 100
        fdr = result["fdr"].to_numpy(float)
        evidence = -np.log10(np.clip(fdr, 1e-300, None))
        passed = fdr <= fdr_threshold
        cutoff = -np.log10(fdr_threshold)

        ax.scatter(gap, evidence, s=28, color=_NEUTRAL, alpha=0.55,
                   edgecolor="none")
        ax.axvline(0, color=_ZERO, lw=1.2)
        ax.axhline(cutoff, color=_HAIRLINE, lw=1.1)

        names = result.sort_values(
            ["fdr", "effect_size"], key=lambda s: s.abs() if s.name == "effect_size" else s,
            ascending=[True, False],
        ).head(name_top)
        colors = sns.color_palette("colorblind", n_colors=max(len(names), 1))
        for color, (_, item) in zip(colors, names.iterrows()):
            ax.scatter(
                100 * item["effect_size"],
                -np.log10(max(item["fdr"], 1e-300)),
                s=60, color=color, edgecolor="white", linewidth=0.8, zorder=4,
            )

        limit = max(float(np.nanmax(np.abs(gap))) * 1.12, 1)
        top = max(cutoff + 0.15, float(np.nanmax(evidence)) * 1.08)
        ax.set_xlim(-limit, limit)
        ax.set_ylim(-0.05, top)
        _fdr_ticks(ax, top)
        ax.set_title(spec["title"], fontsize=11, pad=10)
        ax.set_xlabel(
            "gap in eligible-biopsy net score (percentage points)\n"
            f"← higher in {spec['b_name']}     higher in {spec['a_name']} →",
            fontsize=9,
        )
        ax.set_ylabel("joint FDR  (smaller = stronger)")
        tidy_axes(ax, grid="both", hide=("top", "right"))

        key.axis("off")
        passed_n = int(passed.sum())
        key.set_title(
            f"Largest signals · {passed_n} pass FDR {fdr_threshold:g}",
            fontsize=10, loc="left", pad=7,
        )
        for rank, (color, (rule, item)) in enumerate(zip(colors, names.iterrows()), start=1):
            ypos = 1 - (rank - 0.35) / max(name_top, len(names))
            key.scatter(0.02, ypos - 0.005, s=46, color=color, edgecolor="white",
                        linewidth=0.7, transform=key.transAxes)
            label = rule.replace(" -> ", " → ").replace("_", " ")
            key.text(0.08, ypos, label,
                     transform=key.transAxes, fontsize=7.8, fontweight="bold",
                     va="top", color=_INK)
            key.text(
                0.08, ypos - 0.055,
                f"gap {item['effect_size'] * 100:+.0f} pp · FDR {item['fdr']:.3g}",
                transform=key.transAxes, fontsize=7.7, va="top", color="#706E68",
            )

    fig.suptitle(f"Pooled temporal contrasts — {organ}", fontsize=14, y=0.985)
    fig.text(
        0.5, 0.937,
        (f"Analysis: pairwise rules · Unit: biopsy · FOVs: eligible only · "
         f"FDR correction: 2 contrasts / {total_tests} tests"),
        ha="center", va="top", fontsize=8.5, color="#706E68",
    )
    fig.subplots_adjust(left=0.09, right=0.985, top=0.84, hspace=0.64, wspace=0.07)
    _finish(fig, save)


# ---------------------------------------------------------------------------
# Rule-state context used by the new-findings notebook
# ---------------------------------------------------------------------------

def _cell_share_table(cells, metadata, cell_types, columns):
    """Per-FOV percentage of all cells belonging to each requested cell type."""
    totals = cells.groupby("fov").size()
    counts = (
        cells.loc[cells["cell type"].isin(cell_types)]
        .groupby(["fov", "cell type"]).size().unstack(fill_value=0)
        .reindex(columns=cell_types, fill_value=0)
    )
    shares = counts.div(totals, axis=0).mul(100).rename_axis("FOV").reset_index()
    return metadata[["FOV", *columns]].drop_duplicates().merge(shares, on="FOV", how="left")


def _eligible_by_stage(eligibility, metadata, spec, stages):
    """One rule's eligible FOVs, stage by stage. The panels below share it, so they
    are always drawn on the same fields."""
    for stage in stages:
        mask = (metadata["Organ"] == spec["organ"]) & (metadata[spec["score"]] == stage)
        fovs = metadata.loc[mask, "FOV"].drop_duplicates()
        can_test = eligibility.reindex(columns=fovs).loc[spec["rule"]]
        yield stage, can_test.index[can_test]


def _prevalence_panel(ax, states, eligibility, metadata, spec, stages, bars=False,
                      fov_label="eligible FOVs"):
    """How often the rule attracts and how often it avoids, as two separate lines."""
    attraction, avoidance = [], []
    attraction_n, avoidance_n, eligible_n = [], [], []
    for stage, fovs in _eligible_by_stage(eligibility, metadata, spec, stages):
        state = states.reindex(columns=fovs).loc[spec["rule"]]
        share = 100 / len(fovs) if len(fovs) else np.nan
        attraction_n.append(int((state == 1).sum()))
        avoidance_n.append(int((state == -1).sum()))
        eligible_n.append(len(fovs))
        attraction.append(share * attraction_n[-1])
        avoidance.append(share * avoidance_n[-1])

    shown = []
    for label, short, values, counts, color in (
            ("Attraction", "attract", attraction, attraction_n, "#2878D0"),
            ("Avoidance", "avoid", avoidance, avoidance_n, "#D85D62")):
        if not any(counts):
            continue
        if bars:
            both = any(attraction_n) and any(avoidance_n)
            offset = (-0.18 if short == "attract" else 0.18) if both else 0
            ax.bar(np.arange(len(stages)) + offset, values,
                   width=0.34 if both else 0.56, color=color, label=label)
        else:
            ax.plot(range(len(stages)), values, marker="D", markersize=6, lw=2.2,
                    color=color, label=label)
        shown.append((short, counts))

    labels = [
        f"{stage}\n" + " | ".join(
            [str(counts[position]) for _, counts in shown]
            + [str(eligible_n[position])]
        )
        for position, stage in enumerate(stages)
    ]
    ax.set_ylim(0, 100)
    ax.set_xticks(range(len(stages)), labels)
    ax.set_xlabel(" | ".join([short for short, _ in shown] + [fov_label]),
                  fontsize=7.5, labelpad=2)
    ax.set_ylabel(f"{fov_label}\nwith rule (%)", labelpad=8)
    if shown:
        ax.legend(frameon=False, fontsize=8.2, loc="center left",
                  bbox_to_anchor=(1.01, 0.5), borderaxespad=0)
    tidy_axes(ax, grid="y", hide=("top", "right"))
    ax.tick_params(length=0)


def _shares_panel(ax, cells, eligibility, metadata, spec, stages):
    """Abundance of the rule's two cell types, one dot per eligible FOV."""
    types = list(dict.fromkeys(spec["rule"].split(" -> ")))
    colors = ["#D97832", "#169873"]
    shares = _cell_share_table(cells, metadata, types, []).set_index("FOV")
    spread = np.random.default_rng(0)
    means = {cell_type: [] for cell_type in types}
    labels = []

    for position, (stage, fovs) in enumerate(
            _eligible_by_stage(eligibility, metadata, spec, stages)):
        values = shares.reindex(fovs)
        for cell_type, color, offset in zip(types, colors, (-0.075, 0.075)):
            column = values[cell_type].fillna(0)
            means[cell_type].append(column.mean() if len(column) else np.nan)
            ax.scatter(position + offset + spread.uniform(-0.032, 0.032, len(column)),
                       column, s=16, alpha=0.5, linewidth=0, color=color)
        labels.append(f"{stage}\nn={len(fovs)}")

    for cell_type, color, role, offset in zip(types, colors, ("Ant", "Con"), (-0.075, 0.075)):
        ax.plot(np.arange(len(stages)) + offset, means[cell_type], marker="D",
                markersize=6, lw=2.2, color=color,
                label=f"{role}: {cell_type.replace('_', ' ')}")
    ax.set_ylim(bottom=0)
    ax.set_xticks(range(len(stages)), labels)
    ax.set_ylabel("cells in eligible\nFOVs (%)", labelpad=8)
    ax.legend(frameon=False, fontsize=8.2, loc="center left",
              bbox_to_anchor=(1.01, 0.5), borderaxespad=0)
    tidy_axes(ax, grid="y", hide=("top", "right"))
    ax.tick_params(length=0)


def plot_rule_summary(states, eligibility, cells, metrics, metadata, spec, stages,
                      metric="Lift", reference=1, trend_results=None,
                      pair_results=None, state=None, test_label=None, save=None):
    """One rule in four stacked panels: how often it fires, what it is made of, how
    strong it is where it fires, and what every FOV of the stage was classified as.

    One panel per row rather than four across: the panels share an x axis of three
    stages, so full width each is readable where a quarter of the width is not.
    """
    rule, organ, group_col = spec["rule"], spec["organ"], spec["score"]
    threshold = eligibility.attrs.get("min_cells")
    test_unit = spec.get("test_unit", "FOV")
    detail = (
        f"Plots: FOV · Tests: {test_unit} · Organ: {organ} · "
        f"Score: {group_col.replace(' score', '').lower()} · "
        f"Stages: {' / '.join(stages)} · "
        f"Eligibility: {f'≥{threshold} cells/type' if threshold else 'controlled'} · "
        "n = eligible FOVs"
    )
    # Built at the width LaTeX gives it, so nothing shrinks on the page.
    with plt.rc_context(_PANEL_FONTS):
        fig, axes = plt.subplots(
            4, 1, figsize=(_TEXT_WIDTH, 8.0),
            gridspec_kw={"height_ratios": [1, 1, 1, 1], "hspace": 0.82},
        )
        _prevalence_panel(axes[0], states, eligibility, metadata, spec, stages)
        _shares_panel(axes[1], cells, eligibility, metadata, spec, stages)
        _metric_panel(axes[2], metrics, eligibility, metadata, spec, stages,
                      metric, reference, state=state)
        counts, eligible_counts, nets = _stage_state_counts(
            states, eligibility, metadata, rule, organ, group_col, stages)
        _draw_state_bars(axes[3], counts, eligible_counts, nets, stages)
        state_handles = [
            Line2D([0], [0], marker="s", linestyle="none", markersize=6,
                   color=_STATE_COLORS[name], label=name)
            for name in _STATE_ORDER
        ]
        axes[3].legend(handles=state_handles, frameon=False, fontsize=7.4,
                       loc="center left", bbox_to_anchor=(1.01, 0.5),
                       borderaxespad=0)

        direction = {1: "attraction", -1: "avoidance"}.get(state)
        strength_title = f"how strong ({direction + ' ' if direction else ''}{metric})"
        for ax, name in zip(axes, ("how often it fires", "what it is made of",
                                   strength_title, "every FOV")):
            ax.set_title(name, fontsize=8.4, color="#5F5D58", loc="left", pad=5)

        fig.suptitle(f"{organ} · {rule.replace(' -> ', ' → ')}", fontsize=11.5, y=0.995)
        fig.text(0.5, 0.973, detail, ha="center", va="top", fontsize=7,
                 color="#706E68", linespacing=1.4)
        fig.align_ylabels(axes)
        fig.subplots_adjust(top=0.91, bottom=0.05, left=0.14, right=0.76)
        _finish(fig, save)


def plot_rule_prevalence_all_fovs(states, metadata, spec, stages, save=None):
    """The summary's "how often it fires" panel, counted over every FOV of the stage."""
    detail = (
        f"Plots: FOV · Organ: {spec['organ']} · "
        f"Score: {spec['score'].replace(' score', '').lower()} · "
        f"Stages: {' / '.join(stages)} · n = all FOVs (no eligibility filter)"
    )
    every_fov = states.astype(bool) | True
    with plt.rc_context(_PANEL_FONTS):
        fig, ax = plt.subplots(figsize=(_TEXT_WIDTH, 2.6))
        _prevalence_panel(ax, states, every_fov, metadata, spec, stages,
                          fov_label="all FOVs")
        ax.set_title("how often it fires", fontsize=8.4, color="#5F5D58", loc="left", pad=5)
        fig.suptitle(f"{spec['organ']} · {spec['rule'].replace(' -> ', ' → ')}",
                     fontsize=11.5, y=0.98)
        fig.text(0.5, 0.9, detail, ha="center", va="top", fontsize=7, color="#706E68")
        fig.subplots_adjust(top=0.72, bottom=0.25, left=0.14, right=0.76)
        _finish(fig, save)


def plot_organ_rule_context(states, eligibility, cells, metadata, rule, group_col,
                            stages, stage_results, heading,
                            organs=("Colon", "Duodenum"), save=None):
    """Show one organ rule contrast together with both cell-abundance profiles."""
    antecedent, consequent = rule.split(" -> ")
    shares = _cell_share_table(cells, metadata, [antecedent, consequent],
                               ["Organ", group_col])
    organ_colors = {"Colon": "#2878D0", "Duodenum": "#D97832"}
    cell_colors = {antecedent: "#D97832", consequent: "#169873"}
    threshold = eligibility.attrs.get("min_cells")

    fig = plt.figure(figsize=(12.2, 8.0))
    grid = fig.add_gridspec(2, 2, height_ratios=(1.18, 1), hspace=0.56, wspace=0.25)
    rule_ax = fig.add_subplot(grid[0, :])
    abundance_axes = [fig.add_subplot(grid[1, i]) for i in range(2)]

    for organ in organs:
        means, counts = [], []
        for stage in stages:
            fovs = metadata.loc[
                (metadata["Organ"] == organ) & (metadata[group_col] == stage), "FOV"
            ].drop_duplicates()
            state = states.reindex(columns=fovs).loc[rule]
            can_test = eligibility.reindex(columns=fovs).loc[rule]
            observed = state[can_test]
            means.append(100 * observed.mean() if len(observed) else np.nan)
            counts.append((int((observed != 0).sum()), len(observed)))
        rule_ax.plot(range(len(stages)), means, color=organ_colors[organ], lw=2.1,
                     marker="D", markersize=7, label=organ)
        for position, (mean, count) in enumerate(zip(means, counts)):
            if np.isfinite(mean):
                offset = 9 if organ == "Colon" else -12
                rule_ax.annotate(f"rule/eligible={count[0]}/{count[1]}",
                                 (position, mean), xytext=(0, offset),
                                 textcoords="offset points", ha="center", fontsize=8,
                                 color=organ_colors[organ])

    evidence = []
    for stage in stages:
        result = stage_results.get(stage, pd.DataFrame())
        if rule in result.index:
            evidence.append(f"{stage} {result.at[rule, 'fdr']:.3g}")
    rule_ax.axhline(0, color=_ZERO, lw=1.1)
    rule_ax.set_xticks(range(len(stages)), stages)
    rule_ax.set_ylim(-112, 112)
    rule_ax.set_ylabel("net rule score among eligible FOVs (%)")
    rule_ax.set_title(rule.replace(" -> ", " → "), fontsize=11, pad=26)
    rule_ax.text(0.5, 1.01, "organ FDR: " + " · ".join(evidence),
                 transform=rule_ax.transAxes, ha="center", va="bottom",
                 fontsize=8.3, color="#706E68")
    rule_ax.legend(frameon=False, ncol=len(organs), loc="lower center",
                   bbox_to_anchor=(0.5, 1.18))
    tidy_axes(rule_ax, grid="y", hide=("top", "right"))
    rule_ax.tick_params(length=0)

    for ax, organ in zip(abundance_axes, organs):
        for cell_type in (antecedent, consequent):
            means = []
            for stage in stages:
                fovs = metadata.loc[
                    (metadata["Organ"] == organ) & (metadata[group_col] == stage), "FOV"
                ].drop_duplicates()
                can_test = eligibility.reindex(columns=fovs).loc[rule]
                eligible_fovs = can_test.index[can_test]
                values = shares.loc[shares["FOV"].isin(eligible_fovs), cell_type]
                means.append(values.fillna(0).mean())
            ax.plot(range(len(stages)), means, color=cell_colors[cell_type], lw=2,
                    marker="D", markersize=6, label=cell_type.replace("_", " "))
        ax.set_xticks(range(len(stages)), stages)
        ax.set_ylabel("mean cells in each FOV (%)")
        ax.set_title(f"{organ}: cell abundance", fontsize=10.5)
        ax.legend(frameon=False, fontsize=8.5)
        tidy_axes(ax, grid="y", hide=("top", "right"))
        ax.tick_params(length=0)

    threshold_text = f"eligibility ≥{threshold} cells/type" if threshold else "eligibility-controlled"
    fig.suptitle(heading, fontsize=14, y=0.99)
    fig.text(0.5, 0.950,
             (f"Analysis: pairwise rule states · Unit: FOV · Organs: {' / '.join(organs)} · "
              f"Score: {group_col.replace(' score', '').lower()} · "
              f"Stages: {' / '.join(stages)} · Eligibility: "
              f"{threshold_text.replace('eligibility ', '')} · Rule and abundance: same FOVs"),
             ha="center", va="top", fontsize=8.5, color="#706E68")
    fig.subplots_adjust(top=0.82, left=0.09, right=0.98, bottom=0.08)
    _finish(fig, save)
