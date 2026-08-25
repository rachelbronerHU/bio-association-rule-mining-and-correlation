"""Drawing for the differential-rule analysis: which rules tell two groups apart.

Only plotting lives here; the tests and the tables stay in the notebook.
The shared pieces - saving, titles, group colours - come from `vis_helper.py`.
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import to_rgba
from matplotlib.lines import Line2D

from vis_helper import (_finish, _titled, _category_colors, tidy_axes, spread_labels,
                        NEUTRAL as _NEUTRAL, HAIRLINE as _HAIRLINE,
                        ZERO as _ZERO, INK as _INK)


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
    """Left-aligned and a line above the legend, so the two never run into each other."""
    ax.set_title(_titled(text, scope), fontsize=12, pad=28, loc="left")


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
    plt.tight_layout()
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
    plt.tight_layout()
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
                    fontsize=9, color="#96958f")
        ax.set_xticks([])
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


def _room_for_names(fig):
    """Margins the names can live in: they sit outside the panels, where tight_layout
    cannot see them, so the space is set by hand."""
    fig.subplots_adjust(left=0.205, right=0.99, top=0.845, bottom=0.09, wspace=0.63)


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

    fig.suptitle(_titled(f"{way} rules and the cells behind them (top {top_n})", scope),
                 fontsize=13, y=0.97)
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

    # Each panel keeps its own scale: the two directions rarely cover the same range, and
    # a shared one leaves whichever panel is smaller as mostly empty space.
    fig, axes = plt.subplots(1, 2, figsize=(14.5, 5.4))
    for ax, rising, title in ((axes[0], True, "climbing with severity"),
                              (axes[1], False, "fading with severity")):
        nets, names, slopes, steady, found = _movers(trend, stages, fdr_threshold,
                                                     top_n, rising)
        _trend_panel(ax, nets, names, slopes, steady,
                     f"{title}   ({found} {thing}{'s' if found != 1 else ''})", stages,
                     unit=value_label or _NET_LABEL)
        _plain(ax)

    fig.suptitle(_titled(f"{what} that move with severity (top {top_n} each way)", scope),
                 fontsize=13, y=0.97)
    _room_for_names(fig)
    _finish(fig, save)
