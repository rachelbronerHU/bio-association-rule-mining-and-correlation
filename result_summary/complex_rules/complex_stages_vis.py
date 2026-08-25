"""Drawing for the stage analysis: where a longer rule lives as the disease worsens.

Only plotting lives here; the counting stays in the notebook.
Colours come from `complex_vis`, so a verdict means the same thing in every figure.
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.patches import Patch, Rectangle
from matplotlib.transforms import blended_transform_factory, offset_copy

from vis_helper import _finish, _titled, tidy_axes, spread_labels, INK
from complex_vis import CLASS_COLORS, text_on

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
}
STATE_LABELS = {
    "informative": "the rule is here and adds something",
    "adds":        "here, and beat a real shorter rule",
    "new":         "here, and nothing shorter was mined",
    "covered":     "the pairs cover it, the rule adds nothing",
    "redundant":   "here, but a shorter rule already said it",
    "pairs_only":  "not here at all, but its pairs are",
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
    """Each stage's name over how many FOVs it holds - the denominator, in the open."""
    sizes = counts.drop_duplicates("stage").set_index("stage")["n_fovs"]
    return [f"{s}\nn={int(sizes[s])}" for s in stages]


def plot_stage_bars(counts, stages, scope=None, split_adds=False,
                    split_covered=False, save=None):
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
    reach = float(sum(w.to_numpy() for w in wide.values()).max()) * 1.06
    y = np.arange(len(names))

    fig, axes = plt.subplots(1, len(stages), figsize=(12.5, len(names) * 0.34 + 2.2),
                             sharey=True, gridspec_kw={"wspace": 0.05})
    axes = np.atleast_1d(axes)

    for ax, stage, tick in zip(axes, stages, _stage_ticks(counts, stages)):
        left = np.zeros(len(names))
        for column in columns:
            width = wide[column][stage].to_numpy()
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
    axes[len(stages) // 2].set_xlabel(
        "share of the stage's FOVs      (empty = neither the rule nor its pairs)",
        fontsize=10)

    fig.suptitle(_titled("Where each rule lives, stage by stage", scope),
                 fontsize=13, x=0.5, y=1.0)
    axes[-1].legend(handles=_keys(columns), fontsize=8.5, frameon=False,
                    loc="upper left", bbox_to_anchor=(1.03, 1.0),
                    handlelength=1.1, borderpad=0, labelspacing=0.7)
    _finish(fig, save)


# What each figure measures, in the words the reader sees: the title over it, the label
# on its colour scale, and the note under the title, one sentence per line.
_MEASURES = {
    "earns": {
        "title": "Share of better complex rule out of all FOVs with the rule or its parts",
        "scale": "% - better complex rule out of the FOVs with the rule or its parts",
        "note": [
            "Each cell: the share, then the two counts it divides — FOVs with "
            "informative complex rule | simpler rule only / better.",
            "Dashed outline: fewer than {min_n} FOVs behind it.",
        ],
    },
    "informative": {
        "title": "Share of informative complex rule out of all FOVs in the stage",
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
             "scale": f"% - {STATE_LABELS.get(value, value)}",
             "note": _OTHER_NOTE}
    return _MEASURES.get(value, {}).get(part, plain[part])


def plot_stage_heatmap(counts, stages, value="earns", scope=None, min_n=5, save=None):
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
    behind = _wide(counts, stages, "n_present").reindex(names)
    adds = _wide(counts, stages, "informative").reindex(names).to_numpy()
    covered = _wide(counts, stages, "covered").reindex(names).to_numpy()
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
            ax.annotate(f"{int(adds[row, column])} | {int(covered[row, column])}",
                        xy=(column + 0.04, row), ha="left", va="center", fontsize=8,
                        color=ink)

    ax.set_xticks(range(len(stages)), _stage_ticks(counts, stages), fontsize=10.5)
    ax.set_yticks(range(len(names)), names, fontsize=8.5)
    ax.tick_params(length=0)
    ax.spines[:].set_visible(False)
    ax.set_xticks(np.arange(len(stages) + 1) - 0.5, minor=True)
    ax.set_yticks(np.arange(len(names) + 1) - 0.5, minor=True)
    ax.grid(which="minor", color="white", lw=2.5)

    bar = fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=ramp), ax=ax,
                       pad=0.02, fraction=0.04)
    bar.set_label(_says(value, "scale"), fontsize=9)
    bar.outline.set_visible(False)

    note = [line.format(min_n=min_n) for line in _says(value, "note")]
    ax.set_title(_titled(_says(value, "title"), scope), fontsize=12.5, loc="left",
                 pad=16 + 12 * len(note))
    ax.annotate("\n".join(note), xy=(0, 1.0), xytext=(0, 10),
                xycoords="axes fraction", textcoords="offset points",
                fontsize=8.5, color="#898781", va="bottom", linespacing=1.45)
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


def plot_stage_trend(counts, stages, value="earns", scope=None, top_n=8, min_n=5,
                     save=None):
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

    fig.suptitle(_titled(f"Which rules fade, and which arrive (top {top_n} each way)",
                         scope), fontsize=13, y=0.99)
    fig.text(0.5, 0.925, f"hollow dot: fewer than {min_n} FOVs behind it    "
                         f"(dot size = how many)", fontsize=8, color="#898781",
             ha="center")
    _finish(fig, save)
