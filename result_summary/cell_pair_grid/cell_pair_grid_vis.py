"""Drawing for the cell-pair grid: which cell type sits next to which.

One dot per rule on a center-cell x neighbor-cell grid. Colour = how strong the rule is
(lift), size = the second number: confidence inside a single FOV, or the share of a
group that has the rule. Every size here is measured in inches, so a grid of 32 cell
types stays as readable as a grid of 8, alone or in a panel.

Only plotting lives here; the counting stays in the notebook.
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize

from vis_helper import _finish


CELL_GROUP_COLORS = {"epithelial": "#c0392b", "immune": "#2980b9",
                     "structure": "#27ae60", "other": "#7f8c8d"}
CELL_GROUP_ORDER = ["epithelial", "immune", "structure", "other"]

_RATIO_METRICS = {"Lift", "Conviction"}     # 1 = no association -> the scale diverges at 1
_ZERO_METRICS = {"Leverage"}                # 0 = no association
_RATIO_TICKS = [0.125, 0.25, 0.5, 0.75, 1, 1.5, 2, 4, 8, 16, 32]
_COLOR_QUANTILE = 0.98      # the few strongest rules saturate instead of washing the rest out

_SQUARE_ALONE = 0.30        # inches per cell type in a grid drawn on its own
_SQUARE_PANEL = 0.22        # ... and in a panel, where many grids share the page
_LEGEND_INCHES = 3.0        # the right margin the keys live in


# --- what a dot looks like -------------------------------------------------

def _finite(values):
    """The values as floats, with infinity replaced by the largest finite one."""
    v = np.asarray(values, dtype=float)
    ok = v[np.isfinite(v)]
    return np.where(np.isposinf(v), ok.max() if ok.size else 1.0, v)


def _limit(values):
    """How far the scale reaches from its middle. The most extreme few rules are left
    outside it (they simply take the end color), so the rest of the grid stays readable."""
    v = np.abs(np.asarray(values, dtype=float))
    v = v[np.isfinite(v)]
    if not v.size:
        return 1.0
    return float(np.quantile(v, _COLOR_QUANTILE)) or float(v.max()) or 1.0


def _color_scale(values, col):
    """How a metric becomes a color.

    Lift and conviction are ratios around 1, so they get a scale that is symmetric
    in log2 - a rule twice as likely and a rule half as likely are equally far from
    the middle - while the colorbar is still written in the metric's own units.
    """
    v = _finite(values)
    if col in _RATIO_METRICS:
        def to_log(x):
            f = _finite(x)
            return np.log2(np.where(f > 0, f, np.nan))
        limit = _limit(to_log(v))
        # Whole steps of log2, the same units the dot panels use, so a reader does not
        # have to switch scales between two figures on the same page.
        step = 1 if limit <= 4 else 2
        ticks = [t for t in range(-12, 13, step) if abs(t) <= limit]
        return (to_log, Normalize(-limit, limit), "RdBu_r",
                ticks, [f"{t:g}" for t in ticks])
    if col in _ZERO_METRICS:
        limit = _limit(v)
        return _finite, Normalize(-limit, limit), "RdBu_r", None, None
    return _finite, Normalize(0.0, _limit(v)), "Reds", None, None


def _grid_side(n_cells, per_cell):
    """How wide the grid itself is, in inches."""
    return max(4.2, per_cell * n_cells)


def _dot_area(n_cells, per_cell):
    """The area of the biggest dot - a little smaller than one square of the grid,
    so a full-size dot never spills into its neighbors."""
    square_pt = _grid_side(n_cells, per_cell) / n_cells * 72
    return np.pi / 4 * (0.82 * square_pt) ** 2


def _size_points(values, size_ref, max_area):
    """Dot areas. `size_ref` is shared by a whole panel, so the same size means the
    same number in every grid of that panel."""
    frac = np.clip(_finite(values) / size_ref, 0.0, 1.0) if size_ref else 0.0
    return max_area * (0.06 + 0.94 * frac)


def build_matrix_scale(frames, color_col, size_col, size_ref=None):
    """One color scale and one dot-size reference for a whole panel of grids.

    `frames` : the tables that will be drawn. Pass `size_ref=1.0` when the size is
    already a share, so the dots read as absolute percentages.
    """
    frames = [f for f in frames if len(f)]
    colors = pd.concat([f[color_col] for f in frames]) if frames else pd.Series(dtype=float)
    sizes = pd.concat([f[size_col] for f in frames]) if frames else pd.Series(dtype=float)
    if size_ref is None:
        size_ref = float(_finite(sizes).max()) if len(sizes) else 1.0
    transform, norm, cmap, ticks, tick_labels = _color_scale(colors, color_col)
    return {"transform": transform, "norm": norm, "cmap": cmap,
            "ticks": ticks, "tick_labels": tick_labels, "size_ref": max(size_ref, 1e-9)}


# --- the empty grid, the titles and the keys -------------------------------

def _draw_grid(ax, cell_order, cell_groups, fontsize):
    """The grid a dot sits on: one square per pair of cell types.

    Thin lines run between every row and column and a stronger line separates the cell
    groups, so a dot can be followed back to the names on the edges. The background is
    left white on purpose - any tint behind the dots would change how their color reads.
    """
    size = len(cell_order)
    groups = [(cell_groups or {}).get(cell, "other") for cell in cell_order]

    ax.set_xlim(-0.5, size - 0.5)
    ax.set_ylim(-0.5, size - 0.5)

    for i in range(1, size):                           # the line between two groups
        if groups[i] != groups[i - 1]:
            ax.axhline(i - 0.5, color="0.55", lw=1.0, zorder=2)
            ax.axvline(i - 0.5, color="0.55", lw=1.0, zorder=2)

    ax.set_xticks(range(size))
    ax.set_yticks(range(size))
    ax.set_xticks(np.arange(size) + 0.5, minor=True)   # the lines run *between* squares
    ax.set_yticks(np.arange(size) + 0.5, minor=True)
    ax.grid(which="minor", color="0.90", linewidth=0.6, zorder=1)
    ax.grid(which="major", visible=False)
    ax.tick_params(which="both", length=0)

    ax.set_xticklabels(cell_order, rotation=90, fontsize=fontsize)
    ax.set_yticklabels(cell_order, fontsize=fontsize)
    for labels in (ax.get_xticklabels(), ax.get_yticklabels()):
        for label in labels:
            group = (cell_groups or {}).get(label.get_text(), "other")
            label.set_color(CELL_GROUP_COLORS.get(group, "black"))
    for spine in ax.spines.values():
        spine.set_visible(False)


def _figure_titles(fig, title, subtitle, center):
    """Title and subtitle, placed in inches from the top so they never collide.

    Returns the fraction of the figure left for the grids.
    """
    height = fig.get_figheight()
    if title:
        fig.suptitle(title, fontsize=15, x=center, y=1 - 0.34 / height)
    if subtitle:
        fig.text(center, 1 - 0.66 / height, subtitle, ha="center", va="top",
                 fontsize=10, color="0.35")
    used = 0.30 + 0.34 * bool(title) + 0.40 * bool(subtitle)
    return 1 - used / height


def _matrix_legends(fig, scale, color_col, size_col, size_label, cell_groups,
                    left, top, max_area, outer_label=None):
    """The cell-group key, the colorbar and the dot-size key, stacked in the margin.

    Everything is placed in inches from `top`, so the stack looks the same whether the
    panel has one row of grids or six.
    """
    width, height = fig.get_figwidth(), fig.get_figheight()

    def down(inches):
        return top - inches / height

    groups = [g for g in CELL_GROUP_ORDER if g in set((cell_groups or {}).values())]
    if groups:
        marks = [plt.Line2D([0], [0], marker="s", linestyle="none", markersize=9,
                            color=CELL_GROUP_COLORS[g], label=g) for g in groups]
        fig.legend(handles=marks, loc="upper left", frameon=False,
                   bbox_to_anchor=(left + 0.55 / width, down(0.1)),
                   title="cell group", fontsize=9, title_fontsize=10)

    mappable = plt.cm.ScalarMappable(norm=scale["norm"], cmap=scale["cmap"])
    cbar = fig.colorbar(
        mappable, extend="both" if scale["norm"].vmin < 0 else "max",
        cax=fig.add_axes([left + 0.35 / width, down(4.4), 0.16 / width, 2.4 / height]))
    if scale["ticks"] is not None:
        cbar.set_ticks(scale["ticks"])
        cbar.set_ticklabels(scale["tick_labels"])
    ratio = scale["ticks"] is not None
    cbar.set_label("log2 " + color_col.lower() + "\n(0 = chance)" if ratio
                   else color_col.lower(), fontsize=10)
    cbar.ax.tick_params(labelsize=9)

    ref, is_share = scale["size_ref"], size_col == "share"
    handles = [plt.Line2D([0], [0], marker="o", linestyle="none", color="0.55",
                          markersize=float(np.sqrt(_size_points([v], ref, max_area)[0])),
                          label=f"{v:.0%}" if is_share else f"{v:.2g}")
               for v in (0.25 * ref, 0.5 * ref, ref)]
    # The rows have to clear the biggest dot, which grows with the grid.
    biggest = max(h.get_markersize() for h in handles)
    spacing = max(1.6, biggest / 9)
    fig.legend(handles=handles, loc="upper left", frameon=False,
               bbox_to_anchor=(left + 0.55 / width, down(5.1)),
               title=size_label or size_col.lower(), fontsize=9, title_fontsize=10,
               labelspacing=spacing, borderpad=0.6)

    if outer_label:
        ring = plt.Line2D([0], [0], marker="o", linestyle="none", markersize=13,
                          markerfacecolor="none", markeredgecolor="0.72",
                          markeredgewidth=1.1, label=outer_label)
        fig.legend(handles=[ring], loc="upper left", frameon=False,
                   bbox_to_anchor=(left + 0.55 / width,
                                   down(6.0 + 3 * biggest * 1.2 / 72)),
                   fontsize=9, handletextpad=1.2)


# --- the two calls the notebook makes --------------------------------------

def plot_rule_matrix(df, cell_order, cell_groups=None, color_col="Lift",
                     size_col="Confidence", title=None, subtitle=None, scale=None,
                     size_label=None, ax=None, per_cell=_SQUARE_ALONE,
                     show_diagonal=True, outer_size_col=None, outer_label=None,
                     mark_lone=False, save=None):
    """One grid: a dot per rule, center cell (y) against neighbor cell (x).

    df         : one row per rule, with 'antecedent', 'consequent' and both metrics.
    cell_order : the fixed cell-type order - the same list in every grid, so two
                 grids can be compared square by square.
    cell_groups: {cell type: group}, which colors the names and splits the grid.
    scale      : from `build_matrix_scale`; pass the panel's scale in to share it.
                 Left out, the grid builds its own scale from its own numbers.
    outer_size_col: a second, never-smaller size drawn as a ring behind each dot, so
                 the two can be read at once - the ring is what the dot could have been.
    mark_lone  : outline the dots whose mirror rule was never mined, which is what makes
                 the grid asymmetric.
    """
    if scale is None:
        scale = build_matrix_scale([df], color_col, size_col)
    size = len(cell_order)
    max_area = _dot_area(size, per_cell)

    own_fig = ax is None
    if own_fig:
        side = _grid_side(size, per_cell)
        fig, ax = plt.subplots(figsize=(side + _LEGEND_INCHES + 1.6, side + 2.4))

    _draw_grid(ax, cell_order, cell_groups, fontsize=9 if own_fig else 7)
    if show_diagonal:
        ax.plot([-0.5, size - 0.5], [-0.5, size - 0.5], color="0.72", lw=0.9,
                linestyle=(0, (4, 4)), zorder=2)

    place = {cell: i for i, cell in enumerate(cell_order)}
    rows = df[df["antecedent"].isin(place) & df["consequent"].isin(place)]
    if len(rows) and outer_size_col:
        ax.scatter(rows["consequent"].map(place), rows["antecedent"].map(place),
                   s=_size_points(rows[outer_size_col], scale["size_ref"], max_area),
                   facecolor="none", edgecolor="0.72", linewidth=0.9, zorder=2)
    if len(rows):
        ax.scatter(rows["consequent"].map(place), rows["antecedent"].map(place),
                   s=_size_points(rows[size_col], scale["size_ref"], max_area),
                   c=scale["transform"](rows[color_col]),
                   cmap=scale["cmap"], norm=scale["norm"],
                   edgecolor="0.25", linewidth=0.4, zorder=3)
        if mark_lone:
            mined = set(zip(rows["antecedent"], rows["consequent"]))
            lone = rows[[(c, a) not in mined
                         for a, c in zip(rows["antecedent"], rows["consequent"])]]
            if len(lone):
                ax.scatter(lone["consequent"].map(place), lone["antecedent"].map(place),
                           s=_size_points(lone[size_col], scale["size_ref"], max_area),
                           facecolor="none", edgecolor="#111111", linewidth=1.5, zorder=4)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("neighbor cell", fontsize=11 if own_fig else 9)
    ax.set_ylabel("center cell", fontsize=11 if own_fig else 9)

    if not own_fig:
        if title:
            ax.set_title(title, fontsize=11, pad=8)
        return

    width = fig.get_figwidth()
    left = 1 - _LEGEND_INCHES / width
    top = _figure_titles(fig, title or "which cell type sits next to which",
                         subtitle, center=left / 2)
    plt.tight_layout(rect=[0, 0, left, top])
    _matrix_legends(fig, scale, color_col, size_col, size_label, cell_groups,
                    left=left, top=top, max_area=max_area, outer_label=outer_label)
    _finish(fig, save)


def plot_rule_matrix_panel(frames, cell_order, cell_groups=None, color_col="Lift",
                           size_col="Confidence", num_cols=3, title=None,
                           subtitle=None, size_label=None, size_ref=None,
                           show_diagonal=True, outer_size_col=None, outer_label=None,
                           save=None):
    """One grid per entry of `frames` ({title: table}), all on one shared scale.

    The color scale and the dot-size reference are built once over every table, so a
    big dark dot means the same thing in all of them.
    """
    if not frames:
        print("No data to plot.")
        return
    scale = build_matrix_scale(list(frames.values()), color_col, size_col, size_ref=size_ref)

    size = len(cell_order)
    side = _grid_side(size, _SQUARE_PANEL)
    num_cols = max(1, min(num_cols, len(frames)))
    num_rows = (len(frames) + num_cols - 1) // num_cols

    fig, axes = plt.subplots(num_rows, num_cols, squeeze=False,
                             figsize=((side + 1.2) * num_cols + _LEGEND_INCHES,
                                      (side + 1.6) * num_rows + 1.2))
    axes = axes.flatten()
    for ax, (name, df) in zip(axes, frames.items()):
        plot_rule_matrix(df, cell_order, cell_groups=cell_groups, color_col=color_col,
                         size_col=size_col, title=name, scale=scale, ax=ax,
                         per_cell=_SQUARE_PANEL, show_diagonal=show_diagonal,
                         outer_size_col=outer_size_col)
    for ax in axes[len(frames):]:
        ax.set_visible(False)

    left = 1 - _LEGEND_INCHES / fig.get_figwidth()
    top = _figure_titles(fig, title, subtitle, center=left / 2)
    plt.tight_layout(rect=[0, 0, left, top])
    _matrix_legends(fig, scale, color_col, size_col, size_label, cell_groups,
                    left=left, top=top, max_area=_dot_area(size, _SQUARE_PANEL),
                    outer_label=outer_label)
    _finish(fig, save)




# --- one cell pair, every FOV that has it ----------------------------------

_PAIR_WIDTH = 2.4           # inches per cell pair
_PAIR_HEIGHT = 2.6


def plot_pair_dots(panels, counts, stage_order, stage_colors, num_cols=3,
                   value_col="log2_lift", ylabel="log2 lift\n(0 = chance)",
                   baseline=0.0, footnote=None, title=None, subtitle=None, save=None):
    """One panel per cell pair: every FOV carrying the rule as its own dot.

    panels : {pair name: table with 'stage' and 'log2_lift'}, one row per FOV.
    counts : {pair name: {stage: (FOVs with the rule, eligible FOVs)}}, written
             under each stage so a panel says how common the rule is as well as how
             strong it is.
    """
    if not panels:
        print("No data to plot.")
        return
    num_cols = max(1, min(num_cols, len(panels)))
    num_rows = (len(panels) + num_cols - 1) // num_cols
    fig, axes = plt.subplots(num_rows, num_cols, squeeze=False, sharey=True,
                             figsize=(_PAIR_WIDTH * num_cols + 1.4,
                                      _PAIR_HEIGHT * num_rows + 1.6))
    axes = axes.flatten()
    spread = np.random.default_rng(0)

    for ax, (name, table) in zip(axes, panels.items()):
        if baseline is not None:
            ax.axhline(baseline, color="0.55", lw=1.0, zorder=1)
        for x, stage in enumerate(stage_order):
            values = table.loc[table["stage"] == stage, value_col].to_numpy(dtype=float)
            if len(values):
                ax.scatter(x + spread.uniform(-0.16, 0.16, len(values)), values,
                           s=26, color=stage_colors.get(stage, "0.5"),
                           edgecolor="0.25", linewidth=0.4, alpha=0.85, zorder=3)
                middle = np.median(values)
                ax.plot([x - 0.28, x + 0.28], [middle] * 2, color="0.15", lw=1.6, zorder=4)
                ax.annotate(f"{middle:.2f}", (x + 0.30, middle), fontsize=7,
                            color="0.15", va="center", ha="left", zorder=5)
        ax.set_xticks(range(len(stage_order)))
        ax.set_xticklabels(["{}\n{}/{}".format(stage, *counts[name].get(stage, (0, 0)))
                            for stage in stage_order], fontsize=9)
        ax.set_xlim(-0.6, len(stage_order) - 0.4)
        ax.set_title(name, fontsize=10)
        ax.spines[["top", "right"]].set_visible(False)
        ax.tick_params(labelsize=9)

    for ax in axes[len(panels):]:
        ax.set_visible(False)
    for ax in axes[::num_cols]:
        ax.set_ylabel(ylabel, fontsize=9)

    top = _figure_titles(fig, title, subtitle, center=0.5)
    fig.tight_layout(rect=[0, 0.03, 1, top])
    fig.text(0.5, 0.005, footnote or "each dot is one FOV  |  bar = median  |  "
             "numbers = FOVs with the rule / FOVs holding both cell types",
             ha="center", fontsize=8.5, color="0.35")
    _finish(fig, save)
