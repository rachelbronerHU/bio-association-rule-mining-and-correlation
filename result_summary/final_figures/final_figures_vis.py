"""Drawing for the final figures. Shared pieces come from `vis_helper.py`."""
from itertools import combinations
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.cm import ScalarMappable
from matplotlib.colors import LinearSegmentedColormap, Normalize, to_hex, to_rgb
from matplotlib.lines import Line2D
from matplotlib.patches import Circle, FancyArrowPatch, Patch, Polygon, Rectangle
from matplotlib.ticker import MaxNLocator
from matplotlib.transforms import Bbox, blended_transform_factory, offset_copy

from vis_helper import (_category_colors, _cell_handles, save_figure, rule_name, INK, GRID, ZERO, UP_ARROW,
                        DOWN_ARROW, KIND_COLORS)

_FIGURE_DIR = Path(__file__).resolve().parent / "summary_downloads"
_KEY_DIR = _FIGURE_DIR / "keys"
PANEL_INCHES = 2.5

EDGE_LAYER = {"avoids": 1, "attracts": 2}
PREVALENCE, PREVALENCE_CMAP = Normalize(0, 100), "YlOrRd"
PC_TAG_MAPS = ("Blues", "YlOrBr")
EDGE_HEADS = {"attracts": "-|>,head_length=0.6,head_width=0.4", "avoids": "-[,widthB=0.6,lengthB=0"}
FADED = 0.2
BACKDROP, BACKDROP_SIZE = "#f0efeb", 0.5
PALE, PALE_GREY = 0.3, "#c9c8c2"
TREND_GAP = 0.4
RULE_SIZE = 2.5
KEY_ROW = 0.3
CLASS_COLORS = {
    "Immune–Immune": "#E3DCF2",
    "Immune–Epithel": "#FCE0C8",
    "Immune–Stroma": "#D3EDDC",
    "Tissue structure": "#F6EDC4",
}
STRIPE = 0.25
CLASS_TABS = {
    "Immune–Immune": "#9B8AC4",
    "Immune–Epithel": "#E3A87A",
    "Immune–Stroma": "#7DBF95",
    "Tissue structure": "#D6BE6A",
}


def _fonts(size):
    return {"font.size": size, "xtick.labelsize": size, "ytick.labelsize": size,
            "legend.fontsize": size, "pdf.fonttype": 42}


def _names_left(ax, aligned=()):
    """The left edge of the rule names of `ax` (and of the `aligned` axes), in `ax`'s axes units."""
    ax.get_figure(root=True).draw_without_rendering()
    left = min(label.get_window_extent().x0 for each in [ax, *aligned]
               for label in each.get_yticklabels() if label.get_text())
    return ax.transAxes.inverted().transform((left, 0))[0]


def _class_bands(ax, centers, classes, aligned=(), right=1):
    """A soft band behind each row, from the left edge of its names (and those of the `aligned` axes)
    to `right` in the axes' units (its right edge; 0 for its left, behind the names only), at `centers` in
    data units, coloured by its rule's class (CLASS_COLORS)."""
    left = _names_left(ax, aligned)
    across = blended_transform_factory(ax.transAxes, ax.transData)
    for center, name in zip(centers, classes):
        ax.add_patch(Rectangle((left, center - 0.43), right - left, 0.86, transform=across, lw=0, zorder=0,
                               color=CLASS_COLORS.get(name, "white"), clip_on=False))


def _class_tabs(ax, centers, classes):
    """A small square in the space left at the start of each rule's name, at `centers` in data units,
    in its class's CLASS_TABS colour."""
    size = plt.rcParams["font.size"]
    at = offset_copy(blended_transform_factory(ax.transAxes, ax.transData), ax.figure, x=0.5 * size, units="points")
    ax.scatter([_names_left(ax)] * len(centers), list(centers), s=(0.7 * size) ** 2, marker="s", linewidths=0,
               c=[CLASS_TABS.get(name, "white") for name in classes], transform=at, clip_on=False)


def save_class_key(font_size, colors=CLASS_COLORS, name="rule_classes"):
    """A square per rule class in its `colors` colour, the bands' by default, saved as keys/<name>."""
    _save_key([Patch(color=color, label=group) for group, color in colors.items()], font_size, name)


def _fresh_folder(save):
    """The folder `save` in summary_downloads, emptied of its PDFs."""
    folder = _FIGURE_DIR / save
    for old in folder.glob("*.pdf"):
        old.unlink()
    return folder


def _save_boxes(fig, boxes, folder):
    """`fig` cut to each box of `boxes` ({name: box in display units}), saved as folder/<name>.pdf,
    saying where."""
    for name, box in boxes.items():
        fig.savefig(folder / f"{name}.pdf", bbox_inches=box.transformed(fig.dpi_scale_trans.inverted()))
    print(f"saved {', '.join(boxes)} in {folder}")


def _save_parts(fig, parts, folder, name, piece):
    """`fig` saved as folder/<name>, and each part (axes or subfigure) cut out on its own
    as <piece>_1, <piece>_2 …"""
    save_figure(fig, name, figure_dir=folder)
    _save_boxes(fig, {f"{piece}_{n}": part.get_tightbbox() for n, part in enumerate(parts, 1)}, folder)


def _save_row(fig, parts, save, piece):
    """Empty the folder `save` of its PDFs, then save the figure and its parts in it."""
    folder = _fresh_folder(save)
    _save_parts(fig, parts, folder, save, piece)
    return folder


def _save_key(handles, font_size, name, folder=_KEY_DIR):
    """A legend of `handles`, saved as folder/<name>.pdf, by default in summary_downloads/keys."""
    with plt.rc_context(_fonts(font_size)):
        fig, ax = plt.subplots(figsize=(1, 1))
        ax.legend(handles=handles, loc="center", frameon=False)
        ax.axis("off")
        save_figure(fig, name, figure_dir=folder)
    plt.show()


def pick_fovs(fovs, n, ends=False):
    """`n` FOVs spread evenly along `fovs`, or with `ends` the first `n` and the last `n`."""
    spots = (np.r_[:n, len(fovs) - n:len(fovs)] if ends
             else np.linspace(0, len(fovs) - 1, n).round().astype(int))
    return fovs[spots]


def key_cells(cells, fovs, rules, n):
    """The `n` most common cell types of each FOV in `fovs`, together, and any other type a rule
    of those FOVs names ('center', 'neighbor'); most common first."""
    shown = cells.loc[cells["fov"].isin(fovs) & cells["cell type"].notna(), ["fov", "cell type"]].astype(str)
    top = shown.groupby("fov")["cell type"].value_counts().groupby(level=0).head(n)
    named = rules.loc[rules["FOV"].isin(fovs), ["center", "neighbor"]].stack()
    keep = set(top.index.get_level_values("cell type")) | set(named)
    return [cell for cell in shown["cell type"].value_counts().index if cell in keep]


def _fov_map(ax, cells, colors, dot, scale_um, scale_font, colored=None):
    """Cells by type, with a `scale_um` bar when given; with `colored`, only those types in colour,
    RULE_SIZE times the dot, the rest in BACKDROP underneath, BACKDROP_SIZE times the dot."""
    size = cells["x_um"].max()
    types = cells["cell type"].astype(str)
    color, scale = types.map(colors), 1
    if colored is not None:
        kept = types.isin(colored)
        order = kept.argsort(kind="stable")
        cells, color = cells.iloc[order], color.where(kept, BACKDROP).iloc[order]
        scale = np.where(kept, RULE_SIZE, BACKDROP_SIZE)[order]
    ax.scatter(cells["x_um"], cells["y_um"], s=dot * (400 / size) ** 2 * scale, linewidths=0, c=color.tolist())
    if scale_um:
        ax.plot([size - scale_um, size], [size, size], color=INK, lw=4, solid_capstyle="butt")
        ax.annotate(f"{scale_um} µm", (size, size), xytext=(0, 4), textcoords="offset points",
                    ha="right", va="bottom", color=INK, fontsize=scale_font)
    ax.set(xticks=[], yticks=[], aspect="equal")
    ax.invert_yaxis()
    for spine in ax.spines.values():
        spine.set(color=GRID, linewidth=1)


def fov_tags(n):
    """One colour per marked FOV, dark for the first, light for the last."""
    return [to_hex(color) for color in plt.cm.viridis(np.linspace(0, 0.9, n))]


def pc_tags(n):
    """Per component, `n` tag colours from one family of PC_TAG_MAPS, light for the low end, dark for the high."""
    return [[to_hex(color) for color in plt.get_cmap(name)(np.linspace(0.4, 1, n))] for name in PC_TAG_MAPS]


def stage_tags(stages, n):
    """`n` tag colours moving from the first of `stages`' PCA colour to the last's."""
    colors, order = _category_colors(stages)
    blend = LinearSegmentedColormap.from_list("stages", [colors[order[0]], colors[order[-1]]])
    return [to_hex(color) for color in blend(np.linspace(0, 1, n))]


def _tag(ax, color):
    """A filled triangle in the top left corner, the FOV's tag colour."""
    ax.add_patch(Polygon([(0, 1), (0.15, 1), (0, 0.85)], closed=True, color=color, lw=0,
                         transform=ax.transAxes, clip_on=False, zorder=4))


def _row(n, width, gap_after=None):
    """A figure with a row of `n` square axes `width` long, a blank half-size space after the first
    `gap_after` of them, and how much larger a panel is than PANEL_INCHES."""
    ratios = [1] * n if gap_after is None else [1] * gap_after + [0.5] + [1] * (n - gap_after)
    fig, axes = plt.subplots(1, len(ratios), layout="constrained", squeeze=False,
                             figsize=(width, width / sum(ratios)), width_ratios=ratios)
    axes = list(axes.ravel())
    if gap_after is not None:
        axes.pop(gap_after).axis("off")
    return fig, axes, width / sum(ratios) / PANEL_INCHES


def plot_rules_per_fov(counts, marked, colors, width, height, font_size, spans=(), stripes=None, save=None):
    """Stacked bar of rules per FOV, one colour per column of `counts`, and under each marked FOV
    a dot in its tag colour; no dots when `marked` is empty.

    spans   : (FOVs, colour) pairs; under the bars of each, a line in its colour.
    stripes : {name: group per FOV}; under the bars, a named stripe per entry, each FOV in its group's
              PCA colour, grey when it has none.
    save  : the figure's name; the bars and the y axis are also cut out on their own beside it,
            as <save>_bars and <save>_y_axis, at the same height.
    """
    with plt.rc_context(_fonts(font_size)):
        stripes = stripes or {}
        fig, axes = plt.subplots(1 + len(stripes), 1, layout="constrained", sharex=True, squeeze=False,
                                 figsize=(width, height + STRIPE * len(stripes)),
                                 height_ratios=[height] + [STRIPE] * len(stripes))
        ax, *strips = axes[:, 0]
        for strip, (name, groups) in zip(strips, stripes.items()):
            shades, _ = _category_colors(groups)
            strip.bar(range(len(counts)), 1, width=1, linewidth=0,
                      color=groups.reindex(counts.index).map(shades).fillna(PALE_GREY).tolist())
            strip.set_yticks([0.5], [name])
            strip.tick_params(length=0)
            strip.spines[:].set_visible(False)
        counts.plot.bar(stacked=True, width=1, linewidth=0, color=colors, legend=False, ax=ax)
        if len(marked):
            ax.scatter([counts.index.get_loc(fov) for fov in marked], [0] * len(marked), s=150,
                       c=fov_tags(len(marked)), clip_on=False, zorder=3, transform=ax.get_xaxis_transform())
        for fovs, color in spans:
            spots = [counts.index.get_loc(fov) for fov in fovs]
            ax.plot([min(spots) - 0.5, max(spots) + 0.5], [-0.02, -0.02], color=color, lw=8,
                    solid_capstyle="butt", clip_on=False, transform=ax.get_xaxis_transform())
        ax.set(xticks=[], xlabel="" if stripes else "FOVs", ylabel="Rules", xlim=(-0.5, len(counts) - 0.5))
        (strips or [ax])[-1].set(xticks=[], xlabel="FOVs")
        ax.yaxis.set_major_locator(MaxNLocator(3))
        ax.spines[["top", "right"]].set_visible(False)
    if save:
        save_figure(fig, save, figure_dir=_FIGURE_DIR)
        whole, left = Bbox.union([part.get_tightbbox() for part in [ax, *strips]]), ax.get_window_extent().x0
        _save_boxes(fig, {f"{save}_bars": Bbox([[left, whole.y0], [whole.x1, whole.y1]]),
                          f"{save}_y_axis": Bbox([[whole.x0, whole.y0], [left, whole.y1]])}, _FIGURE_DIR)
    plt.show()


def save_significance_key(colors, labels, font_size):
    """The bar colours of `plot_rules_per_fov` with their `labels`, saved as keys/significance."""
    _save_key([Patch(color=c, label=l) for c, l in zip(colors, labels)], font_size, "significance")


def plot_fov_maps(cells, fovs, cell_colors, key, width, font_size, dot=10, scale_um=100, scale_bars=0,
                  scale_font=None, gap_after=None, tags=None, save=None):
    """A row of FOV maps, cells coloured by type, the first `scale_bars` with a `scale_um` bar.

    key  : the cell types named in the cell type key, in its order; no key when None.
    dot  : cell dot area on a 400 um map of PANEL_INCHES; larger maps get smaller dots.
    scale_font : font size of the scale bar's label; `font_size` when None.
    gap_after : a blank space after this many maps; none when None.
    tags : the corner triangle colours, one per map; `fov_tags` when None, no triangles when empty.
    save : a folder in summary_downloads, emptied of its PDFs, then given the row,
           each map on its own (map_1, map_2 …) and the cell type key.
    """
    tags = fov_tags(len(fovs)) if tags is None else tags
    with plt.rc_context(_fonts(font_size)):
        fig, axes, scale = _row(len(fovs), width, gap_after)
        for n, (fov, ax) in enumerate(zip(fovs, axes)):
            _fov_map(ax, cells[cells["fov"].eq(fov)], cell_colors, dot * scale ** 2,
                     scale_um if n < scale_bars else None, scale_font)
            if tags:
                _tag(ax, tags[n])
    if save:
        _save_row(fig, axes, save, "map")
        if key is not None:
            save_cell_key(key, cell_colors, font_size, save)
    plt.show()


def save_cell_key(key, cell_colors, font_size, save):
    """The cell type key, the types of `key` in its order, saved as <save>/cells.pdf in summary_downloads."""
    handles = _cell_handles(key, cell_colors)
    for handle in handles:
        handle.set_markersize(0.8 * font_size)
    _save_key(handles, font_size, "cells", _FIGURE_DIR / save)


def _crossings(order, pairs):
    """How many lines cross when the nodes sit on a circle in `order`, one line per pair."""
    spot = {node: i for i, node in enumerate(order)}
    a, b = np.sort([[spot[x], spot[y]] for x, y in pairs], axis=1).T
    inside = lambda x: (a[:, None] < x) & (x < b[:, None])
    shared = (a[:, None] == a) | (a[:, None] == b) | (b[:, None] == a) | (b[:, None] == b)
    return np.sum((inside(a) ^ inside(b)) & ~shared) // 2


def _circle_order(nodes, pairs):
    """`nodes` reordered, two at a time, while that leaves fewer crossing lines."""
    order = sorted(nodes)
    best = _crossings(order, pairs) if pairs else 0
    improved = best > 0
    while improved:
        improved = False
        for i, j in combinations(range(len(order)), 2):
            order[i], order[j] = order[j], order[i]
            count = _crossings(order, pairs)
            if count < best:
                best, improved = count, True
            else:
                order[i], order[j] = order[j], order[i]
    return order


def _network(ax, rules, colors, node, line, strongest, shown=None):
    """One FOV's rules on a circle of cell types: centre to neighbour, → attracts, ⊣ avoids,
    thicker for a stronger |log2 lift|. A rule of a type with itself is a small loop.
    With `shown` (centre, neighbour), the other rules are faded behind it."""
    nodes = set(rules["center"]) | set(rules["neighbor"])
    pairs = {tuple(sorted(pair)) for pair in zip(rules["center"], rules["neighbor"]) if pair[0] != pair[1]}
    order = _circle_order(nodes, list(pairs))
    angles = np.pi / 2 - 2 * np.pi * np.arange(len(order)) / max(len(order), 1)
    xy = dict(zip(order, np.c_[np.cos(angles), np.sin(angles)]))
    strength = np.log2(rules["Lift"].clip(lower=2.0 ** -strongest)).abs().clip(upper=strongest)
    for rule, lw in zip(rules.itertuples(), line * strength):
        color, layer = KIND_COLORS[rule.Kind], EDGE_LAYER[rule.Kind]
        alpha, layer = (FADED, 0) if shown and (rule.center, rule.neighbor) != shown else (1, layer)
        if rule.center == rule.neighbor:
            ax.add_patch(Circle(1.1 * xy[rule.center], 0.1, fill=False, color=color, lw=lw, alpha=alpha,
                                zorder=layer))
        else:
            ax.add_patch(FancyArrowPatch(xy[rule.center], xy[rule.neighbor], arrowstyle=EDGE_HEADS[rule.Kind],
                                         connectionstyle="arc3,rad=0.2", mutation_scale=5 * line,
                                         shrinkA=node ** 0.5, shrinkB=node ** 0.5, color=color, lw=lw,
                                         alpha=alpha, zorder=layer))
    if order:
        ax.scatter(*np.array([xy[cell] for cell in order]).T, s=node, linewidths=0, zorder=3,
                   c=[colors[cell] for cell in order])
    ax.set(xticks=[], yticks=[], aspect="equal", xlim=(-1.25, 1.25), ylim=(-1.25, 1.25))
    ax.spines[:].set_visible(False)


def save_edge_key(font_size, line=1.5, strongest=3):
    """→ attracts and ⊣ avoids in their colours, then the line width for each |log2 lift|, as on a
    network of PANEL_INCHES drawn with `line` and `strongest`; saved as keys/edges."""
    rows = [(kind, KIND_COLORS[kind], EDGE_HEADS[kind], line * strongest) for kind in KIND_COLORS]
    rows += [(f"|log2 lift| {s}", INK, "-", line * s) for s in (1, 2, strongest)]
    with plt.rc_context(_fonts(font_size)):
        fig, ax = plt.subplots(figsize=(4, 0.5 * len(rows)))
        for y, (label, color, head, lw) in enumerate(rows):
            ax.add_patch(FancyArrowPatch((0, -y), (1, -y), arrowstyle=head, mutation_scale=5 * line,
                                         color=color, lw=lw))
            ax.text(1.3, -y, label, va="center")
        ax.set(xlim=(0, 1.3), ylim=(0.5 - len(rows), 0.5))
        ax.axis("off")
        save_figure(fig, "edges", figure_dir=_KEY_DIR)
    plt.show()


def plot_rule_networks(rules, fovs, cell_colors, width, font_size, node=150, line=1.5, strongest=3,
                       gap_after=None, tags=None, save=None):
    """A row of rule networks, one per FOV, drawn by `_network`; the nodes of each are ordered
    on their circle so as few lines cross as can be found.

    rules     : one row per rule, with 'FOV', 'center', 'neighbor', 'Kind' and 'Lift'.
    node      : node dot area; line : line width per unit of |log2 lift|, which stops at `strongest`;
                both for a network of PANEL_INCHES.
    gap_after : a blank space after this many networks; none when None.
    tags      : the corner triangle colours, one per network; `fov_tags` when None, no triangles when empty.
    save      : a folder in summary_downloads, emptied of its PDFs, then given the row and each
                network on its own (network_1 …).
    """
    tags = fov_tags(len(fovs)) if tags is None else tags
    with plt.rc_context(_fonts(font_size)):
        fig, axes, scale = _row(len(fovs), width, gap_after)
        node, line = node * scale ** 2, line * scale
        for n, (fov, ax) in enumerate(zip(fovs, axes)):
            _network(ax, rules[rules["FOV"].eq(fov)], cell_colors, node, line, strongest)
            if tags:
                _tag(ax, tags[n])
    if save:
        _save_row(fig, axes, save, "network")
    plt.show()


def _shade(color, score):
    """`color` per score: at 0 pale, PALE of it on PALE_GREY; at 1 the colour itself."""
    color, score = np.array(to_rgb(color)), np.asarray(score, float)[:, None]
    pale = PALE * color + (1 - PALE) * np.array(to_rgb(PALE_GREY))
    return pale + (color - pale) * score


def _rule_map(ax, cells, score, colors, dot, scale_um):
    """The FOV in BACKDROP, and the cells with a `score` RULE_SIZE times the dot in their type's colour,
    pale when not counted for the rule, the counted on top."""
    _fov_map(ax, cells, colors, dot, scale_um, None, colored=())
    scored = cells.assign(score=score).dropna(subset="score").sort_values("score")
    shades = [_shade(colors[cell_type], [value])[0]
              for cell_type, value in zip(scored["cell type"].astype(str), scored["score"])]
    ax.scatter(scored["x_um"], scored["y_um"], s=RULE_SIZE * dot * (400 / cells["x_um"].max()) ** 2,
               linewidths=0, c=shades)


def _rule_key(ax, rule, colors, across=False):
    """Per cell type of `rule`, centre on top (first when `across`, all on one line): its counted and not
    counted dot as in `_rule_map`, then its name."""
    size = plt.rcParams["font.size"]
    if across:
        step = size * (3.5 + 0.6 * max(map(len, rule)))
        for n, cell_type in enumerate(rule):
            for x, color in zip([0.5 * size, 1.7 * size], _shade(colors[cell_type], [1, 0])):
                ax.scatter([0], [0.5], s=(0.8 * size) ** 2, linewidths=0, c=[color],
                           transform=offset_copy(ax.transAxes, ax.figure, x=n * step + x, units="points"))
            ax.text(0, 0.5, cell_type, va="center",
                    transform=offset_copy(ax.transAxes, ax.figure, x=n * step + 2.7 * size, units="points"))
        ax.set(xlim=(0, 1), ylim=(0, 1))
        ax.axis("off")
        return
    for row, cell_type in enumerate(rule):
        ax.scatter([0, 1], [row, row], s=(0.8 * size) ** 2, linewidths=0, c=_shade(colors[cell_type], [1, 0]))
        ax.text(1.7, row, cell_type, va="center")
    ax.set(xlim=(-0.6, 6), ylim=(len(rule) + 0.5, -1.5))
    ax.axis("off")


def _pair_grid(n, sides, width, vertical):
    """A figure with a row of `n` square axes per side, `sides` rows (columns when `vertical`), and one
    key axes after them across all sides, `width` long, a square wide, or KEY_ROW of one high when
    `vertical`; the rows of axes, the key, and how much larger a panel is than PANEL_INCHES."""
    side = width / (n + 1)
    fig = plt.figure(layout="constrained",
                     figsize=(sides * side, (n + KEY_ROW) * side) if vertical else (width, sides * side))
    grid = (fig.add_gridspec(n + 1, sides, height_ratios=[1] * n + [KEY_ROW]) if vertical
            else fig.add_gridspec(sides, n + 1))
    rows = [[fig.add_subplot(grid[(at, row) if vertical else (row, at)]) for at in range(n)] for row in range(sides)]
    return fig, rows, fig.add_subplot(grid[n, :] if vertical else grid[:, n]), side / PANEL_INCHES


def plot_rule_examples(cells, examples, rules, cell_colors, key, width, font_size, network=True, dot=10,
                       node=150, line=1.5, strongest=3, scale_um=None, vertical=False, notes=None, save=None):
    """One panel per rule, titled by the rule in its kind's colour, a row per example FOV: the FOV's map,
    its map with only the rule's two cell types in colour, its rule map by `_rule_map`, with `network` its
    rule network with the rule in front; and one key of the rule map by `_rule_key` beside all rows
    (one line under them when `vertical`); a `scale_um` bar on every map when given;
    drawn as in `plot_fov_maps` and `plot_rule_networks`.

    examples : {name: ((centre, neighbour), kind, [(FOV, score, tag colour), ...])}, the score as
               `rule_examples_helper.rule_scores` gives it, the tag in the corner of the FOV's maps.
    rules    : as in `plot_rule_networks`.
    key      : the cell types named in the cell type key, in its order.
    vertical : each example as a column, top to bottom, `width` long, instead of a row.
    notes    : {name: text} printed above each panel; nothing when None.
    save     : a folder in summary_downloads, emptied of its PDFs, then given each panel named after its
               rule, each of its squares on its own (<name>_1 …) and the cell type key.
    """
    folder = _fresh_folder(save) if save else None
    for name, (rule, kind, sides) in examples.items():
        if notes:
            print(notes[name])
        with plt.rc_context(_fonts(font_size)):
            fig, rows, key_ax, scale = _pair_grid(3 + network, len(sides), width, vertical)
            for axes, (fov, score, tag) in zip(rows, sides):
                shown = cells[cells["fov"].eq(fov)]
                _fov_map(axes[0], shown, cell_colors, dot * scale ** 2, scale_um, None)
                _fov_map(axes[1], shown, cell_colors, dot * scale ** 2, scale_um, None, colored=rule)
                _rule_map(axes[2], shown, score, cell_colors, dot * scale ** 2, scale_um)
                if network:
                    _network(axes[3], rules[rules["FOV"].eq(fov)], cell_colors, node * scale ** 2,
                             line * scale, strongest, shown=rule)
                for ax in axes[:3]:
                    _tag(ax, tag)
            _rule_key(key_ax, rule, cell_colors, across=vertical)
            rows[0][0].set_title(rule_name(rule, kind), loc="left", color=KIND_COLORS[kind])
        if save:
            _save_parts(fig, [ax for axes in rows for ax in axes] + [key_ax], folder, name, name)
        plt.show()
    if save:
        save_cell_key(key, cell_colors, font_size, save)


def stars(fdr):
    return "***" if fdr <= 0.001 else "**" if fdr <= 0.01 else "*" if fdr <= 0.05 else ""


def _count_rows(ax, counts, color, heights=None, rising=None, unit="", ends=False):
    """One line per row across its columns, at `heights` (default `counts`), each point
    labelled with its count and `unit`, only the first and last with `ends`; with `rising` (one bool
    per row), an up or down arrow at the end."""
    x = np.arange(counts.shape[1])
    heights = counts if heights is None else heights
    for row, (line, height) in enumerate(zip(counts.to_numpy(float), heights.to_numpy(float))):
        y = row + 0.9 - 0.5 * height / max(np.nanmax(height), 1e-9)
        shown = ~np.isnan(y)
        ax.plot(x[shown], y[shown], color=color, lw=3, marker="o", markersize=8)
        for at in x[shown][[0, -1]] if ends else x[shown]:
            ax.annotate(f"{line[at]:.0f}{unit}", (at, y[at]), xytext=(0, 6), textcoords="offset points",
                        ha="center", va="bottom", color=color)
        if rising is not None:
            up = rising.iloc[row]
            ax.plot(x[-1] + 0.5, y[-1], "^" if up else "v", markersize=14,
                    color=UP_ARROW if up else DOWN_ARROW)
    ax.set(xlim=(-0.4, len(x) - 0.4 + 0.5 * (rising is not None)), ylim=(len(counts), 0),
           xticks=[], yticks=[])
    ax.spines[:].set_visible(False)


def _trend_width(groups, square, trend):
    """Width of a trend block in inches: two line plots, the squares, and TREND_GAP inches on each side."""
    return 2 * trend + groups * square + 2 * TREND_GAP


def _trend_block(sub, found, tested, fdr, kind, cell, square, trend, header, names=True, units=None,
                 unit=None):
    """The share of testable units with each rule per group, as a line and as squares, and the testable
    units per group; with `fdr`, its stars and an up or down arrow from the first group to the last; with
    `names`, the slanted group names above, each with its total from `units` when given; with `unit` (the
    unit's name), a header over each line plot. `cell` is a row's height, `square` a group's width, `trend` a
    line plot's width and `header` the space above the rules, in inches. Returns the axes of the rule names."""
    rows, groups = found.shape
    width = _trend_width(groups, square, trend)
    box = lambda left, size: sub.add_axes([left / width, 0, size / width, rows * cell / (rows * cell + header)])
    ax_found, ax_share = box(0, trend), box(trend + TREND_GAP, groups * square)
    ax_tested = box(trend + groups * square + 2 * TREND_GAP, trend)
    share = 100 * found / tested
    rising = None if fdr is None else share.iloc[:, -1] >= share.iloc[:, 0]
    _count_rows(ax_found, share, KIND_COLORS[kind], rising=rising, unit="%", ends=True)
    ax_found.set_yticks(np.arange(rows) + 0.5, [rule_name(rule, kind) for rule in found.index])
    ax_found.tick_params(length=0)
    ax_share.imshow(share, cmap=PREVALENCE_CMAP, norm=PREVALENCE, aspect="auto",
                    extent=(-0.5, groups - 0.5, rows, 0))
    for row in range(1, rows):
        ax_share.axhline(row, color="white", lw=4)
    for column in range(1, groups):
        ax_share.axvline(column - 0.5, color="white", lw=4)
    labels = share.columns.astype(str)
    if units is not None:
        labels = [f"{group} (n={units[column]})" for group, column in zip(labels, share.columns)]
    ax_share.set_xticks(range(groups), labels, rotation=45, ha="left", rotation_mode="anchor")
    ax_share.set_yticks(np.arange(rows) + 0.5, [] if fdr is None else fdr.map(stars))
    ax_share.tick_params(length=0, top=True, labeltop=names, bottom=False, labelbottom=False,
                         left=False, labelleft=False, right=True, labelright=True)
    ax_share.spines[:].set_visible(False)
    _count_rows(ax_tested, tested, ZERO)
    if unit and names:
        for ax, label in [(ax_found, f"{unit[0].upper()}{unit[1:]}s with rule"), (ax_tested, f"Testable {unit}s")]:
            ax.set_title(label, loc="left", fontsize="medium")
    return ax_found


def plot_rule_trend(title, blocks, font_size, cell=1.8, square=1.8, trend=3.5, extra_text=False, save=None):
    """One titled figure: blocks of rules across groups, one under the other, drawn by `_trend_block`,
    the group names above the first only.

    blocks : dicts with 'found' and 'tested' (rule x group counts), 'kind', 'unit', 'units' (total per
             group), and optionally 'fdr' (per rule) for stars and arrows and 'classes' (per rule) for bands.
    cell, square, trend : a row's height, a group's width and a line plot's width, in inches.
    extra_text : also a header over each line plot.
    save   : a folder in summary_downloads, given the figure, named after its title, and each of its
             blocks on its own (<name>_1, <name>_2 …), the first with the title, all as wide so they line up.
    """
    longest = max(len(str(group)) for group in blocks[0]["found"].columns) + 8 * ("units" in blocks[0])
    header = (1.5 + 0.4 * longest) * font_size / 72
    headers = [header] + [font_size / 72] * (len(blocks) - 1)
    heights = [3 * font_size / 72] + [len(block["found"]) * cell + h for block, h in zip(blocks, headers)]
    with plt.rc_context(_fonts(font_size)):
        fig = plt.figure(figsize=(_trend_width(blocks[0]["found"].shape[1], square, trend), sum(heights)))
        head, *subs = fig.subfigures(len(heights), 1, height_ratios=heights, squeeze=False)[:, 0]
        head.text(0.5, 0, title, ha="center", va="bottom", fontsize=1.2 * font_size)
        named = [_trend_block(sub, block["found"], block["tested"], block.get("fdr"), block["kind"], cell,
                              square, trend, h, names=n == 0, units=block.get("units"),
                              unit=block.get("unit") if extra_text else None)
                 for n, (sub, block, h) in enumerate(zip(subs, blocks, headers))]
        for ax, block in zip(named, blocks):
            if "classes" in block:
                _class_bands(ax, np.arange(len(block["found"])) + 0.5,
                             block["classes"].reindex(block["found"].index), named)
    if save:
        folder = _FIGURE_DIR / save
        name = title.replace(" | ", "_").replace(" ", "_")
        boxes = [sub.get_tightbbox() for sub in subs]
        boxes[0] = Bbox.union([head.get_tightbbox(), boxes[0]])
        left, right = min(box.x0 for box in boxes), max(box.x1 for box in boxes)
        boxes = [Bbox([[left, box.y0], [right, box.y1]]) for box in boxes]
        save_figure(fig, name, figure_dir=folder)
        _save_boxes(fig, {f"{name}_{n}": box for n, box in enumerate(boxes, 1)}, folder)
    plt.show()


def save_prevalence_key(unit, font_size):
    """The colour scale of `plot_rule_trend`, as % of testable `unit`s, saved as keys/prevalence_<unit>."""
    with plt.rc_context(_fonts(font_size)):
        key, ax = plt.subplots(figsize=(6, 0.4))
        key.colorbar(ScalarMappable(PREVALENCE, PREVALENCE_CMAP), cax=ax, orientation="horizontal",
                     label=f"% of testable {unit}s")
        save_figure(key, f"prevalence_{unit}", figure_dir=_KEY_DIR)
    plt.show()


def plot_pca(coords, variance, stage, marked, pair, width, font_size, line=None, title=None, save=None):
    """FOVs on PC1 and PC2, coloured by their `stage`, a ring in its tag colour around each FOV
    of `marked` ({FOV: colour}), a box around the two FOVs of `pair`; none when empty; a
    dashed `line` between its two (PC1, PC2) ends; and `title` above.

    save : a folder in summary_downloads, emptied of its PDFs, then given the PCA.
    """
    with plt.rc_context(_fonts(font_size)):
        fig, ax = plt.subplots(layout="constrained", figsize=(width / 2, width / 2))
        _pca(ax, coords, variance, stage, marked, pair, line)
        ax.set_title(title)
    if save:
        save_figure(fig, "pca", figure_dir=_fresh_folder(save))
    plt.show()


def plot_pca_pair(pcas, stage, width, font_size, marked={}, pair=(), line=None, save=None):
    """PCAs of the same FOVs side by side, coloured by their `stage`, each titled by its name; the
    first with `marked`, `pair` and `line` as in `plot_pca`.

    pcas : {name: (coords, variance)}.
    save : a folder in summary_downloads, emptied of its PDFs, then given the PCAs together and
           each on its own (pca_1, pca_2 …).
    """
    with plt.rc_context(_fonts(font_size)):
        fig, axes, _ = _row(len(pcas), width)
        for i, (ax, (name, (coords, variance))) in enumerate(zip(axes, pcas.items())):
            _pca(ax, coords, variance, stage, *((marked, pair, line) if i == 0 else ()))
            ax.set_title(name)
    if save:
        _save_row(fig, axes, save, "pca")
    plt.show()


def _pca(ax, coords, variance, stage, marked={}, pair=(), line=None):
    colors, _ = _category_colors(coords[stage])
    at = coords.set_index("FOV")[["PC1", "PC2"]]
    ax.axhline(0, color=GRID, lw=1, zorder=0)
    ax.axvline(0, color=GRID, lw=1, zorder=0)
    ax.scatter(at["PC1"], at["PC2"], s=150, c=coords[stage].map(colors).tolist(), linewidths=0)
    ax.scatter(at.loc[list(marked), "PC1"], at.loc[list(marked), "PC2"], s=320, facecolors="none",
               edgecolors=list(marked.values()), linewidths=2.5)
    if len(pair):
        low, high = at.loc[pair].min().to_numpy(), at.loc[pair].max().to_numpy()
        pad = 0.02 * np.ptp(at.to_numpy(), axis=0).max()
        ax.add_patch(Rectangle(low - pad, *(high - low + 2 * pad), fill=False, edgecolor=INK, lw=1.5,
                               ls=(0, (4, 2))))
    if line is not None:
        ax.plot(*np.transpose(line), color=INK, lw=1.5, ls=(0, (4, 2)), zorder=0)
    ax.set(xlabel=f"PC1 ({variance[0]:.0f}%)", ylabel=f"PC2 ({variance[1]:.0f}%)", xticks=[], yticks=[])
    ax.spines[["top", "right"]].set_visible(False)


def save_stage_key(stages, font_size, name="stages"):
    """A dot per stage of `stages` in its PCA colour, saved as keys/<name>."""
    colors, order = _category_colors(stages)
    save_group_key({stage: colors[stage] for stage in order}, font_size, name)


def save_group_key(colors, font_size, name, markers={}):
    """A dot per group of `colors` ({group: colour}) with its name, saved as keys/<name>; a group of
    `markers` ({group: marker}) gets that marker instead of a dot."""
    _save_key([Line2D([], [], marker=markers.get(group, "o"), ls="", color=color, markersize=0.8 * font_size,
                      label=group)
               for group, color in colors.items()], font_size, name)


def plot_loadings(weights, rules, n, width, font_size, classes=None, kinds=None, save=None):
    """Per component, a row of `weights` over `rules`, its `n` rules of lowest and `n` of highest
    weight as bars, side by side, each in its rule's kind colour (with `kinds`) or else in that component's
    `pc_tags` family, dark for positive and light for negative; above its two ends, the way those rules move FOVs on the PCA (PC1 left / right, PC2 down / up).

    classes : the class per rule, a square before each rule's name; none when None.
    kinds   : the kind per rule, its bar in that kind's colour and its name with its arrow; → when None.

    save : a folder in summary_downloads, emptied of its PDFs, then given the components together
           and each on its own (pc_1, pc_2 …).
    """
    rules = np.asarray(rules).astype(str)
    kinds = {} if kinds is None else kinds
    names = np.array([("    " if classes is not None else "") + rule_name(rule, kinds.get(rule, "attracts"))
                      for rule in rules])
    orders = [np.argsort(weight) for weight in weights]
    picks = [np.r_[order[:n], order[-n:]] if len(order) > 2 * n else order for order in orders]
    with plt.rc_context(_fonts(font_size)):
        fig, axes = plt.subplots(1, len(weights), layout="constrained", figsize=(width, n + 1))
        for number, (ax, weight, shown, (light, *_, dark)) in enumerate(zip(axes, weights, picks, pc_tags(2)), 1):
            shades = np.where(weight[shown] > 0, dark, light)
            ax.barh(names[shown], weight[shown],
                    color=[KIND_COLORS.get(kinds.get(rule), shade) for rule, shade in zip(rules[shown], shades)])
            ax.axvline(0, color=INK, lw=1)
            ax.set(xlabel=f"PC{number} weight")
            low, high = ("left", "right") if number == 1 else ("down", "up")
            ax.text(0, 1, f"← PC{number} {low}", transform=ax.transAxes, ha="left", va="bottom")
            ax.text(1, 1, f"PC{number} {high} →", transform=ax.transAxes, ha="right", va="bottom")
            ax.xaxis.set_major_locator(MaxNLocator(3))
            ax.tick_params(axis="y", length=0)
            ax.spines[["top", "right", "left"]].set_visible(False)
        _names_to_left(fig, axes)
        for ax, shown in zip(axes, picks) if classes is not None else []:
            _class_tabs(ax, range(len(shown)), classes.reindex(rules[shown]))
    if save:
        _save_row(fig, axes, save, "pc")
    plt.show()


def _names_to_left(fig, axes):
    """The y labels of each of `axes` moved left of its widest one and aligned on their left; the layout then frozen."""
    fig.draw_without_rendering()
    for ax in axes:
        widest = max(label.get_window_extent().width for label in ax.get_yticklabels()) * 72 / fig.dpi
        ax.tick_params(axis="y", pad=widest + 6)
        plt.setp(ax.get_yticklabels(), ha="left")
    fig.draw_without_rendering()
    fig.set_layout_engine("none")


def plot_both_groups(runs, references, best_color, width, font_size, save=None):
    """A row per predictor in its colour, best first: the best rule run in `best_color` and in bold, and the best
    run of each reference. Left, the % right of each group joined by a line: filled for the first, hollow for the
    other. Right, their average as a bar from 50% (guessing).

    runs       : the runs of one prediction and unit, as saved by run_predictions.py.
    references : {reference rule set: (its name, its colour)}.
    save       : a folder in summary_downloads, emptied of its PDFs, then given the plot, each panel on its own
                 (panel_1, panel_2) and the key of the two groups.
    """
    rules = runs[runs["min_patients"].notna()]
    best = rules.loc[rules["balanced_accuracy"].idxmax()]
    tops = [(name, runs.loc[runs["rule_set"].eq(reference), "balanced_accuracy"].idxmax(), color)
            for reference, (name, color) in references.items()]
    rows = sorted([("Rules", best.name, best_color), *tops],
                  key=lambda row: runs.loc[row[1], "balanced_accuracy"], reverse=True)
    names, picked, colors = zip(*rows)
    shown, y = runs.loc[list(picked)], np.arange(len(rows))
    weights = ["bold" if name == "Rules" else "normal" for name in names]
    with plt.rc_context(_fonts(font_size)):
        fig, (sides, average) = plt.subplots(1, 2, sharey=True, layout="constrained",
                                             figsize=(width, 0.9 * len(rows) + 1.5))
        sides.axvline(50, color=INK, lw=1.5, ls=(0, (4, 2)), zorder=0)
        sides.hlines(y, shown[["positive_right", "negative_right"]].min(axis=1),
                     shown[["positive_right", "negative_right"]].max(axis=1), colors=colors, lw=3, zorder=1)
        sides.scatter(shown["positive_right"], y, s=400, color=colors, zorder=2, clip_on=False)
        sides.scatter(shown["negative_right"], y, s=400, facecolors="white", edgecolors=colors, linewidths=3,
                      zorder=2, clip_on=False)
        sides.set(xlim=(-3, 103), xticks=[0, 50, 100], xlabel="Each group right (%)", yticks=y, yticklabels=names)
        sides.invert_yaxis()
        average.barh(y, shown["balanced_accuracy"] - 50, left=50, height=0.4, color=colors)
        for at, value, color, weight in zip(y, shown["balanced_accuracy"], colors, weights):
            average.annotate(f"{value:.0f}", (value, at), xytext=(8, 0), textcoords="offset points", va="center",
                             color=color, fontweight=weight)
        average.set(xlim=(50, 100), xticks=[50, 75, 100], xlabel="% right (average)")
        for label, weight in zip(sides.get_yticklabels(), weights):
            label.set_fontweight(weight)
        for ax in (sides, average):
            ax.grid(axis="x", color=GRID, lw=1)
            ax.set_axisbelow(True)
            ax.tick_params(axis="y", length=0)
            ax.spines[["top", "right", "left"]].set_visible(False)
    if save:
        folder = _save_row(fig, [sides, average], save, "panel")
    plt.show()
    if save:
        _save_key([Line2D([], [], marker="o", ls="", markersize=0.8 * font_size, color=INK, label=best["positive"]),
                   Line2D([], [], marker="o", ls="", markersize=0.8 * font_size, mfc="white", mec=INK, mew=3,
                          label=best["negative"])], font_size, "groups", folder)


def plot_rule_weights(weights, kinds, sides, width, font_size, classes=None, save=None):
    """`plot_weights` for rules: each rule named with its kind's arrow and in its kind's colour.

    kinds   : the kind per rule.
    classes : the class per rule, a band behind its name as in `plot_rule_trend`; none when None.
    """
    names = {rule: rule_name(rule, kinds[rule]) for rule in weights.index}
    plot_weights(weights.rename(names), {names[rule]: KIND_COLORS[kind] for rule, kind in kinds.items()},
                 sides, width, font_size, None if classes is None else classes.rename(names), save)


def plot_weights(weights, colors, sides, width, font_size, classes=None, save=None):
    """A bar per name of `weights` (name → weight), lowest at the bottom, in its colour of `colors`; to the right
    pushes toward the second of `sides` (the two groups' names), to the left toward the first.

    classes : the class per name, a band behind it as in `plot_rule_trend`; none when None.
    save    : a folder in summary_downloads, emptied of its PDFs, then given the bars.
    """
    weights = weights.sort_values()
    with plt.rc_context(_fonts(font_size)):
        fig, ax = plt.subplots(layout="constrained", figsize=(width / 2, 0.5 * len(weights) + 1))
        ax.barh(weights.index, weights, color=[colors[name] for name in weights.index])
        ax.axvline(0, color=INK, lw=1)
        ax.set(xlabel="Weight")
        at_zero = blended_transform_factory(ax.transData, ax.transAxes)
        ax.text(0, 1, f"← {sides[0]}  ", transform=at_zero, ha="right", va="bottom")
        ax.text(0, 1, f"  {sides[1]} →", transform=at_zero, ha="left", va="bottom")
        ax.xaxis.set_major_locator(MaxNLocator(3))
        ax.tick_params(axis="y", length=0)
        ax.spines[["top", "right", "left"]].set_visible(False)
        _names_to_left(fig, [ax])
        if classes is not None:
            _class_bands(ax, range(len(weights)), classes.reindex(weights.index), right=0)
    if save:
        save_figure(fig, "weights", figure_dir=_fresh_folder(save))
    plt.show()


def plot_permutation(shuffled, real, p, label, run_color, best_color, width, font_size, save=None):
    """The best score of each search on shuffled labels as a histogram, and a line at the `real` best with
    it and `p` beside it; `label` names the score.

    save : a folder in summary_downloads, emptied of its PDFs, then given the plot.
    """
    with plt.rc_context(_fonts(font_size)):
        fig, ax = plt.subplots(layout="constrained", figsize=(width / 2, width / 3))
        ax.hist(shuffled, bins=np.arange(0, 101, 2.5), color=run_color, edgecolor="white")
        ax.axvline(real, color=best_color, lw=3)
        ax.text(real, 1, f" {real:.0f}\n p = {p:.2f}", transform=ax.get_xaxis_transform(), va="top")
        ax.set(xlim=(30, 100), xlabel=label, ylabel="Searches")
        ax.yaxis.set_major_locator(MaxNLocator(3, integer=True))
        ax.spines[["top", "right"]].set_visible(False)
    if save:
        save_figure(fig, "permutation", figure_dir=_fresh_folder(save))
    plt.show()
