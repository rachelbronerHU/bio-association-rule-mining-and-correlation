"""Drawing for the final figures. Shared pieces come from `vis_helper.py`."""
from itertools import combinations
from textwrap import fill
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize, to_hex
from matplotlib.lines import Line2D
from matplotlib.patches import Circle, FancyArrowPatch, Patch, Polygon, Rectangle
from matplotlib.ticker import MaxNLocator

from vis_helper import (_category_colors, _cell_handles, save_figure, INK, GRID, ZERO, ATTRACTION, AVOIDANCE,
                        UP_ARROW, DOWN_ARROW)

_FIGURE_DIR = Path(__file__).resolve().parent / "summary_downloads"
PANEL_INCHES = 2.5

KIND_COLORS = {"attracts": ATTRACTION, "avoids": AVOIDANCE}
EDGE_LAYER = {"avoids": 1, "attracts": 2}
PREVALENCE, PREVALENCE_CMAP = Normalize(0, 100), "YlOrRd"
PC_TAG_MAPS = ("Blues", "YlOrBr")
TRAJECTORY_MAP = "plasma"
EDGE_HEADS = {"attracts": "-|>,head_length=0.6,head_width=0.4", "avoids": "-[,widthB=0.6,lengthB=0"}


def _fonts(size):
    return {"font.size": size, "xtick.labelsize": size, "ytick.labelsize": size,
            "legend.fontsize": size, "pdf.fonttype": 42}


def _fresh_folder(save):
    """The folder `save` in summary_downloads, emptied of its PDFs."""
    folder = _FIGURE_DIR / save
    for old in folder.glob("*.pdf"):
        old.unlink()
    return folder


def _save_parts(fig, parts, folder, name, piece):
    """`fig` saved as folder/<name>, and each part (axes or subfigure) cut out on its own
    as <piece>_1, <piece>_2 …"""
    save_figure(fig, name, figure_dir=folder)
    for n, part in enumerate(parts, 1):
        box = part.get_tightbbox().transformed(fig.dpi_scale_trans.inverted())
        fig.savefig(folder / f"{piece}_{n}.pdf", bbox_inches=box)


def _save_row(fig, parts, save, piece):
    """Empty the folder `save` of its PDFs, then save the figure and its parts in it."""
    folder = _fresh_folder(save)
    _save_parts(fig, parts, folder, save, piece)
    return folder


def _save_key(handles, path, font_size):
    with plt.rc_context(_fonts(font_size)):
        fig, ax = plt.subplots(figsize=(1, 1))
        ax.legend(handles=handles, loc="center", frameon=False)
        ax.axis("off")
        fig.savefig(path, bbox_inches="tight")
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


def _fov_map(ax, cells, colors, dot, scale_um, scale_font):
    size = cells["x_um"].max()
    ax.scatter(cells["x_um"], cells["y_um"], s=dot * (400 / size) ** 2, linewidths=0,
               c=cells["cell type"].astype(str).map(colors).tolist())
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


def trajectory_tags(n):
    """`n` tag colours from TRAJECTORY_MAP, dark purple for the first to orange for the last."""
    return [to_hex(color) for color in plt.get_cmap(TRAJECTORY_MAP)(np.linspace(0, 0.85, n))]


def _tag(ax, color):
    """A filled triangle in the top left corner, the FOV's tag colour."""
    ax.add_patch(Polygon([(0, 1), (0.15, 1), (0, 0.85)], closed=True, color=color, lw=0,
                         transform=ax.transAxes, clip_on=False, zorder=4))


def _row(n, width, gap_after=None):
    """A figure with a row of `n` square axes, a blank half-width space after the first
    `gap_after` of them, and how much larger a panel is than PANEL_INCHES."""
    ratios = [1] * n if gap_after is None else [1] * gap_after + [0.5] + [1] * (n - gap_after)
    fig, axes = plt.subplots(1, len(ratios), layout="constrained", squeeze=False, width_ratios=ratios,
                             figsize=(width, width / sum(ratios)))
    axes = list(axes[0])
    if gap_after is not None:
        axes.pop(gap_after).axis("off")
    return fig, axes, width / sum(ratios) / PANEL_INCHES


def plot_rules_per_fov(counts, marked, colors, labels, width, font_size, save=None):
    """Stacked bar of rules per FOV, one colour per column of `counts`, and under each marked FOV
    a dot in its tag colour; no dots when `marked` is empty.

    save : the figure's name; its colour key is saved beside it as <save>_key.
    """
    with plt.rc_context(_fonts(font_size)):
        fig, ax = plt.subplots(layout="constrained", figsize=(width, 0.6 * width))
        counts.plot.bar(stacked=True, width=1, linewidth=0, color=colors, legend=False, ax=ax)
        if len(marked):
            ax.scatter([counts.index.get_loc(fov) for fov in marked], [0] * len(marked), s=150,
                       c=fov_tags(len(marked)), clip_on=False, zorder=3, transform=ax.get_xaxis_transform())
        ax.set(xticks=[], xlabel="", xlim=(-0.5, len(counts) - 0.5))
        ax.yaxis.set_major_locator(MaxNLocator(3))
        ax.spines[["top", "right"]].set_visible(False)
    save_figure(fig, save, figure_dir=_FIGURE_DIR)
    if save:
        _save_key([Patch(color=c, label=l) for c, l in zip(colors, labels)],
                  _FIGURE_DIR / f"{save}_key.pdf", font_size)
    plt.show()


def plot_fov_maps(cells, fovs, cell_colors, key, width, font_size, dot=10, scale_um=100,
                  scale_font=None, gap_after=None, tags=None, save=None):
    """A row of FOV maps, cells coloured by type, each with a `scale_um` bar.

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
            _fov_map(ax, cells[cells["fov"].eq(fov)], cell_colors, dot * scale ** 2, scale_um, scale_font)
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
    _save_key(handles, _FIGURE_DIR / save / "cells.pdf", font_size)


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


def _network(ax, rules, colors, node, line, strongest):
    """One FOV's rules on a circle of cell types: centre to neighbour, → attracts, ⊣ avoids,
    thicker for a stronger |log2 lift|. A rule of a type with itself is a small loop."""
    nodes = set(rules["center"]) | set(rules["neighbor"])
    pairs = {tuple(sorted(pair)) for pair in zip(rules["center"], rules["neighbor"]) if pair[0] != pair[1]}
    order = _circle_order(nodes, list(pairs))
    angles = np.pi / 2 - 2 * np.pi * np.arange(len(order)) / max(len(order), 1)
    xy = dict(zip(order, np.c_[np.cos(angles), np.sin(angles)]))
    strength = np.log2(rules["Lift"].clip(lower=2.0 ** -strongest)).abs().clip(upper=strongest)
    for rule, lw in zip(rules.itertuples(), line * strength):
        color, layer = KIND_COLORS[rule.Kind], EDGE_LAYER[rule.Kind]
        if rule.center == rule.neighbor:
            ax.add_patch(Circle(1.1 * xy[rule.center], 0.1, fill=False, color=color, lw=lw, zorder=layer))
        else:
            ax.add_patch(FancyArrowPatch(xy[rule.center], xy[rule.neighbor], arrowstyle=EDGE_HEADS[rule.Kind],
                                         connectionstyle="arc3,rad=0.2", mutation_scale=5 * line,
                                         shrinkA=node ** 0.5, shrinkB=node ** 0.5, color=color, lw=lw,
                                         zorder=layer))
    if order:
        ax.scatter(*np.array([xy[cell] for cell in order]).T, s=node, linewidths=0, zorder=3,
                   c=[colors[cell] for cell in order])
    ax.set(xticks=[], yticks=[], aspect="equal", xlim=(-1.25, 1.25), ylim=(-1.25, 1.25))
    ax.spines[:].set_visible(False)


def _save_edge_key(path, font_size, line, strongest):
    """→ attracts and ⊣ avoids in their colours, then the line width for each |log2 lift|."""
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
        fig.savefig(path, bbox_inches="tight")
    plt.show()


def plot_rule_networks(rules, fovs, cell_colors, width, font_size, node=150, line=1.5, strongest=3,
                       gap_after=None, save=None):
    """A row of rule networks, one per FOV, drawn by `_network`; the nodes of each are ordered
    on their circle so as few lines cross as can be found.

    rules     : one row per rule, with 'FOV', 'center', 'neighbor', 'Kind' and 'Lift'.
    node      : node dot area; line : line width per unit of |log2 lift|, which stops at `strongest`;
                both for a network of PANEL_INCHES.
    gap_after : a blank space after this many networks; none when None.
    save      : a folder in summary_downloads, emptied of its PDFs, then given the row, each
                network on its own (network_1 …) and the edge key.
    """
    with plt.rc_context(_fonts(font_size)):
        fig, axes, scale = _row(len(fovs), width, gap_after)
        node, line = node * scale ** 2, line * scale
        for fov, ax, tag in zip(fovs, axes, fov_tags(len(fovs))):
            _network(ax, rules[rules["FOV"].eq(fov)], cell_colors, node, line, strongest)
            _tag(ax, tag)
    if save:
        folder = _save_row(fig, axes, save, "network")
        _save_edge_key(folder / "edges.pdf", font_size, line, strongest)
    plt.show()


def stars(fdr):
    return "***" if fdr <= 0.001 else "**" if fdr <= 0.01 else "*" if fdr <= 0.05 else ""


def _count_rows(ax, counts, color, heights=None, rising=None):
    """One line per row across its columns, at `heights` (default `counts`), each point
    labelled with its count; with `rising` (one bool per row), an up or down arrow at the end."""
    x = np.arange(counts.shape[1])
    heights = counts if heights is None else heights
    for row, (line, height) in enumerate(zip(counts.to_numpy(float), heights.to_numpy(float))):
        y = row + 0.9 - 0.5 * height / max(np.nanmax(height), 1e-9)
        shown = ~np.isnan(y)
        ax.plot(x[shown], y[shown], color=color, lw=3, marker="o", markersize=8)
        for at in x[shown]:
            ax.annotate(f"{line[at]:.0f}", (at, y[at]), xytext=(0, 6), textcoords="offset points",
                        ha="center", va="bottom", color=color)
        if rising is not None:
            up = rising.iloc[row]
            ax.plot(x[-1] + 0.5, y[-1], "^" if up else "v", markersize=14,
                    color=UP_ARROW if up else DOWN_ARROW)
    ax.set(xlim=(-0.4, len(x) - 0.4 + 0.5 * (rising is not None)), ylim=(len(counts), 0),
           xticks=[], yticks=[])
    ax.spines[:].set_visible(False)


def _trend_width(groups, cell, trend):
    """Width of a trend block in inches: two line plots, the squares, and a square's gap on each side."""
    return 2 * trend + (groups + 2) * cell


def _trend_block(sub, found, tested, fdr, kind, cell, trend):
    """Units with each rule per group, drawn at their share of the testable units, that share
    as squares, and the testable units per group; with `fdr`, its stars and an up or down arrow
    from the first group to the last. `cell` is a square's side
    and `trend` a line plot's width, in inches; `sub` is one row taller than the rules, for the
    group names."""
    rows, groups = found.shape
    width = _trend_width(groups, cell, trend)
    box = lambda left, size: sub.add_axes([left / width, 0, size / width, rows / (rows + 1)])
    ax_found, ax_share = box(0, trend), box(trend + cell, groups * cell)
    ax_tested = box(trend + (groups + 2) * cell, trend)
    share = 100 * found / tested
    rising = None if fdr is None else share.iloc[:, -1] >= share.iloc[:, 0]
    _count_rows(ax_found, found, KIND_COLORS[kind], share, rising)
    ax_found.set_yticks(np.arange(rows) + 0.5, found.index.str.replace(" -> ", " → "))
    ax_found.tick_params(length=0)
    ax_share.imshow(share, cmap=PREVALENCE_CMAP, norm=PREVALENCE, aspect="auto",
                    extent=(-0.5, groups - 0.5, rows, 0))
    for row in range(1, rows):
        ax_share.axhline(row, color="white", lw=4)
    for column in range(1, groups):
        ax_share.axvline(column - 0.5, color="white", lw=4)
    ax_share.set_xticks(range(groups), [fill(str(group), 9, break_long_words=False)
                                        for group in share.columns])
    ax_share.set_yticks(np.arange(rows) + 0.5, [] if fdr is None else fdr.map(stars))
    ax_share.tick_params(length=0, top=True, labeltop=True, bottom=False, labelbottom=False,
                         left=False, labelleft=False, right=True, labelright=True)
    ax_share.spines[:].set_visible(False)
    _count_rows(ax_tested, tested, ZERO)


def plot_rule_trends(panels, unit, font_size, cell=1.8, trend=3.5, save=None):
    """One titled figure per panel: blocks of rules across groups, one under the other, drawn
    by `_trend_block` at the same size in every panel.

    panels : {title: blocks}; a block is a dict with 'found' and 'tested' (rule x group counts),
             'kind', and optionally 'fdr' (per rule) for stars and arrows.
    unit   : what was counted, "FOV" or "patient", for the colour scale's label.
    cell, trend : a square's side and a line plot's width, in inches.
    save   : a folder in summary_downloads, emptied of its PDFs, then given each panel, named
             after its title, each of its blocks on its own (<panel>_1, <panel>_2 …) and the colour scale.
    """
    if not panels:
        print("No rule passes.")
        return
    folder = _fresh_folder(save) if save else None
    for title, blocks in panels.items():
        heights = [cell] + [(len(block["found"]) + 1) * cell for block in blocks]
        with plt.rc_context(_fonts(font_size)):
            fig = plt.figure(figsize=(_trend_width(blocks[0]["found"].shape[1], cell, trend), sum(heights)))
            head, *subs = fig.subfigures(len(heights), 1, height_ratios=heights, squeeze=False)[:, 0]
            head.text(0.5, 0.5, title, ha="center", va="center", fontsize=1.2 * font_size)
            for sub, block in zip(subs, blocks):
                _trend_block(sub, block["found"], block["tested"], block.get("fdr"), block["kind"], cell, trend)
        if save:
            name = title.replace(" · ", "_").replace(" ", "_")
            _save_parts(fig, subs, folder, name, name)
        plt.show()
    if save:
        with plt.rc_context(_fonts(font_size)):
            key, ax = plt.subplots(figsize=(6, 0.4))
            key.colorbar(ScalarMappable(PREVALENCE, PREVALENCE_CMAP), cax=ax, orientation="horizontal",
                         label=f"% of testable {unit}s")
            key.savefig(folder / "prevalence.pdf", bbox_inches="tight")
        plt.show()


def plot_pca(coords, variance, stage, marked, pair, width, font_size, line=None, save=None):
    """FOVs on PC1 and PC2, coloured by their `stage`, a ring in its tag colour around each FOV
    of `marked` ({FOV: colour}), a box around the two FOVs of `pair`; none when empty; and a
    dashed `line` between its two (PC1, PC2) ends.

    save : a folder in summary_downloads, emptied of its PDFs, then given the PCA and the stage key.
    """
    with plt.rc_context(_fonts(font_size)):
        fig, ax = plt.subplots(layout="constrained", figsize=(width / 2, width / 2))
        _pca(ax, coords, variance, stage, marked, pair)
        if line is not None:
            ax.plot(*np.transpose(line), color=INK, lw=1.5, ls=(0, (4, 2)), zorder=0)
    if save:
        folder = _fresh_folder(save)
        save_figure(fig, "pca", figure_dir=folder)
        _save_stage_key(coords[stage], folder, font_size)
    plt.show()


def plot_pca_pair(pcas, stage, width, font_size, save=None):
    """PCAs of the same FOVs side by side, coloured by their `stage`, each titled by its name.

    pcas : {name: (coords, variance)}.
    save : a folder in summary_downloads, emptied of its PDFs, then given the PCAs together,
           each on its own (pca_1, pca_2 …) and the stage key.
    """
    with plt.rc_context(_fonts(font_size)):
        fig, axes, _ = _row(len(pcas), width)
        for ax, (name, (coords, variance)) in zip(axes, pcas.items()):
            _pca(ax, coords, variance, stage)
            ax.set_title(name)
    if save:
        folder = _save_row(fig, axes, save, "pca")
        _save_stage_key(coords[stage], folder, font_size)
    plt.show()


def _pca(ax, coords, variance, stage, marked={}, pair=()):
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
    ax.set(xlabel=f"PC1 ({variance[0]:.0f}%)", ylabel=f"PC2 ({variance[1]:.0f}%)", xticks=[], yticks=[])
    ax.spines[["top", "right"]].set_visible(False)


def _save_stage_key(stages, folder, font_size):
    colors, order = _category_colors(stages)
    _save_key([Line2D([], [], marker="o", ls="", color=colors[name], markersize=0.8 * font_size, label=name)
               for name in order], folder / "stages.pdf", font_size)


def plot_loadings(weights, rules, n, width, font_size, save=None):
    """Per component, a row of `weights` over `rules`, its `n` rules of lowest and `n` of highest
    weight as bars, side by side, in that component's `pc_tags` family: dark for positive, light for negative.

    save : a folder in summary_downloads, emptied of its PDFs, then given the components together
           and each on its own (pc_1, pc_2 …).
    """
    names = np.char.replace(np.asarray(rules).astype(str), " -> ", " → ")
    with plt.rc_context(_fonts(font_size)):
        fig, axes = plt.subplots(1, len(weights), layout="constrained", figsize=(width, n + 1))
        for number, (ax, weight, (light, *_, dark)) in enumerate(zip(axes, weights, pc_tags(2)), 1):
            order = np.argsort(weight)
            shown = np.r_[order[:n], order[-n:]] if len(order) > 2 * n else order
            ax.barh(names[shown], weight[shown], color=np.where(weight[shown] > 0, dark, light))
            ax.axvline(0, color=INK, lw=1)
            ax.set(xlabel=f"PC{number} weight")
            ax.xaxis.set_major_locator(MaxNLocator(3))
            ax.tick_params(axis="y", length=0)
            ax.spines[["top", "right", "left"]].set_visible(False)
    if save:
        _save_row(fig, axes, save, "pc")
    plt.show()
