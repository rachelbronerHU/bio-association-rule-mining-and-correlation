"""Drawing for the final figures. Shared pieces come from `vis_helper.py`."""
from itertools import combinations
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.colors import to_hex
from matplotlib.lines import Line2D
from matplotlib.patches import Circle, FancyArrowPatch, Patch, Polygon, Rectangle
from matplotlib.ticker import MaxNLocator
from matplotlib.transforms import blended_transform_factory

from vis_helper import (PANEL_FONTS, SHARE_COLORS, TEXT_WIDTH, _cell_handles, _prevalence_norm, _shares_panel,
                        draw_row_trends, figure_titles, save_figure, INK, GRID, ATTRACTION, AVOIDANCE)

_FIGURE_DIR = Path(__file__).resolve().parent / "summary_downloads"

KIND_COLORS = {"attracts": ATTRACTION, "avoids": AVOIDANCE}
EDGE_LAYER = {"avoids": 1, "attracts": 2}
EDGE_HEADS = {"attracts": "-|>,head_length=0.6,head_width=0.4", "avoids": "-[,widthB=0.6,lengthB=0"}
CLASS_COLORS = {
    "Immune–Immune": "#E3DCF2",
    "Immune–Epithel": "#FCE0C8",
    "Immune–Stroma": "#D3EDDC",
    "Tissue structure": "#F6EDC4",
}


def _class_bands(fig, ax, classes):
    """A soft band behind each row's name and trend, coloured by the rule's class."""
    across = blended_transform_factory(fig.transFigure, ax.transData)
    right = ax.get_position().x1
    for row, name in enumerate(classes):
        ax.add_patch(Rectangle((0.005, row + 0.07), right - 0.005, 0.86, transform=across,
                               color=CLASS_COLORS.get(name, "white"), lw=0, zorder=0,
                               clip_on=False))


def _class_key(fig, axes_top):
    """One line of coloured dots naming the classes, just under the titles."""
    points = fig.get_figheight() * 72
    key_top = axes_top + 27 / points
    dots = [Line2D([], [], ls="", marker="o", markersize=7, markerfacecolor=color,
                   markeredgecolor=INK, markeredgewidth=0.3, label=name)
            for name, color in CLASS_COLORS.items()]
    fig.legend(handles=dots, loc="upper center", bbox_to_anchor=(0.5, key_top), ncol=len(dots),
               frameon=False, fontsize=7.5, handletextpad=0.2, columnspacing=1.4,
               borderpad=0, borderaxespad=0)
    return key_top - 42 / points


def _by_class(found, tested, classes):
    """Rows grouped by class, in the key's order; unchanged when `classes` is None."""
    if classes is None:
        return found, tested, None
    rank = {name: i for i, name in enumerate(CLASS_COLORS)}
    classes = classes.reindex(found.index)
    order = classes.map(rank).sort_values(kind="stable").index
    return found.loc[order], tested.loc[order], classes.loc[order]


def _fonts(size):
    return {"font.size": size, "xtick.labelsize": size, "ytick.labelsize": size,
            "legend.fontsize": size, "pdf.fonttype": 42}


def _save_row(fig, axes, save, piece):
    """Empty the folder `save` of its PDFs, then save the row in it, and each axes cut out
    on its own as <piece>_1, <piece>_2 …"""
    folder = _FIGURE_DIR / save
    for old in folder.glob("*.pdf"):
        old.unlink()
    save_figure(fig, save, figure_dir=folder)
    for n, ax in enumerate(axes, 1):
        box = ax.get_tightbbox().transformed(fig.dpi_scale_trans.inverted())
        fig.savefig(folder / f"{piece}_{n}.pdf", bbox_inches=box)
    return folder


def _save_key(handles, path, font_size):
    with plt.rc_context(_fonts(font_size)):
        fig, ax = plt.subplots(figsize=(1, 1))
        ax.legend(handles=handles, loc="center", frameon=False)
        ax.axis("off")
        fig.savefig(path, bbox_inches="tight")
    plt.show()


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


def _tag(ax, color):
    """A filled triangle in the top left corner, the FOV's tag colour."""
    ax.add_patch(Polygon([(0, 1), (0.15, 1), (0, 0.85)], closed=True, color=color, lw=0,
                         transform=ax.transAxes, clip_on=False, zorder=4))


def plot_rules_per_fov(counts, marked, colors, labels, width, font_size, save=None):
    """Stacked bar of rules per FOV, one colour per column of `counts`, and under each marked FOV
    a dot in its tag colour.

    save : the figure's name; its colour key is saved beside it as <save>_key.
    """
    with plt.rc_context(_fonts(font_size)):
        fig, ax = plt.subplots(layout="constrained", figsize=(width, 0.6 * width))
        counts.plot.bar(stacked=True, width=1, linewidth=0, color=colors, legend=False, ax=ax)
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
                  scale_font=None, save=None):
    """A row of FOV maps, cells coloured by type, each with a `scale_um` bar.

    key  : the cell types named in the cell type key, in its order.
    dot  : cell dot area on a 400 um map; larger maps get smaller dots.
    scale_font : font size of the scale bar's label; `font_size` when None.
    save : a folder in summary_downloads, emptied of its PDFs, then given the row,
           each map on its own (map_1, map_2 …) and the cell type key.
    """
    with plt.rc_context(_fonts(font_size)):
        fig, axes = plt.subplots(1, len(fovs), layout="constrained", squeeze=False,
                                 figsize=(width, width / len(fovs)))
        for fov, ax, tag in zip(fovs, axes[0], fov_tags(len(fovs))):
            _fov_map(ax, cells[cells["fov"].eq(fov)], cell_colors, dot, scale_um, scale_font)
            _tag(ax, tag)
    if save:
        folder = _save_row(fig, axes[0], save, "map")
        _save_key(_cell_handles(key, cell_colors), folder / "cells.pdf", font_size)
    plt.show()


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
                       save=None):
    """A row of rule networks, one per FOV, drawn by `_network`; the nodes of each are ordered
    on their circle so as few lines cross as can be found.

    rules     : one row per rule, with 'FOV', 'center', 'neighbor', 'Kind' and 'Lift'.
    node      : node dot area; line : line width per unit of |log2 lift|, which stops at `strongest`.
    save      : a folder in summary_downloads, emptied of its PDFs, then given the row, each
                network on its own (network_1 …) and the edge key.
    """
    with plt.rc_context(_fonts(font_size)):
        fig, axes = plt.subplots(1, len(fovs), layout="constrained", squeeze=False,
                                 figsize=(width, width / len(fovs)))
        for fov, ax, tag in zip(fovs, axes[0], fov_tags(len(fovs))):
            _network(ax, rules[rules["FOV"].eq(fov)], cell_colors, node, line, strongest)
            _tag(ax, tag)
    if save:
        folder = _save_row(fig, axes[0], save, "network")
        _save_edge_key(folder / "edges.pdf", font_size, line, strongest)
    plt.show()


def _cell_share_rows(ax, rules, cells, eligibility, metadata, organ, score, stages):
    """One small cell share panel per rule, top to bottom, on the FOVs where it was testable."""
    ax.axis("off")
    for row, rule in enumerate(rules):
        inset = ax.inset_axes([0, 1 - (row + 0.9) / len(rules), 1, 0.8 / len(rules)])
        _shares_panel(inset, cells, eligibility, metadata,
                      {"organ": organ, "score": score, "rule": rule}, stages, compact=True)


def _share_key(ax):
    key = [Line2D([], [], color=color, marker="D", markersize=3, lw=1, label=label)
           for color, label in zip(SHARE_COLORS, ("first cell type", "second cell type"))]
    ax.legend(handles=key, title="cells in testable FOVs (%)", loc="lower center",
              bbox_to_anchor=(0.5, 1), ncol=2, frameon=False, fontsize=6, title_fontsize=7,
              handlelength=1.5, columnspacing=0.8, borderaxespad=0.3)


def _share(found, tested):
    return 100 * found / tested.replace(0, np.nan)


def _rule_rows(ax_trend, ax, ax_cells, found, tested, fdr=None, skipped=None, line_color=None,
               cells=None, eligibility=None, metadata=None, organ=None, score=None, norm=None):
    """The rows of a rule change heatmap: trend, share of testable FOVs per group, FDR,
    and, when `ax_cells` is given, the rule's two cell types. `norm` defaults to these rows' own."""
    share = _share(found, tested)
    norm = norm or _prevalence_norm(share)
    text = (share.map(lambda v: "-" if np.isnan(v) else f"{v:.0f}%")
            + "\n" + found.astype(str) + "/" + tested.astype(str))
    groups = list(share.columns)
    if fdr is not None:
        share["FDR"], text["FDR"] = np.nan, ""
    sns.heatmap(share, annot=text, fmt="", cmap="YlOrRd", norm=norm,
                cbar=False, linewidths=0.5, linecolor="white",
                annot_kws={"fontsize": 6}, ax=ax)
    ax.xaxis.tick_top()
    ax.set(xlabel="", ylabel="", yticks=[])
    ax.tick_params(length=0)
    if fdr is not None:
        ax.axvline(len(groups), color="white", lw=6)
        for row, value in enumerate(fdr.reindex(share.index)):
            ax.text(len(groups) + 0.5, row + 0.5, "-" if np.isnan(value) else f"{value:.2g}",
                    ha="center", va="center", fontsize=6.5, color=INK)
    draw_row_trends(ax_trend, share[groups], skipped, color=line_color)
    if ax_cells is not None:
        _cell_share_rows(ax_cells, share.index, cells, eligibility, metadata, organ, score, groups)


def plot_rule_change_heatmap(found, tested, title, organ=None, subtitle=None, params=None,
                             note=None, skipped=None, fdr=None, classes=None, line_color=None,
                             cells=None, eligibility=None, metadata=None, score=None, save=None):
    """Share of testable FOVs that have each rule, per group, with the rule's trend on the left.

    found / tested : rule x group counts.
    skipped        : a group shown but not used to rank the rules, drawn as an open point.
    fdr            : optional value per rule, written in its own last column.
    classes        : optional class per rule (see `rule_class`), drawn as a soft band behind its row;
                     rows are then grouped by class, in the key's order.
    line_color     : optional single colour for the trend lines, which then end in an up or down arrow.
    cells, eligibility, metadata, score : when given, a column on the right with each rule's
                     two cell types per group, on its testable FOVs (rule x FOV `eligibility`).
    """
    found, tested, classes = _by_class(found, tested, classes)
    with_cells = cells is not None
    with plt.rc_context(PANEL_FONTS):
        fig, (ax_trend, ax, *ax_cells) = plt.subplots(
            1, 3 if with_cells else 2,
            figsize=(TEXT_WIDTH, 1.6 + 0.32 * len(found) + (0.2 if classes is not None else 0)),
            gridspec_kw={"width_ratios": [1, 2.6, 1.8] if with_cells else [1, 2.6], "wspace": 0.03},
        )
        ax_cells = ax_cells[0] if with_cells else None
        _rule_rows(ax_trend, ax, ax_cells, found, tested, fdr, skipped, line_color,
                   cells, eligibility, metadata, organ, score)
        ax_trend.set_title("change across groups", fontsize=7)
        if with_cells:
            _share_key(ax_cells)
        axes_top = figure_titles(fig, title, organ, subtitle=subtitle, params=params, note=note)
        if classes is not None:
            axes_top = _class_key(fig, axes_top)
        fig.subplots_adjust(left=0.25, right=0.99, bottom=0.02, top=axes_top)
        if classes is not None:
            _class_bands(fig, ax_trend, classes)
    save_figure(fig, save, figure_dir=_FIGURE_DIR)
    plt.show()


_TOP_WIDTHS = {"width_ratios": [1, 2.6, 1.8], "wspace": 0.03, "left": 0.25, "right": 0.99}


def _top_block(fig, grid, row, block, cells, metadata, widest, norm, key):
    """One block of `plot_top_gaps` in one grid row: trend, squares, cell shares."""
    ax_trend, ax, ax_cells = (fig.add_subplot(grid[row, column]) for column in range(3))
    found, tested, classes = _by_class(block["found"], block["tested"], block.get("classes"))
    _rule_rows(ax_trend, ax, ax_cells, found, tested, block["fdr"],
               block["skipped"], block["line_color"], cells, block["eligibility"], metadata,
               block["organ"], block["score"], norm)
    pad = (widest - found.shape[1] - 1) / 2
    ax.set_xlim(-pad, widest - pad)
    if classes is not None:
        _class_bands(fig, ax_trend, classes)
    if key:
        _share_key(ax_cells)
    return ax


def plot_top_gaps(blocks, cells, metadata, title, subtitle=None, params=None, paper=False, save=None):
    """The strongest rules of several rule change heatmaps, one block each, drawn like their rows.

    blocks : dicts with 'title', 'found', 'tested', 'fdr', 'skipped', 'line_color',
             'eligibility', 'organ', 'score' and optionally 'classes',
             as `plot_rule_change_heatmap` takes them.
    paper  : one figure per block, saved as save_1, save_2 …, with no titles; they are printed.
    Squares keep one width and one colour scale in every block, centred under the widest.
    """
    norm = _prevalence_norm(pd.concat([_share(block["found"], block["tested"]) for block in blocks]))
    rows = [len(block["found"]) for block in blocks]
    widest = max(block["found"].shape[1] for block in blocks) + 1.5
    row_inch, gap_inch = 0.32, 0.5
    with_classes = any(block.get("classes") is not None for block in blocks)
    if paper:
        print(title, subtitle, params, sep="\n")
        for number, block in enumerate(blocks):
            print(f"\n{number + 1}. {block['title']}")
            head_inch = (0.55 + 0.25 * with_classes) if number == 0 else 0.25
            height = head_inch + row_inch * len(block["found"]) + 0.05
            with plt.rc_context(PANEL_FONTS):
                fig = plt.figure(figsize=(TEXT_WIDTH, height))
                grid = fig.add_gridspec(1, 3, bottom=0.05 / height, top=1 - head_inch / height,
                                        **_TOP_WIDTHS)
                _top_block(fig, grid, 0, block, cells, metadata, widest, norm, key=number == 0)
                if number == 0 and with_classes:
                    _class_key(fig, 1 - 27 / (height * 72))
            save_figure(fig, save and f"{save}_{number + 1}", figure_dir=_FIGURE_DIR)
            plt.show()
        return
    with plt.rc_context(PANEL_FONTS):
        fig = plt.figure(figsize=(TEXT_WIDTH, 1.4 + 0.2 * with_classes
                                  + row_inch * sum(rows) + gap_inch * len(blocks)))
        axes_top = figure_titles(fig, title, subtitle=subtitle, params=params)
        if with_classes:
            axes_top = _class_key(fig, axes_top)
        grid = fig.add_gridspec(len(blocks), 3, height_ratios=rows,
                                hspace=gap_inch / (row_inch * np.mean(rows)),
                                bottom=0.02, top=axes_top, **_TOP_WIDTHS)
        for number, block in enumerate(blocks):
            ax = _top_block(fig, grid, number, block, cells, metadata, widest, norm, key=number == 0)
            fig.text(0.01, ax.get_position().y1 + 20 / 72 / fig.get_figheight(), block["title"],
                     fontsize=8, color=INK, fontweight="medium", va="bottom")
    save_figure(fig, save, figure_dir=_FIGURE_DIR)
    plt.show()
