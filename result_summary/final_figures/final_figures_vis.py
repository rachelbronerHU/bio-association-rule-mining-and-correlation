"""Drawing for the final figures. Shared pieces come from `vis_helper.py`."""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle
from matplotlib.transforms import blended_transform_factory

from vis_helper import (PANEL_FONTS, SHARE_COLORS, TEXT_WIDTH, _prevalence_norm, _shares_panel,
                        draw_row_trends, figure_titles, save_figure, INK, ATTRACTION, AVOIDANCE)

_FIGURE_DIR = Path(__file__).resolve().parent / "summary_downloads"

KIND_COLORS = {"attracts": ATTRACTION, "avoids": AVOIDANCE}
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


def plot_rules_per_fov(counts, marked, colors, labels, save=None):
    """Stacked bar of rules per FOV, one colour per column of `counts`.

    counts : FOV x group, in the order drawn.
    marked : {FOV: letter}, each drawn as a dot under its bar.
    """
    with plt.rc_context(PANEL_FONTS):
        fig, ax = plt.subplots(figsize=(TEXT_WIDTH, 2.2))
        counts.plot.bar(stacked=True, width=1, color=colors, ax=ax)
        ax.legend(labels, frameon=False)
        spots = [counts.index.get_loc(fov) for fov in marked]
        ax.plot(spots, np.zeros(len(spots)), "o", color=INK, markersize=4,
                transform=ax.get_xaxis_transform(), clip_on=False, zorder=3)
        ax.set_xticks(spots, list(marked.values()), rotation=0)
        ax.tick_params(axis="x", length=0, pad=6)
        ax.set(xlabel=f"FOVs, from most rules that passed FDR to fewest ({len(counts)})",
               ylabel="number of rules")
        fig.tight_layout()
    save_figure(fig, save, figure_dir=_FIGURE_DIR)
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
    if classes is not None:
        rank = {name: i for i, name in enumerate(CLASS_COLORS)}
        classes = classes.reindex(found.index)
        order = classes.map(rank).sort_values(kind="stable").index
        found, tested, classes = found.loc[order], tested.loc[order], classes.loc[order]
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
    _rule_rows(ax_trend, ax, ax_cells, block["found"], block["tested"], block["fdr"],
               block["skipped"], block["line_color"], cells, block["eligibility"], metadata,
               block["organ"], block["score"], norm)
    pad = (widest - block["found"].shape[1] - 1) / 2
    ax.set_xlim(-pad, widest - pad)
    if key:
        _share_key(ax_cells)
    return ax


def plot_top_gaps(blocks, cells, metadata, title, subtitle=None, params=None, paper=False, save=None):
    """The strongest rules of several rule change heatmaps, one block each, drawn like their rows.

    blocks : dicts with 'title', 'found', 'tested', 'fdr', 'skipped', 'line_color',
             'eligibility', 'organ' and 'score', as `plot_rule_change_heatmap` takes them.
    paper  : one figure per block, saved as save_1, save_2 …, with no titles; they are printed.
    Squares keep one width and one colour scale in every block, centred under the widest.
    """
    norm = _prevalence_norm(pd.concat([_share(block["found"], block["tested"]) for block in blocks]))
    rows = [len(block["found"]) for block in blocks]
    widest = max(block["found"].shape[1] for block in blocks) + 1.5
    row_inch, gap_inch = 0.32, 0.5
    if paper:
        print(title, subtitle, params, sep="\n")
        for number, block in enumerate(blocks):
            print(f"\n{number + 1}. {block['title']}")
            head_inch = 0.55 if number == 0 else 0.25
            height = head_inch + row_inch * len(block["found"]) + 0.05
            with plt.rc_context(PANEL_FONTS):
                fig = plt.figure(figsize=(TEXT_WIDTH, height))
                grid = fig.add_gridspec(1, 3, bottom=0.05 / height, top=1 - head_inch / height,
                                        **_TOP_WIDTHS)
                _top_block(fig, grid, 0, block, cells, metadata, widest, norm, key=number == 0)
            save_figure(fig, save and f"{save}_{number + 1}", figure_dir=_FIGURE_DIR)
            plt.show()
        return
    with plt.rc_context(PANEL_FONTS):
        fig = plt.figure(figsize=(TEXT_WIDTH, 1.4 + row_inch * sum(rows) + gap_inch * len(blocks)))
        axes_top = figure_titles(fig, title, subtitle=subtitle, params=params)
        grid = fig.add_gridspec(len(blocks), 3, height_ratios=rows,
                                hspace=gap_inch / (row_inch * np.mean(rows)),
                                bottom=0.02, top=axes_top, **_TOP_WIDTHS)
        for number, block in enumerate(blocks):
            ax = _top_block(fig, grid, number, block, cells, metadata, widest, norm, key=number == 0)
            fig.text(0.01, ax.get_position().y1 + 20 / 72 / fig.get_figheight(), block["title"],
                     fontsize=8, color=INK, fontweight="medium", va="bottom")
    save_figure(fig, save, figure_dir=_FIGURE_DIR)
    plt.show()
