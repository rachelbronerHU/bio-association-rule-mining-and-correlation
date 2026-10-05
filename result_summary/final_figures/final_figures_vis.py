"""Drawing for the final figures. Shared pieces come from `vis_helper.py`."""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle
from matplotlib.transforms import blended_transform_factory

from vis_helper import (PANEL_FONTS, TEXT_WIDTH, _prevalence_norm, draw_row_trends,
                        figure_titles, save_figure, INK, ATTRACTION, AVOIDANCE)

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


def plot_rule_change_heatmap(found, tested, title, organ=None, subtitle=None, params=None,
                             note=None, skipped=None, fdr=None, classes=None, line_color=None,
                             save=None):
    """Share of testable FOVs that have each rule, per group, with the rule's trend on the left.

    found / tested : rule x group counts, with 'Total' as the last column.
    skipped        : a group shown but not used to rank the rules, drawn as an open point.
    fdr            : optional value per rule, written in its own column before 'Total'.
    classes        : optional class per rule (see `rule_class`), drawn as a soft band behind its row;
                     rows are then grouped by class, in the key's order.
    line_color     : optional single colour for the trend lines, which then end in an up or down arrow.
    """
    if classes is not None:
        rank = {name: i for i, name in enumerate(CLASS_COLORS)}
        classes = classes.reindex(found.index)
        order = classes.map(rank).sort_values(kind="stable").index
        found, tested, classes = found.loc[order], tested.loc[order], classes.loc[order]
    share = 100 * found / tested.replace(0, np.nan)
    text = (share.map(lambda v: "-" if np.isnan(v) else f"{v:.0f}%")
            + "\n" + found.astype(str) + "/" + tested.astype(str))
    if fdr is not None:
        share.insert(share.shape[1] - 1, "FDR", np.nan)
        text.insert(text.shape[1] - 1, "FDR", "")
    with plt.rc_context(PANEL_FONTS):
        fig, (ax_trend, ax) = plt.subplots(
            1, 2, figsize=(TEXT_WIDTH, 1.6 + 0.32 * len(share) + (0.2 if classes is not None else 0)),
            gridspec_kw={"width_ratios": [1, 2.6], "wspace": 0.03},
        )
        sns.heatmap(share, annot=text, fmt="", cmap="YlOrRd", norm=_prevalence_norm(share),
                    cbar=False, linewidths=0.5, linecolor="white",
                    annot_kws={"fontsize": 6}, ax=ax)
        ax.xaxis.tick_top()
        ax.set(xlabel="", ylabel="", yticks=[])
        ax.tick_params(length=0)
        ax.axvline(share.shape[1] - 1, color="white", lw=6)
        if fdr is not None:
            x = share.columns.get_loc("FDR") + 0.5
            for row, value in enumerate(fdr.reindex(share.index)):
                ax.text(x, row + 0.5, "-" if np.isnan(value) else f"{value:.2g}", ha="center", va="center",
                        fontsize=6.5, color=INK)
        draw_row_trends(ax_trend, share.drop(columns=["FDR", "Total"], errors="ignore"), skipped,
                        color=line_color)
        ax_trend.set_title("change across groups", fontsize=7)
        axes_top = figure_titles(fig, title, organ, subtitle=subtitle, params=params, note=note)
        if classes is not None:
            axes_top = _class_key(fig, axes_top)
        fig.subplots_adjust(left=0.25, right=0.99, bottom=0.02, top=axes_top)
        if classes is not None:
            _class_bands(fig, ax_trend, classes)
    save_figure(fig, save, figure_dir=_FIGURE_DIR)
    plt.show()
