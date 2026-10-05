"""Drawing for the final figures. Shared pieces come from `vis_helper.py`."""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from vis_helper import (PANEL_FONTS, TEXT_WIDTH, _prevalence_norm, draw_row_trends,
                        figure_titles, save_figure, INK)

_FIGURE_DIR = Path(__file__).resolve().parent / "summary_downloads"


def plot_rule_change_heatmap(found, tested, title, organ=None, subtitle=None, params=None,
                             note=None, skipped=None, fdr=None, save=None):
    """Share of testable FOVs that have each rule, per group, with the rule's trend on the left.

    found / tested : rule x group counts, with 'Total' as the last column.
    skipped        : a group shown but not used to rank the rules, drawn as an open point.
    fdr            : optional value per rule, written in its own column before 'Total'.
    """
    share = 100 * found / tested.replace(0, np.nan)
    text = (share.map(lambda v: "-" if np.isnan(v) else f"{v:.0f}%")
            + "\n" + found.astype(str) + "/" + tested.astype(str))
    if fdr is not None:
        share.insert(share.shape[1] - 1, "FDR", np.nan)
        text.insert(text.shape[1] - 1, "FDR", "")
    with plt.rc_context(PANEL_FONTS):
        fig, (ax_trend, ax) = plt.subplots(
            1, 2, figsize=(TEXT_WIDTH, 1.6 + 0.32 * len(share)),
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
        draw_row_trends(ax_trend, share.drop(columns=["FDR", "Total"], errors="ignore"), skipped)
        ax_trend.set_title("change across groups", fontsize=7)
        figure_titles(fig, title, organ, subtitle=subtitle, params=params, note=note)
        fig.subplots_adjust(left=0.25, right=0.99, bottom=0.02)
    save_figure(fig, save, figure_dir=_FIGURE_DIR)
    plt.show()
