"""The same rule seen in real fields, one figure per state.

Shared by the differential-rule and complex-rule notebooks, so both get the same
three-level title: organ and rule, then what the figure shows, then the scope.
Stages run down the rows and the two views across, which is what lets each field
be drawn as large as the page allows.
"""
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from vis_helper import save_figure, plot_fov, set_cell_colors, resolve_cell_colors

TEXT_WIDTH = 6.85

_STATE_NAMES = {1: "Attraction", -1: "Avoidance", 0: "No rule"}
_STATE_SLUGS = {1: "attraction", -1: "avoidance", 0: "no_rule"}
_SUBTITLES = {1: "Attraction examples",
              -1: "Avoidance examples",
              0: "Fields without the rule"}

_TITLE_ABOVE = 0.95      # the three-level title, before any panel caption
_LINE = 0.145            # one caption line
_LEGEND_BLOCK = 0.55
_LABEL_MARGIN = 0.55
_EDGE = 0.06
_MAX_HEIGHT = 9.6
_WSPACE = 0.05
_CELL_SIZE = 10

_METRIC_ORDER = ("lift", "conf", "conv", "sup", "lev")
_PASSING = "#6B7A63"
_NOT_PASSING = "#A2645A"


def _caption(metrics, fdr):
    """Short metric names over two lines, in the order a reader scans them."""
    parts = [f"{name} {metrics[name]:.3g}" for name in _METRIC_ORDER
             if pd.notna(metrics.get(name, np.nan))]
    if pd.notna(fdr):
        parts.append(f"fdr {fdr:.3g}")
    if not parts:
        return ""
    half = (len(parts) + 1) // 2
    return " · ".join(parts[:half]) + "\n" + " · ".join(parts[half:])


def _scope(score, stages, min_cells):
    return (f"Unit: FOV · Score: {score.replace(' score', '').lower()} · "
            f"Stages: {' / '.join(stages)} · Eligibility: ≥{min_cells} cells/type")


def _target(save, state):
    if not save:
        return None
    path = Path(save)
    return str(path.with_name(f"{path.stem}_{_STATE_SLUGS[state]}{path.suffix}"))


def _layout(n_stages, head_lines, row_lines):
    """Largest square field that fits, once the captions have been given their room.

    A caption sits above its panel, so the tallest one sets both the header block
    and the gap between rows. Sizing these from the text is what keeps a long
    "below gate" line from landing on the field above it.
    """
    title_block = _TITLE_ABOVE + _LINE * head_lines
    gap = _LINE * row_lines + 0.10
    by_width = (TEXT_WIDTH - _LABEL_MARGIN - _EDGE) / (2 + _WSPACE)
    by_height = ((_MAX_HEIGHT - title_block - _LEGEND_BLOCK - (n_stages - 1) * gap)
                 / n_stages)
    field = min(by_width, by_height)
    plot_width = (2 + _WSPACE) * field
    plot_height = n_stages * field + (n_stages - 1) * gap
    height = plot_height + title_block + _LEGEND_BLOCK
    left = _LABEL_MARGIN + (TEXT_WIDTH - _LABEL_MARGIN - _EDGE - plot_width) / 2
    return (height, title_block, gap / field,
            left / TEXT_WIDTH, (left + plot_width) / TEXT_WIDTH)


def _blank(ax):
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)


def _one_state(examples, state, stages, cells, metadata, organ, score, metric,
               min_cells, max_fdr, save, figure_dir, eligible_n):
    rule = examples["rule"].iat[0]
    ant = list(examples["antecedent_cells"].iat[0])
    con = list(examples["consequent_cells"].iat[0])
    rows = examples.set_index("stage")
    cell_colors = resolve_cell_colors(ant + con)

    captions = {}
    for stage in stages:
        if stage not in rows.index:
            continue
        item = rows.loc[stage]
        if isinstance(item, pd.DataFrame):
            item = item.iloc[0]
        text = _caption(item.get("metrics") or {}, item.get("fdr"))
        why = item.get("why")
        if isinstance(why, str) and why:
            text = f"{text}\n{why}" if text else why
        captions[stage] = text

    row_lines = max((text.count("\n") + 1 for text in captions.values() if text),
                    default=1)
    first = stages[0] if stages else None
    head_lines = max(2, 1 + (captions.get(first, "").count("\n") + 1
                             if captions.get(first) else 1))
    height, title_block, hspace, left, right = _layout(len(stages), head_lines, row_lines)
    fig, axes = plt.subplots(len(stages), 2, figsize=(TEXT_WIDTH, height),
                             squeeze=False, facecolor="white",
                             gridspec_kw={"wspace": _WSPACE, "hspace": hspace})

    for row, stage in enumerate(stages):
        full, marked = axes[row, 0], axes[row, 1]
        if stage not in rows.index:
            for ax in (full, marked):
                _blank(ax)
            marked.set_visible(False)
            message = ("0 eligible FOVs" if eligible_n.get(stage) == 0
                       else f"No eligible {_STATE_NAMES[state].lower()} example")
            full.text((2 + _WSPACE) / 2, 0.5, message, ha="center", va="center",
                      color="#898781", fontsize=8.2)
            full.set_ylabel(stage, fontsize=10, fontweight="bold", labelpad=8)
            continue

        item = rows.loc[stage]
        if isinstance(item, pd.DataFrame):
            item = item.iloc[0]
        fov = item["FOV"]
        biopsy = item.get("Biopsy", np.nan)
        location = fov if pd.isna(biopsy) else f"biopsy {biopsy} · {fov}"
        plot_fov(fov, "", cells, metadata, ax=full, show_legend=False,
                 cell_size=_CELL_SIZE, colors=cell_colors)
        plot_fov(fov, "", cells, metadata, target_ant_cells=ant, target_cons_cells=con,
                 ax=marked, show_legend=False, cell_size=_CELL_SIZE, colors=cell_colors)

        caption = captions.get(stage, "")
        passes = state != 0 and pd.notna(item.get("fdr")) and item["fdr"] <= max_fdr

        full.set_title(f"full FOV\n{location}" if row == 0 else location, fontsize=9.0)
        marked.set_title(f"rule cells\n{caption}" if row == 0 else caption,
                         fontsize=7.5, color=_PASSING if passes else _NOT_PASSING)
        for ax in (full, marked):
            ax.set_xlabel("")
            ax.set_xticks([])
            ax.set_yticks([])
        marked.set_ylabel("")
        full.set_ylabel(stage, fontsize=10, fontweight="bold", labelpad=8)

    handles = [
        Line2D([0], [0], marker="o", linestyle="none",
               markerfacecolor=cell_colors.get(cell, "black"), markeredgecolor="none",
               markersize=7, label=cell.replace("_", " "))
        for cell in dict.fromkeys(ant + con)
    ]
    fig.legend(handles=handles, title="Cell type", frameon=False, ncol=len(handles),
               loc="lower center", bbox_to_anchor=(0.5, 0.006),
               fontsize=8.5, title_fontsize=9)

    fig.suptitle(f"{organ} · {rule.replace(' -> ', ' → ')}", fontsize=13,
                 y=1 - 0.30 / height)
    fig.text(0.5, 1 - 0.58 / height, _SUBTITLES[state],
             ha="center", va="top", fontsize=10)
    fig.text(0.5, 1 - 0.82 / height, _scope(score, stages, min_cells),
             ha="center", va="top", fontsize=8.5, color="#706E68")
    fig.subplots_adjust(top=1 - title_block / height, bottom=_LEGEND_BLOCK / height,
                        left=left, right=right, wspace=_WSPACE, hspace=hspace)

    save_figure(fig, save, figure_dir=figure_dir)
    plt.show()
    return fig


def plot_rule_fovs(examples, stages, cells, metadata, organ, score, metric="Lift",
                   min_cells=20, max_fdr=0.05, save=None, figure_dir=None):
    """One figure per observed state: stages down the rows, full and highlighted across."""
    eligible_n = getattr(examples, "attrs", {}).get("eligible_n", {})
    examples = pd.DataFrame(examples)
    if examples.empty:
        print("No representative FOVs to plot.")
        return []

    if figure_dir is None:
        figure_dir = Path.cwd() / "summary_downloads"
    set_cell_colors(cells)
    present = set(pd.to_numeric(examples["state"], errors="coerce").dropna().astype(int))

    figures = []
    for state in (1, -1, 0):
        if state not in present:
            continue
        subset = examples[examples.state.eq(state)].copy()
        figures.append(_one_state(
            subset, state, stages, cells, metadata, organ, score, metric, min_cells,
            max_fdr, _target(save, state), figure_dir, eligible_n,
        ))
    return figures
