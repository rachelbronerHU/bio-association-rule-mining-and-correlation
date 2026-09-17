"""The same fields, seen once per rule: one figure per state.

Shared by the differential-rule and complex-rule notebooks, so both get the same
three-level title: organ and rule, then what the figure shows, then the scope.
Stages run down the rows. The first column is the untouched field; every column
after it highlights one rule's cells in that same field, which is how a longer
rule and its shorter one are compared without leaving the figure.
"""
import textwrap
from pathlib import Path
from typing import NamedTuple

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
_LABEL_MARGIN = 0.40
_EDGE = 0.03
_MAX_HEIGHT = 9.6
_WSPACE = 0.02
_CELL_SIZE = 8
_LABEL_WIDTH = 30
_CAPTION_WIDTH = 34      # wider than this and a caption spills over the next column

_METRIC_ORDER = ("lift", "conf", "conv", "sup", "lev")
_PASSING = "#6B7A63"
_NOT_PASSING = "#A2645A"


class View(NamedTuple):
    """One highlighted column: whose cells to colour, and what to say about them."""

    label: str
    cells: tuple                  # the rule's cell types, antecedent then consequent
    antecedent: tuple
    consequent: tuple
    captions: dict                # stage -> text
    passing: dict                 # stage -> did this rule clear the FDR gate here


def _caption(item):
    """Short metric names over two lines, then the reason when there is no rule."""
    metrics = item.get("metrics") or {}
    parts = [f"{name} {metrics[name]:.3g}" for name in _METRIC_ORDER
             if pd.notna(metrics.get(name, np.nan))]
    if pd.notna(item.get("fdr")):
        parts.append(f"fdr {item['fdr']:.3g}")
    half = (len(parts) + 1) // 2
    text = (" · ".join(parts[:half]) + "\n" + " · ".join(parts[half:])) if parts else ""

    why = item.get("why")
    if isinstance(why, str) and why:
        why = textwrap.fill(why, _CAPTION_WIDTH)
        return f"{text}\n{why}" if text else why
    return text


def _row_of(frame, stage):
    """The one row for this stage, or None."""
    if frame is None or stage not in frame.index:
        return None
    item = frame.loc[stage]
    return item.iloc[0] if isinstance(item, pd.DataFrame) else item


def view_of(examples, label, stages, max_fdr):
    """An examples frame turned into one column of the figure."""
    rows = examples.set_index("stage")
    first = examples.iloc[0]
    antecedent = tuple(first["antecedent_cells"])
    consequent = tuple(first["consequent_cells"])
    captions, passing = {}, {}
    for stage in stages:
        item = _row_of(rows, stage)
        if item is None:
            continue
        captions[stage] = _caption(item)
        passing[stage] = bool(pd.notna(item.get("fdr")) and item["fdr"] <= max_fdr)
    return View(label, tuple(dict.fromkeys(antecedent + consequent)),
                antecedent, consequent, captions, passing)


def _scope(score, stages, min_cells):
    return (f"Unit: FOV · Score: {score.replace(' score', '').lower()} · "
            f"Stages: {' / '.join(stages)} · Eligibility: ≥{min_cells} cells/type")


def _target(save, state):
    if not save:
        return None
    path = Path(save)
    return str(path.with_name(f"{path.stem}_{_STATE_SLUGS[state]}{path.suffix}"))


def _layout(n_stages, n_cols, head_lines, row_lines):
    """Largest square field that fits, once the captions have been given their room.

    A caption sits above its panel, so the tallest one sets both the header block
    and the gap between rows. Sizing these from the text is what keeps a long
    "below gate" line from landing on the field above it.
    """
    title_block = _TITLE_ABOVE + _LINE * head_lines
    gap = _LINE * row_lines + 0.10
    by_width = (TEXT_WIDTH - _LABEL_MARGIN - _EDGE) / (n_cols + (n_cols - 1) * _WSPACE)
    by_height = ((_MAX_HEIGHT - title_block - _LEGEND_BLOCK - (n_stages - 1) * gap)
                 / n_stages)
    field = min(by_width, by_height)
    plot_width = (n_cols + (n_cols - 1) * _WSPACE) * field
    height = n_stages * field + (n_stages - 1) * gap + title_block + _LEGEND_BLOCK
    left = _LABEL_MARGIN + (TEXT_WIDTH - _LABEL_MARGIN - _EDGE - plot_width) / 2
    return (height, title_block, gap / field,
            left / TEXT_WIDTH, (left + plot_width) / TEXT_WIDTH)


def _lines(text):
    return text.count("\n") + 1 if text else 0


def _wrapped(label):
    return textwrap.fill(label, _LABEL_WIDTH)


def _blank(ax):
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)


def _bare(ax):
    ax.set_xlabel("")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_ylabel("")


def _one_state(examples, state, views, stages, cells, metadata, organ, score,
               min_cells, save, figure_dir, eligible_n):
    rule = examples["rule"].iat[0]
    rows = examples.set_index("stage")
    shown_cells = tuple(dict.fromkeys(cell for view in views for cell in view.cells))
    cell_colors = resolve_cell_colors(list(shown_cells))

    n_cols = 1 + len(views)
    every = [view.captions.get(stage, "") for view in views for stage in stages]
    row_lines = max((_lines(text) for text in every if text), default=1)
    # The first row carries a column label as well, and a long rule name wraps.
    first = stages[0] if stages else None
    head_lines = max(
        [2] + [_lines(_wrapped(view.label)) + _lines(view.captions.get(first, ""))
               for view in views])
    height, title_block, hspace, left, right = _layout(
        len(stages), n_cols, head_lines, row_lines)
    fig, axes = plt.subplots(len(stages), n_cols, figsize=(TEXT_WIDTH, height),
                             squeeze=False, facecolor="white",
                             gridspec_kw={"wspace": _WSPACE, "hspace": hspace})

    span = (n_cols + (n_cols - 1) * _WSPACE) / 2
    for row, stage in enumerate(stages):
        item = _row_of(rows, stage)
        if item is None:
            for ax in axes[row]:
                _blank(ax)
                ax.set_visible(False)
            axes[row, 0].set_visible(True)
            message = ("0 eligible FOVs" if eligible_n.get(stage) == 0
                       else f"No eligible {_STATE_NAMES[state].lower()} example")
            axes[row, 0].text(span, 0.5, message, ha="center", va="center",
                              color="#898781", fontsize=8.2)
            axes[row, 0].set_ylabel(stage, fontsize=10, fontweight="bold", labelpad=8)
            continue

        fov = item["FOV"]
        biopsy = item.get("Biopsy", np.nan)
        location = fov if pd.isna(biopsy) else f"biopsy {biopsy} · {fov}"
        plot_fov(fov, "", cells, metadata, ax=axes[row, 0], show_legend=False,
                 cell_size=_CELL_SIZE, colors=cell_colors)
        axes[row, 0].set_title(f"full FOV\n{location}" if row == 0 else location,
                               fontsize=9.0)

        for column, view in enumerate(views, start=1):
            ax = axes[row, column]
            plot_fov(fov, "", cells, metadata, target_ant_cells=list(view.antecedent),
                     target_cons_cells=list(view.consequent), ax=ax, show_legend=False,
                     cell_size=_CELL_SIZE, colors=cell_colors)
            caption = view.captions.get(stage, "")
            ax.set_title(f"{_wrapped(view.label)}\n{caption}" if row == 0 else caption,
                         fontsize=7.5,
                         color=_PASSING if view.passing.get(stage) else _NOT_PASSING)

        for ax in axes[row]:
            _bare(ax)
        axes[row, 0].set_ylabel(stage, fontsize=10, fontweight="bold", labelpad=8)

    handles = [
        Line2D([0], [0], marker="o", linestyle="none",
               markerfacecolor=cell_colors.get(cell, "black"), markeredgecolor="none",
               markersize=7, label=cell.replace("_", " "))
        for cell in shown_cells
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


def plot_rule_fovs(examples, stages, cells, metadata, organ, score,
                   min_cells=20, max_fdr=0.05, save=None, figure_dir=None, others=()):
    """One figure per observed state: stages down the rows, each rule its own column.

    `others` : (label, frame) pairs covering the same fields — a shorter rule, say.
    Each becomes one more highlighted column, so the rules are compared in one field
    rather than across two figures.
    """
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
        views = [view_of(subset, "rule cells", stages, max_fdr)]
        for label, frame in others:
            frame = pd.DataFrame(frame)
            frame = frame[frame.FOV.isin(set(subset.FOV))]
            if not frame.empty:
                views.append(view_of(frame, label, stages, max_fdr))
        figures.append(_one_state(
            subset, state, views, stages, cells, metadata, organ, score, min_cells,
            _target(save, state), figure_dir, eligible_n,
        ))
    return figures
