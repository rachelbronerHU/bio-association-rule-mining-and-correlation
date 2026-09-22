"""The same fields, seen once per rule.

Shared by the differential-rule and complex-rule notebooks, so both get the same
three-level title: organ and rule, then what the figure shows, then the scope.

A rule on its own gets one figure per state, stages down the rows: the untouched
field, the rule's cells in it, and the cells it was counted on. A rule with a
shorter rule to compare against gets one figure per stage instead, the two rules
side by side with what each one counted underneath. Stages down the rows there
as well would halve every field, and the counted cells are what needs reading.
"""
import textwrap
from pathlib import Path
from typing import NamedTuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from vis_helper import (save_figure, plot_fov, plot_counted_cells, set_cell_colors,
                        resolve_cell_colors)
import rule_metrics as rm

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
_CENTER_COLOR = "#1A1A1A"
_NEIGHBOR_COLOR = "#D08C34"
_CHECK_LABEL = "cells counted for the rule"


class View(NamedTuple):
    """One rule in the figure: whose cells to colour, and what to say about them."""

    label: str
    cells: tuple                  # the rule's cell types, antecedent then consequent
    antecedent: tuple
    consequent: tuple
    captions: dict                # stage -> text
    passing: dict                 # stage -> did this rule clear the FDR gate here
    antecedent_items: tuple       # the same cells with their role, e.g. Goblet_CENTER
    consequent_items: tuple


class _Box(NamedTuple):
    """Where the panels sit, once the captions have been given their room."""

    height: float
    title_block: float
    hspace: float
    left: float
    right: float


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
    """An examples frame turned into one rule of the figure."""
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
                antecedent, consequent, captions, passing,
                rm.items_of(first["antecedent_items"]),
                rm.items_of(first["consequent_items"]))


def _scope(score, stages, min_cells):
    return (f"Unit: FOV · Score: {score.replace(' score', '').lower()} · "
            f"Stages: {' / '.join(stages)} · Eligibility: ≥{min_cells} cells/type")


def _target(save, *parts):
    """The file name with what the figure shows added to it."""
    if not save:
        return None
    path = Path(save)
    tail = [str(part).lower().replace(" ", "_") for part in parts]
    return str(path.with_name("_".join([path.stem, *tail]) + path.suffix))


def _layout(n_rows, n_cols, head_lines, row_lines):
    """Largest square field that fits, once the captions have been given their room.

    A caption sits above its panel, so the tallest one sets both the header block
    and the gap between rows. Sizing these from the text is what keeps a long
    "below gate" line from landing on the field above it.
    """
    title_block = _TITLE_ABOVE + _LINE * head_lines
    gap = _LINE * row_lines + 0.10
    by_width = (TEXT_WIDTH - _LABEL_MARGIN - _EDGE) / (n_cols + (n_cols - 1) * _WSPACE)
    by_height = ((_MAX_HEIGHT - title_block - _LEGEND_BLOCK - (n_rows - 1) * gap)
                 / n_rows)
    field = min(by_width, by_height)
    plot_width = (n_cols + (n_cols - 1) * _WSPACE) * field
    height = n_rows * field + (n_rows - 1) * gap + title_block + _LEGEND_BLOCK
    left = _LABEL_MARGIN + (TEXT_WIDTH - _LABEL_MARGIN - _EDGE - plot_width) / 2
    return _Box(height, title_block, gap / field,
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


def _where(item):
    biopsy = item.get("Biopsy", np.nan)
    return item["FOV"] if pd.isna(biopsy) else f"biopsy {biopsy} · {item['FOV']}"


def _full_panel(ax, fov, cells, metadata, cell_colors, title):
    """The field as it is, every cell type in its own colour."""
    plot_fov(fov, "", cells, metadata, ax=ax, show_legend=False,
             cell_size=_CELL_SIZE, colors=cell_colors)
    ax.set_title(title, fontsize=9.0)


def _rule_panel(ax, fov, cells, metadata, cell_colors, view, stage, named):
    """The same field with everything but the rule's cell types greyed out."""
    plot_fov(fov, "", cells, metadata, target_ant_cells=list(view.antecedent),
             target_cons_cells=list(view.consequent), ax=ax, show_legend=False,
             cell_size=_CELL_SIZE, colors=cell_colors)
    caption = view.captions.get(stage, "")
    ax.set_title(f"{_wrapped(view.label)}\n{caption}" if named else caption,
                 fontsize=7.5,
                 color=_PASSING if view.passing.get(stage) else _NOT_PASSING)


def _check_panel(ax, fov, cells, metadata, view, named):
    """The same field again, with only the cells the rule was counted on."""
    counted = rm.counted_cells(view.antecedent_items, view.consequent_items, cells, fov)
    plot_counted_cells(ax, fov, cells, metadata,
                       [(counted.neighbors, _NEIGHBOR_COLOR),
                        (counted.centers, _CENTER_COLOR)], _CELL_SIZE)
    line = (f"{len(counted.centers)} centers · "
            f"{counted.total}/{(cells['fov'] == fov).sum()} cells")
    ax.set_title(f"{_wrapped(_CHECK_LABEL)}\n{line}" if named else line, fontsize=7.5)


def _key(shown_cells, cell_colors):
    """One dot per cell type, then one per role in the counted panels."""
    def dot(color, label):
        return Line2D([0], [0], marker="o", linestyle="none", markeredgecolor="none",
                      markerfacecolor=color, markersize=7, label=label)

    return ([dot(cell_colors.get(cell, "black"), cell.replace("_", " "))
             for cell in shown_cells]
            + [dot(_CENTER_COLOR, "rule center"), dot(_NEIGHBOR_COLOR, "rule neighbor")])


def _frame(fig, box, organ, rule, subtitle, scope, shown_cells, cell_colors,
           save, figure_dir):
    """The three-level title, the key underneath, and the saved file."""
    handles = _key(shown_cells, cell_colors)
    fig.legend(handles=handles, title="Cell type · role", frameon=False,
               ncol=len(handles), loc="lower center", bbox_to_anchor=(0.5, 0.006),
               fontsize=8.5, title_fontsize=9)

    fig.suptitle(f"{organ} · {rule.replace(' -> ', ' → ')}", fontsize=13,
                 y=1 - 0.30 / box.height)
    fig.text(0.5, 1 - 0.58 / box.height, subtitle, ha="center", va="top", fontsize=10)
    fig.text(0.5, 1 - 0.82 / box.height, scope, ha="center", va="top", fontsize=8.5,
             color="#706E68")
    fig.subplots_adjust(top=1 - box.title_block / box.height,
                        bottom=_LEGEND_BLOCK / box.height, left=box.left,
                        right=box.right, wspace=_WSPACE, hspace=box.hspace)

    save_figure(fig, save, figure_dir=figure_dir)
    plt.show()
    return fig


def _state_figure(examples, state, view, stages, cells, metadata, organ, score,
                  min_cells, save, figure_dir, eligible_n, subtitle):
    """One rule: stages down the rows, the field, the rule, and what it counted."""
    rows = examples.set_index("stage")
    cell_colors = resolve_cell_colors(list(view.cells))

    captions = [view.captions.get(stage, "") for stage in stages]
    row_lines = max((_lines(text) for text in captions if text), default=1)
    # The first row carries a column label as well, and a long rule name wraps.
    first = stages[0] if stages else None
    head_lines = max(2, _lines(_wrapped(view.label))
                     + _lines(view.captions.get(first, "")))
    box = _layout(len(stages), 3, head_lines, row_lines)
    fig, axes = plt.subplots(len(stages), 3, figsize=(TEXT_WIDTH, box.height),
                             squeeze=False, facecolor="white",
                             gridspec_kw={"wspace": _WSPACE, "hspace": box.hspace})

    for row, stage in enumerate(stages):
        item = _row_of(rows, stage)
        if item is None:
            for ax in axes[row]:
                _blank(ax)
                ax.set_visible(False)
            axes[row, 0].set_visible(True)
            message = ("0 eligible FOVs" if eligible_n.get(stage) == 0
                       else f"No eligible {_STATE_NAMES[state].lower()} example")
            axes[row, 0].text((3 + 2 * _WSPACE) / 2, 0.5, message, ha="center",
                              va="center", color="#898781", fontsize=8.2)
        else:
            fov, where = item["FOV"], _where(item)
            _full_panel(axes[row, 0], fov, cells, metadata, cell_colors,
                        f"full FOV\n{where}" if row == 0 else where)
            _rule_panel(axes[row, 1], fov, cells, metadata, cell_colors, view, stage,
                        row == 0)
            _check_panel(axes[row, 2], fov, cells, metadata, view, row == 0)
            for ax in axes[row]:
                _bare(ax)
        axes[row, 0].set_ylabel(stage, fontsize=10, fontweight="bold", labelpad=8)

    return _frame(fig, box, organ, examples["rule"].iat[0],
                  subtitle or _SUBTITLES[state], _scope(score, stages, min_cells),
                  view.cells, cell_colors, save, figure_dir)


def _stage_figure(examples, state, stage, views, stages, cells, metadata, organ,
                  score, min_cells, save, figure_dir, subtitle):
    """One stage: the field and each rule above, what each rule counted below."""
    item = _row_of(examples.set_index("stage"), stage)
    if item is None:
        return None
    fov = item["FOV"]
    shown_cells = tuple(dict.fromkeys(cell for view in views for cell in view.cells))
    cell_colors = resolve_cell_colors(list(shown_cells))

    head_lines = max([2] + [_lines(_wrapped(view.label))
                            + _lines(view.captions.get(stage, "")) for view in views])
    box = _layout(2, 1 + len(views), head_lines, 2)
    fig, axes = plt.subplots(2, 1 + len(views), figsize=(TEXT_WIDTH, box.height),
                             squeeze=False, facecolor="white",
                             gridspec_kw={"wspace": _WSPACE, "hspace": box.hspace})

    _full_panel(axes[0, 0], fov, cells, metadata, cell_colors, f"full FOV\n{_where(item)}")
    for column, view in enumerate(views, start=1):
        _rule_panel(axes[0, column], fov, cells, metadata, cell_colors, view, stage, True)
        _check_panel(axes[1, column], fov, cells, metadata, view, True)
    for ax in axes.ravel():
        _bare(ax)
    axes[1, 0].set_visible(False)

    return _frame(fig, box, organ, examples["rule"].iat[0],
                  f"{subtitle or _SUBTITLES[state]} · {stage}",
                  _scope(score, stages, min_cells), shown_cells, cell_colors,
                  save, figure_dir)


def plot_rule_fovs(examples, stages, cells, metadata, organ, score,
                   min_cells=20, max_fdr=0.05, save=None, figure_dir=None, others=(),
                   subtitle=None):
    """One figure per state, or per state and stage when there is a rule to compare.

    `others` : (label, frame) pairs covering the same fields — a shorter rule, say.
    Each becomes one more rule in the figure, so the rules are compared in one
    field rather than across two figures.
    """
    eligible_n = getattr(examples, "attrs", {}).get("eligible_n", {})
    examples = pd.DataFrame(examples)
    if examples.empty:
        print("No representative FOVs to plot.")
        return []

    if "why" not in examples.columns:
        fields = rm.Fields(cells)
        examples = rm.explain_missing(examples, cells, fields=fields, max_fdr=max_fdr)
        others = [(label, rm.explain_missing(pd.DataFrame(frame), cells, fields=fields,
                                             max_fdr=max_fdr))
                  for label, frame in others]

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

        if len(views) == 1:
            figures.append(_state_figure(
                subset, state, views[0], stages, cells, metadata, organ, score,
                min_cells, _target(save, _STATE_SLUGS[state]), figure_dir,
                eligible_n, subtitle))
            continue
        for stage in stages:
            figure = _stage_figure(
                subset, state, stage, views, stages, cells, metadata, organ, score,
                min_cells, _target(save, _STATE_SLUGS[state], stage), figure_dir,
                subtitle)
            if figure is not None:
                figures.append(figure)
    return figures
