"""Drawing for the differential-rule analysis: which rules tell two groups apart.

Only plotting lives here; the tests and the tables stay in the notebook.
The shared pieces - saving, titles, group colours - come from `vis_helper.py`.
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from matplotlib.colors import to_rgba, to_hex
from matplotlib.lines import Line2D
from matplotlib.transforms import blended_transform_factory
from scipy.cluster.hierarchy import fcluster

import compare_rules as cr
import differential_stats as ds
from vis_helper import (save_figure, _titled, _category_colors, tidy_axes, spread_labels,
                        plot_fov, plot_fov_panel, set_cell_colors, TEXT_WIDTH, PANEL_FONTS,
                        _cell_share_table, _eligible_by_stage, _shares_panel,
                        NEUTRAL as _NEUTRAL, HAIRLINE as _HAIRLINE,
                        ZERO as _ZERO, INK as _INK, ATTRACTION, AVOIDANCE, KIND_COLORS, rule_name)
from matplotlib.colors import LinearSegmentedColormap

_LIFT_CMAP = LinearSegmentedColormap.from_list("lift", [AVOIDANCE, "#f7f7f5", ATTRACTION])
_UNTESTABLE = "#bdbcb5"


_FIGURE_DIR = Path(__file__).resolve().parent.parent / "summary_downloads"

_TEXT_WIDTH = TEXT_WIDTH
_PANEL_FONTS = PANEL_FONTS


def _finish(fig, save=None, dpi=200):
    """Save differential-rule figures beside their notebooks, then show them."""
    save_figure(fig, save, dpi=dpi, figure_dir=_FIGURE_DIR)
    plt.show()


def _group_colors(label_a, label_b):
    """One colour per group. Stages keep the colours they have everywhere else."""
    colors, _ = _category_colors(pd.Series([label_a, label_b]))
    return to_rgba(colors[label_a]), to_rgba(colors[label_b])


def _legend(ax, label_a, label_b, color_a, color_b, extra=()):
    """Two dots naming the groups above the axes, so colour is never the only clue, and any `extra` marks."""
    marks = [Line2D([0], [0], marker="o", linestyle="none", markersize=8,
                    markerfacecolor=c, markeredgecolor="white", label=l)
             for l, c in ((label_a, color_a), (label_b, color_b))] + list(extra)
    ax.legend(handles=marks, fontsize=9, frameon=False, ncol=min(len(marks), 3),
              loc="lower right", bbox_to_anchor=(1.0, 1.0))


def _kind_marks():
    """Legend marks for the rule kinds, as in the final figures: → attracts, ⊣ avoids, in their colours."""
    return [Line2D([0], [0], marker=marker, linestyle="none", markersize=11, color=KIND_COLORS[kind],
                   label=kind) for kind, marker in (("attracts", r"$\rightarrow$"), ("avoids", r"$\dashv$"))]


def _title(ax, text, scope):
    """Short figure title followed by a separate scope subtitle."""
    fig = ax.figure
    fig.suptitle(text, fontsize=13.5, y=0.985)
    if scope:
        fig.text(0.5, 0.943, str(scope), ha="center", va="top",
                 fontsize=8.5, color="#706E68")


# FDR values worth a tick. A short axis can afford the in-between values; a long one
# only has room for the decades, or the labels pile up on each other.
_TICKS_NEAR = [1, 0.5, 0.2, 0.1, 0.05, 0.01, 1e-3]
_TICKS_FAR = [1, 0.1, 0.05, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6]


def _fdr_ticks(ax, top):
    """Label the y axis with FDR values instead of -log10 of them."""
    wanted = _TICKS_NEAR if top <= 3.4 else _TICKS_FAR
    shown = [v for v in wanted if -np.log10(v) <= top]
    ax.set_yticks([-np.log10(v) for v in shown])
    ax.set_yticklabels([f"{v:g}" for v in shown])


def _place_names(ax, spots, top):
    """One name per row above the dots, each with a thin leader down to its dot.

    The surest rules all land on the same line - the permutation test cannot go below
    1 / (shuffles + 1) - and rule names are long, so there is no room sideways. A fixed
    ladder of rows cannot overlap, however crowded the plot gets.
    """
    if not spots:
        return
    floor = max(y for _, y, _, _ in spots) + 0.10 * top
    rows = np.linspace(0.97 * top, floor, len(spots))
    left, right = ax.get_xlim()
    for (x, y, text, color), row in zip(sorted(spots), rows):
        # Names near an edge lean inwards, so a long one never runs off the plot.
        side = "left" if x < 0.5 * left else "right" if x > 0.5 * right else "center"
        ax.annotate(text, xy=(x, y), xytext=(x, row), fontsize=7, color=color,
                    ha=side, va="center", linespacing=1.35, zorder=5,
                    arrowprops=dict(arrowstyle="-", color="0.6", lw=0.6,
                                    shrinkA=3, shrinkB=4))


def plot_volcano(result, kinds, first, last, max_fdr=0.05, min_effect=None, name_top=5, unit="FOVs",
                 scope=None, save=None):
    """Every tested rule as a dot: x its gap in mean log2 lift (`last` minus `first`), y its FDR, size
    its testable `unit` in both groups. The rules that pass `ds.passing` are coloured by the group
    they are higher in, and the `name_top` biggest gaps among them named with their `kinds` arrow and colour."""
    if result is None or result.empty:
        print("No data to plot.")
        return
    gap = result["gap"].to_numpy(float)
    height = -np.log10(result["fdr"].to_numpy(float))
    passed = ds.passing(result, max_fdr, min_effect).to_numpy()
    cutoff = -np.log10(max_fdr)
    color_last, color_first = _group_colors(last, first)
    colors = [color_last if p and g > 0 else color_first if p else to_rgba(_NEUTRAL)
              for p, g in zip(passed, gap)]
    testable = (result[f"n {first}"] + result[f"n {last}"]).to_numpy(float)
    sizes = 15 + 185 * testable / testable.max()

    fig, ax = plt.subplots(figsize=(9.5, 5.6))
    ax.axvline(0, color=_ZERO, lw=1.4, zorder=1)
    ax.axhline(cutoff, color=_HAIRLINE, lw=1.0, zorder=1)
    for side in ((-min_effect, min_effect) if min_effect is not None else ()):
        ax.axvline(side, color=_HAIRLINE, lw=1.0, ls=(0, (5, 4)), zorder=1)
    ax.scatter(gap, height, s=sizes, c=colors, alpha=0.9, edgecolor="white", linewidth=0.8, zorder=3)

    limit = float(np.abs(gap).max()) * 1.15 or 1.0
    ax.set_xlim(-limit, limit)
    top = max(float(height.max()), cutoff) * 1.4
    ax.set_ylim(-0.04 * top, top)
    _fdr_ticks(ax, top)

    named = pd.Series(np.abs(gap)).where(passed).dropna().nlargest(name_top).index
    together = {}
    for i in named:
        together.setdefault((round(gap[i], 6), round(height[i], 6)), []).append(result.index[i])
    _place_names(ax, [(x, y, "\n".join(rule_name(rule, kinds[rule]) for rule in names),
                       KIND_COLORS[kinds[names[0]]]) for (x, y), names in together.items()], top)

    ax.annotate(f"FDR {max_fdr:g}", xy=(-limit, cutoff), xytext=(3, 3),
                textcoords="offset points", fontsize=8, color=_INK)
    ax.set_xlabel(f"mean log2 lift: {last} − {first}  (→ closer in {last})", fontsize=10)
    ax.set_ylabel("FDR  (lower = surer)", fontsize=10)
    _title(ax, f"{first} vs {last}: rules whose lift differs", scope)
    size_marks = [Line2D([0], [0], marker="o", linestyle="none", markersize=np.sqrt(sizes[i]),
                         markerfacecolor=_NEUTRAL, markeredgecolor="white", label=f"{testable[i]:.0f} {unit}")
                  for i in (testable.argmin(), testable.argmax())]
    _legend(ax, f"higher in {last}", f"higher in {first}", color_last, color_first, size_marks + _kind_marks())
    tidy_axes(ax, grid="both")
    fig.tight_layout(rect=(0, 0, 1, 0.88))
    _finish(fig, save)


def plot_close_ups(table, groups, result, kinds, rules, first, last, max_fdr=0.05, min_effect=None,
                   unit="FOVs", scope=None, save=None):
    """One panel per rule of `rules`: the log2 lift of each testable unit in `first` and `last`
    (filled where found, hollow at 0 where not), each group's mean as a diamond, joined by the gap
    the volcano shows; titled with the rule in its `kinds` arrow and colour, and with the gap and FDR,
    coloured like the volcano when it passes."""
    rules = [rule for rule in rules if rule in result.index]
    if not rules:
        print("No rules to show.")
        return
    color_last, color_first = _group_colors(last, first)
    passed = ds.passing(result, max_fdr, min_effect)
    rng = np.random.default_rng(0)
    fig, axes = plt.subplots(1, len(rules), figsize=(3.2 * len(rules), 4.2), sharey=True, squeeze=False)
    for ax, rule in zip(axes[0], rules):
        means, labels = [], []
        for x, (group, color) in enumerate(((first, color_first), (last, color_last))):
            values = table.loc[rule, groups.index[groups.eq(group)]].dropna()
            found = values.ne(0).to_numpy()
            spread = x + rng.uniform(-0.18, 0.18, len(values))
            ax.scatter(spread[found], values[found], s=30, color=color, alpha=0.8,
                       edgecolor="white", linewidth=0.5, zorder=3)
            ax.scatter(spread[~found], values[~found], s=30, facecolor="none", edgecolor=color,
                       linewidth=0.8, alpha=0.6, zorder=3)
            means.append(values.mean())
            labels.append(f"{group}\n{len(values)} testable\n{int(found.sum())} found")
        ax.plot([0, 1], means, color=_INK, lw=1.4, zorder=4)
        ax.scatter([0, 1], means, marker="D", s=70, color=[color_first, color_last],
                   edgecolor=_INK, linewidth=0.9, zorder=5)
        ax.axhline(0, color=_ZERO, lw=1.1, zorder=1)
        ax.set_xticks([0, 1], labels)
        ax.set_xlim(-0.5, 1.5)
        gap, fdr = result.loc[rule, "gap"], result.loc[rule, "fdr"]
        shade = (color_last if gap > 0 else color_first) if passed[rule] else _INK
        ax.set_title(f"gap {gap:+.2f} · FDR {fdr:.2g}", fontsize=9.5, color=shade, pad=4)
        ax.text(0.5, 1.09, rule_name(rule, kinds[rule]), transform=ax.transAxes, ha="center", va="bottom",
                fontsize=10.5, color=KIND_COLORS[kinds[rule]])
        tidy_axes(ax, grid="y", hide=("top", "right"))
        ax.tick_params(length=0)
    axes[0][0].set_ylabel(f"log2 lift per {unit[:-1]}  (hollow = not found)", fontsize=10)
    _title(axes[0][0], "Close-ups: the values behind the volcano", scope)
    fig.tight_layout(rect=(0, 0, 1, 0.88))
    _finish(fig, save)


def _cluster_lines(tree, n_clusters, axes, across):
    """Cut `tree` into `n_clusters` and draw a dashed line between neighbouring clusters on `axes`,
    across the rows when `across`, down the columns otherwise. Returns each leaf's cluster, numbered
    1, 2, … in the order drawn."""
    labels = pd.factorize(fcluster(tree.linkage, n_clusters, "maxclust")[tree.reordered_ind])[0] + 1
    for edge in np.flatnonzero(labels[1:] != labels[:-1]) + 1:
        for ax in axes:
            (ax.axhline if across else ax.axvline)(edge, color=_INK, lw=1.3, ls=(0, (5, 3)))
    return labels


def _cluster_names(labels, ax, across):
    """C1, C2, … beside `ax`, at the middle of each cluster."""
    for n in np.unique(labels):
        middle = np.flatnonzero(labels == n).mean() + 0.5
        if across:
            ax.text(-0.3, middle, f"C{n}", transform=blended_transform_factory(ax.transAxes, ax.transData),
                    ha="right", va="center", fontsize=9, color=_INK)
        else:
            ax.text(middle, 1.3, f"C{n}", transform=blended_transform_factory(ax.transData, ax.transAxes),
                    ha="center", va="bottom", fontsize=9, color=_INK)


def _patient_colors(patients):
    """One colour per patient, shuffled so patients next to each other in the list do not look alike."""
    names = patients.dropna().unique()
    palette = np.random.default_rng(0).permutation(np.array(sns.color_palette("husl", len(names))))
    return patients.map(dict(zip(names, map(to_hex, palette)))).fillna(_NEUTRAL).rename("Patient")


def _bottom_strip(grid, colors, flip):
    """`colors` (one per unit) as a strip under the heatmap, or beside it when `flip`, taking over the units' labels."""
    ax = grid.ax_heatmap
    units = grid.data2d.index if flip else grid.data2d.columns
    rgba = np.array([to_rgba(c) for c in colors.reindex(units)])
    box = ax.get_position()
    size = grid.ax_row_colors.get_position().width if flip else grid.ax_col_colors.get_position().height
    labels = [label.get_text() for label in (ax.get_yticklabels() if flip else ax.get_xticklabels())]
    if flip:
        strip = ax.figure.add_axes([box.x1 + 0.4 * size, box.y0, size, box.height])
        strip.imshow(rgba[:, None], aspect="auto")
        strip.set_xticks([0], [colors.name], rotation=90)
        strip.yaxis.tick_right()
        strip.set_yticks(range(len(labels)), labels)
        ax.set_yticks([])
    else:
        strip = ax.figure.add_axes([box.x0, box.y0 - 1.4 * size, box.width, size])
        strip.imshow(rgba[None], aspect="auto")
        strip.yaxis.tick_right()
        strip.set_yticks([0], [colors.name])
        strip.set_xticks(range(len(labels)), labels, rotation=90)
        ax.set_xticks([])
    strip.tick_params(length=0)
    strip.spines[:].set_visible(False)


def plot_lift_heatmap(table, groups, kinds, clip=2.0, flip=False, scope=None, save=None, n_clusters=None,
                      linkage="average", rule_clusters=None, untestable_as=0, patients=None):
    """Rules x units of log2 lift, both sides clustered (euclidean distance, `linkage` method), with a
    strip coloured by the group of each unit and every rule named with its `kinds` arrow and colour;
    `flip` puts the units on the rows. Colours stop at ±`clip`. A NaN (not testable) is clustered as
    `untestable_as` (a number, or "mean": the rule's mean over its testable units) and drawn in its
    own grey. `n_clusters` and `rule_clusters` cut the units and the rules into that many clusters,
    split by dashed lines. `patients`, the patient of each unit, adds a strip coloured by patient.
    Returns the cluster of each unit (C1, C2, … in the order drawn) when `n_clusters` is given."""
    if table.shape[0] < 2 or table.shape[1] < 2:
        print("Not enough rules or units to cluster.")
        return
    groups = groups.reindex(table.columns)
    colors, order = _category_colors(groups)
    strip = groups.map(colors).fillna(_NEUTRAL).rename(groups.name)
    fill = table.mean(axis=1) if untestable_as == "mean" else untestable_as
    filled = table.T.fillna(fill).T.fillna(0)
    shown, filled = (table.T, filled.T) if flip else (table, filled)
    long = max(6, 0.22 * len(table))
    width, height = (long, 8) if flip else (12, long)
    few_units = table.shape[1] <= 80
    grid = sns.clustermap(
        filled, mask=shown.isna(), method=linkage, metric="euclidean",
        cmap=_LIFT_CMAP, center=0, vmin=-clip, vmax=clip,
        row_colors=strip if flip else None, col_colors=None if flip else strip,
        xticklabels=True if flip else few_units, yticklabels=few_units if flip else True,
        figsize=(width, height), dendrogram_ratio=(1.5 / width, 1.5 / height),
        cbar_pos=(0.02, 1 - 1.35 / height, 0.02, 1.2 / height), cbar_kws={"label": "log2 lift"})
    ax = grid.ax_heatmap
    ax.set_facecolor(_UNTESTABLE)
    unit_tree, rule_tree = ((grid.dendrogram_row, grid.dendrogram_col) if flip
                            else (grid.dendrogram_col, grid.dendrogram_row))
    clusters = None
    if n_clusters:
        strip_ax = grid.ax_row_colors if flip else grid.ax_col_colors
        labels = _cluster_lines(unit_tree, n_clusters, (ax, strip_ax), flip)
        _cluster_names(labels, strip_ax, flip)
        clusters = pd.Series([f"C{n}" for n in labels], index=grid.data2d.index if flip else grid.data2d.columns,
                             name="cluster")
    if rule_clusters:
        _cluster_lines(rule_tree, rule_clusters, (ax,), not flip)
    if patients is not None:
        _bottom_strip(grid, _patient_colors(patients.reindex(table.columns)), flip)
    ax.set_xlabel("")
    ax.set_ylabel("")
    rule_labels = ax.get_xticklabels if flip else ax.get_yticklabels
    rules = [label.get_text() for label in rule_labels()]
    (ax.set_xticklabels if flip else ax.set_yticklabels)([rule_name(rule, kinds[rule]) for rule in rules])
    for label, rule in zip(rule_labels(), rules):
        label.set_color(KIND_COLORS[kinds[rule]])
    marks = [Line2D([0], [0], marker="s", linestyle="none", markersize=9, markerfacecolor=colors[g],
                    markeredgecolor="none", label=g) for g in order]
    untestable = Line2D([0], [0], marker="s", linestyle="none", markersize=9, markerfacecolor=_UNTESTABLE,
                        markeredgecolor="none", label="not testable")
    grid.figure.legend(handles=marks + _kind_marks() + [untestable], frameon=False,
                       loc="lower right", bbox_to_anchor=(0.98, 1.0), ncol=len(marks) + 3)
    grid.figure.suptitle(f"log2 lift, clustered{f'  ·  {scope}' if scope else ''}", y=1 + 0.6 / height,
                         va="bottom", fontsize=13)
    _finish(grid.figure, save)
    return clusters


def plot_cluster_maps(clusters, cells, fovs, score, per_cluster=None, per_row=4, save=None):
    """One panel of cell maps per cluster of `clusters` (FOV -> cluster, in heatmap order), each map titled
    with its cluster, `score` group and patient; `per_cluster` maps picked evenly along the cluster, None for all."""
    set_cell_colors(cells)
    about = fovs.set_index("FOV")
    for name, members in clusters.groupby(clusters, sort=False):
        chosen = members.index
        if per_cluster and len(chosen) > per_cluster:
            chosen = chosen[np.linspace(0, len(chosen) - 1, per_cluster).round().astype(int)]
        print(f"{name}: {len(members)} FOVs, {len(chosen)} shown")
        plot_fov_panel({fov: f"{name} · {about.at[fov, score]} · {about.at[fov, 'PatientID']}"
                        for fov in chosen}, cells, fovs, num_cols=per_row, one_legend=True,
                       save=save and str(_FIGURE_DIR / f"{save}_{name.lower()}"))


# One hue per rule, so a line can be followed across the stages and found again in the
# names. Kept to eight - past that neighbouring hues stop being told apart - and ordered
# so the ones drawn next to each other are the furthest apart in colour.
_LINE_COLORS = ["#2a78d6", "#eb6834", "#199e70", "#c98500",
                "#d55181", "#4a3aa7", "#008300", "#e34948"]


def _cell_types_in(rules):
    """Every cell type named in these rules, in the order they first come up."""
    seen = []
    for rule in rules:
        for part in str(rule).replace("->", ",").split(","):
            name = part.strip()
            if name and name not in seen:
                seen.append(name)
    return seen


def _trend_panel(ax, nets, names, slopes, steady, title, stages, unit="", colors=None):
    """One direction's rules: a line per rule, named outside on the left with an arrow in."""
    ax.set_title(title, fontsize=10.5, color=_INK, loc="left", pad=14)
    if not len(nets):
        ax.annotate("none", xy=(0.5, 0.5), xycoords="axes fraction", ha="center",
                    va="center", fontsize=9, color="#96958f")
        ax.set_xticks([])
        ax.set_yticks([])
        for side in ax.spines.values():
            side.set_visible(False)
        ax.grid(False)
        return

    # The unit rides under the title instead of on a rotated y label, which would have to
    # share the left margin with the names.
    notes = [unit] + (["dashed: went back at one step"] if not steady.all() else [])
    ax.annotate("      ".join(n for n in notes if n), xy=(0, 1.012),
                xycoords="axes fraction", fontsize=8, color="#898781")

    # Only this panel's own lines set the scale - an empty half-panel just to match the
    # other side helps nobody.
    low, high = float(nets.min()), float(max(nets.max(), 0.0))
    pad = max((high - low) * 0.10, 2.0)
    ax.set_ylim(low - pad, high + pad)

    x = np.arange(len(stages))
    ax.set_xlim(-0.08, len(stages) - 0.92)
    rows = spread_labels(nets[:, 0], (high - low + 2 * pad) * 0.075, low - pad * 0.6, high + pad * 0.6)

    for i, (row, name, slope, is_steady, at) in enumerate(
            zip(nets, names, slopes, steady, rows)):
        color = (colors or {}).get(name) or _LINE_COLORS[i % len(_LINE_COLORS)]
        # Dashed means the rule went back at one step: the drift is real, but not gradual.
        ax.plot(x, row, color=color, lw=1.8, alpha=0.9, solid_capstyle="round", zorder=3,
                linestyle="-" if is_steady else (0, (4.5, 2.5)))
        ax.scatter(x, row, s=24, color=color, edgecolor="white", linewidth=1.1, zorder=4)
        # A slope that rounds to nothing is written '0', never '-0'.
        moved = f"{slope:+.0f}" if abs(slope) >= 0.5 else "0"
        # Outside the panel, right-aligned so the names end in a straight column, with an
        # arrow in to the line's own starting point - the only way to tell crowded starts apart.
        ax.annotate(f"{name}   {moved}", xy=(x[0], row[0]), xytext=(-0.15, at),
                    textcoords=("axes fraction", "data"), annotation_clip=False,
                    fontsize=8, va="center", ha="right", color=color, zorder=5,
                    arrowprops=dict(arrowstyle="->", color=color, alpha=0.5, lw=0.8,
                                    shrinkA=4, shrinkB=4))

    ax.set_xticks(x, stages, fontsize=10)


def _movers(trend, stages, fdr_threshold, top_n, rising=None):
    """The rows a panel draws: the steepest few that passed, optionally one direction only."""
    nets = trend[[f"net_{stage}" for stage in stages]].to_numpy(dtype=float) * 100
    slope = trend["slope"].to_numpy(dtype=float) * 100
    wanted = (trend["fdr"] <= fdr_threshold).to_numpy(copy=True)
    if rising is not None:
        wanted &= slope > 0 if rising else slope < 0

    order = np.argsort(-np.abs(np.where(wanted, slope, 0)))[:int(wanted.sum())][:top_n]
    return (nets[order], trend.index[order].astype(str), slope[order],
            trend["steady"].to_numpy()[order], int(wanted.sum()))


def _plain(ax):
    """The chrome every trend panel shares: a zero line, hairline grid, no frame."""
    ax.axhline(0, color=_ZERO, lw=1.5, zorder=2)
    tidy_axes(ax, grid="y", hide=("top", "right", "bottom", "left"))
    ax.tick_params(length=0, labelsize=9, colors="#898781")


_NET_LABEL = "net: attracts minus avoids (% of the stage's FOVs)"
_SHARE_LABEL = "share of the cells (%)"


def _room_for_names(fig, top=0.845):
    """Margins the names can live in: they sit outside the panels, where tight_layout
    cannot see them, so the space is set by hand."""
    fig.subplots_adjust(left=0.205, right=0.99, top=top, bottom=0.09, wspace=0.63)


def plot_rules_with_cells(trend, cell_trend, stages, rising=True, scope=None,
                          fdr_threshold=0.05, top_n=8, cell_colors=None, save=None):
    """One direction's rules beside the cell types those rules are built from.

    Same three stages on both sides, each panel on its own scale - a rule's net is a
    share of FOVs and a cell type's share is a share of cells, so one axis for both would
    invent a relationship. Read across: if the rules move and their cell types do not,
    the change is in how the cells are arranged, not in how many of them there are.
    """
    nets, names, slopes, steady, found = _movers(trend, stages, fdr_threshold, top_n, rising)
    way = "Climbing" if rising else "Fading"
    if not len(nets):
        print(f"No {way.lower()} rules to plot.")
        return

    # Only the cell types those rules are made of, steepest first so the busiest ones
    # are the ones that get a colour of their own.
    types = [t for t in _cell_types_in(names) if t in cell_trend.index]
    shown = cell_trend.loc[types].reindex(cell_trend.loc[types]["slope"].abs()
                                          .sort_values(ascending=False).index)
    if cell_colors is None:
        shown = shown.head(len(_LINE_COLORS))

    fig, axes = plt.subplots(1, 2, figsize=(14.5, 5.4))
    _trend_panel(axes[0], nets, names, slopes, steady,
                 f"the rules   ({found} {way.lower()})", stages, unit=_NET_LABEL)
    _trend_panel(axes[1], shown[[f"net_{s}" for s in stages]].to_numpy(dtype=float) * 100,
                 shown.index.astype(str), shown["slope"].to_numpy(dtype=float) * 100,
                 shown["steady"].to_numpy(), "how common those cell types are", stages,
                 unit=_SHARE_LABEL, colors=cell_colors)
    for ax in axes:
        _plain(ax)

    fig.suptitle(f"{way} rules and the cells behind them (top {top_n})",
                 fontsize=13, y=0.985)
    if scope:
        fig.text(0.5, 0.943, str(scope), ha="center", va="top",
                 fontsize=8.5, color="#706E68")
    _room_for_names(fig)
    _finish(fig, save)


def plot_trend(trend, stages, scope=None, fdr_threshold=0.05, top_n=8,
               what="Rules", value_label=None, save=None):
    """Rules whose net climbs or fades as the disease worsens, one line per rule.

    One point per stage, so the line itself is the trend the test measured, and the
    number beside each name is its slope. Climbing and fading are drawn apart,
    otherwise the lines cross and neither is readable. Any table `compare_stages`
    returns can be drawn - give `what` and `value_label` for cell types instead of rules.
    """
    thing = what.rstrip("s").lower()             # 'Cell types' -> 'cell type'

    movers = {
        rising: _movers(trend, stages, fdr_threshold, top_n, rising)
        for rising in (True, False)
    }
    # Nothing passing is a result worth stating, but it does not need a full-height
    # panel of empty axes to state it.
    height = 5.4 if any(len(m[0]) for m in movers.values()) else 1.7

    # Each panel keeps its own scale: the two directions rarely cover the same range, and
    # a shared one leaves whichever panel is smaller as mostly empty space.
    fig, axes = plt.subplots(1, 2, figsize=(14.5, height))
    for ax, rising, title in ((axes[0], True, "climbing with severity"),
                              (axes[1], False, "fading with severity")):
        nets, names, slopes, steady, found = movers[rising]
        _trend_panel(ax, nets, names, slopes, steady,
                     f"{title}   ({found} {thing}{'s' if found != 1 else ''})", stages,
                     unit=value_label or _NET_LABEL)
        _plain(ax)

    title_y = 0.985 if height > 3 else 0.93
    fig.suptitle(f"{what} that move with severity (top {top_n} each way)",
                 fontsize=13, y=title_y)
    if scope:
        fig.text(0.5, title_y - 0.042 * (5.4 / height), str(scope), ha="center",
                 va="top", fontsize=8.5, color="#706E68")
    _room_for_names(fig, top=0.845 if height > 3 else 0.52)
    _finish(fig, save)


def plot_state_breakdown(states, selected, fovs_by_stage, stages, scope=None, save=None):
    """For selected rules, split each stage into attract, avoid, and no-rule FOVs."""
    if selected.empty:
        print("No eligible trends passed the cutoff.")
        return

    colors = {"Attract": ATTRACTION, "No rule": "#dddcd5", "Avoid": AVOIDANCE}
    ncols = 2 if len(selected) > 1 else 1
    nrows = int(np.ceil(len(selected) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(11, 2.65 * nrows + 1),
                             squeeze=False)

    for ax, rule in zip(axes.flat, selected.index):
        shares, totals = [], []
        for stage in stages:
            values = states.reindex(columns=fovs_by_stage[stage]).loc[rule].dropna()
            totals.append(len(values))
            shares.append([100 * (values == value).mean() for value in (1, 0, -1)])

        shares = np.asarray(shares)
        bottom = np.zeros(len(stages))
        for column, label in enumerate(("Attract", "No rule", "Avoid")):
            bars = ax.bar(np.arange(len(stages)), shares[:, column], bottom=bottom,
                          color=colors[label], width=0.68, label=label)
            for bar, percent in zip(bars, shares[:, column]):
                if percent >= 10:
                    ax.text(bar.get_x() + bar.get_width() / 2,
                            bar.get_y() + bar.get_height() / 2, f"{percent:.0f}%",
                            ha="center", va="center", fontsize=8,
                            color="white" if label != "No rule" else _INK)
            bottom += shares[:, column]

        slope = selected.loc[rule, "slope"] * 100
        arrow = "increasing" if slope > 0 else "decreasing"
        ax.set_title(f"{rule}\n{arrow} {abs(slope):.0f} pp/stage · "
                     f"FDR {selected.loc[rule, 'fdr']:.3g}", fontsize=9.5, loc="left")
        ax.set_xticks(np.arange(len(stages)),
                      [f"{stage}\nn={total}" for stage, total in zip(stages, totals)])
        ax.set_ylim(0, 100)
        ax.set_ylabel("eligible FOVs (%)", fontsize=8)
        tidy_axes(ax, grid="y", hide=("top", "right", "left"))
        ax.tick_params(length=0, labelsize=8)

    for ax in axes.flat[len(selected):]:
        ax.set_visible(False)

    handles = [Line2D([0], [0], marker="s", linestyle="none", markersize=8,
                      markerfacecolor=colors[label], markeredgecolor="none", label=label)
               for label in ("Attract", "No rule", "Avoid")]
    fig.legend(handles=handles, loc="upper center", ncol=3, frameon=False,
               bbox_to_anchor=(0.5, 0.89))
    fig.suptitle("What changes inside the strongest eligible trends",
                 fontsize=13, y=0.99)
    if scope:
        fig.text(0.5, 0.947, str(scope), ha="center", va="top",
                 fontsize=8.5, color="#706E68")
    fig.subplots_adjust(top=0.78, hspace=0.72, wspace=0.32)
    _finish(fig, save)


# ---------------------------------------------------------------------------
# Focused evidence views used by the paper-alignment and time notebooks
# ---------------------------------------------------------------------------

_STATE_COLORS = {
    "Attraction": ATTRACTION,
    "No rule": "#E8E7E2",
    "Avoidance": AVOIDANCE,
    "Rule not testable": "#B8B4C7",
}

_STAGE_COLORS = {"Control": "#67B58A", "Mild": "#E7A33E", "Severe": "#D85D62",
                 "<30": "#9E8FD0", "30-100": "#7258B8", ">100": "#4A2F8F"}
_STATE_ORDER = ["Attraction", "No rule", "Avoidance", "Rule not testable"]


def _rule_test_text(spec, trend_results=None, pair_results=None, test_label=None):
    """Compact corrected-test summary for one rule panel."""
    rule, organ, group_col = spec["rule"], spec["organ"], spec["score"]
    scope_key = (organ, group_col)
    trend = (trend_results or {}).get(scope_key, pd.DataFrame())
    pair = (pair_results or {}).get(scope_key, pd.DataFrame())
    tests = []
    prefix = f"{test_label} " if test_label else ""
    if rule in trend.index:
        tests.append(f"ordered {prefix}FDR {trend.at[rule, 'fdr']:.3g}")
    if isinstance(pair, dict):
        labels = {
            ("Control", "Mild"): "C–M",
            ("Mild", "Severe"): "M–S",
            ("Control", "Severe"): "C–S",
        }
        for comparison, result in pair.items():
            if rule in result.index:
                label = labels.get(comparison, "–".join(comparison))
                tests.append(f"{label} {prefix}FDR {result.at[rule, 'fdr']:.3g}")
    elif rule in pair.index:
        tests.append(f"C–S {prefix}FDR {pair.at[rule, 'fdr']:.3g}")
    return " · ".join(tests) if tests else spec.get("test_text", "")


def _stage_state_counts(states, eligibility, metadata, rule, organ, group_col,
                        stages, unit_col="FOV"):
    """Counts and net scores for one rule in each displayed stage."""
    stage_counts, eligible_counts, nets = [], [], []
    for stage in stages:
        mask = (metadata["Organ"] == organ) & (metadata[group_col] == stage)
        units = metadata.loc[mask, unit_col].drop_duplicates()
        state = states.reindex(columns=units).loc[rule]
        can_test = eligibility.reindex(columns=units).loc[rule]
        eligible_n = int(can_test.sum())
        counts = {
            "Attraction": int((can_test & (state > 0)).sum()),
            "No rule": int((can_test & (state == 0)).sum()),
            "Avoidance": int((can_test & (state < 0)).sum()),
            "Rule not testable": int((~can_test).sum()),
        }
        stage_counts.append(counts)
        eligible_counts.append((eligible_n, len(units)))
        nets.append(
            100 * (counts["Attraction"] - counts["Avoidance"]) / eligible_n
            if eligible_n else np.nan
        )
    return stage_counts, eligible_counts, nets


def _draw_state_bars(ax, stage_counts, eligible_counts, nets, stages,
                     unit_label="FOVs", label_threshold=8, compact=False, show_net=True):
    """Draw the shared all-unit rule-state bars."""
    x = np.arange(len(stages))
    bottom = np.zeros(len(stages))
    ax.set_ylim(0, 114)          # set first: the labels below size themselves against it
    for label in _STATE_ORDER:
        percentages = np.array([
            100 * counts[label] / total if total else 0
            for counts, (_, total) in zip(stage_counts, eligible_counts)
        ])
        bars = ax.bar(
            x, percentages, bottom=bottom, width=0.64,
            color=_STATE_COLORS[label], edgecolor="white", linewidth=0.8,
        )
        # A band only gets the count over the percentage when it is tall enough to
        # hold two lines; a middling band gets them side by side, a thin one nothing.
        axes_points = ax.get_window_extent().height * 72 / ax.figure.dpi
        span = np.diff(ax.get_ylim())[0] or 114
        for bar, percent, counts in zip(bars, percentages, stage_counts):
            band_points = percent / span * axes_points
            if percent < label_threshold or band_points < 9:
                continue
            ink = _INK if label == "No rule" else "white"
            two_lines = band_points >= 20
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_y() + bar.get_height() / 2,
                f"{counts[label]}\n{percent:.0f}%" if two_lines
                else f"{counts[label]} · {percent:.0f}%",
                ha="center", va="center", fontsize=8, color=ink,
            )
        bottom += percentages

    for position, net in enumerate(nets):
        if not show_net:
            continue
        prefix = "" if compact else "net "
        text = f"{prefix}n/a" if np.isnan(net) else f"{prefix}{net:+.0f}%"
        ax.text(position, 103, text, ha="center", va="bottom",
                fontsize=7 if compact else 8.5,
                color="#454440", fontweight="bold")
    ax.set_xticks(x, [
        f"{stage}\n{eligible_n}/{total}" if compact
        else f"{stage}\n{eligible_n}/{total} eligible"
        for stage, (eligible_n, total) in zip(stages, eligible_counts)
    ])
    ax.set_ylabel(f"all {unit_label} (%)")
    tidy_axes(ax, grid="y", hide=("top", "right", "left", "bottom"))
    ax.tick_params(length=0)


def plot_rule_states(states, eligibility, metadata, specs, stages, heading,
                     trend_results=None, pair_results=None, unit_col="FOV",
                     unit_label="FOVs", test_label=None, save=None):
    """Show attraction, no rule, avoidance and insufficient cells for every unit.

    Each item in ``specs`` names a rule, organ and grouping column. The full bar uses
    every unit in that group, so loss of a cell type remains visible instead of being
    hidden by the eligibility filter.
    """
    n_panels = len(specs)
    figure_height = 3.7 * n_panels + 1.8
    fig, axes = plt.subplots(
        n_panels, 1, figsize=(11.8, figure_height), squeeze=False,
    )
    axes = axes[:, 0]
    groupings = list(dict.fromkeys(
        spec["score"].replace(" score", "").lower() for spec in specs
    ))
    grouping_scope = (
        f"{groupings[0]} score" if len(groupings) == 1
        else " + ".join(f"{grouping} score" for grouping in groupings)
    )
    figure_subtitle = (
        f"Analysis: pairwise rule states · Unit: FOV · Score: {grouping_scope} · "
        f"Stages: {' / '.join(stages)} · Eligibility: rule can be tested · "
        "Bars: all FOVs"
    )
    if any(isinstance(value, dict) for value in (pair_results or {}).values()):
        figure_subtitle += (
            "\nFDR correction: all rules and C–M / M–S / C–S comparisons"
        )

    for ax, spec in zip(axes, specs):
        rule, organ, group_col = spec["rule"], spec["organ"], spec["score"]
        summary = _stage_state_counts(
            states, eligibility, metadata, rule, organ, group_col, stages, unit_col,
        )
        _draw_state_bars(ax, *summary, stages, unit_label)

        panel = spec.get("panel", rule.replace(" -> ", " → "))
        tests = _rule_test_text(spec, trend_results, pair_results, test_label)
        detail = group_col.lower()
        if tests:
            detail += f" · {tests}"
        ax.set_title(panel, fontsize=10.8, pad=31, loc="center")
        ax.text(0.5, 1.02, detail, transform=ax.transAxes, ha="center",
                va="bottom", fontsize=8.6, color="#66645F")

    handles = [
        Line2D([0], [0], marker="s", linestyle="none", markersize=9,
               markerfacecolor=_STATE_COLORS[label], markeredgecolor="none", label=label)
        for label in _STATE_ORDER
    ]
    # Keep the header a fixed physical size. Fixed fractional positions create
    # several inches of empty space when a figure contains many rule panels.
    title_y = 1 - 0.15 / figure_height
    subtitle_y = 1 - 0.63 / figure_height
    legend_y = 1 - 1.05 / figure_height
    axes_top = 1 - 1.55 / figure_height
    fig.legend(handles=handles, ncol=4, frameon=False, loc="upper center",
               bbox_to_anchor=(0.5, legend_y))
    fig.suptitle(heading, fontsize=14.5, x=0.5, ha="center", y=title_y)
    fig.text(0.5, subtitle_y, figure_subtitle, ha="center", va="top", fontsize=8.6,
             color="#66645F", linespacing=1.5)
    fig.subplots_adjust(left=0.12, right=0.985, top=axes_top, bottom=0.07, hspace=0.9)
    _finish(fig, save)


def _metric_panel(ax, metrics, eligibility, metadata, spec, stages, metric="Lift",
                  reference=1, unit_col="FOV", state=None, connect=True):
    """One axes of raw metric values, drawn only where an eligible FOV has the rule."""
    rule, organ, group_col = spec["rule"], spec["organ"], spec["score"]
    columns = [unit_col, metric] + (["state"] if state is not None else [])
    rows = metrics.loc[metrics["Clean_Rule"] == rule, columns].copy()
    if state is not None:
        rows = rows[rows["state"] == state]
    values_by_fov = rows.dropna(subset=[metric]).groupby(unit_col)[metric].mean()
    plot_rows, labels, medians = [], [], []

    for position, stage in enumerate(stages):
        fovs = metadata.loc[
            (metadata["Organ"] == organ) & (metadata[group_col] == stage), unit_col
        ].drop_duplicates()
        can_test = eligibility.reindex(columns=fovs).loc[rule]
        eligible_fovs = can_test.index[can_test]
        values = values_by_fov.reindex(eligible_fovs).dropna()
        plot_rows.extend(
            {"stage": stage, "position": position, metric: value}
            for value in values
        )
        labels.append(f"{stage}\n{len(values)}/{len(eligible_fovs)}")
        medians.append(values.median() if len(values) else np.nan)

    data = pd.DataFrame(plot_rows)
    if not data.empty:
        colors = {stage: _STAGE_COLORS.get(stage, "#777570") for stage in stages}
        sns.stripplot(
            data=data, x="stage", y=metric, hue="stage", legend=False,
            order=stages, palette=colors,
            jitter=0.18, size=5, alpha=0.72, edgecolor="white", linewidth=0.6, ax=ax,
        )
        if connect:
            ax.plot(range(len(stages)), medians, color="#777570", lw=1.5, zorder=4)
        ax.scatter(range(len(stages)), medians, marker="D", s=58,
                   color=[colors[stage] for stage in stages],
                   edgecolor="white", linewidth=0.8, zorder=5)
    if reference is not None:
        ax.axhline(reference, color=_ZERO, lw=1.1)
    ax.set_xticks(range(len(stages)), labels)
    ax.set_xlabel("")
    ax.set_xlim(-0.5, len(stages) - 0.5)
    direction = {1: "attraction", -1: "avoidance"}.get(state)
    ax.set_ylabel(f"{direction.title() + ' ' if direction else ''}{metric} where the rule fires")
    tidy_axes(ax, grid="y", hide=("top", "right"))
    ax.tick_params(length=0)


def plot_rule_metric(metrics, eligibility, metadata, spec, stages, metric="Lift",
                     reference=1, heading=None, unit_col="FOV", state=None,
                     trend_results=None, pair_results=None, save=None):
    """The metric panel on its own, titled and captioned."""
    rule, organ, group_col = spec["rule"], spec["organ"], spec["score"]
    fig, ax = plt.subplots(figsize=(8.8, 4.4))
    _metric_panel(ax, metrics, eligibility, metadata, spec, stages, metric,
                  reference, unit_col, state)
    direction = {1: "attraction", -1: "avoidance"}.get(state)
    label = f"{direction.title()} {metric}" if direction else metric
    tests = _rule_test_text(
        spec, trend_results, pair_results,
        f"{direction} {metric}" if direction else metric,
    )
    detail = (
        f"Analysis: pairwise rules · Unit: FOV · Organ: {organ} · "
        f"Score: {group_col.replace(' score', '').lower()} · Stages: {' / '.join(stages)} · "
        "Eligibility: rule can be tested · "
        f"State: {direction or 'either'} · Metric: {metric}"
        "\nDots: rule-bearing FOVs · Diamonds: stage medians"
    )
    if tests:
        detail += f"\nTests: {tests}"
    fig.suptitle(heading or f"{rule.replace(' -> ', ' → ')} — {label}",
                 fontsize=13.5, y=0.99)
    fig.text(0.5, 0.935, detail, ha="center", va="top", fontsize=8.3,
             color="#706E68", linespacing=1.35)
    fig.subplots_adjust(top=0.72, bottom=0.19, left=0.12, right=0.98)
    _finish(fig, save)


def plot_time_coverage(metadata, day_col, group_col, groups, save=None):
    """One dot per biopsy, showing where the temporal evidence is concentrated."""
    data = metadata.dropna(subset=[day_col]).drop_duplicates("Biopsy").copy()
    colors = {"<30": "#58A6A6", "30-100": "#E3A33D", ">100": "#806FB3"}
    fig, ax = plt.subplots(figsize=(10.5, 3.5))
    organs = [organ for organ in ("Colon", "Duodenum") if organ in set(data["Organ"])]
    y_by_organ = {organ: i for i, organ in enumerate(organs)}
    rng = np.random.default_rng(4)
    for group in groups:
        part = data[data[group_col] == group]
        y = np.array([y_by_organ[o] for o in part["Organ"]], float)
        y += rng.uniform(-0.09, 0.09, len(part))
        ax.scatter(part[day_col], y, s=54, color=colors.get(group, _NEUTRAL),
                   edgecolor="white", linewidth=0.8, alpha=0.85, label=group)
    ax.set_yticks(range(len(organs)), organs)
    ax.set_xlabel("days after transplantation")
    ax.set_ylabel("")
    fig.suptitle("Biopsy coverage across time", fontsize=13, y=0.99)
    fig.text(
        0.5, 0.935,
        f"Unit: biopsy · Time variable: {day_col.lower()} · Groups: {' / '.join(groups)} · Color: time group",
        ha="center", va="top", fontsize=8.5, color="#706E68",
    )
    ax.legend(title="post-transplant window", frameon=False, ncol=len(groups),
              loc="lower center", bbox_to_anchor=(0.5, 1.08))
    tidy_axes(ax, grid="x", hide=("top", "right", "left"))
    ax.tick_params(axis="y", length=0)
    fig.subplots_adjust(top=0.67)
    _finish(fig, save)


def plot_temporal_screen(result, scope=None, fdr_threshold=0.05, name_top=8, save=None):
    """All tested rules: largest between-bin change against temporal FDR."""
    if result.empty:
        print("No temporal rules could be tested.")
        return
    x = result["range"].to_numpy(float) * 100
    fdr = result["fdr"].to_numpy(float)
    y = -np.log10(np.clip(fdr, 1e-300, None))
    passed = fdr <= fdr_threshold
    fig, (ax, key) = plt.subplots(
        1, 2, figsize=(12, 5.4), gridspec_kw={"width_ratios": [4.2, 1.7]},
    )
    ax.scatter(x[~passed], y[~passed], s=28, color=_NEUTRAL, alpha=0.55,
               edgecolor="none")
    ax.scatter(x[passed], y[passed], s=48, color="#6A5AA8", alpha=0.9,
               edgecolor="white", linewidth=0.7)
    ax.axhline(-np.log10(fdr_threshold), color=_HAIRLINE, lw=1.1)
    names = result.sort_values(["fdr", "range"], ascending=[True, False]).head(name_top)
    colors = sns.color_palette("colorblind", n_colors=max(len(names), 1))
    for color, (rule, row) in zip(colors, names.iterrows()):
        px = 100 * row["range"]
        py = -np.log10(max(row["fdr"], 1e-300))
        ax.scatter(px, py, s=62, facecolor=color, edgecolor="white",
                   linewidth=0.8, zorder=4)

    key.axis("off")
    status = ("FDR-significant signals" if passed.any()
              else "Largest exploratory signals\n(none passes FDR 0.05)")
    key.set_title(status, fontsize=10.5, loc="left", pad=14)
    for rank, (color, (rule, row)) in enumerate(zip(colors, names.iterrows()), start=1):
        ypos = 1 - (rank - 0.4) / max(name_top, len(names))
        key.scatter(0.02, ypos - 0.005, s=48, color=color, edgecolor="white",
                    linewidth=0.7, transform=key.transAxes)
        key.text(0.08, ypos, rule.replace(" -> ", " → "),
                 transform=key.transAxes, fontsize=8.5, fontweight="bold",
                 va="top", color=_INK)
        key.text(0.08, ypos - 0.055,
                 f"gap {100 * row['range']:.0f} pp · FDR {row['fdr']:.3g}",
                 transform=key.transAxes, fontsize=7.8, va="top", color="#706E68")
    top = max(-np.log10(fdr_threshold) + 0.15, np.nanmax(y) * 1.08)
    ax.set_ylim(-0.05, top)
    _fdr_ticks(ax, top)
    ax.set_xlabel("largest gap between any two post-transplant windows (percentage points)")
    ax.set_ylabel("FDR across all tested rules  (smaller = stronger)")
    ax.set_title(_titled("Temporal pair-rule screen", scope), fontsize=13, pad=38)
    # Two lines: one long one runs out of the axes and under the key's heading.
    ax.text(
        0, 1.015,
        ("Analysis: pairwise rules · Unit: biopsy · Groups: <30 / 30–100 / >100 days\n"
         f"Eligibility: rule can be tested · Tested rules: {len(result)}"),
        transform=ax.transAxes, ha="left", va="bottom", fontsize=8.5,
        color="#706E68", linespacing=1.4,
    )
    tidy_axes(ax, grid="both", hide=("top", "right"))
    fig.subplots_adjust(wspace=0.08)
    _finish(fig, save)


def plot_time_profiles(values, metadata, rules, result, groups, group_col,
                       scope=None, save=None):
    """Biopsy-level rule scores across Control and post-transplant time bins."""
    rules = [rule for rule in rules if rule in values.index]
    if not rules:
        print("No requested temporal rules are available.")
        return
    ncols = 2 if len(rules) > 1 else 1
    nrows = int(np.ceil(len(rules) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(12, 3.7 * nrows + 0.8),
                             squeeze=False)
    colors = {"Control": "#67B58A", "<30": "#58A6A6",
              "30-100": "#E3A33D", ">100": "#806FB3"}
    rng = np.random.default_rng(7)

    for ax, rule in zip(axes.flat, rules):
        means = []
        for position, group in enumerate(groups):
            units = metadata.loc[metadata[group_col] == group, "Biopsy"].drop_duplicates()
            observed = values.reindex(columns=units).loc[rule].dropna() * 100
            jitter = rng.uniform(-0.12, 0.12, len(observed))
            ax.scatter(position + jitter, observed, s=33, color=colors.get(group, _NEUTRAL),
                       alpha=0.62, edgecolor="white", linewidth=0.6)
            mean = observed.mean() if len(observed) else np.nan
            means.append(mean)
            if len(observed):
                ax.scatter(position, mean, s=92, marker="D", color=colors.get(group, _INK),
                           edgecolor="white", linewidth=1.2, zorder=5)
            ax.text(position, -108, f"eligible n={len(observed)}", ha="center", va="top",
                    fontsize=8, color="#66645F")
        ax.plot(range(len(groups)), means, color="#4A4844", lw=1.4, alpha=0.7,
                zorder=3)
        ax.axhline(0, color=_ZERO, lw=1.2)
        ax.set_xticks(range(len(groups)), groups)
        ax.set_ylim(-118, 108)
        ax.set_ylabel("biopsy rule balance (pp)\nattraction FOVs − avoidance FOVs")
        evidence = f"temporal FDR {result.at[rule, 'fdr']:.3g}" if rule in result.index else "context rule"
        ax.set_title(f"{rule.replace(' -> ', ' → ')}\n{evidence}", fontsize=10, pad=8)
        tidy_axes(ax, grid="y", hide=("top", "right"))
        ax.tick_params(length=0)

    for ax in axes.flat[len(rules):]:
        ax.set_visible(False)
    parts = str(scope or "").split(" — ", 1)
    organ = parts[0] if parts and parts[0] else ""
    context = parts[1] if len(parts) > 1 else ""
    title = _titled("Biopsy-level temporal pair-rule profiles", organ)
    subtitle = (f"Analysis: pairwise rules · Unit: biopsy · Groups: {' / '.join(groups)} · "
                "FOVs: eligible only")
    if context:
        subtitle = f"Context: {context} · {subtitle}"
    fig.suptitle(title, fontsize=14, y=0.99)
    fig.text(0.5, 0.947, subtitle, ha="center", va="top", fontsize=8.5,
             color="#706E68")
    fig.subplots_adjust(top=0.81, hspace=0.62, wspace=0.28)
    _finish(fig, save)


def plot_pooled_time_contrasts(results, contrasts, organ, fdr_threshold=0.05,
                               name_top=6, save=None):
    """Two pooled biopsy-level time contrasts with one joint FDR family."""
    shown = [(key, contrasts[key], results[key]) for key in contrasts if key in results]
    if not shown:
        print("No pooled temporal contrasts could be tested.")
        return

    total_tests = sum(len(result) for _, _, result in shown)
    fig, axes = plt.subplots(
        len(shown), 2, figsize=(12, 4.25 * len(shown) + 1.4),
        gridspec_kw={"width_ratios": [4.8, 1.7]}, squeeze=False,
    )

    for row, (key_name, spec, result) in enumerate(shown):
        ax, key = axes[row]
        gap = result["effect_size"].to_numpy(float) * 100
        fdr = result["fdr"].to_numpy(float)
        evidence = -np.log10(np.clip(fdr, 1e-300, None))
        passed = fdr <= fdr_threshold
        cutoff = -np.log10(fdr_threshold)

        ax.scatter(gap, evidence, s=28, color=_NEUTRAL, alpha=0.55,
                   edgecolor="none")
        ax.axvline(0, color=_ZERO, lw=1.2)
        ax.axhline(cutoff, color=_HAIRLINE, lw=1.1)

        names = result.sort_values(
            ["fdr", "effect_size"], key=lambda s: s.abs() if s.name == "effect_size" else s,
            ascending=[True, False],
        ).head(name_top)
        colors = sns.color_palette("colorblind", n_colors=max(len(names), 1))
        for color, (_, item) in zip(colors, names.iterrows()):
            ax.scatter(
                100 * item["effect_size"],
                -np.log10(max(item["fdr"], 1e-300)),
                s=60, color=color, edgecolor="white", linewidth=0.8, zorder=4,
            )

        limit = max(float(np.nanmax(np.abs(gap))) * 1.12, 1)
        top = max(cutoff + 0.15, float(np.nanmax(evidence)) * 1.08)
        ax.set_xlim(-limit, limit)
        ax.set_ylim(-0.05, top)
        _fdr_ticks(ax, top)
        ax.set_title(spec["title"], fontsize=11, pad=10)
        ax.set_xlabel(
            "gap in eligible-biopsy net score (percentage points)\n"
            f"← higher in {spec['b_name']}     higher in {spec['a_name']} →",
            fontsize=9,
        )
        ax.set_ylabel("joint FDR  (smaller = stronger)")
        tidy_axes(ax, grid="both", hide=("top", "right"))

        key.axis("off")
        passed_n = int(passed.sum())
        key.set_title(
            f"Largest signals · {passed_n} pass FDR {fdr_threshold:g}",
            fontsize=10, loc="left", pad=7,
        )
        for rank, (color, (rule, item)) in enumerate(zip(colors, names.iterrows()), start=1):
            ypos = 1 - (rank - 0.35) / max(name_top, len(names))
            key.scatter(0.02, ypos - 0.005, s=46, color=color, edgecolor="white",
                        linewidth=0.7, transform=key.transAxes)
            label = rule.replace(" -> ", " → ").replace("_", " ")
            key.text(0.08, ypos, label,
                     transform=key.transAxes, fontsize=7.8, fontweight="bold",
                     va="top", color=_INK)
            key.text(
                0.08, ypos - 0.055,
                f"gap {item['effect_size'] * 100:+.0f} pp · FDR {item['fdr']:.3g}",
                transform=key.transAxes, fontsize=7.7, va="top", color="#706E68",
            )

    fig.suptitle(f"Pooled temporal contrasts — {organ}", fontsize=14, y=0.985)
    fig.text(
        0.5, 0.937,
        (f"Analysis: pairwise rules · Unit: biopsy · FOVs: eligible only · "
         f"FDR correction: 2 contrasts / {total_tests} tests"),
        ha="center", va="top", fontsize=8.5, color="#706E68",
    )
    fig.subplots_adjust(left=0.09, right=0.985, top=0.84, hspace=0.64, wspace=0.07)
    _finish(fig, save)


# ---------------------------------------------------------------------------
# Rule-state context used by the new-findings notebook
# ---------------------------------------------------------------------------

def _prevalence_panel(ax, states, eligibility, metadata, spec, stages, bars=False,
                      fov_label="eligible FOVs"):
    """How often the rule attracts and how often it avoids, as two separate lines."""
    attraction, avoidance = [], []
    attraction_n, avoidance_n, eligible_n = [], [], []
    for stage, fovs in _eligible_by_stage(eligibility, metadata, spec, stages):
        state = states.reindex(columns=fovs).loc[spec["rule"]]
        share = 100 / len(fovs) if len(fovs) else np.nan
        attraction_n.append(int((state == 1).sum()))
        avoidance_n.append(int((state == -1).sum()))
        eligible_n.append(len(fovs))
        attraction.append(share * attraction_n[-1])
        avoidance.append(share * avoidance_n[-1])

    shown = []
    for label, short, values, counts, color in (
            ("Attraction", "attract", attraction, attraction_n, ATTRACTION),
            ("Avoidance", "avoid", avoidance, avoidance_n, AVOIDANCE)):
        if not any(counts):
            continue
        if bars:
            both = any(attraction_n) and any(avoidance_n)
            offset = (-0.18 if short == "attract" else 0.18) if both else 0
            ax.bar(np.arange(len(stages)) + offset, values,
                   width=0.34 if both else 0.56, color=color, label=label)
        else:
            ax.plot(range(len(stages)), values, marker="D", markersize=6, lw=2.2,
                    color=color, label=label)
        shown.append((short, counts))

    labels = [
        f"{stage}\n" + " | ".join(
            [str(counts[position]) for _, counts in shown]
            + [str(eligible_n[position])]
        )
        for position, stage in enumerate(stages)
    ]
    ax.set_ylim(0, 100)
    ax.set_xticks(range(len(stages)), labels)
    ax.set_xlabel(" | ".join([short for short, _ in shown] + [fov_label]),
                  fontsize=7.5, labelpad=2)
    ax.set_ylabel(f"{fov_label}\nwith rule (%)", labelpad=8)
    if shown:
        ax.legend(frameon=False, fontsize=8.2, loc="center left",
                  bbox_to_anchor=(1.01, 0.5), borderaxespad=0)
    tidy_axes(ax, grid="y", hide=("top", "right"))
    ax.tick_params(length=0)


def plot_rule_summary(states, eligibility, cells, metrics, metadata, spec, stages,
                      metric="Lift", reference=1, trend_results=None,
                      pair_results=None, state=None, test_label=None, save=None):
    """One rule in four stacked panels: how often it fires, what it is made of, how
    strong it is where it fires, and what every FOV of the stage was classified as.

    One panel per row rather than four across: the panels share an x axis of three
    stages, so full width each is readable where a quarter of the width is not.
    """
    rule, organ, group_col = spec["rule"], spec["organ"], spec["score"]
    test_unit = spec.get("test_unit", "FOV")
    detail = (
        f"Plots: FOV · Tests: {test_unit} · Organ: {organ} · "
        f"Score: {group_col.replace(' score', '').lower()} · "
        f"Stages: {' / '.join(stages)} · "
        "Eligibility: rule can be tested · "
        "n = eligible FOVs"
    )
    # Built at the width LaTeX gives it, so nothing shrinks on the page.
    with plt.rc_context(_PANEL_FONTS):
        fig, axes = plt.subplots(
            4, 1, figsize=(_TEXT_WIDTH, 8.0),
            gridspec_kw={"height_ratios": [1, 1, 1, 1], "hspace": 0.82},
        )
        _prevalence_panel(axes[0], states, eligibility, metadata, spec, stages)
        _shares_panel(axes[1], cells, eligibility, metadata, spec, stages)
        _metric_panel(axes[2], metrics, eligibility, metadata, spec, stages,
                      metric, reference, state=state)
        counts, eligible_counts, nets = _stage_state_counts(
            states, eligibility, metadata, rule, organ, group_col, stages)
        _draw_state_bars(axes[3], counts, eligible_counts, nets, stages)
        state_handles = [
            Line2D([0], [0], marker="s", linestyle="none", markersize=6,
                   color=_STATE_COLORS[name], label=name)
            for name in _STATE_ORDER
        ]
        axes[3].legend(handles=state_handles, frameon=False, fontsize=7.4,
                       loc="center left", bbox_to_anchor=(1.01, 0.5),
                       borderaxespad=0)

        direction = {1: "attraction", -1: "avoidance"}.get(state)
        strength_title = f"how strong ({direction + ' ' if direction else ''}{metric})"
        for ax, name in zip(axes, ("how often it fires", "what it is made of",
                                   strength_title, "every FOV")):
            ax.set_title(name, fontsize=8.4, color="#5F5D58", loc="left", pad=5)

        fig.suptitle(f"{organ} · {rule.replace(' -> ', ' → ')}", fontsize=11.5, y=0.995)
        fig.text(0.5, 0.973, detail, ha="center", va="top", fontsize=7,
                 color="#706E68", linespacing=1.4)
        fig.align_ylabels(axes)
        fig.subplots_adjust(top=0.91, bottom=0.05, left=0.14, right=0.76)
        _finish(fig, save)


def plot_rule_prevalence_all_fovs(states, metadata, spec, stages, save=None):
    """The summary's "how often it fires" panel, counted over every FOV of the stage."""
    detail = (
        f"Plots: FOV · Organ: {spec['organ']} · "
        f"Score: {spec['score'].replace(' score', '').lower()} · "
        f"Stages: {' / '.join(stages)} · n = all FOVs (no eligibility filter)"
    )
    every_fov = states.astype(bool) | True
    with plt.rc_context(_PANEL_FONTS):
        fig, ax = plt.subplots(figsize=(_TEXT_WIDTH, 2.6))
        _prevalence_panel(ax, states, every_fov, metadata, spec, stages,
                          fov_label="all FOVs")
        ax.set_title("how often it fires", fontsize=8.4, color="#5F5D58", loc="left", pad=5)
        fig.suptitle(f"{spec['organ']} · {spec['rule'].replace(' -> ', ' → ')}",
                     fontsize=11.5, y=0.98)
        fig.text(0.5, 0.9, detail, ha="center", va="top", fontsize=7, color="#706E68")
        fig.subplots_adjust(top=0.72, bottom=0.25, left=0.14, right=0.76)
        _finish(fig, save)


def rule_file(spec, kind, prefix=""):
    """File name for one rule's figure, e.g. Duodenum_clinical_Paneth_to_Epithelial_summary.pdf."""
    rule = spec["rule"].replace(" -> ", "_to_").replace(" ", "")
    score = spec["score"].split()[0].lower()
    return f"{prefix}{spec['organ']}_{score}_{rule}_{kind}.pdf"


def show_rule(spec, states, eligibility, cells, rules, metadata, stages, metric="Lift",
              reference=1, trend_results=None, pair_results=None, prefix=""):
    """One rule: the four-panel summary, then representative FOVs."""
    plot_rule_summary(
        states, eligibility, cells, rules, metadata, spec, stages,
        metric=metric, reference=reference, state=spec.get("state"),
        trend_results=trend_results, pair_results=pair_results,
        save=rule_file(spec, "summary", prefix),
    )
    examples = ds.representative_fovs(
        rules, eligibility, metadata, spec["rule"], spec["organ"], spec["score"],
        stages, metric, cells=cells,
    )
    cr.plot_rule_fovs(
        examples, stages, cells, metadata, spec["organ"], spec["score"],
        save=rule_file(spec, "fovs", prefix),
    )


def plot_organ_rule_context(states, eligibility, cells, metadata, rule, group_col,
                            stages, stage_results, heading,
                            organs=("Colon", "Duodenum"), save=None):
    """Show one organ rule contrast together with both cell-abundance profiles."""
    antecedent, consequent = rule.split(" -> ")
    shares = _cell_share_table(cells, metadata, [antecedent, consequent],
                               ["Organ", group_col])
    organ_colors = {"Colon": "#2878D0", "Duodenum": "#D97832"}
    cell_colors = {antecedent: "#D97832", consequent: "#169873"}

    fig = plt.figure(figsize=(12.2, 8.0))
    grid = fig.add_gridspec(2, 2, height_ratios=(1.18, 1), hspace=0.56, wspace=0.25)
    rule_ax = fig.add_subplot(grid[0, :])
    abundance_axes = [fig.add_subplot(grid[1, i]) for i in range(2)]

    for organ in organs:
        means, counts = [], []
        for stage in stages:
            fovs = metadata.loc[
                (metadata["Organ"] == organ) & (metadata[group_col] == stage), "FOV"
            ].drop_duplicates()
            state = states.reindex(columns=fovs).loc[rule]
            can_test = eligibility.reindex(columns=fovs).loc[rule]
            observed = state[can_test]
            means.append(100 * observed.mean() if len(observed) else np.nan)
            counts.append((int((observed != 0).sum()), len(observed)))
        rule_ax.plot(range(len(stages)), means, color=organ_colors[organ], lw=2.1,
                     marker="D", markersize=7, label=organ)
        for position, (mean, count) in enumerate(zip(means, counts)):
            if np.isfinite(mean):
                offset = 9 if organ == "Colon" else -12
                rule_ax.annotate(f"rule/eligible={count[0]}/{count[1]}",
                                 (position, mean), xytext=(0, offset),
                                 textcoords="offset points", ha="center", fontsize=8,
                                 color=organ_colors[organ])

    evidence = []
    for stage in stages:
        result = stage_results.get(stage, pd.DataFrame())
        if rule in result.index:
            evidence.append(f"{stage} {result.at[rule, 'fdr']:.3g}")
    rule_ax.axhline(0, color=_ZERO, lw=1.1)
    rule_ax.set_xticks(range(len(stages)), stages)
    rule_ax.set_ylim(-112, 112)
    rule_ax.set_ylabel("net rule score among eligible FOVs (%)")
    rule_ax.set_title(rule.replace(" -> ", " → "), fontsize=11, pad=26)
    rule_ax.text(0.5, 1.01, "organ FDR: " + " · ".join(evidence),
                 transform=rule_ax.transAxes, ha="center", va="bottom",
                 fontsize=8.3, color="#706E68")
    rule_ax.legend(frameon=False, ncol=len(organs), loc="lower center",
                   bbox_to_anchor=(0.5, 1.18))
    tidy_axes(rule_ax, grid="y", hide=("top", "right"))
    rule_ax.tick_params(length=0)

    for ax, organ in zip(abundance_axes, organs):
        for cell_type in (antecedent, consequent):
            means = []
            for stage in stages:
                fovs = metadata.loc[
                    (metadata["Organ"] == organ) & (metadata[group_col] == stage), "FOV"
                ].drop_duplicates()
                can_test = eligibility.reindex(columns=fovs).loc[rule]
                eligible_fovs = can_test.index[can_test]
                values = shares.loc[shares["FOV"].isin(eligible_fovs), cell_type]
                means.append(values.fillna(0).mean())
            ax.plot(range(len(stages)), means, color=cell_colors[cell_type], lw=2,
                    marker="D", markersize=6, label=cell_type.replace("_", " "))
        ax.set_xticks(range(len(stages)), stages)
        ax.set_ylabel("mean cells in each FOV (%)")
        ax.set_title(f"{organ}: cell abundance", fontsize=10.5)
        ax.legend(frameon=False, fontsize=8.5)
        tidy_axes(ax, grid="y", hide=("top", "right"))
        ax.tick_params(length=0)

    fig.suptitle(heading, fontsize=14, y=0.99)
    fig.text(0.5, 0.950,
             (f"Analysis: pairwise rule states · Unit: FOV · Organs: {' / '.join(organs)} · "
              f"Score: {group_col.replace(' score', '').lower()} · "
              f"Stages: {' / '.join(stages)} · Eligibility: rule can be tested · "
              "Rule and abundance: same FOVs"),
             ha="center", va="top", fontsize=8.5, color="#706E68")
    fig.subplots_adjust(top=0.82, left=0.09, right=0.98, bottom=0.08)
    _finish(fig, save)
