"""Drawing for the rule-space analysis: PCAs, loadings, and what the FOVs are made of.

Everything here is used only by `rule_space_helper.py` and the rule-space notebooks.
The shared pieces - saving, cell colours, category colours, titles - stay in
`vis_helper.py`, which this module builds on.
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Patch, Rectangle

import vis_helper as vh
from vis_helper import save_figure, _finish, _titled, _category_colors

_GROUP_COLORS = ["#4c5c68", "#c9a227"]   # groups are a split, not a category: never the category palette

# Only a handful of cell types are stacked in one bar, and they need to be told apart at
# a glance. The shared cell colours are spread over every cell type at once, so next to
# each other they can come out near-identical - these are for the stack only.
_STACK_COLORS = list(plt.get_cmap("tab10").colors)

# Red/blue with a wide pale middle: anything within half a standard deviation of its
# average stays near-white, so only real deviations carry colour.
_SOFT_DIVERGING = LinearSegmentedColormap.from_list("soft_diverging", [
    (0.000, "#2166ac"), (0.250, "#8db3d5"), (0.375, "#f4f4f4"),
    (0.625, "#f4f4f4"), (0.750, "#dd9384"), (1.000, "#b2182b")])


# ---------------------------------------------------------------------------
# 1. One PCA: how much each component carries, where the FOVs fall, what drives it
# ---------------------------------------------------------------------------

def plot_pca_scree(explained_variance, subtitle=None, save=None, scope=None):
    """Bars = how much each component explains; red line = running total."""
    ev = np.asarray(explained_variance, dtype=float)
    names = [f"PC{i + 1}" for i in range(len(ev))]

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(names, ev, color="skyblue", edgecolor="black")
    ax.plot(names, np.cumsum(ev), color="red", marker="o", linestyle="-",
            label="running total")
    for i, v in enumerate(ev):
        ax.annotate(f"{v:.1f}%", (i, v), ha="center", va="bottom",
                    fontsize=8, xytext=(0, 2), textcoords="offset points")

    fig.suptitle(_titled("PCA explained variance", scope), fontsize=14)
    if subtitle:
        ax.set_title(subtitle, fontsize=9, color="0.35")
    ax.set_ylabel("explained variance (%)", fontsize=11)
    ax.set_xlabel("component", fontsize=11)
    ax.legend(frameon=False)
    ax.grid(axis="y", linestyle="--", alpha=0.7)
    ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()
    _finish(fig, save)


_PC_BAR_COLORS = ["#4477aa", "#ee9944", "#88ccaa"]


def plot_abundance_correlation_bars(rho, n_fovs, top_n=8, subtitle=None, save=None,
                                    scope=None):
    """How closely each cell type's share of the cells follows each component.

    `rho` : cell types x components, Spearman rho. `n_fovs` sets the shaded band -
    the range where a value is too small to mean anything at this many FOVs.
    Only the `top_n` cell types with the largest correlation are drawn.
    """
    if rho.empty:
        print("No data to plot.")
        return

    pcs = list(rho.columns)
    top = (rho.reindex(rho.abs().max(axis=1).sort_values(ascending=False).index)
              .head(top_n).iloc[::-1])                  # biggest ends up at the top
    band = 1.96 / np.sqrt(max(n_fovs - 1, 1))            # |rho| that reaches p < 0.05

    y = np.arange(len(top))
    height = 0.8 / len(pcs)
    fig, ax = plt.subplots(figsize=(7, max(3.2, len(top) * 0.42)))

    ax.axvspan(-band, band, color="0.88", zorder=0)
    for i, pc in enumerate(pcs):
        offset = (i - (len(pcs) - 1) / 2) * height
        ax.barh(y + offset, top[pc].to_numpy(dtype=float), height=height,
                color=_PC_BAR_COLORS[i % len(_PC_BAR_COLORS)], label=pc,
                edgecolor="white", linewidth=0.5, zorder=3)

    ax.axvline(0, color="0.2", lw=1.0, zorder=4)
    ax.set_yticks(y)
    ax.set_yticklabels(top.index.astype(str), fontsize=9)
    ax.set_xlim(-1, 1)
    ax.set_xlabel("how closely they move together", fontsize=10)
    ax.set_ylabel("cell type", fontsize=10)
    fig.suptitle(_titled("Does each component follow how common a cell type is?", scope)
                 + f"   (grey: too small to matter at {n_fovs} FOVs)", fontsize=11)
    if subtitle:
        ax.set_title(subtitle, fontsize=7.5, color="0.35")
    ax.legend(fontsize=9, frameon=False, loc="lower right")
    ax.grid(axis="x", color="0.92", lw=0.8)
    ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()
    _finish(fig, save)


def plot_patient_spread(spread, save=None):
    """Whether one patient's FOVs sit together, against how many FOVs they have.

    One dot per patient. x = the mean distance between that patient's own FOVs,
    divided by the mean distance between any two FOVs in the same plane. Left of 1.0
    means that patient's FOVs sit closer together than FOVs in general.
    """
    if spread.empty:
        print("No data to plot.")
        return

    x = spread["spread"].to_numpy(dtype=float)
    k = spread["n_FOV"].to_numpy(dtype=float)
    y = k + np.random.default_rng(0).uniform(-0.2, 0.2, len(k))   # jitter, so dots don't stack

    fig, ax = plt.subplots(figsize=(7.4, 3.6))
    ax.axvline(1.0, color="0.45", ls="--", lw=1.1, zorder=1)
    ax.scatter(x, y, s=38, color="#4477aa", alpha=0.85,
               edgecolor="white", linewidth=0.6, zorder=3)

    below = int((x < 1).sum())
    ax.set_xlabel("how far apart one patient's FOVs are, next to any two FOVs", fontsize=10)
    ax.set_ylabel("FOVs the patient has", fontsize=10)
    ax.set_title(f"Do a patient's FOVs sit together?   {below} of {len(x)} patients "
                 f"are closer together than average", fontsize=11)
    ax.set_yticks(sorted({int(v) for v in k}))
    ax.annotate("same as any two FOVs", xy=(1.0, ax.get_ylim()[0]), xytext=(5, 6),
                textcoords="offset points", fontsize=8, color="0.45")
    ax.grid(color="0.92", lw=0.8)
    ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()
    _finish(fig, save)


def plot_pca_scatter(df_pca, explained_variance, color_by, subtitle=None,
                     label_fovs=None, box_fovs=None, x="PC1", y="PC2", fov_col="FOV",
                     max_legend=12, save=None, scope=None, ax=None):
    """One dot per FOV, colored by `color_by`. The static version for the write-up.

    `label_fovs` : the FOVs to name on the plot — either a list, or the
        {FOV: description} dict that `get_representative_fovs_for_pc` returns.
        Names are pushed apart with leader lines so they never overlap.
    `max_legend` : with more colors than this (e.g. one per patient) the legend
        would be unreadable, so it is replaced by a count in the title.

    A numeric `color_by` column (e.g. a cell type's share of the FOV) is drawn
    as a colorbar instead of a legend.
    """
    if df_pca.empty:
        print("No data to plot.")
        return

    ix = [int(x[2:]) - 1, int(y[2:]) - 1]
    ev = np.asarray(explained_variance, dtype=float)
    numeric = pd.api.types.is_numeric_dtype(df_pca[color_by])

    own = ax is None
    if own:
        fig, ax = plt.subplots(figsize=(9, 7))
    else:
        fig = ax.figure


    ordered = []
    if numeric:
        v = df_pca[color_by].to_numpy(dtype=float)
        sc = ax.scatter(df_pca[x], df_pca[y], s=48, c=v, cmap="viridis",
                        alpha=0.9, edgecolor="DarkSlateGrey", linewidth=0.5)
        fig.colorbar(sc, ax=ax, label=color_by)
    else:
        vals = df_pca[color_by].fillna("Unknown").astype(str)
        colors, ordered = _category_colors(vals)
        for value in ordered:
            m = (vals == value).to_numpy()
            ax.scatter(df_pca.loc[m, x], df_pca.loc[m, y], s=48, label=str(value),
                       color=colors[value], alpha=0.85,
                       edgecolor="DarkSlateGrey", linewidth=0.5)

    ax.axhline(0, color="0.85", lw=0.8, zorder=0)
    ax.axvline(0, color="0.85", lw=0.8, zorder=0)

    # One square around the boxed FOVs, so the maps beside the plot are easy to find.
    boxed = df_pca[df_pca[fov_col].isin(list(box_fovs or []))]
    if not boxed.empty:
        pad = 0.018 * max(df_pca[x].max() - df_pca[x].min(),
                          df_pca[y].max() - df_pca[y].min())
        x0, x1 = boxed[x].min() - pad, boxed[x].max() + pad
        y0, y1 = boxed[y].min() - pad, boxed[y].max() + pad
        # Blue, and filled: the stages are green/orange/red and every other marker is
        # outlined in black, so this is the only thing on the plot in this colour.
        ax.add_patch(Rectangle((x0, y0), x1 - x0, y1 - y0, facecolor="#4477aa",
                               alpha=0.18, zorder=1))
        ax.add_patch(Rectangle((x0, y0), x1 - x0, y1 - y0, fill=False,
                               edgecolor="#1f4e79", linewidth=2.2, zorder=6))


    if label_fovs is not None:
        names = list(label_fovs.keys()) if isinstance(label_fovs, dict) else list(label_fovs)
        texts = []
        for fov in names:
            row = df_pca[df_pca[fov_col] == fov]
            if row.empty:
                continue
            row = row.iloc[0]
            ax.scatter([row[x]], [row[y]], s=110, facecolors="none",
                       edgecolor="black", linewidth=1.3, zorder=5)
            texts.append(ax.text(row[x], row[y], str(fov), fontsize=8, zorder=6))
        if texts:
            try:
                from adjustText import adjust_text
                adjust_text(texts, ax=ax,
                            arrowprops=dict(arrowstyle="-", color="0.45", lw=0.7),
                            expand=(1.6, 1.9))
            except ImportError:
                for t in texts:
                    t.set_ha("left")

    title = _titled(f"PCA colored by {color_by}", scope)
    if len(ordered) > max_legend:
        title += f"  ({len(ordered)} values, too many to list)"
    else:
        ax.legend(title=color_by, fontsize=9, title_fontsize=10,
                  bbox_to_anchor=(1.02, 1), loc="upper left", frameon=True)
    # Inside a panel the figure is shared, so the title has to sit on the axes.
    if own:
        fig.suptitle(title, fontsize=13)
        if subtitle:
            ax.set_title(subtitle, fontsize=9, color="0.35")
    else:
        ax.set_title(title, fontsize=11)

    ax.set_xlabel(f"{x} ({ev[ix[0]]:.1f}%)", fontsize=11)
    ax.set_ylabel(f"{y} ({ev[ix[1]]:.1f}%)", fontsize=11)
    ax.grid(color="0.93", lw=0.8)
    ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)

    if own:
        plt.tight_layout()
        _finish(fig, save)


def plot_pca_loadings(weights, feature_names, component_idx=0, subtitle=None,
                      top_n=10, save=None, scope=None):
    """The rules pulling hardest on one component, both ways.

    Green = rules that push a FOV to the positive side, salmon = to the negative
    side. Reading both ends together tells you what that component means.
    """
    series = pd.Series(np.asarray(weights, dtype=float), index=list(feature_names))
    top = pd.concat([series.nlargest(top_n), series.nsmallest(top_n)]).sort_values()
    top = top[~top.index.duplicated()]             # a short list can overlap

    colors = ["salmon" if v < 0 else "mediumseagreen" for v in top.to_numpy()]
    # Near-square, so two of these sit side by side in the write-up and stay readable.
    fig, ax = plt.subplots(figsize=(7, max(4, len(top) * 0.42)))
    ax.barh(top.index.astype(str), top.to_numpy(), color=colors, edgecolor="black")

    fig.suptitle(_titled(f"Top rules for PC{component_idx + 1}", scope), fontsize=14)
    if subtitle:
        ax.set_title(subtitle, fontsize=9, color="0.35")
    ax.set_xlabel(f"weight in PC{component_idx + 1}", fontsize=11)
    ax.set_ylabel("rule", fontsize=11)
    ax.axvline(0, color="black", linewidth=1.2)
    ax.grid(axis="x", linestyle="--", alpha=0.7)
    ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(axis="y", labelsize=9)
    plt.tight_layout()
    _finish(fig, save)


# ---------------------------------------------------------------------------
# 2. What the FOVs in each composition group are made of
# ---------------------------------------------------------------------------

def _positions(group):
    """x for each FOV, with a gap between the groups so they read as two blocks."""
    return np.arange(len(group), dtype=float) + (group.to_numpy() == "B") * 3


def _strips(axes, labels, x, names):
    """The thin rows above the plot: one colour band per label, with its own key.

    The first row is the composition group, which gets its own two colours - drawing it
    from the category palette made A and B look like Colon and Duodenum.
    """
    for i, (ax, values, name) in enumerate(zip(axes, labels, names)):
        if i == 0:
            colors = dict(zip(sorted(values.unique()), _GROUP_COLORS))
        else:
            colors, _ = _category_colors(values.astype(str))
        ax.bar(x, 1, width=1.0, color=[colors[v] for v in values.astype(str)])
        ax.set_ylabel(name, rotation=0, ha="right", va="center", fontsize=9)
        ax.set_yticks([])
        ax.set_xlim(x.min() - 1, x.max() + 1)
        for side in ax.spines.values():
            side.set_visible(False)
        ax.legend(handles=[Patch(facecolor=c, label=v) for v, c in colors.items()],
                  fontsize=8, ncol=len(colors), frameon=False,
                  loc="lower left", bbox_to_anchor=(1.005, 0))


def plot_composition_split(table, means, scope, label, save=None):
    """How the two composition groups fall by `label`, and what each is made of."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4),
                                   gridspec_kw={"width_ratios": [1, 1.3]})
    colors, ordered = _category_colors(pd.Series(table.columns.astype(str)))
    table[ordered].plot.bar(ax=ax1, rot=0, width=0.7,
                            color=[colors[o] for o in ordered], edgecolor="white")
    for bar in ax1.patches:
        if bar.get_height():
            ax1.annotate(int(bar.get_height()),
                         (bar.get_x() + bar.get_width() / 2, bar.get_height()),
                         ha="center", va="bottom", fontsize=9)
    ax1.set_xlabel("group of FOVs with similar cell counts")
    ax1.set_ylabel("FOVs")
    ax1.set_title(f"How the two groups fall by {label}", fontsize=11)
    ax1.legend(title=label, fontsize=9, title_fontsize=9)

    (means * 100).T.plot.barh(ax=ax2, color=_GROUP_COLORS, edgecolor="white")
    ax2.set_xlabel("% of the cells in the FOV, averaged over the group")
    ax2.set_title("What the two groups are made of", fontsize=11)
    ax2.legend(title="group", fontsize=9, title_fontsize=9)

    for ax in (ax1, ax2):
        ax.grid(color="0.93", lw=0.8)
        ax.set_axisbelow(True)
        ax.spines[["top", "right"]].set_visible(False)
    fig.suptitle(f"Same cells, different arrangement?   {scope}", fontsize=13)
    fig.tight_layout()
    save_figure(fig, save)
    plt.show()


def plot_composition_bars(shares, labels, cells, scope, save=None):
    """One bar per FOV: the share of its cells each of the split's cell types holds.

    The rest of the tissue is the grey block on top, so you can see how much of the
    FOV the split was actually made on.
    """
    x = _positions(labels[0])
    fig, axes = plt.subplots(3, 1, figsize=(13, 5.4), sharex=True,
                             gridspec_kw={"height_ratios": [0.5, 0.5, 8]})
    _strips(axes[:2], labels, x, ["group", labels[1].name])

    ax, bottom = axes[2], np.zeros(len(shares))
    for i, cell in enumerate(cells):
        ax.bar(x, shares[cell], bottom=bottom, width=1.0,
               color=_STACK_COLORS[i % len(_STACK_COLORS)], label=cell)
        bottom += shares[cell].to_numpy()
    ax.bar(x, 100 - bottom, bottom=bottom, width=1.0, color="0.88",
           label="every other cell type")

    ax.set_ylim(0, 100)
    ax.set_xlim(x.min() - 1, x.max() + 1)
    ax.set_xticks([])
    ax.set_ylabel("% of the cells in the FOV")
    ax.set_xlabel("one bar per FOV, sorted inside each group")
    ax.legend(fontsize=8, loc="upper left", bbox_to_anchor=(1.005, 1), frameon=False)
    fig.suptitle(f"What each FOV is made of   {scope}", fontsize=13)
    fig.tight_layout()
    save_figure(fig, save)
    plt.show()


def plot_composition_heatmap(z, difference, labels, cells, scope, save=None):
    """Every cell type, every FOV. Colour is how far from that cell type's average.

    A heatmap of raw shares would show only Epithelial and Muscle, so each cell type is
    standardised: red is more than usual for that type, blue is less. Rows cannot be
    compared with each other - only along a row.

    Rows arrive sorted by `difference`, the gap between the two groups, which is drawn
    again as bars down the side. That turns what is otherwise speckle into blocks: the
    cell types high in group A gather at the top, group B's at the bottom, and the ones
    that tell the groups apart not at all fade into the pale band between them.
    """
    x = _positions(labels[0])
    rows = list(z.columns)
    fig, axes = plt.subplots(3, 3, figsize=(9, 10), sharex="col",
                             gridspec_kw={"height_ratios": [0.5, 0.5, 14],
                                          "width_ratios": [14, 2.6, 0.3], "wspace": 0.04})
    for ax in axes[:2, 1:].flat:
        ax.axis("off")
    _strips(axes[:2, 0], labels, x, ["group", labels[1].name])

    ax = axes[2, 0]
    grid = np.full((len(rows), int(x.max()) + 1), np.nan)
    grid[:, x.astype(int)] = z.to_numpy().T
    im = ax.imshow(grid, aspect="auto", cmap=_SOFT_DIVERGING, vmin=-2, vmax=2,
                   extent=(-0.5, x.max() + 0.5, len(rows) - 0.5, -0.5))
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels([f"{r}  *" if r in cells else r for r in rows], fontsize=8)
    ax.set_xticks([])
    ax.set_xlabel("one column per FOV, sorted inside each group   "
                  "(* = a cell type the split was made on)")

    side = axes[2, 1]
    y = np.arange(len(rows))
    side.barh(y, difference[rows], height=0.8,
              color=[_SOFT_DIVERGING(0.5 + np.clip(d, -2, 2) / 4) for d in difference[rows]])
    side.axvline(0, color="0.5", lw=0.8)
    side.set_ylim(len(rows) - 0.5, -0.5)
    side.set_yticks([])
    side.set_xlabel("group A - group B\n(standard deviations)", fontsize=8)
    side.tick_params(labelsize=8)
    side.grid(axis="x", color="0.93", lw=0.8)
    side.set_axisbelow(True)
    side.spines[["top", "right", "left"]].set_visible(False)

    fig.colorbar(im, cax=axes[2, 2],
                 label="standard deviations from that cell type's average")
    fig.suptitle(f"Every cell type, every FOV   {scope}", fontsize=13)
    save_figure(fig, save)
    plt.show()
