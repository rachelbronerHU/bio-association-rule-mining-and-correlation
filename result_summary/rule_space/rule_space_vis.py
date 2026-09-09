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
from vis_helper import save_figure, _finish, _titled, _category_colors

# Groups are a split, not a category: never the category palette. Blue against gold,
# far enough apart to survive colour blindness.
_GROUP_COLORS = ["#2F6FA8", "#C98A1F"]

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


def plot_group_spread(spread, label, scope=None, subtitle=None, save=None):
    """How tightly each group's own FOVs sit together, one bar per group.

    A bar left of 1.0 means those FOVs are more alike than FOVs in general here. The
    number on each bar is how many FOVs it rests on.
    """
    if spread.empty:
        print("No data to plot.")
        return

    values = spread["spread"].to_numpy(dtype=float)
    names = [str(name) for name in spread.index]
    y = np.arange(len(values))

    fig, ax = plt.subplots(figsize=(7.4, 0.55 * len(values) + 2.2))
    ax.barh(y, values, height=0.6, color="#4477aa", alpha=0.9,
            edgecolor="white", linewidth=0.8, zorder=3)
    ax.axvline(1.0, color="0.45", ls="--", lw=1.1, zorder=4)

    for position, value, count in zip(y, values, spread["n_FOV"]):
        ax.annotate(f"{value:.2f}  ({int(count)} FOVs)", xy=(value, position),
                    xytext=(5, 0), textcoords="offset points",
                    va="center", fontsize=9, color="0.3")

    ax.set_yticks(y, names)
    ax.invert_yaxis()                          # tightest group at the top
    ax.set_xlim(0, max(1.05, values.max() * 1.28))
    ax.set_xlabel(f"how far apart one {label.lower()} group's FOVs are, "
                  "next to any two FOVs", fontsize=10)
    ax.set_title(_titled(f"Do the FOVs of one {label.lower()} sit together?", scope),
                 fontsize=11)
    if subtitle:
        ax.annotate(subtitle, xy=(0, 1), xytext=(0, 22), xycoords="axes fraction",
                    textcoords="offset points", fontsize=8, color="0.35")
    ax.annotate("same as any two FOVs", xy=(1.0, len(values) - 0.4), xytext=(5, 0),
                textcoords="offset points", fontsize=8, color="0.45")
    ax.grid(axis="x", color="0.92", lw=0.8)
    ax.set_axisbelow(True)
    ax.spines[["top", "right", "left"]].set_visible(False)
    plt.tight_layout()
    _finish(fig, save)


def plot_pca_scatter(df_pca, explained_variance, color_by, subtitle=None,
                     label_fovs=None, label_colors=None, box_fovs=None,
                     x="PC1", y="PC2", fov_col="FOV",
                     max_legend=12, save=None, scope=None, ax=None):
    """One dot per FOV, colored by `color_by`. The static version for the write-up.

    `label_fovs` : the FOVs to name on the plot — either a list, or the
        {FOV: description} dict that `get_representative_fovs_for_pc` returns.
        Names are pushed apart with leader lines so they never overlap.
    `label_colors` : {FOV: color} for those names and their rings. Pass the same
        mapping the maps use and one corner reads as one colour across both figures.
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
        fig, ax = plt.subplots(figsize=(7.5, 5.8))
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
            tone = (label_colors or {}).get(fov, "black")
            ax.scatter([row[x]], [row[y]], s=110, facecolors="none",
                       edgecolor=tone, linewidth=1.3, zorder=5)
            # Set off the dot from the start: adjustText is not always installed, and
            # a name printed on top of its own ring is the harder one to read.
            texts.append(ax.annotate(str(fov), (row[x], row[y]), xytext=(13, 7),
                                     textcoords="offset points", fontsize=8, zorder=6,
                                     ha="left", va="center", color=tone,
                                     fontweight="bold" if label_colors else "normal"))
        if texts:
            try:
                from adjustText import adjust_text
                adjust_text(texts, ax=ax,
                            arrowprops=dict(arrowstyle="-", color="0.45", lw=0.7),
                            expand=(2.2, 2.6))
            except ImportError:
                pass

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


def plot_pca_panel(rows, color_by, title=None, subtitle=None, save=None):
    """Several PCAs in one figure, so they can be read against each other.

    `rows` : lists of (coords, variance, label) - one list per row of the panel. A
        short row leaves its remaining slots blank. The label is what that tile was
        built from, and is the only thing that differs between them.
    `subtitle` : the settings, carried once for the whole panel.
    """
    ncols = max(len(row) for row in rows)
    fig, axes = plt.subplots(len(rows), ncols, squeeze=False,
                             figsize=(6 * ncols, 5 * len(rows)))
    for axes_row, tiles in zip(axes, rows):
        for ax, (coords, variance, label) in zip(axes_row, tiles):
            plot_pca_scatter(coords, variance, color_by=color_by, scope=label, ax=ax)
        for ax in axes_row[len(tiles):]:
            ax.axis("off")

    for ax in axes.flat[1:]:            # the same colours everywhere - one legend is enough
        if ax.get_legend():
            ax.get_legend().remove()

    # Offsets in inches, not figure fractions: a panel three rows tall is three times
    # the height, and a fraction would put the title three times further from the top.
    height = fig.get_size_inches()[1]
    lines = 0 if not subtitle else subtitle.count("\n") + 1
    title_y = 1 - 0.20 / height
    subtitle_y = 1 - 0.46 / height
    strip = 1 - (0.52 + 0.20 * lines) / height

    fig.tight_layout(rect=(0, 0, 1, strip))
    fig.suptitle(title, fontsize=15, y=title_y, va="top")
    if subtitle:
        fig.text(0.5, subtitle_y, subtitle, ha="center", va="top",
                 fontsize=9.5, color="0.35", linespacing=1.6)
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


# ---------------------------------------------------------------------------
# 6. The write-up's summary figures: one per question that spans every scope
# ---------------------------------------------------------------------------

# Organ hues, the same ones the organ-coloured scatters use everywhere else in the
# summary, so blue always means colon. Checked as a pair for colour blindness. The weak
# all-FOV scope is drawn in ink instead: it is context for the other two, not a third
# category of its own.
ORGAN_HUES = {"Colon": "#1f77b4", "Duodenum": "#ff7f0e"}
CONTEXT_INK = "#52514e"

# Who is ahead: two hues with a grey middle for "too close to call".
AHEAD = "#1B9AAA"       # the rules are ahead
BEHIND = "#A11D5B"      # the cell counts are ahead
LEVEL = "#b8b7b1"       # the two are within 0.02, which is not a result
_QUIET_GRID = "#e8e7e1"


def _bare(ax, grid_axis="both"):
    """Recessive furniture, so the marks are the only thing with weight."""
    ax.grid(axis=grid_axis, color=_QUIET_GRID, lw=0.8)
    ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(colors=CONTEXT_INK, labelsize=9)


def plot_threshold_sweep(table, organs, contrasts, save=None):
    """Separation against how common a rule must be, one line per scope.

    table     : thresholds (index, as shares) x scope names (columns)
    organs    : {scope: organ} for the hue; a scope missing from it is drawn as context
    contrasts : {scope: label} naming the stage contrast, which picks the line style

    Colour carries the organ and the dash carries the contrast, so five lines need two
    hues rather than five. Every line is labelled at its own peak, which is the thing
    the figure exists to show.
    """
    if table.empty:
        print("No data to plot.")
        return

    x = table.index.to_numpy(dtype=float) * 100
    styles = {label: style for label, style in
              zip(dict.fromkeys(contrasts.values()), ["-", "--", ":"])}

    fig, ax = plt.subplots(figsize=(9.5, 5.6))
    ax.axhline(0, color="#8d8c85", lw=1.0, zorder=1)

    for scope in table.columns:
        y = table[scope].to_numpy(dtype=float)
        organ = organs.get(scope)
        color = ORGAN_HUES.get(organ, CONTEXT_INK)
        ax.plot(x, y, styles.get(contrasts.get(scope), "-"), color=color,
                lw=2.0, marker="o", markersize=6, markerfacecolor="white",
                markeredgewidth=1.6, zorder=3)

        # Name each line at its best threshold: that peak is the whole point. A peak at
        # either end is labelled inwards, so the text stays on the page.
        best = int(np.nanargmax(y))
        side = "left" if best == 0 else ("right" if best == len(x) - 1 else "center")
        nudge = {"left": 8, "right": -8, "center": 0}[side]
        ax.annotate(f"{scope}   {y[best]:+.2f}", xy=(x[best], y[best]),
                    xytext=(nudge, 11), textcoords="offset points", ha=side,
                    fontsize=8.5, color=color, fontweight="bold", zorder=6)
        ax.scatter([x[best]], [y[best]], s=100, facecolors="none", edgecolor=color,
                   linewidth=2.0, zorder=5)

    ax.set_xticks(x, [f"{v:g}%" for v in x])
    ax.set_xlabel("a rule must fire in more than this share of the FOVs",
                  fontsize=10, color=CONTEXT_INK)
    ax.set_ylabel("separation", fontsize=10, color=CONTEXT_INK)
    ax.set_title("Rare rules tell the organs apart, common rules tell the stages apart",
                 fontsize=12.5, color="#252525")

    handles = [plt.Line2D([], [], color=ORGAN_HUES[organ], lw=2.2, label=organ)
               for organ in ORGAN_HUES]
    handles += [plt.Line2D([], [], color=CONTEXT_INK, lw=1.8, ls=style, label=label)
                for label, style in styles.items()]
    # Below the plot: the peak labels own the space inside it.
    ax.legend(handles=handles, fontsize=8.5, frameon=False, ncol=5,
              loc="lower center", bbox_to_anchor=(0.5, 0.005),
              bbox_transform=fig.transFigure)
    _bare(ax, grid_axis="y")
    fig.tight_layout(rect=(0, 0.07, 1, 1))
    _finish(fig, save)


def plot_rules_against_counts(table, save=None):
    """Each scope's rule PCA beside a PCA of cell composition, one row per scope.

    table : scope (index) x ['rules', 'counts']

    A filled dot is the rules and a hollow dot the cell counts, so which is which never
    rests on colour. The bar between them is coloured by who is ahead, and greyed where
    the two are within 0.02 of each other.
    """
    if table.empty:
        print("No data to plot.")
        return

    rows = table.iloc[::-1]                      # first row of the table ends up on top
    y = np.arange(len(rows))
    rules = rows["rules"].to_numpy(dtype=float)
    counts = rows["counts"].to_numpy(dtype=float)

    fig, ax = plt.subplots(figsize=(9.5, 0.56 * len(rows) + 2.2))
    ax.axvline(0, color="#8d8c85", lw=1.0, zorder=1)

    for i, (mine, theirs) in enumerate(zip(rules, counts)):
        gap = mine - theirs
        tone = LEVEL if abs(gap) < 0.02 else (AHEAD if gap > 0 else BEHIND)
        ax.plot([mine, theirs], [i, i], color=tone, lw=3.6,
                solid_capstyle="round", zorder=2)
        ax.scatter([theirs], [i], s=80, facecolors="white", edgecolor=CONTEXT_INK,
                   linewidth=1.7, zorder=4)
        ax.scatter([mine], [i], s=80, color=CONTEXT_INK, zorder=5)
        ax.annotate(f"{mine:+.2f}  vs  {theirs:+.2f}", xy=(max(mine, theirs), i),
                    xytext=(11, 0), textcoords="offset points", va="center",
                    fontsize=8, color=CONTEXT_INK)

    ax.set_yticks(y, rows.index, fontsize=9.5)
    low, high = min(rules.min(), counts.min()), max(rules.max(), counts.max())
    ax.set_xlim(low - 0.04, high + 0.12)      # room for the value beside the longest row
    ax.set_xlabel("separation", fontsize=10, color=CONTEXT_INK)
    ax.set_title("The rules against counting the cells, over the same FOVs",
                 fontsize=12.5, color="#252525")

    dot = dict(marker="o", color="none", markeredgecolor=CONTEXT_INK, markersize=9)
    ax.legend(handles=[plt.Line2D([], [], markerfacecolor=CONTEXT_INK, label="rules", **dot),
                       plt.Line2D([], [], markerfacecolor="white", label="cell counts", **dot),
                       plt.Line2D([], [], color=AHEAD, lw=3.6, label="rules ahead"),
                       plt.Line2D([], [], color=BEHIND, lw=3.6, label="counts ahead"),
                       plt.Line2D([], [], color=LEVEL, lw=3.6, label="level")],
              fontsize=8.5, frameon=False, ncol=5, loc="lower center",
              bbox_to_anchor=(0.5, -0.02), bbox_transform=fig.transFigure)
    _bare(ax, grid_axis="x")
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    _finish(fig, save)


def plot_sign_slope(table, organs, save=None):
    """What keeping the avoidance rules does to each scope, one slope per scope.

    table  : scope (index) x ['attraction', 'both']
    organs : {scope: organ} for the hue

    Two positions and a line between them, so the direction of the change is the shape
    of the mark rather than something to read off an axis.
    """
    if table.empty:
        print("No data to plot.")
        return

    fig, ax = plt.subplots(figsize=(7.8, 5.8))
    ax.axhline(0, color="#8d8c85", lw=1.0, zorder=1)

    for scope in table.index:
        start = float(table.at[scope, "attraction"])
        end = float(table.at[scope, "both"])
        color = ORGAN_HUES.get(organs.get(scope), CONTEXT_INK)
        ax.plot([0, 1], [start, end], color=color, lw=2.0, marker="o", markersize=7,
                markerfacecolor="white", markeredgewidth=1.6, zorder=3)
        ax.annotate(f"{scope}  {end:+.2f}", xy=(1, end), xytext=(11, 0),
                    textcoords="offset points", va="center", fontsize=8.5,
                    color=color, fontweight="bold")
        ax.annotate(f"{start:+.2f}", xy=(0, start), xytext=(-11, 0),
                    textcoords="offset points", va="center", ha="right",
                    fontsize=8, color=CONTEXT_INK)

    ax.set_xlim(-0.45, 1.75)
    ax.set_xticks([0, 1], ["attraction rules\nonly", "attraction and\navoidance"],
                  fontsize=9.5)
    ax.set_ylabel("separation", fontsize=10, color=CONTEXT_INK)
    ax.set_title("Avoidance rules pay in the duodenum and cost a little in the colon",
                 fontsize=11.5, color="#252525")
    ax.legend(handles=[plt.Line2D([], [], color=ORGAN_HUES[organ], lw=2.2, label=organ)
                       for organ in ORGAN_HUES],
              fontsize=8.5, frameon=False, loc="upper left")
    _bare(ax, grid_axis="y")
    ax.spines["bottom"].set_visible(False)
    ax.tick_params(bottom=False)
    fig.tight_layout()
    _finish(fig, save)


def plot_variance_bars(table, save=None):
    """How much of each scope its first two components carry, and what it was built on.

    table : scope (index) x ['PC1', 'PC2', 'FOVs', 'rules']

    One hue in two steps, because the two components measure the same thing. The counts
    ride along as text so the bar stays the only mark.
    """
    if table.empty:
        print("No data to plot.")
        return

    rows = table.iloc[::-1]
    y = np.arange(len(rows))
    pc1 = rows["PC1"].to_numpy(dtype=float)
    pc2 = rows["PC2"].to_numpy(dtype=float)

    fig, ax = plt.subplots(figsize=(9.5, 0.52 * len(rows) + 2.0))
    ax.barh(y, pc1, height=0.6, color="#2878D0", zorder=3, label="PC1")
    # A hair of surface between the segments, so the split stays visible.
    ax.barh(y, pc2, height=0.6, left=pc1 + 0.22, color="#A8C8EC", zorder=3, label="PC2")
    for i, (first, second, fovs, rules) in enumerate(
            zip(pc1, pc2, rows["FOVs"], rows["rules"])):
        ax.annotate(f"{first + second:.1f}%      {int(fovs)} FOVs, {int(rules)} rules",
                    xy=(first + second + 1.1, i), va="center", fontsize=8.5,
                    color=CONTEXT_INK)

    ax.set_yticks(y, rows.index, fontsize=9.5)
    ax.set_xlim(0, (pc1 + pc2).max() * 2.15)
    ax.set_xlabel("share of all the variation the first two components carry (%)",
                  fontsize=10, color=CONTEXT_INK)
    ax.set_title("How thin each picture is", fontsize=12.5, color="#252525")
    ax.legend(fontsize=8.5, frameon=False, ncol=2, loc="lower right")
    _bare(ax, grid_axis="x")
    fig.tight_layout()
    _finish(fig, save)
