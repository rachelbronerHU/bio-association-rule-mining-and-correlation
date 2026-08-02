"""Visualization helpers for the result-summary pairwise-rule analysis.

Only plotting lives here; data prep / aggregation stays in the notebook.
"""
import os

import matplotlib.pyplot as plt
from matplotlib.colors import PowerNorm, Normalize, to_rgb
from matplotlib.patches import Patch
from matplotlib.ticker import MultipleLocator
import numpy as np
import pandas as pd
import seaborn as sns

# Stage colors for the fingerprint strips (baseline greens -> warm severity).
_STAGE_PALETTE = {
    "Control_S": "#4d9221", "Control": "#a1d99b",
    "Mild": "#fdae6b", "Severe": "#e34a33", "Unknown": "#dddddd",
}


def _prevalence_norm(df_pct, gamma=0.5):
    """Non-linear color norm so a few very-prevalent rules don't wash out the rest.

    gamma < 1 stretches the low/mid range (PowerNorm: color ~ value**gamma).
    """
    vmax = float(np.nanmax(df_pct.to_numpy())) if df_pct.size else 1.0
    return PowerNorm(gamma=gamma, vmin=0.0, vmax=max(vmax, 1e-9))


def _build_annotations(df_plot, orig_col_by_display, total_col, total_units, stage_totals):
    """Return (percentage matrix, 'count (pct%)' annotation matrix)."""
    df_pct = df_plot.astype(float).copy()
    annot = []
    for idx, row in df_plot.iterrows():
        annot_row = []
        for display_col, val in row.items():
            orig = orig_col_by_display[display_col]
            total = total_units if orig == total_col else stage_totals.get(orig, val)
            pct = (val / total * 100) if total else 0.0
            df_pct.at[idx, display_col] = pct
            annot_row.append(f"{int(val)} ({pct:.1f}%)")
        annot.append(annot_row)
    return df_pct, annot


def plot_rule_stage_heatmap(
    df_agg,
    df_metadata,
    score_col,
    organs=None,
    stages=None,
    id_col="FOV",
    stage_order=None,
    rule_abundance=None,
    norm_gamma=0.5,
    max_rules_to_show=30,
    cmap="YlOrRd",
    save=None,
):
    """Heatmap of top-rule prevalence (% of `id_col` units) across stages.

    Parameters
    ----------
    df_agg : DataFrame indexed by Clean_Rule, with a 'Total {id_col}' column + one per stage.
    df_metadata : metadata carrying `id_col` and `score_col` (used for the n= denominators).
    score_col : stage column name (e.g. 'Pathological score').
    id_col : counting unit — 'FOV' (default), 'Biopsy', or 'PatientID'. Must match df_agg.
    stage_order : optional list giving the left-to-right order of stage columns
        (e.g. ['Control_S', 'Control', 'Mild', 'Severe']); unlisted stages keep their order.
    organs / stages : only used to annotate the title.
    rule_abundance : DataFrame indexed by Clean_Rule with 'ant' and 'con' columns
        (mean cell fraction over the rule's own FOVs). When given, draws back-to-back
        antecedent/consequent abundance bars beside the heatmap.
    norm_gamma : PowerNorm gamma for the color scale (None = linear). <1 boosts low values.
    """
    if df_agg.empty:
        print("No data to plot.")
        return

    total_col = f"Total {id_col}"
    df_plot = df_agg.head(max_rules_to_show).copy()

    # Optional explicit column order (Total stays first).
    if stage_order is not None:
        stage_cols = [c for c in df_plot.columns if c != total_col]
        ordered = [c for c in stage_order if c in stage_cols]
        rest = [c for c in stage_cols if c not in ordered]
        df_plot = df_plot[[total_col] + ordered + rest]
    total_units = df_metadata[id_col].nunique()
    stage_totals = df_metadata.groupby(score_col)[id_col].nunique()

    # Rename columns to include n= counts, remembering the mapping back to originals.
    orig_col_by_display = {}
    renamed = {}
    for col in df_plot.columns:
        if col == total_col:
            disp = f"{col}\n(n={total_units})"
        elif col in stage_totals:
            disp = f"{col}\n(n={stage_totals[col]})"
        else:
            disp = col
        renamed[col] = disp
        orig_col_by_display[disp] = col
    df_plot = df_plot.rename(columns=renamed)

    df_pct, annot = _build_annotations(df_plot, orig_col_by_display, total_col, total_units, stage_totals)

    draw_bars = rule_abundance is not None
    height = max(6, len(df_plot) * 0.3)
    if draw_bars:
        fig, (ax_bar, ax) = plt.subplots(
            1, 2, figsize=(15.5, height),
            gridspec_kw={"width_ratios": [8, 30], "wspace": 0.06},
        )
    else:
        fig, ax = plt.subplots(figsize=(14, height))

    norm = _prevalence_norm(df_pct, norm_gamma) if norm_gamma else None
    sns.heatmap(
        df_pct, annot=annot, fmt="", cmap=cmap, ax=ax, norm=norm,
        cbar_kws={"label": f"Percentage of {id_col} (%)"},
    )

    if draw_bars:
        rule_index = df_agg.head(max_rules_to_show).index
        ra = rule_abundance.reindex(rule_index)
        ant = ra["ant"].to_numpy(dtype=float)
        con = ra["con"].to_numpy(dtype=float)
        both = np.concatenate([ant, con])
        vmax = float(np.nanmax(both)) if np.isfinite(both).any() else 1.0
        y = np.arange(len(rule_index)) + 0.5

        # Back-to-back bars: length = abundance; color (shared scale) reinforces it at a glance.
        cnorm = Normalize(vmin=0.0, vmax=vmax)
        pur = plt.get_cmap("Purples")
        ant_colors = pur(0.25 + 0.75 * cnorm(np.nan_to_num(ant)))
        con_colors = pur(0.25 + 0.75 * cnorm(np.nan_to_num(con)))
        ax_bar.barh(y, -ant, height=0.8, color=ant_colors)   # antecedent grows left
        ax_bar.barh(y, con, height=0.8, color=con_colors)    # consequent grows right
        ax_bar.axvline(0, color="0.35", lw=1.0)              # center divider

        ax_bar.set_ylim(ax.get_ylim())                       # align rows with the heatmap
        ax_bar.set_yticks(y)
        ax_bar.set_yticklabels(rule_index, fontsize=9)
        ax_bar.set_ylabel("Cleaned Rule", fontsize=12)
        ax_bar.set_xlim(-vmax * 1.08, vmax * 1.08)
        ax_bar.set_xticks([-vmax, 0, vmax])
        ax_bar.set_xticklabels([f"{vmax:.2f}", "0", f"{vmax:.2f}"], fontsize=8)
        ax_bar.set_xlabel("mean cell fraction\n(in the rule's FOVs)", fontsize=8)
        ax_bar.annotate("Antecedent", xy=(0.25, 1.0), xycoords="axes fraction",
                        ha="center", va="bottom", fontsize=10, fontweight="bold")
        ax_bar.annotate("Consequent", xy=(0.75, 1.0), xycoords="axes fraction",
                        ha="center", va="bottom", fontsize=10, fontweight="bold")
        ax_bar.spines[["top", "right", "left"]].set_visible(False)
        ax_bar.tick_params(left=False)
        ax.set_yticks([])

    title = f"Top Rules Prevalence across {score_col} (by {id_col})"
    if organs:
        title += f" ({', '.join(organs)})"
    if stages:
        title += f" | Stages: {', '.join(map(str, stages))}"
    ax.set_title(title, fontsize=14, pad=15)
    ax.set_ylabel("" if draw_bars else "Cleaned Rule", fontsize=12)
    ax.set_xlabel("Stage", fontsize=12)
    ax.tick_params(axis="both", which="major", labelsize=10,
                   top=True, labeltop=True, bottom=False, labelbottom=False)
    plt.tight_layout()
    _finish(fig, save)


def _similarity_order(frame):
    """Index order that puts similar rows next to each other (hierarchical clustering).

    Euclidean/average linkage: correlation and cosine are undefined for all-zero rows
    (e.g. a patient where none of the top rules fire), euclidean handles them fine.
    """
    if len(frame) < 3:
        return frame.index
    from scipy.cluster.hierarchy import linkage, leaves_list
    order = leaves_list(linkage(frame.fillna(0).to_numpy(), method="average", metric="euclidean"))
    return frame.index[order]


def plot_patient_fingerprint(
    mat,
    df_fovs,
    score_col,
    patient_col="PatientID",
    stage_order=None,
    strip_scores=("Pathological score", "Clinical score"),
    order="cluster",
    cluster_rules=False,
    max_rules=30,
    cmap="YlOrRd",
    save=None,
    scope=None,
):
    """Rules x patients heatmap: each cell = share of that patient's FOVs that have the rule.

    Stage strips above the columns color every patient by stage (one strip per `strip_scores`).
    `order='cluster'` puts patients with similar rules next to each other, so any grouping by
    stage shows up on its own in the strips; `order='stage'` sorts by stage instead.
    `cluster_rules=True` also groups similar rules together.
    """
    if mat.empty:
        print("No data to plot.")
        return
    mat = mat.head(max_rules)

    # One stage value per patient, per score.
    stage_by_score = {}
    for sc in strip_scores:
        if sc in df_fovs.columns:
            stage_by_score[sc] = df_fovs.drop_duplicates(patient_col).set_index(patient_col)[sc]

    if order == "cluster":
        patients = list(_similarity_order(mat.T))       # each row = one patient's rule profile
    else:
        # Order patient columns by the primary score's stage, then id.
        primary = stage_by_score.get(score_col)
        rank = {s: i for i, s in enumerate(stage_order or [])}
        def _key(p):
            s = primary.get(p, "Unknown") if primary is not None else "Unknown"
            return (rank.get(s, len(rank)), str(p))
        patients = sorted(mat.columns, key=_key)
    mat = mat[patients]
    if cluster_rules:
        mat = mat.loc[list(_similarity_order(mat))]

    # The colour bar gets its own column, so the stage strips line up with the heatmap
    # instead of being pushed wider than it.
    n_strip = len(stage_by_score)
    strip_h, main_h = 0.16, max(6, len(mat) * 0.32)
    fig = plt.figure(figsize=(max(10, len(patients) * 0.18),
                              main_h + strip_h * n_strip + 0.5))
    gs = fig.add_gridspec(n_strip + 1, 2, width_ratios=[45, 1],
                          height_ratios=[strip_h] * n_strip + [main_h],
                          hspace=0.10, wspace=0.015)
    strip_axes = [fig.add_subplot(gs[i, 0]) for i in range(n_strip)]
    main_ax = fig.add_subplot(gs[n_strip, 0])
    cax = fig.add_subplot(gs[n_strip, 1])

    for ax_s, (sc, series) in zip(strip_axes, stage_by_score.items()):
        colors = [_STAGE_PALETTE.get(series.get(p, "Unknown"), "#dddddd") for p in patients]
        rgb = np.array([to_rgb(c) for c in colors]).reshape(1, len(patients), 3)
        ax_s.imshow(rgb, aspect="auto", extent=[0, len(patients), 0, 1])
        ax_s.set_xlim(0, len(patients))
        ax_s.set_xticks([])
        ax_s.set_yticks([0.5])
        ax_s.set_yticklabels([sc.replace(" score", "")], fontsize=8)

    sns.heatmap(mat, ax=main_ax, cmap=cmap, vmin=0, vmax=1, cbar_ax=cax,
                cbar_kws={"label": "share of the patient's FOVs"})
    how = ("grouped by how similar their rules are" if order == "cluster"
           else f"ordered by {score_col}")
    main_ax.set_xlabel(f"Patients (n={len(patients)}), {how}", fontsize=11)
    main_ax.set_ylabel("Rule", fontsize=11)
    main_ax.set_xticks(np.arange(len(patients)) + 0.5)
    main_ax.set_xticklabels(patients, rotation=90, fontsize=6)
    main_ax.tick_params(axis="y", labelsize=9)

    stages_present = [s for s in (stage_order or list(_STAGE_PALETTE)) if s in _STAGE_PALETTE]
    handles = [Patch(color=_STAGE_PALETTE[s], label=s) for s in stages_present]
    main_ax.legend(handles=handles, title="Stage", bbox_to_anchor=(1.05, 1.0),
                   loc="upper left", fontsize=8, title_fontsize=9)

    (strip_axes[0] if n_strip else main_ax).set_title(
        _titled("Which rules each patient has", scope), fontsize=13, pad=10)
    _finish(fig, save)          # no tight_layout: it would undo the gridspec alignment


def rule_row_abundance(rule_rows, fov_frac, side="con"):
    """Abundance for every rule occurrence, measured in its own FOV.

    side 'ant' / 'con' = the share of cells of that side's cell type(s) in that FOV.
    side 'both'        = antecedent share x consequent share, which is exactly what
                         lift divides by (lift = P(A and B) / (P(A) x P(B))).
    """
    def one_side(which):
        vals = []
        for rule, fov in zip(rule_rows["Clean_Rule"], rule_rows["FOV"]):
            ant, con = rule.split(" -> ")
            items = (ant if which == "ant" else con).split(", ")
            vals.append(fov_frac.loc[fov, items].mean() if fov in fov_frac.index else np.nan)
        return np.array(vals, dtype=float)

    if side == "both":
        return one_side("ant") * one_side("con")
    return one_side(side)


def _plain_log_ticks(ax, which="x"):
    """Label a log axis with ordinary numbers (0.01, 0.05, 0.2) instead of 10^-2."""
    from matplotlib.ticker import FuncFormatter, NullFormatter
    axis = ax.xaxis if which == "x" else ax.yaxis
    lo, hi = (ax.get_xlim() if which == "x" else ax.get_ylim())
    nice = [0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5,
            1, 2, 5, 10, 20, 50, 100, 200, 500]
    ticks = [t for t in nice if lo <= t <= hi]
    if len(ticks) >= 2:
        axis.set_ticks(ticks)
    axis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
    axis.set_minor_formatter(NullFormatter())


def plot_metric_vs_abundance(rule_rows, fov_frac, metric="Lift", show_trend=True,
                             title_note=None, save=None):
    """One score against how common the cells are: antecedent on the left, consequent on the right.

    One dot per rule occurrence (one rule in one FOV) — nothing is averaged. Both panels
    share the same y-axis, so you can see straight away which side the score follows.
    Infinite conviction is drawn at the top with a red ring instead of being thrown away.
    """
    colors = {"Lift": "#d95f02", "Confidence": "#1b9e77", "Conviction": "#7570b3"}
    color = colors.get(metric, "#4477aa")
    yscale = "linear" if metric == "Confidence" else "log"

    y = rule_rows[metric].to_numpy(dtype=float)
    is_inf = ~np.isfinite(y)
    if is_inf.any():                      # park infinity just above the largest real value
        y = np.where(is_inf, np.nanmax(y[~is_inf]) * 1.3, y)

    fig, axes = plt.subplots(1, 2, figsize=(8, 3.4), sharey=True)
    for ax, side in zip(axes, ["ant", "con"]):
        x = rule_row_abundance(rule_rows, fov_frac, side)
        ok = np.isfinite(x) & (x > 0)
        ax.scatter(x[ok], y[ok], s=10, color=color, alpha=0.3, linewidths=0)
        if is_inf.any():
            m = ok & is_inf
            ax.scatter(x[m], y[m], s=26, facecolors="none", edgecolor="red", linewidth=1.0,
                       label=f"∞ ({int(m.sum())}, drawn at top)")
            ax.legend(frameon=False, fontsize=8, loc="lower right")
        if show_trend and ok.sum() > 40:
            b = pd.DataFrame({"x": x[ok], "y": y[ok]})
            b["_bin"] = pd.qcut(b["x"], 8, labels=False, duplicates="drop")
            t = b.groupby("_bin")[["x", "y"]].median()
            ax.plot(t["x"], t["y"], color=color, lw=2.2)
        ax.set_xscale("log")
        ax.set_xlabel(f"share of cells that are the "
                      f"{'antecedent' if side == 'ant' else 'consequent'}", fontsize=9)
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(color="0.92", lw=0.8)
        ax.set_axisbelow(True)

    axes[0].set_yscale(yscale)
    axes[0].set_ylabel(metric.lower(), fontsize=10)
    for ax in axes:                       # plain numbers, not 10^-2
        _plain_log_ticks(ax, "x")
        if yscale == "log":
            _plain_log_ticks(ax, "y")
        ax.tick_params(labelsize=8)
    note = f" — {title_note}" if title_note else ""
    fig.suptitle(f"Does {metric.lower()} follow how common the cells are?{note}"
                 f"   (one dot per rule per FOV, n={len(rule_rows)})", fontsize=10)
    plt.tight_layout()
    _finish(fig, save)


def plot_rule_metric_scatter(table, x="med_Lift", y="med_Conviction", size_col="n_Patient",
                             color_col="med_Confidence", annotate_top=8, save=None):
    """How strong each rule is vs how many patients have it.

    x / y = the rule's median lift and conviction (log axes); dot size = `size_col`;
    color = `color_col`. Conviction = infinity is drawn at the top with a red ring.
    Only the `annotate_top` strongest rules are named, and the names are nudged apart
    so they never sit on top of each other. Returns those named rules as a table.
    """
    if table.empty:
        print("No data to plot.")
        return None
    df = table.copy()
    xv = df[x].to_numpy(dtype=float)
    yv = df[y].to_numpy(dtype=float).copy()
    is_inf = ~np.isfinite(yv)
    finite_max = np.nanmax(yv[np.isfinite(yv)]) if np.isfinite(yv).any() else 1.0
    yv[is_inf] = finite_max * 1.2                      # cap inf onto the log axis
    sizes = 15 + 6 * df[size_col].to_numpy(dtype=float)

    fig, ax = plt.subplots(figsize=(9, 7))
    sc = ax.scatter(xv, yv, s=sizes, c=df[color_col].to_numpy(dtype=float), cmap="viridis",
                    alpha=0.85, edgecolor="white", linewidth=0.5)
    if is_inf.any():
        ax.scatter(xv[is_inf], yv[is_inf], s=sizes[is_inf], facecolors="none",
                   edgecolor="red", linewidth=1.2, label="conviction = ∞ (shown at top)")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.axvline(np.nanmedian(xv), ls="--", color="0.6", lw=0.8)
    ax.axhline(np.nanmedian(yv), ls="--", color="0.6", lw=0.8)
    ax.set_xlabel("lift (log)", fontsize=11)
    ax.set_ylabel("conviction (log)", fontsize=11)
    fig.colorbar(sc, ax=ax, label="confidence")

    # Name only the strongest few, and let adjustText push the names apart.
    rank = df[[x, y]].rank(pct=True).mean(axis=1)
    named = list(rank.sort_values(ascending=False).head(annotate_top).index)
    texts = []
    for r in named:
        ry = finite_max * 1.2 if not np.isfinite(df.loc[r, y]) else df.loc[r, y]
        texts.append(ax.text(df.loc[r, x], ry, str(r), fontsize=8))
    if texts:
        try:
            from adjustText import adjust_text
            adjust_text(texts, ax=ax,
                        arrowprops=dict(arrowstyle="-", color="0.5", lw=0.6),
                        expand=(1.5, 1.8))
        except ImportError:
            for t in texts:
                t.set_ha("left")

    ax.set_title(f"How strong each rule is vs how many patients have it "
                 f"(dot size = {size_col})", fontsize=12)
    if is_inf.any():
        ax.legend(fontsize=8, loc="lower right")
    plt.tight_layout()
    _finish(fig, save)
    cols = [c for c in ["antecedent", "consequent", x, color_col, y, size_col,
                        "n_FOV", "ant_abund", "con_abund"] if c in df.columns]
    return df.loc[named, cols].sort_values(x, ascending=False).round(3)


def plot_rule_metric_scatter_interactive(table, size_col="n_Patient", color_col="med_Confidence",
                                         show=True):
    """Interactive (plotly) scatter: how strong each rule is vs how many patients have it.

    Hover a point for the rule name and all its numbers — no labels drawn on the plot.
    x = lift, y = conviction (both log; conviction = inf is drawn at the top with a red ring),
    point size = `size_col`, color = `color_col`. Returns the figure.
    """
    import plotly.graph_objects as go

    if table.empty:
        print("No data to plot.")
        return None
    df = table.reset_index()
    rule_col = df.columns[0]                       # 'Clean_Rule'
    lift = df["med_Lift"].to_numpy(dtype=float)
    conv = df["med_Conviction"].to_numpy(dtype=float)
    is_inf = ~np.isfinite(conv)
    finite_max = np.nanmax(conv[np.isfinite(conv)]) if np.isfinite(conv).any() else 1.0
    yplot = np.where(is_inf, finite_max * 1.2, conv)
    conv_txt = np.where(is_inf, "∞", np.round(conv, 2).astype(str))
    sizes = df[size_col].to_numpy(dtype=float)

    def _col(name):
        return np.round(df[name].to_numpy(dtype=float), 3) if name in df.columns else np.full(len(df), np.nan)
    cd = np.column_stack([df[rule_col].astype(str), conv_txt, _col(color_col),
                          _col("n_FOV"), _col(size_col), _col("ant_abund"), _col("con_abund")])

    fig = go.Figure(go.Scatter(
        x=lift, y=yplot, mode="markers",
        marker=dict(size=sizes, sizemode="area",
                    sizeref=2.0 * max(sizes.max(), 1) / (28.0 ** 2), sizemin=4,
                    color=df[color_col].to_numpy(dtype=float), colorscale="Viridis",
                    showscale=True,
                    colorbar=dict(title="confidence", x=1.02, len=0.9, y=0.5, yanchor="middle"),
                    line=dict(width=0.5, color="white")),
        showlegend=False,
        customdata=cd,
        hovertemplate=("<b>%{customdata[0]}</b><br>"
                       "lift = %{x:.2f}   conviction = %{customdata[1]}<br>"
                       "confidence = %{customdata[2]}<br>"
                       "n_FOV = %{customdata[3]}   " + f"{size_col}" + " = %{customdata[4]}<br>"
                       "ant_abund = %{customdata[5]}   con_abund = %{customdata[6]}"
                       "<extra></extra>"),
    ))
    if is_inf.any():
        fig.add_trace(go.Scatter(
            x=lift[is_inf], y=yplot[is_inf], mode="markers",
            marker=dict(size=16, color="rgba(0,0,0,0)", line=dict(color="red", width=2)),
            name="conviction = ∞ (shown at top)", hoverinfo="skip"))
    fig.add_vline(x=float(np.nanmedian(lift)), line_dash="dash", line_color="gray", opacity=0.5)
    fig.add_hline(y=float(np.nanmedian(yplot)), line_dash="dash", line_color="gray", opacity=0.5)
    fig.update_xaxes(type="log", title="lift (log)")
    fig.update_yaxes(type="log", title="conviction (log)")
    fig.update_layout(
        title=f"How strong each rule is vs how many patients have it (dot size = {size_col})",
        width=950, height=680, showlegend=bool(is_inf.any()),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0))
    if show:
        fig.show()
    return fig


# ===========================================================================
# Rule space: FOVs as vectors of rule strengths (PCA) + the FOV maps
# ===========================================================================
# Every figure here is matplotlib, so it can be written to a file and pulled
# into result_summary.tex. The one plotly version is a twin for exploring on
# screen only — plotly cannot be saved as an image without kaleido.

FIGURE_DIR = "summary_downloads"          # written next to result_summary.tex

# Stage colors for dots on a white background (stronger than the strip palette).
PCA_STAGE_COLORS = {
    "Control_S": "#2E7D32",     # dark green
    "Control": "#81C784",       # light green
    "Mild": "#FFA726",          # orange
    "Severe": "#D32F2F",        # red
}
PCA_STAGE_ORDER = ["Control_S", "Control", "Mild", "Severe"]

# Filled once by set_cell_colors(); every FOV map then uses the same colors.
_CELL_COLORS = {}
_OTHER_COLOR = (0.5, 0.5, 0.5)


def save_figure(fig, name, dpi=200, figure_dir=None):
    """Write a figure next to the LaTeX summary, and return the path(s).

    Given a bare `name`, writes both a PDF (vector, what \\includegraphics uses) and a
    PNG (for a quick look). Give an explicit extension to write only that one.
    """
    if not name:
        return None
    directory = figure_dir if figure_dir is not None else FIGURE_DIR
    stem, ext = os.path.splitext(name)
    os.makedirs(directory, exist_ok=True)

    paths = []
    for suffix in ([ext] if ext else [".pdf", ".png"]):
        path = os.path.join(directory, stem + suffix)
        fig.savefig(path, dpi=dpi, bbox_inches="tight", facecolor=fig.get_facecolor())
        print(f"saved {path}")
        paths.append(path)
    return paths[0] if len(paths) == 1 else paths


def _finish(fig, save=None, dpi=200):
    """Save (when asked) and show — the last line of every figure below."""
    save_figure(fig, save, dpi=dpi)
    plt.show()


def set_cell_colors(df_cells, cmap="tab20b"):
    """Give every cell type one color, shared by all the FOV maps.

    Call this once after loading the cells. Returns the {cell type: color} map.
    """
    global _CELL_COLORS
    types = sorted(df_cells["cell type"].dropna().astype(str).unique())
    try:
        palette = plt.colormaps.get_cmap(cmap).resampled(max(len(types), 1))
    except AttributeError:                              # older matplotlib
        palette = plt.cm.get_cmap(cmap, max(len(types), 1))
    _CELL_COLORS = {ct: palette(i) for i, ct in enumerate(types)}
    return _CELL_COLORS


def _category_colors(values, order=None):
    """Colors for a categorical coloring: stage colors when the values are stages.

    Returns ({value: color}, ordered_values).
    """
    present = list(pd.unique(pd.Series(values).dropna()))
    if set(present) <= set(PCA_STAGE_COLORS):
        ordered = [s for s in PCA_STAGE_ORDER if s in present]
        return {s: PCA_STAGE_COLORS[s] for s in ordered}, ordered

    ordered = list(order) if order is not None else sorted(present, key=str)
    base = plt.get_cmap("tab10" if len(ordered) <= 10 else "tab20")
    n = 10 if len(ordered) <= 10 else 20
    return {v: base(i % n) for i, v in enumerate(ordered)}, ordered


# ---------------------------------------------------------------------------
# How much of the picture each component carries
# ---------------------------------------------------------------------------

def _titled(base, scope):
    """Put the scope (organ / stages) into the title, so a figure reads on its own."""
    return f"{base}  —  {scope}" if scope else base


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


# ---------------------------------------------------------------------------
# Where each FOV sits (static, for the write-up)
# ---------------------------------------------------------------------------

def plot_pca_scatter(df_pca, explained_variance, color_by, subtitle=None,
                     label_fovs=None, box_fovs=None, x="PC1", y="PC2", fov_col="FOV",
                     max_legend=12, save=None, scope=None):
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

    fig, ax = plt.subplots(figsize=(9, 7))
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
        from matplotlib.patches import Rectangle
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
    fig.suptitle(title, fontsize=13)
    if subtitle:
        ax.set_title(subtitle, fontsize=9, color="0.35")

    ax.set_xlabel(f"{x} ({ev[ix[0]]:.1f}%)", fontsize=11)
    ax.set_ylabel(f"{y} ({ev[ix[1]]:.1f}%)", fontsize=11)
    ax.grid(color="0.93", lw=0.8)
    ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()
    _finish(fig, save)


def plot_pca_scatter_interactive(df_pca, explained_variance, color_by, subtitle=None,
                                 annotate_pc1=None, annotate_pc2=None,
                                 fov_col="FOV", show=True):
    """The same picture, interactive — hover a dot for its FOV. Screen only.

    Cannot be saved as an image (needs kaleido); use `plot_pca_scatter` for the
    write-up. `annotate_pc1` / `annotate_pc2` are the {FOV: description} dicts
    from `get_representative_fovs_for_pc`; their FOVs get an arrow and a label,
    PC1's placed above the dot and PC2's below so the two sets never collide.
    """
    import plotly.express as px

    if df_pca.empty:
        print("No data to plot.")
        return None

    ev = np.asarray(explained_variance, dtype=float)
    color_map, order = None, None
    if set(df_pca[color_by].dropna().astype(str)) <= set(PCA_STAGE_COLORS):
        color_map = PCA_STAGE_COLORS
        order = {color_by: PCA_STAGE_ORDER}

    fig = px.scatter(
        df_pca, x="PC1", y="PC2", color=color_by,
        color_discrete_map=color_map, category_orders=order,
        hover_name=fov_col,
        title=f"PCA colored by {color_by}",
        subtitle=subtitle,
        labels={"PC1": f"PC1 ({ev[0]:.1f}%)", "PC2": f"PC2 ({ev[1]:.1f}%)"},
        template="plotly_white", width=900, height=600,
    )
    fig.update_traces(marker=dict(size=10, opacity=0.8,
                                  line=dict(width=1, color="DarkSlateGrey")))

    pc1 = dict(annotate_pc1 or {})
    pc2 = dict(annotate_pc2 or {})
    named = sorted(set(pc1) | set(pc2),
                   key=lambda f: df_pca.loc[df_pca[fov_col] == f, "PC1"].iloc[0])
    for i, fov in enumerate(named):
        row = df_pca[df_pca[fov_col] == fov].iloc[0]
        in1, in2 = fov in pc1, fov in pc2
        if in1 and not in2:                       # PC1 labels go above
            ay, ax_ = -50 - (i % 3) * 20, (30 if row["PC1"] > 0 else -30)
        elif in2 and not in1:                     # PC2 labels go below
            ay, ax_ = 50 + (i % 3) * 20, (30 if row["PC1"] > 0 else -30)
        else:                                     # extreme on both: point outwards
            ay = -50 if row["PC2"] > 0 else 50
            ax_ = 50 if row["PC1"] > 0 else -50
        fig.add_annotation(
            x=row["PC1"], y=row["PC2"], text=str(fov), showarrow=True,
            arrowhead=2, arrowsize=1, arrowwidth=2, arrowcolor="#636363",
            ax=ax_, ay=ay, font=dict(size=10, color="black"),
            bgcolor="rgba(255, 255, 255, 0.8)", bordercolor="#c7c7c7",
            borderwidth=1, borderpad=4)

    if show:
        fig.show()
    return fig


# ---------------------------------------------------------------------------
# Which rules pull the FOVs apart
# ---------------------------------------------------------------------------

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


def plot_pc_abundance_correlation(corr, pvals=None, subtitle=None, save=None):
    """How much of each component is just how common a cell type is.

    `corr` : cell types x PCs of Spearman rho, `pvals` the matching FDR-adjusted
    p-values. A cell reading near +1 or -1 means that component mostly follows
    that cell type's share of the FOV rather than the spatial rules.
    A star marks rho with an adjusted p below 0.05.
    """
    if corr.empty:
        print("No data to plot.")
        return

    annot = corr.round(2).astype(str)
    if pvals is not None:
        stars = pvals.reindex_like(corr) < 0.05
        annot = annot.where(~stars, annot + "*")

    height = max(3.0, 0.42 * len(corr) + 1.4)
    fig, ax = plt.subplots(figsize=(1.6 * len(corr.columns) + 3.5, height))
    sns.heatmap(corr, annot=annot, fmt="", cmap="coolwarm", center=0,
                vmin=-1, vmax=1, ax=ax, linewidths=0.5, linecolor="white",
                cbar_kws={"label": "Spearman rho"})

    fig.suptitle("PC scores vs cell-type abundance", fontsize=14)
    if subtitle:
        ax.set_title(subtitle, fontsize=9, color="0.35")
    ax.set_xlabel("component", fontsize=11)
    ax.set_ylabel("cell type", fontsize=11)
    ax.tick_params(axis="y", labelsize=9, rotation=0)
    ax.tick_params(axis="x", labelsize=10)
    plt.tight_layout()
    _finish(fig, save)


# ---------------------------------------------------------------------------
# The FOVs themselves
# ---------------------------------------------------------------------------

def _add_scale_bar_50um(ax, x_max, y_max):
    """A 50 um bar in the bottom-right corner, so sizes are readable."""
    bar_um, x_end, y_line = 50.0, x_max - 25, y_max - 25
    x_start = x_end - bar_um
    ax.plot([x_start, x_end], [y_line, y_line], color="black", linewidth=4,
            solid_capstyle="butt")
    ax.text((x_start + x_end) / 2, y_line - 12, "50 µm", color="black",
            ha="center", va="bottom", fontsize=10)


def plot_fov(fov_id, description, df_cells, df_fovs,
             target_ant_cells=None, target_cons_cells=None, ax=None, save=None):
    """A map of one FOV: every cell drawn where it sits, colored by its type.

    Give `target_ant_cells` / `target_cons_cells` to grey out everything except
    one rule's two cell types, which is how a rule is shown in a real image.
    Call `set_cell_colors(df_cells)` once first so the colors match everywhere.
    """
    df_fov = df_cells[df_cells["fov"] == fov_id].copy()
    if df_fov.empty:
        print(f"No cells found for FOV {fov_id}")
        return

    meta = df_fovs[df_fovs["FOV"] == fov_id]
    size_um = meta["Size [um]"].iloc[0] if not meta.empty else 400
    cell_size = 90 if size_um == 400 else 45

    own_fig = ax is None
    if own_fig:
        fig, ax = plt.subplots(figsize=(10, 10), facecolor="#ffffff")
    ax.set_facecolor("#eaeaeaff")

    if target_ant_cells or target_cons_cells:
        targets = list(target_ant_cells or []) + list(target_cons_cells or [])
        title = (f"Highlighted Rule in {fov_id}\n"
                 f"{', '.join(target_ant_cells or [])} -> {', '.join(target_cons_cells or [])}")
        other = ~df_fov["cell type"].isin(targets)
        if other.any():
            g = df_fov[other]
            ax.scatter(g["x_um"], g["y_um"], s=cell_size, c=[_OTHER_COLOR],
                       alpha=0.2, linewidths=0, label="Other")
        for ct, g in df_fov[~other].groupby("cell type"):
            ax.scatter(g["x_um"], g["y_um"], s=cell_size,
                       c=[_CELL_COLORS.get(ct, (0, 0, 0))], label=ct,
                       alpha=1.0, linewidths=0)
        legend_types = targets
    else:
        title = f"FOV: {fov_id}" + (f" ({description})" if description else "")
        for ct, g in df_fov.groupby("cell type"):
            ax.scatter(g["x_um"], g["y_um"], s=cell_size,
                       c=[_CELL_COLORS.get(ct, (1, 1, 1))], label=ct,
                       alpha=0.9, linewidths=0)
        legend_types = sorted(df_fov["cell type"].dropna().astype(str).unique())

    handles = [plt.Line2D([0], [0], marker="o", color="w", markeredgecolor="none",
                          markerfacecolor=_CELL_COLORS.get(ct, "black"),
                          markersize=8, label=ct) for ct in legend_types]

    x_max, y_max = df_fov["x_um"].max(), df_fov["y_um"].max()
    ax.set_axisbelow(True)
    ax.xaxis.set_major_locator(MultipleLocator(100.0))
    ax.yaxis.set_major_locator(MultipleLocator(100.0))
    ax.xaxis.set_minor_locator(MultipleLocator(25.0))
    ax.yaxis.set_minor_locator(MultipleLocator(25.0))
    ax.grid(which="minor", color="white", linestyle="-", linewidth=0.8, alpha=0.7)
    ax.grid(which="major", color="lightgray", linestyle="-", linewidth=0.8, alpha=0.8)
    if not pd.isna(x_max) and not pd.isna(y_max):
        _add_scale_bar_50um(ax, x_max=x_max, y_max=y_max)

    ax.set_title(title, color="black", fontsize=14)
    ax.set_xlabel("x (µm)", color="black")
    ax.set_ylabel("y (µm)", color="black")
    ax.set_aspect("equal", adjustable="box")
    ax.invert_yaxis()
    ax.tick_params(axis="both", colors="black", labelsize=9)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.legend(handles=handles, title="Cell type", bbox_to_anchor=(1.02, 1),
              loc="upper left", fontsize=9, title_fontsize=10, frameon=True)

    if own_fig:
        plt.tight_layout()
        _finish(fig, save)


def plot_fov_panel(target_fovs, df_cells, df_fovs, num_cols=1, save=None):
    """One map per FOV, side by side, lettered A, B, C … for the write-up.

    `target_fovs` is the {FOV: description} dict from
    `get_representative_fovs_for_pc`, so the panel reads left-to-right along
    the component.
    """
    if not target_fovs:
        print("No FOVs to plot.")
        return
    fovs = list(target_fovs)
    num_cols = max(1, min(num_cols, len(fovs)))
    num_rows = (len(fovs) + num_cols - 1) // num_cols

    fig, axes = plt.subplots(num_rows, num_cols,
                             figsize=(10 * num_cols, 10 * num_rows),
                             facecolor="#ffffff", squeeze=False)
    axes = axes.flatten()

    for i, fov in enumerate(fovs):
        plot_fov(fov, target_fovs[fov], df_cells, df_fovs, ax=axes[i])
        axes[i].text(-0.05, 1.05, chr(65 + i), transform=axes[i].transAxes,
                     fontsize=20, fontweight="bold", va="top", ha="right")
    for ax in axes[len(fovs):]:
        ax.set_visible(False)

    plt.tight_layout()
    _finish(fig, save)


# ===========================================================================
# Rule matrix: which cell type sits next to which
# ===========================================================================
# One dot per rule on a center-cell x neighbor-cell grid. Color = how strong the
# rule is (lift), size = the second number: confidence inside a single FOV, or the
# share of a group that has the rule. Every size below is measured in inches, so a
# grid of 32 cell types stays as readable as a grid of 8, alone or in a panel.

CELL_GROUP_COLORS = {"epithelial": "#c0392b", "immune": "#2980b9",
                     "structure": "#27ae60", "other": "#7f8c8d"}
CELL_GROUP_ORDER = ["epithelial", "immune", "structure", "other"]

_RATIO_METRICS = {"Lift", "Conviction"}     # 1 = no association -> the scale diverges at 1
_ZERO_METRICS = {"Leverage"}                # 0 = no association
_RATIO_TICKS = [0.125, 0.25, 0.5, 0.75, 1, 1.5, 2, 4, 8, 16, 32]
_COLOR_QUANTILE = 0.98      # the few strongest rules saturate instead of washing the rest out

_SQUARE_ALONE = 0.30        # inches per cell type in a grid drawn on its own
_SQUARE_PANEL = 0.22        # ... and in a panel, where many grids share the page
_LEGEND_INCHES = 3.0        # the right margin the keys live in


# --- what a dot looks like -------------------------------------------------

def _finite(values):
    """The values as floats, with infinity replaced by the largest finite one."""
    v = np.asarray(values, dtype=float)
    ok = v[np.isfinite(v)]
    return np.where(np.isposinf(v), ok.max() if ok.size else 1.0, v)


def _limit(values):
    """How far the scale reaches from its middle. The most extreme few rules are left
    outside it (they simply take the end color), so the rest of the grid stays readable."""
    v = np.abs(np.asarray(values, dtype=float))
    v = v[np.isfinite(v)]
    if not v.size:
        return 1.0
    return float(np.quantile(v, _COLOR_QUANTILE)) or float(v.max()) or 1.0


def _color_scale(values, col):
    """How a metric becomes a color.

    Lift and conviction are ratios around 1, so they get a scale that is symmetric
    in log2 - a rule twice as likely and a rule half as likely are equally far from
    the middle - while the colorbar is still written in the metric's own units.
    """
    v = _finite(values)
    if col in _RATIO_METRICS:
        def to_log(x):
            f = _finite(x)
            return np.log2(np.where(f > 0, f, np.nan))
        limit = _limit(to_log(v))
        ticks = [t for t in _RATIO_TICKS if abs(np.log2(t)) <= limit]
        return (to_log, Normalize(-limit, limit), "RdBu_r",
                [np.log2(t) for t in ticks], [f"{t:g}" for t in ticks])
    if col in _ZERO_METRICS:
        limit = _limit(v)
        return _finite, Normalize(-limit, limit), "RdBu_r", None, None
    return _finite, Normalize(0.0, _limit(v)), "Reds", None, None


def _grid_side(n_cells, per_cell):
    """How wide the grid itself is, in inches."""
    return max(4.2, per_cell * n_cells)


def _dot_area(n_cells, per_cell):
    """The area of the biggest dot - a little smaller than one square of the grid,
    so a full-size dot never spills into its neighbors."""
    square_pt = _grid_side(n_cells, per_cell) / n_cells * 72
    return np.pi / 4 * (0.82 * square_pt) ** 2


def _size_points(values, size_ref, max_area):
    """Dot areas. `size_ref` is shared by a whole panel, so the same size means the
    same number in every grid of that panel."""
    frac = np.clip(_finite(values) / size_ref, 0.0, 1.0) if size_ref else 0.0
    return max_area * (0.06 + 0.94 * frac)


def build_matrix_scale(frames, color_col, size_col, size_ref=None):
    """One color scale and one dot-size reference for a whole panel of grids.

    `frames` : the tables that will be drawn. Pass `size_ref=1.0` when the size is
    already a share, so the dots read as absolute percentages.
    """
    frames = [f for f in frames if len(f)]
    colors = pd.concat([f[color_col] for f in frames]) if frames else pd.Series(dtype=float)
    sizes = pd.concat([f[size_col] for f in frames]) if frames else pd.Series(dtype=float)
    if size_ref is None:
        size_ref = float(_finite(sizes).max()) if len(sizes) else 1.0
    transform, norm, cmap, ticks, tick_labels = _color_scale(colors, color_col)
    return {"transform": transform, "norm": norm, "cmap": cmap,
            "ticks": ticks, "tick_labels": tick_labels, "size_ref": max(size_ref, 1e-9)}


# --- the empty grid, the titles and the keys -------------------------------

def _draw_grid(ax, cell_order, cell_groups, fontsize):
    """The grid a dot sits on: one square per pair of cell types.

    Thin lines run between every row and column and a stronger line separates the cell
    groups, so a dot can be followed back to the names on the edges. The background is
    left white on purpose - any tint behind the dots would change how their color reads.
    """
    size = len(cell_order)
    groups = [(cell_groups or {}).get(cell, "other") for cell in cell_order]

    ax.set_xlim(-0.5, size - 0.5)
    ax.set_ylim(-0.5, size - 0.5)

    for i in range(1, size):                           # the line between two groups
        if groups[i] != groups[i - 1]:
            ax.axhline(i - 0.5, color="0.55", lw=1.0, zorder=2)
            ax.axvline(i - 0.5, color="0.55", lw=1.0, zorder=2)

    ax.set_xticks(range(size))
    ax.set_yticks(range(size))
    ax.set_xticks(np.arange(size) + 0.5, minor=True)   # the lines run *between* squares
    ax.set_yticks(np.arange(size) + 0.5, minor=True)
    ax.grid(which="minor", color="0.90", linewidth=0.6, zorder=1)
    ax.grid(which="major", visible=False)
    ax.tick_params(which="both", length=0)

    ax.set_xticklabels(cell_order, rotation=90, fontsize=fontsize)
    ax.set_yticklabels(cell_order, fontsize=fontsize)
    for labels in (ax.get_xticklabels(), ax.get_yticklabels()):
        for label in labels:
            group = (cell_groups or {}).get(label.get_text(), "other")
            label.set_color(CELL_GROUP_COLORS.get(group, "black"))
    for spine in ax.spines.values():
        spine.set_visible(False)


def _figure_titles(fig, title, subtitle, center):
    """Title and subtitle, placed in inches from the top so they never collide.

    Returns the fraction of the figure left for the grids.
    """
    height = fig.get_figheight()
    if title:
        fig.suptitle(title, fontsize=15, x=center, y=1 - 0.34 / height)
    if subtitle:
        fig.text(center, 1 - 0.66 / height, subtitle, ha="center", va="top",
                 fontsize=10, color="0.35")
    used = 0.30 + 0.34 * bool(title) + 0.40 * bool(subtitle)
    return 1 - used / height


def _matrix_legends(fig, scale, color_col, size_col, size_label, cell_groups,
                    left, top, max_area):
    """The cell-group key, the colorbar and the dot-size key, stacked in the margin.

    Everything is placed in inches from `top`, so the stack looks the same whether the
    panel has one row of grids or six.
    """
    width, height = fig.get_figwidth(), fig.get_figheight()

    def down(inches):
        return top - inches / height

    groups = [g for g in CELL_GROUP_ORDER if g in set((cell_groups or {}).values())]
    if groups:
        marks = [plt.Line2D([0], [0], marker="s", linestyle="none", markersize=9,
                            color=CELL_GROUP_COLORS[g], label=g) for g in groups]
        fig.legend(handles=marks, loc="upper left", frameon=False,
                   bbox_to_anchor=(left + 0.55 / width, down(0.1)),
                   title="cell group", fontsize=9, title_fontsize=10)

    mappable = plt.cm.ScalarMappable(norm=scale["norm"], cmap=scale["cmap"])
    cbar = fig.colorbar(
        mappable, extend="both" if scale["norm"].vmin < 0 else "max",
        cax=fig.add_axes([left + 0.35 / width, down(4.4), 0.16 / width, 2.4 / height]))
    if scale["ticks"] is not None:
        cbar.set_ticks(scale["ticks"])
        cbar.set_ticklabels(scale["tick_labels"])
    cbar.set_label(color_col.lower(), fontsize=10)
    cbar.ax.tick_params(labelsize=9)

    ref, is_share = scale["size_ref"], size_col == "share"
    handles = [plt.Line2D([0], [0], marker="o", linestyle="none", color="0.55",
                          markersize=float(np.sqrt(_size_points([v], ref, max_area)[0])),
                          label=f"{v:.0%}" if is_share else f"{v:.2g}")
               for v in (0.25 * ref, 0.5 * ref, ref)]
    fig.legend(handles=handles, loc="upper left", frameon=False,
               bbox_to_anchor=(left + 0.55 / width, down(5.1)),
               title=size_label or size_col.lower(), fontsize=9, title_fontsize=10,
               labelspacing=1.6, borderpad=0.6)


# --- the two calls the notebook makes --------------------------------------

def plot_rule_matrix(df, cell_order, cell_groups=None, color_col="Lift",
                     size_col="Confidence", title=None, subtitle=None, scale=None,
                     size_label=None, ax=None, per_cell=_SQUARE_ALONE,
                     show_diagonal=True, save=None):
    """One grid: a dot per rule, center cell (y) against neighbor cell (x).

    df         : one row per rule, with 'antecedent', 'consequent' and both metrics.
    cell_order : the fixed cell-type order - the same list in every grid, so two
                 grids can be compared square by square.
    cell_groups: {cell type: group}, which colors the names and splits the grid.
    scale      : from `build_matrix_scale`; pass the panel's scale in to share it.
                 Left out, the grid builds its own scale from its own numbers.
    """
    if scale is None:
        scale = build_matrix_scale([df], color_col, size_col)
    size = len(cell_order)
    max_area = _dot_area(size, per_cell)

    own_fig = ax is None
    if own_fig:
        side = _grid_side(size, per_cell)
        fig, ax = plt.subplots(figsize=(side + _LEGEND_INCHES + 1.6, side + 2.4))

    _draw_grid(ax, cell_order, cell_groups, fontsize=9 if own_fig else 7)
    if show_diagonal:
        ax.plot([-0.5, size - 0.5], [-0.5, size - 0.5], color="0.72", lw=0.9,
                linestyle=(0, (4, 4)), zorder=2)

    place = {cell: i for i, cell in enumerate(cell_order)}
    rows = df[df["antecedent"].isin(place) & df["consequent"].isin(place)]
    if len(rows):
        ax.scatter(rows["consequent"].map(place), rows["antecedent"].map(place),
                   s=_size_points(rows[size_col], scale["size_ref"], max_area),
                   c=scale["transform"](rows[color_col]),
                   cmap=scale["cmap"], norm=scale["norm"],
                   edgecolor="0.25", linewidth=0.4, zorder=3)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("neighbor cell", fontsize=11 if own_fig else 9)
    ax.set_ylabel("center cell", fontsize=11 if own_fig else 9)

    if not own_fig:
        if title:
            ax.set_title(title, fontsize=11, pad=8)
        return

    width = fig.get_figwidth()
    left = 1 - _LEGEND_INCHES / width
    top = _figure_titles(fig, title or "which cell type sits next to which",
                         subtitle, center=left / 2)
    plt.tight_layout(rect=[0, 0, left, top])
    _matrix_legends(fig, scale, color_col, size_col, size_label, cell_groups,
                    left=left, top=top, max_area=max_area)
    _finish(fig, save)


def plot_rule_matrix_panel(frames, cell_order, cell_groups=None, color_col="Lift",
                           size_col="Confidence", num_cols=3, title=None,
                           subtitle=None, size_label=None, size_ref=None,
                           show_diagonal=True, save=None):
    """One grid per entry of `frames` ({title: table}), all on one shared scale.

    The color scale and the dot-size reference are built once over every table, so a
    big dark dot means the same thing in all of them.
    """
    if not frames:
        print("No data to plot.")
        return
    scale = build_matrix_scale(list(frames.values()), color_col, size_col, size_ref=size_ref)

    size = len(cell_order)
    side = _grid_side(size, _SQUARE_PANEL)
    num_cols = max(1, min(num_cols, len(frames)))
    num_rows = (len(frames) + num_cols - 1) // num_cols

    fig, axes = plt.subplots(num_rows, num_cols, squeeze=False,
                             figsize=((side + 1.2) * num_cols + _LEGEND_INCHES,
                                      (side + 1.6) * num_rows + 1.2))
    axes = axes.flatten()
    for ax, (name, df) in zip(axes, frames.items()):
        plot_rule_matrix(df, cell_order, cell_groups=cell_groups, color_col=color_col,
                         size_col=size_col, title=name, scale=scale, ax=ax,
                         per_cell=_SQUARE_PANEL, show_diagonal=show_diagonal)
    for ax in axes[len(frames):]:
        ax.set_visible(False)

    left = 1 - _LEGEND_INCHES / fig.get_figwidth()
    top = _figure_titles(fig, title, subtitle, center=left / 2)
    plt.tight_layout(rect=[0, 0, left, top])
    _matrix_legends(fig, scale, color_col, size_col, size_label, cell_groups,
                    left=left, top=top, max_area=_dot_area(size, _SQUARE_PANEL))
    _finish(fig, save)


