"""Visualization helpers shared by the result-summary investigations.

Only plotting lives here; data prep / aggregation stays in the notebook.
"""
import os

import matplotlib.pyplot as plt
from matplotlib.colors import PowerNorm, Normalize, to_rgb
from skimage.color import rgb2lab, deltaE_ciede2000
from matplotlib.patches import Patch
from matplotlib.ticker import MultipleLocator
import numpy as np
import pandas as pd
import seaborn as sns

# Stage colors for the fingerprint strips: the same green -> amber -> red climb as
# PCA_STAGE_COLORS below, one step lighter so a strip stays quiet behind the heatmap.
_STAGE_PALETTE = {
    "Control_S": "#3E9068", "Control": "#7FCBA0",
    "Mild": "#F2AA4C", "Severe": "#DE6767", "Unknown": "#e3e2dc",
}


def _prevalence_norm(df_pct, gamma=0.5):
    """Non-linear color norm so a few very-prevalent rules don't wash out the rest.

    gamma < 1 stretches the low/mid range (PowerNorm: color ~ value**gamma).
    """
    vmax = float(np.nanmax(df_pct.to_numpy())) if df_pct.size else 1.0
    return PowerNorm(gamma=gamma, vmin=0.0, vmax=max(vmax, 1e-9))


def _build_annotations(df_plot, orig_col_by_display, total_col, total_units, stage_totals,
                       denominators=None):
    """Return (percentage matrix, 'count (pct%)' annotation matrix)."""
    df_pct = df_plot.astype(float).copy()
    annot = []
    for idx, row in df_plot.iterrows():
        annot_row = []
        for display_col, val in row.items():
            orig = orig_col_by_display[display_col]
            if denominators is None:
                total = total_units if orig == total_col else stage_totals.get(orig, val)
            else:
                total = denominators.at[idx, orig]
            pct = (val / total * 100) if total else 0.0
            df_pct.at[idx, display_col] = pct
            count = f"{int(val)}" if denominators is None else f"{int(val)}/{int(total)}"
            annot_row.append(f"{count} ({pct:.1f}%)")
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
    denominators=None,
    save=None,
    scope=None,
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
    denominators : optional rule x column table of eligible-unit counts. When given,
        every percentage uses its own rule-specific denominator.
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
        if denominators is not None:
            disp = f"{col}\n(rule-specific n)"
        elif col == total_col:
            disp = f"{col}\n(n={total_units})"
        elif col in stage_totals:
            disp = f"{col}\n(n={stage_totals[col]})"
        else:
            disp = col
        renamed[col] = disp
        orig_col_by_display[disp] = col
    df_plot = df_plot.rename(columns=renamed)

    if denominators is not None:
        denominators = denominators.reindex(
            index=df_plot.index, columns=df_agg.columns, fill_value=0
        ).fillna(0)
    df_pct, annot = _build_annotations(
        df_plot, orig_col_by_display, total_col, total_units, stage_totals, denominators
    )

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
        cbar_kws={"label": f"Percentage of {'eligible ' if denominators is not None else ''}{id_col} (%)"},
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

    title = ("Eligibility-controlled top rule prevalence"
             if denominators is not None else "Top rule prevalence")
    score = score_col.replace(" score", "").lower()
    details = [scope, f"Score: {score}", f"Unit: {id_col}"]
    if organs:
        details.append("Organ: " + " / ".join(organs))
    shown_stages = stage_order or stages
    if shown_stages:
        details.append("Stages: " + " / ".join(map(str, shown_stages)))
    details.append("Denominator: eligible units" if denominators is not None
                   else "Denominator: all units")
    fig.suptitle(title, fontsize=14, y=0.99)
    fig.text(0.5, 0.955, " · ".join(x for x in details if x),
             ha="center", va="top", fontsize=8.5, color="#706E68")
    ax.set_ylabel("" if draw_bars else "Cleaned Rule", fontsize=12)
    ax.set_xlabel(score_col, fontsize=12)
    ax.tick_params(axis="both", which="major", labelsize=10,
                   top=True, labeltop=True, bottom=False, labelbottom=False)
    fig.tight_layout(rect=(0, 0, 1, 0.91))
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
    by="patient",
    consistency=None,
):
    """Rules x patients heatmap: each cell = share of that patient's FOVs that have the rule.

    With `by='fov'` the columns are FOVs and the values are log2 of lift, so the scale is
    diverging around 0 - above 0 the cell types attract, below 0 they avoid.

    Stage strips above the columns color every column by stage (one strip per `strip_scores`).
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

    if by == "fov":
        patients = list(mat.columns)                    # already grouped by patient, keep it
    elif order == "cluster":
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
    widths = [45, 1] if consistency is None else [45, 7, 1]
    gs = fig.add_gridspec(n_strip + 1, len(widths), width_ratios=widths,
                          height_ratios=[strip_h] * n_strip + [main_h],
                          hspace=0.10, wspace=0.015)
    strip_axes = [fig.add_subplot(gs[i, 0]) for i in range(n_strip)]
    main_ax = fig.add_subplot(gs[n_strip, 0])
    ax_bar = fig.add_subplot(gs[n_strip, 1]) if consistency is not None else None
    cax = fig.add_subplot(gs[n_strip, -1])

    for ax_s, (sc, series) in zip(strip_axes, stage_by_score.items()):
        colors = [_STAGE_PALETTE.get(series.get(p, "Unknown"), "#dddddd") for p in patients]
        rgb = np.array([to_rgb(c) for c in colors]).reshape(1, len(patients), 3)
        ax_s.imshow(rgb, aspect="auto", extent=[0, len(patients), 0, 1])
        ax_s.set_xlim(0, len(patients))
        ax_s.set_xticks([])
        ax_s.set_yticks([0.5])
        ax_s.set_yticklabels([sc.replace(" score", "")], fontsize=8)

    if by == "fov":
        lim = float(np.nanpercentile(np.abs(mat.to_numpy()), 98)) or 1.0
        sns.heatmap(mat, ax=main_ax, cmap="RdBu_r", vmin=-lim, vmax=lim, cbar_ax=cax,
                    cbar_kws={"label": "log2 of lift"})
        main_ax.set_xlabel(f"FOVs (n={len(patients)}), grouped by patient", fontsize=11)
    else:
        sns.heatmap(mat, ax=main_ax, cmap=cmap, vmin=0, vmax=1, cbar_ax=cax,
                    cbar_kws={"label": "share of the patient's FOVs"})
        how = ("grouped by how similar their rules are" if order == "cluster"
               else f"ordered by {score_col}")
        main_ax.set_xlabel(f"Patients (n={len(patients)}), {how}", fontsize=11)
    main_ax.set_ylabel("Rule", fontsize=11)
    main_ax.set_xticks(np.arange(len(patients)) + 0.5)
    main_ax.set_xticklabels(patients, rotation=90, fontsize=6)
    main_ax.tick_params(axis="y", labelsize=9)

    if by == "fov":                        # a line where one patient's FOVs end
        owner = df_fovs.set_index("FOV")["PatientID"].reindex(patients).to_numpy()
        for b in np.flatnonzero(owner[1:] != owner[:-1]) + 1:
            main_ax.axvline(b, color="0.25", lw=0.7)

    if ax_bar is not None:                 # how consistent each rule is within a patient
        vals = consistency.reindex(mat.index).to_numpy(dtype=float)
        y = np.arange(len(mat)) + 0.5
        ax_bar.barh(y, np.nan_to_num(vals), height=0.8, color="#4477aa")
        ax_bar.set_ylim(main_ax.get_ylim())
        ax_bar.set_xlim(0, 1)
        ax_bar.set_xticks([0, 1])
        ax_bar.set_yticks([])
        ax_bar.set_xlabel("Avg of %FOVs / Patient", fontsize=8)
        ax_bar.tick_params(labelsize=8)
        ax_bar.spines[["top", "right", "left"]].set_visible(False)

    stages_present = [s for s in (stage_order or list(_STAGE_PALETTE)) if s in _STAGE_PALETTE]
    handles = [Patch(color=_STAGE_PALETTE[s], label=s) for s in stages_present]
    # Outside the whole figure, so the consistency bars cannot cover it.
    fig.legend(handles=handles, title="Stage", loc="upper left",
               bbox_to_anchor=(1.0, 0.98), bbox_transform=fig.transFigure,
               fontsize=8, title_fontsize=9)

    what = "FOV" if by == "fov" else "patient"
    fig.suptitle(f"Which rules each {what} has", fontsize=13, y=0.99)
    fig.text(
        0.5, 0.957,
        (f"{scope or 'Rules: all'} · Score: {score_col.replace(' score', '').lower()} · "
         f"Unit: {what} · Rules shown: {len(mat)}"),
        ha="center", va="top", fontsize=8.5, color="#706E68",
    )
    fig.subplots_adjust(top=0.90)
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


def _plain_log_ticks(ax, which="x", nice=None):
    """Label a log axis with ordinary numbers (0.01, 0.05, 0.2) instead of 10^-2.

    `nice` gives a denser or sparser set of candidate ticks; those inside the axis
    limits are used.
    """
    from matplotlib.ticker import FuncFormatter, NullFormatter
    axis = ax.xaxis if which == "x" else ax.yaxis
    lo, hi = (ax.get_xlim() if which == "x" else ax.get_ylim())
    nice = nice or [0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5,
                    1, 2, 5, 10, 20, 50, 100, 200, 500]
    ticks = [t for t in nice if lo <= t <= hi]
    if len(ticks) >= 2:
        axis.set_ticks(ticks)
    axis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
    axis.set_minor_formatter(NullFormatter())


def plot_metric_vs_abundance(rule_rows, fov_frac, metric="Lift", show_trend=True,
                             title_note=None, save=None, scope=None):
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
    note = f" · {title_note}" if title_note else ""
    fig.suptitle(f"{metric} and cell abundance", fontsize=12.5, y=0.99)
    fig.text(
        0.5, 0.94,
        (f"{scope or 'Rules: all'} · Unit: rule occurrence (rule × FOV) · "
         f"Observations: {len(rule_rows)}{note}"),
        ha="center", va="top", fontsize=8.5, color="#706E68",
    )
    fig.tight_layout(rect=(0, 0, 1, 0.87))
    _finish(fig, save)


def plot_rule_metric_scatter(table, x="med_Lift", y="med_Conviction", size_col="n_Patient",
                             color_col="med_Confidence", annotate_top=8, save=None,
                             scope=None):
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

    fig.suptitle("Rule strength and reproducibility", fontsize=13, y=0.99)
    fig.text(
        0.5, 0.95,
        (f"{scope or 'Rules: all'} · Unit: rule · X: {x} · Y: {y} · "
         f"Size: {size_col} · Color: {color_col}"),
        ha="center", va="top", fontsize=8.5, color="#706E68",
    )
    if is_inf.any():
        ax.legend(fontsize=8, loc="lower right")
    fig.tight_layout(rect=(0, 0, 1, 0.91))
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
# A green -> amber -> red climb, because the stages are an order, not a list of names.
# Checked for colour blindness: every pair of the three stages that share a plot stays
# apart under protanopia, deuteranopia and tritanopia.
PCA_STAGE_COLORS = {
    "Control_S": "#1F7A4D",     # deep green
    "Control": "#4FB477",       # green
    "Mild": "#E8890C",          # amber
    "Severe": "#D03B3B",        # red
}
PCA_STAGE_ORDER = ["Control_S", "Control", "Mild", "Severe"]

# Corner colors for the archetype figures: the names on the PCA and the map titles that
# go with them. Deliberately clear of the stage palette above - no green, amber or red -
# so a corner color can never be read as a stage. All dark enough to stay legible as text
# on white, spread in lightness as well as hue, and ordered so that the first few are the
# furthest apart. Cycles when there are more corners than colors.
CORNER_COLORS = ["#4B2E83",     # violet
                 "#1B9AAA",     # teal
                 "#A11D5B",     # magenta
                 "#8A6A2F",     # bronze
                 "#3C6FB4",     # blue
                 "#55566A"]     # slate


def corner_colors(names):
    """{name: color} for the archetype corners, in the order the names first appear.

    One place, so the PCA labels and the map titles cannot drift apart.
    """
    ordered = list(dict.fromkeys(names))
    return {name: CORNER_COLORS[i % len(CORNER_COLORS)]
            for i, name in enumerate(ordered)}

# The ink and furniture every figure shares, so the whole summary reads as one set.
INK = "#52514e"           # text that is not a title
NEUTRAL = "#b8b7b1"       # measured, but not worth colouring in
HAIRLINE = "#c3c2b7"      # cutoff lines
ZERO = "#8d8c85"          # the line a value is read against
GRID = "#e1e0d9"


def spread_labels(values, min_gap, lo, hi):
    """Nudge labels apart until none overlap, keeping their order and staying inside.

    Works in whatever units it is handed; for a log axis, pass the logs.
    """
    values = np.asarray(values, dtype=float)
    placed = values.copy()
    order = np.argsort(placed)
    for lower, upper in zip(order, order[1:]):
        placed[upper] = max(placed[upper], placed[lower] + min_gap)
    placed -= placed.mean() - values.mean()          # keep the block where the marks are
    placed += min(0.0, hi - placed.max()) + max(0.0, lo - placed.min())
    return placed


def tidy_axes(ax, grid=None, hide=("top", "right")):
    """Hairline grid behind the marks, and no frame around them.

    grid : 'x', 'y', 'both', or None for no grid.
    hide : which spines to drop; () keeps the frame.
    """
    if grid:
        ax.grid(axis=grid, color=GRID, lw=0.8)
        ax.set_axisbelow(True)
    if hide:
        ax.spines[list(hide)].set_visible(False)


# Filled once by set_cell_colors(); every FOV map then uses the same colors.
_CELL_COLORS = {}
_OTHER_COLOR = (0.5, 0.5, 0.5)
_COUNTED_GREY = "#D5D5D5"       # a cell that took no part, as faint as a greyed one
# The part each cell plays where a rule is counted. Kept apart from the cell-type
# palette, and in drawing order: the center goes on top.
ROLE_COLORS = {"center": "#1A1A1A", "antecedent": "#D08C34", "consequent": "#7048E8"}
_CELL_COLOR_OVERRIDES = {
    "Endothelial": "#0072B2",
    "Epithelial": "#7A9E3F",
}

# Spares used only when a rule's own cell types are too close to tell apart.
_COLOR_RESERVE = (
    "#D55E00", "#0072B2", "#009E73", "#CC79A7", "#E69F00",
    "#56B4E9", "#332288", "#882255", "#117733", "#DDCC77",
)
MIN_CELL_DELTA_E = 20.0


def _lab(color):
    return rgb2lab(np.array(to_rgb(color), dtype=float).reshape(1, 1, 3))


def _delta_e(one, other):
    return float(deltaE_ciede2000(_lab(one), _lab(other))[0, 0])


def _too_close(colors, floor):
    """The first pair of names whose colors are within `floor`, or None."""
    names = list(colors)
    worst = None
    for i, first in enumerate(names):
        for second in names[i + 1:]:
            distance = _delta_e(colors[first], colors[second])
            if distance < floor and (worst is None or distance < worst[0]):
                worst = (distance, first, second)
    return worst


def _crowded(name, colors):
    """How close this name sits to the rest: smaller means more crowded."""
    return min(
        (_delta_e(colors[name], colors[other])
         for other in colors if other != name),
        default=float("inf"),
    )


def _which_to_move(first, second, colors):
    """Prefer moving the type without a chosen color, then the more crowded one."""
    pinned = (first in _CELL_COLOR_OVERRIDES, second in _CELL_COLOR_OVERRIDES)
    if pinned[0] != pinned[1]:
        return second if pinned[0] else first
    return first if _crowded(first, colors) <= _crowded(second, colors) else second


def _replacement(colors, avoid, floor):
    """The reserve color furthest from both the rule's colors and the palette."""
    others = [color for name, color in colors.items() if name != avoid]
    palette = [color for name, color in _CELL_COLORS.items() if name != avoid]
    best, best_score = None, -1.0
    for candidate in _COLOR_RESERVE:
        within = min((_delta_e(candidate, color) for color in others),
                     default=float("inf"))
        if within < floor:
            continue
        score = within + 0.25 * min(
            (_delta_e(candidate, color) for color in palette),
            default=float("inf"),
        )
        if score > best_score:
            best, best_score = candidate, score
    return best


def resolve_cell_colors(cell_types, floor=MIN_CELL_DELTA_E):
    """The palette with a rule's own cell types pulled apart when two look alike.

    The base palette holds fewer distinct colors than there are cell types, so
    two of a rule's types can arrive identical. Pass the result to every panel of
    one figure and to its legend, so the whole figure agrees.
    """
    wanted = sorted({name for name in cell_types if name in _CELL_COLORS})
    if len(wanted) < 2:
        return dict(_CELL_COLORS)

    chosen = {name: _CELL_COLORS[name] for name in wanted}
    for _ in range(len(wanted)):
        clash = _too_close(chosen, floor)
        if clash is None:
            break
        _, first, second = clash
        move = _which_to_move(first, second, chosen)
        spare = _replacement(chosen, move, floor)
        if spare is None:
            break
        chosen[move] = spare
    return {**_CELL_COLORS, **chosen}


def save_figure(fig, name, dpi=200, figure_dir=None):
    """Write a figure next to the LaTeX summary, and return the path.

    PDF only - vector, and the one thing \\includegraphics needs. Give an explicit
    extension to write that format instead.
    """
    if not name:
        return None
    directory = figure_dir if figure_dir is not None else FIGURE_DIR
    stem, ext = os.path.splitext(name)
    os.makedirs(directory, exist_ok=True)

    path = os.path.join(directory, stem + (ext or ".pdf"))
    fig.savefig(path, dpi=dpi, bbox_inches="tight", facecolor=fig.get_facecolor())
    print(f"saved {path}")
    return path


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
    _CELL_COLORS.update({
        cell: color for cell, color in _CELL_COLOR_OVERRIDES.items()
        if cell in _CELL_COLORS
    })
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


def figure_titles(fig, title, organ=None, subtitle=None, params=None, note=None,
                  align="center"):
    """Draw a short title hierarchy with a fixed physical gap above the axes."""
    lines = [
        (subtitle, 9.8, "#5F5D58", "medium"),
        (params, 8.5, "#898781", "normal"),
        (note, 8.1, "#898781", "normal"),
    ]
    lines = [line for line in lines if line[0]]
    height_points = fig.get_figheight() * 72
    x = 0.5 if align == "center" else 0.01
    ha = "center" if align == "center" else "left"
    y = 1 - 7 / height_points
    fig.suptitle(_titled(title, organ), fontsize=13.5, x=x, y=y, ha=ha)
    for line, fontsize, color, weight in lines:
        y -= (17 if fontsize > 9 else 15) / height_points
        fig.text(x, y, line, fontsize=fontsize, color=color, fontweight=weight,
                 ha=ha, va="top")
    axes_top = y - 42 / height_points
    fig.subplots_adjust(top=axes_top)
    return axes_top


# ---------------------------------------------------------------------------
# Which rules pull the FOVs apart
# ---------------------------------------------------------------------------

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
             target_ant_cells=None, target_cons_cells=None, ax=None, save=None,
             show_legend=True, cell_size=None, colors=None):
    """A map of one FOV: every cell drawn where it sits, colored by its type.

    Give `target_ant_cells` / `target_cons_cells` to grey out everything except
    one rule's two cell types, which is how a rule is shown in a real image.
    Call `set_cell_colors(df_cells)` once first so the colors match everywhere.
    `colors` overrides that shared map for this one panel, which is how a figure
    pulls apart two rule cell types that would otherwise look the same.
    """
    colors = _CELL_COLORS if colors is None else colors
    df_fov = df_cells[df_cells["fov"] == fov_id].copy()
    if df_fov.empty:
        print(f"No cells found for FOV {fov_id}")
        return

    meta = df_fovs[df_fovs["FOV"] == fov_id]
    size_um = meta["Size [um]"].iloc[0] if not meta.empty else 400
    cell_size = cell_size if cell_size is not None else (90 if size_um == 400 else 45)

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
                       c=[colors.get(ct, (0, 0, 0))], label=ct,
                       alpha=1.0, linewidths=0)
        legend_types = targets
    else:
        # The description goes on its own line: in a panel a one-line title runs past
        # the map and into the letter beside it.
        title = f"FOV: {fov_id}" + (f"\n{description}" if description else "")
        for ct, g in df_fov.groupby("cell type"):
            ax.scatter(g["x_um"], g["y_um"], s=cell_size,
                       c=[colors.get(ct, (1, 1, 1))], label=ct,
                       alpha=0.9, linewidths=0)
        legend_types = sorted(df_fov["cell type"].dropna().astype(str).unique())

    handles = [plt.Line2D([0], [0], marker="o", color="w", markeredgecolor="none",
                          markerfacecolor=colors.get(ct, "black"),
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
    if show_legend:
        ax.legend(handles=handles, title="Cell type", bbox_to_anchor=(1.02, 1),
                  loc="upper left", fontsize=9, title_fontsize=10, frameon=True)

    if own_fig:
        plt.tight_layout()
        _finish(fig, save)


def plot_counted_cells(ax, fov_id, df_cells, df_fovs, counted, cell_size=None):
    """One field drawn in grey, with only the cells a rule was counted on in color.

    `counted` is what `rule_metrics.counted_cells` returns. Consequents first, then
    the other antecedents, then the centers on top, so the cell a patch is built
    around is never hidden by one of its neighbors.
    """
    grey = dict.fromkeys(df_cells["cell type"].dropna().unique(), _COUNTED_GREY)
    plot_fov(fov_id, "", df_cells, df_fovs, ax=ax, show_legend=False,
             cell_size=cell_size, colors=grey)
    block = df_cells[df_cells["fov"] == fov_id]
    for positions, role in ((counted.consequent, "consequent"),
                            (counted.antecedent, "antecedent"),
                            (counted.centers, "center")):
        here = block.iloc[list(positions)]
        ax.scatter(here["x_um"], here["y_um"], s=cell_size, c=ROLE_COLORS[role],
                   linewidths=0)


def role_key(parts):
    """One (color, label) per part the figure's rules have cells for.

    `parts` : one `rule_metrics.Parts` per rule drawn. A part is named by its cell
    types, because which antecedent is the center cannot otherwise be read off the
    picture; where two rules disagree on it, the colour is left to speak alone.
    """
    entries = []
    for role, color in ROLE_COLORS.items():
        seen = {getattr(part, role) for part in parts if getattr(part, role)}
        if not seen:
            continue
        types = seen.pop() if len(seen) == 1 else ()
        named = ", ".join(one.replace("_", " ") for one in types)
        entries.append((color, f"{role} ({named})" if named else role))
    return entries


def _complex_plot(module, name, args, kwargs):
    """Keep old notebooks working while complex layouts live with their investigation."""
    import importlib
    plot = getattr(importlib.import_module(module), name)
    return plot(*args, **kwargs)


def plot_rule_fov_pairs(*args, **kwargs):
    return _complex_plot("complex_vis", "plot_rule_fov_pairs", args, kwargs)


def plot_stage_rule_fovs(*args, **kwargs):
    return _complex_plot("complex_stages_vis", "plot_stage_rule_fovs", args, kwargs)


def plot_group_profiles(values, metadata, rules, groups, group_col, unit_col,
                        result=None, scope=None, ylabel="value", save=None):
    """One unit-level profile figure per rule across ordered groups."""
    rules = [rule for rule in rules if rule in values.index]
    if not rules:
        print("No requested rules are available.")
        return []

    rng = np.random.default_rng(7)
    figures = []
    for rule in rules:
        fig, ax = plt.subplots(figsize=(6.4, 4.1))
        means = []
        for position, group in enumerate(groups):
            units = metadata.loc[metadata[group_col] == group, unit_col].drop_duplicates()
            observed = values.reindex(columns=units).loc[rule].dropna()
            jitter = rng.uniform(-0.12, 0.12, len(observed))
            color = _STAGE_PALETTE.get(group, "#777777")
            ax.scatter(position + jitter, observed, s=32, color=color, alpha=0.68,
                       edgecolor="white", linewidth=0.6)
            mean = observed.mean() if len(observed) else np.nan
            means.append(mean)
            if len(observed):
                ax.scatter(position, mean, s=86, marker="D", color=color,
                           edgecolor="white", linewidth=1.1, zorder=5)
            ax.text(position, -0.12, f"eligible n={len(observed)}", ha="center",
                    va="top", fontsize=8, color="#66645F",
                    transform=ax.get_xaxis_transform())
        ax.plot(range(len(groups)), means, color="#4A4844", lw=1.3, alpha=0.7)
        ax.set_xticks(range(len(groups)), groups)
        ax.set_ylim(-0.04, 1.04)
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda value, _: f"{value:.0%}"))
        ax.set_ylabel(ylabel)
        evidence = None
        if result is not None and rule in result.index and "fdr" in result:
            evidence = f"Patient-level FDR = {result.at[rule, 'fdr']:.3g}"
        tidy_axes(ax, grid="y", hide=("top", "right"))
        ax.tick_params(length=0)
        figure_titles(
            fig, str(rule).replace(" -> ", " → "), organ=scope,
            subtitle="Patient-level rule profile · informative complex rule",
            params=evidence,
        )
        fig.subplots_adjust(bottom=0.17)
        _finish(fig, save)
        figures.append(fig)
    return figures


def plot_fov_panel(target_fovs, df_cells, df_fovs, num_cols=1, save=None,
                   title_groups=None):
    """One map per FOV, side by side, lettered A, B, C … for the write-up.

    `target_fovs` is the {FOV: description} dict from
    `get_representative_fovs_for_pc`, so the panel reads left-to-right along
    the component.
    `title_groups` : {FOV: group name}. Each group gets a row of its own and one title
    color, so the maps that belong together read as one at a glance. `num_cols` is then
    the longest group, and a shorter row leaves its places empty.
    """
    if not target_fovs:
        print("No FOVs to plot.")
        return
    fovs = list(target_fovs)
    if title_groups:
        grouped = {}
        for fov in fovs:
            grouped.setdefault(title_groups[fov], []).append(fov)
        rows = list(grouped.values())
    else:
        num_cols = max(1, min(num_cols, len(fovs)))
        rows = [fovs[i:i + num_cols] for i in range(0, len(fovs), num_cols)]
    num_cols = max(len(row) for row in rows)

    fig, axes = plt.subplots(len(rows), num_cols,
                             figsize=(10 * num_cols, 10 * len(rows)),
                             facecolor="#ffffff", squeeze=False)

    group_colors = corner_colors(list(title_groups.values())) if title_groups else {}
    letter = 0
    for row_axes, row_fovs in zip(axes, rows):
        for ax, fov in zip(row_axes, row_fovs):
            plot_fov(fov, target_fovs[fov], df_cells, df_fovs, ax=ax)
            color = group_colors.get((title_groups or {}).get(fov), "black")
            ax.title.set_color(color)
            ax.text(-0.07, 1.06, chr(65 + letter), transform=ax.transAxes, color=color,
                    fontsize=20, fontweight="bold", va="bottom", ha="right")
            letter += 1
        for ax in row_axes[len(row_fovs):]:
            ax.set_visible(False)

    plt.tight_layout()
    _finish(fig, save)
