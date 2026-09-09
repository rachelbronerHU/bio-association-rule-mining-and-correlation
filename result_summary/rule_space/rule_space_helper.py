"""FOVs as vectors in rule space: build the matrix, run the PCA, draw what reads it.

Three pieces, so a notebook stays a list of questions:

    Settings - module-level, set from a cell in each notebook
    Data     - everything loaded from disk, loaded once
    Scope    - the result of one PCA: its matrix, coordinates and variance

A notebook then reads:

    rs.POSITIVE_ONLY = True
    data = rs.load(RESULT_CSV_PATH)
    colon = rs.run(data, stages=['Control', 'Severe'], organs=['Colon'],
                   prefix='cs_colon', save=True)
    rs.fov_panel(data, colon, save=True)
    rs.pair(data, colon, save=True)

Nothing here draws: the PCA figures live in rule_space_vis, the shared ones
in vis_helper.
"""
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_samples
from sklearn.preprocessing import StandardScaler

import data_helper as dh
import vis_helper as vh
import rule_space_vis as rsv
from data_helper import base_items, clean_items

REGION_COLS = ['in_CryptVilli', 'in_BrunnerGland', 'in_SMV', 'in_Muscle',
               'in_LP', 'in_Submucosa', 'in_Follicle', 'in_Lumen']


# ---------------------------------------------------------------------------
# 1. The settings, the data, and one run's result
# ---------------------------------------------------------------------------

# --- Settings -------------------------------------------------------------
# Every notebook sets these in its own cell, so a run's whole setup reads in one place.
METRIC            = 'Lift'      # what a cell of the matrix holds
POSITIVE_ONLY     = True        # keep only lift > 1 (the cell types attract)
RULE_MAX_ITEMS    = 2           # 2 = pairwise
NO_SELF           = True        # drop rules where a cell type repeats
MIN_FOV_THRESHOLD = 0.02        # a rule must be seen in more than this share of FOVs
WITH_SCALE        = True        # log-shift + standardise before the PCA
RULES_TO_EXCLUDE  = ()          # drop every rule naming one of these cell types
SCORE_COL         = 'Pathological score'
TOP_LOADINGS      = 8           # how many rules to name at each end of a component
N_COLORED         = 2           # how many top cell types to colour a PCA by

# The floor the mining used: a rule was only kept where a cell type reached this many
# cells and this share of the FOV. Set it to match the run these rules came from - only
# `presence_matrices` reads it, to ask which rules could have been found at all.
MIN_PATCHES       = 15          # cells of a type an FOV needs
MIN_SUPPORT       = 0.01        # and that share of the FOV's cells

# The biopsy metadata worth colouring a PCA by, for `run(..., colors=...)`.
METADATA_COLORS   = tuple(dh.METADATA_COLS)

# Comparing one set of settings against another: turn this on and every PCA plot is
# written, under a name carrying the settings, into its own folder. Nothing is
# overwritten between runs, so a notebook can be re-run per setting and the results
# lined up afterwards. Everything that is not a PCA - the scree, the loadings, the
# tissue maps, the spread bars - is left unwritten, so a sweep never touches the
# write-up's folder. Off, only the figures a call asks for are written, to
# `summary_downloads`, under their plain names - which is what the write-up reads.
SAVE_ALL          = False
SAVE_ALL_DIR      = 'param_runs'

_DEFAULT_FIGURE_DIR = vh.FIGURE_DIR


def _number(value):
    """A number as a piece of a filename: no dot, which would read as a file type."""
    return f"{value:g}".replace('.', 'p').replace('-', 'm')


def _param_tag(min_fov=None):
    """The settings that change what a figure shows, as one filename-safe string."""
    parts = [str(METRIC).lower(),
             f"items{RULE_MAX_ITEMS}",
             'pos' if POSITIVE_ONLY else 'posneg',
             'noself' if NO_SELF else 'withself',
             f"fov{_number(MIN_FOV_THRESHOLD if min_fov is None else min_fov)}",
             f"patch{_number(MIN_PATCHES)}",
             f"sup{_number(MIN_SUPPORT)}"]
    if not WITH_SCALE:
        parts.append('raw')
    if RULES_TO_EXCLUDE:
        parts.append('no-' + '-'.join(sorted(RULES_TO_EXCLUDE)))
    return "_".join(parts)


def _slug(text):
    """A column name as a piece of a filename."""
    return str(text).replace(' ', '_').lower()


def figure_name(name, save=True, pca=False, colored_by=None, min_fov=None):
    """The name a figure is written under, or None when it is not written.

    Every figure in this module goes through here, so the two modes are decided in one
    place. It also points the shared saver at the right folder, since that folder is
    part of the same decision.

    `pca` marks the figures a sweep is about - the ones showing where the FOVs fall.
    Under SAVE_ALL only those are written, so a sweep leaves `summary_downloads` alone.

    A swept name splits the work in two: the tag carries the settings that built the
    matrix, and the name carries what the figure shows. So the sign of the rules is
    dropped from the front of a prefix - the tag already states it, and a prefix saying
    'posneg' would go stale the moment POSITIVE_ONLY is swept - while `colored_by` is
    added for the figures whose colouring is not already in their name.
    """
    if not name:
        return None
    if SAVE_ALL:
        if not pca:
            return None
        vh.FIGURE_DIR = SAVE_ALL_DIR
        for sign in ('posneg_', 'pos_'):
            if name.startswith(sign):
                name = name[len(sign):]
                break
        if colored_by:
            name = f"{name}_{_slug(colored_by)}"
        return f"{name}__{_param_tag(min_fov)}"
    vh.FIGURE_DIR = _DEFAULT_FIGURE_DIR
    return name if save else None


@dataclass
class Data:
    """Everything read from disk, plus the two tables derived from it."""
    results: pd.DataFrame             # one row per rule per FOV
    fovs: pd.DataFrame                # one row per FOV, with its metadata
    cells: pd.DataFrame               # one row per cell
    labels: pd.DataFrame              # one row per FOV: everything we can colour by
    fractions: pd.DataFrame           # FOV x cell type, each row summing to 1


@dataclass
class Scope:
    """One PCA: what it was run on, and what came out."""
    prefix: str                       # figures are saved as <prefix>_<what>
    stages: tuple | None
    organs: tuple | None
    matrix: pd.DataFrame              # FOVs x rules
    coords: pd.DataFrame              # FOVs x PCs, with the metadata joined on
    variance: np.ndarray              # % explained, per component
    model: PCA = field(repr=False)
    built_from: str = 'rules'         # or 'cell counts', or 'rules without X'
    min_fov: float = MIN_FOV_THRESHOLD  # the share this one was built with

    @property
    def label(self):
        """'Colon, Control vs Severe' - what a figure title says it is looking at."""
        organs = ", ".join(self.organs) if self.organs else "both organs"
        stages = " vs ".join(self.stages) if self.stages else "all stages"
        what = "" if self.built_from == 'rules' else f", {self.built_from}"
        return f"{organs}, {stages}{what}"


def load(result_csv_path=None, load_max_items=4):
    """Read the cells, FOVs and rules once, and derive the two lookup tables.

    Rules are loaded wide and unfiltered; the settings above narrow them per run, so
    changing one never means reloading. The run comes from `data_helper.RESULT_CSV_PATH`
    unless this notebook wants a different one.
    """
    result_csv_path = result_csv_path or dh.RESULT_CSV_PATH
    cells, fovs, _ = dh.load_spatial_data()
    results = dh.load_results(result_csv_path, rule_max_items=load_max_items,
                              kind=None)
    vh.set_cell_colors(cells)
    return Data(results=results, fovs=fovs, cells=cells,
                labels=_fov_labels(cells, fovs),
                fractions=dh.fov_fractions(cells))


def _fov_labels(cells, fovs):
    """One row per FOV holding everything we might colour a PCA by.

    'Region' is the tissue region most of that FOV's cells sit in.
    """
    cols = (['FOV', 'Organ', 'Biopsy', 'PatientID', 'Pathological score', 'Clinical score']
            + list(dh.METADATA_COLS))
    labels = fovs[[c for c in cols if c in fovs.columns]].drop_duplicates('FOV').copy()

    present = [c for c in REGION_COLS if c in cells.columns]
    if present:
        share = cells.groupby('fov')[present].mean()
        share.columns = [c.replace('in_', '') for c in present]
        labels = labels.merge(share.idxmax(axis=1).rename('Region'),
                              left_on='FOV', right_index=True, how='left')
    labels['Region'] = labels.get('Region', 'Unknown').fillna('Unknown')
    return labels


# ---------------------------------------------------------------------------
# 2. From rules to a PCA
# ---------------------------------------------------------------------------

def _filter(data, stages, organs, exclude=(), fovs=None):
    """The FOVs this run is about, and the rules allowed to be columns.

    `fovs` names FOVs by hand, on top of the stage and organ cuts rather than instead
    of them, so a subset still knows where it came from.
    """
    chosen = data.fovs
    if stages is not None:
        chosen = chosen[chosen[SCORE_COL].isin(stages)]
    if organs is not None:
        chosen = chosen[chosen['Organ'].isin(organs)]
    if fovs is not None:
        chosen = chosen[chosen['FOV'].isin(fovs)]

    rules = data.results[data.results['FOV'].isin(chosen['FOV'])].copy()
    items = rules['Antecedents'].apply(base_items) + rules['Consequents'].apply(base_items)

    keep = items.apply(len) <= RULE_MAX_ITEMS
    if NO_SELF:
        keep &= items.apply(lambda b: len(b) == len(set(b)))
    if POSITIVE_ONLY:
        keep &= rules['Lift'] > 1
    if RULES_TO_EXCLUDE or exclude:
        drop = set(RULES_TO_EXCLUDE) | set(exclude)
        keep &= items.apply(lambda b: not (set(b) & drop))

    rules = rules[keep].copy()
    rules['Clean_Rule'] = (rules['Antecedents'].apply(clean_items) + ' -> '
                           + rules['Consequents'].apply(clean_items))
    return rules


def _matrix(rules, min_fov):
    """FOVs x rules, each cell that rule's strength in that FOV."""
    n_fovs = rules['FOV'].nunique()
    counts = rules.groupby('Clean_Rule')['FOV'].nunique()
    kept = counts[counts > n_fovs * min_fov].index
    rules = rules[rules['Clean_Rule'].isin(kept)]

    mat = (rules.drop_duplicates(['FOV', 'Clean_Rule'])
                .pivot(index='FOV', columns='Clean_Rule', values=METRIC))

    # A rule that never fired means 'no association', not missing data.
    fill = 1.0 if METRIC in ('Lift', 'Conviction') else 0.0
    mat = mat.fillna(fill)

    # Conviction can be infinite; park it just above the largest real value.
    mat = mat.replace([np.inf, -np.inf], np.nan)
    top = mat.max().max()
    mat = mat.fillna(top * 1.1 if pd.notna(top) else fill)
    if not WITH_SCALE:
        return mat

    # Move the neutral point to zero, then give every rule the same weight.
    shifted = np.log2(mat + 1e-9) if METRIC in ('Lift', 'Conviction') else np.log1p(mat)
    return pd.DataFrame(StandardScaler().fit_transform(shifted),
                        index=mat.index, columns=mat.columns)


def _settings(scope, color_by=None):
    """The two small grey lines printed under a figure.

    Top line: what the figure shows. Second line: the settings that built it, so a plot
    on its own says which run it came from.
    """
    params = " | ".join([f"metric: {METRIC}",
                         f"positive only: {'yes' if POSITIVE_ONLY else 'no'}",
                         f"self-rules: {'no' if NO_SELF else 'yes'}",
                         f"max cell types: {RULE_MAX_ITEMS}",
                         f"rule in >{scope.min_fov:.0%} of FOVs",
                         scope.label])
    if not color_by:
        return params

    per_group = separation_by_group(scope, color_by)
    if per_group is None:
        return f"color: {color_by}\n{params}"
    detail = ", ".join(f"{name} {value:+.2f}" for name, value in per_group.items())
    return (f"color: {color_by}  |  separation {per_group.mean():+.2f}  ({detail})"
            f"\n{params}")


def run(data, stages=None, organs=None, prefix=None, colors=(), save=False,
        exclude=(), fovs=None, diagnostics=True, min_fov=None):
    """Filter, build the matrix, run the PCA, and draw how much each component carries.

    `colors` names extra colourings to draw as scatters (organ, a metadata column...).
    The stage-coloured scatter comes from `pair()` instead, because that one also
    carries the boxed pair. `exclude` drops every rule naming those cell types.
    `fovs` keeps those FOVs only, rebuilding the rules and the matrix from them - a
    new PCA, not the old one re-drawn. `diagnostics` draws the scree and the loadings bars - turn it off when you only
    want the PCA back to lay out yourself.

    `min_fov` is how often a rule must fire to be kept, for this run only; it defaults
    to MIN_FOV_THRESHOLD. The scope remembers it, so a figure states the share it was
    actually built with rather than whatever the setting holds when it is drawn.
    """
    min_fov = MIN_FOV_THRESHOLD if min_fov is None else min_fov
    mat = _matrix(_filter(data, stages, organs, exclude, fovs), min_fov)
    return _fit(mat, data, stages, organs, prefix, colors, save,
                diagnostics=diagnostics, min_fov=min_fov)


def _fit(mat, data, stages, organs, prefix, colors, save, built_from='rules',
         diagnostics=True, min_fov=None):
    """PCA on a ready matrix, then the scree, the colourings and the loadings."""
    n = min(10, *mat.shape)
    model = PCA(n_components=n).fit(mat)

    coords = pd.DataFrame(model.transform(mat), index=mat.index,
                          columns=[f'PC{i + 1}' for i in range(n)]).reset_index()
    coords = coords.merge(data.labels, on='FOV', how='left')
    for c in coords.columns:                       # never let a colour be blank
        if coords[c].dtype == object:
            coords[c] = coords[c].fillna('Unknown')

    scope = Scope(prefix=prefix, stages=stages, organs=organs, matrix=mat,
                  coords=coords, variance=model.explained_variance_ratio_ * 100,
                  model=model, built_from=built_from,
                  min_fov=MIN_FOV_THRESHOLD if min_fov is None else min_fov)
    print(f"{scope.label}: {mat.shape[0]} FOVs x {mat.shape[1]} columns  |  "
          f"PC1 {scope.variance[0]:.1f}%  PC2 {scope.variance[1]:.1f}%")

    name = ((lambda what, pca=False: figure_name(f"{prefix}_{what}", save, pca))
            if prefix else (lambda what, pca=False: None))
    for col in colors:
        rsv.plot_pca_scatter(coords, scope.variance, color_by=col, scope=scope.label,
                            subtitle=_settings(scope, col),
                            save=name(f"pca_{col.replace(' ', '_').lower()}", pca=True))
    if diagnostics:
        rsv.plot_pca_scree(scope.variance, subtitle=_settings(scope),
                          scope=scope.label, save=name('scree'))
        for i in (0, 1):
            rsv.plot_pca_loadings(model.components_[i], mat.columns, component_idx=i,
                                 subtitle=_settings(scope), top_n=TOP_LOADINGS,
                                 scope=scope.label, save=name(f'loadings_pc{i + 1}'))
    return scope


# ---------------------------------------------------------------------------
# 3. Reading one PCA
# ---------------------------------------------------------------------------

def separation_by_group(scope, label=None, components=('PC1', 'PC2')):
    """How well each group on its own is picked out in this PCA.

    Every FOV is compared with the FOVs of its own group and with those of the nearest
    other group. 0 means that group is fully mixed into the others, 1 means it sits
    apart with clear space around it, and below 0 means its FOVs are typically nearer
    another group than their own.

    Groups are worth reading one by one: one tight group among scattered ones scores
    high while the scattered ones score low, and that is a different picture from two
    groups genuinely sitting apart.

    None when there is nothing to measure: a label this scope does not carry, or one
    that leaves fewer than two groups here.
    """
    coords = scope.coords
    if label is None:
        label = SCORE_COL
    if label not in coords.columns:
        return None
    groups = coords[label].fillna('Unknown').astype(str)
    if not 2 <= groups.nunique() <= len(coords) - 1:
        return None
    per_fov = silhouette_samples(coords[list(components)].to_numpy(dtype=float), groups)
    return pd.Series(per_fov, index=groups.to_numpy()).groupby(level=0).mean()


def separation(scope, label=None, components=('PC1', 'PC2')):
    """How far apart the groups of `label` sit in this PCA, as one number.

    Each group is averaged on its own first, and the groups are then averaged evenly.
    Averaging over the FOVs instead would let the biggest group decide the answer: with
    103 Mild FOVs against 36 Severe, a dense Mild blob scores high on its own and carries
    the total, even where the two groups are not apart at all.

    It is here to compare one PCA with another - the rules against the cell counts,
    say - which two pictures side by side cannot do reliably.
    """
    per_group = separation_by_group(scope, label, components)
    return np.nan if per_group is None else float(per_group.mean())


def panel(rows, color_by=None, title=None, save=None, what='panel', subtitle=None):
    """Several PCAs in one figure. Each list is one row; a short row leaves blanks.

    Every tile is titled by what it was built from and by how far apart the groups
    sit in it, which is the whole point of holding them side by side.

    `what` names the figure when SAVE_ALL is on and no `save` name was given. Two
    unnamed panels over the same scope would otherwise land on one file, so give the
    second one its own `what`.
    """
    color_by = color_by or SCORE_COL
    tiles = [[(scope.coords, scope.variance, _tile_label(scope, color_by))
              for scope in row] for row in rows]
    name = figure_name(save or f"{rows[0][0].prefix}_{what}", save is not None,
                       pca=True, colored_by=color_by)
    rsv.plot_pca_panel(tiles, color_by, title or rows[0][0].label,
                       subtitle=subtitle or _settings(rows[0][0]), save=name)


def _tile_label(scope, color_by):
    """What one tile of a panel was built from, and how far apart its groups sit."""
    score = separation(scope, color_by)
    if not np.isfinite(score):
        return scope.built_from
    return f"{scope.built_from} | separation {score:+.2f}"


def loadings(scope, save=False):
    """The rules pulling hardest on PC1 and on PC2, one figure each."""
    for i in (0, 1):
        rsv.plot_pca_loadings(scope.model.components_[i], scope.matrix.columns,
                              component_idx=i, subtitle=_settings(scope),
                              top_n=TOP_LOADINGS, scope=scope.label,
                              save=figure_name(f"{scope.prefix}_loadings_pc{i + 1}", save, min_fov=scope.min_fov))


def spread_along(scope, component='PC1', num=5):
    """`num` FOVs spread along one component, lowest to highest.

    Spread over the component's range, not its quantiles: the scores are skewed, so
    quantiles would put most of them inside the crowded middle. Where a position has no
    FOV near it the nearest unused one is taken, and each title prints that FOV's real
    score, so a gap shows rather than hides.
    """
    coords = scope.coords
    if coords.empty:
        return {}
    names = ['low', 'mid-low', 'middle', 'mid-high', 'high']
    lo, hi = coords[component].min(), coords[component].max()

    picked = {}
    for k, frac in enumerate(np.linspace(0, 1, num)):
        target = lo + frac * (hi - lo)
        order = (coords[component] - target).abs().sort_values().index
        i = next((i for i in order if coords.at[i, 'FOV'] not in picked), None)
        if i is None:
            break
        row = coords.loc[i]
        label = names[k] if num == len(names) else f'{frac:.0%}'
        picked[row['FOV']] = (f"{component} {label} | {row[component]:.2f} | "
                              f"{row.get('Organ', '?')} ({row.get(SCORE_COL, '?')})")
    return picked


def outliers(scope, k=3, components=('PC1', 'PC2')):
    """Returns (the FOVs left, the FOVs sitting far out). Draws nothing.

    Each FOV's distance from the middle, over the median distance - so 1 is an
    ordinary FOV and `k` is how far out one must sit to be an outlier: 2 trims hard,
    5 barely trims. The middle is the median, which the far FOVs cannot drag towards
    themselves. Pass the kept FOVs to `run(..., fovs=kept)` to fit the PCA again.
    """
    coords = scope.coords
    pts = coords[list(components)].to_numpy(dtype=float)
    far = np.linalg.norm(pts - np.median(pts, axis=0), axis=1)
    usual = np.median(far)
    ratio = far / usual if usual else np.zeros(len(far))

    out = coords.loc[ratio > k, 'FOV'].tolist()
    print(f"{scope.label}: {len(out)} of {len(coords)} FOVs sit more than {k}x the "
          f"usual {usual:.2f} out from the middle")
    for fov, times in sorted(zip(out, ratio[ratio > k]), key=lambda both: -both[1]):
        print(f"  {fov:26s} {times:.1f}x")
    return coords.loc[ratio <= k, 'FOV'].tolist(), out


def closest_pair(scope):
    """Two FOVs sitting together in the plot, with empty space around them.

    Skips FOVs where no rule fired (they all land on the same point) and pairs from one
    patient (two FOVs of a biopsy look alike whatever the method does). Of the rest,
    takes the pair with the most space around them next to their own gap.
    """
    from scipy.spatial.distance import pdist, squareform

    coords, mat = scope.coords, scope.matrix
    fovs = list(coords['FOV'])
    patient = coords['PatientID'].to_numpy()
    active = (mat.reindex(fovs).nunique(axis=1) > 1).to_numpy()

    D = squareform(pdist(coords[['PC1', 'PC2']].to_numpy(dtype=float)))
    np.fill_diagonal(D, np.inf)
    space = D.copy()                       # untouched, for measuring the empty ring

    D[~active, :] = np.inf                 # no rule fired
    D[:, ~active] = np.inf
    for i in range(len(fovs)):             # same patient is no evidence
        D[i, patient == patient[i]] = np.inf
    span = coords['PC1'].max() - coords['PC1'].min()
    D[(D < 0.015 * span) | (D > 0.035 * span)] = np.inf   # touching, but still two dots

    best = None
    for i in range(len(fovs)):
        for j in range(i + 1, len(fovs)):
            if not np.isfinite(D[i, j]):
                continue
            margin = np.delete(np.minimum(space[i], space[j]), [i, j]).min()
            if best is None or margin / D[i, j] > best[0]:
                best = (margin / D[i, j], fovs[i], fovs[j], D[i, j], margin)

    if best is None:
        print('no pair found')
        return []
    _, a, b, gap, margin = best
    print(f"  pair: {a} + {b}  ({gap:.2f} apart, nothing else within {margin:.2f})")
    return [a, b]


def fov_panel(data, scope, component='PC1', num=5, save=False):
    """The FOVs along one component, drawn side by side as maps, lettered A-E."""
    picked = spread_along(scope, component=component, num=num)
    for fov, desc in picked.items():
        print(f"  {fov:26s} {desc}")
    vh.plot_fov_panel(picked, data.cells, data.fovs, num_cols=len(picked),
                      save=figure_name(f"{scope.prefix}_fovs_{component.lower()}", save, min_fov=scope.min_fov))
    return picked


_DIRECTIONS = ['right', 'down-right', 'down', 'down-left',
               'left', 'up-left', 'up', 'up-right']


def _compass(point):
    """The word for the direction a point lies in, seen from the middle."""
    turn = np.degrees(np.arctan2(point[1], point[0])) % 360
    return _DIRECTIONS[int(round((360 - turn) / 45)) % len(_DIRECTIONS)]


def archetype_fovs(scope, num=8, per_corner=1, gap=0.2):
    """The FOVs at the edges of the cloud: {FOV: (its description, its corner)}.

    Directions, not the corners of the plot: a cloud shaped like a triangle has three
    tips and no fourth corner, and this finds the tips it has. PC1 and PC2 are put on
    one scale first, so the wider one does not win every direction.

    `num` directions are swept clockwise from the right, so the maps come out in the
    order they sit round the plot, and the FOV furthest out in each is taken. Winners
    within `gap` of one another - a share of the cloud's width - are one corner, named
    for where its furthest-out FOV lies. `per_corner` then takes that many of the
    corner's FOVs, furthest out first.
    """
    coords = scope.coords
    if coords.empty:
        return {}
    pts = coords[['PC1', 'PC2']].to_numpy(dtype=float)
    span = np.ptp(pts, axis=0)
    pts = (pts - np.median(pts, axis=0)) / np.where(span > 0, span, 1)
    out = np.linalg.norm(pts, axis=1)                  # how far out from the middle

    corners = []                                       # each: the FOV indexes found
    for angle in np.linspace(0, -2 * np.pi, num, endpoint=False):
        i = int(np.argmax(pts @ np.array([np.cos(angle), np.sin(angle)])))
        # Every corner this winner is close to becomes one corner, so three FOVs strung
        # along one edge are not read as two corners with a gap between them.
        joined = [c for c in corners
                  if min(np.linalg.norm(pts[i] - pts[j]) for j in c) <= gap]
        if not joined:
            corners.append([i])
            continue
        joined[0][:] = list(dict.fromkeys(sum(joined, [i])))
        for other in joined[1:]:
            corners.remove(other)

    picked = {}
    for seeds in corners:
        near = [j for j in range(len(pts))
                if min(np.linalg.norm(pts[j] - pts[i]) for i in seeds) <= gap]
        share = np.array(near)[np.argsort(-out[near])][:per_corner]
        way = _compass(pts[share[0]])                  # the tip says which corner it is
        for n, j in enumerate(share, start=1):
            row = coords.iloc[j]
            picked.setdefault(row['FOV'],
                              (f"{way} corner, {n} of {len(share)} | "
                               f"PC1 {row['PC1']:.1f}, PC2 {row['PC2']:.1f} | "
                               f"{row.get('Organ', '?')} ({row.get(SCORE_COL, '?')})", way))
    return picked


def archetype_panel(data, scope, num=8, per_corner=1, gap=0.2, save=False):
    """Where the edge FOVs sit, and then those FOVs as maps - four to a row.

    The scatter names them, and the maps of one corner share a title color.
    """
    picked = archetype_fovs(scope, num=num, per_corner=per_corner, gap=gap)
    titles = {fov: desc for fov, (desc, _) in picked.items()}
    corners = {fov: way for fov, (_, way) in picked.items()}
    for fov, desc in titles.items():
        print(f"  {fov:26s} {desc}")

    # The maps colour each corner's title from this same mapping, so a corner keeps one
    # colour across both figures. The palette avoids the stage colours on purpose.
    by_corner = vh.corner_colors(list(corners.values()))
    rsv.plot_pca_scatter(scope.coords, scope.variance, color_by=SCORE_COL,
                        label_fovs=titles, scope=scope.label,
                        label_colors={fov: by_corner[way]
                                      for fov, way in corners.items()},
                        subtitle=_settings(scope, SCORE_COL),
                        save=figure_name(f"{scope.prefix}_archetypes_pca", save,
                                         pca=True, colored_by=SCORE_COL, min_fov=scope.min_fov))
    vh.plot_fov_panel(titles, data.cells, data.fovs, num_cols=per_corner,
                      title_groups=corners,
                      save=figure_name(f"{scope.prefix}_archetypes", save, min_fov=scope.min_fov))
    return picked


def pair(data, scope, color_by=None, save=False):
    """The scope's scatter - the panel FOVs named, the closest pair squared - and then
    that pair drawn as maps.

    `color_by` is the stage unless given; the named FOVs and the boxed pair do not
    depend on it, so only the colours change. Each colouring saves under its own name.
    """
    color_by = color_by or SCORE_COL
    two = closest_pair(scope)
    rsv.plot_pca_scatter(scope.coords, scope.variance, color_by=color_by,
                        label_fovs=spread_along(scope), box_fovs=two,
                        scope=scope.label, subtitle=_settings(scope, color_by),
                        save=figure_name(
                            f"{scope.prefix}_pca_{color_by.replace(' ', '_').lower()}",
                            save, pca=True))
    vh.plot_fov_panel({f: str(data.fovs.set_index('FOV').loc[f, SCORE_COL]) for f in two},
                      data.cells, data.fovs, num_cols=2,
                      save=figure_name(f"{scope.prefix}_pair_fovs", save, min_fov=scope.min_fov))
    return two


def group_spread(scope, label=None, min_fovs=2, save=False):
    """How tightly each group's own FOVs sit together, next to any two FOVs.

    Below 1 means that group's FOVs are more alike than FOVs in general here. The
    atlas describes the healthy gut as stereotypical across individuals and disease
    as varied, so Control is the group expected to sit tightest.

    The same measure as `patient_spread`, one level up: that one asks whether a
    patient's FOVs sit together, this one whether a stage's do.
    """
    from scipy.spatial.distance import pdist

    label = label or SCORE_COL
    coords = scope.coords
    baseline = pdist(coords[['PC1', 'PC2']].to_numpy(dtype=float)).mean()
    rows = []
    for name, group in coords.groupby(label):
        points = group[['PC1', 'PC2']].to_numpy(dtype=float)
        if len(points) < min_fovs:
            continue
        rows.append({label: name, 'n_FOV': len(points),
                     'spread': pdist(points).mean() / baseline})
    table = pd.DataFrame(rows).set_index(label).sort_values('spread')
    tightest = table.index[0] if len(table) else None
    print(f"  {scope.label}: tightest is {tightest} at {table['spread'].iloc[0]:.2f}"
          if tightest is not None else "  nothing to measure")
    rsv.plot_group_spread(table, label, scope=scope.label,
                          subtitle=_settings(scope),
                          save=figure_name(f"{scope.prefix}_group_spread", save, min_fov=scope.min_fov))
    return table


def patient_spread(scope, min_fovs=2, save=False):
    """How far apart each patient's own FOVs sit, next to any two FOVs.

    Below 1 means that patient's FOVs sit closer together than FOVs in general.
    Distances between pairs are used so the value does not drift with how many FOVs a
    patient has. Patients with fewer than `min_fovs` are skipped - one FOV has no spread.
    """
    from scipy.spatial.distance import pdist

    coords = scope.coords
    baseline = pdist(coords[['PC1', 'PC2']].to_numpy(dtype=float)).mean()
    rows = []
    for patient, grp in coords.groupby('PatientID'):
        pts = grp[['PC1', 'PC2']].to_numpy(dtype=float)
        if len(pts) < min_fovs:
            continue
        rows.append({'PatientID': patient, 'n_FOV': len(pts),
                     'spread': pdist(pts).mean() / baseline})
    table = pd.DataFrame(rows).set_index('PatientID').sort_values('spread')
    print(f"  {len(table)} patients with {min_fovs}+ FOVs | "
          f"{(table['spread'] < 1).sum()} sit closer together than average | "
          f"median {table['spread'].median():.2f}")
    rsv.plot_patient_spread(table, save=figure_name(f"{scope.prefix}_patient_spread", save, min_fov=scope.min_fov))
    return table


# ---------------------------------------------------------------------------
# 4. Is a component only following how common a cell type is?
# ---------------------------------------------------------------------------

def abundance_correlation(scope, fractions, n_pcs=2):
    """Spearman rho between each component and each cell type's share, across FOVs.

    Returns (rho, adjusted p), both cell types x components. p-values are FDR-corrected
    over the whole table, since this is one test per cell type per component.
    """
    from scipy.stats import spearmanr
    from statsmodels.stats.multitest import multipletests

    coords = scope.coords
    pcs = [f'PC{i + 1}' for i in range(n_pcs) if f'PC{i + 1}' in coords.columns]
    frac = fractions.reindex(coords['FOV']).fillna(0.0)

    rho = pd.DataFrame(index=frac.columns, columns=pcs, dtype=float)
    raw = rho.copy()
    for pc in pcs:
        y = coords[pc].to_numpy(dtype=float)
        for cell in frac.columns:
            x = frac[cell].to_numpy(dtype=float)
            if np.std(x) == 0:                        # absent everywhere: leave NaN
                continue
            rho.loc[cell, pc], raw.loc[cell, pc] = spearmanr(x, y)

    flat = raw.to_numpy(dtype=float).ravel()
    ok = np.isfinite(flat)
    adj = np.full_like(flat, np.nan)
    if ok.any():
        adj[ok] = multipletests(flat[ok], method='fdr_bh')[1]
    return rho, pd.DataFrame(adj.reshape(raw.shape), index=raw.index, columns=raw.columns)


def abundance(data, scope, top_n=8, save=False):
    """How much of a component is just how common a cell type is.

    Draws the correlation over every cell type, then colours this PCA by the few that
    came top *here* - each scope has its own rule set, so each has its own answer.
    """
    rho, padj = abundance_correlation(scope, data.fractions)
    rsv.plot_abundance_correlation_bars(
        rho, n_fovs=len(scope.coords), top_n=top_n, scope=scope.label,
        subtitle=_settings(scope),
        save=figure_name(f"{scope.prefix}_abundance_correlation", save, min_fov=scope.min_fov))

    # The same PCA coloured by the cell types that came top, one figure each.
    top = list(rho.abs().max(axis=1).sort_values(ascending=False).index)[:N_COLORED]
    df = scope.coords.merge(data.fractions[top], left_on='FOV', right_index=True, how='left')
    for i, cell in enumerate(top, start=1):
        rsv.plot_pca_scatter(df, scope.variance, color_by=cell, scope=scope.label,
                            subtitle=_settings(scope, f'share of {cell} cells'),
                            save=figure_name(f"{scope.prefix}_abundance_top{i}", save,
                                             pca=True, min_fov=scope.min_fov))
    return rho, padj


# ---------------------------------------------------------------------------
# 5. Two ways to ask whether the rules add anything
# ---------------------------------------------------------------------------

def linked_cells(data, scope, min_rho=0.5):
    """The cell types whose share follows this PCA's components strongly.

    A cut on the size of the correlation, not on its significance. With a few hundred
    FOVs almost everything is significant - at 281 FOVs |rho| of 0.12 already clears
    p < 0.05 while accounting for 1% of the variation, which would drop 27 of the 32
    cell types. 0.5 means the two move together over a quarter of the way.
    """
    rho, _ = abundance_correlation(scope, data.fractions)
    strongest = rho.abs().max(axis=1).sort_values(ascending=False)
    return list(strongest[strongest >= min_rho].index)


def scatter(scope, color_by=None, save=False):
    """Where the FOVs fall in this PCA - the picture to hold beside another one."""
    color_by = color_by or SCORE_COL
    rsv.plot_pca_scatter(scope.coords, scope.variance, color_by=color_by, scope=scope.label,
                        subtitle=_settings(scope, color_by),
                        save=figure_name(
                            f"{scope.prefix}_pca_{color_by.replace(' ', '_').lower()}",
                            save, pca=True))


def composition_pca(data, scope, exclude=()):
    """The same FOVs, but the matrix is what they are made of instead of what rules they have.

    The baseline for 'do the rules add anything?'. If the groups come apart here just as
    well, the rules are not carrying anything the cell counts do not. Shares are left as
    they are: they are already on one scale, and standardising would give a type seen in
    1% of cells the same say as one seen in 40%. `exclude` drops those cell types'
    columns, which is the cell-count twin of dropping their rules.

    Draws nothing - the notebook lays it out beside the one it is a baseline for.
    """
    mat = (data.fractions.drop(columns=list(exclude), errors='ignore')
           .reindex(scope.coords['FOV']).fillna(0.0))
    tag = f"_no_{'_'.join(c.lower() for c in exclude)}" if exclude else ""
    what = f"cell counts without {', '.join(exclude)}" if exclude else 'cell counts'
    return _fit(mat, data, scope.stages, scope.organs,
                f"{scope.prefix}_composition{tag}", (), save=False,
                built_from=what, diagnostics=False, min_fov=scope.min_fov)


def leave_one_out(data, scope, cells=None):
    """Re-run the PCA once per composition-linked cell type, dropping only that one's rules.

    One question at a time: does this separation lean on Goblet's rules? On Muscle's?
    Dropping them all at once cannot tell them apart, and takes far more FOVs with it.
    A rule goes if the cell type appears anywhere in it, antecedent or consequent.

    Each cell type gives back a pair - the rules without it, and the cell counts without
    it - so a panel row shows the two answers to 'is this cell type what separates them?'
    beside each other.

    Draws nothing - returns one row per cell type, for the notebook to lay out.
    """
    cells = cells if cells is not None else linked_cells(data, scope)
    rows = []
    for cell in cells:
        rules = run(data, stages=scope.stages, organs=scope.organs,
                    prefix=f"{scope.prefix}_no_{cell.lower()}", exclude=[cell],
                    diagnostics=False)
        rules.built_from = f"rules without {cell}"
        rows.append([rules, composition_pca(data, scope, exclude=[cell])])
    return rows


# ---------------------------------------------------------------------------
# 6. Is the separation about which rules exist, or how strong they are?
# ---------------------------------------------------------------------------

def presence_matrices(data, scope):
    """Two more PCAs over the same FOVs and rules: did the rule fire, and could it.

    Most of the lift matrix is the fill value, so the PCA may be reading which rules
    exist rather than how strong they are. These two pull those apart:

        fired     1 where the rule appears, 0 where it does not - the same matrix with
                  the lift values thrown away
        possible  1 where the FOV holds enough of both the rule's cell types to clear
                  the mining's support floor, wherever those cells happen to sit

    So `possible` knows nothing about arrangement. If it separates the groups as well as
    the other two, the separation is composition. `possible` is a necessary condition,
    not a sufficient one: a rule can still fail to fire with the cells all present.
    """
    rules = _filter(data, scope.stages, scope.organs)
    rules = rules[rules['Clean_Rule'].isin(scope.matrix.columns)]
    same = dict(index=scope.matrix.index, columns=scope.matrix.columns)

    fired = (rules.drop_duplicates(['FOV', 'Clean_Rule'])
                  .pivot(index='FOV', columns='Clean_Rule', values=METRIC)
                  .notna().reindex(**same).fillna(False))

    counts = (data.cells.groupby(['fov', 'cell type']).size().unstack(fill_value=0)
              .reindex(scope.matrix.index).fillna(0))
    enough = counts.ge(np.maximum(MIN_PATCHES, MIN_SUPPORT * counts.sum(axis=1)), axis=0)

    named = (rules.drop_duplicates('Clean_Rule').set_index('Clean_Rule')
             .apply(lambda r: set(base_items(r['Antecedents']))
                              | set(base_items(r['Consequents'])), axis=1))
    possible = pd.DataFrame(
        {rule: enough[[c for c in cells if c in enough.columns]].all(axis=1)
         for rule, cells in named.items()}).reindex(**same).fillna(False)

    return (_binary_pca(fired, data, scope, 'fired', 'which rules fired'),
            _binary_pca(possible, data, scope, 'possible', 'which rules could fire'))


def _binary_pca(mat, data, scope, tag, what):
    """PCA on a 0/1 matrix, standardised like the lift matrix so the two compare.

    A rule that fired everywhere or nowhere carries no information and would divide by
    a zero spread, so it is dropped.
    """
    mat = mat.astype(float)
    mat = mat.loc[:, mat.std() > 0]
    scaled = pd.DataFrame(StandardScaler().fit_transform(mat),
                          index=mat.index, columns=mat.columns)
    return _fit(scaled, data, scope.stages, scope.organs, f"{scope.prefix}_{tag}", (),
                save=False, built_from=what, diagnostics=False, min_fov=scope.min_fov)
