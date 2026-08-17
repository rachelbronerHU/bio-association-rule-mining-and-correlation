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
import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

import data_helper as dh
import vis_helper as vh
import rule_space_vis as rsv
from data_helper import base_items, check_rule_overlap, clean_items

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


@dataclass
class Data:
    """Everything read from disk, plus the two tables derived from it."""
    results: pd.DataFrame             # one row per rule per FOV
    fovs: pd.DataFrame                # one row per FOV, with its metadata
    cells: pd.DataFrame               # one row per cell
    labels: pd.DataFrame              # one row per FOV: everything we can colour by
    fractions: pd.DataFrame           # FOV x cell type, each row summing to 1
    support: tuple = (10, 0.01)       # the mining's (absolute, relative) floor


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
                              positive_only=False)
    vh.set_cell_colors(cells)
    return Data(results=results, fovs=fovs, cells=cells,
                labels=_fov_labels(cells, fovs),
                fractions=_fov_fractions(cells),
                support=_support_floor(result_csv_path))


def _support_floor(result_csv_path):
    """How many patches a rule needed, taken from the run that produced these rules.

    A run records its settings under 'settings' or under 'CONFIG', naming the same two
    floors differently.
    """
    cfg = Path(result_csv_path).parent / 'run_config.json'
    raw = json.loads(cfg.read_text()) if cfg.exists() else {}
    c = raw.get('settings') or raw.get('CONFIG') or {}
    # Approximate twice over: the floor backs joint patches but is checked per cell type,
    # and under WEIGHTED it counts weight, not patches.
    return (c.get('min_patches', c.get('MIN_ABS_SUPPORT', 10)),
            c.get('min_support', c.get('MIN_SUPPORT', 0.01)))


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


def _fov_fractions(cells):
    """FOV x cell type, each row summing to 1: what each FOV is made of."""
    comp = cells.groupby(['fov', 'cell type']).size().unstack(fill_value=0)
    return comp.div(comp.sum(axis=1), axis=0)


# ---------------------------------------------------------------------------
# 2. From rules to a PCA
# ---------------------------------------------------------------------------

def _filter(data, stages, organs, exclude=()):
    """The FOVs this run is about, and the rules allowed to be columns."""
    fovs = data.fovs
    if stages is not None:
        fovs = fovs[fovs[SCORE_COL].isin(stages)]
    if organs is not None:
        fovs = fovs[fovs['Organ'].isin(organs)]

    rules = data.results[data.results['FOV'].isin(fovs['FOV'])].copy()
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


def _matrix(rules):
    """FOVs x rules, each cell that rule's strength in that FOV."""
    n_fovs = rules['FOV'].nunique()
    counts = rules.groupby('Clean_Rule')['FOV'].nunique()
    kept = counts[counts > n_fovs * MIN_FOV_THRESHOLD].index
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
    """The small grey line printed under every figure."""
    parts = [f"metric: {METRIC}",
             f"positive only: {'yes' if POSITIVE_ONLY else 'no'}",
             f"no self-rules: {'yes' if NO_SELF else 'no'}",
             f"max cell types: {RULE_MAX_ITEMS}",
             f"rule seen in >{MIN_FOV_THRESHOLD:.0%} of FOVs",
             scope.label]
    return " | ".join(([f"color: {color_by}"] if color_by else []) + parts)


def run(data, stages=None, organs=None, prefix=None, colors=(), save=False,
        exclude=(), diagnostics=True):
    """Filter, build the matrix, run the PCA, and draw how much each component carries.

    `colors` names extra colourings to draw as scatters (organ, a metadata column...).
    The stage-coloured scatter comes from `pair()` instead, because that one also
    carries the boxed pair. `exclude` drops every rule naming those cell types.
    `diagnostics` draws the scree and the loadings bars - turn it off when you only
    want the PCA back to lay out yourself.
    """
    mat = _matrix(_filter(data, stages, organs, exclude))
    return _fit(mat, data, stages, organs, prefix, colors, save, diagnostics=diagnostics)


def _fit(mat, data, stages, organs, prefix, colors, save, built_from='rules',
         diagnostics=True):
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
                  model=model, built_from=built_from)
    print(f"{scope.label}: {mat.shape[0]} FOVs x {mat.shape[1]} columns  |  "
          f"PC1 {scope.variance[0]:.1f}%  PC2 {scope.variance[1]:.1f}%")

    name = (lambda what: f"{prefix}_{what}") if (save and prefix) else (lambda what: None)
    for col in colors:
        rsv.plot_pca_scatter(coords, scope.variance, color_by=col, scope=scope.label,
                            subtitle=_settings(scope, col),
                            save=name(f"pca_{col.replace(' ', '_').lower()}"))
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

def loadings(scope, save=False):
    """The rules pulling hardest on PC1 and on PC2, one figure each."""
    for i in (0, 1):
        rsv.plot_pca_loadings(scope.model.components_[i], scope.matrix.columns,
                              component_idx=i, subtitle=_settings(scope),
                              top_n=TOP_LOADINGS, scope=scope.label,
                              save=f"{scope.prefix}_loadings_pc{i + 1}" if save else None)


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
                      save=f"{scope.prefix}_fovs_{component.lower()}" if save else None)
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
                        save=f"{scope.prefix}_pca_{color_by.replace(' ', '_').lower()}"
                             if save else None)
    vh.plot_fov_panel({f: str(data.fovs.set_index('FOV').loc[f, SCORE_COL]) for f in two},
                      data.cells, data.fovs, num_cols=2,
                      save=f"{scope.prefix}_pair_fovs" if save else None)
    return two


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
    rsv.plot_patient_spread(table, save=f"{scope.prefix}_patient_spread" if save else None)
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
        save=f"{scope.prefix}_abundance_correlation" if save else None)

    # The same PCA coloured by the cell types that came top, one figure each.
    top = list(rho.abs().max(axis=1).sort_values(ascending=False).index)[:N_COLORED]
    df = scope.coords.merge(data.fractions[top], left_on='FOV', right_index=True, how='left')
    for i, cell in enumerate(top, start=1):
        rsv.plot_pca_scatter(df, scope.variance, color_by=cell, scope=scope.label,
                            subtitle=_settings(scope, f'share of {cell} cells'),
                            save=f"{scope.prefix}_abundance_top{i}" if save else None)
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
                        save=f"{scope.prefix}_pca_{color_by.replace(' ', '_').lower()}"
                             if save else None)


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
                built_from=what, diagnostics=False)


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
    min_abs, min_rel = data.support
    enough = counts.ge(np.maximum(min_abs, min_rel * counts.sum(axis=1)), axis=0)

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
                save=False, built_from=what, diagnostics=False)
