"""Preparation and automatic selection for the complex-rule investigations.

Nothing here re-mines rules or changes their stored classification / mining FDR.
"""
import ast
import json
from dataclasses import asdict, dataclass
from functools import lru_cache
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from statsmodels.stats.multitest import multipletests

import data_helper as dh
import differential_stats as ds

STAGES = ['Control', 'Mild', 'Severe']
ORGANS = ['Colon', 'Duodenum']
ROOT = Path(__file__).resolve().parent.parent


@dataclass(frozen=True)
class Config:
    max_items: int = 4
    min_cells: int = 20
    mining_fdr: float = .05
    support: float = .01
    confidence: float = .5  # attraction only; avoidance has a separate opportunity gate
    expected_support: float = .01
    score: str = 'Clinical score'
    min_fovs: int = 5
    min_patients: int = 3
    min_group_fovs: int = 10
    visible_gap: float = .15
    attraction_lift_gap: float = .30
    avoidance_lift_gap: float = .15
    permutations: int = 4999
    top_n: int = 8


@lru_cache(None)
def _parsed_items(value):
    return tuple(sorted(ast.literal_eval(value)))


def items(value):
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return ()
    return tuple(sorted(value)) if isinstance(value, (list, tuple)) else _parsed_items(value)


def plain(item):
    return item.replace('_CENTER', '').replace('_NEIGHBOR', '')


def rule_name(ant, con):
    """Keep center identity visible: unmarked items are neighbors."""
    label = lambda item: plain(item) + (' [C]' if item.endswith('_CENTER') else '')
    return ' + '.join(map(label, ant)) + ' -> ' + ' + '.join(map(label, con))


def cell_rule_name(ant, con):
    """Arrow and cell types define the reported rule; mining centers remain internal."""
    side = lambda values: ' + '.join(sorted(map(plain, values)))
    return side(ant) + ' -> ' + side(con)


def prepare(raw, metadata, max_items=4):
    """Parse unique sides once, preserve exact roles, and use distinct cell types."""
    raw = raw[raw.FOV.isin(metadata.FOV)].copy()
    definitions = raw[['Antecedents', 'Consequents']].drop_duplicates().copy()
    definitions['ant'] = definitions.Antecedents.map(items)
    definitions['con'] = definitions.Consequents.map(items)
    definitions['types'] = [tuple(sorted(set(map(plain, a+c))))
                            for a, c in zip(definitions.ant, definitions.con)]
    definitions['n_items'] = [len(a+c) for a, c in zip(definitions.ant, definitions.con)]
    # Repeated labels are not a three/four-cell-type niche.
    keep = ((definitions.n_items <= max_items)
            & (definitions.n_items == definitions.types.map(len))
            & ~definitions.types.map(lambda x: bool(set(x) & set(dh.DROP_CELLS))))
    definitions = definitions[keep].copy()
    definitions['Clean_Rule'] = [rule_name(a, c) for a, c in zip(definitions.ant, definitions.con)]
    definitions['Cell_Rule'] = [cell_rule_name(a, c) for a, c in zip(definitions.ant, definitions.con)]
    definitions['stored_rule'] = [' + '.join(a)+' -> '+' + '.join(c)
                                  for a, c in zip(definitions.ant, definitions.con)]
    result = raw.merge(definitions, on=['Antecedents', 'Consequents'], validate='many_to_one')
    result['state'] = result.Kind.map({'attracts': 1, 'avoids': -1})
    # P(A)P(B) = observed support - leverage, including zero-Lift avoidance.
    result['Expected_support'] = (result.Support - result.Leverage).clip(lower=0)
    if result.duplicated(['FOV', 'Clean_Rule']).any():
        raise ValueError('More than one occurrence/state for an exact role-aware rule and FOV')
    return result


def load(config=Config()):
    cells, metadata, _ = dh.load_spatial_data()
    raw = dh._read_rules(dh.RESULT_CSV_PATH, config.max_items)
    rules = prepare(raw, metadata, config.max_items)
    settings = json.loads((Path(dh.RESULT_CSV_PATH).parent / 'run_config.json').read_text())
    print(asdict(config))
    print(f'{len(rules):,} stored center-specific occurrences; distinct cell types only')
    print('Mining settings:', settings)
    return cells, metadata, rules


def investigation_rows(rules, config, mode='all', informative=True):
    """A per-FOV investigation gate, not a new mining threshold."""
    gate = np.where(rules.Kind.eq('attracts'),
                    (rules.Support >= config.support) & (rules.Confidence >= config.confidence),
                    rules.Expected_support >= config.expected_support)
    keep = (rules.n_items >= 3) & (rules.Individual_FDR <= config.mining_fdr) & gate
    if informative:
        keep &= rules.Adds_Information
    if mode == 'new':
        keep &= rules.Complex_Class.eq('new')
    elif mode == 'non-new':
        keep &= ~rules.Complex_Class.eq('new')
    elif mode != 'all':
        raise ValueError(mode)
    selected = rules[keep].copy()
    return collapse_centers(selected) if 'Cell_Rule' in selected else selected


def collapse_centers(rows):
    """Average passing center variants in each FOV, preserving the arrow sides.

    The minimum constituent mining FDR is retained only as an audit field; it is
    not a newly corrected FDR for the pooled cell-type rule. If opposite states
    occur, the mean Lift determines the descriptive pooled state.
    """
    if rows.empty:
        return rows.assign(Clean_Rule=pd.Series(dtype='object'))
    numeric = [name for name in ('Support', 'Confidence', 'Lift', 'Leverage',
                                  'Conviction', 'Expected_support') if name in rows]
    grouped = rows.groupby(['FOV', 'Cell_Rule'], sort=False)
    result = grouped.first().reset_index()
    means = grouped[numeric].mean().reset_index()
    result = result.drop(columns=numeric).merge(means, on=['FOV', 'Cell_Rule'], validate='one_to_one')
    result['n_centers'] = grouped.size().to_numpy()
    result['center_rules'] = grouped.Clean_Rule.agg(tuple).to_numpy()
    result['center_states'] = grouped.state.agg(lambda x: tuple(sorted(set(x)))).to_numpy()
    result['Individual_FDR'] = grouped.Individual_FDR.min().to_numpy()
    result['Simpler_Rules'] = grouped.Simpler_Rules.agg(
        lambda values: tuple(sorted(set().union(*(items(value) for value in values))))
    ).to_numpy()
    result['Adds_Information'] = grouped.Adds_Information.any().to_numpy()
    if 'Complex_Class' in rows:
        priority = {'new': 5, 'stronger_effect': 4, 'consequent_is_noise': 3,
                    'simpler_are_noise': 2, 'consequent_driven': 1, 'redundant_by_simpler': 0}
        result['Complex_Class'] = grouped.Complex_Class.agg(
            lambda values: max(values, key=lambda value: priority.get(value, -1))
        ).to_numpy()
    result['Clean_Rule'] = result.Cell_Rule
    result['state'] = np.sign(result.Lift - 1).astype('int8')
    result['Kind'] = result.state.map({1: 'attracts', -1: 'avoids', 0: 'neutral'})
    return result[result.state.ne(0)].reset_index(drop=True)


def matrices(rules, rows, cells, metadata, config, extra=None):
    key = 'Cell_Rule' if 'Cell_Rule' in rules else 'Clean_Rule'
    definitions = rules[rules.n_items >= 3].drop_duplicates(key).set_index(key)
    types = definitions.types.to_dict()
    types.update(extra or {})
    cell_counts = cells.groupby(['cell type','fov']).size().unstack(fill_value=0)
    enough = cell_counts.reindex(columns=metadata.FOV,fill_value=0).ge(config.min_cells)
    masks = {t: enough.reindex(list(t),fill_value=False).all(axis=0).to_numpy()
             for t in set(tuple(v) for v in types.values())}
    eligibility = pd.DataFrame([masks[tuple(t)] for t in types.values()],index=list(types),columns=metadata.FOV)
    eligibility.attrs['min_cells'] = config.min_cells
    states = rows.pivot(index='Clean_Rule', columns='FOV', values='state').reindex(
        index=eligibility.index, columns=eligibility.columns).fillna(0).astype('int8')
    return states, eligibility


def occurrence_table(states, eligibility, rows, metadata, config):
    """Both directions and all denominators; All is pooled, never an extra sample."""
    output = []
    for organ in ORGANS:
        for group in ['All', *STAGES]:
            scope = metadata[metadata.Organ.eq(organ)]
            if group != 'All':
                scope = scope[scope[config.score].eq(group)]
            ids = scope.FOV.tolist()
            eligible = eligibility[ids]
            for kind, state in [('attracts', 1), ('avoids', -1)]:
                present = states[ids].eq(state) & eligible
                count = present.sum(axis=1)
                patients = ds.aggregate_fovs(present.astype(float).where(eligible), scope, 'PatientID')
                eligible_patients = patients.notna().sum(axis=1)
                hit_patients = patients.gt(0).sum(axis=1)
                block = pd.DataFrame(dict(rule=states.index, organ=organ, stage=group, kind=kind,
                                          total=len(ids), eligible=eligible.sum(axis=1).values,
                                          hits=count.values, patients=eligible_patients.values,
                                          hit_patients=hit_patients.values))
                block['share'] = block.hits / block.eligible.replace(0, np.nan)
                output.append(block)
    return pd.concat(output, ignore_index=True)


def independent_values(values, metadata, organ, score, unit):
    """Equal biopsy weighting for patients; omit cross-stage patients from unpaired tests."""
    meta = metadata[metadata.Organ.eq(organ) & metadata[score].isin(STAGES)].copy()
    if unit == 'FOV':
        return values[meta.FOV], meta, 0
    if unit == 'Biopsy':
        return ds.aggregate_fovs(values, meta, unit), meta, 0
    cross_stage = meta.groupby('PatientID')[score].nunique().gt(1)
    omitted = cross_stage[cross_stage].index
    meta = meta[~meta.PatientID.isin(omitted)]
    biopsy = ds.aggregate_fovs(values, meta, 'Biopsy')
    biopsy_meta = meta.drop_duplicates('Biopsy').drop(columns='FOV').rename(columns={'Biopsy': 'FOV'})
    patients = ds.aggregate_fovs(biopsy, biopsy_meta, 'PatientID')
    return patients, meta, len(omitted)


def stage_tests(states, eligible, counts, metadata, config, metrics=None):
    """Same differential permutation methods; FDR spans organs/kinds/contrasts per unit."""
    results = []
    pooled = counts[counts.stage.eq('All')]
    for organ in ORGANS:
        for kind, state in [('attracts', 1), ('avoids', -1)]:
            candidates = pooled[(pooled.organ == organ) & (pooled.kind == kind)
                                & (pooled.hits >= config.min_fovs)
                                & (pooled.hit_patients >= config.min_patients)].rule
            values = (states.loc[candidates].eq(state).astype(float) if metrics is None
                      else metrics[state].reindex(candidates)).where(eligible.loc[candidates])
            for unit in ['FOV', 'Biopsy', 'PatientID']:
                numbers, meta, omitted = independent_values(values, metadata, organ, config.score, unit)
                ordered, pairs = ds.severity_screen(numbers, meta, organ, config.score, STAGES,
                                                    min_eligible=config.min_patients,
                                                    min_present=config.min_patients if metrics is None else 0,
                                                    n_permutations=config.permutations, unit_col=unit)
                for contrast, frame in [('ordered', ordered), *[('–'.join(k), v) for k,v in pairs.items()]]:
                    frame = frame.rename_axis('rule').reset_index()
                    frame['organ'], frame['kind'], frame['unit'] = organ, kind, unit
                    frame['contrast'], frame['omitted_patients'] = contrast, omitted
                    results.append(frame)
    result = pd.concat(results, ignore_index=True)
    result['fdr'] = np.nan
    for unit, group in result.groupby('unit'):
        valid = group.p_value.dropna()
        if len(valid):
            result.loc[valid.index, 'fdr'] = multipletests(valid, method='fdr_bh')[1]
    return result


def strength_screen(analysis, metadata):
    """Conditional Lift changes, not changes in how often a rule passes the gates."""
    config = analysis['config']
    matrices = {s: ds.direction_metric_matrix(analysis['rows'], analysis['states'],
                                               analysis['eligible'], state=s) for s in [1,-1]}
    tests = stage_tests(analysis['states'], analysis['eligible'], analysis['counts'], metadata, config, matrices)
    q = tests[tests.unit.eq('PatientID')].groupby(['organ','rule','kind']).fdr.min().to_dict()
    rows = analysis['rows']
    rows = rows[[bool(analysis['eligible'].at[r,f]) for r,f in zip(rows.Clean_Rule,rows.FOV)]]
    rows = rows[['FOV','Clean_Rule','Kind','Lift']].merge(
        metadata[['FOV','Organ','PatientID',config.score]], on='FOV', validate='many_to_one')
    grouped = rows.groupby(['Organ','Clean_Rule','Kind',config.score]).agg(
        median=('Lift','median'),fovs=('FOV','nunique'),patients=('PatientID','nunique'))
    output = []
    for (organ,rule,kind), block in grouped.groupby(level=[0,1,2],sort=False):
        block = block.droplevel([0,1,2])
        gaps = []
        for a,b in combinations(STAGES,2):
            if a not in block.index or b not in block.index:
                continue
            x,y = block.loc[a],block.loc[b]
            if min(x.fovs,y.fovs) >= config.min_fovs and min(x.patients,y.patients) >= config.min_patients:
                gaps.append((abs(y['median']-x['median']),a+'–'+b))
        gap,contrast = max(gaps,default=(0.,'none'))
        fdr = q.get((organ,rule,kind),np.nan)
        threshold = config.attraction_lift_gap if kind=='attracts' else config.avoidance_lift_gap
        if contrast!='none' and (gap >= threshold or fdr <= .05):
            output.append(dict(organ=organ,rule=rule,kind=kind,gap=gap,contrast=contrast,
                               patient_fdr=fdr,stream='conditional Lift',test_metric='Lift'))
    specs = pd.DataFrame(output,columns=['organ','rule','kind','gap','contrast','patient_fdr','stream','test_metric'])
    specs = specs.sort_values('gap',ascending=False).groupby(['organ','kind'],sort=False).head(2).reset_index(drop=True)
    return dict(analysis,tests=tests,test_metric='Lift'), specs


def candidates(counts, tests, config, purpose='changes'):
    """Separate visible/statistical streams plus recurrent-rule ranking, without named rules."""
    output = []
    viable = counts[(counts.stage == 'All') & (counts.hits >= config.min_fovs)
                    & (counts.hit_patients >= config.min_patients)][['organ','rule','kind']]
    counts = counts.merge(viable,on=['organ','rule','kind'],validate='many_to_one')
    patient_q = tests[tests.unit.eq('PatientID')].groupby(['organ','rule','kind']).fdr.min().to_dict()
    for (organ, rule, kind), group in counts.groupby(['organ', 'rule', 'kind'], sort=False):
        pooled = group[group.stage.eq('All')].iloc[0]
        if pooled.hits < config.min_fovs or pooled.hit_patients < config.min_patients:
            continue
        by_stage = group.set_index('stage')
        gaps = []
        for a,b in combinations(STAGES, 2):
            x,y = by_stage.loc[a], by_stage.loc[b]
            if min(x.eligible,y.eligible) >= config.min_group_fovs and min(x.patients,y.patients) >= config.min_patients:
                gaps.append((abs(x.share-y.share), a+'–'+b))
        gap, contrast = max(gaps, default=(0., 'none'))
        q = patient_q.get((organ,rule,kind), np.nan)
        visible, supported = gap >= config.visible_gap, pd.notna(q) and q <= .05
        recurrent = pooled.hits >= 10 and pooled.hit_patients >= 5 and pooled.share >= .10
        if purpose == 'changes' and not (visible or supported):
            continue
        if purpose == 'recurrence' and not recurrent:
            continue
        stream = 'both' if visible and supported else ('statistical' if supported else 'visible')
        if purpose == 'recurrence':
            stream = 'recurrence'
        output.append(dict(organ=organ, rule=rule, kind=kind, gap=gap, contrast=contrast,
                           patient_fdr=q, hits=int(pooled.hits), patients=int(pooled.hit_patients),
                           eligible=int(pooled.eligible), share=pooled.share, stream=stream))
    columns = ['organ','rule','kind','gap','contrast','patient_fdr','hits','patients','eligible','share','stream']
    table = pd.DataFrame(output, columns=columns)
    if table.empty:
        return table
    sort = ['share','patients','hits'] if purpose == 'recurrence' else ['gap','patients','hits']
    ranked = table.sort_values(sort, ascending=False)
    # Each organ/kind has its own slots; a supported stream cannot be crowded out by visible gaps.
    selected = []
    for _, group in ranked.groupby(['organ','kind'], sort=False):
        quota = max(2, config.top_n // 4)
        if purpose == 'changes':
            supported = group[group.patient_fdr <= .05].sort_values('patient_fdr').head(1)
            group = pd.concat([supported,group]).drop_duplicates(['organ','rule','kind'])
        selected.append(group.head(quota))
    return pd.concat(selected).drop_duplicates(['organ','rule','kind']).reset_index(drop=True)


def analyse(rules, cells, metadata, config=Config(), mode='all', extra=None, informative=True):
    rows = investigation_rows(rules, config, mode, informative)
    states, eligible = matrices(rules, rows, cells, metadata, config, extra)
    counts = occurrence_table(states, eligible, rows, metadata, config)
    tests = stage_tests(states, eligible, counts, metadata, config)
    return dict(rows=rows, states=states, eligible=eligible, counts=counts, tests=tests,
                config=config, mode=mode, informative=informative)


def stage_counts(data, organ, rule, parent):
    """Both rules use the complex rule's eligible FOVs in every displayed stage."""
    states, eligible, metadata, config = (data[key] for key in
                                          ['states', 'eligible', 'metadata', 'config'])
    records = []
    for stage in STAGES:
        scope = metadata[metadata.Organ.eq(organ) & metadata[config.score].eq(stage)]
        ids = scope.FOV[eligible.loc[rule, scope.FOV].to_numpy()].tolist()
        patients = scope.set_index('FOV').loc[ids, 'PatientID']
        for name, role in [(parent, 'parent'), (rule, 'complex')]:
            values = states.loc[name, ids]
            attraction = int(values.eq(1).sum())
            avoidance = int(values.eq(-1).sum())
            records.append(dict(stage=stage, role=role, rule=name, eligible=len(ids),
                                patients=patients.nunique(), attraction=attraction,
                                avoidance=avoidance,
                                rule_patients=patients[values.to_numpy() != 0].nunique(),
                                net=(attraction - avoidance) / len(ids) if ids else np.nan))
    return pd.DataFrame(records)


def matched_parents(rule, kind, rows, all_rules):
    """Measured same-FOV parents; average their center variants before ranking."""
    current = rows[(rows.Clean_Rule == rule) & (rows.Kind == kind)]
    output = []
    available = all_rules[all_rules.Kind.eq(kind) & all_rules.FOV.isin(current.FOV)]
    lookup = {fov: frame.set_index('stored_rule') for fov,frame in available.groupby('FOV')}
    for _, row in current.iterrows():
        frame = lookup.get(row.FOV)
        if frame is None:
            continue
        names = list(items(row.Simpler_Rules))
        parents = frame.loc[frame.index.intersection(names)]
        if parents.empty:
            continue
        parent_key = 'Cell_Rule' if 'Cell_Rule' in parents else 'Clean_Rule'
        parents = parents.groupby(parent_key, sort=False).agg(
            Lift=('Lift', 'mean'), Individual_FDR=('Individual_FDR', 'min')
        ).reset_index()
        parent = parents.loc[parents.Lift.idxmax() if kind == 'attracts' else parents.Lift.idxmin()]
        output.append(dict(FOV=row.FOV, complex_lift=row.Lift, parent_lift=parent.Lift,
                           parent=parent[parent_key], parent_fdr=parent.Individual_FDR,
                           gain=(row.Lift/parent.Lift if kind == 'attracts'
                                 else parent.Lift/row.Lift if row.Lift > 0 else np.inf)))
    return pd.DataFrame(output, columns=['FOV','complex_lift','parent_lift','parent','parent_fdr','gain'])


def save_tables(analysis, specs, prefix):
    folder = ROOT / 'summary_downloads'
    folder.mkdir(exist_ok=True)
    for name in ['counts', 'tests']:
        analysis[name].to_csv(folder / f'{prefix}_{name}.csv', index=False)
    specs.to_csv(folder / f'{prefix}_selected.csv', index=False)


def gain_candidates(analysis, all_rules, metadata):
    """Rank measured same-field improvements; absent parents and infinite ratios cannot win."""
    config = analysis['config']
    rows = analysis['rows'].copy()
    rows = rows[[bool(analysis['eligible'].at[r,f]) for r,f in zip(rows.Clean_Rule,rows.FOV)]]
    rows['parent_key'] = rows.Simpler_Rules.map(items)
    rows = rows.explode('parent_key').dropna(subset=['parent_key'])
    parent_key = 'Cell_Rule' if 'Cell_Rule' in all_rules else 'Clean_Rule'
    parent = all_rules[['FOV','Kind','stored_rule',parent_key,'Lift']].rename(
        columns={'stored_rule':'parent_key',parent_key:'parent_cell_rule','Lift':'parent_lift'})
    matched = rows.merge(parent,on=['FOV','Kind','parent_key'],validate='many_to_one')
    matched = matched.groupby(['FOV','Clean_Rule','Kind','parent_cell_rule'],sort=False).agg(
        Lift=('Lift','first'), parent_lift=('parent_lift','mean')
    ).reset_index()
    matched['rank_lift'] = np.where(matched.Kind.eq('attracts'),-matched.parent_lift,matched.parent_lift)
    matched = matched.sort_values('rank_lift').drop_duplicates(['FOV','Clean_Rule'])
    # Use raw Lift separation for avoidance: zero is legitimate, not an arbitrary 1e6 strength.
    matched['improvement'] = np.where(matched.Kind.eq('attracts'),
                                      matched.Lift-matched.parent_lift,matched.parent_lift-matched.Lift)
    info = metadata.set_index('FOV')
    matched['organ'] = matched.FOV.map(info.Organ)
    matched['patient'] = matched.FOV.map(info.PatientID)
    table = matched.groupby(['organ','Clean_Rule','Kind']).agg(
        gain=('improvement','median'),matched_fovs=('FOV','nunique'),patients=('patient','nunique')).reset_index()
    table = table[(table.matched_fovs >= config.min_fovs) & (table.patients >= config.min_patients)
                  & (table.gain >= .1)].sort_values('gain',ascending=False)
    table = table.groupby(['organ','Kind'],sort=False).head(1)
    return table.rename(columns={'Clean_Rule':'rule','Kind':'kind'}).assign(stream='measured gain',contrast='Control–Severe')


def test_labels(analysis, spec):
    tests = analysis['tests']
    rows = tests[(tests.rule == spec['rule']) & (tests.organ == spec['organ'])
                 & (tests.kind == spec.get('kind', 'attracts'))]
    # Same named contrast for all units, never three independently minimized p-values.
    contrast = spec.get('contrast', 'Control–Severe')
    if contrast == 'none':
        contrast = 'Control–Severe'
    labels = []
    for unit, label in [('FOV','FOV'), ('Biopsy','biopsy'), ('PatientID','patient')]:
        q = rows.loc[(rows.unit == unit) & (rows.contrast == contrast), 'fdr']
        labels.append(f'{label}: {q.iloc[0]:.3g}' if len(q) and pd.notna(q.iloc[0]) else f'{label}: not tested')
    return contrast+' '+analysis.get('test_metric','occurrence')+' FDR · '+' · '.join(labels)
