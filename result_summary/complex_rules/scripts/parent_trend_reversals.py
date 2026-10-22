"""Find opposite Control-to-Severe trends for a complex rule and a fixed shorter parent."""

import numpy as np
import pandas as pd

import complex_investigation as ci
import data_helper as dh
import rule_metrics as rm


def prepare(rules, cells, metadata, config):
    """Apply the same occurrence gates to both rules, then keep measured parent links."""
    gate = np.where(
        rules.Kind.eq('attracts'),
        (rules.Support >= config.support) & (rules.Confidence >= config.confidence),
        rules.Expected_support >= config.expected_support,
    )
    passed = rules[(dh.rule_fdr(rules) <= config.mining_fdr) & gate].copy()
    grouped = ci.collapse_centers(passed)
    links = passed.loc[passed.n_items >= 3, ['FOV', 'Cell_Rule', 'Kind', 'simpler_rules']]
    links = links.explode('simpler_rules').rename(columns={'Cell_Rule': 'rule', 'simpler_rules': 'parent'})
    parents = passed[['FOV', 'Kind', 'Cell_Rule']].drop_duplicates().rename(columns={'Cell_Rule': 'parent'})
    links = links.merge(parents, on=['FOV', 'Kind', 'parent'])
    info = metadata.set_index('FOV')
    links['organ'] = links.FOV.map(info.Organ)
    pairs = links.groupby(['organ', 'rule', 'parent']).FOV.nunique().rename('matched_fovs').reset_index()
    pairs = pairs[pairs.matched_fovs >= 3]

    names = pd.Index(pd.unique(pd.concat([pairs.rule, pairs.parent], ignore_index=True)))
    fovs = metadata.FOV
    states = grouped[grouped.Clean_Rule.isin(names)].pivot(
        index='Clean_Rule', columns='FOV', values='state'
    ).reindex(index=names, columns=fovs).fillna(0).astype('int8')
    fields = rm.Fields(cells)
    eligible = ci.cell_rule_testable_fovs(list(names), cells, metadata, config, fields=fields)
    return dict(rows=grouped, pairs=pairs, states=states, eligible=eligible,
                metadata=metadata, config=config, fields=fields)


def select(data, min_eligible=10, min_patients=3, min_gap=.20, min_hits=5,
           top_n=2, min_mild_eligible=8):
    """Rank balanced sign reversals; no named rule or significance gate enters selection."""
    pairs = data['pairs'].copy()
    if pairs.empty:
        return pairs
    names = data['states'].index
    child_index = names.get_indexer(pairs.rule)
    parent_index = names.get_indexer(pairs.parent)
    states = data['states'].to_numpy()
    eligible = data['eligible'].to_numpy()
    metadata = data['metadata']
    config = data['config']
    for stage, label in [('Control', 'control'), ('Severe', 'severe')]:
        organ_masks = np.stack([
            (metadata.Organ.eq(organ) & metadata[config.score].eq(stage)).to_numpy()
            for organ in ci.ORGANS
        ])
        scoped = organ_masks[pairs.organ.map({organ: i for i, organ in enumerate(ci.ORGANS)})]
        common = eligible[child_index] & eligible[parent_index] & scoped
        denominator = common.sum(axis=1)
        pairs[label + '_eligible'] = denominator
        for role, index in [('parent', parent_index), ('complex', child_index)]:
            values = states[index]
            attraction = ((values == 1) & common).sum(axis=1)
            avoidance = ((values == -1) & common).sum(axis=1)
            pairs[label + '_' + role + '_hits'] = attraction + avoidance
            pairs[label + '_' + role + '_net'] = np.divide(
                attraction - avoidance, denominator,
                out=np.full(len(pairs), np.nan), where=denominator > 0,
            )
    pairs['parent_gap'] = pairs.severe_parent_net - pairs.control_parent_net
    pairs['complex_gap'] = pairs.severe_complex_net - pairs.control_complex_net
    pairs['balanced_gap'] = np.minimum(pairs.parent_gap.abs(), pairs.complex_gap.abs())
    pairs = pairs[(pairs.control_eligible >= min_eligible)
                  & (pairs.severe_eligible >= min_eligible)
                  & (pairs.parent_gap * pairs.complex_gap < 0)
                  & (pairs.balanced_gap >= min_gap)
                  & (pairs[['control_parent_hits', 'severe_parent_hits']].max(axis=1) >= min_hits)
                  & (pairs[['control_complex_hits', 'severe_complex_hits']].max(axis=1) >= min_hits)]
    pairs = pairs.sort_values(['balanced_gap', 'matched_fovs'], ascending=False)
    selected = []
    organ_counts = {organ: 0 for organ in ci.ORGANS}
    seen_rules = set()
    for pair in pairs.itertuples(index=False):
        if organ_counts[pair.organ] >= top_n or (pair.organ, pair.rule) in seen_rules:
            continue
        counts = ci.stage_counts(data, pair.organ, pair.rule, pair.parent)
        endpoints = counts[counts.stage.isin(['Control', 'Severe'])]
        if endpoints.patients.min() < min_patients or counts.eligible.min() < min_mild_eligible:
            continue
        if min(endpoints.groupby('role').rule_patients.max()) < min_patients:
            continue
        selected.append(pair._asdict())
        organ_counts[pair.organ] += 1
        seen_rules.add((pair.organ, pair.rule))
        if len(selected) >= top_n * len(ci.ORGANS):
            break
    columns = ['organ', 'rule', 'parent', 'parent_gap', 'complex_gap', 'balanced_gap',
               'matched_fovs', 'control_eligible', 'severe_eligible']
    return pd.DataFrame(selected).reindex(columns=columns)
