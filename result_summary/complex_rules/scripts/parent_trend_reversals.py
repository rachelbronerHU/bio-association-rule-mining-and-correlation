"""Find opposite Control-to-Severe trends for a complex rule and a fixed shorter parent."""

import numpy as np
import pandas as pd

import complex_investigation as ci


def role_preserving_parent(parent_ant, parent_con, child_ant, child_con):
    """A shorter parent can remove cell types, but cannot flip the arrow."""
    return (set(map(ci.plain, parent_ant)).issubset(map(ci.plain, child_ant))
            and set(map(ci.plain, parent_con)).issubset(map(ci.plain, child_con)))


def prepare(rules, cells, metadata, config):
    """Apply the same occurrence gates to both rules, then keep measured parent links."""
    gate = np.where(
        rules.Kind.eq('attracts'),
        (rules.Support >= config.support) & (rules.Confidence >= config.confidence),
        rules.Expected_support >= config.expected_support,
    )
    passed = rules[(rules.Individual_FDR <= config.mining_fdr) & gate].copy()
    grouped = ci.collapse_centers(passed)
    complex_rows = passed[passed.n_items >= 3]
    links = complex_rows[['FOV', 'Cell_Rule', 'Kind', 'Simpler_Rules']].copy()
    links['parent_key'] = links.Simpler_Rules.map(ci.items)
    links = links.explode('parent_key').dropna(subset='parent_key')
    parents = passed[['FOV', 'Kind', 'stored_rule', 'Cell_Rule', 'n_items']].rename(
        columns={'stored_rule': 'parent_key', 'Cell_Rule': 'parent'}
    )
    links = links.merge(parents, on=['FOV', 'Kind', 'parent_key'], validate='many_to_one')
    links = links.rename(columns={'Cell_Rule': 'rule'})
    links = links[links.n_items < links.rule.map(
        rules.drop_duplicates('Cell_Rule').set_index('Cell_Rule').n_items
    )]
    info = metadata.set_index('FOV')
    links['organ'] = links.FOV.map(info.Organ)
    pairs = links.groupby(['organ', 'rule', 'parent']).FOV.nunique().rename('matched_fovs').reset_index()
    pairs = pairs[pairs.matched_fovs >= 3]
    definitions = rules.drop_duplicates('Cell_Rule').set_index('Cell_Rule')
    # A true contextual parent preserves the arrow's two cell-type sides.
    pairs = pairs[[
        role_preserving_parent(definitions.at[parent, 'ant'], definitions.at[parent, 'con'],
                               definitions.at[rule, 'ant'], definitions.at[rule, 'con'])
        for rule, parent in zip(pairs.rule, pairs.parent)
    ]]

    names = pd.Index(pd.unique(pd.concat([pairs.rule, pairs.parent], ignore_index=True)))
    fovs = metadata.FOV
    states = grouped[grouped.Clean_Rule.isin(names)].pivot(
        index='Clean_Rule', columns='FOV', values='state'
    ).reindex(index=names, columns=fovs).fillna(0).astype('int8')
    cell_counts = cells.groupby(['cell type', 'fov']).size().unstack(fill_value=0)
    enough = cell_counts.reindex(columns=fovs, fill_value=0).ge(config.min_cells)
    eligible = pd.DataFrame(
        [enough.reindex(list(definitions.at[name, 'types']), fill_value=False).all(axis=0).to_numpy()
         for name in names], index=names, columns=fovs,
    )
    eligible.attrs['min_cells'] = config.min_cells
    return dict(rows=grouped, pairs=pairs, states=states, eligible=eligible,
                metadata=metadata, config=config)


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
        common = eligible[child_index] & scoped
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
