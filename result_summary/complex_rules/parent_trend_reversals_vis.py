"""Compact plots for automatically found shorter-parent versus complex trend reversals."""

import hashlib

import matplotlib.pyplot as plt
import numpy as np
from IPython.display import display

import differential_stats as ds
import differential_vis as dv
import complex_investigation as ci
import parent_trend_reversals as reversal


def filename(spec, panel):
    identity = spec['organ'] + spec['rule'] + spec['parent']
    tag = hashlib.sha1(identity.encode()).hexdigest()[:10]
    return str(ci.ROOT / 'summary_downloads' / f'reversal_{spec["organ"]}_{tag}_{panel}.pdf')


def comparison(data, spec, save=None):
    table = reversal.stage_counts(data, spec['organ'], spec['rule'], spec['parent'])
    fields = [field for field in ['attraction', 'avoidance'] if table[field].sum()]
    if len(fields) == 2:
        fields.append('net')
    fig, axes = plt.subplots(len(fields), 1,
                             figsize=(dv._TEXT_WIDTH, 2.9 if len(fields) == 1 else 6.3),
                             sharex=True, gridspec_kw={'hspace': .42})
    axes = np.atleast_1d(axes)
    colors = {'parent': '#777777', 'complex': '#2878D0'}
    for role, name in [('parent', spec['parent']), ('complex', spec['rule'])]:
        block = table[table.role.eq(role)].set_index('stage').loc[ci.STAGES]
        denom = block.eligible.to_numpy()
        for ax, field in zip(axes, fields):
            values = (100 * block.net.to_numpy() if field == 'net' else
                      100 * block[field].to_numpy() / np.where(denom, denom, np.nan))
            ax.plot(range(3), values, marker='D', lw=2, ms=5, color=colors[role],
                    label=role.capitalize())
    labels = {'attraction': 'Attraction (% eligible FOVs)',
              'avoidance': 'Avoidance (% eligible FOVs)',
              'net': 'Net: attraction − avoidance (points)'}
    for ax, field in zip(axes, fields):
        ax.set_ylabel(labels[field], fontsize=8)
        ax.set_xlim(-.25, 2.25)
        dv.tidy_axes(ax, grid='y', hide=('top', 'right'))
        if field == 'net':
            ax.axhline(0, color='#999999', lw=.8)
            ax.set_ylim(-100, 100)
        else:
            ax.set_ylim(0, 100)
    axes[0].legend(frameon=False, loc='center left', bbox_to_anchor=(1.01, .5), fontsize=8)
    axes[-1].set_xticks(range(3), [f'{stage}\n{n} eligible FOVs'
                                  for stage, n in zip(ci.STAGES,
                                      table[table.role.eq('complex')].eligible)])
    fig.suptitle(f'{spec["organ"]} · parent versus complex by stage', fontsize=11)
    fig.text(.5, .935, f'Parent: {spec["parent"]}\nComplex: {spec["rule"]}',
             ha='center', va='top', fontsize=8)
    fig.subplots_adjust(top=.72 if len(fields) == 1 else .82,
                        bottom=.19 if len(fields) == 1 else .11, left=.19, right=.79)
    dv._finish(fig, save)


def show_selected(index, selected, data, cells):
    if not 0 <= index < len(selected):
        print(f'No result at index {index}; {len(selected)} selected reversals.')
        return
    spec = selected.iloc[index].to_dict()
    print(f'[{index}] {spec["organ"]}: {spec["parent"]}  vs  {spec["rule"]}')
    print('Same complex-rule eligibility; net = attraction share − avoidance share.')
    table = reversal.stage_counts(data, spec['organ'], spec['rule'], spec['parent'])
    display(table[['stage', 'role', 'attraction', 'avoidance', 'eligible',
                   'patients', 'rule_patients', 'net']])
    comparison(data, spec, filename(spec, 'summary'))
    common = data['eligible'].loc[[spec['rule'], spec['parent']]].copy()
    common.loc[spec['parent']] = common.loc[spec['rule']].to_numpy()
    for role, name in [('parent', spec['parent']), ('complex', spec['rule'])]:
        examples = ds.representative_fovs(
            data['rows'], common, data['metadata'], name, spec['organ'],
            data['config'].score, ci.STAGES, 'Lift',
        )
        dv.plot_pair_rule_fovs(
            examples, ci.STAGES, cells, data['metadata'], spec['organ'],
            data['config'].score, min_cells=data['config'].min_cells,
            highlight_label='Highlighted rule cell types',
            selection='each observed state near median Lift; typical eligible no-rule FOV',
            save=filename(spec, role + '_fovs'), split_states=True,
        )
