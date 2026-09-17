"""Compact plots for automatically found shorter-parent versus complex trend reversals."""

import hashlib

import matplotlib.pyplot as plt
import numpy as np
from IPython.display import display

import differential_stats as ds
import differential_vis as dv
import compare_rules as cr
import rule_metrics as rm
import complex_investigation as ci
import parent_trend_reversals as reversal


def filename(spec, panel):
    identity = spec['organ'] + spec['rule'] + spec['parent']
    tag = hashlib.sha1(identity.encode()).hexdigest()[:10]
    return str(ci.ROOT / 'summary_downloads' / f'reversal_{spec["organ"]}_{tag}_{panel}.pdf')


_PARENT_GREY = '#777777'
_FIELD_COLORS = {'attraction': '#2878D0', 'avoidance': '#E66A4E'}
_FIELD_LABELS = {'attraction': 'Attraction (%)', 'avoidance': 'Avoidance (%)'}
_COUNT_WORDS = {'attraction': 'attracted', 'avoidance': 'avoided', 'eligible': 'eligible'}


def arrow(rule):
    return rule.replace(' -> ', ' → ')


def comparison(data, spec, save=None):
    table = reversal.stage_counts(data, spec['organ'], spec['rule'], spec['parent'])
    by_role = {role: table[table.role.eq(role)].set_index('stage').loc[ci.STAGES]
               for role in ('parent', 'complex')}
    fields = [field for field in ('attraction', 'avoidance') if by_role['complex'][field].sum()]
    if not fields:
        print('The complex rule never attracts or avoids here; no panel to draw.')
        return

    height = 1.95 + 1.9 * len(fields)
    fig, axes = plt.subplots(len(fields), 1, figsize=(dv._TEXT_WIDTH, height),
                             sharex=True, gridspec_kw={'hspace': .3})
    axes = np.atleast_1d(axes)
    for ax, field in zip(axes, fields):
        for role, color in (('parent', _PARENT_GREY), ('complex', _FIELD_COLORS[field])):
            block = by_role[role]
            denom = block.eligible.to_numpy()
            share = 100 * block[field].to_numpy() / np.where(denom, denom, np.nan)
            ax.plot(range(3), share, marker='D', lw=2, ms=5, color=color,
                    label=role.capitalize())
        ax.set_ylabel(_FIELD_LABELS[field], fontsize=8)
        ax.set_xlim(-.25, 2.25)
        ax.set_ylim(0, 100)
        dv.tidy_axes(ax, grid='y', hide=('top', 'right'))
        ax.legend(frameon=False, loc='center left', bbox_to_anchor=(1.01, .5), fontsize=8)

    counted = fields + ['eligible']
    axes[-1].set_xticks(range(3), [
        stage + '\n' + ' | '.join(str(int(by_role['complex'].loc[stage, name]))
                       for name in counted)
        for stage in ci.STAGES])
    axes[-1].set_xlabel(' | '.join(_COUNT_WORDS[name] for name in counted)
                        + ' FOVs, complex rule', fontsize=8)

    fig.suptitle(f'{spec["organ"]} · {arrow(spec["rule"])}', fontsize=12,
                 y=1 - .28 / height)
    fig.text(.5, 1 - .55 / height, 'Complex rule versus its shorter parent',
             ha='center', va='top', fontsize=9.5)
    fig.text(.5, 1 - .76 / height,
             f'Parent: {arrow(spec["parent"])} · % of eligible FOVs',
             ha='center', va='top', fontsize=8, color='#706E68')
    fig.subplots_adjust(top=1 - 1.0 / height, bottom=.95 / height, left=.13, right=.80)
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
        examples = rm.explain_missing(examples, cells)
        cr.plot_rule_fovs(
            examples, ci.STAGES, cells, data['metadata'], spec['organ'],
            data['config'].score, min_cells=data['config'].min_cells,
            max_fdr=data['config'].mining_fdr,
            save=filename(spec, role + '_fovs'),
        )
