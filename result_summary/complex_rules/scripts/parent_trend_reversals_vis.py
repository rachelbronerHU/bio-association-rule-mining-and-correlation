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
import complex_investigation_vis as civ
import parent_trend_reversals as reversal


def filename(spec, panel):
    identity = spec['organ'] + spec['rule'] + spec['parent']
    tag = hashlib.sha1(identity.encode()).hexdigest()[:10]
    return str(ci.ROOT / 'summary_downloads' / f'reversal_{spec["organ"]}_{tag}_{panel}.pdf')


def show_selected(index, selected, data, cells):
    if not 0 <= index < len(selected):
        print(f'No result at index {index}; {len(selected)} selected reversals.')
        return
    spec = selected.iloc[index].to_dict()
    print(f'[{index}] {spec["organ"]}: {spec["parent"]}  vs  {spec["rule"]}')
    print('FOVs testable for both rules; net = attraction share − avoidance share.')
    table = ci.stage_counts(data, spec['organ'], spec['rule'], spec['parent'])
    display(table[['stage', 'role', 'attraction', 'avoidance', 'eligible',
                   'patients', 'rule_patients', 'net']])
    civ.parent_vs_complex(table, spec['organ'], spec['rule'], spec['parent'],
                          filename(spec, 'summary'))
    common = data['eligible'].loc[[spec['rule'], spec['parent']]].copy()
    common.loc[:, :] = common.all(axis=0).to_numpy()
    fields = data['fields']
    for role, name in [('parent', spec['parent']), ('complex', spec['rule'])]:
        examples = ds.representative_fovs(
            data['rows'], common, data['metadata'], name, spec['organ'],
            data['config'].score, ci.STAGES, 'Lift',
        )
        examples = rm.explain_missing(examples, cells, fields=fields)
        cr.plot_rule_fovs(
            examples, ci.STAGES, cells, data['metadata'], spec['organ'],
            data['config'].score,
            max_fdr=data['config'].mining_fdr,
            save=filename(spec, role + '_fovs'), fields=fields,
        )
