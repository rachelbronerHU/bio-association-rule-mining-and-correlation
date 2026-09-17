"""Explain one rule-bearing FOV beside an eligible FOV without that rule.

The patch reconstruction follows the binary CN settings used by this run. It is a
diagnostic, not a replacement for the stored mining result or its shuffle FDR.
"""

import ast
import json
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from bs4 import BeautifulSoup
from IPython.display import display
from sklearn.neighbors import NearestNeighbors

import data_helper as dh
from vis_helper import plot_fov, set_cell_colors


ROOT = Path(__file__).resolve().parent


def selected_from_notebook(path):
    """Read the selected-rule table from an already executed source notebook."""
    notebook = json.loads(Path(path).read_text(encoding='utf-8'))
    tables = []
    for cell in notebook['cells']:
        source = ''.join(cell.get('source', []))
        if not any(marker in source for marker in
                   ('display(pd.DataFrame(SPECS)', 'display(SPECS', 'display(selected)')):
            continue
        for output in cell.get('outputs', []):
            html = output.get('data', {}).get('text/html')
            if not html:
                continue
            soup = BeautifulSoup(''.join(html), 'html.parser')
            for markup in soup.find_all('table'):
                head = markup.find('thead')
                body = markup.find('tbody')
                if head is None or body is None:
                    continue
                header_rows = [[tag.get_text(strip=True).lower()
                                for tag in row.find_all(['th','td'])]
                               for row in head.find_all('tr')]
                headers = next((row for row in header_rows if {'organ','rule'}.issubset(row)),
                               header_rows[-1])
                values = [[tag.get_text(strip=True) for tag in row.find_all(['th','td'])]
                          for row in body.find_all('tr')]
                table = pd.DataFrame([row for row in values if len(row) == len(headers)],
                                     columns=headers)
                if {'organ', 'rule'}.issubset(table.columns):
                    columns = ['organ', 'rule', *[name for name in ('direction','kind') if name in table]]
                    tables.append(table[columns])
    if tables:
        result = pd.concat(tables, ignore_index=True).dropna().drop_duplicates().reset_index(drop=True)
        result['rule'] = result.rule.map(canonical_rule)
        return result.drop_duplicates().reset_index(drop=True)
    # Paper alignment names are deliberately fixed by the paper, not by results.
    pairs = re.findall(r'show_rule\([\'\"]([^\'\"]+)[\'\"],\s*[\'\"]([^\'\"]+)[\'\"]\)',
                       '\n'.join(''.join(cell.get('source', [])) for cell in notebook['cells']))
    if pairs:
        return pd.DataFrame([{'rule': canonical_rule(rule), 'organ': organ}
                             for rule, organ in pairs]).drop_duplicates()
    if Path(path).stem == 'rule_occurrence_overview':
        for cell in notebook['cells']:
            if 'display(named)' not in ''.join(cell.get('source', [])):
                continue
            for output in cell.get('outputs', []):
                html = output.get('data', {}).get('text/html')
                if not html:
                    continue
                soup = BeautifulSoup(''.join(html), 'html.parser')
                rules = [row.find(['th','td']).get_text(strip=True)
                         for row in soup.select('tbody tr')]
                return pd.DataFrame([{'organ': organ, 'rule': canonical_rule(rule)}
                                     for organ in ('Colon','Duodenum') for rule in rules])
    raise ValueError(f'No displayed selected-rule table in {path}; run the source notebook first.')


def cell_types(rule):
    ant, con = rule.split(' -> ', 1)
    return ant.split(' + '), con.split(' + ')


def canonical_rule(rule):
    """Old executed notebooks may still display center marks until rerun."""
    left, right = rule.replace(' [C]', '').split(' -> ', 1)
    return ' + '.join(sorted(left.split(' + '))) + ' -> ' + ' + '.join(sorted(right.split(' + ')))


def binary_patches(field, antecedent, consequent, center, radius=25,
                   min_patch=2, max_one_type_share=.9):
    """Reconstruct valid center-cell transactions and A/B/A∩B masks."""
    field = field.reset_index(drop=True)
    coords = field[['x_um', 'y_um']].to_numpy()
    labels = field['cell type'].to_numpy()
    neighbors = NearestNeighbors(radius=radius).fit(coords).radius_neighbors(
        coords, return_distance=False)
    valid = np.zeros(len(field), dtype=bool)
    ant = np.zeros(len(field), dtype=bool)
    con = np.zeros(len(field), dtype=bool)
    for index, full in enumerate(neighbors):
        if len(full) < min_patch:
            continue
        _, counts = np.unique(labels[full], return_counts=True)
        if counts.max() / len(full) > max_one_type_share:
            continue
        valid[index] = True
        around = set(labels[full[full != index]])
        ant[index] = labels[index] == center and set(antecedent).difference([center]).issubset(around)
        con[index] = set(consequent).issubset(around)
    joint = ant & con
    denominator = int(valid.sum())
    support_a = ant.sum() / denominator if denominator else np.nan
    support_b = con.sum() / denominator if denominator else np.nan
    support = joint.sum() / denominator if denominator else np.nan
    confidence = support / support_a if support_a else np.nan
    lift = confidence / support_b if support_b else np.nan
    metrics = dict(valid=denominator, antecedent=int(ant.sum()), consequent=int(con.sum()),
                   joint=int(joint.sum()), Support=support, Confidence=confidence,
                   Lift=lift, Expected_support=support_a * support_b)
    return dict(valid=valid, ant=ant, con=con, joint=joint, missed=ant & ~con,
                metrics=metrics)


def _stored_variant(rules, fov, antecedent, consequent, center):
    ant = tuple(sorted([center + '_CENTER', *[name + '_NEIGHBOR'
                                               for name in antecedent if name != center]]))
    con = tuple(sorted(name + '_NEIGHBOR' for name in consequent))
    name = ' + '.join(ant) + ' -> ' + ' + '.join(con)
    rows = rules[rules.FOV.eq(fov) & rules.stored_rule.eq(name)]
    return rows.iloc[0] if len(rows) else None


def diagnostic_table(fov, rule, cells, rules, config, informative=True, mode='all'):
    antecedent, consequent = cell_types(rule)
    field = cells[cells.fov.eq(fov)]
    records, masks = [], {}
    for center in antecedent:
        result = binary_patches(field, antecedent, consequent, center)
        stored = _stored_variant(rules, fov, antecedent, consequent, center)
        metrics = result['metrics']
        status = 'not stored by mining'
        if stored is not None:
            status = ('mining FDR > cutoff' if stored.Individual_FDR > config.mining_fdr
                      else 'investigation support/confidence gate')
            if stored.Individual_FDR <= config.mining_fdr:
                passed = ((stored.Support >= config.support and stored.Confidence >= config.confidence)
                          if stored.Kind == 'attracts'
                          else stored.Expected_support >= config.expected_support)
                if passed:
                    status = 'passes investigation gate'
                    if informative and len(antecedent + consequent) >= 3 and not stored.Adds_Information:
                        status = 'filtered: shorter rule explains it'
                    elif mode == 'new' and stored.Complex_Class != 'new':
                        status = 'filtered: not new class'
                    elif mode == 'non-new' and stored.Complex_Class == 'new':
                        status = 'filtered: new class'
        records.append(dict(center=center, status=status,
                            stored_kind=stored.Kind if stored is not None else None,
                            stored_lift=stored.Lift if stored is not None else np.nan,
                            stored_fdr=stored.Individual_FDR if stored is not None else np.nan,
                            **metrics))
        masks[center] = result
    return pd.DataFrame(records), masks


def plot_patches(fov, rule, center, cells, metadata, result):
    """Large version of the verification notebook's transaction panels."""
    field = cells[cells.fov.eq(fov)].reset_index(drop=True)
    x, y = field.x_um.to_numpy(), field.y_um.to_numpy()
    set_cell_colors(cells)
    ant, con = cell_types(rule)
    fig, axes = plt.subplots(2, 4, figsize=(17, 9.2), facecolor='white')
    plot_fov(fov, '', cells, metadata, ax=axes[0, 0], show_legend=False, cell_size=22)
    plot_fov(fov, '', cells, metadata, target_ant_cells=ant, target_cons_cells=con,
             ax=axes[0, 1], show_legend=False, cell_size=22)
    masks = [('valid', 'Valid patches', '#2878D0'),
             ('ant', 'Antecedent present', '#355DB9'),
             ('con', 'Consequent present', '#9C62BC'),
             ('joint', 'Both present', '#16866A'),
             ('missed', 'Antecedent without consequent', '#D85D62')]
    for ax, (key, title, color) in zip(list(axes.flat)[2:], masks):
        ax.scatter(x, y, s=9, color='#B5B5B5', alpha=.20, linewidths=0)
        ax.scatter(x[result[key]], y[result[key]], s=23, color=color, linewidths=0)
        ax.set_title(f'{title} · {int(result[key].sum())}', fontsize=10)
        ax.set_aspect('equal', adjustable='box')
        ax.invert_yaxis()
        ax.set_xticks([])
        ax.set_yticks([])
    axes[0, 0].set_title('Full FOV', fontsize=10)
    axes[0, 1].set_title('Named cell types', fontsize=10)
    axes[1, 3].axis('off')
    fig.suptitle(f'{fov} · {rule.replace(" -> ", " → ")} · center tested: {center}', fontsize=14)
    fig.text(.5,.93,'Binary 25-µm CN reconstruction; grey = other cells. Stored mining FDR is checked separately.',
             ha='center',fontsize=9,color='#66645F')
    fig.subplots_adjust(top=.86,bottom=.06,left=.04,right=.98,hspace=.27,wspace=.15)
    plt.show()
    return fig


def compare_fields(rule, organ, cells, metadata, rules, config, stage=None, state=None,
                   informative=True, mode='all', group_col=None, groups=None):
    """Choose one hit and one eligible non-hit in the same stage, then explain both."""
    import complex_investigation as ci

    ant, con = cell_types(rule)
    names = ant + con
    counts = cells[cells['cell type'].isin(names)].groupby(['fov','cell type']).size().unstack(fill_value=0)
    eligible = counts.reindex(index=metadata.FOV, columns=names, fill_value=0).ge(config.min_cells).all(axis=1)
    raw = rules[rules.Cell_Rule.eq(rule)].copy()
    if informative and len(names) >= 3:
        raw = raw[raw.Adds_Information]
    if mode == 'new':
        raw = raw[raw.Complex_Class.eq('new')]
    elif mode == 'non-new':
        raw = raw[raw.Complex_Class.ne('new')]
    elif mode != 'all':
        raise ValueError(mode)
    gate = np.where(raw.Kind.eq('attracts'),
                    (raw.Support >= config.support) & (raw.Confidence >= config.confidence),
                    raw.Expected_support >= config.expected_support)
    passed = ci.collapse_centers(raw[(raw.Individual_FDR <= config.mining_fdr) & gate])
    states = passed.set_index('FOV').state.to_dict()
    group_col = group_col or config.score
    options = [stage] if stage else (groups or ['Severe','Control','Mild'])
    chosen = None
    for label in options:
        scope = metadata[metadata.Organ.eq(organ) & metadata[group_col].eq(label)]
        ids = [fov for fov in scope.FOV if eligible.loc[fov]]
        absent = [fov for fov in ids if states.get(fov,0)==0]
        for direction in ([state] if state is not None else [1,-1]):
            hits = [fov for fov in ids if states.get(fov,0)==direction]
            if hits and absent:
                chosen = label, direction, ids, hits, absent
                break
        if chosen is not None:
            break
    if chosen is None:
        print('No stage has both a rule-bearing and an eligible no-rule FOV under these gates.')
        return
    label, direction, ids, hits, absent = chosen
    hit_rows = passed[passed.FOV.isin(hits)]
    target = hit_rows.Lift.median()
    hit = hit_rows.iloc[(hit_rows.Lift-target).abs().argmin()].FOV
    no_rule = sorted(absent)[len(absent)//2]
    endpoint = (mode + ' informative' if mode != 'all' else 'informative') if informative else 'all passing'
    name = 'attraction' if direction == 1 else 'avoidance'
    print(f'{organ} · {label}: {len(hits)}/{len(ids)} eligible FOVs show {endpoint} {name}.')
    print(f'Comparison: rule-bearing {hit} versus eligible no-rule {no_rule}.')
    print('A missing stored row does not prove spatial independence: candidate pruning and shuffle FDR matter.')
    hit_table, hit_masks = diagnostic_table(hit, rule, cells, rules, config, informative, mode)
    no_table, no_masks = diagnostic_table(no_rule, rule, cells, rules, config, informative, mode)
    display(pd.concat({'rule FOV':hit_table,'no-rule FOV':no_table}))
    center = hit_table.sort_values('stored_fdr',na_position='last').center.iloc[0]
    no_row = no_table.set_index('center').loc[center]
    mining = json.loads((Path(dh.RESULT_CSV_PATH).parent / 'run_config.json').read_text())['settings']
    minimum = mining.get('min_lift', 1.2)
    maximum = mining.get('avoidance_max_lift', .8)
    if no_row.status == 'not stored by mining' and pd.notna(no_row.Lift):
        if maximum < no_row.Lift < minimum:
            print(f'Likely explanation: the no-rule FOV has joint patches, but reconstructed '
                  f'Lift {no_row.Lift:.2f} is near independence (mined attraction ≥{minimum:g}; '
                  f'avoidance ≤{maximum:g}). This does not rule out other mining filters.')
        else:
            print('The reconstructed metrics alone do not explain why mining omitted this variant; '
                  'candidate pruning or shuffle significance may matter.')
    else:
        print(f'No-rule FOV status for this center: {no_row.status}.')
    plot_patches(hit,rule,center,cells,metadata,hit_masks[center])
    plot_patches(no_rule,rule,center,cells,metadata,no_masks[center])
