"""Histograms of rule values, each rule coloured by whether it passed its significance test."""
import ast
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from spatial_association_rules.validation.permutations import sample_folder

import complex_investigation as ci
import data_helper as dh
import differential_vis as dv
import vis_helper as vh

P_BINS = np.logspace(-4.2, 0, 43)
METRIC_BINS = np.logspace(-1, 1, 81)
COLORS = [vh.RISING, '#5b7d99', '#aebfcd']
OPPOSITE = vh.DOWN_ARROW
SHAPES = {'pairwise': 'Pairwise', 'complex-antecedents': 'Multiple antecedents',
          'complex-consequents': 'Multiple consequents'}


def significance(rules, p_cut, max_fdr, p_column='P_Value', fdr_column='Individual_FDR'):
    """0 passed FDR, 1 p at or below the cut only, 2 p above the cut, -1 no p-value."""
    p = rules[p_column]
    return pd.Series(np.select([rules[fdr_column] <= max_fdr, p <= p_cut, p > p_cut], [0, 1, 2], -1),
                     index=rules.index)


def with_opposite(groups, opposite):
    """significance() with a new group 0 in front: passed FDR and opposite to all its significant simpler rules."""
    return (groups + 1).where(groups.ge(0), -1).mask(groups.eq(0) & opposite, 0)


def group_labels(p_cut, max_fdr):
    return [f'passed FDR ≤ {max_fdr}', f'p ≤ {p_cut}, not after FDR', f'p > {p_cut}']


def pairwise_or_complex(rules):
    names = np.where(rules.Rule_Type.eq('pairwise'), 'Pairwise', 'Complex')
    return pd.Series(pd.Categorical(names, ['Pairwise', 'Complex']), index=rules.index)


def shape_and_size(rules):
    """'Pairwise', or the complex shape and its number of items."""
    size = ' · ' + (rules.len_ant + rules.len_con).astype(str) + ' items'
    names = rules.Rule_Type.map(SHAPES) + size.where(rules.Rule_Type.ne('pairwise'), '')
    order = sorted(names.unique(), key=lambda name: (name != 'Pairwise', name))
    return pd.Series(pd.Categorical(names, order), index=rules.index)


def counts(values, panels, groups, bins):
    """One value per rule, counted into bins: {panel: counts, group x bin}."""
    return {name: np.array([np.histogram(here[groups[here.index].eq(group)], bins)[0]
                            for group in range(groups.max() + 1)])
            for name, here in values.groupby(panels, observed=True)}


def tested_by_batch(comparisons):
    """{(field, fixed cell types): ids of the rules that batch of conditional shuffles tested}."""
    tested = comparisons[comparisons.status.eq('tested')]
    fixed = tested.fixed_types.map(lambda text: tuple(ast.literal_eval(text)))
    return tested.groupby(['FOV', fixed]).rule_idx.unique().to_dict()


def shuffled_counts(rules, panels, groups, tested=None, bins=METRIC_BINS, metrics=('lift', 'conviction')):
    """Every shuffled value of every rule, counted like counts(), one field at a time.

    tested: None for each rule's own shuffles; tested_by_batch() for the conditional shuffles,
    so each batch counts only the rules it tested, not the shorter rules it measured beside them.
    Values beyond the bins are counted at the ends: lift 0 at the left, infinite conviction at the right.
    Returns {metric: {panel: counts, group x bin}}, without empty panels.
    """
    n_bins = len(bins) - 1
    code = (panels.cat.codes * 3 + groups).where(groups.ge(0), -1)
    totals = {metric: np.zeros(len(panels.cat.categories) * 3 * n_bins, dtype=np.int64)
              for metric in metrics}
    folder = Path(dh.RESULT_CSV_PATH).parent / 'permutations'
    for fov, here in rules.groupby('FOV'):
        by_rule = pd.Series(code[here.index].to_numpy(), index=here.rule_idx)
        fov_folder = sample_folder(folder, fov)
        paths = [fov_folder / 'individual.npz'] if tested is None else sorted(fov_folder.glob('conditional_*.npz'))
        for path in paths:
            shuffles = np.load(path)
            ids = shuffles['rule_idx']
            if tested is not None:
                ids = np.where(np.isin(ids, tested.get((fov, tuple(shuffles['fixed_types'])), [])), ids, -1)
            rule_code = by_rule.reindex(ids, fill_value=-1).to_numpy()
            keep = rule_code >= 0
            for metric in metrics:
                values = shuffles[metric][:, keep]
                place = np.searchsorted(bins, np.clip(values, bins[0], bins[-1]), side='right') - 1
                place = np.minimum(place, n_bins - 1) + rule_code[keep] * n_bins
                totals[metric] += np.bincount(place[~np.isnan(values)], minlength=len(totals[metric]))
    return {metric: {name: table for name, table in zip(panels.cat.categories, total.reshape(-1, 3, n_bins))
                     if table.any()}
            for metric, total in totals.items()}


def histograms(counts, bins, labels, xlabel, unit, save, line=None, colors=COLORS):
    """One stacked histogram per panel, each its own figure, saved as save_<panel>.

    counts: {title: counts, group x bin}. line: (x, text) for a dashed line.
    """
    for title, table in counts.items():
        with plt.rc_context(vh.PANEL_FONTS):
            fig, ax = plt.subplots(figsize=(vh.TEXT_WIDTH, 2.6))
            _panel(ax, title, table, bins, labels, unit, colors)
            if line:
                ax.axvline(line[0], color=vh.HAIRLINE, lw=.8, ls='--')
                ax.text(line[0], .98, f' {line[1]}', transform=ax.get_xaxis_transform(),
                        ha='left', va='top', color=vh.INK, fontsize=7)
            ax.set_xscale('log')
            ax.xaxis.set_major_formatter(lambda value, _: f'{value:g}')
            ax.set_xlabel(xlabel)
            fig.tight_layout()
            name = title.replace(' · ', '_').replace(' ', '_')
            dv._finish(fig, str(ci.ROOT / 'summary_downloads' / f'{save}_{name}.pdf'))


def field_bars(rules, groups, labels, raw, colors=COLORS):
    """Rules in each group per field, one stacked bar figure per rule size; 'not tested' from the raw rules."""
    size = rules.len_ant + rules.len_con
    in_field = pd.crosstab([size, rules.FOV], groups)
    raw = raw[(raw.len_ant + raw.len_con).isin(size)]
    in_field['not tested'] = raw.groupby([raw.len_ant + raw.len_con, 'FOV']).size() - in_field.sum(axis=1)
    for items, counts in in_field.groupby(level=0):
        counts = counts.loc[counts.sum(axis=1).sort_values(ascending=False).index]
        counts.plot.bar(stacked=True, width=1, color=colors + [vh.GRID], figsize=(12, 3), xticks=[])
        plt.legend(labels + ['not tested'], frameon=False)
        plt.title('Pairwise rules' if items == 2 else f'Rules with {items} items')
        plt.xlabel(f'fields, from most rules to fewest ({len(counts)})')
        plt.ylabel('number of raw rules')
        plt.tight_layout()
        plt.show()


def _panel(ax, title, table, bins, labels, unit, colors):
    total = table.sum()
    base = np.zeros(table.shape[1])
    for row, color in zip(table[::-1], colors[::-1]):
        ax.stairs(base + row, bins, baseline=base, fill=True, color=color)
        base = base + row
    handles = [plt.Rectangle((0, 0), 1, 1, color=color) for color in colors]
    ax.legend(handles, [f'{label}, {row.sum():,} ({row.sum() / total:.0%})' for label, row in zip(labels, table)],
              frameon=False, loc='lower left', bbox_to_anchor=(0, 1), ncol=2, borderaxespad=.2)
    ax.set_title(f'{title} · {total:,} {unit}', loc='left', pad=4 + 12 * -(-len(labels) // 2))
    ax.set_ylabel(unit)
    ax.yaxis.set_major_formatter(lambda value, _: f'{value:,.0f}')
    vh.tidy_axes(ax, grid='y')
