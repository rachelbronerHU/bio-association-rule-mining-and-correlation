"""Today's classification next to the new one, over the same occurrences.

One dot per longer rule and field: how much support it has, and how far it improves on
the shorter rule that judged it. The dots never move between panels - only their colour,
which is the verdict each classification gives them.
"""
import textwrap
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

import complex_investigation as ci
import data_helper as dh
import differential_stats as ds
import differential_vis as dv
import reclassify as rc
import rule_metrics as rm
import vis_helper as vh
from complex_vis import CLASS_COLORS
from spatial_association_rules import transactions as mt
from vis_helper import tidy_axes

TYPE_LABELS = {'ant-complex': 'Multiple antecedents', 'con-complex': 'Multiple consequents'}

# An avoiding rule is not counted on how often its cells met but on how often they could
# have, so both axes read its expected support where an attracting rule reads its support.
ACROSS_LABELS = {'occasions': {'attracts': 'support of the longer rule',
                               'avoids': 'expected support of the longer rule'},
                 'kept_share': {'attracts': "share of the shorter rule's support that is kept",
                                'avoids': "share of the shorter rule's expected support that is kept"}}
OCCASIONS = {'attracts': 'Support', 'avoids': 'Expected_support'}

LEAST = 10          # a band is widened to this many rules before anything is chosen
# The two sides of a counting patch. No cell type is drawn in either colour.
TAKING_PART = {'antecedent': '#e8590c', 'consequent': '#5f3dc4'}

# Weakest verdict first, so the rules a classification keeps are drawn on top.
OLD_LABELS = {'redundant_by_simpler': 'a shorter rule explains it',
              'consequent_driven': 'consequents explain it',
              'simpler_are_noise': 'shorter rules lack support',
              'consequent_is_noise': 'consequent link lacks support',
              'stronger_effect': 'stronger than shorter',
              'new': 'no shorter rule mined'}

NEW_COLORS = {rc.NOT_IMPROVING: CLASS_COLORS['redundant_by_simpler'],
              rc.SIMPLER_ARE_NOISE: CLASS_COLORS['simpler_are_noise'],
              rc.STRONGER: CLASS_COLORS['stronger_effect'],
              rc.NO_SHORTER: CLASS_COLORS['new']}

OLD = ('Complex_Class', CLASS_COLORS, OLD_LABELS, 'today')
NEW = ('Complex_Class_2', NEW_COLORS, {}, 'new')


def load(config):
    """The fields, their labels, and the rules with both classifications on them."""
    cells, metadata, rules = ci.load(config)
    counts = rc.transaction_counts(Path(dh.RESULT_CSV_PATH).parent)
    return cells, metadata, rc.reclassify(rules, counts, config.mining_fdr)


def dots(rules, config, gain):
    """Every longer-rule occurrence that passes the investigation gates, kept or dropped.

    One row per stored arrangement: both classifications judge each center on its own.
    `gain` says which stored metric each rule shape is read on.
    """
    rows = ci.investigation_rows(rules, config, informative=False, collapse=False)
    rows = rows[rows.n_items == 3].copy()
    rows['occasions'] = np.where(rows.Kind.eq('attracts'), rows.Support, rows.Expected_support)
    return rows.join(_against_shorter(rows, rules, config.mining_fdr, gain))


def _against_shorter(rows, rules, max_fdr, gain):
    """Each rule next to the one shorter rule that judged it, the hardest one to beat.

    The gain is read on the metric `gain` names for the rule's shape; avoidance counts
    the other way round, where the smaller value is the stronger one.
    The kept share is the longer rule's occasions over that same shorter rule's, so it can
    never pass 1. Shorter rules that missed the FDR bar judge nothing, and are used only
    when no other one was measured.
    """
    measured = {name: dict(zip(zip(rules.FOV, rules.stored_rule), rules[name]))
                for name in ('Lift', 'Conviction', 'Support', 'Expected_support',
                             'Individual_FDR')}
    found = []
    for row in rows.itertuples():
        values = measured[gain[row.Rule_Type]]
        keys = [(row.FOV, name) for name in row.Simpler_Rules_2 if (row.FOV, name) in values]
        judging = [key for key in keys if measured['Individual_FDR'][key] <= max_fdr] or keys
        if not judging:
            found.append({})
            continue
        hardest = (max if row.Kind == 'attracts' else min)(judging, key=lambda k: values[k])
        improvement = getattr(row, gain[row.Rule_Type]) - values[hardest]
        occasions = measured[OCCASIONS[row.Kind]][hardest]
        found.append(dict(gain=improvement if row.Kind == 'attracts' else -improvement,
                          kept_share=row.occasions / occasions if occasions else np.nan,
                          shorter=hardest[1], shorter_occasions=occasions,
                          shorter_value=values[hardest]))
    table = pd.DataFrame(found, index=rows.index,
                         columns=['gain', 'kept_share', 'shorter', 'shorter_occasions',
                                  'shorter_value'])
    table['gain'] = table.gain.where(np.isfinite(table.gain.to_numpy(dtype=float)))
    return table


def examples(rows, rule_type, kind, places, band, flattest_too=True):
    """The rules to look at as fields, taken from the ones the new verdict keeps.

    `places` even steps along the kept share, from its left end to its right, each giving
    the strongest gain within `band` of it, and the flattest one as well when asked.
    """
    here = rows[rows.Rule_Type.eq(rule_type) & rows.Kind.eq(kind)
                & rows.Complex_Class_2.eq(rc.STRONGER) & rows.kept_share.notna()]
    picks, taken = [], set()
    for place in np.linspace(0, 1, places):
        around = _band(here, place, band)
        wanted = [('strongest', around.gain.sort_values(ascending=False))]
        if flattest_too:
            wanted.append(('flattest', around.gain.sort_values()))
        for what, ranked in wanted:
            # Bands overlap where the rules bunch up, so each step takes the best left.
            position = next((one for one in ranked.index if one not in taken), None)
            if position is None:
                continue
            taken.add(position)
            row = here.loc[position]
            picks.append({'where': f'{place:.0%} along', 'pick': what, 'position': position,
                          'field': row.FOV, 'rule': row.Clean_Rule,
                          'shorter rule': row.shorter, 'share kept': row.kept_share,
                          'gain': row.gain, 'occasions': row.occasions})
    return pd.DataFrame(picks, index=range(1, len(picks) + 1))


def _band(here, place, band):
    """The rules around one point on the kept share, widened until it holds a few."""
    start, width = here.kept_share.min(), here.kept_share.max() - here.kept_share.min()
    away = (here.kept_share - (start + place * width)).abs()
    around = here[away <= band * width]
    return around if len(around) >= LEAST else here.loc[away.nsmallest(LEAST).index]


def show_rule(picks, index, rows, rules, cells, metadata, config, prefix):
    """One picked rule as fields, in two rows.

    Above: the whole field, the rule's cell types, the shorter rule's. Below, under each
    of those two, the cells that actually make the rule hold, in their own cell-type
    colours, with the rule's other cells in one colour that belongs to no cell type.
    """
    pick = picks.loc[index]
    longer = rows.loc[pick.position]
    shorter = rules[rules.FOV.eq(longer.FOV) & rules.stored_rule.eq(pick['shorter rule'])].iloc[0]
    info = metadata.drop_duplicates('FOV').set_index('FOV')
    fov, settings = longer.FOV, rm.current_settings()

    fig, axes = plt.subplots(2, 3, figsize=(dv._TEXT_WIDTH, 6.4), facecolor='white')
    vh.set_cell_colors(cells)
    # One colour map for every panel and the key, so the same type reads the same way.
    types = dict.fromkeys(_named_cells(longer) + _named_cells(shorter))
    colours = vh.resolve_cell_colors(list(types))
    _field(axes[0, 0], fov, cells, metadata, f'full field\n{fov}', colours)
    for column, rule in [(1, longer), (2, shorter)]:
        _field(axes[0, column], fov, cells, metadata, _caption(rule, column == 2), colours,
               ant=dh.base_items(rule.Antecedents), con=dh.base_items(rule.Consequents))
        _taking_part(axes[1, column], fov, cells, metadata, rule, settings, colours)
    axes[1, 0].set_visible(False)

    fig.suptitle(f'{info.at[fov, "Organ"]} · {longer.Cell_Rule.replace(" -> ", " → ")}',
                 fontsize=12, y=.985)
    fig.text(.5, .945, f'point {index} · {info.at[fov, config.score]}', ha='center', fontsize=9)
    _cell_key(fig, types, colours)
    fig.subplots_adjust(top=.88, bottom=.07, left=.02, right=.98, wspace=.03, hspace=.12)
    dv._finish(fig, str(ci.ROOT / 'summary_downloads' /
                        f'{prefix}_{longer.Rule_Type}_{longer.Kind}_{index}_fovs.pdf'))


def _named_cells(rule):
    """The cell types a rule names, antecedents then consequents."""
    return dh.base_items(rule.Antecedents) + dh.base_items(rule.Consequents)


def _caption(rule, is_shorter):
    """The rule's name over its metrics, as the field figures write them."""
    values = ds.rule_metrics(rule)
    name = ('shorter: ' if is_shorter else '') + rule.Cell_Rule.replace(' -> ', ' → ')
    return (f'{textwrap.fill(name, 28)}\n'
            f'lift {values["lift"]:.3g} · conf {values["conf"]:.3g} · conv {values["conv"]:.3g}\n'
            f'sup {values["sup"]:.3g} · lev {values["lev"]:.3g} · fdr {rule.Individual_FDR:.3g}')


def _field(ax, fov, cells, metadata, title, colours, ant=None, con=None):
    """One map, greyed outside the rule's cell types."""
    vh.plot_fov(fov, '', cells, metadata, target_ant_cells=ant, target_cons_cells=con,
                ax=ax, show_legend=False, cell_size=6, colors=colours)
    _bare(ax, title)


def _taking_part(ax, fov, cells, metadata, rule, settings, colours):
    """The cells of the patches where every item of the rule is present."""
    # No cell type is named, so every cell is drawn as background and nothing highlighted.
    vh.plot_fov(fov, '', cells, metadata, target_ant_cells=['none'], ax=ax,
                show_legend=False, cell_size=6, colors=colours)
    block = cells[cells.fov.eq(fov)]
    taking = _counting_cells(block, rule, settings)
    for side, colour in TAKING_PART.items():
        here = block.iloc[sorted(taking[side])]
        ax.scatter(here.x_um, here.y_um, s=6, c=colour, linewidths=0)
    counted = len(taking['antecedent'] | taking['consequent'])
    _bare(ax, f'cells that count for it\n{counted} of {len(block)} cells')


def _counting_cells(block, rule, settings):
    """Which cells make the rule hold: one set per side of the arrow.

    A patch counts when its center carries the rule's center item and every other item
    is on one of its neighbours - the same condition the mining counts support by.
    """
    coords = block[['x_um', 'y_um']].to_numpy(dtype=float)
    labels = block['cell type'].to_numpy(dtype=object)
    sides = {'antecedent': rule.ant, 'consequent': rule.con}
    wanted = {side: {mt.strip_role(item) for item in items if not mt.is_center(item)}
              for side, items in sides.items()}
    middle = {side: {mt.strip_role(item) for item in items if mt.is_center(item)}
              for side, items in sides.items()}
    center_label = next(iter(middle['antecedent'] | middle['consequent']))

    taking = {side: set() for side in sides}
    patches = mt.measure_patches(mt.find_patches(coords, settings), coords, settings)
    for patch in patches:
        if labels[patch.center] != center_label:
            continue
        if mt.is_crowded_by_one_type(labels[patch.members], settings.max_one_type_share):
            continue
        beside = set(labels[patch.neighbors])
        if not (wanted['antecedent'] | wanted['consequent']) <= beside:
            continue
        taking['antecedent' if middle['antecedent'] else 'consequent'].add(patch.center)
        for neighbor in patch.neighbors:
            for side in sides:
                if labels[neighbor] in wanted[side]:
                    taking[side].add(neighbor)
    return taking


def _bare(ax, title):
    ax.set_title(title, fontsize=6.5)
    ax.set_xlabel('')
    ax.set_ylabel('')
    ax.set_xticks([])
    ax.set_yticks([])


def _cell_key(fig, types, colours):
    """One key for the cell types and for the cells the rule is not counted on."""
    keys = [Line2D([], [], marker='o', linestyle='', markersize=5, color=colours[name],
                   label=name.replace('_', ' ')) for name in types]
    keys += [Line2D([], [], marker='o', linestyle='', markersize=5, color=colour,
                    label=f'{side}s that count') for side, colour in TAKING_PART.items()]
    fig.legend(handles=keys, loc='lower center', ncol=len(keys), frameon=False,
               fontsize=7, bbox_to_anchor=(.5, .005))


def show(rows, rule_type, kind, prefix, gain, across='occasions', mark=None):
    """One rule shape and one direction: today's verdicts above, the new ones below."""
    here = rows[rows.Rule_Type.eq(rule_type) & rows.Kind.eq(kind)]
    value = gain[rule_type].lower()
    fig, axes = plt.subplots(2, 1, figsize=(dv._TEXT_WIDTH, 7.6), sharex=True, sharey=True)
    for ax, (column, colors, labels, when) in zip(axes, [OLD, NEW]):
        _panel(ax, here, column, colors, when, across, value)
        if when == 'new':   # the picks come from the new verdict, so only it is marked
            _mark(ax, mark, across)
    axes[1].set_xlabel(ACROSS_LABELS[across][kind])

    missing = here[[across, 'gain']].isna().any(axis=1).sum()
    fig.suptitle(f'{TYPE_LABELS[rule_type]} · {kind}: {len(here) - missing} occurrences\n'
                 f'{missing} more cannot be placed: no shorter rule was measured, or it never met',
                 fontsize=10)
    fig.tight_layout(rect=(0, .20, 1, .94))
    for (column, colors, labels, when), height in zip([OLD, NEW], [.19, .09]):
        _legend(fig, colors, labels, when, height)
    name = f'{prefix}_{rule_type}_{kind}' + ('' if across == 'occasions' else '_kept')
    dv._finish(fig, str(ci.ROOT / 'summary_downloads' / f'{name}.pdf'))


def _panel(ax, here, column, colors, title, across, value):
    for name in colors:
        group = here[here[column].eq(name)]
        ax.scatter(group[across], group.gain, s=7, linewidth=0, alpha=.6,
                   color=colors[name])
    ax.axhline(0, color='#999999', linewidth=.8, zorder=0)
    ax.set_title(title, fontsize=9)
    ax.set_ylabel(f'gain in {value}', fontsize=8)
    tidy_axes(ax, grid='y')


def _mark(ax, picks, across):
    """Ring the rules the table lists, with the number the table gives them."""
    if picks is None:
        return
    x = picks['share kept'] if across == 'kept_share' else picks['occasions']
    ax.scatter(x, picks.gain, s=70, facecolor='none', edgecolor='#333333',
               linewidth=.9, zorder=5)
    for number, place in zip(picks.index, zip(x, picks.gain)):
        ax.annotate(str(number), place, textcoords='offset points', xytext=(6, 4), fontsize=7)


def _legend(fig, colors, labels, when, height):
    """One key per classification, both under the panels."""
    keys = [Line2D([], [], marker='o', linestyle='', markersize=5, color=color,
                   label=labels.get(name, name)) for name, color in colors.items()]
    fig.legend(handles=keys, title=when, loc='upper center', ncol=3, frameon=False,
               fontsize=7, title_fontsize=7.5, bbox_to_anchor=(.5, height),
               columnspacing=1.2, handletextpad=.4)
