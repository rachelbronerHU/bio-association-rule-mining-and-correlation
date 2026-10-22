"""Small adapters around the differential plots and diagnostic context panels."""
import hashlib
import textwrap

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from IPython.display import Markdown, display
from matplotlib.lines import Line2D

import differential_stats as ds
import differential_vis as dv
import compare_rules as cr
import data_helper as dh
import rule_metrics as rm
import vis_helper as vh
from vis_helper import ATTRACTION, AVOIDANCE
import complex_investigation as ci


def filename(prefix, spec, panel):
    identity = spec['organ'] + spec['rule'] + spec.get('kind', '')
    tag = hashlib.sha1(identity.encode()).hexdigest()[:10]
    return str(ci.ROOT / 'summary_downloads' / f'{prefix}_{spec["organ"]}_{tag}_{panel}.pdf')


def _abundance_panel(ax, cells, eligible, metadata, spec, groups):
    """Cell composition is a diagnostic for the rule, never a selected finding."""
    types = sorted(set(spec['rule'].replace(' -> ', ' + ').split(' + ')))
    shares = dv._cell_share_table(cells, metadata.drop_duplicates('FOV'), types, []).set_index('FOV')
    colors = plt.get_cmap('tab10').colors
    spread = np.random.default_rng(13)
    offsets = np.linspace(-.14, .14, len(types))
    for number, (name, offset) in enumerate(zip(types, offsets)):
        means = []
        for position, (_, fovs) in enumerate(dv._eligible_by_stage(eligible, metadata, spec, groups)):
            values = shares.reindex(fovs)[name].fillna(0)
            means.append(values.mean() if len(values) else np.nan)
            ax.scatter(position + offset + spread.uniform(-.022, .022, len(values)),
                       values, s=12, alpha=.4, color=colors[number], linewidths=0)
        ax.plot(np.arange(len(groups)) + offset, means, marker='D', ms=4, lw=1.5,
                color=colors[number], label=name.replace('_', ' '))
    ax.set_xlim(-.5, len(groups)-.5)
    ax.set_xticks(range(len(groups)), groups)
    ax.set_ylim(bottom=0)
    ax.set_ylabel('cells in eligible\nFOVs (%)')
    ax.legend(frameon=False, fontsize=7, loc='center left', bbox_to_anchor=(1.01,.5))
    dv.tidy_axes(ax, grid='y', hide=('top','right'))


def summary(analysis, metadata, spec, save=None, pooled=False, cells=None):
    config = analysis['config']
    groups = ['All', *ci.STAGES] if pooled else ci.STAGES
    shown = pd.concat([metadata.assign(**{config.score:'All'}), metadata]) if pooled else metadata
    plot_spec = dict(spec, score=config.score)
    states, eligible, rows = (analysis[key] for key in ['states','eligible','rows'])
    with plt.rc_context(dv._PANEL_FONTS):
        # How often the rule fires is drawn by parent_vs_complex, with the parent
        # beside it, so it is not repeated here.
        nrows = 3 if cells is not None else 2
        fig, axes = plt.subplots(nrows, 1, figsize=(dv._TEXT_WIDTH, 6.9 if cells is not None else 5.0),
                                 gridspec_kw={'height_ratios':[1,1.1,1] if cells is not None else [1.1,1],
                                              'hspace':.72})
        metric_state = {'attracts':1,'avoids':-1}[spec['kind']] if analysis.get('test_metric')=='Lift' else None
        metric_axis = 1 if cells is not None else 0
        state_axis = 2 if cells is not None else 1
        if cells is not None:
            _abundance_panel(axes[0], cells, eligible, shown, plot_spec, groups)
        dv._metric_panel(axes[metric_axis], rows, eligible, shown, plot_spec, groups,
                         state=metric_state, connect=not pooled)
        counts = dv._stage_state_counts(states, eligible, shown, spec['rule'], spec['organ'], config.score, groups)
        dv._draw_state_bars(axes[state_axis], *counts, groups, show_net=False)
        titles = ['what cell types are present? (diagnostic, not a finding)'] if cells is not None else []
        titles += ['how strong is it? (Lift)', 'every FOV, including ineligible fields']
        for ax,title in zip(axes, titles):
            ax.set_title(title, loc='left', color='#5F5D58', pad=6)
            if pooled:
                ax.axvline(.5, color='#B7B4AE', lw=.8, ls=':')
        axes[metric_axis].set_xlabel('rule-bearing / eligible FOVs · dots = FOVs · diamonds = medians', fontsize=7)
        handles = [dv.Line2D([0],[0], marker='s', linestyle='none', markersize=6,
                             color=dv._STATE_COLORS[n], label=n) for n in dv._STATE_ORDER]
        axes[state_axis].legend(handles=handles, ncol=1, frameon=False, loc='center left',
                       bbox_to_anchor=(1.01,.5), fontsize=7)
        title = f'{spec["organ"]} · {spec["rule"].replace(" -> ", " → ")}'
        fig.suptitle(textwrap.fill(title, 78), fontsize=11, y=.995)
        subtitle = (f'{config.score} · rule can be tested · passing occurrences\n'
                    f'attraction support ≥{config.support:g}; avoidance expected support ≥{config.expected_support:g}')
        fig.text(.5,.927,subtitle,ha='center',va='top',fontsize=7,color='#706E68')
        fig.align_ylabels(axes)
        fig.subplots_adjust(top=.87 if cells is not None else .835,bottom=.075 if cells is not None else .095,
                            left=.14,right=.76)
        dv._finish(fig, save)


def fixed_parent(analysis, all_rules, metadata, spec, quiet=False):
    """The one shorter rule the comparison is against: most often the best measured one.

    Returns (name, rows, matched_fovs). The rows are collapsed over center variants,
    so `Clean_Rule` is the parent name and there is one row per field.
    """
    matched = ci.matched_parents(spec['rule'], spec.get('kind','attracts'),
                                 analysis['rows'], all_rules)
    matched = matched[matched.FOV.isin(metadata.loc[metadata.Organ.eq(spec['organ']),'FOV'])]
    if matched.empty:
        if not quiet:
            print('No measured shorter rule for a prevalence comparison.')
        return None, None, 0
    config = analysis['config']
    source = pd.DataFrame()
    for parent in matched.parent.value_counts().index:
        candidate = all_rules[all_rules.Cell_Rule.eq(parent)].copy()
        gate = np.where(candidate.Kind.eq('attracts'),
                        (candidate.Support >= config.support) & (candidate.Confidence >= config.confidence),
                        candidate.Expected_support >= config.expected_support)
        candidate = candidate[(dh.rule_fdr(candidate) <= config.mining_fdr) & gate]
        if candidate.FOV.isin(metadata.loc[metadata.Organ.eq(spec['organ']),'FOV']).any():
            source = candidate
            break
    if source.empty:
        if not quiet:
            print('No shorter rule passes the same investigation gates in this organ.')
        return None, None, 0
    return parent, ci.collapse_centers(source), matched[matched.parent.eq(parent)].FOV.nunique()


def parent_comparison(analysis, all_rules, metadata, spec, save=None):
    """Compare actual matched FOVs; zero Lift stays zero, never a clipped huge gain."""
    matched = ci.matched_parents(spec['rule'], spec.get('kind','attracts'), analysis['rows'], all_rules)
    scope = metadata[metadata.Organ.eq(spec['organ'])].set_index('FOV')
    matched = matched[matched.FOV.isin(scope.index)]
    matched = matched[matched.FOV.map(analysis['eligible'].loc[spec['rule']]).fillna(False)]
    matched['stage'] = matched.FOV.map(scope[analysis['config'].score])
    if matched.empty:
        print('No measured same-FOV parent: no numerical gain is assigned.')
        return
    fig, ax = plt.subplots(figsize=(dv._TEXT_WIDTH, 3.5))
    counts = []
    for i,stage in enumerate(ci.STAGES):
        block = matched[matched.stage.eq(stage)]
        counts.append(len(block))
        offsets = np.linspace(-.08,.08,len(block))
        for offset,(_,row) in zip(offsets,block.iterrows()):
            ax.plot([i-.15+offset,i+.15+offset], [row.parent_lift,row.complex_lift],
                    color='#B7B4AE', lw=.7, alpha=.6)
        for col,offset,color,label in [('parent_lift',-.15,'#9A8D70','strongest stored parent'),
                                       ('complex_lift',.15,'#2878D0','complex rule')]:
            ax.scatter(i+offset+offsets,block[col],s=18,alpha=.65,color=color,
                       label=label if i==0 else None)
            if len(block):
                ax.scatter(i+offset,block[col].median(),marker='D',s=65,color=color,edgecolor='white',zorder=4)
    ax.axhline(1,color='#AAAAAA',lw=1)
    ax.set_xticks(range(3),[f'{s}\n{n} matched FOVs' for s,n in zip(ci.STAGES,counts)])
    ax.set_ylabel('Lift · matched fields only')
    ax.set_xlim(-.5,2.5)
    dv.tidy_axes(ax,grid='y',hide=('top','right'))
    ax.legend(frameon=False,fontsize=8)
    title = f'{spec["organ"]} · {spec["rule"].replace(" -> "," → ")}'
    fig.suptitle(textwrap.fill(title,78),fontsize=11)
    ax.set_title('Does the complex rule improve on a measured shorter rule?',fontsize=9,pad=12)
    fig.tight_layout(rect=(0,0,1,.85))
    dv._finish(fig,save)
    display(matched.groupby('stage',sort=False).agg(matched_fovs=('FOV','nunique'),
             median_complex_lift=('complex_lift','median'),median_parent_lift=('parent_lift','median'),
             significant_parents=('parent_fdr',lambda x:int((x<=analysis['config'].mining_fdr).sum()))))


def parent_in_same_fields(all_rules, examples, parent):
    """The shorter rule measured in the fields the complex rule is drawn in.

    Its cell types come from its own definition, never from the complex rule, so
    only its own cells are highlighted. Where it was not mined in one of those
    fields the row is left at state 0, and `explain_missing` fills in the gate it
    misses.
    """
    carried = examples[examples.state.ne(0)]
    occurrences = ci.collapse_centers(all_rules[all_rules.Cell_Rule.eq(parent)])
    if carried.empty or occurrences.empty:
        return pd.DataFrame()

    definition = occurrences.iloc[0]
    antecedents, consequents = definition['Antecedents'], definition['Consequents']
    identity = dict(rule=parent, antecedent_items=antecedents, consequent_items=consequents,
                    antecedent_cells=ci.dh.base_items(antecedents),
                    consequent_cells=ci.dh.base_items(consequents))
    # Every field the parent was mined in, gates or not: a rule that was measured
    # and then dropped is not the same as one that was never there.
    mined = occurrences.set_index('FOV')

    rows = []
    for _, example in carried.iterrows():
        here = mined.loc[example.FOV] if example.FOV in mined.index else None
        if isinstance(here, pd.DataFrame):
            here = here.iloc[0]
        rows.append(dict(
            identity, stage=example.stage, FOV=example.FOV,
            kind=example.get('kind', 'attracts'),
            state=0 if here is None else int(here.state),
            metrics={} if here is None else ds.rule_metrics(here),
            fdr=np.nan if here is None else dh.rule_fdr(here),
        ))
    frame = pd.DataFrame(rows)
    frame.attrs['eligible_n'] = examples.attrs.get('eligible_n', {})
    return frame


_PARENT_GREY = '#777777'
_FIELD_COLORS = {'attraction': ATTRACTION, 'avoidance': AVOIDANCE}
_FIELD_LABELS = {'attraction': 'Attraction (%)', 'avoidance': 'Avoidance (%)'}
_COUNT_WORDS = {'attraction': 'attracted', 'avoidance': 'avoided', 'eligible': 'eligible'}


def arrow(rule):
    return rule.replace(' -> ', ' → ')


def parent_counts(analysis, metadata, spec, parent, grouped):
    """Per-stage counts for the complex rule and its parent, over the same fields.

    `analysis['states']` holds only the complex rules, so the parent's row is built
    from its own passing occurrences. Both shares use the FOVs where both rules
    are testable.
    """
    fovs = analysis['states'].columns
    states = pd.DataFrame(0, index=[spec['rule'], parent], columns=fovs, dtype='int8')
    states.loc[spec['rule']] = analysis['states'].loc[spec['rule']].reindex(
        fovs, fill_value=0).to_numpy()
    if grouped is not None and len(grouped):
        states.loc[parent, grouped.FOV] = grouped.state.to_numpy()
    fields = analysis['fields']
    eligible = pd.concat([
        analysis['eligible'].loc[[spec['rule']]],
        ci.cell_rule_testable_fovs([parent], fields.cells, metadata, analysis['config'], fields=fields),
    ])
    return ci.stage_counts(dict(states=states, eligible=eligible, metadata=metadata,
                                config=analysis['config']),
                           spec['organ'], spec['rule'], parent)


def parent_vs_complex(table, organ, rule, parent, save=None):
    """One panel per direction, the shorter rule and the complex one in each.

    Both rules are counted where both rules are testable, so the two
    lines share a denominator. A direction the complex rule never takes is not
    drawn at all, and its count is left off the axis.
    """
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

    fig.suptitle(f'{organ} · {arrow(rule)}', fontsize=12, y=1 - .28 / height)
    fig.text(.5, 1 - .55 / height, 'Complex rule versus its shorter parent',
             ha='center', va='top', fontsize=9.5)
    fig.text(.5, 1 - .76 / height, f'Parent: {arrow(parent)} · % of FOVs testable for both rules',
             ha='center', va='top', fontsize=8, color='#706E68')
    fig.subplots_adjust(top=1 - 1.0 / height, bottom=.95 / height, left=.13, right=.80)
    dv._finish(fig, save)


def show_selected(index, specs, analysis, rules, cells, metadata, prefix,
                  pooled=False, parents=False, maps=True):
    if not 0 <= index < len(specs):
        print(f'No result at index {index}; {len(specs)} selected results.')
        return
    spec = specs.iloc[index].to_dict()
    display(Markdown(f'**[{index}] {spec["organ"]} · {spec["rule"]}**'))
    if spec.get('claim'):
        display(Markdown(spec['claim'] + '\n\n' + spec.get('caveat','') + '\n\n' + spec.get('source','')))
    counts = analysis['counts']
    block = counts[(counts.rule == spec['rule']) & (counts.organ == spec['organ'])]
    display(block[['stage','kind','hits','eligible','total','hit_patients','patients']].reset_index(drop=True))
    parent, grouped, _ = fixed_parent(analysis, rules, metadata, spec)
    if parent:
        parent_vs_complex(parent_counts(analysis, metadata, spec, parent, grouped),
                          spec['organ'], spec['rule'], parent,
                          filename(prefix, spec, 'parent_vs_complex'))
    summary(analysis,metadata,spec,filename(prefix,spec,'summary'),pooled,cells)
    if parents:
        parent_comparison(analysis,rules,metadata,spec,filename(prefix,spec,'parents'))
    if maps:
        config = analysis['config']
        examples = ds.representative_fovs(analysis['rows'],analysis['eligible'],metadata,spec['rule'],
                                          spec['organ'],config.score,ci.STAGES,'Lift')
        # The complex rule and its parent are explained over the same fields, so
        # the patches of each field are built once and shared.
        fields = analysis['fields']
        examples = rm.explain_missing(examples, cells, fields=fields)
        shorter = parent_in_same_fields(rules, examples, parent) if parent else pd.DataFrame()
        if len(shorter):
            shorter = rm.explain_missing(shorter, cells, fields=fields)
        cr.plot_rule_fovs(examples,ci.STAGES,cells,metadata,spec['organ'],config.score,
                          max_fdr=config.mining_fdr,
                          others=[(f'shorter: {parent.replace(" -> "," → ")}', shorter)]
                                 if len(shorter) else (),
                          save=filename(prefix,spec,'fovs'), fields=fields)


def show_rule(picks, index, rows, rules, cells, metadata, config, prefix):
    """One picked rule as fields, in two rows.

    Above: the whole field, the rule's cell types, the shorter rule's. Below, under each
    of those two, the cells that actually make the rule hold, coloured by the part they
    play in it: the center, the other antecedents, the consequents.
    """
    pick = picks.loc[index]
    longer = rows.loc[pick.position]
    shorter = rules[rules.FOV.eq(longer.FOV) & rules.Cell_Rule.eq(pick['shorter rule'])].iloc[0]
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
        _taking_part(axes[1, column], fov, cells, metadata, rule, settings)
    axes[1, 0].set_visible(False)

    fig.suptitle(f'{info.at[fov, "Organ"]} · {longer.Cell_Rule.replace(" -> ", " → ")}',
                 fontsize=12, y=.985)
    fig.text(.5, .945, f'point {index} · {info.at[fov, config.score]}', ha='center', fontsize=9)
    _cell_key(fig, types, colours, [rm.rule_parts(one.ant, one.con)
                                    for one in (longer, shorter)])
    fig.subplots_adjust(top=.88, bottom=.11, left=.02, right=.98, wspace=.03, hspace=.12)
    dv._finish(fig, str(ci.ROOT / 'summary_downloads' /
                        f'{prefix}_{longer.Kind}_{index}_fovs.pdf'))


def _named_cells(rule):
    """The cell types a rule names, antecedents then consequents."""
    return dh.base_items(rule.Antecedents) + dh.base_items(rule.Consequents)


def _caption(rule, is_shorter):
    """The rule's name over its metrics, as the field figures write them."""
    values = ds.rule_metrics(rule)
    name = ('shorter: ' if is_shorter else '') + rule.Cell_Rule.replace(' -> ', ' → ')
    fdr = '' if is_shorter else f' · fdr {dh.rule_fdr(rule):.3g}'
    return (f'{textwrap.fill(name, 28)}\n'
            f'lift {values["lift"]:.3g} · conf {values["conf"]:.3g} · conv {values["conv"]:.3g}\n'
            f'sup {values["sup"]:.3g} · lev {values["lev"]:.3g}{fdr}')


def _field(ax, fov, cells, metadata, title, colours, ant=None, con=None):
    """One map, greyed outside the rule's cell types."""
    vh.plot_fov(fov, '', cells, metadata, target_ant_cells=ant, target_cons_cells=con,
                ax=ax, show_legend=False, cell_size=6, colors=colours)
    _bare(ax, title)


def _taking_part(ax, fov, cells, metadata, rule, settings):
    """The cells of the patches where every item of the rule is present."""
    counted = rm.counted_cells(rule.ant, rule.con, cells, fov, settings)
    vh.plot_counted_cells(ax, fov, cells, metadata, counted,
                          rm.rule_parts(rule.ant, rule.con), 6)
    _bare(ax, f'cells that count for it\n'
              f'{counted.total} of {cells.fov.eq(fov).sum()} cells')


def _bare(ax, title):
    ax.set_title(title, fontsize=6.5)
    ax.set_xlabel('')
    ax.set_ylabel('')
    ax.set_xticks([])
    ax.set_yticks([])


def _cell_key(fig, types, colours, parts):
    """One key for the cell types, one for the part each plays where it counts."""
    def dot(colour, label):
        return Line2D([], [], marker='o', linestyle='', markersize=5, color=colour,
                      label=label)

    for keys, title, above in (
            ([dot(colours[name], name.replace('_', ' ')) for name in types],
             'Cell type', .048),
            ([dot(colour, label) for colour, label in vh.role_key(parts)],
             None, .005)):
        fig.legend(handles=keys, title=title, loc='lower center', ncol=len(keys),
                   frameon=False, fontsize=7, title_fontsize=7.5,
                   bbox_to_anchor=(.5, above))
