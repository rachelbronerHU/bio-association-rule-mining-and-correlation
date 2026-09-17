"""Small adapters around the differential plots and diagnostic context panels."""
import hashlib
import textwrap

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from IPython.display import Markdown, display

import differential_stats as ds
import differential_vis as dv
import compare_rules as cr
import rule_metrics as rm
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
        nrows = 4 if cells is not None else 3
        fig, axes = plt.subplots(nrows, 1, figsize=(dv._TEXT_WIDTH, 8.5 if cells is not None else 6.6),
                                 gridspec_kw={'height_ratios':[1,1,1.1,1] if cells is not None else [1,1.1,1],
                                              'hspace':.72})
        dv._prevalence_panel(axes[0], states, eligible, shown, plot_spec, groups, bars=pooled)
        axes[0].set_xlim(-.5, len(groups)-.5)
        metric_state = {'attracts':1,'avoids':-1}[spec['kind']] if analysis.get('test_metric')=='Lift' else None
        metric_axis = 2 if cells is not None else 1
        state_axis = 3 if cells is not None else 2
        if cells is not None:
            _abundance_panel(axes[1], cells, eligible, shown, plot_spec, groups)
        dv._metric_panel(axes[metric_axis], rows, eligible, shown, plot_spec, groups,
                         state=metric_state, connect=not pooled)
        counts = dv._stage_state_counts(states, eligible, shown, spec['rule'], spec['organ'], config.score, groups)
        dv._draw_state_bars(axes[state_axis], *counts, groups, show_net=False)
        titles = ['how often does this complex rule fire?']
        if cells is not None:
            titles.append('what cell types are present? (diagnostic, not a finding)')
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
        endpoint = analysis['mode']+' informative' if analysis.get('informative',True) else 'all passing'
        subtitle = (f'{config.score} · ≥{config.min_cells} cells/type · {endpoint} occurrences\n'
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
        candidate = candidate[(candidate.Individual_FDR <= config.mining_fdr) & gate]
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
            fdr=np.nan if here is None else here.Individual_FDR,
        ))
    frame = pd.DataFrame(rows)
    frame.attrs['eligible_n'] = examples.attrs.get('eligible_n', {})
    return frame


def parent_eligibility(analysis, spec, parent):
    """The complex rule's own eligible fields, under the parent's name.

    The two rules are only comparable over the same fields, so the parent is never
    given a denominator of its own here.
    """
    return analysis['eligible'].loc[[spec['rule']]].rename(index={spec['rule']: parent})


def parent_prevalence(analysis, all_rules, metadata, spec, save=None):
    """One fixed, most frequently best measured parent; not a changing per-FOV union."""
    config = analysis['config']
    parent, grouped, matched_fovs = fixed_parent(analysis, all_rules, metadata, spec)
    if parent is None:
        return
    eligible = parent_eligibility(analysis, spec, parent)
    states = pd.DataFrame(0,index=[parent],columns=eligible.columns,dtype='int8')
    if len(grouped):
        states.loc[parent,grouped.FOV] = grouped.state.to_numpy()
    fig,ax = plt.subplots(figsize=(dv._TEXT_WIDTH,3.15))
    dv._prevalence_panel(ax,states,eligible,metadata,
                         dict(organ=spec['organ'],rule=parent,score=config.score),ci.STAGES)
    ax.set_title('How often does one fixed measured shorter rule fire?',loc='left',fontsize=9)
    fig.suptitle(f'{spec["organ"]} · {parent.replace(" -> "," → ")}',fontsize=11)
    fig.text(.5,.88,'Same FOV eligibility as the complex rule; one parent identity across all stages',
             ha='center',fontsize=7,color='#706E68')
    fig.subplots_adjust(top=.76,bottom=.23,left=.15,right=.77)
    dv._finish(fig,save)
    print(f'Fixed parent: {parent}; most often the best measured shorter rule in '
          f'{matched_fovs} matched FOVs.')


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
    summary(analysis,metadata,spec,filename(prefix,spec,'summary'),pooled,cells)
    parent_prevalence(analysis,rules,metadata,spec,filename(prefix,spec,'parent_prevalence'))
    if parents:
        parent_comparison(analysis,rules,metadata,spec,filename(prefix,spec,'parents'))
    if maps:
        config = analysis['config']
        examples = ds.representative_fovs(analysis['rows'],analysis['eligible'],metadata,spec['rule'],
                                          spec['organ'],config.score,ci.STAGES,'Lift')
        # The complex rule and its parent are explained over the same fields, so
        # the patches of each field are built once and shared.
        fields = rm.Fields(cells)
        examples = rm.explain_missing(examples, cells, fields=fields)
        parent, _, _ = fixed_parent(analysis, rules, metadata, spec, quiet=True)
        shorter = parent_in_same_fields(rules, examples, parent) if parent else pd.DataFrame()
        if len(shorter):
            shorter = rm.explain_missing(shorter, cells, fields=fields)
        cr.plot_rule_fovs(examples,ci.STAGES,cells,metadata,spec['organ'],config.score,
                          min_cells=config.min_cells,max_fdr=config.mining_fdr,
                          others=[(f'shorter: {parent.replace(" -> "," → ")}', shorter)]
                                 if len(shorter) else (),
                          save=filename(prefix,spec,'fovs'))


def show_threshold(index, specs, baseline, stricter, metadata, prefix='support', cells=None):
    if not 0 <= index < len(specs):
        print('No comparison at this index.')
        return
    spec = specs.iloc[index].to_dict()
    # The same exact rule and denominator at both cutoffs: do not compare two different top lists.
    summary(baseline,metadata,spec,filename(prefix+'_001',spec,'summary'),pooled=True,cells=cells)
    summary(stricter,metadata,spec,filename(prefix+'_003',spec,'summary'),pooled=True,cells=cells)


def show_denominators(index, specs, analysis, metadata, prefix='denominators'):
    if not 0 <= index < len(specs):
        print('No denominator comparison at this index.')
        return
    spec = specs.iloc[index].to_dict()
    table = analysis['counts']
    block = table[(table.rule == spec['rule']) & (table.organ == spec['organ'])
                  & (table.kind == spec['kind']) & table.stage.isin(ci.STAGES)].set_index('stage').loc[ci.STAGES]
    fig,ax = plt.subplots(figsize=(dv._TEXT_WIDTH,3.7))
    for offset,denom,color,label in [(-.15,'total','#B7B4AE','all FOVs'),(.15,'eligible','#2878D0','eligible FOVs')]:
        ax.bar(np.arange(3)+offset,100*block.hits/block[denom].replace(0,np.nan),width=.28,color=color,label=label)
    ax.set_xticks(range(3),[f'{s}\n{r.hits} | {r.eligible} | {r.total}' for s,r in block.iterrows()])
    ax.set_xlabel('rule-bearing eligible | eligible | total FOVs',fontsize=8)
    ax.set_ylabel('prevalence (%)')
    ax.set_ylim(0,100)
    ax.legend(frameon=False)
    dv.tidy_axes(ax,grid='y',hide=('top','right'))
    fig.suptitle(textwrap.fill(f'{spec["organ"]} · {spec["rule"]}',78),fontsize=11)
    ax.set_title('Same eligible numerator; only the denominator changes',fontsize=9)
    fig.tight_layout(rect=(0,0,1,.85))
    dv._finish(fig,filename(prefix,spec,'summary'))


def show_classes(kind, rules, config, prefix):
    import complex_vis as cv
    passed = ci.investigation_rows(rules,config,informative=False)
    counts = passed[passed.Kind.eq(kind)].groupby(['Rule_Type','Complex_Class']).size().unstack(fill_value=0)
    labels = {'ant-complex':'multiple antecedents','con-complex':'multiple consequents',
              'both-complex':'both sides complex'}
    classes = {'redundant_by_simpler':'a shorter rule explains it','consequent_driven':'consequents already explain it',
               'simpler_are_noise':'shorter rules lack support','consequent_is_noise':'consequent link lacks support',
               'stronger_effect':'stronger than shorter rules','new':'no shorter rule mined'}
    counts = counts.reindex(columns=list(classes),fill_value=0)
    cv.plot_class_split(counts,labels=labels,class_labels=classes,params=kind,
                       title='How stored longer-rule occurrences are classified',
                       unit_label='FOV-level occurrences',
                       subtitle='Three/four distinct cell types · passing investigation gates · one count per FOV',
                       save=str(ci.ROOT/'summary_downloads'/f'{prefix}_{kind}_classes.pdf'))
