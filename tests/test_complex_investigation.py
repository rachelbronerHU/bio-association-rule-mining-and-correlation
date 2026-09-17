"""Small correctness checks for investigation filters, roles and denominators."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT/'result_summary'),str(ROOT/'result_summary/differential_rules'),
               str(ROOT/'result_summary/complex_rules')]
import complex_investigation as ci
import complex_hypotheses as hypotheses


def test_expected_support_keeps_zero_lift_avoidance():
    rows=pd.DataFrame(dict(n_items=[3,3,3],Individual_FDR=[.01]*3,Adds_Information=[True]*3,
                           Kind=['attracts','avoids','avoids'],Support=[.02,0.,0.],
                           Expected_support=[.015,.04,.02],Confidence=[.8,0.,0.],Complex_Class=['new']*3))
    selected=ci.investigation_rows(rows,ci.Config(support=.03,expected_support=.03))
    assert selected.index.tolist()==[1]


def test_center_variants_share_one_reported_rule_but_keep_arrow_direction():
    one=ci.rule_name(('A_CENTER','B_NEIGHBOR'),('C_NEIGHBOR',))
    two=ci.rule_name(('A_NEIGHBOR','B_CENTER'),('C_NEIGHBOR',))
    assert one!=two and '[C]' in one and '[C]' in two
    assert ci.cell_rule_name(('A_CENTER','B_NEIGHBOR'),('C_NEIGHBOR',)) == 'A + B -> C'
    assert ci.cell_rule_name(('A_NEIGHBOR','B_CENTER'),('C_NEIGHBOR',)) == 'A + B -> C'
    assert ci.cell_rule_name(('C_CENTER',),('A_NEIGHBOR','B_NEIGHBOR')) == 'C -> A + B'


def test_center_variants_average_metrics_once_per_fov():
    rows=pd.DataFrame([
        dict(FOV='f1',Cell_Rule='A + B -> C',Clean_Rule='A [C] + B -> C',
             Lift=1.4,Support=.02,Confidence=.6,Leverage=.01,Expected_support=.01,
             Individual_FDR=.02,Kind='attracts',state=1,Adds_Information=True,
             Simpler_Rules="['A_CENTER -> C_NEIGHBOR']",Complex_Class='stronger_effect'),
        dict(FOV='f1',Cell_Rule='A + B -> C',Clean_Rule='A + B [C] -> C',
             Lift=2.,Support=.04,Confidence=.8,Leverage=.03,Expected_support=.01,
             Individual_FDR=.04,Kind='attracts',state=1,Adds_Information=False,
             Simpler_Rules="['B_CENTER -> C_NEIGHBOR']",Complex_Class='redundant_by_simpler'),
        dict(FOV='f1',Cell_Rule='C -> A + B',Clean_Rule='C [C] -> A + B',
             Lift=.5,Support=.02,Confidence=.5,Leverage=-.01,Expected_support=.03,
             Individual_FDR=.01,Kind='avoids',state=-1,Adds_Information=True,
             Simpler_Rules='[]',Complex_Class='new'),
    ])
    grouped=ci.collapse_centers(rows)
    assert len(grouped)==2
    forward=grouped[grouped.Clean_Rule.eq('A + B -> C')].iloc[0]
    assert forward.n_centers==2 and np.isclose(forward.Lift,1.7)
    assert np.isclose(forward.Support,.03)
    assert forward.Individual_FDR==.02
    assert len(forward.Simpler_Rules)==2
    assert grouped[grouped.Clean_Rule.eq('C -> A + B')].iloc[0].state==-1


def test_mixed_center_states_use_mean_lift_without_merging_reverse_arrow():
    rows=pd.DataFrame([
        dict(FOV='f1',Cell_Rule='A + B -> C',Clean_Rule='A [C] + B -> C',
             Lift=1.6,Individual_FDR=.01,state=1,Kind='attracts',
             Simpler_Rules='[]',Adds_Information=True),
        dict(FOV='f1',Cell_Rule='A + B -> C',Clean_Rule='A + B [C] -> C',
             Lift=.6,Individual_FDR=.02,state=-1,Kind='avoids',
             Simpler_Rules='[]',Adds_Information=True),
        dict(FOV='f1',Cell_Rule='C -> A + B',Clean_Rule='C [C] -> A + B',
             Lift=.5,Individual_FDR=.03,state=-1,Kind='avoids',
             Simpler_Rules='[]',Adds_Information=True),
    ])
    grouped=ci.collapse_centers(rows)
    assert len(grouped)==2
    forward=grouped[grouped.Cell_Rule.eq('A + B -> C')].iloc[0]
    assert np.isclose(forward.Lift,1.1)
    assert forward.state==1 and forward.Kind=='attracts'
    assert forward.center_states==(-1,1)
    assert grouped[grouped.Cell_Rule.eq('C -> A + B')].iloc[0].state==-1


def test_patient_mean_weights_biopsies_equally_and_omits_cross_stage():
    meta=pd.DataFrame(dict(FOV=['a','b','c','d','e'],Biopsy=['b1','b1','b2','b3','b4'],
                           PatientID=['p1','p1','p1','p2','p2'],Organ=['Colon']*5,
                           stage=['Control','Control','Control','Mild','Severe']))
    values=pd.DataFrame([[1.,1.,0.,1.,0.]],index=['r'],columns=meta.FOV)
    result,_,omitted=ci.independent_values(values,meta,'Colon','stage','PatientID')
    assert result.at['r','p1']==.5
    assert omitted==1 and 'p2' not in result


def test_absent_registered_rule_has_eligibility_and_zero_state():
    rules=pd.DataFrame(dict(n_items=[3],Clean_Rule=['r'],types=[('A','B','C')]))
    rows=pd.DataFrame(columns=['Clean_Rule','FOV','state'])
    cells=pd.DataFrame({'cell type':['A','B','C','A','B'],'fov':['f1']*3+['f2']*2})
    meta=pd.DataFrame({'FOV':['f1','f2']})
    state,eligible=ci.matrices(rules,rows,cells,meta,ci.Config(min_cells=1),{'absent':('A','B','C')})
    assert state.loc['absent'].tolist()==[0,0]
    assert eligible.loc['absent'].tolist()==[True,False]


def test_known_registry_checks_all_centers_and_contextual_reverse():
    specs=hypotheses.known_specs()
    assert len(specs)==27
    for _,group in specs.groupby(['niche','organ']):
        assert len(group[group.family=='centered'])==len(group.iloc[0].types)
        assert len(group[group.family=='contextual'])==2


def test_visible_effect_can_survive_nonsignificant_patient_fdr():
    counts=pd.DataFrame([dict(organ='Colon',rule='r',kind='attracts',stage=s,hits=h,
                             hit_patients=p,eligible=n,patients=10,share=h/n)
                         for s,h,p,n in [('All',20,8,60),('Control',15,6,20),
                                         ('Mild',4,3,20),('Severe',1,1,20)]])
    tests=pd.DataFrame([dict(organ='Colon',rule='r',kind='attracts',unit='PatientID',fdr=.8)])
    selected=ci.candidates(counts,tests,ci.Config())
    assert selected.iloc[0].stream=='visible'
    assert np.isclose(selected.iloc[0].gap,.7)


def test_measured_parent_is_same_field_and_same_kind():
    rows=pd.DataFrame([dict(FOV='f1',Clean_Rule='complex',Kind='avoids',Lift=0.,
                            Simpler_Rules="['parent']")])
    parents=pd.DataFrame([dict(FOV=fov,Kind=kind,stored_rule='parent',Clean_Rule='parent',
                               Lift=lift,Individual_FDR=.01)
                          for fov,kind,lift in [('f1','avoids',.4),('f2','avoids',.1),('f1','attracts',2.)]])
    matched=ci.matched_parents('complex','avoids',rows,parents)
    assert matched.parent_lift.tolist()==[.4]
    assert matched.complex_lift.tolist()==[0.]
    absent=rows.assign(Simpler_Rules="['missing']")
    assert ci.matched_parents('complex','avoids',absent,parents).empty


def test_configured_lift_screen_accepts_real_zero_values():
    meta=pd.DataFrame(dict(FOV=[f'f{i}' for i in range(6)],Biopsy=[f'b{i}' for i in range(6)],
                           PatientID=[f'p{i}' for i in range(6)],Organ=['Colon']*6,
                           stage=['Control']*3+['Severe']*3))
    values=pd.DataFrame([[0.,0.,0.,.4,.4,.4]],index=['r'],columns=meta.FOV)
    counts=pd.DataFrame([dict(organ='Colon',rule='r',kind='avoids',stage='All',hits=6,hit_patients=6)])
    eligible=values.notna()
    tests=ci.stage_tests(values,eligible,counts,meta,ci.Config(score='stage',permutations=19),{1:values,-1:values})
    result=tests[(tests.unit=='PatientID') & (tests.contrast=='Control–Severe')]
    assert len(result)==1 and np.isclose(abs(result.effect_size.iloc[0]),.4)
