import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path[:0] = [str(Path(__file__).resolve().parents[1] / 'result_summary'),
                str(Path(__file__).resolve().parents[1] / 'result_summary' / 'differential_rules'),
                str(Path(__file__).resolve().parents[1] / 'result_summary' / 'complex_rules')]

import complex_investigation as ci
import parent_trend_reversals as reversals


def test_parent_keeps_arrow_sides_but_not_center_identity():
    child_ant = ('Muscle_CENTER', 'CD4T_NEIGHBOR')
    child_con = ('CD8T_NEIGHBOR', 'Macrophage_NEIGHBOR')
    assert reversals.role_preserving_parent(('Muscle_CENTER',), ('CD8T_NEIGHBOR',),
                                            child_ant, child_con)
    assert reversals.role_preserving_parent(('CD4T_CENTER',), ('CD8T_NEIGHBOR',),
                                            child_ant, child_con)
    assert not reversals.role_preserving_parent(('CD8T_NEIGHBOR',), ('Muscle_CENTER',),
                                                child_ant, child_con)


def test_opposite_trends_use_complex_eligibility():
    fovs = [f'f{i}' for i in range(30)]
    metadata = pd.DataFrame({
        'FOV': fovs,
        'Organ': ['Colon'] * 30,
        'Clinical score': ['Control'] * 10 + ['Mild'] * 10 + ['Severe'] * 10,
        'PatientID': [f'p{i // 2}' for i in range(30)],
    })
    states = pd.DataFrame(0, index=['child', 'parent'], columns=fovs, dtype='int8')
    states.loc['parent', fovs[:2] + fovs[20:28]] = -1
    states.loc['child', fovs[:8] + fovs[20:22]] = -1
    eligible = pd.DataFrame(True, index=states.index, columns=fovs)
    eligible.loc['child', fovs[10:12]] = False
    pairs = pd.DataFrame([dict(organ='Colon', rule='child', parent='parent', matched_fovs=6)])
    data = dict(states=states, eligible=eligible, metadata=metadata,
                pairs=pairs, config=ci.Config())
    counts = reversals.stage_counts(data, 'Colon', 'child', 'parent')
    assert counts[counts.stage.eq('Mild')].eligible.tolist() == [8, 8]
    selected = reversals.select(data)
    assert len(selected) == 1
    assert np.isclose(selected.iloc[0].parent_gap, -.6)
    assert np.isclose(selected.iloc[0].complex_gap, .6)
