import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'result_summary')]
import rule_diagnostics as rd


def test_binary_patch_masks_show_hits_and_misses():
    field = pd.DataFrame({'x_um':[0.,1.,2.,50.], 'y_um':[0.,0.,0.,0.],
                          'cell type':['A','B','C','A']})
    result = rd.binary_patches(field, ['A'], ['B'], 'A', max_one_type_share=1.)
    assert result['metrics']['joint'] == 1
    assert result['metrics']['antecedent'] == 1
    assert result['metrics']['valid'] == 3


def test_every_diagnostic_companion_reads_a_source_rule_list():
    for family in ('complex_rules', 'differential_rules'):
        folder = ROOT / 'result_summary' / family
        for companion in (folder / 'rule_diagnostics').glob('*_diagnostics.ipynb'):
            source = folder / companion.name.replace('_diagnostics.ipynb', '.ipynb')
            specs = rd.selected_from_notebook(source)
            assert len(specs) and {'organ','rule'}.issubset(specs.columns)
