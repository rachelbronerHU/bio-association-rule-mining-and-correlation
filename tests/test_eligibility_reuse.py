"""Regression checks for shared eligibility and parent-comparison denominators."""
import sys
import unittest
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'result_summary'),
               str(ROOT / 'result_summary/differential_rules/scripts'),
               str(ROOT / 'result_summary/complex_rules/scripts')]
import rule_metrics as rm
import complex_investigation as ci
import complex_investigation_vis as vis
import parent_trend_reversals as reversals
import parent_trend_reversals_vis as reversal_vis


class EligibilityReuseTests(unittest.TestCase):
    def setUp(self):
        self.cells = pd.DataFrame({
            'fov': ['f1'] * 60 + ['f2'] * 60,
            'cell type': ['A', 'B', 'C'] * 40,
            'x_um': np.tile(np.arange(60) % 10, 2),
            'y_um': np.tile(np.arange(60) // 10, 2),
        })
        self.settings = rm.Settings(
            weighting=rm.Weighting.BINARY, method=rm.Method.CN, radius=25,
            min_support=.01, min_lift=1.2, max_items_per_rule=4,
            min_patches=15, min_label_count=5, avoidance_max_lift=.8,
            avoidance_min_expected_meetings=30,
        )
        self.fields = rm.Fields(self.cells, self.settings)
        self.metadata = pd.DataFrame({'FOV': ['f1', 'f2']})
        self.names = ['A + B -> C']

    def test_masks_and_plots_reuse_patches_and_preserve_results(self):
        expected = ci.testable_fovs(self.names, self.cells, self.metadata, ci.Config(),
                                   fields=rm.Fields(self.cells, self.settings))
        expected_cells = rm.counted_cells(['A_CENTER'], ['B_NEIGHBOR'],
                                         self.cells, 'f1', self.settings)
        with patch.object(rm.mining_transactions, 'find_patches',
                          wraps=rm.mining_transactions.find_patches) as build:
            with patch.object(self.fields, 'side_supports',
                              wraps=self.fields.side_supports) as supports:
                for _ in range(3):
                    actual = ci.testable_fovs(self.names, self.cells, self.metadata, ci.Config(),
                                             fields=self.fields)
                    pd.testing.assert_frame_equal(actual, expected)
                self.assertEqual(supports.call_count, 2)
            self.assertEqual(self.fields.counted_cells(['A_CENTER'], ['B_NEIGHBOR'], 'f1'),
                             expected_cells)
            ci.testable_fovs(['A -> C'], self.cells, self.metadata, ci.Config(), fields=self.fields)
            self.assertEqual(build.call_count, 2)  # Once per FOV, including the new parent.

    def test_thresholds_and_kinds_have_separate_cached_results(self):
        normal = ci.testable_fovs(self.names, self.cells, self.metadata, ci.Config(),
                                 fields=self.fields)
        strict = ci.testable_fovs(self.names, self.cells, self.metadata,
                                 replace(ci.Config(), support=.9, expected_support=.9),
                                 fields=self.fields)
        self.assertTrue(normal.to_numpy().all())
        self.assertFalse(strict.to_numpy().any())
        definitions = {'parent': (['A_CENTER'], ['B_NEIGHBOR'])}
        avoidance = rm.can_pass_fovs(definitions, self.cells, kinds=(rm.AVOIDS,), fields=self.fields)
        attraction = rm.can_pass_fovs(definitions, self.cells, kinds=(rm.ATTRACTS,), fields=self.fields)
        self.assertFalse(avoidance.to_numpy().any())
        self.assertTrue(attraction.to_numpy().all())
        attraction.iloc[:, :] = False
        again = rm.can_pass_fovs(definitions, self.cells, kinds=(rm.ATTRACTS,), fields=self.fields)
        self.assertTrue(again.to_numpy().all())

    def test_empty_rules_and_missing_settings(self):
        result = ci.testable_fovs([], self.cells, self.metadata, ci.Config(), fields=self.fields)
        self.assertEqual(result.shape, (0, 2))
        self.assertTrue(all(dtype == bool for dtype in result.dtypes))
        self.fields.settings = None
        with self.assertRaisesRegex(ValueError, 'valid run_config.json'):
            rm.can_pass_fovs({}, self.cells, fields=self.fields)


class ParentComparisonTests(unittest.TestCase):
    def setUp(self):
        self.child, self.parent = 'A + B -> C', 'A -> C'
        self.metadata = pd.DataFrame({
            'FOV': ['both', 'parent', 'complex', 'neither', 'mild'],
            'Organ': ['Colon'] * 5,
            'Clinical score': ['Control'] * 4 + ['Mild'],
            'PatientID': ['p1', 'p2', 'p3', 'p4', 'p5'],
        })
        self.data = dict(
            states=pd.DataFrame([[1, 0, -1, 0, 1], [1, -1, 0, 0, 1]],
                                index=[self.child, self.parent], columns=self.metadata.FOV),
            eligible=pd.DataFrame([[True, False, True, False, True],
                                   [True, True, False, False, True]],
                                  index=[self.child, self.parent], columns=self.metadata.FOV),
            metadata=self.metadata, config=ci.Config(),
        )

    def tearDown(self):
        plt.close('all')

    def test_intersection_counts_exclude_fovs_testable_for_only_one_rule(self):
        table = ci.stage_counts(self.data, 'Colon', self.child, self.parent)
        control = table[table.stage.eq('Control')]
        self.assertTrue(control.eligible.eq(1).all())
        self.assertTrue(control.attraction.eq(1).all())
        self.assertTrue(control.avoidance.eq(0).all())
        self.assertTrue(control.net.eq(1).all())
        self.assertTrue(table[table.stage.eq('Severe')].eligible.eq(0).all())

    def test_plot_shares_use_the_intersection_denominator(self):
        table = ci.stage_counts(self.data, 'Colon', self.child, self.parent)
        with patch.object(vis.dv, '_finish') as finish:
            vis.parent_vs_complex(table, 'Colon', self.child, self.parent)
            fig = finish.call_args.args[0]
            for line in fig.axes[0].lines:
                np.testing.assert_allclose(line.get_ydata(), [100, 100, np.nan])
            fig.canvas.draw()

    def test_reversal_ranking_uses_only_jointly_testable_fovs(self):
        metadata = pd.DataFrame({
            'FOV': ['c_both', 'c_parent', 'c_complex', 's_both', 's_parent', 's_complex', 'm'],
            'Organ': ['Colon'] * 7,
            'Clinical score': ['Control'] * 3 + ['Severe'] * 3 + ['Mild'],
            'PatientID': [f'p{i}' for i in range(7)],
        })
        data = dict(
            states=pd.DataFrame([[-1, 0, 1, 1, 0, -1, 1], [1, -1, 0, -1, 1, 0, 1]],
                                index=[self.child, self.parent], columns=metadata.FOV),
            eligible=pd.DataFrame([[True, False, True, True, False, True, True],
                                   [True, True, False, True, True, False, True]],
                                  index=[self.child, self.parent], columns=metadata.FOV),
            pairs=pd.DataFrame([dict(organ='Colon', rule=self.child, parent=self.parent,
                                     matched_fovs=3)]),
            metadata=metadata, config=ci.Config(),
        )
        selected = reversals.select(data, min_eligible=1, min_patients=1,
                                    min_hits=1, min_mild_eligible=1)
        self.assertEqual(len(selected), 1)
        row = selected.iloc[0]
        self.assertEqual((row.control_eligible, row.severe_eligible), (1, 1))
        self.assertEqual((row.parent_gap, row.complex_gap), (-2, 2))

    def test_reversal_examples_use_the_same_intersection(self):
        self.data.update(rows=pd.DataFrame(), fields=object())
        selected = pd.DataFrame([dict(organ='Colon', rule=self.child, parent=self.parent)])
        with patch.object(reversal_vis, 'display'), patch('builtins.print'), \
                patch.object(vis, 'parent_vs_complex'), \
                patch.object(reversal_vis.ds, 'representative_fovs', return_value=pd.DataFrame()) as choose, \
                patch.object(rm, 'explain_missing', return_value=pd.DataFrame()), \
                patch.object(reversal_vis.cr, 'plot_rule_fovs'):
            reversal_vis.show_selected(0, selected, self.data, pd.DataFrame())
        self.assertEqual(choose.call_count, 2)
        for call in choose.call_args_list:
            mask = call.args[1]
            for name in (self.child, self.parent):
                self.assertEqual(mask.loc[name].tolist(), [True, False, False, False, True])


if __name__ == '__main__':
    unittest.main()
