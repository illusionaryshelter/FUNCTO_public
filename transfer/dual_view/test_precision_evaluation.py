"""Synthetic checks for frozen two-run precision comparison and decision labels."""

import copy
from contextlib import ExitStack
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image

from .evaluation import _row
from .precision_evaluation import (_comparison_inputs, _decision, evaluate,
                                   precision_summary, render)


def _dump(path, value):
    Path(path).write_text(json.dumps(value))


class PrecisionTests(unittest.TestCase):
    def test_comparison_rejects_different_domain_before_truth_is_opened(self):
        module = 'transfer.dual_view.precision_evaluation.'
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            baseline = root/'baseline'
            web = root/'web'
            web.mkdir()
            record = {'declared_task_ids': ['one'], 'backends': ['ransac'],
                      'sampling': {'sampling_radius_m': .03}, 'index': str(root/'index'),
                      'cache_root': str(root/'cache'), 'tasks': [{'id': 'one', 'cases': ['camera'],
                                                               'surface_directory': 'fixed'}]}
            _dump(web/'evaluation.json', {'prediction_index': str(baseline/'predictions.json'),
                'read_after_all_predictions_frozen': True,
                'rows': [{'scheme': 'dual', 'task': 'one', 'backend': 'ransac'}]})
            with patch(module+'frozen_index', side_effect=[record, copy.deepcopy(record)]) as freeze:
                _comparison_inputs(root/'trial', baseline, web)
            self.assertEqual(freeze.call_count, 2)
            for field, changed in (('declared_task_ids', ['another']), ('backends', ['teaserpp']),
                                   ('sampling', {'sampling_radius_m': .06}), ('cache_root', 'different'),
                                   ('tasks', [{'id': 'one', 'cases': ['other'], 'surface_directory': 'fixed'}])):
                old = copy.deepcopy(record)
                old[field] = changed
                with self.subTest(field=field), patch(module+'frozen_index', side_effect=[record, old]), \
                     patch(module+'read_archive', side_effect=AssertionError('Truth must not open')), \
                     self.assertRaises(ValueError):
                    _comparison_inputs(root/'trial', baseline, web)

    def test_rejected_diagnostic_and_affine_rotation_do_not_enter_sim3_success_domain(self):
        rows = []
        for task, accepted, is_sim3, error in (('true', True, True, 4.), ('affine', True, False, 40.),
                                                ('rejected', False, True, .01)):
            rows.append({'task': task, 'scheme': 'trial', 'backend': 'ransac', 'status': 'completed',
                         'decision_status': 'accepted' if accepted else 'rejected', 'accepted': accepted,
                         'primary_metrics': {'center_error_m': .045, 'center_error_radius_ratio': 1.5,
                                             'top1_preferred_support_hit': True} if accepted else None,
                         'diagnostic_first_mode_metrics': {'center_error_m': .000001},
                         'transform_metrics': {'reference': {'truth_is_sim3': is_sim3},
                                               'primary_estimate': {'rotation_vs_polar_deg': error} if accepted else None}})
        result = precision_summary(rows)[0]
        self.assertEqual(result['total'], 3)
        self.assertEqual(result['accepted'], 2)
        self.assertEqual(result['rotation_true_sim3_deg']['count'], 1)
        self.assertEqual(result['rotation_true_sim3_deg']['median'], 4)
        self.assertEqual(result['rotation_anisotropic_polar_deg']['median'], 40)
        self.assertEqual(result['center_error_mm']['median'], 45)
        self.assertEqual(result['ratio_above_one'], 2)

    def test_invalid_decision_cannot_mislabel_trial(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)/'decision.json'
            for value in ({'status': 'best_case', 'reason': 'x'}, {'status': 'adopted', 'reason': ''}):
                _dump(path, value)
                with self.assertRaises(ValueError):
                    _decision(path)

    def test_two_task_report_has_only_two_dual_methods_and_reuses_baseline_assets(self):
        module = 'transfer.dual_view.precision_evaluation.'
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            baseline, web, trial, destination = root/'baseline', root/'web', root/'trial', root/'comparison'
            (web/'assets').mkdir(parents=True)
            trial.mkdir()
            states, prior_rows, loaded = [], {}, {}
            metric = {'center_error_m': .045, 'top1_preferred_support_hit': True,
                      'mode_metrics': [{'center_error_m': .045, 'preferred_support_hit': True}]}
            result = {'accepted': True, 'reasons': [], 'modes': [{'hypothesis_indices': [0]}],
                      'diagnostics': {'hypotheses': [], 'timing_seconds': {'scoring': .25},
                                      'sampling_radius_m': .03}}
            for task in ('one', 'two'):
                (baseline/task).mkdir(parents=True)
                (trial/task).mkdir()
                evidence = task+'_demo.npy'
                np.save(root/evidence, np.zeros(1, bool))
                views, cases = [], []
                for view in ('A', 'B'):
                    pair = task+'_'+view+'.json'
                    _dump(root/pair, {'preferred_support_id': 0,
                        'target_reference_evidence_center_world': [0., 0., 1.],
                        'target_reference_center_visible': True, 'true_affine_transform': np.eye(4).tolist()})
                    case = {'id': task+'_'+view, 'view': view, 'evidence': evidence, 'pair_truth': pair}
                    cases.append(case)
                    views.append({'id': view, 'case': case, 'frame_dir': root, 'rgb_path': root/'unused.png',
                        'K': np.eye(3), 'T_world_camera_cv': np.eye(4), 'observation': {}})
                    Image.new('RGB', (4, 4)).save(web/'assets'/f'{task}_dual_ransac_{view}_mode0.png')
                _dump(baseline/task/'ransac.json', result)
                states.append({'id': task, 'cases': cases, 'status': 'completed', 'seconds': 2.,
                    'preparation_seconds': .5, 'stage_seconds': {'shared_target_ranking': .2},
                    'surface_directory': str(root/'surface'), 'backends': {'ransac': {'status': 'completed', 'seconds': 1.5}}})
                prior_rows[(task, 'ransac')] = _row(task, 'dual', 'ransac', 'A+B', 'completed', result, metric)
                loaded[task] = {'source': {}, 'source_rgb': root/'unused.png',
                                'query_indices': [], 'query_region': [], 'views': views}
            record = {'backends': ['ransac'], 'index': str(root/'index.json'), 'cache_root': str(root/'cache'),
                      'sampling': {'sampling_radius_m': .03}, 'tasks': states}
            surface = {'xyz': np.array([[0., 0., 1.]]), 'observation_xyz': np.zeros((2, 3)),
                       'observation_view_index': np.array([0, 1]), 'observation_pixels': np.zeros((2, 2), int),
                       'representative_observation': np.array([0])}
            with ExitStack() as stack:
                for name, value in {'_comparison_inputs': (record, record, prior_rows),
                    'read_manifest': (root/'index.json', {'data': str(root)}),
                    'read_archive': {'support_id_image': np.zeros((4, 4), int)},
                    'read_rgb': np.zeros((4, 4, 3), np.uint8), 'restore_prediction': result,
                    'load_surface': surface, '_transform_evaluation': None,
                    'region_picture': (Image.new('RGB', (4, 4)), {'sample_count': 0, 'visible_samples': 0}),
                }.items():
                    stack.enter_context(patch(module+name, return_value=value))
                stack.enter_context(patch('transfer.dual_view.evaluation.evaluate_prediction', return_value=metric))
                stack.enter_context(patch(module+'load_observations', side_effect=lambda _, task, __: loaded[task['id']]))
                source = stack.enter_context(patch(module+'source_picture', return_value=Image.new('RGB', (4, 4))))
                answer = evaluate(trial, destination, baseline_results=baseline, baseline_web=web)
            self.assertEqual(source.call_count, 2)
            self.assertEqual(len(answer), 2)
            report = json.loads((destination/'evaluation.json').read_text())
            self.assertEqual(len(report['rows']), 4)
            self.assertEqual({row['scheme'] for row in report['rows']}, {'baseline', 'trial'})
            self.assertTrue(all(row['primary_metrics']['center_error_radius_ratio'] == 1.5 for row in report['rows']))
            self.assertEqual(len(list((destination/'assets').iterdir())), 6)
            html = (destination/'index.html').read_text()
            self.assertEqual(html.count('class="dual-comparison"'), 2)
            self.assertIn('../web/assets/one_dual_ransac_A_mode0.png', html)
            self.assertNotIn('data-scheme="old"', html)
            self.assertNotIn('value="dual_original"', html)
            self.assertIn('误差/采样半径 1.500', html)
            snapshots = {path.name: path.read_bytes() for path in (destination/'assets').iterdir()}
            decision = root/'decision.json'
            _dump(decision, {'status': 'not_adopted', 'reason': '固定旋转门槛未通过。'})
            with patch(module+'read_archive', side_effect=AssertionError('Render-only must not read GT')), \
                 patch(module+'load_observations', side_effect=AssertionError('Render-only must not load input')):
                render(destination, decision_json=decision)
                render(destination)
            self.assertIn('未采用试验', (destination/'index.html').read_text())
            self.assertEqual(json.loads((destination/'decision.json').read_text())['status'], 'not_adopted')
            self.assertEqual(snapshots, {path.name: path.read_bytes() for path in (destination/'assets').iterdir()})


if __name__ == '__main__':
    unittest.main()
