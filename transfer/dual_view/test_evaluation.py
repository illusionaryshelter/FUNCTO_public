"""Truth-free synthetic checks for frozen evaluation and three-scheme reports.

All files below are tiny fabricated terminal records. No existing prediction,
target truth, feature cache, or experiment output is opened by these tests.
"""

import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from contextlib import ExitStack

import numpy as np
from PIL import Image

from .evaluation import (_row, _summary, _dual_region_metrics, _center_radius_metric, _runtime_summary,
                         _supervisor_metrics, evaluate, frozen_index)
from .visualization import write_html
# BEGIN held-out transform metric test imports.
from .transform_metrics import affine_reference, evaluate_transforms, transform_error
from scipy.spatial.transform import Rotation
# END metric test imports.


def _dump(path, value):
    Path(path).write_text(json.dumps(value))


class FreezeTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.folder = self.root / "task"
        self.folder.mkdir()
        self.backend = {"status": "completed", "accepted": False,
                        "decision_status": "rejected_no_supported_region_in_H"}
        self.state = {"id": "task", "cases": [], "status": "completed",
                      "target_truth_used": False, "backends": {"ransac": self.backend}}
        self.record = {"scheme": "dual_view_observed_partial", "frozen": True,
                       "target_truth_used": False, "backends": ["ransac"],
                       "declared_task_ids": ["task"], "tasks": [self.state]}
        self.prediction = {"accepted": False, "reasons": [], "modes": [],
            "diagnostics": {"scheme": "dual_view_observed_partial", "native_backend": "ransac",
                            "target_truth_used": False,
                            "decision_status": "rejected_no_supported_region_in_H"}}
        np.savez_compressed(self.folder / "ransac.npz")
        self.save()

    def tearDown(self):
        self.temporary.cleanup()

    def save(self):
        _dump(self.root / "predictions.json", self.record)
        _dump(self.folder / "frozen.json", self.state)
        _dump(self.folder / "ransac.json", self.prediction)

    def test_complete_truth_free_record_passes(self):
        self.assertEqual(frozen_index(self.root), self.record)

    def test_unfrozen_run_and_mismatched_terminal_record_fail(self):
        self.record["frozen"] = False
        self.save()
        with self.assertRaises(ValueError):
            frozen_index(self.root)
        self.record["frozen"] = True
        self.save()
        _dump(self.folder / "frozen.json", {**self.state, "status": "running"})
        with self.assertRaises(ValueError):
            frozen_index(self.root)

    def test_missing_backend_artifact_fails(self):
        (self.folder / "ransac.npz").unlink()
        with self.assertRaises(FileNotFoundError):
            frozen_index(self.root)

    def test_unfinished_declared_task_cannot_disappear_from_denominator(self):
        self.record["declared_task_ids"].append("not_started")
        self.save()
        with self.assertRaises(ValueError):
            frozen_index(self.root)

    def test_task_level_truth_use_is_rejected(self):
        self.state["target_truth_used"] = True
        self.save()
        with self.assertRaises(ValueError):
            frozen_index(self.root)

    def test_other_scheme_or_backend_cannot_be_mislabeled_as_dual(self):
        for field, value in (("scheme", "joint_single_view"), ("native_backend", "teaserpp"),
                             ("target_truth_used", True)):
            original = self.prediction["diagnostics"][field]
            self.prediction["diagnostics"][field] = value
            self.save()
            with self.subTest(field=field), self.assertRaises(ValueError):
                frozen_index(self.root)
            self.prediction["diagnostics"][field] = original

    def test_terminal_missing_input_retains_declared_task(self):
        self.state.update(status="missing_input", backends={}, reasons=["source_model_inapplicable"])
        self.save()
        self.assertEqual(len(frozen_index(self.root)["tasks"]), 1)


class StatisticsTests(unittest.TestCase):
    def row(self, task, accepted, *, scheme="dual", view="A+B", status="completed",
            decision=None, error=.1, correct=False, warnings=()):
        result = {"accepted": accepted, "reasons": [], "modes": [],
                  "diagnostics": {"quality_warnings": list(warnings), "decision_status": decision}}
        metric = {"center_error_m": error, "top1_preferred_support_hit": correct,
                  "mode_metrics": [{"center_error_m": error, "preferred_support_hit": correct}]}
        return _row(task, scheme, "ransac", view, status, result, metric)

    def test_rejected_good_diagnostic_does_not_improve_accepted_metrics(self):
        accepted = self.row("accepted_wrong", True, error=.1, correct=False)
        rejected = self.row("rejected_good", False, error=.001, correct=True)
        self.assertIsNone(rejected["primary_metrics"])
        self.assertEqual(rejected["diagnostic_first_mode_metrics"]["center_error_m"], .001)
        summary = _summary([accepted, rejected])[0]
        self.assertEqual(summary["accepted"], 1)
        self.assertEqual(summary["correct_surface"], 0)
        self.assertEqual(summary["wrong_surface"], 1)
        self.assertEqual(summary["median_center_error_mm"], 100)
        self.assertEqual(summary["rejected"], 1)

    def test_status_counts_partition_complete_declared_domain(self):
        rows = [self.row("accepted", True), self.row("rejected", False),
                self.row("incomplete", False, decision="search_incomplete"),
                self.row("missing", False, status="missing_input"),
                self.row("failed", False, status="failed")]
        summary = _summary(rows)[0]
        self.assertEqual(summary["total"], 5)
        self.assertEqual(sum(summary[key] for key in
            ("accepted", "rejected", "incomplete", "missing", "failed")), 5)
        self.assertEqual(rows[2]["display_status"], "搜索未完成 / 不接受")

    def test_three_schemes_and_two_single_views_have_separate_denominators(self):
        rows = [self.row("same_task", False, scheme=scheme, view=view)
                for scheme in ("old", "new") for view in ("A", "B")]
        rows.append(self.row("same_task", True))
        summary = _summary(rows)
        self.assertEqual(len(summary), 5)
        self.assertTrue(all(item["total"] == 1 for item in summary))
        self.assertEqual(sum(item["accepted"] for item in summary), 1)

    def test_warning_preserved_and_html_has_incomplete_column(self):
        row = self.row("sample", True, warnings=["finite_search_not_global"])
        self.assertEqual(row["display_status"], "接受（有警告）")
        summary = _summary([self.row("partial", False, decision="search_incomplete")])
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "index.html"
            write_html(path, [], summary)
            html = path.read_text()
        self.assertIn("搜索未完成", html)
        self.assertIn("<th>搜索未完成</th>", html)

    def test_input_binding_failure_remains_ten_failed_comparison_rows(self):
        # A broken observed input cannot erase its task or cause evaluation to
        # proceed to held-out truth. Two backends have 4 single-view + 1 joint
        # rows apiece; each remains an explicit evaluation-binding failure.
        record = {"backends": ["ransac", "teaserpp"], "index": "fabricated_index.json",
                  "cache_root": "unused_cache", "tasks": [
                      {"id": "broken", "status": "failed", "cases": [
                          {"id": "broken_A", "view": "A"},
                          {"id": "broken_B", "view": "B"}], "backends": {}}]}
        module = "transfer.dual_view.evaluation."
        with tempfile.TemporaryDirectory() as folder:
            destination = Path(folder) / "report"
            with patch(module+"frozen_index", return_value=record), \
                 patch(module+"read_manifest", return_value=(Path(folder)/"input.json", {"data": folder})), \
                 patch(module+"load_observations", side_effect=ValueError("synthetic observed input mismatch")), \
                 patch(module+"read_archive", side_effect=AssertionError("Truth must not be read")) as truth_reader:
                summary = evaluate(Path(folder)/"predictions", destination,
                    old_results=Path(folder)/"old", new_results=Path(folder)/"new")
            truth_reader.assert_not_called()
            rows = json.loads((destination/"evaluation.json").read_text())["rows"]
        self.assertEqual(len(rows), 10)
        self.assertTrue(all(row["status"] == "failed" and not row["accepted"] for row in rows))
        self.assertTrue(all(row["primary_metrics"] is None for row in rows))
        self.assertEqual(sum(item["total"] for item in summary), 10)
        self.assertEqual(sum(item["failed"] for item in summary), 10)


# BEGIN evaluation-only transform contracts and historical-asset rendering.
class TransformMetricsTests(unittest.TestCase):
    def setUp(self):
        points = np.array([[-.03, -.02, 0.], [.03, -.02, 0.], [0., .04, 0.],
                           [-.2, -.1, .1], [.2, -.1, .1], [0., .2, .1]])
        self.source = {'query_xyz': points, 'check_xyz': points.copy(),
            'query_region': np.array([True, True, True, False, False, False]),
            'check_query_region': np.array([True, True, True, False, False, False]),
            'center': np.zeros(3), 'normal': np.array([0., 0., 1.])}

    def test_true_sim3_has_zero_scale_rotation_and_point_error(self):
        rotation = Rotation.from_euler('xyz', [13., -17., 41.], degrees=True).as_matrix()
        affine = np.eye(4)
        affine[:3, :3], affine[:3, 3] = 1.25*rotation, [.1, .2, .3]
        metric = transform_error((1.25, rotation, affine[:3, 3]), affine, self.source)
        for key in ('scale_signed_error_percent', 'rotation_vs_polar_deg', 'normal_error_deg',
                    'local_mapping_rmse_mm', 'context_mapping_rmse_mm', 'mapped_center_error_mm'):
            self.assertAlmostEqual(metric[key], 0, places=9)
        self.assertEqual(metric['local_point_count'], 6)
        self.assertEqual(metric['context_point_count'], 6)

    def test_same_center_and_normal_do_not_hide_scale_or_in_plane_rotation(self):
        metric = transform_error((.35, Rotation.from_euler('z', 95., degrees=True).as_matrix(),
                                  np.zeros(3)), np.eye(4), self.source)
        self.assertAlmostEqual(metric['scale_signed_error_percent'], -65)
        self.assertAlmostEqual(metric['rotation_vs_polar_deg'], 95)
        self.assertAlmostEqual(metric['normal_error_deg'], 0)
        self.assertAlmostEqual(metric['mapped_center_error_mm'], 0)
        self.assertGreater(metric['local_mapping_rmse_mm'], 30)
        self.assertGreater(metric['context_mapping_rmse_mm'], metric['local_mapping_rmse_mm'])

    def test_anisotropic_affine_has_polar_rotation_but_no_true_scalar(self):
        rotation = Rotation.from_euler('z', 37., degrees=True).as_matrix()
        affine = np.eye(4)
        affine[:3, :3] = rotation @ np.diag([1.18, 1., .85])
        reference = affine_reference(affine)
        self.assertFalse(reference['truth_is_sim3'])
        self.assertIsNone(reference['true_scale'])
        np.testing.assert_allclose(reference['polar_rotation'], rotation, atol=1e-12)
        metric = transform_error((1., rotation, np.zeros(3)), affine, self.source)
        self.assertIsNone(metric['scale_signed_error_percent'])
        self.assertIsNone(metric['scale_absolute_error_percent'])
        self.assertAlmostEqual(metric['rotation_vs_polar_deg'], 0)
        self.assertGreater(metric['local_mapping_rmse_mm'], 0)

    def test_affine_normal_uses_inverse_transpose(self):
        source = dict(self.source, normal=np.array([1., 1., 1.])/np.sqrt(3))
        affine = np.diag([2., 1., .5, 1.])
        expected = np.array([.5, 1., 2.])/np.linalg.norm([.5, 1., 2.])
        rotation, _ = Rotation.align_vectors(expected[None], source['normal'][None])
        metric = transform_error((1., rotation.as_matrix(), np.zeros(3)), affine, source)
        self.assertAlmostEqual(metric['normal_error_deg'], 0, places=9)

    def test_rejected_diagnostics_never_become_primary_or_certified(self):
        transform = (1., np.eye(3), np.zeros(3))
        result = {'accepted': False, 'diagnostics': {'hypotheses': [
            {'hypothesis_id': 7, 'transform': transform}], 'finite_candidate_unique': True},
            'modes': [{'hypothesis_index': 7}]}
        metrics = evaluate_transforms(result, self.source, np.eye(4), primary_transform=transform)
        self.assertIsNone(metrics['primary_estimate'])
        self.assertFalse(metrics['transform_certified'])
        self.assertEqual(metrics['displayed_candidate_role'], 'rejected_diagnostics_only')
        self.assertEqual(metrics['displayed_candidates'][0]['hypothesis_id'], 7)

    def test_competitor_transform_does_not_enter_accepted_primary_errors(self):
        primary = (1., np.eye(3), np.zeros(3))
        competitor = (.35, Rotation.from_euler('z', 95., degrees=True).as_matrix(), np.zeros(3))
        result = {'accepted': True, 'diagnostics': {'hypotheses': [
            {'hypothesis_id': 0, 'transform': primary}, {'hypothesis_id': 1, 'transform': competitor}]},
            'modes': [{'type': 'primary', 'hypothesis_indices': [0]},
                      {'type': 'competitor_different_surface', 'hypothesis_index': 1}]}
        metrics = evaluate_transforms(result, self.source, np.eye(4), primary_transform=primary)
        self.assertAlmostEqual(metrics['primary_estimate']['local_mapping_rmse_mm'], 0)
        self.assertEqual(len(metrics['displayed_candidates']), 1)
        extra = metrics['additional_diagnostic_modes'][0]
        self.assertEqual(extra['role'], 'competitor_or_rejected_diagnostic_not_primary')
        self.assertGreater(extra['candidates'][0]['local_mapping_rmse_mm'], 30)

    def test_competitor_correct_surface_cannot_improve_accepted_topk_or_samples(self):
        points = np.array([[0., 0., 1.], [.01, 0., 1.], [0., 0., 2.], [.01, 0., 2.]])
        modes = [{'seed_index': seed, 'center': points[seed], 'normal': np.array([0., 0., 1.]),
                  'mask': np.array([i in (seed, seed+1) for i in range(4)]),
                  'sample_indices': np.array([seed, seed+1])} for seed in (0, 2)]
        metrics = _dual_region_metrics({'accepted': True, 'modes': modes}, points,
            {'support_id': np.array([-1, -1, 5, 5]), 'preferred_support_id': np.asarray(5)}, points[2],
            sampling_radius_m=.03)
        self.assertFalse(metrics['accepted_topk_preferred_support_hit'])
        self.assertEqual(metrics['mode_count'], 1)
        self.assertEqual(metrics['all_modes_unique_samples'], 2)
        self.assertEqual(metrics['aggregate_scope'], 'primary_mode_only')
        self.assertTrue(metrics['mode_metrics'][1]['preferred_support_hit'])
        self.assertEqual(metrics['mode_metrics'][1]['role'], 'competitor_diagnostic')

    # BEGIN the recorded physical radius is a descriptive denominator only.
    def test_center_ratio_does_not_change_acceptance_or_include_rejected_primary(self):
        metric = _center_radius_metric({'center_error_m': .045, 'accepted': True}, .03)
        self.assertAlmostEqual(metric['center_error_radius_ratio'], 1.5)
        self.assertTrue(metric['accepted'])
        self.assertIsNone(_center_radius_metric(None, .03))
        self.assertIsNone(_center_radius_metric({'center_error_m': None}, .03)['center_error_radius_ratio'])
        result = {'accepted': False, 'reasons': [], 'diagnostics': {}, 'modes': [{}]}
        row = _row('rejected', 'dual', 'ransac', 'A+B', 'completed', result,
                   {**metric, 'mode_metrics': [metric]})
        self.assertIsNone(row['primary_metrics'])
        self.assertAlmostEqual(row['diagnostic_first_mode_metrics']['center_error_radius_ratio'], 1.5)

    def test_center_ratio_rejects_invalid_radius_and_error(self):
        for radius in (0, -.03, float('nan'), float('inf')):
            with self.subTest(radius=radius), self.assertRaises(ValueError):
                _center_radius_metric({'center_error_m': .001}, radius)
        for error in (-.01, float('nan'), float('inf')):
            with self.subTest(error=error), self.assertRaises(ValueError):
                _center_radius_metric({'center_error_m': error}, .03)
    # END physical-radius semantics.

    def test_task_total_and_supervisor_wall_do_not_double_count_shared_preparation(self):
        record = {'backends': ['ransac', 'teaserpp'], 'tasks': [{'seconds': 10.,
            'preparation_seconds': 4., 'stage_seconds': {'shared_target_ranking': 2.},
            'backends': {'ransac': {'seconds': 2.}, 'teaserpp': {'seconds': 3.}}}]}
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            path = root/'resources.json'
            _dump(path, {'command': ['python', '--output', str(root/'predictions')],
                'elapsed_s': 12., 'status': 'completed', 'returncode': 0})
            supervisor = _supervisor_metrics(path, root/'predictions')
            with self.assertRaises(ValueError):
                _supervisor_metrics(path, root/'another-run')
        summary = _runtime_summary(record, supervisor)
        self.assertEqual(summary['sum_task_seconds'], 10.)
        self.assertEqual(summary['supervisor']['wall_seconds'], 12.)
        self.assertEqual(summary['recorded_preparation_stage_sums_seconds']['shared_target_ranking'], 2.)
        self.assertEqual(summary['backends']['ransac']['sum_seconds'], 2.)
        self.assertEqual(summary['backends']['teaserpp']['sum_seconds'], 3.)

    def test_original_image_references_and_transform_warning_survive_render(self):
        transform = (1., np.eye(3), np.zeros(3))
        result = {'accepted': True, 'diagnostics': {'hypotheses': [
            {'hypothesis_id': 2, 'transform': transform}]}, 'modes': [{'hypothesis_indices': [2]}]}
        metrics = evaluate_transforms(result, self.source, np.eye(4))
        panel = {'view': 'A+B', 'display_status': '接受', 'images': [
            {'path': '../web/assets/frozen.png', 'caption': '原双视角'}], 'transform_metrics': metrics,
            'runtime': {'task_seconds': 3., 'backend_seconds': 1., 'timing_seconds': {}}}
        cards = [{'id': 'synthetic', 'source': None, 'methods': [
            {'scheme': 'dual_original', 'label': '原双视角', 'panels': [panel]}]}]
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)/'index.html'
            write_html(path, cards, [], baseline_link='../web/index.html')
            html = path.read_text()
            self.assertEqual(sorted(p.name for p in Path(folder).iterdir()), ['index.html'])
        self.assertIn('src="../web/assets/frozen.png"', html)
        self.assertIn('href="../web/index.html"', html)
        self.assertIn('完整旋转', html)
        self.assertIn('不证明完整变换可识别', html)
        self.assertIn('未记录', html)

    def test_two_task_evaluation_reuses_all_historical_images_and_keeps_timings(self):
        # Two tasks catch accidental replacement of the manifest by a per-task
        # historical JSON. Images are tiny fabricated assets, never real data.
        module = 'transfer.dual_view.evaluation.'
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            baseline, old_web, output = root/'original', root/'web', root/'revision'
            (old_web/'assets').mkdir(parents=True)
            output.mkdir()
            states, baseline_rows, loaded_by_task = [], {}, {}
            metric = {'center_error_m': .001, 'top1_preferred_support_hit': True,
                      'mode_metrics': [{'center_error_m': .001, 'preferred_support_hit': True}]}
            result = {'accepted': True, 'reasons': [], 'modes': [{'hypothesis_indices': [0]}],
                      'diagnostics': {'hypotheses': [], 'timing_seconds': {'scoring': .25},
                                      'sampling_radius_m': .03}}
            for task in ('one', 'two'):
                (baseline/task).mkdir(parents=True)
                (output/task).mkdir()
                cases, views = [], []
                evidence = task+'_evidence.npy'
                np.save(root/evidence, np.zeros(1, bool))
                for view in ('A', 'B'):
                    pair_path = task+'_'+view+'.json'
                    _dump(root/pair_path, {'preferred_support_id': 0,
                        'target_reference_evidence_center_world': [0., 0., 1.],
                        'target_reference_center_visible': True, 'true_affine_transform': np.eye(4).tolist()})
                    case = {'id': task+'_'+view, 'view': view, 'evidence': evidence, 'pair_truth': pair_path}
                    cases.append(case)
                    views.append({'id': view, 'case': case, 'frame_dir': root, 'rgb_path': root/'unused.png',
                        'K': np.eye(3), 'T_world_camera_cv': np.eye(4),
                        'observation': {'pixels': np.array([[0, 0]]), 'xyz': np.array([[0., 0., 1.]])}})
                    for name in (f'{task}_old_ransac_{view}.png', f'{task}_new_ransac_{view}.png',
                                 f'{task}_dual_ransac_{view}_mode0.png'):
                        Image.new('RGB', (4, 4)).save(old_web/'assets'/name)
                Image.new('RGB', (4, 4)).save(old_web/'assets'/f'{task}_source.png')
                _dump(baseline/task/'ransac.json', result)
                state = {'id': task, 'cases': cases, 'status': 'completed', 'seconds': 2.,
                    'preparation_seconds': .5, 'stage_seconds': {'shared_target_ranking': .2},
                    'surface_directory': str(root/'surface'),
                    'backends': {'ransac': {'status': 'completed', 'seconds': 1.5}}}
                states.append(state)
                baseline_rows[(task, 'ransac')] = _row(task, 'dual', 'ransac', 'A+B', 'completed', result, metric)
                loaded_by_task[task] = {'source': {}, 'source_rgb': root/'unused.png',
                    'query_indices': [], 'query_region': [], 'views': views}
            record = {'backends': ['ransac'], 'index': str(root/'index.json'),
                      'cache_root': str(root/'cache'), 'tasks': states,
                      'sampling': {'sampling_radius_m': .03}}
            surface = {'xyz': np.array([[0., 0., 1.]]), 'observation_xyz': np.zeros((2, 3)),
                'observation_view_index': np.array([0, 1]), 'observation_pixels': np.zeros((2, 2), int),
                'representative_observation': np.array([0])}
            with ExitStack() as stack:
                replacements = {
                    'frozen_index': record, '_original_dual': (record, baseline_rows),
                    'read_manifest': (root/'index.json', {'data': str(root)}),
                    'read_archive': {'support_id_image': np.zeros((4, 4), int)},
                    'read_rgb': np.zeros((4, 4, 3), np.uint8), 'restore_prediction': result,
                    'load_surface': surface, 'evaluate_prediction': metric, '_transform_evaluation': None,
                    'region_picture': (Image.new('RGB', (4, 4)), {'sample_count': 0, 'visible_samples': 0}),
                }
                for name, value in replacements.items():
                    stack.enter_context(patch(module+name, return_value=value))
                stack.enter_context(patch(module+'load_observations', side_effect=lambda _, task, __: loaded_by_task[task['id']]))
                source_draw = stack.enter_context(patch(module+'source_picture', side_effect=AssertionError('Historical source must be reused')))
                single_draw = stack.enter_context(patch(module+'single_view_picture', side_effect=AssertionError('Historical target must be reused')))
                summary = evaluate(output, root/'web-revision', old_results=root/'old', new_results=root/'new',
                    baseline_dual_results=baseline, baseline_dual_web=old_web)
            source_draw.assert_not_called()
            single_draw.assert_not_called()
            rows = json.loads((root/'web-revision/evaluation.json').read_text())['rows']
            self.assertEqual(len(rows), 12)
            self.assertEqual(sum(item['total'] for item in summary), 12)
            assets = list((root/'web-revision/assets').iterdir())
            self.assertEqual(len(assets), 4)
            self.assertTrue(all('_dual_' in asset.name for asset in assets))
            self.assertTrue(all(row['runtime']['preparation_stage_seconds']['shared_target_ranking'] == .2
                for row in rows if row['scheme'] == 'dual'))
# END transform/reference and no-copy report checks.


if __name__ == "__main__":
    unittest.main()
