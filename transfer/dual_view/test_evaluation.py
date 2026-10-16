"""Truth-free synthetic checks for frozen evaluation and three-scheme reports.

All files below are tiny fabricated terminal records. No existing prediction,
target truth, feature cache, or experiment output is opened by these tests.
"""

import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from .evaluation import _row, _summary, evaluate, frozen_index
from .visualization import write_html


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
        self.record = {"scheme": "dual_view_continuous", "frozen": True,
                       "target_truth_used": False, "backends": ["ransac"],
                       "declared_task_ids": ["task"], "tasks": [self.state]}
        self.prediction = {"accepted": False, "reasons": [], "modes": [],
            "diagnostics": {"scheme": "dual_view_continuous", "native_backend": "ransac",
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


if __name__ == "__main__":
    unittest.main()
