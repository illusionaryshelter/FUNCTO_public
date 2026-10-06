"""Memory pressure may reclaim unused pages, never disable the host limits.

Use simulated memory readings rather than allocating gigabytes or starting a
GPU worker. The failure tests protect the hard-stop behavior after cleanup.
"""

from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from local_models import runtime


class EndLoop(Exception):
    """Stop the otherwise persistent monitor after one successful iteration."""


class WorkerExit(Exception):
    """Represent os._exit without terminating the test process."""


class MemoryGuardTest(unittest.TestCase):
    def run_guard(self, rss_gib, available_gib, should_exit):
        worker = Mock()
        parent = Mock()
        # Keep a nonzero caller footprint so tests cover the combined limit.
        parent.memory_info.return_value = SimpleNamespace(rss=2**30)
        worker.memory_info.side_effect = [SimpleNamespace(rss=(rss - 1) * 2**30) for rss in rss_gib]
        with patch("psutil.Process", side_effect=[worker, parent]), \
             patch("psutil.virtual_memory", side_effect=[SimpleNamespace(available=x * 2**30) for x in available_gib]), \
             patch.object(runtime, "_reclaim_cpu_memory") as reclaim, \
             patch.object(runtime.os, "_exit", side_effect=WorkerExit) as exit_worker, \
             patch.object(runtime.time, "sleep", side_effect=EndLoop):
            with self.assertRaises(WorkerExit if should_exit else EndLoop):
                runtime._guard_memory(123)
        return reclaim, exit_worker

    def test_normal_memory_does_not_trigger_reclamation(self):
        # BEGIN bounded-worker cleanup starts before the outer 8 GiB cap.
        reclaim, exit_worker = self.run_guard([5], [6], False)
        reclaim.assert_not_called()
        exit_worker.assert_not_called()

    def test_dead_checkpoint_pages_are_reclaimed_before_outer_limit(self):
        reclaim, exit_worker = self.run_guard([6.4, 5.2], [6, 7.2], False)
        reclaim.assert_called_once_with()
        exit_worker.assert_not_called()
        # END early cleanup leaves both hard-stop tests below unchanged.

    def test_reclaimed_unused_pages_allow_inference_to_continue(self):
        reclaim, exit_worker = self.run_guard([12.4, 9.2], [2.5, 5.7], False)
        reclaim.assert_called_once_with()
        exit_worker.assert_not_called()

    def test_live_combined_rss_still_stops_after_reclamation(self):
        reclaim, exit_worker = self.run_guard([12.4, 12.1], [2.5, 2.8], True)
        reclaim.assert_called_once_with()
        exit_worker.assert_called_once_with(75)

    def test_low_host_availability_still_stops_below_rss_limit(self):
        reclaim, exit_worker = self.run_guard([8, 7], [0.9, 1.2], True)
        reclaim.assert_called_once_with()
        exit_worker.assert_called_once_with(75)


if __name__ == "__main__":
    unittest.main()
