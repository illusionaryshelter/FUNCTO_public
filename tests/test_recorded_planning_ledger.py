"""The diagnostic CLI preserves failed planning work for an explicit retry."""

# BEGIN failed planning results must leave a reusable immutable request ledger.
from copy import deepcopy
import json
from pathlib import Path
import sys
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent/"planning"))
from manipulation.planning import PlanningBudget, PlanningResult
from manipulation.planning_problem import PlanningEvent
from recorded_task_planning import plan_and_record


def test_failed_result_archives_ledger_and_passes_previous_work_to_next_attempt(tmp_path):
    problem = SimpleNamespace(events=(PlanningEvent("close", "skill", .5, 3, "close", "explicit test event"),),
                              phase_count=1, anchors=np.array([0., .5, 1.]))
    budget = PlanningBudget()
    # BEGIN this recorder test intercepts planning before any geometry query.
    # Object identity proves the supplied source binding survives both attempts.
    robot_validator = object()
    # BEGIN the recorder forwards one caller-owned resident backend unchanged.
    backend = SimpleNamespace(robot_validator=robot_validator)
    # END this handoff test never constructs native models.
    # END no test geometry is claimed by this handoff-only sentinel.
    first = {"request_id": "same-explicit-request", "consumed_planning": {"initializations": 1},
             "history": [{"value": "F", "source": "unchanged failed attempt fact"}]}
    second = {**deepcopy(first), "consumed_planning": {"initializations": 2}}
    outputs = [PlanningResult(False, None, {"reason": "synthetic failed initialization"}, request_ledger=first),
               PlanningResult(False, None, {"reason": "synthetic second failed initialization"}, request_ledger=second)]
    # BEGIN the recorder invokes the already bound backend, including on retry.
    backend.plan = plan = Mock(side_effect=outputs)
    with patch("torch.cuda.max_memory_allocated", return_value=0), \
            patch("torch.cuda.max_memory_reserved", return_value=0):
        result = plan_and_record(problem, None, {}, tmp_path/"first", robot_validator=robot_validator, budget=budget,
                                provenance={"scope": "CPU handoff regression; no native solve"}, execution_limits={}, backend=backend)
        saved = json.loads((tmp_path/"first"/"request-ledger.json").read_text())
        assert saved == first and not result.success
        plan_and_record(problem, None, {}, tmp_path/"second", robot_validator=robot_validator, budget=budget,
                        provenance={"scope": "same CPU request"}, execution_limits={}, backend=backend, request_ledger=saved)
    assert plan.call_args_list[1].kwargs["request_ledger"] == first
    assert backend.robot_validator is robot_validator
    assert plan.call_count == 2
    # END retries retain the caller's model and pass the spent-work ledger.
    assert json.loads((tmp_path/"second"/"request-ledger.json").read_text()) == second
    report = json.loads((tmp_path/"second"/"planning-report.json").read_text())
    assert not report["success"] and Path(report["request_ledger_path"]).name == "request-ledger.json"
# END absence of an accepted trajectory cannot erase spent work.
