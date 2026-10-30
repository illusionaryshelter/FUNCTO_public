"""Prepare an explicit observed request and call the shared preference backend.

The supplied factory owns skill representation and observed task construction.
The shared runner owns RGB-D scene preparation, robot/device binding and result
recording. The planner owns numerical policy and all candidate decisions.
"""
import argparse
from functools import partial
import importlib.util
import json
from pathlib import Path
import sys

PROJECT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT))

from manipulation.evidence_archive import read_evidence_json
from manipulation.planning import PlanningBudget
from manipulation.preference_planner import PreferenceMotionBackend, bind_pose_objectives
from run_recorded_skill import parse_args as parse_planning_args, run_prepared


def run(request_path, factory_path, output):
    """Pass one complete task and measured entrance through the shared APIs."""
    request_path, factory_path, output = map(Path, (request_path, factory_path, output))
    request = json.loads(request_path.read_text())
    request['request_path'] = str(request_path.resolve())
    specification = importlib.util.spec_from_file_location('observed_problem_factory', factory_path)
    factory = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(factory)
    factory.prepare(request)
    feedback_path = Path(request['inputs'])/'feedback.json'
    feedback = json.loads(feedback_path.read_text())
    feedback['incoming_finger_command'] = json.loads(Path(request['incoming_command']).read_text())
    feedback_path.write_text(json.dumps(feedback, indent=2)+'\n')

    output = output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    command = json.loads((Path(request['inputs'])/'planner-command.json').read_text())
    command[command.index('--output')+1] = str(output/'planning-run')
    command[command.index('--planning-seconds')+1] = str(PlanningBudget.planning_seconds)
    prepared = parse_planning_args(command)
    problem, provenance = factory.build_problem(request)
    problem = bind_pose_objectives(problem, request['pose_objectives'])
    provenance['pose_objectives'] = request['pose_objectives']
    continuation = (None if prepared.continuation is None
                    else read_evidence_json(prepared.continuation))
    terminal = (None if continuation is None else
                continuation['original_binding']['validation']['export']['reference_termination'])
    result = run_prepared(prepared, problem=problem, provenance=provenance,
        backend_factory=partial(PreferenceMotionBackend, incoming_terminal_contract=terminal))
    (output/'planning-command.json').write_text(json.dumps(command, indent=2)+'\n')
    summary = dict(planning_success=result.success,
        trajectory=str(output/'planning-run/planning/trajectory.npz'),
        report=str(output/'planning-run/planning/planning-report.json'))
    (output/'result-summary.json').write_text(json.dumps(summary, indent=2)+'\n')
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('request', 'factory', 'output'):
        parser.add_argument('--'+name, type=Path, required=True)
    args = parser.parse_args()
    result = run(args.request, args.factory, args.output)
    print(json.dumps(dict(planning_success=result.success)), flush=True)
