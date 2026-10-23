"""Initial rotation anchors preserve material geometry and continuous tool motion."""
# BEGIN frame-generic reference tests; all inputs are explicit synthetic geometry.
from copy import deepcopy
from itertools import product
from types import SimpleNamespace
import unittest

import numpy as np
from scipy.spatial.transform import Rotation

from manipulation.planning_problem import reference_phase
from manipulation.representation import build_representation, generate_reference, sample_poses
from manipulation.timed_reference import BoundToolReference


FIELD = "is_initial_relative_rotation_rpy"


def pose(position=(0., 0., 0.), rpy=(0., 0., 0.)):
    value = np.eye(4)
    value[:3, :3] = Rotation.from_euler("xyz", rpy).as_matrix()
    value[:3, 3] = position
    return value


def fixture(*, moving_reference=False):
    """Supply noncommuting source rotations, body offsets, and an independent entry."""
    times = np.array([2., 4., 6.])
    body = np.array([
        pose((.12, -.09, .17), (.1, -.2, .3)),
        pose((.21, .04, .25), (-.15, .22, .55)),
        pose((.34, .15, .08), (.25, .08, .8)),
    ])
    representation = build_representation(
        kind="manipulation", timestamps=times, frame_indices=[2, 7, 11],
        T_world_carrier=body, T_world_reference=np.repeat(np.eye(4)[None], 3, axis=0),
        L_R=np.eye(4), approach_time=2., interaction_time=5., release_time=5.6,
        events=[{"event_id": "declared_release", "timestamp": 5.6}])
    functional = pose((.5, -.3, .2), (.3, -.25, .45))
    if moving_reference:
        functional = np.array([functional,
            pose((.6, -.22, .24), (.25, -.1, .6)),
            pose((.67, -.12, .29), (.18, .12, .75))])
    arguments = dict(T_world_function=functional,
        J=pose((.04, -.03, .02), (.12, -.17, .23)),
        A=pose((.08, -.02, .11), (-.3, .2, -.45)),
        T_world_body=pose((.42, -.31, .6), (.5, -.12, 1.35)))
    arguments["T_world_eef"] = arguments["T_world_body"] @ np.linalg.inv(arguments["A"])
    return representation, arguments


def functional_samples(representation, arguments, result, progress):
    frame = arguments["T_world_function"]
    if frame.ndim == 2:
        return np.broadcast_to(frame, (len(progress), 4, 4))
    clock = result["object_timestamps"]
    times = clock[0] + np.asarray(progress) * (clock[-1] - clock[0])
    return sample_poses(frame, representation["timestamps"], times)


class InitialRelativeRotationReferenceTest(unittest.TestCase):
    def test_all_boolean_selections_preserve_relative_rotation_and_material_position(self):
        representation, arguments = fixture(moving_reference=True)
        original = generate_reference(representation, **arguments)
        progress = np.r_[0., .137, .381, .694, 1.]
        material = original["L_body_material"]
        frame = functional_samples(representation, arguments, original, progress)
        old = original["object_path"].sample_numpy(progress) @ material
        old_local = np.swapaxes(frame[:, :3, :3], -1, -2) @ old[:, :3, :3]
        actual_local = frame[0, :3, :3].T @ (arguments["T_world_body"] @ material)[:3, :3]
        old_rpy = Rotation.from_matrix(old_local[0]).as_euler("xyz")
        actual_rpy = Rotation.from_matrix(actual_local).as_euler("xyz")
        for flags in product((False, True), repeat=3):
            with self.subTest(flags=flags):
                declared = {**representation, FIELD: list(flags)}
                result = generate_reference(declared, **arguments)
                new = result["object_path"].sample_numpy(progress) @ material
                local = np.swapaxes(frame[:, :3, :3], -1, -2) @ new[:, :3, :3]
                selected = np.where(flags, actual_rpy, old_rpy)
                np.testing.assert_allclose(local[0], Rotation.from_euler("xyz", selected).as_matrix(), atol=2e-14)
                # This checks the physical relative-rotation invariant instead
                # of reproducing the implementation's left-correction formula.
                np.testing.assert_allclose(local[0].T @ local, old_local[0].T @ old_local, atol=2e-14)
                np.testing.assert_allclose(new[:, :3, 3], old[:, :3, 3], atol=2e-14)
                np.testing.assert_allclose(result["eef_path"].sample_numpy(progress) @ arguments["A"],
                                           result["object_path"].sample_numpy(progress), atol=2e-14)
                np.testing.assert_array_equal(result["H_m"], representation["H_m"])
                self.assertEqual(result["events"], representation["events"])

    def test_yaw_only_keeps_roll_pitch_curve_and_moves_the_offset_tool(self):
        representation, arguments = fixture()
        original = generate_reference(representation, **arguments)
        result = generate_reference({**representation, FIELD: [False, False, True]}, **arguments)
        progress = np.linspace(0., 1., 19)
        frame = arguments["T_world_function"][:3, :3]
        material = result["L_body_material"]
        old = original["object_path"].sample_numpy(progress) @ material
        new = result["object_path"].sample_numpy(progress) @ material
        old_rpy = Rotation.from_matrix(frame.T @ old[:, :3, :3]).as_euler("xyz")
        new_rpy = Rotation.from_matrix(frame.T @ new[:, :3, :3]).as_euler("xyz")
        np.testing.assert_allclose(new_rpy[:, :2], old_rpy[:, :2], atol=2e-14)
        change = np.unwrap(new_rpy[:, 2] - old_rpy[:, 2])
        np.testing.assert_allclose(change, np.full(len(progress), change[0]), atol=2e-14)
        self.assertGreater(np.linalg.norm(result["eef_poses"][-1, :3, 3] - original["eef_poses"][-1, :3, 3]), .01)
        self.assertEqual(result["rotation_transfer"][FIELD], [False, False, True])

    def test_false_flags_retain_nominal_reference_and_true_flags_bind_current_rotation(self):
        representation, arguments = fixture()
        original = generate_reference(representation, **arguments)
        unchanged = generate_reference({**representation, FIELD: [False] * 3}, **arguments)
        anchored = generate_reference({**representation, FIELD: [True] * 3}, **arguments)
        progress = np.linspace(0., 1., 17)
        np.testing.assert_allclose(unchanged["eef_path"].sample_numpy(progress),
                                   original["eef_path"].sample_numpy(progress), atol=2e-14)
        np.testing.assert_allclose(anchored["object_path"].sample_numpy(0.)[:3, :3],
                                   arguments["T_world_body"][:3, :3], atol=2e-14)

    def test_body_coordinate_changes_and_world_transform_preserve_physical_tool_motion(self):
        representation, arguments = fixture(moving_reference=True)
        representation[FIELD] = [False, False, True]
        original = generate_reference(representation, **arguments)
        progress = np.array([0., .17, .43, .79, 1.])
        tool = original["eef_path"].sample_numpy(progress)
        source_change = pose((.03, -.06, .07), (.4, -.3, .2))
        current_change = pose((-.02, .05, .04), (-.2, .1, -.35))
        changed = deepcopy(representation)
        changed["H_m"] = changed["H_m"] @ source_change
        inputs = {**arguments,
            "J": np.linalg.inv(source_change) @ arguments["J"] @ current_change,
            "A": arguments["A"] @ current_change,
            "T_world_body": arguments["T_world_body"] @ current_change}
        reexpressed = generate_reference(changed, **inputs)
        np.testing.assert_allclose(reexpressed["eef_path"].sample_numpy(progress), tool, atol=3e-14)
        world = pose((.3, -.7, .2), (.6, -.2, .4))
        moved = generate_reference(representation, **{**arguments,
            "T_world_function": world @ arguments["T_world_function"],
            "T_world_body": world @ arguments["T_world_body"],
            "T_world_eef": world @ arguments["T_world_eef"]})
        np.testing.assert_allclose(moved["eef_path"].sample_numpy(progress), world @ tool, atol=3e-14)

    def test_continuous_derivatives_and_physical_boundary_binding_share_new_reference(self):
        import torch
        representation, arguments = fixture(moving_reference=True)
        result = generate_reference({**representation, FIELD: [False, False, True]}, **arguments)
        path = result["eef_path"]
        def values(progress):
            position, orientation = path.sample(progress)
            return torch.cat((position, orientation))
        sample = torch.tensor(.317, dtype=torch.float64, requires_grad=True)
        derivative = torch.autograd.functional.jacobian(values, sample)
        step = 1e-6
        finite = (values(sample + step) - values(sample - step)) / (2 * step)
        self.assertTrue(torch.isfinite(derivative).all())
        np.testing.assert_allclose(derivative.detach().numpy(), finite.detach().numpy(), atol=2e-9, rtol=2e-8)
        phase = reference_phase(result, skill_id="synthetic", source="synthetic material motion")
        bound = BoundToolReference(SimpleNamespace(phases=(phase,)), [0., 1.], [0., 1.], 3.6,
                                  arguments["T_world_eef"], np.zeros(6))
        event = result["eef_boundaries"]["goals"][0]
        np.testing.assert_allclose(event["pose"], path.sample_numpy(event["progress"]), atol=2e-14)
        np.testing.assert_allclose(bound.sample_numpy(0.), arguments["T_world_eef"], atol=2e-13)
        times = np.array([3.6 * event["progress"], 3.3, 3.6])
        np.testing.assert_allclose(bound.sample_numpy(times), path.sample_numpy(times / 3.6), atol=2e-13)

    def test_time_varying_contact_uses_same_rotated_object_and_retained_contact_relation(self):
        representation, arguments = fixture()
        contact = np.array([pose((.04, .01, .08), (0., .1, .2)),
                            pose((.06, .03, .08), (.1, .05, .15)),
                            pose((.08, .04, .09), (.12, .02, .08))])
        representation.update(C_D=contact, is_initial_relative_rotation_rpy=[False, False, True])
        arguments.pop("A")
        arguments["contact_motion"] = True
        arguments["T_world_eef"] = arguments["T_world_body"] @ np.linalg.inv(arguments["J"]) @ contact[0]
        result = generate_reference(representation, **arguments)
        progress = np.array([0., .23, .71, 1.])
        times = 2. + 3.6 * progress
        relative = np.linalg.inv(arguments["J"]) @ sample_poses(contact, representation["timestamps"], times)
        np.testing.assert_allclose(result["eef_path"].sample_numpy(progress),
                                   result["object_path"].sample_numpy(progress) @ relative, atol=2e-14)

    def test_tool_reference_uses_calibrated_tool_frame_with_the_same_flag_contract(self):
        source = np.array([pose((.1, .0, .2), (.1, -.2, .3)),
                           pose((.2, .1, .1), (.2, .1, .5))])
        representation = build_representation(kind="grasp", timestamps=[2., 4.], frame_indices=[0, 1],
            T_world_carrier=np.repeat(np.eye(4)[None], 2, axis=0), T_world_eef=source,
            L_G=np.eye(4), interaction_time=4.)
        representation[FIELD] = [True, True, True]
        entry = pose((.4, -.2, .3), (.4, .2, -.3))
        result = generate_reference(representation, T_world_function=pose((.7, .0, .2)), T_world_eef=entry)
        np.testing.assert_allclose(result["eef_path"].sample_numpy(0.)[:3, :3], entry[:3, :3], atol=2e-14)

    def test_flags_require_three_booleans_at_the_consuming_interface(self):
        representation, arguments = fixture()
        for value in ([0, 0, 1], [False, True], [False, False, True, False], "false,false,true", None):
            with self.subTest(value=value), self.assertRaisesRegex(ValueError, "three Boolean"):
                generate_reference({**representation, FIELD: value}, **arguments)
# END initial-anchor selection is tested independently of object/task identities.


if __name__ == "__main__":
    unittest.main()
