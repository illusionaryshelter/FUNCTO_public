"""Coordinate, event-time and missing-input counterexamples for PRD references."""
# BEGIN fixed-frame and reference mathematics regressions.
import unittest
import numpy as np
from scipy.spatial.transform import Rotation
from manipulation.motion import MotionModel, _inverse
from manipulation.regions import source_region_frame
from manipulation.representation import build_representation, generate_reference, sample_poses, trajectory_summary


def pose(x=0., y=0., z=0., angle=0.):
    result = np.eye(4)
    result[:3, :3] = Rotation.from_euler("z", angle).as_matrix()
    result[:3, 3] = [x, y, z]
    return result


class RepresentationTest(unittest.TestCase):
    def test_source_camera_convention_and_fixed_body_binding(self):
        body, camera = pose(.4, -.1, .3, .8), pose(0., 0., 2., .3)
        points = np.array([[0., 0., .5], [.2, 0., .5], [0., .2, .5], [.2, .2, .5]])
        frame = source_region_frame(T_world_body=body, T_world_camera=camera, points_world=points)
        np.testing.assert_allclose(body @ frame["L_body_region"], frame["T_world_function"], atol=1e-14)
        np.testing.assert_allclose(frame["T_world_function"][:3, 3], [.1, .1, .5])
        np.testing.assert_allclose(frame["T_world_function"][:3, 2], [0, 0, 1])
        np.testing.assert_allclose(frame["T_world_function"][:3, :3].T @ frame["T_world_function"][:3, :3], np.eye(3), atol=1e-14)
        supplied = source_region_frame(T_world_body=body, T_world_camera=camera, T_world_function=pose(.1, .2, .3, -.2))
        np.testing.assert_allclose(body @ supplied["L_body_region"], pose(.1, .2, .3, -.2), atol=1e-14)

    def test_moving_reference_and_offset_region_roundtrip_each_time(self):
        times = [12., 12.3, 13.]
        carrier = np.array([pose(.5, .1, .7, a) for a in [.1, .2, .4]])
        reference = np.array([pose(x, .3, .2, a) for x, a in zip([.2, .3, .35], [0., .1, .3])])
        eef, L = carrier @ pose(.07, .03, .1, -.2), pose(.1, .02, .04, .5)
        rep = build_representation(kind="manipulation", frame_indices=[5, 8, 20], timestamps=times,
            T_world_carrier=carrier, T_world_reference=reference, T_world_eef=eef, L_R=L,
            approach_time=12., interaction_time=12.6)
        np.testing.assert_allclose(reference @ L @ rep["H_m"], carrier, atol=1e-14)
        np.testing.assert_allclose(carrier @ rep["C_D"], eef, atol=1e-14)
        np.testing.assert_allclose(rep["summaries"]["H_m"]["interaction_pose"], sample_poses(rep["H_m"], times, 12.6), atol=1e-14)
        np.testing.assert_array_equal(rep["frame_indices"], [5, 8, 20])

    def test_grasp_region_origin_is_not_the_body_origin(self):
        carrier = np.array([pose(.1, .2), pose(.2, .3)])
        L = pose(.1, .05, .02, .3)
        relative = np.array([pose(.0, -.2, .1), pose(.0, -.03)])
        rep = build_representation(kind="grasp", frame_indices=[0, 1], timestamps=[3., 4.],
            T_world_carrier=carrier, T_world_eef=carrier @ L @ relative, L_G=L, approach_time=3., interaction_time=4.)
        np.testing.assert_allclose(rep["H_g"], relative, atol=1e-14)
        current = pose(.5, -.4, .2, 1.)
        np.testing.assert_allclose(generate_reference(rep, T_world_function=current)["eef_poses"], current @ relative, atol=1e-14)

    def test_no_source_eef_still_generates_current_held_eef(self):
        body = np.array([pose(.2, .1, .4), pose(.3, .1, .2, .5)])
        rep = build_representation(kind="manipulation", frame_indices=[4, 7], timestamps=[2., 3.],
            T_world_carrier=body, T_world_reference=np.repeat(pose(.1, .2)[None], 2, axis=0),
            L_R=pose(.02), approach_time=2., interaction_time=3.)
        self.assertIn("C_D", rep["missing"])
        J, A = pose(.03, .01, 0., .2), pose(.1, .0, .15, -.5)
        first = generate_reference(rep, T_world_function=pose(.7), J=J, A=A)
        second = generate_reference(rep, T_world_function=pose(.7), J=J, A=pose(.0, .1, .2, .1))
        self.assertEqual(first["status"], "reference")
        np.testing.assert_allclose(first["object_poses"], second["object_poses"], atol=1e-14)
        np.testing.assert_allclose(first["eef_poses"] @ A, first["object_poses"], atol=1e-14)
        self.assertFalse(np.allclose(first["eef_poses"], second["eef_poses"]))
        missing = generate_reference(rep, T_world_function=pose(.7), J=J)
        self.assertIn("object_poses", missing)
        self.assertIn("eef_poses", missing["missing"])

    def test_event_interpolation_precedes_attachment_and_stops_at_release(self):
        rep = build_representation(kind="manipulation", frame_indices=[0, 10], timestamps=[5., 7.],
            T_world_carrier=np.array([pose(), pose(angle=np.pi/2)]),
            T_world_reference=np.repeat(np.eye(4)[None], 2, axis=0), L_R=np.eye(4),
            approach_time=5., interaction_time=6., release_time=6.5)
        A = pose(-.2)
        result = generate_reference(rep, T_world_function=np.eye(4), J=np.eye(4), A=A)
        np.testing.assert_allclose(result["summaries"]["eef"]["interaction_pose"], pose(angle=np.pi/4) @ _inverse(A), atol=1e-12)
        self.assertEqual(result["eef_timestamps"][-1], 6.5)
        self.assertEqual(result["object_timestamps"][-1], 7.)
        self.assertNotIn("retreat", result)
        self.assertIn("approach_direction", result["summaries"]["eef"])
        self.assertNotIn("approach_direction", result["summaries"]["object"])

    def test_source_body_redefinition_preserves_physical_reference(self):
        body = np.array([pose(.1, .2, .3), pose(.3, .2, .1, .5)])
        S, J, A = pose(.04, -.03, .02, -.4), pose(.1, .03, .0, .2), pose(.0, .0, .1)
        reps = [build_representation(kind="manipulation", frame_indices=[0, 1], timestamps=[1., 2.],
                    T_world_carrier=values, T_world_reference=np.repeat(np.eye(4)[None], 2, axis=0), L_R=np.eye(4))
                for values in (body, body @ S)]
        a = generate_reference(reps[0], T_world_function=pose(.8), J=J, A=A)
        b = generate_reference(reps[1], T_world_function=pose(.8), J=_inverse(S) @ J, A=A)
        np.testing.assert_allclose(a["object_poses"], b["object_poses"], atol=1e-14)
        np.testing.assert_allclose(a["eef_poses"], b["eef_poses"], atol=1e-14)

    def test_scalar_mechanism_needs_no_cabinet_registration_or_source_eef(self):
        model = MotionModel("revolute", np.array([[0.], [1.]]), pose(.4), np.eye(3), np.zeros(3), 1e-4, 1e-4, {})
        rep = build_representation(kind="manipulation", frame_indices=[3, 5], timestamps=[10., 12.], interaction_time=11.)
        result = generate_reference(rep, motion_model=model, source_coordinates=[.3, .7], current_coordinates=.2, A=pose(.0, .0, .1))
        np.testing.assert_allclose(result["coordinates"].ravel(), [.2, .4, .6])
        np.testing.assert_allclose(result["object_poses"], model.poses([[.2], [.4], [.6]]), atol=1e-14)
        self.assertEqual(result["status"], "reference")

    def test_nonattached_contact_uses_time_varying_CD(self):
        body, contact = np.array([pose(.1), pose(.2, .1, angle=.3)]), np.array([pose(.0, .1), pose(.03, .1, angle=-.1)])
        rep = build_representation(kind="manipulation", frame_indices=[0, 1], timestamps=[1., 2.], T_world_carrier=body,
            T_world_reference=np.repeat(np.eye(4)[None], 2, axis=0), T_world_eef=body @ contact, L_R=np.eye(4))
        J = pose(.03, .02, angle=.5)
        result = generate_reference(rep, T_world_function=pose(.7), J=J, contact_motion=True)
        np.testing.assert_allclose(result["eef_poses"], pose(.7) @ body @ contact, atol=1e-14)
        with self.assertRaises(ValueError):
            generate_reference(rep, T_world_function=pose(.7), J=J, A=np.eye(4), contact_motion=True)

    # BEGIN continuous reference regressions test between exported pose knots.
    def test_rotating_correspondence_offset_is_not_a_cartesian_chord(self):
        body = np.array([pose(), pose(angle=np.pi/2)])
        rep = build_representation(kind="manipulation", frame_indices=[0, 1], timestamps=[2., 4.],
            T_world_carrier=body, T_world_reference=np.repeat(np.eye(4)[None], 2, axis=0), L_R=np.eye(4))
        J, A = pose(.2), pose(.0, .1)
        result = generate_reference(rep, T_world_function=np.eye(4), J=J, A=A)
        expected = pose(angle=np.pi/4) @ J
        np.testing.assert_allclose(result["object_path"].sample_numpy(.5), expected, atol=1e-12)
        np.testing.assert_allclose(result["eef_path"].sample_numpy(.5), expected @ _inverse(A), atol=1e-12)
        self.assertGreater(np.linalg.norm(expected[:3, 3]-result["object_poses"][:, :3, 3].mean(axis=0)), .05)

    def test_moving_function_and_relative_motion_are_sampled_before_product(self):
        body = np.array([pose(.2, angle=0.), pose(.5, angle=np.pi/2)])
        function = np.array([pose(angle=0.), pose(.1, .2, angle=np.pi/2)])
        rep = build_representation(kind="manipulation", frame_indices=[5, 6], timestamps=[10., 14.],
            T_world_carrier=body, T_world_reference=np.repeat(np.eye(4)[None], 2, axis=0),
            L_R=np.eye(4), release_time=13.)
        J, A = pose(.15, .03, angle=.2), pose(.04, .01, angle=.1)
        result = generate_reference(rep, T_world_function=function, J=J, A=A)
        expected = sample_poses(function, [10., 14.], 11.5) @ sample_poses(body, [10., 14.], 11.5) @ J
        # Halfway through the held [10,13] interval is original time 11.5.
        np.testing.assert_allclose(result["eef_path"].sample_numpy(.5), expected @ _inverse(A), atol=1e-12)
        np.testing.assert_allclose(result["object_path"].sample_numpy(.375), expected, atol=1e-12)
        np.testing.assert_allclose(result["eef_path"].sample_numpy(1.), result["eef_poses"][-1], atol=1e-12)
        self.assertEqual(result["eef_timestamps"][-1], 13.)
        self.assertEqual(result["object_timestamps"][-1], 14.)

    def test_time_varying_contact_is_composed_between_knots(self):
        body = np.array([pose(.1), pose(.2, .1, angle=np.pi/2)])
        contact = np.array([pose(.0, .1), pose(.04, .15, angle=-.4)])
        function = np.array([pose(.5), pose(.7, angle=.5)])
        rep = build_representation(kind="manipulation", frame_indices=[0, 1], timestamps=[1., 3.],
            T_world_carrier=body, T_world_reference=np.repeat(np.eye(4)[None], 2, axis=0),
            T_world_eef=body @ contact, L_R=np.eye(4))
        result = generate_reference(rep, T_world_function=function, J=pose(.2, angle=.3), contact_motion=True)
        expected = (sample_poses(function, [1., 3.], 2.) @ sample_poses(body, [1., 3.], 2.)
                    @ sample_poses(contact, [1., 3.], 2.))
        np.testing.assert_allclose(result["eef_path"].sample_numpy(.5), expected, atol=1e-12)

    def test_scalar_contact_preserves_fk_and_relative_contact_between_knots(self):
        model = MotionModel("revolute", np.array([[0.], [1.]]), pose(.4), np.eye(3), np.zeros(3), 1e-4, 1e-4, {})
        body = np.array([pose(), pose(angle=.5)])
        contact = np.array([pose(.0, .1), pose(.03, .1, angle=-.1)])
        rep = build_representation(kind="manipulation", frame_indices=[0, 1], timestamps=[1., 3.],
            T_world_carrier=body, T_world_eef=body @ contact)
        J = pose(.03, .02, angle=.5)
        result = generate_reference(rep, motion_model=model, source_coordinates=[.3, .9],
                                    current_coordinates=.2, J=J, contact_motion=True)
        expected = model.poses([[.5]])[0] @ _inverse(J) @ sample_poses(contact, [1., 3.], 2.)
        np.testing.assert_allclose(result["eef_path"].sample_numpy(.5), expected, atol=1e-12)
    # END sample-path arithmetic is covered beyond the original/event knots.

    def test_zero_displacement_and_missing_events_do_not_erase_motion(self):
        poses = np.array([pose(), pose(angle=.5)])
        summary = trajectory_summary(poses, [2., 3.], approach_time=2., interaction_time=3.)
        self.assertIn("interaction_pose", summary)
        self.assertNotIn("approach_direction", summary)
        self.assertIn("approach_direction", summary["missing"])
        self.assertIn("interaction_pose", trajectory_summary(poses[:1], [2.], interaction_time=2.))
        with self.assertRaises(ValueError):
            sample_poses(poses, [2., 3.], 3.01)
# END fixed-frame and reference mathematics regressions.
