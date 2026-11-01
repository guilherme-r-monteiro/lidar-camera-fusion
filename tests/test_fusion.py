import numpy as np

from fusion.detector import Detection
from fusion.fusion import fused_position, monocular_depth
from fusion.geometry import Calibration, Projected, project
from fusion.tracker import GATE_METERS, Tracker
from fusion.tracklets import Group, near_face_depth


def identity_calibration() -> Calibration:
    projection = np.array([[700.0, 0.0, 600.0, 0.0], [0.0, 700.0, 180.0, 0.0], [0.0, 0.0, 1.0, 0.0]])
    return Calibration(np.eye(4), np.eye(4), projection)


def test_project_drops_points_behind_the_camera() -> None:
    points = np.array([[0.0, 0.0, 10.0, 1.0], [0.0, 0.0, -5.0, 1.0]])
    projected = project(points, identity_calibration())
    assert len(projected.depth) == 1 and projected.depth[0] == 10.0
    assert np.allclose(projected.pixels[0], [600.0, 180.0])


def test_fused_position_uses_nearest_half_of_points() -> None:
    pixels = np.tile(np.array([[100.0, 100.0]]), (10, 1))
    depth = np.array([10.0] * 5 + [40.0] * 5)
    xyz = np.stack([np.zeros(10), np.zeros(10), depth], axis=1)
    position, count = fused_position((50.0, 50.0, 150.0, 150.0), Projected(pixels, depth, xyz))
    assert count == 10 and position is not None and position[2] == 10.0


def test_too_few_points_gives_no_position() -> None:
    projected = Projected(np.array([[100.0, 100.0]]), np.array([10.0]), np.array([[0.0, 0.0, 10.0]]))
    assert fused_position((50.0, 50.0, 150.0, 150.0), projected)[0] is None


def test_monocular_depth_follows_pinhole_model() -> None:
    depth = monocular_depth((0.0, 0.0, 10.0, 70.0), Group.VEHICLE, identity_calibration())
    assert abs(depth - 700.0 * 1.6 / 70.0) < 1e-9


def test_near_face_is_closer_than_center() -> None:
    center = np.array([0.0, 0.0, 0.0, 1.0])
    calibration = Calibration(np.array([[0, -1, 0, 0], [0, 0, -1, 0], [1, 0, 0, 20], [0, 0, 0, 1]], dtype=float), np.eye(4), np.eye(3, 4))
    assert near_face_depth(center, (4.0, 2.0, 1.5), 0.0, calibration) == 18.0


def test_tracker_keeps_id_inside_gate_and_spawns_outside() -> None:
    tracker = Tracker()
    first = tracker.update([np.array([0.0, 0.0, 10.0])], [Group.VEHICLE])
    near = tracker.update([np.array([0.0, 0.0, 10.5])], [Group.VEHICLE])
    far = tracker.update([np.array([0.0, 0.0, 10.5 + GATE_METERS + 1])], [Group.VEHICLE])
    assert near == first and far != first


def test_detection_dataclass_is_usable() -> None:
    assert Detection("car", 0.9, (0.0, 0.0, 1.0, 1.0)).label == "car"
