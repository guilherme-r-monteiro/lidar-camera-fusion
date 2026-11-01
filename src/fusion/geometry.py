from dataclasses import dataclass
from pathlib import Path

import numpy as np

MIN_DEPTH = 0.5


@dataclass(frozen=True)
class Calibration:
    velo_to_cam: np.ndarray
    rect: np.ndarray
    projection: np.ndarray

    @property
    def focal_y(self) -> float:
        return float(self.projection[1, 1])


@dataclass(frozen=True)
class Projected:
    pixels: np.ndarray
    depth: np.ndarray
    camera_xyz: np.ndarray


def read_numbers(path: Path, key: str) -> np.ndarray:
    for line in path.read_text().splitlines():
        if line.startswith(f"{key}:"):
            return np.array([float(v) for v in line.split(":", 1)[1].split()])
    raise KeyError(key)


def load_calibration(velo_to_cam_file: Path, cam_to_cam_file: Path) -> Calibration:
    transform = np.eye(4)
    transform[:3, :3] = read_numbers(velo_to_cam_file, "R").reshape(3, 3)
    transform[:3, 3] = read_numbers(velo_to_cam_file, "T")
    rect = np.eye(4)
    rect[:3, :3] = read_numbers(cam_to_cam_file, "R_rect_00").reshape(3, 3)
    projection = read_numbers(cam_to_cam_file, "P_rect_02").reshape(3, 4)
    return Calibration(transform, rect, projection)


def to_camera(points_velo: np.ndarray, calibration: Calibration) -> np.ndarray:
    homogeneous = np.hstack([points_velo[:, :3], np.ones((len(points_velo), 1))])
    return (calibration.rect @ calibration.velo_to_cam @ homogeneous.T).T[:, :3]


def project(points_velo: np.ndarray, calibration: Calibration) -> Projected:
    camera = to_camera(points_velo, calibration)
    in_front = camera[:, 2] > MIN_DEPTH
    camera = camera[in_front]
    homogeneous = np.hstack([camera, np.ones((len(camera), 1))])
    image = (calibration.projection @ homogeneous.T).T
    return Projected(image[:, :2] / image[:, 2:3], camera[:, 2], camera)


def load_scan(path: Path) -> np.ndarray:
    return np.fromfile(path, dtype=np.float32).reshape(-1, 4)
