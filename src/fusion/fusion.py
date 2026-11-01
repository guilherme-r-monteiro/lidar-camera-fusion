from dataclasses import dataclass

import numpy as np

from fusion.detector import Detection
from fusion.geometry import Calibration, Projected
from fusion.tracklets import Group

CENTRAL_FRACTION = 0.5
MINIMUM_POINTS = 5
NEAR_FRACTION = 0.5
CLASS_GROUPS = {
    "car": Group.VEHICLE,
    "truck": Group.VEHICLE,
    "bus": Group.VEHICLE,
    "person": Group.PERSON,
    "bicycle": Group.RIDER,
    "motorcycle": Group.RIDER,
}
HEIGHT_PRIOR_METERS = {Group.VEHICLE: 1.6, Group.PERSON: 1.75, Group.RIDER: 1.75}


@dataclass(frozen=True)
class Observation:
    detection: Detection
    group: Group
    fused_xyz: np.ndarray | None
    monocular_depth: float
    point_count: int


def group_of(label: str) -> Group | None:
    return CLASS_GROUPS.get(label)


def central_region(box: tuple[float, float, float, float]) -> tuple[float, float, float, float]:
    x1, y1, x2, y2 = box
    margin_x, margin_y = (x2 - x1) * (1 - CENTRAL_FRACTION) / 2, (y2 - y1) * (1 - CENTRAL_FRACTION) / 2
    return x1 + margin_x, y1 + margin_y, x2 - margin_x, y2 - margin_y


def fused_position(box: tuple[float, float, float, float], projected: Projected) -> tuple[np.ndarray | None, int]:
    x1, y1, x2, y2 = central_region(box)
    pixels = projected.pixels
    inside = (pixels[:, 0] >= x1) & (pixels[:, 0] <= x2) & (pixels[:, 1] >= y1) & (pixels[:, 1] <= y2)
    count = int(inside.sum())
    if count < MINIMUM_POINTS:
        return None, count
    depths = projected.depth[inside]
    near = depths <= np.quantile(depths, NEAR_FRACTION)
    return np.median(projected.camera_xyz[inside][near], axis=0), count


def monocular_depth(box: tuple[float, float, float, float], group: Group, calibration: Calibration) -> float:
    pixel_height = max(box[3] - box[1], 1.0)
    return calibration.focal_y * HEIGHT_PRIOR_METERS[group] / pixel_height


def observe(detections: list[Detection], projected: Projected, calibration: Calibration) -> list[Observation]:
    observations = []
    for detection in detections:
        group = group_of(detection.label)
        if group is None:
            continue
        position, count = fused_position(detection.box, projected)
        observations.append(Observation(detection, group, position, monocular_depth(detection.box, group, calibration), count))
    return observations
