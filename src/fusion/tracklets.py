import xml.etree.ElementTree as ET
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path

import numpy as np

from fusion.geometry import Calibration, to_camera


class Group(StrEnum):
    VEHICLE = "vehicle"
    PERSON = "person"
    RIDER = "rider"


OBJECT_GROUPS = {
    "Car": Group.VEHICLE,
    "Van": Group.VEHICLE,
    "Truck": Group.VEHICLE,
    "Pedestrian": Group.PERSON,
    "Person (sitting)": Group.PERSON,
    "Cyclist": Group.RIDER,
}


@dataclass(frozen=True)
class GroundTruth:
    track_id: int
    group: Group
    center_camera: np.ndarray
    near_depth: float


def read_text(node: ET.Element, tag: str) -> str:
    child = node.find(tag)
    if child is None or child.text is None:
        raise ValueError(tag)
    return child.text.strip()


def near_face_depth(center_velo: np.ndarray, size: tuple[float, float, float], yaw: float, calibration: Calibration) -> float:
    length, width, height = size
    signs = np.array([[x, y, z] for x in (-1, 1) for y in (-1, 1) for z in (-1, 1)], dtype=float)
    corners = signs * np.array([length, width, height]) / 2
    rotation = np.array([[np.cos(yaw), -np.sin(yaw), 0.0], [np.sin(yaw), np.cos(yaw), 0.0], [0.0, 0.0, 1.0]])
    in_velo = corners @ rotation.T + center_velo[:3]
    return float(to_camera(in_velo, calibration)[:, 2].min())


def parse_tracklets(path: Path, calibration: Calibration) -> dict[int, list[GroundTruth]]:
    frames: dict[int, list[GroundTruth]] = {}
    root = ET.parse(path).getroot()
    for track_id, item in enumerate(root.find("tracklets").findall("item")):
        group = OBJECT_GROUPS.get(read_text(item, "objectType"))
        if group is None:
            continue
        height = float(read_text(item, "h"))
        first = int(read_text(item, "first_frame"))
        for offset, pose in enumerate(item.find("poses").findall("item")):
            center_velo = np.array(
                [[float(read_text(pose, "tx")), float(read_text(pose, "ty")), float(read_text(pose, "tz")) + height / 2, 1.0]]
            )
            center = to_camera(center_velo, calibration)[0]
            size = (float(read_text(item, "l")), float(read_text(item, "w")), height)
            near = near_face_depth(center_velo[0], size, float(read_text(pose, "rz")), calibration)
            frames.setdefault(first + offset, []).append(GroundTruth(track_id, group, center, near))
    return frames
