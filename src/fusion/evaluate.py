import json
import statistics
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from fusion.detector import OnnxDetector
from fusion.fusion import Observation, observe
from fusion.geometry import Calibration, load_calibration, load_scan, project
from fusion.tracker import Tracker
from fusion.tracklets import GroundTruth, Group, parse_tracklets

CONFIDENCE = 0.3
RANGE_BINS = ((0.0, 15.0), (15.0, 30.0), (30.0, 60.0))
WITHIN_FRACTION = 0.10
MAX_EVALUATED_DEPTH = 60.0
COMPATIBLE_GROUPS = {
    Group.VEHICLE: {Group.VEHICLE},
    Group.PERSON: {Group.PERSON},
    Group.RIDER: {Group.RIDER, Group.PERSON},
}


@dataclass(frozen=True)
class Match:
    frame: int
    track_id: int
    depth: float
    near_depth: float
    fused_depth: float | None
    monocular_depth: float
    tracker_id: int | None


def project_point(point: np.ndarray, calibration: Calibration) -> np.ndarray:
    image = calibration.projection @ np.append(point, 1.0)
    return image[:2] / image[2]


def contains(box: tuple[float, float, float, float], pixel: np.ndarray) -> bool:
    return box[0] <= pixel[0] <= box[2] and box[1] <= pixel[1] <= box[3]


def area(box: tuple[float, float, float, float]) -> float:
    return (box[2] - box[0]) * (box[3] - box[1])


def match_truth(truth: GroundTruth, observations: list[Observation], calibration: Calibration) -> int | None:
    if truth.center_camera[2] <= 0 or truth.center_camera[2] > MAX_EVALUATED_DEPTH:
        return None
    pixel = project_point(truth.center_camera, calibration)
    candidates = [
        i
        for i, o in enumerate(observations)
        if o.group in COMPATIBLE_GROUPS[truth.group] and contains(o.detection.box, pixel)
    ]
    return min(candidates, key=lambda i: area(observations[i].detection.box)) if candidates else None


def run_frames(root: Path, calibration: Calibration, truths: dict[int, list[GroundTruth]], detector: OnnxDetector) -> tuple[list[Match], int]:
    tracker = Tracker()
    matches: list[Match] = []
    in_range = 0
    frames = sorted((root / "image_02" / "data").glob("*.png"))
    for index, image_path in enumerate(frames):
        scan = load_scan(root / "velodyne_points" / "data" / f"{image_path.stem}.bin")
        observations = observe(detector.detect(image_path, CONFIDENCE), project(scan, calibration), calibration)
        located = [i for i, o in enumerate(observations) if o.fused_xyz is not None]
        ids = tracker.update([observations[i].fused_xyz for i in located], [observations[i].group for i in located])
        tracker_ids = dict(zip(located, ids))
        for truth in truths.get(index, []):
            if 0 < truth.center_camera[2] <= MAX_EVALUATED_DEPTH:
                in_range += 1
            chosen = match_truth(truth, observations, calibration)
            if chosen is None:
                continue
            observation = observations[chosen]
            fused = float(observation.fused_xyz[2]) if observation.fused_xyz is not None else None
            matches.append(
                Match(index, truth.track_id, float(truth.center_camera[2]), truth.near_depth, fused, observation.monocular_depth, tracker_ids.get(chosen))
            )
    return matches, in_range


def error_summary(depths: list[float], estimates: list[float]) -> dict[str, float]:
    if not depths:
        return {"count": 0}
    signed = [e - d for d, e in zip(depths, estimates)]
    relative = [abs(s) / d for s, d in zip(signed, depths)]
    return {
        "count": len(depths),
        "median_signed_error_m": statistics.median(signed),
        "median_abs_error_m": statistics.median(abs(s) for s in signed),
        "median_rel_error": statistics.median(relative),
        "within_10_percent": sum(r <= WITHIN_FRACTION for r in relative) / len(relative),
    }


def depth_report(matches: list[Match]) -> dict[str, object]:
    report: dict[str, object] = {}
    for low, high in (*RANGE_BINS, (0.0, MAX_EVALUATED_DEPTH)):
        selected = [m for m in matches if low <= m.depth < high and m.fused_depth is not None]
        fused = [m.fused_depth for m in selected]
        mono = [m.monocular_depth for m in selected]
        report[f"{low:.0f}-{high:.0f}m"] = {
            "against_near_face": {
                "fused": error_summary([m.near_depth for m in selected], fused),
                "monocular": error_summary([m.near_depth for m in selected], mono),
            },
            "against_center": {
                "fused": error_summary([m.depth for m in selected], fused),
                "monocular": error_summary([m.depth for m in selected], mono),
            },
        }
    return report


def tracking_report(matches: list[Match]) -> dict[str, int]:
    by_truth: dict[int, list[int]] = {}
    for match in sorted(matches, key=lambda m: m.frame):
        if match.tracker_id is not None:
            by_truth.setdefault(match.track_id, []).append(match.tracker_id)
    switches = sum(sum(a != b for a, b in zip(ids, ids[1:])) for ids in by_truth.values())
    return {
        "ground_truth_tracks_matched": len(by_truth),
        "tracker_ids_used": len({i for ids in by_truth.values() for i in ids}),
        "id_switches": switches,
        "matched_observations": sum(len(ids) for ids in by_truth.values()),
    }


def main(data_dir: Path, model_path: Path, output: Path) -> None:
    date_dir = data_dir / "2011_09_26"
    root = date_dir / "2011_09_26_drive_0001_sync"
    calibration = load_calibration(date_dir / "calib_velo_to_cam.txt", date_dir / "calib_cam_to_cam.txt")
    truths = parse_tracklets(root / "tracklet_labels.xml", calibration)
    matches, in_range = run_frames(root, calibration, truths, OnnxDetector(model_path))
    report = {
        "ground_truth_in_range": in_range,
        "matched": len(matches),
        "matched_with_lidar_points": sum(m.fused_depth is not None for m in matches),
        "depth": depth_report(matches),
        "tracking": tracking_report(matches),
    }
    output.write_text(json.dumps(report, indent=2))
    sys.stdout.write(json.dumps(report["depth"]["0-60m"]) + "\n")


if __name__ == "__main__":
    main(Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3]))
