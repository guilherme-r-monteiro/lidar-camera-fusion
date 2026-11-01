# lidar-camera-fusion

Fuses YOLOv8n detections with LiDAR points on a public KITTI drive to estimate object distance, tracks the objects, and scores both against KITTI 3D tracklet labels.

Design note: `docs/design.md`. Raw results: `results/fusion-results.json`.

- Data: KITTI raw drive 2011_09_26_drive_0001, 108 frames (https://www.cvlibs.net/datasets/kitti/raw_data.php).
- Detector: YOLOv8n exported to ONNX, CPU.
- Fusion: LiDAR points projected into the image, nearest-half median depth inside the central half of each box.
- Baseline: monocular distance from a class height prior and the pinhole model.

## Results

Distance to the near face of the 3D box, for detections matched to a tracklet and holding at least five LiDAR points:

| Range | Matched | LiDAR fusion: median error, within 10% | Camera only: median error, within 10% |
|---|---|---|---|
| 0-15 m | 40 | 0.75 m, 72% | 1.02 m, 55% |
| 15-30 m | 145 | 0.34 m, 98% | 2.34 m, 48% |
| 30-60 m | 131 | 0.25 m, 92% | 3.47 m, 50% |
| 0-60 m | 316 | 0.35 m, 92% | 2.45 m, 50% |

Of 464 tracklet objects within 60 m, 337 were matched to a detection, and 316 of those held enough LiDAR points.

Tracking with a greedy nearest-neighbour tracker: 14 tracklet identities matched, 27 tracker identities used, 23 identity switches over 316 matched observations.

What the results say:
- Fusion cuts the median distance error from 2.45 m to 0.35 m, and the share within 10% from 50% to 92%. The camera-only error grows with range. The fused error does not.
- Close objects are worse (0.75 m). Large nearby boxes hold many background points and the near-face rule helps less there.
- Against the box centre the numbers differ (fused median error 1.70 m, bias about -1.7 m) because LiDAR measures the near surface. The choice of reference is part of the result and both are in the raw file.
- The tracker is weak. It has no motion model and no ego-motion compensation, so identities split when detections flicker. It is a baseline, not a result to rely on.

## Limits

- One drive, 108 frames, a few vehicles and pedestrians. Differences by range bin rest on 40 to 145 matches.
- Only matched detections are scored, so a missed object does not count against distance error. The match rate (337 of 464) is the detection recall at this setting.
- The monocular baseline uses one fixed height per class. A better prior or a trained depth model would narrow the gap.
- LiDAR and camera are rectified and synchronised by KITTI. Real systems need to handle time offsets and calibration drift. That is not tested here.
- No radar and no 3D detector. The fused position is a point estimate, not a 3D box.
- The tracker's ID switches are measured on matched observations only.
- Python and NumPy, CPU. Latency was not measured, so no real-time claim is made.

## Run it

```sh
uv venv .venv && uv pip install --python .venv/bin/python -e . pytest
mkdir -p data && cd data
B=https://s3.eu-central-1.amazonaws.com/avg-kitti/raw_data
curl -LO $B/2011_09_26_drive_0001/2011_09_26_drive_0001_sync.zip
curl -LO $B/2011_09_26_calib.zip
curl -LO $B/2011_09_26_drive_0001/2011_09_26_drive_0001_tracklets.zip
for f in *.zip; do unzip -q -o $f; done; cd ..
.venv/bin/python -m pytest
.venv/bin/python -m fusion.evaluate data models/yolov8n.onnx results/fusion-results.json
```

The ONNX model comes from `scripts/setup.sh` in https://github.com/guilherme-r-monteiro/vision-agent-mcp.
