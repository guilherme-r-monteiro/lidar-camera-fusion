# Design note

Target: BairesDev, Senior Perception Engineer. The posting asks for real-time perception pipelines for autonomy, sensor fusion of LiDAR, camera and radar, object detection, tracking, and processing workflows that meet latency demands.

## Question

Does fusing camera detections with LiDAR points give a better object distance than a camera-only estimate, and how well does a simple tracker keep object identities?

## Data

KITTI raw drive 2011_09_26_drive_0001 (108 frames, synchronised camera and Velodyne scans, calibration, and 3D tracklet labels). It is public. The data is not redistributed here.

## Method

1. Detect objects in the left colour image with YOLOv8n (ONNX).
2. Project every LiDAR point into the image with the KITTI calibration (Velodyne to camera, rectification, projection matrix).
3. Take the LiDAR points inside the central half of each detection box. Keep the nearest half of their depths, so background points seen through the box do not pull the estimate back. The median position is the fused 3D position.
4. Baseline: monocular distance from the pinhole model and a fixed height prior per class.
5. Ground truth: KITTI tracklet 3D boxes. A detection is matched to a tracklet when it contains the projected tracklet centre and has a compatible class.
6. Tracking: greedy nearest-neighbour association on the ground plane with a 4 m gate and a class constraint.
7. Report distance error per range bin, and identity switches against tracklet identities.

## Scope limits chosen before the run

- One drive of 108 frames, mostly vehicles.
- Radar is not available in KITTI, so only LiDAR and camera are fused.
- Only detections matched to a tracklet are scored. Detection recall is reported as a ratio, not optimised.
- The ground truth for distance is the near face of the 3D box, because LiDAR sees the near surface. Errors against the box centre are also reported.
