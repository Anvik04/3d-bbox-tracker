# 3D Bounding Box Tracker (Camera + LiDAR Fusion)

An end-to-end Python/PyTorch pipeline for 3D multi-object detection and tracking of moving objects, utilizing sensor fusion (Camera + LiDAR) in the KITTI data layout.

## Quick Start: Real-Time Vehicle Detection Demo

Run the monocular 3D vehicle detection/tracking demo with no LiDAR required (camera only):

```bash
python scripts/run_mono_demo.py --source 0
python scripts/run_mono_demo.py --source path/to/video.mp4 --save-video outputs/mono_demo.mp4
```

Calibration note: edit [configs/mono_camera.yaml](configs/mono_camera.yaml) with your camera height and tilt before running.

## Architecture Overview

```
                      +-------------------+
                      | Camera Image      |
                      |   (3, H_i, W_i)   |
                      +---------+---------+
                                |
                                v
                      +-------------------+
                      |   CNN Backbone    |
                      +---------+---------+
                                | Image Feature Map
                                v (Bilinear Sampling / Projection)
  +-----------------+ +-------------------+
  | LiDAR PC        | |                   |
  |  (N, 4) [XYZI]  | |                   |
  +--------+--------+ |                   |
           |          |  Calibrated       |
           v          |  Fusion Layer     |
  +-----------------+ |                   |
  | Pillar Encoder  | |                   |
  | (PointNet BEV)  | |                   |
  +--------+--------+ +---------+---------+
           |                    |
           +----------+---------+
                      | Fused BEV Map (C_lidar + C_img, H_bev, W_bev)
                      v
              +---------------+
              |  3D Detection |
              |     Head      |
              +-------+-------+
                      | Raw 3D BBoxes & Scores
                      v (Rotated BEV NMS)
              +---------------+
              | 3D Detections |
              +-------+-------+
                      | Detections: [x,y,z,l,w,h,yaw]
                      v (Hungarian Matching / 3D IoU + Distance)
              +---------------+
              |   3D Kalman   |
              |    Tracker    |
              +-------+-------+
                      |
                      v
              +---------------+
              | Active Tracks | (Track ID, Velocity, 3D BBox)
              +---------------+
```

## Repository Structure

```
3d-bbox-tracker/
├── README.md                 # Project guide (this file)
├── DECISIONS.md              # Design decisions log
├── pyproject.toml            # Dependencies and tool configurations
├── configs/
│   └── kitti_fusion.yaml     # Config hyperparams (sensor, model, tracker)
├── data/
│   └── fixtures/             # Generated synthetic miniature KITTI dataset
├── src/
│   ├── data/
│   │   ├── calib.py          # Calib matrix parsing and projection transformations
│   │   ├── kitti_dataset.py  # KITTI format PyTorch Dataset loader
│   │   └── synth_fixtures.py # Synthetic data generation engine
│   ├── models/
│   │   ├── pillar_encoder.py # PointPillars voxelization & feature extractor
│   │   ├── image_backbone.py # Camera feature maps CNN backbone
│   │   ├── fusion.py         # Bilinear grid projection sensor fusion
│   │   ├── detection_head.py # 3D regression & focal class heads
│   │   └── detector.py       # End-to-end model and Rotated NMS decoder
│   ├── tracking/
│   │   ├── kalman_3d.py      # Constant-velocity 3D Kalman Filter
│   │   └── tracker.py        # Hungarian tracking association and lifecycles
│   ├── training/
│   │   ├── losses.py         # Focal loss and Smooth L1 regression loss
│   │   └── train.py          # Model training loop
│   ├── eval/
│   │   └── metrics.py        # 3D IoU, Average Precision (AP), and MOTA
│   └── viz/
│       └── visualize.py      # Open3D renderers & 2D Matplotlib fallbacks
├── scripts/
│   ├── generate_fixtures.py  # CLI wrapper to create synthetic data
│   └── run_inference.py      # Inference CLI to output tracks & plots
└── tests/                    # Comprehensive unit & integration tests
```

## How to Get Started

### 1. Installation
Ensure Python 3.11 is installed, then run:
```bash
# Clone the repository
git clone <repo-url> 3d-bbox-tracker
cd 3d-bbox-tracker

# Create virtual environment and install in editable mode with dev tools
python3.11 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

### 2. Generate Synthetic KITTI Dataset
Generate the small 10-frame mini-KITTI dataset with moving cars:
```bash
python scripts/generate_fixtures.py
```

### 3. Run Unit and Smoke Tests
Verify all components are functional and correctly integrated:
```bash
python -m pytest
```

### 4. Train Model
Run the training loop on the generated synthetic dataset (runs on CPU/GPU and saves a checkpoint):
```bash
python src/training/train.py
```

### 5. Run Inference and Tracking Pipeline
Process the sequence through the trained detector and tracker to output predictions and visualizations:
```bash
python scripts/run_inference.py
```
This command outputs:
- Standard KITTI prediction files to `outputs/predict/`
- Projected 2D wireframe overlays on images to `outputs/viz/image_2/`
- Headless-friendly 3D BEV tracking scatter plots to `outputs/viz/3d/`

---

## Real-Time Monocular Vehicle Detection (No LiDAR)

This repository now has an additive no-LiDAR inference path for real-time vehicle 3D perception from a webcam or video file. Instead of the existing Camera+LiDAR fusion model, it uses a COCO-pretrained YOLOv8n 2D detector, a simple monocular ground-plane lift step, and the existing 3D Kalman tracker to produce live 3D vehicle boxes.

### How it differs from the fusion pipeline
- The original path trains a Camera+LiDAR fusion detector on synthetic KITTI fixtures.
- The new monocular path uses only RGB frames and a simple camera-height/tilt calibration to estimate a box in front of the camera.
- It is intentionally lightweight and designed to run on CPU with a webcam or recorded video.

### Run the demo
```bash
python scripts/run_mono_demo.py --source 0
python scripts/run_mono_demo.py --source path/to/video.mp4 --save-video outputs/mono_demo.mp4
```

Note: [scripts/webcam_demo.py](scripts/webcam_demo.py) is the earlier motion-based prototype and is not the current recommended entry point for the monocular vehicle-detection demo; use [scripts/run_mono_demo.py](scripts/run_mono_demo.py) instead.

### Calibration notes
The mono configuration file at [configs/mono_camera.yaml](configs/mono_camera.yaml) expects:
- camera_height_m: the height of the camera center above the ground plane (measure with a tape measure)
- tilt_deg: a rough camera tilt estimate relative to the ground plane
- fov_deg: the webcam or phone horizontal field-of-view spec, used to derive a focal length from the frame width
- principal_point: usually left at the image center unless you have a more precise calibration

## Swapping to Real Sensor Data (KITTI or nuScenes)

This codebase has been architected to conform to standard KITTI layouts, making dataset upgrades direct:

1. **KITTI Dataset**: 
   Point the data directories in `configs/kitti_fusion.yaml` (or pass the data root folder directly to `KITTIDataset`) to your downloaded official KITTI directory structure (`velodyne/`, `image_2/`, `calib/`, `label_2/`). No code modifications are needed.

2. **nuScenes Dataset**:
   To load nuScenes data, write a wrapper dataloader similar to `kitti_dataset.py` that translates the nuScenes coordinate system (where x is right, y is forward, z is up in LiDAR) to this pipeline's conventions, and extracts the calibration transformations.
