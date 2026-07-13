# Design Decisions and Assumptions

This document lists the architectural design decisions and implementation choices for the 3D perception and tracking pipeline.

## 1. Voxelization and Pillar Feature Encoding
- **Decision**: Vectorized grouping and scattering in pure PyTorch.
- **Rationale**: Standard PointPillars implementations use custom C++/CUDA voxelization extensions (like `spconv`). To keep the codebase fully cross-platform and lightweight (installable within seconds on CPU/macOS/GPU), we implemented a vectorized sorting and start-mask prefix sum grouping logic. This runs entirely in PyTorch without custom compilation.

## 2. Differentiable Camera-LiDAR Fusion
- **Decision**: Implement coordinate projection matrices directly in PyTorch and use `F.grid_sample`.
- **Rationale**: Converting tensors to NumPy/CPU for projection during the forward pass breaks the autograd computation graph. Projecting grid centers using PyTorch tensors and sampling via bilinear interpolation is fully differentiable and compatible with both GPU and CPU execution.

## 3. Continuous Yaw Regression
- **Decision**: Predict $(\sin(\theta), \cos(\theta))$ instead of raw $\theta$ yaw angle.
- **Rationale**: Raw angle regression suffers from boundary discontinuity at $-\pi$ and $\pi$ (where $-\pi + \epsilon \approx \pi - \epsilon$ in orientation but is far apart in regression space). Regressing sine and cosine values, then decoding using `arctan2`, provides smooth gradients across all angles.

## 4. Multi-Object Tracking Association Fallback
- **Decision**: Cost matrix is calculated as `1.0 - BEV_IoU`. If BEV IoU is zero, fall back to center-to-center distance scaled by a maximum threshold (4.0m).
- **Rationale**: For fast-moving objects or low frame-rates, consecutive detections might not overlap at all, yielding an IoU of 0. Fallback to center-to-center distance ensures continuous tracking and prevents identity switches or birth/death cycles.

## 5. Headless 3D Visualization
- **Decision**: Auto-fallback from Open3D to Matplotlib 3D scatter plots.
- **Rationale**: Open3D visualizer requires an active X-server/window server and crashes in headless environments or CI. When `output_path` is provided, we bypass the GUI window creation and output Matplotlib-based 3D scatter plots instead.

## 6. Pre-NMS Top-K Selection
- **Decision**: Sort detections by score and select only the top 100 highest-scoring boxes before running rotated NMS.
- **Rationale**: In the early training phases or with untrained models, many grid cells output false positive bounding boxes exceeding the score threshold. Running rotated NMS (which constructs Shapely polygons and calculates intersection areas) on thousands of boxes scales quadratically and causes severe performance bottlenecks. Limiting inputs to the top 100 ensures that the test suite and evaluation metrics run in milliseconds.
