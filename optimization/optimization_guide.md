# Performance Optimization Guide

## Current Performance Analysis
- Target FPS: 30
- Current FPS: Measure with optimization/profile_performance.py
- Bottleneck: Usually inference on CPU
- Primary Constraint: CPU or visualization overhead

## Optimization Techniques

### 1. Model Optimization
- Use yolov8n.pt for fastest inference.
- Use yolov8m.pt when accuracy priority is higher.
- Avoid large models for strict real-time CPU targets.

### 2. Input Optimization
- Reduce resolution from 640x480 to 416x416 or 320x240.
- Typical impact: 20-40 percent FPS improvement.
- Trade-off: reduced detail may lower detection quality.

### 3. Inference Optimization
- Enable GPU (CUDA) when available.
- Typical impact: 3x to 5x faster inference.
- Confirm with torch.cuda.is_available().

### 4. Visualization Optimization
- Disable non-essential overlays during performance runs.
- Draw fewer labels and avoid expensive per-frame effects.
- Keep only core overlays (FPS and boxes) for max speed.

### 5. Frame Processing Optimization
- Process every second frame if strict FPS needed.
- Resize frame before inference.
- Keep conversion and copy operations minimal.

### 6. Output Optimization
- Disable video recording while benchmarking.
- Disable screenshot notifications and heavy logging.
- Export logs at shutdown instead of every frame.

## Quick Wins
1. Use yolov8n.pt.
2. Lower frame resolution.
3. Disable demo mode.
4. Disable performance panel and recording.

## Typical Profiling Breakdown (CPU)
- Camera capture: 3-8 ms
- YOLO inference: 10-30 ms
- Visualization: 3-10 ms
- Display and I/O: 3-8 ms

## Recommended Max-Performance Config
```python
# config.py
model.model_path = "yolov8n.pt"
webcam.frame_width = 416
webcam.frame_height = 416
output.save_video = False
detection.enable_fps_display = True
```

Expected: 25-30 FPS on CPU, and higher on GPU.
