# YOLOv8 Real-Time Object Detection - User Guide

## Quick Start

### Installation
1. Create virtual environment: python -m venv .venv
2. Activate on Windows: .venv\\Scripts\\Activate.ps1
3. Install dependencies: pip install -r requirements.txt
4. Run application: python main.py

### Basic Usage
1. Start application with python main.py
2. Allow webcam access when prompted by OS
3. Press q to exit
4. Review generated files in outputs folder

## Keyboard Controls

### Navigation
- h: Show help overlay
- q or ESC: Quit application
- SPACE: Pause or resume

### Detection Control
- +: Increase confidence threshold
- -: Decrease confidence threshold
- c: Reset threshold to default
- f: Open class filter menu

### Capture and Recording
- s: Save screenshot (PNG)
- r: Toggle video recording (MP4)
- l: Show detection log summary

### Display Options
- d: Toggle demo mode
- p: Toggle performance panel
- t: Toggle threshold indicator

## Configuration
Edit config.py defaults through AppConfig sections:
- model: model path, confidence and IoU thresholds, device
- webcam: camera id, resolution, target FPS, frame flip
- output: output directories and recording defaults
- detection: FPS display, counting, class filtering controls

## Output Files
- Screenshots: outputs/screenshots
- Videos: outputs/videos
- Detection logs: outputs/detections
- Reports: outputs/reports

## Troubleshooting

### Webcam Issues
- Ensure webcam is not used by another app
- Try webcam.camera_id = 1 in config.py
- Verify camera permissions in OS settings

### Low FPS
- Use model.model_path = yolov8n.pt
- Lower webcam.frame_width and webcam.frame_height
- Disable recording and demo mode
- Use CUDA-enabled PyTorch if GPU available

### High Memory Usage
- Disable recording
- Lower frame size
- Close heavy background applications

### CUDA Not Used
- Run: python -c "import torch; print(torch.cuda.is_available())"
- Install CUDA-compatible torch build

## Advanced Usage
- Run practice viva: python practice_viva.py
- Run test suite: python test_runner.py
- Run performance profile: python optimization/profile_performance.py

## References
- README.md for project architecture and setup
- controls_reference.txt for full key map
- docs/DEPLOYMENT_CHECKLIST.md for release readiness
