# YOLOv8 Real-Time Object Detection System

A beginner-friendly Deep Learning mini assignment project that performs **real-time object detection from webcam** using a **pretrained YOLOv8 model** (no model training required).

## What This Project Does
- Captures live frames from your webcam.
- Uses pretrained YOLOv8 weights to detect objects in real time.
- Draws bounding boxes, class labels, and confidence scores.
- Supports optional output saving (videos/screenshots).
- Provides a full environment verification script before coding main logic.

## Key Features
- ⚡ Real-time webcam object detection
- 🧠 Pretrained model inference (transfer learning usage)
- 🖥️ Works on Windows, macOS, and Linux
- 🛠️ Config-driven setup via config.py
- ✅ Installation validator (test_installation.py)

## Syllabus Coverage (Unit III)
- CNN-based visual feature extraction
- Transfer Learning with pretrained YOLOv8 weights
- Object Detection concepts (bounding boxes, classes, confidence, IoU)
- Real-time deep learning inference pipeline

---

## 1) System Requirements

### Software
- Python **3.9+** (recommended: 3.10 or 3.11)
- pip (latest)
- VS Code

### Operating Systems
- Windows 10/11
- macOS 12+
- Linux (Ubuntu 20.04+ or equivalent)

### Hardware (Recommended)
- CPU: Intel i5 / Ryzen 5 or better
- RAM: Minimum 8 GB (16 GB recommended)
- GPU: Optional NVIDIA GPU with CUDA support for faster inference
- Webcam: Built-in or external USB webcam

---

## 2) Installation Instructions

### A) Windows (PowerShell)
```powershell
# 1) Create and enter project folder
mkdir yolo
cd yolo

# 2) Create files/folders quickly
ni main.py, config.py, utils.py, requirements.txt, test_installation.py, README.md, .gitignore -ItemType File
mkdir models, outputs, logs, outputs\videos, outputs\screenshots
ni models\.gitkeep, outputs\videos\.gitkeep, outputs\screenshots\.gitkeep, logs\.gitkeep -ItemType File

# 3) Create virtual environment
python -m venv .venv

# 4) Activate virtual environment
.\.venv\Scripts\Activate.ps1

# 5) Verify venv is active
python -c "import sys; print(sys.prefix)"

# 6) Upgrade pip
python -m pip install --upgrade pip

# 7) Install dependencies
pip install -r requirements.txt

# 8) Download/check YOLO model once
python -c "from ultralytics import YOLO; YOLO('yolov8n.pt'); print('Model ready')"

# 9) Verify full installation
python test_installation.py
```

### B) macOS (zsh/bash)
```bash
# 1) Create and enter project folder
mkdir -p yolo && cd yolo

# 2) Create files/folders quickly
touch main.py config.py utils.py requirements.txt test_installation.py README.md .gitignore
mkdir -p models outputs/videos outputs/screenshots logs
touch models/.gitkeep outputs/videos/.gitkeep outputs/screenshots/.gitkeep logs/.gitkeep

# 3) Create virtual environment
python3 -m venv .venv

# 4) Activate virtual environment
source .venv/bin/activate

# 5) Verify venv is active
python -c "import sys; print(sys.prefix)"

# 6) Upgrade pip
python -m pip install --upgrade pip

# 7) Install dependencies
pip install -r requirements.txt

# 8) Download/check YOLO model once
python -c "from ultralytics import YOLO; YOLO('yolov8n.pt'); print('Model ready')"

# 9) Verify full installation
python test_installation.py
```

### C) Linux (bash)
```bash
# 1) Create and enter project folder
mkdir -p yolo && cd yolo

# 2) Create files/folders quickly
touch main.py config.py utils.py requirements.txt test_installation.py README.md .gitignore
mkdir -p models outputs/videos outputs/screenshots logs
touch models/.gitkeep outputs/videos/.gitkeep outputs/screenshots/.gitkeep logs/.gitkeep

# 3) Create virtual environment
python3 -m venv .venv

# 4) Activate virtual environment
source .venv/bin/activate

# 5) Verify venv is active
python -c "import sys; print(sys.prefix)"

# 6) Upgrade pip
python -m pip install --upgrade pip

# 7) Install dependencies
pip install -r requirements.txt

# 8) Download/check YOLO model once
python -c "from ultralytics import YOLO; YOLO('yolov8n.pt'); print('Model ready')"

# 9) Verify full installation
python test_installation.py
```

---

## 3) Project Structure

```text
project/
├── main.py
├── config.py
├── utils.py
├── requirements.txt
├── test_installation.py
├── README.md
├── .gitignore
├── models/
│   └── .gitkeep
├── outputs/
│   ├── videos/
│   │   └── .gitkeep
│   └── screenshots/
│       └── .gitkeep
└── logs/
    └── .gitkeep
```

### File Roles
- main.py: Entry point (Phase 2 will contain full real-time loop).
- config.py: Centralized class-based configuration.
- utils.py: Reusable helper functions.
- requirements.txt: Python dependency list.
- test_installation.py: Complete environment and webcam/model verification.
- .gitignore: Ignores local caches, outputs, model weights, and IDE files.

---

## 4) Usage Instructions

### Run Setup Validation
```bash
python test_installation.py
```

### Run Main Script
```bash
python main.py
```

### Expected Runtime Controls (for upcoming real-time script)
- q: Quit application
- s: Save screenshot (if enabled)
- r: Toggle recording (if implemented in Phase 2)

---

## 5) Configuration Guide (config.py)

Update settings in AppConfig defaults:
- Model threshold tuning:
  - confidence_threshold (higher = fewer false positives)
  - iou_threshold (NMS overlap control)
- Webcam tuning:
  - frame_width/frame_height for speed vs quality
  - target_fps for smoother video
- Visualization:
  - box_color, text_color, box_thickness

### Example Config Profiles
- Fast CPU profile: 640x480, confidence 0.5, no recording
- Better quality profile: 1280x720, confidence 0.4, GPU enabled
- Demo profile: enable FPS display + counting

---

## 6) Troubleshooting

### Webcam not opening
- Close Zoom/Teams/Meet/OBS and retry.
- Change camera_id from 0 to 1 in config.py.
- Verify camera permissions in OS privacy settings.

### CUDA/GPU not detected
- Install CUDA-compatible PyTorch from official selector.
- Update NVIDIA driver.
- Check with:
```bash
python -c "import torch; print(torch.cuda.is_available()); print(torch.cuda.device_count())"
```

### Import errors
- Confirm venv is active.
- Reinstall:
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### YOLO model download issues
- Ensure internet is available for first run.
- Retry:
```bash
python -c "from ultralytics import YOLO; YOLO('yolov8n.pt')"
```

### Performance optimization tips
- Use lower resolution (640x480).
- Use yolov8n.pt for fastest inference.
- Use GPU when available.
- Close background apps consuming CPU/RAM.

---

## 7) OODA Loop and YOLOv8

### OODA = Observe, Orient, Decide, Act
- Observe: Webcam captures scene.
- Orient: YOLOv8 extracts features and identifies objects.
- Decide: System selects detections over confidence threshold.
- Act: Draw boxes/labels and optionally save output.

This maps directly to real-time deep learning inference workflow.

---

## 8) Expected Output

When running detection, you should see:
- Live webcam video window
- Bounding boxes around detected objects
- Class labels (example: person, laptop, bottle)
- Confidence values (example: 0.87)

### How to interpret confidence
- 0.90: Very confident prediction
- 0.60: Moderate confidence
- < 0.50: Usually filtered out by threshold

---

## 9) Deep Learning Concepts Covered

### CNN Architecture
YOLOv8 uses convolutional layers to learn spatial visual patterns like edges, textures, and object parts.

### Transfer Learning
Instead of training from scratch, pretrained weights are reused to solve your detection task immediately.

### Object Detection
Unlike simple classification, detection predicts:
- What object is present (class)
- Where it is (bounding box coordinates)
- How sure the model is (confidence score)

---

## 10) Viva Preparation

### Common Questions
1. Why YOLOv8 for real-time detection?
   - It is fast, accurate, and easy to deploy with pretrained weights.

2. Why use transfer learning?
   - It saves time/compute and works well when custom training data is not available.

3. What does IoU threshold do?
   - It controls Non-Maximum Suppression and removes duplicate overlapping boxes.

4. CPU vs GPU inference difference?
   - GPU gives lower latency and higher FPS due to parallel computation.

5. Why confidence threshold is needed?
   - To suppress weak/uncertain predictions and reduce false positives.

### Demo Talking Points
- Show webcam detection in real time.
- Explain one detected object with confidence.
- Change a threshold in config.py and show effect live.

---

## 11) Virtual Environment Guide

### Why you need a venv
- Prevents dependency conflicts across projects.
- Keeps project packages isolated.
- Makes setup reproducible.

### Windows
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -c "import sys; print(sys.prefix)"
deactivate
```

### macOS/Linux
```bash
python3 -m venv .venv
source .venv/bin/activate
python -c "import sys; print(sys.prefix)"
deactivate
```

### Common venv issues
- Activation blocked on PowerShell:
```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
```
- python3 not found: install Python 3 and retry.
- Wrong pip path: run python -m pip ... inside activated venv.

### Best practices
- Always activate venv before install/run.
- Keep requirements.txt updated.
- Avoid global pip install for project dependencies.

---

## 12) Installation Commands Summary (Copy-Paste)

### Windows
```powershell
mkdir yolo
cd yolo
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
python -c "from ultralytics import YOLO; YOLO('yolov8n.pt')"
python test_installation.py
python main.py
```

### macOS
```bash
mkdir -p yolo && cd yolo
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
python -c "from ultralytics import YOLO; YOLO('yolov8n.pt')"
python test_installation.py
python main.py
```

### Linux
```bash
mkdir -p yolo && cd yolo
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
python -c "from ultralytics import YOLO; YOLO('yolov8n.pt')"
python test_installation.py
python main.py
```

---

## 13) Author and Assignment Info
- Author: Your Name
- Date Created: 03 April 2026
- Course: Deep Learning
- Assignment: Mini Project - YOLOv8 Real-Time Object Detection System
- Scope: Phase 1 (Setup and Environment)
