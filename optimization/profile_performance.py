"""Performance profiling script for YOLOv8 application."""

from __future__ import annotations

import time
from typing import List

import numpy as np

from config import AppConfig
from detector.main_detector import MainDetector
from utils import check_memory, get_system_info


def profile_application() -> None:
    """Profile initialization and inference performance using synthetic frames."""
    print("=" * 60)
    print("PERFORMANCE PROFILING")
    print("=" * 60)

    print("\nSYSTEM INFORMATION")
    print("-" * 60)
    sys_info = get_system_info()
    print(f"Platform: {sys_info.get('platform')}")
    print(f"Python Version: {sys_info.get('python_version')}")
    mem = check_memory()
    print(f"Available Memory: {mem.get('available_gb')} GB")

    config = AppConfig.default()
    print("\nCONFIGURATION")
    print("-" * 60)
    print(f"Model: {config.model.model_path}")
    print(f"Resolution: {config.webcam.frame_width}x{config.webcam.frame_height}")
    print(f"Target FPS: {config.webcam.target_fps}")

    print("\nINITIALIZATION TIME")
    print("-" * 60)
    t0 = time.time()
    detector = MainDetector(config=config)
    init_time_ms = (time.time() - t0) * 1000.0
    print(f"Detector Initialization: {init_time_ms:.2f} ms")

    inference_times_s: List[float] = []
    frame_times_s: List[float] = []
    num_frames = 30

    print("\nINFERENCE PERFORMANCE")
    print("-" * 60)
    print(f"Running {num_frames} synthetic frames...")

    try:
        for i in range(num_frames):
            frame = np.random.randint(
                0,
                255,
                (config.webcam.frame_height, config.webcam.frame_width, 3),
                dtype=np.uint8,
            )
            frame_start = time.time()
            _, inference_ms = detector.run_inference(frame)
            elapsed = time.time() - frame_start
            frame_times_s.append(elapsed)
            inference_times_s.append(inference_ms / 1000.0)

            if (i + 1) % 10 == 0:
                print(f"Progress: {i + 1}/{num_frames}")
    finally:
        detector.shutdown()

    print("\nRESULTS")
    print("-" * 60)

    avg_inference_ms = float(np.mean(inference_times_s) * 1000.0) if inference_times_s else 0.0
    avg_frame_ms = float(np.mean(frame_times_s) * 1000.0) if frame_times_s else 0.0
    avg_fps = 1000.0 / avg_frame_ms if avg_frame_ms > 0 else 0.0
    min_fps = 1.0 / max(frame_times_s) if frame_times_s else 0.0
    max_fps = 1.0 / min(frame_times_s) if frame_times_s else 0.0

    print(f"Average Inference Time: {avg_inference_ms:.2f} ms")
    print(f"Average Frame Time: {avg_frame_ms:.2f} ms")
    print(f"Average FPS: {avg_fps:.2f}")
    print(f"Min FPS: {min_fps:.2f}")
    print(f"Max FPS: {max_fps:.2f}")

    print("\nRECOMMENDATIONS")
    print("-" * 60)
    if avg_fps < 20:
        print("FPS below 20. Consider: nano model, lower resolution, GPU, less overlays.")
    elif avg_fps < 25:
        print("FPS below 25. Consider reducing overlays or disabling demo mode.")
    else:
        print("FPS is good (>25).")

    print("\n" + "=" * 60)


if __name__ == "__main__":
    profile_application()
