"""Shared utility functions for logging, device checks, paths, and frame ops."""

from __future__ import annotations

import logging
import os
import platform
from pathlib import Path
import sys
import time
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

try:
    import torch
except Exception:
    torch = None


def setup_logger(
    name: str = "yolo_detector",
    log_dir: str = "./logs",
    log_level: str = "INFO",
    enable_logging: bool = True,
) -> logging.Logger:
    """Configure and return logger instance.

    Args:
        name: Logger name.
        log_dir: Directory for log files.
        log_level: Logging level (e.g., INFO, DEBUG).
        enable_logging: If False, logger remains quiet.
    """
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger

    level = getattr(logging, str(log_level).upper(), logging.INFO)
    logger.setLevel(level)
    logger.propagate = False

    if not enable_logging:
        logger.addHandler(logging.NullHandler())
        return logger

    Path(log_dir).mkdir(parents=True, exist_ok=True)
    fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(name)s | %(message)s")

    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(fmt)
    logger.addHandler(stream_handler)

    file_handler = logging.FileHandler(Path(log_dir) / f"{name}.log", encoding="utf-8")
    file_handler.setFormatter(fmt)
    logger.addHandler(file_handler)

    return logger


def log_message(logger: logging.Logger, level: str, message: str) -> None:
    """Log a message with dynamic log level."""
    log_fn = getattr(logger, level.lower(), logger.info)
    log_fn(message)


def print_separator(title: str = "", width: int = 80, fill: str = "=") -> str:
    """Return a formatted separator for CLI output."""
    if not title.strip():
        return fill * width
    return f" {title.strip()} ".center(width, fill)


def ensure_directories(paths: Iterable[str]) -> None:
    """Create multiple directories if missing."""
    for p in paths:
        Path(p).mkdir(parents=True, exist_ok=True)


def get_device(preferred_device: Optional[str] = None) -> str:
    """Return effective execution device based on availability.

    Args:
        preferred_device: Optional requested device.

    Returns:
        "cuda" when available and requested/auto, otherwise "cpu".
    """
    requested = (preferred_device or "").strip().lower()
    if requested in {"cuda", "gpu"}:
        return "cuda" if torch is not None and torch.cuda.is_available() else "cpu"
    if requested == "cpu":
        return "cpu"
    return "cuda" if torch is not None and torch.cuda.is_available() else "cpu"


def get_system_info() -> Dict[str, Any]:
    """Return operating system and Python runtime details."""
    info: Dict[str, Any] = {
        "platform": platform.system(),
        "platform_release": platform.release(),
        "platform_version": platform.version(),
        "architecture": platform.machine(),
        "python_version": platform.python_version(),
        "cpu_count": os.cpu_count() or 0,
    }

    if torch is not None:
        info["torch_version"] = getattr(torch, "__version__", "unknown")
        info["cuda_available"] = bool(torch.cuda.is_available())
        info["cuda_device_count"] = int(torch.cuda.device_count()) if torch.cuda.is_available() else 0
    else:
        info["torch_version"] = "unavailable"
        info["cuda_available"] = False
        info["cuda_device_count"] = 0

    return info


def check_memory() -> Dict[str, Optional[float]]:
    """Return memory snapshot in GB (best-effort without external dependencies)."""
    total_gb: Optional[float] = None
    avail_gb: Optional[float] = None

    try:
        if platform.system() == "Windows":
            import ctypes

            class MEMORYSTATUSEX(ctypes.Structure):
                _fields_ = [
                    ("dwLength", ctypes.c_ulong),
                    ("dwMemoryLoad", ctypes.c_ulong),
                    ("ullTotalPhys", ctypes.c_ulonglong),
                    ("ullAvailPhys", ctypes.c_ulonglong),
                    ("ullTotalPageFile", ctypes.c_ulonglong),
                    ("ullAvailPageFile", ctypes.c_ulonglong),
                    ("ullTotalVirtual", ctypes.c_ulonglong),
                    ("ullAvailVirtual", ctypes.c_ulonglong),
                    ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
                ]

            stat = MEMORYSTATUSEX()
            stat.dwLength = ctypes.sizeof(MEMORYSTATUSEX)
            ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(stat))
            total_gb = round(stat.ullTotalPhys / (1024**3), 2)
            avail_gb = round(stat.ullAvailPhys / (1024**3), 2)
        elif hasattr(os, "sysconf"):
            pages = os.sysconf("SC_PHYS_PAGES")
            page_size = os.sysconf("SC_PAGE_SIZE")
            avail_pages = os.sysconf("SC_AVPHYS_PAGES") if hasattr(os, "sysconf") else 0
            total_gb = round((pages * page_size) / (1024**3), 2)
            avail_gb = round((avail_pages * page_size) / (1024**3), 2)
    except Exception:
        pass

    return {"total_gb": total_gb, "available_gb": avail_gb}


def ensure_path_exists(path_str: str) -> Path:
    """Create path if it does not exist and return Path object."""
    path = Path(path_str)
    path.mkdir(parents=True, exist_ok=True)
    return path


def get_safe_path(path_str: str, create: bool = False) -> Path:
    """Return validated absolute path.

    Args:
        path_str: Input path string.
        create: Create directory when missing.
    """
    if not path_str or not str(path_str).strip():
        raise ValueError("Path must be a non-empty string.")

    path = Path(path_str).expanduser().resolve()
    if create:
        path.mkdir(parents=True, exist_ok=True)
    return path


def create_output_dirs(base_output_dir: str = "./outputs") -> Dict[str, str]:
    """Create output root and required subfolders.

    Returns:
        Dict with resolved output paths.
    """
    root = ensure_path_exists(base_output_dir)
    videos = ensure_path_exists(str(root / "videos"))
    screenshots = ensure_path_exists(str(root / "screenshots"))
    return {
        "output": str(root),
        "videos": str(videos),
        "screenshots": str(screenshots),
    }


def validate_frame(frame: Any) -> bool:
    """Validate frame object before inference or display."""
    return isinstance(frame, np.ndarray) and frame.size > 0 and frame.ndim in (2, 3)


def get_frame_shape(frame: np.ndarray) -> Tuple[int, ...]:
    """Return frame shape safely."""
    if not validate_frame(frame):
        raise ValueError("Invalid frame passed to get_frame_shape.")
    return frame.shape


def calculate_aspect_ratio(frame: np.ndarray) -> float:
    """Calculate frame aspect ratio as width / height."""
    h, w = frame.shape[:2]
    if h == 0:
        return 0.0
    return w / float(h)


def format_inference_time(inference_ms: float) -> str:
    """Format inference time with equivalent FPS."""
    if inference_ms <= 0:
        return "0.00 ms (0.00 FPS)"
    fps = 1000.0 / inference_ms
    return f"{inference_ms:.2f} ms ({fps:.2f} FPS)"


def calculate_fps(process_seconds: float) -> float:
    """Convert processing time (seconds) to frames per second."""
    if process_seconds <= 0:
        return 0.0
    return 1.0 / process_seconds


def get_inference_stats(inference_history_ms: Sequence[float]) -> Dict[str, float]:
    """Return summary stats for inference history in milliseconds."""
    if not inference_history_ms:
        return {"avg_ms": 0.0, "min_ms": 0.0, "max_ms": 0.0, "avg_fps": 0.0}

    avg_ms = sum(inference_history_ms) / len(inference_history_ms)
    return {
        "avg_ms": avg_ms,
        "min_ms": min(inference_history_ms),
        "max_ms": max(inference_history_ms),
        "avg_fps": (1000.0 / avg_ms) if avg_ms > 0 else 0.0,
    }


def validate_box_coordinates(
    box: Any,
    frame_shape: Tuple[int, ...],
) -> bool:
    """Validate bounding box coordinates against frame bounds."""
    if box is None or len(box) != 4:
        return False

    h, w = frame_shape[:2]
    x1, y1, x2, y2 = [int(v) for v in box]
    if x1 < 0 or y1 < 0 or x2 < 0 or y2 < 0:
        return False
    if x1 >= w or x2 > w or y1 >= h or y2 > h:
        return False
    if x2 <= x1 or y2 <= y1:
        return False
    return True


def calculate_box_area(box: Tuple[int, int, int, int]) -> int:
    """Compute area of a bounding box."""
    x1, y1, x2, y2 = box
    return max(0, x2 - x1) * max(0, y2 - y1)


def check_box_overlap(
    box1: Tuple[int, int, int, int],
    box2: Tuple[int, int, int, int],
    iou_threshold: float = 0.5,
) -> bool:
    """Return True when IoU between two boxes is above threshold."""
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    inter = max(0, x2 - x1) * max(0, y2 - y1)
    if inter == 0:
        return False

    area1 = calculate_box_area(box1)
    area2 = calculate_box_area(box2)
    union = area1 + area2 - inter
    if union <= 0:
        return False

    iou = inter / union
    return iou >= iou_threshold


def get_class_name(class_id: int, names: Any) -> str:
    """Map class id to class name with fallback."""
    if isinstance(names, dict):
        return str(names.get(class_id, class_id))
    if isinstance(names, list) and 0 <= class_id < len(names):
        return str(names[class_id])
    return str(class_id)


def yolo_results_to_dict(yolo_results: Any) -> List[Dict[str, Any]]:
    """Convert YOLO result object list into plain dict list."""
    output: List[Dict[str, Any]] = []
    if not yolo_results:
        return output

    result = yolo_results[0]
    names = getattr(result, "names", {})
    boxes = getattr(result, "boxes", None)
    if boxes is None:
        return output

    xyxy = boxes.xyxy.cpu().numpy()
    conf = boxes.conf.cpu().numpy()
    cls_ids = boxes.cls.cpu().numpy().astype(int)

    for i in range(len(xyxy)):
        x1, y1, x2, y2 = [int(v) for v in xyxy[i]]
        class_id = int(cls_ids[i])
        output.append(
            {
                "box": (x1, y1, x2, y2),
                "confidence": float(conf[i]),
                "class_id": class_id,
                "class_name": get_class_name(class_id, names),
            }
        )
    return output


def filter_by_confidence(
    detections: List[Dict[str, Any]],
    threshold: float,
) -> List[Dict[str, Any]]:
    """Return detections with confidence >= threshold."""
    return [d for d in detections if float(d.get("confidence", 0.0)) >= threshold]


def banner(text: str, width: int = 70, fill: str = "=") -> str:
    """Backward-compatible banner helper."""
    clean = f" {text.strip()} "
    return clean.center(width, fill)
