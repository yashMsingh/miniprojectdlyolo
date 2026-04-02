"""Detector package for YOLOv8 real-time object detection."""

from .base_detector import BaseDetector
from .camera_handler import CameraHandler
from .detection_processor import DetectionProcessor
from .inference_pipeline import InferencePipeline
from .main_detector import MainDetector

__all__ = [
    "BaseDetector",
    "CameraHandler",
    "DetectionProcessor",
    "InferencePipeline",
    "MainDetector",
]
