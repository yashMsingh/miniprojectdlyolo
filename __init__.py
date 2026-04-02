"""YOLOv8 Real-Time Object Detection package."""

from detector import (
    BaseDetector,
    CameraHandler,
    DetectionProcessor,
    InferencePipeline,
    MainDetector,
)
from visualization import FrameDecorator, StatsTracker, Visualizer

__version__ = "0.2.0"

__all__ = [
    "BaseDetector",
    "CameraHandler",
    "DetectionProcessor",
    "InferencePipeline",
    "MainDetector",
    "Visualizer",
    "StatsTracker",
    "FrameDecorator",
]
