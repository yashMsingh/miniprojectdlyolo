"""Centralized configuration for the YOLOv8 real-time detection system."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path
from typing import Any, Dict, Tuple

import cv2

try:
    import torch
except Exception:  # pragma: no cover - fallback for environments without torch
    torch = None


@dataclass
class ModelConfig:
    """Model-related settings."""

    model_path: str = "yolov8n.pt"
    confidence_threshold: float = 0.5
    iou_threshold: float = 0.45
    device: str = "cpu"


@dataclass
class WebcamConfig:
    """Webcam capture settings."""

    camera_id: int = 0
    frame_width: int = 640
    frame_height: int = 480
    target_fps: int = 30
    flip_frame: bool = True


@dataclass
class OutputConfig:
    """Output and recording settings."""

    output_dir: str = "./outputs"
    save_video: bool = False
    save_screenshots: bool = False
    video_codec: str = "mp4v"
    video_fps: int = 30


@dataclass
class DetectionConfig:
    """Detection logic options."""

    enable_counting: bool = True
    enable_fps_display: bool = True
    enable_class_filter: bool = False
    min_confidence: float = 0.5


@dataclass
class UIConfig:
    """Visualization and annotation settings."""

    box_thickness: int = 2
    text_font: int = cv2.FONT_HERSHEY_SIMPLEX
    text_font_size: float = 0.7
    text_color: Tuple[int, int, int] = (0, 255, 0)  # Green (BGR)
    box_color: Tuple[int, int, int] = (0, 255, 0)  # Green (BGR)
    fps_color: Tuple[int, int, int] = (0, 0, 255)  # Red (BGR)
    label_bg_color: Tuple[int, int, int] = (0, 0, 0)  # Black (BGR)


@dataclass
class LoggingConfig:
    """Logging behavior settings."""

    log_dir: str = "./logs"
    log_level: str = "INFO"
    enable_logging: bool = True


@dataclass
class AppConfig:
    """Main app configuration object that groups all sections."""

    model: ModelConfig
    webcam: WebcamConfig
    output: OutputConfig
    detection: DetectionConfig
    ui: UIConfig
    logging: LoggingConfig

    @staticmethod
    def _auto_detect_device() -> str:
        """Return 'cuda' when available, else 'cpu'."""
        if torch is None:
            return "cpu"
        return "cuda" if torch.cuda.is_available() else "cpu"

    @classmethod
    def default(cls) -> "AppConfig":
        """Create default config with automatic device detection."""
        return cls(
            model=ModelConfig(device=cls._auto_detect_device()),
            webcam=WebcamConfig(),
            output=OutputConfig(),
            detection=DetectionConfig(),
            ui=UIConfig(),
            logging=LoggingConfig(),
        )

    def to_dict(self) -> Dict[str, Any]:
        """Convert full config to plain dictionary."""
        return asdict(self)

    def save_json(self, json_path: str = "config.json") -> None:
        """Save config to a JSON file.

        Args:
            json_path: Target file path.
        """
        path = Path(json_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "AppConfig":
        """Create AppConfig from dictionary data."""
        return cls(
            model=ModelConfig(**data.get("model", {})),
            webcam=WebcamConfig(**data.get("webcam", {})),
            output=OutputConfig(**data.get("output", {})),
            detection=DetectionConfig(**data.get("detection", {})),
            ui=UIConfig(**data.get("ui", {})),
            logging=LoggingConfig(**data.get("logging", {})),
        )

    @classmethod
    def load_json(cls, json_path: str = "config.json") -> "AppConfig":
        """Load config from JSON file. Falls back to defaults if file is missing."""
        path = Path(json_path)
        if not path.exists():
            return cls.default()
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        cfg = cls.from_dict(data)
        if not cfg.model.device:
            cfg.model.device = cls._auto_detect_device()
        return cfg


if __name__ == "__main__":
    # Example usage:
    # 1) Create defaults and print them.
    config = AppConfig.default()
    print("Loaded default configuration:")
    print(json.dumps(config.to_dict(), indent=2))

    # 2) Save to config.json, then load back.
    config.save_json("config.json")
    loaded_config = AppConfig.load_json("config.json")
    print("\nReloaded configuration device:", loaded_config.model.device)
