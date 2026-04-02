"""Base detector implementation for model loading and camera lifecycle."""

from __future__ import annotations

from dataclasses import asdict
from typing import Any, Dict, Optional

import cv2
from ultralytics import YOLO

from config import AppConfig
from utils import get_device, setup_logger


class BaseDetector:
    """Handles shared detector initialization tasks.

    Responsibilities:
    - Load and validate configuration.
    - Load YOLO model and move execution to selected device.
    - Initialize and validate webcam capture.
    - Expose lifecycle helpers for cleanup.
    """

    def __init__(self, config: Optional[AppConfig] = None) -> None:
        """Initialize model, logger, and camera state.

        Args:
            config: Optional prebuilt AppConfig. If None, defaults are used.

        Raises:
            RuntimeError: If model loading fails.
        """
        self.config = config or AppConfig.default()
        self.logger = setup_logger(
            name=self.__class__.__name__,
            log_dir=self.config.logging.log_dir,
            log_level=self.config.logging.log_level,
            enable_logging=self.config.logging.enable_logging,
        )

        self.model: Optional[YOLO] = None
        self.cap: Optional[cv2.VideoCapture] = None
        self.frame_count: int = 0
        self.is_running: bool = False

        self._load_model()

    def _load_model(self) -> None:
        """Load YOLO model and validate model readiness."""
        try:
            # Use config-driven model path and device for portability and reproducibility.
            self.config.model.device = get_device(self.config.model.device)
            self.model = YOLO(self.config.model.model_path)
            self.logger.info(
                "Model loaded successfully: %s on device=%s",
                self.config.model.model_path,
                self.config.model.device,
            )
            model_info = self.get_model_info()
            self.logger.info("Model info: %s", model_info)
        except Exception as exc:
            message = f"Failed to load model '{self.config.model.model_path}': {exc}"
            self.logger.exception(message)
            raise RuntimeError(message) from exc

    def get_model_info(self) -> Dict[str, Any]:
        """Return model metadata (parameters, layers, and model path)."""
        if self.model is None:
            return {"loaded": False, "error": "model is not initialized"}

        try:
            net = getattr(self.model, "model", None)
            params = sum(p.numel() for p in net.parameters()) if net is not None else 0
            trainable = (
                sum(p.numel() for p in net.parameters() if p.requires_grad)
                if net is not None
                else 0
            )
            layers = len(list(net.modules())) if net is not None else 0

            return {
                "loaded": True,
                "model_path": self.config.model.model_path,
                "device": self.config.model.device,
                "parameters": params,
                "trainable_parameters": trainable,
                "layers": layers,
                "confidence_threshold": self.config.model.confidence_threshold,
                "iou_threshold": self.config.model.iou_threshold,
            }
        except Exception as exc:
            self.logger.warning("Unable to collect model metadata: %s", exc)
            return {"loaded": True, "warning": str(exc)}

    def check_camera_available(self) -> bool:
        """Check whether camera can be opened with configured camera id."""
        temp_cap = cv2.VideoCapture(self.config.webcam.camera_id)
        available = temp_cap.isOpened()
        temp_cap.release()
        return available

    def initialize_camera(self) -> cv2.VideoCapture:
        """Initialize webcam capture with configured properties.

        Returns:
            Initialized cv2.VideoCapture object.

        Raises:
            RuntimeError: If camera is unavailable or cannot stream frames.
        """
        try:
            self.cap = cv2.VideoCapture(self.config.webcam.camera_id)
            if not self.cap.isOpened():
                raise RuntimeError(
                    f"Camera index {self.config.webcam.camera_id} could not be opened."
                )

            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.config.webcam.frame_width)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.config.webcam.frame_height)
            self.cap.set(cv2.CAP_PROP_FPS, self.config.webcam.target_fps)

            ok, _ = self.cap.read()
            if not ok:
                raise RuntimeError("Camera opened but failed to read initial frame.")

            self.logger.info(
                "Camera initialized: id=%d, width=%d, height=%d, target_fps=%d",
                self.config.webcam.camera_id,
                self.config.webcam.frame_width,
                self.config.webcam.frame_height,
                self.config.webcam.target_fps,
            )
            return self.cap
        except Exception as exc:
            if self.cap is not None:
                self.cap.release()
                self.cap = None
            message = f"Camera initialization failed: {exc}"
            self.logger.exception(message)
            raise RuntimeError(message) from exc

    def release_resources(self) -> None:
        """Release camera and close OpenCV windows safely."""
        try:
            if self.cap is not None:
                self.cap.release()
                self.cap = None
            cv2.destroyAllWindows()
            self.is_running = False
            self.logger.info("Resources released successfully.")
        except Exception as exc:
            self.logger.warning("Resource release encountered an issue: %s", exc)

    def config_as_dict(self) -> Dict[str, Any]:
        """Return serializable configuration snapshot for debug logging."""
        try:
            return asdict(self.config)
        except Exception:
            return {"warning": "config serialization failed"}
