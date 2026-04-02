"""Camera capture utility with reconnect and frame metadata support."""

from __future__ import annotations

import time
from collections import deque
from typing import Any, Deque, Dict, Optional, Tuple

import cv2
import numpy as np

from config import AppConfig
from utils import calculate_fps, validate_frame


class CameraHandler:
    """Wraps camera operations to keep capture logic isolated."""

    def __init__(
        self,
        cap: cv2.VideoCapture,
        config: AppConfig,
        logger,
        reconnect_attempts: int = 3,
    ) -> None:
        """Initialize camera handler.

        Args:
            cap: Active OpenCV capture object.
            config: Application configuration.
            logger: Logger instance.
            reconnect_attempts: Number of reconnect retries on failure.
        """
        self.cap = cap
        self.config = config
        self.logger = logger
        self.reconnect_attempts = reconnect_attempts

        self.frame_count: int = 0
        self.last_read_timestamp: float = time.perf_counter()
        self.processing_times_ms: Deque[float] = deque(maxlen=120)

    def _reconnect(self) -> bool:
        """Try reconnecting the camera if stream is lost."""
        for attempt in range(1, self.reconnect_attempts + 1):
            try:
                if self.cap is not None:
                    self.cap.release()
                self.cap = cv2.VideoCapture(self.config.webcam.camera_id)
                self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.config.webcam.frame_width)
                self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.config.webcam.frame_height)
                self.cap.set(cv2.CAP_PROP_FPS, self.config.webcam.target_fps)

                if self.cap.isOpened():
                    self.logger.warning("Camera reconnected on attempt %d", attempt)
                    return True
                self.logger.warning("Reconnect attempt %d failed", attempt)
            except Exception as exc:
                self.logger.error("Reconnect attempt %d raised: %s", attempt, exc)
        return False

    def read_frame(self) -> Tuple[bool, Optional[np.ndarray]]:
        """Read a frame from camera.

        Returns:
            Tuple of (success, frame).
        """
        start = time.perf_counter()
        try:
            if self.cap is None or not self.cap.isOpened():
                self.logger.error("Camera not opened. Attempting reconnect.")
                if not self._reconnect():
                    return False, None

            ok, frame = self.cap.read()
            if not ok or frame is None:
                self.logger.warning("Dropped frame detected. Attempting reconnect.")
                if self._reconnect():
                    ok, frame = self.cap.read()
                if not ok or frame is None:
                    return False, None

            if self.config.webcam.flip_frame:
                frame = cv2.flip(frame, 1)

            if not self.is_frame_valid(frame):
                self.logger.warning("Invalid frame shape/content.")
                return False, None

            self.frame_count += 1
            now = time.perf_counter()
            elapsed_ms = (now - start) * 1000.0
            self.processing_times_ms.append(elapsed_ms)
            self.last_read_timestamp = now
            return True, frame
        except Exception as exc:
            self.logger.exception("Frame read failed: %s", exc)
            return False, None

    def get_frame_info(self) -> Dict[str, Any]:
        """Return current frame and camera metadata."""
        width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)) if self.cap else 0
        height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) if self.cap else 0
        camera_fps = float(self.cap.get(cv2.CAP_PROP_FPS)) if self.cap else 0.0
        avg_read_ms = (
            sum(self.processing_times_ms) / len(self.processing_times_ms)
            if self.processing_times_ms
            else 0.0
        )
        computed_fps = calculate_fps(avg_read_ms / 1000.0) if avg_read_ms > 0 else 0.0

        return {
            "frame_count": self.frame_count,
            "resolution": (width, height),
            "camera_fps": camera_fps,
            "read_time_ms": round(avg_read_ms, 3),
            "computed_fps": round(computed_fps, 2),
        }

    @staticmethod
    def is_frame_valid(frame: Optional[np.ndarray]) -> bool:
        """Validate frame object and shape before processing."""
        return validate_frame(frame)

    def release(self) -> None:
        """Release capture resource safely."""
        try:
            if self.cap is not None:
                self.cap.release()
        except Exception as exc:
            self.logger.warning("Camera release failed: %s", exc)
