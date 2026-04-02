"""Video recording and screenshot capture utilities."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Optional, Tuple

import cv2
import numpy as np


class VideoRecorder:
    """Handles video file writing and screenshot capture."""

    def __init__(self, logger: Any) -> None:
        """Initialize video recorder.

        Args:
            logger: Logger instance.
        """
        self.logger = logger
        self.video_writer: Optional[cv2.VideoWriter] = None
        self.is_recording = False
        self.recording_start_time: Optional[float] = None
        self.output_filepath: Optional[str] = None
        self.frame_count = 0

    def start_recording(
        self,
        output_path: str,
        fps: float = 30.0,
        frame_width: int = 640,
        frame_height: int = 480,
        codec: str = "mp4v",
    ) -> bool:
        """Start video recording.

        Args:
            output_path: Where to save video file.
            fps: Frames per second.
            frame_width: Video width in pixels.
            frame_height: Video height in pixels.
            codec: Video codec (mp4v, MJPG, etc).

        Returns:
            True if recording started successfully.
        """
        try:
            Path(output_path).parent.mkdir(parents=True, exist_ok=True)

            fourcc = cv2.VideoWriter_fourcc(*codec)
            self.video_writer = cv2.VideoWriter(
                str(output_path),
                fourcc,
                fps,
                (frame_width, frame_height),
            )

            if not self.video_writer.isOpened():
                self.logger.error(f"VideoWriter failed to open: {output_path}")
                self.video_writer = None
                return False

            self.is_recording = True
            self.recording_start_time = time.perf_counter()
            self.output_filepath = output_path
            self.frame_count = 0
            self.logger.info(f"Video recording started: {output_path}")
            return True
        except Exception as exc:
            self.logger.exception(f"Failed to start video recording: {exc}")
            self.video_writer = None
            self.is_recording = False
            return False

    def write_frame(self, frame: np.ndarray) -> bool:
        """Write single frame to video file.

        Args:
            frame: Frame to write.

        Returns:
            True if write succeeded.
        """
        if not self.is_recording or self.video_writer is None:
            return False

        try:
            self.video_writer.write(frame)
            self.frame_count += 1
            return True
        except Exception as exc:
            self.logger.error(f"Frame write failed: {exc}")
            return False

    def stop_recording(self) -> Tuple[bool, Optional[str]]:
        """Stop recording and close video file.

        Returns:
            Tuple of (success, output_filepath).
        """
        if not self.is_recording or self.video_writer is None:
            return False, None

        try:
            self.video_writer.release()
            self.video_writer = None
            self.is_recording = False

            duration = time.perf_counter() - self.recording_start_time
            self.logger.info(
                f"Video recording stopped: {self.output_filepath} "
                f"({self.frame_count} frames, {duration:.2f}s)"
            )
            return True, self.output_filepath
        except Exception as exc:
            self.logger.exception(f"Failed to stop video recording: {exc}")
            return False, None

    def get_recording_duration(self) -> float:
        """Get elapsed recording time in seconds."""
        if not self.is_recording or self.recording_start_time is None:
            return 0.0
        return time.perf_counter() - self.recording_start_time

    def save_screenshot(
        self, frame: np.ndarray, output_dir: str = "./outputs/screenshots"
    ) -> Tuple[bool, Optional[str]]:
        """Save frame as PNG screenshot.

        Args:
            frame: Frame to save.
            output_dir: Output directory.

        Returns:
            Tuple of (success, filepath).
        """
        try:
            Path(output_dir).mkdir(parents=True, exist_ok=True)
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            filepath = Path(output_dir) / f"detection_{timestamp}.png"

            success = cv2.imwrite(str(filepath), frame)
            if success:
                self.logger.info(f"Screenshot saved: {filepath}")
                return True, str(filepath)
            else:
                self.logger.warning(f"Failed to write screenshot: {filepath}")
                return False, None
        except Exception as exc:
            self.logger.exception(f"Screenshot save failed: {exc}")
            return False, None

    def reset_writer(self) -> None:
        """Clean up recording resources."""
        if self.video_writer is not None:
            try:
                self.video_writer.release()
            except Exception:
                pass
            self.video_writer = None
        self.is_recording = False
        self.recording_start_time = None
        self.output_filepath = None
        self.frame_count = 0
