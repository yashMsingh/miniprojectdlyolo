"""Frame visualizer for bounding boxes, labels, and overlay stats."""

from __future__ import annotations

from typing import Any, Dict, List, Tuple

import cv2
import numpy as np

from config import AppConfig
from utils import validate_box_coordinates


class Visualizer:
    """Draws detections and diagnostics onto frames."""

    def __init__(self, config: AppConfig, logger) -> None:
        self.config = config
        self.logger = logger

    def draw_box(self, frame: np.ndarray, detection: Dict[str, Any]) -> np.ndarray:
        """Draw a single bounding box on frame."""
        box = detection.get("box")
        if not validate_box_coordinates(box, frame.shape):
            return frame

        x1, y1, x2, y2 = box
        cv2.rectangle(
            frame,
            (x1, y1),
            (x2, y2),
            self.config.ui.box_color,
            self.config.ui.box_thickness,
        )
        return frame

    def draw_label(self, frame: np.ndarray, detection: Dict[str, Any]) -> np.ndarray:
        """Draw class label and confidence near a bounding box."""
        box = detection.get("box")
        if not validate_box_coordinates(box, frame.shape):
            return frame

        x1, y1, _, _ = box
        class_name = detection.get("class_name", "object")
        confidence = detection.get("confidence", 0.0)
        text = f"{class_name}: {confidence * 100:.0f}%"

        (text_w, text_h), baseline = cv2.getTextSize(
            text,
            self.config.ui.text_font,
            self.config.ui.text_font_size,
            1,
        )
        y_text = max(y1 - 8, text_h + 4)

        cv2.rectangle(
            frame,
            (x1, y_text - text_h - baseline - 4),
            (x1 + text_w + 6, y_text + 2),
            self.config.ui.label_bg_color,
            -1,
        )
        cv2.putText(
            frame,
            text,
            (x1 + 3, y_text),
            self.config.ui.text_font,
            self.config.ui.text_font_size,
            self.config.ui.text_color,
            1,
            cv2.LINE_AA,
        )
        return frame

    def draw_fps(self, frame: np.ndarray, fps: float) -> np.ndarray:
        """Draw FPS in the top-left corner."""
        label = f"FPS: {fps:.2f}"
        cv2.putText(
            frame,
            label,
            (10, 25),
            self.config.ui.text_font,
            0.7,
            self.config.ui.fps_color,
            2,
            cv2.LINE_AA,
        )
        return frame

    def draw_stats(self, frame: np.ndarray, stats: Dict[str, Any]) -> np.ndarray:
        """Draw detection counts and timing stats in frame corners."""
        h, w = frame.shape[:2]
        objects = int(stats.get("total_detections", 0))
        inference_ms = float(stats.get("inference_ms", 0.0))
        frame_idx = int(stats.get("frame_count", 0))

        top_right = f"Objects: {objects}"
        bottom_left = f"Inference: {inference_ms:.2f} ms"
        bottom_right = f"Frame: {frame_idx}"

        cv2.putText(frame, top_right, (w - 180, 25), self.config.ui.text_font, 0.6, self.config.ui.text_color, 2, cv2.LINE_AA)
        cv2.putText(frame, bottom_left, (10, h - 12), self.config.ui.text_font, 0.6, self.config.ui.text_color, 2, cv2.LINE_AA)
        cv2.putText(frame, bottom_right, (w - 170, h - 12), self.config.ui.text_font, 0.6, self.config.ui.text_color, 2, cv2.LINE_AA)
        return frame

    def annotate_frame(
        self,
        frame: np.ndarray,
        detections: List[Dict[str, Any]],
        stats: Dict[str, Any],
    ) -> np.ndarray:
        """Apply complete annotation pipeline on a frame."""
        try:
            output = frame.copy()
            for det in detections:
                output = self.draw_box(output, det)
                output = self.draw_label(output, det)

            if self.config.detection.enable_fps_display:
                output = self.draw_fps(output, float(stats.get("fps", 0.0)))

            output = self.draw_stats(output, stats)
            return output
        except Exception as exc:
            self.logger.exception("Frame annotation failed: %s", exc)
            return frame

    def draw_threshold_indicator(self, frame: np.ndarray, threshold: float) -> np.ndarray:
        """Draw current confidence threshold indicator in top-right."""
        try:
            h, w = frame.shape[:2]
            threshold_pct = int(threshold * 100)
            text = f"Confidence: {threshold_pct}%"
            
            (text_w, text_h), baseline = cv2.getTextSize(
                text,
                self.config.ui.text_font,
                0.6,
                1,
            )
            
            x_pos = w - text_w - 15
            y_pos = 50
            
            cv2.rectangle(
                frame,
                (x_pos - 5, y_pos - text_h - 8),
                (w - 10, y_pos + 5),
                (0, 165, 255),  # Orange color
                1,
            )
            cv2.putText(
                frame,
                text,
                (x_pos, y_pos),
                self.config.ui.text_font,
                0.6,
                (0, 165, 255),
                1,
                cv2.LINE_AA,
            )
            return frame
        except Exception as exc:
            self.logger.debug("Threshold indicator failed: %s", exc)
            return frame

    def draw_recording_indicator(self, frame: np.ndarray, recording: bool, duration_sec: float = 0.0) -> np.ndarray:
        """Draw REC indicator when video recording is active."""
        try:
            if not recording:
                return frame
            
            h, w = frame.shape[:2]
            
            # Blinking red dot
            color = (0, 0, 255) if int(duration_sec * 2) % 2 == 0 else (0, 0, 200)
            cv2.circle(frame, (w - 30, 25), 6, color, -1)
            
            # REC text
            text = "REC"
            cv2.putText(
                frame,
                text,
                (w - 55, 32),
                self.config.ui.text_font,
                0.7,
                color,
                2,
                cv2.LINE_AA,
            )
            
            # Duration
            minutes = int(duration_sec // 60)
            seconds = int(duration_sec % 60)
            duration_text = f"{minutes:02d}:{seconds:02d}"
            cv2.putText(
                frame,
                duration_text,
                (w - 55, 50),
                self.config.ui.text_font,
                0.5,
                (0, 255, 0),
                1,
                cv2.LINE_AA,
            )
            
            return frame
        except Exception as exc:
            self.logger.debug("Recording indicator failed: %s", exc)
            return frame

    def draw_screenshot_notification(self, frame: np.ndarray, show_notification: bool, notification_countdown: int = 15) -> np.ndarray:
        """Draw screenshot notification when screenshot is saved."""
        try:
            if not show_notification or notification_countdown <= 0:
                return frame
            
            h, w = frame.shape[:2]
            
            text = "📸 Screenshot saved!"
            alpha = notification_countdown / 15.0  # Fade out
            
            (text_w, text_h), _ = cv2.getTextSize(
                text,
                self.config.ui.text_font,
                0.7,
                1,
            )
            
            x_pos = (w - text_w) // 2
            y_pos = h // 2
            
            overlay = frame.copy()
            cv2.rectangle(
                overlay,
                (x_pos - 10, y_pos - text_h - 15),
                (x_pos + text_w + 10, y_pos + 10),
                (0, 200, 0),
                -1,
            )
            
            frame = cv2.addWeighted(overlay, alpha * 0.7, frame, 1 - (alpha * 0.7), 0)
            
            cv2.putText(
                frame,
                text,
                (x_pos, y_pos),
                self.config.ui.text_font,
                0.7,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )
            
            return frame
        except Exception as exc:
            self.logger.debug("Screenshot notification failed: %s", exc)
            return frame
