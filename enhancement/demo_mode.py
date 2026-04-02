"""Demo mode with enhanced visual effects."""

from __future__ import annotations

from typing import Any, Dict, List

import cv2
import numpy as np


class DemoMode:
    """Provides enhanced visualization effects for demo mode."""

    def __init__(self, config: Any, logger: Any = None) -> None:
        """Initialize demo mode.

        Args:
            config: Application configuration.
            logger: Logger instance.
        """
        self.config = config
        self.logger = logger
        self.enabled = False

    def enable(self) -> None:
        """Enable demo mode."""
        self.enabled = True
        if self.logger:
            self.logger.info("Demo mode enabled")

    def disable(self) -> None:
        """Disable demo mode."""
        self.enabled = False
        if self.logger:
            self.logger.info("Demo mode disabled")

    def toggle(self) -> bool:
        """Toggle demo mode on/off.

        Returns:
            New enabled state.
        """
        self.enabled = not self.enabled
        return self.enabled

    def apply_demo_enhancements(
        self,
        frame: np.ndarray,
        detections: List[Dict[str, Any]],
    ) -> np.ndarray:
        """Apply all demo mode enhancements to frame.

        Args:
            frame: Input frame.
            detections: Detection list.

        Returns:
            Enhanced frame.
        """
        if not self.enabled:
            return frame

        output = frame.copy()
        output = self.draw_grid_overlay(output)
        output = self.draw_confidence_gradient_boxes(output, detections)
        output = self.draw_class_legend(output, detections)
        return output

    def draw_grid_overlay(self, frame: np.ndarray) -> np.ndarray:
        """Draw reference grid on frame."""
        h, w = frame.shape[:2]
        grid_spacing = 80

        overlay = frame.copy()
        for x in range(0, w, grid_spacing):
            cv2.line(overlay, (x, 0), (x, h), (100, 100, 100), 1)
        for y in range(0, h, grid_spacing):
            cv2.line(overlay, (0, y), (w, y), (100, 100, 100), 1)

        return cv2.addWeighted(overlay, 0.1, frame, 0.9, 0)

    def draw_confidence_gradient_boxes(
        self,
        frame: np.ndarray,
        detections: List[Dict[str, Any]],
    ) -> np.ndarray:
        """Draw boxes with color gradient based on confidence."""
        for det in detections:
            conf = float(det.get("confidence", 0.5))
            box = det.get("box", (0, 0, 0, 0))
            x1, y1, x2, y2 = [int(v) for v in box]

            if conf < 0.6:
                color = (0, 0, 255)
            elif conf < 0.8:
                color = (0, 165, 255)
            else:
                color = (0, 255, 0)

            thickness = int(2 + (conf * 2))
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)

        return frame

    def draw_class_legend(
        self,
        frame: np.ndarray,
        detections: List[Dict[str, Any]],
    ) -> np.ndarray:
        """Draw legend of detected classes on frame."""
        h, w = frame.shape[:2]

        classes_present = set(d.get("class_name", "unknown") for d in detections)
        if not classes_present:
            return frame

        legend_x = w - 220
        legend_y = h - 80

        cv2.rectangle(frame, (legend_x - 5, legend_y - 5), (w - 5, h - 5), (0, 0, 0), -1)
        cv2.rectangle(frame, (legend_x - 5, legend_y - 5), (w - 5, h - 5), (200, 200, 200), 1)

        y = legend_y + 15
        cv2.putText(
            frame,
            "Detected Classes:",
            (legend_x, y),
            self.config.ui.text_font,
            0.5,
            (200, 200, 200),
            1,
        )
        y += 20

        for cls in sorted(classes_present):
            cv2.putText(
                frame,
                f"• {cls}",
                (legend_x + 5, y),
                self.config.ui.text_font,
                0.4,
                (0, 255, 0),
                1,
            )
            y += 18

        return frame
