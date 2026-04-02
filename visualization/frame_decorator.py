"""Additional frame overlays for timestamp and compact info panels."""

from __future__ import annotations

from datetime import datetime
from typing import Any, Dict

import cv2
import numpy as np

from config import AppConfig


class FrameDecorator:
    """Adds non-detection overlays to improve UX readability."""

    def __init__(self, config: AppConfig, logger) -> None:
        self.config = config
        self.logger = logger

    def add_timestamp(self, frame: np.ndarray) -> np.ndarray:
        """Draw current timestamp at top-left."""
        try:
            ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            cv2.putText(
                frame,
                ts,
                (10, 48),
                self.config.ui.text_font,
                0.5,
                self.config.ui.text_color,
                1,
                cv2.LINE_AA,
            )
            return frame
        except Exception as exc:
            self.logger.debug("Timestamp overlay failed: %s", exc)
            return frame

    def add_info_panel(self, frame: np.ndarray, stats: Dict[str, Any]) -> np.ndarray:
        """Draw a compact translucent info panel with recent metrics."""
        try:
            h, w = frame.shape[:2]
            panel_x1, panel_y1 = w - 230, h - 120
            panel_x2, panel_y2 = w - 10, h - 10

            overlay = frame.copy()
            cv2.rectangle(overlay, (panel_x1, panel_y1), (panel_x2, panel_y2), (30, 30, 30), -1)
            frame = cv2.addWeighted(overlay, 0.45, frame, 0.55, 0)

            lines = [
                f"FPS: {float(stats.get('fps', 0.0)):.2f}",
                f"Obj: {int(stats.get('total_detections', 0))}",
                f"Inf: {float(stats.get('inference_ms', 0.0)):.1f} ms",
                f"Frame: {int(stats.get('frame_count', 0))}",
            ]

            y = panel_y1 + 24
            for line in lines:
                cv2.putText(
                    frame,
                    line,
                    (panel_x1 + 10, y),
                    self.config.ui.text_font,
                    0.55,
                    self.config.ui.text_color,
                    1,
                    cv2.LINE_AA,
                )
                y += 22
            return frame
        except Exception as exc:
            self.logger.debug("Info panel overlay failed: %s", exc)
            return frame

    def format_all_info(self, frame: np.ndarray, stats: Dict[str, Any]) -> np.ndarray:
        """Apply all decorative overlays."""
        frame = self.add_timestamp(frame)
        frame = self.add_info_panel(frame, stats)
        return frame

    def add_object_counts(self, frame: np.ndarray, class_counts: Dict[str, int]) -> np.ndarray:
        """Draw object count summary at bottom of frame."""
        try:
            if not class_counts:
                return frame
            
            h, w = frame.shape[:2]
            
            # Build count string: "Total: X | Person: Y | Car: Z"
            total = sum(class_counts.values())
            count_parts = [f"Total: {total}"]
            
            # Show top 5 most detected classes
            sorted_counts = sorted(class_counts.items(), key=lambda x: x[1], reverse=True)[:5]
            for cls_name, count in sorted_counts:
                count_parts.append(f"{cls_name}: {count}")
            
            count_text = " | ".join(count_parts)
            
            (text_w, text_h), baseline = cv2.getTextSize(
                count_text,
                self.config.ui.text_font,
                0.6,
                1,
            )
            
            # Background panel
            panel_y = h - 35
            overlay = frame.copy()
            cv2.rectangle(
                overlay,
                (5, panel_y - text_h - 8),
                (text_w + 15, panel_y + 5),
                (0, 0, 0),
                -1,
            )
            frame = cv2.addWeighted(overlay, 0.4, frame, 0.6, 0)
            
            # Text
            cv2.putText(
                frame,
                count_text,
                (10, panel_y),
                self.config.ui.text_font,
                0.6,
                (0, 255, 0),
                1,
                cv2.LINE_AA,
            )
            
            return frame
        except Exception as exc:
            self.logger.debug("Object counts overlay failed: %s", exc)
            return frame

    def add_active_filters_panel(self, frame: np.ndarray, enabled_classes: list, all_classes: list) -> np.ndarray:
        """Draw panel showing which object classes are currently enabled for filtering."""
        try:
            if not enabled_classes or not all_classes:
                return frame
            
            h, w = frame.shape[:2]
            
            # Determine filter status
            disabled_count = len(all_classes) - len(enabled_classes)
            if disabled_count == 0:
                filter_text = "✓ All classes enabled"
                color = (0, 255, 0)
            else:
                filter_text = f"⚠ {disabled_count}/{len(all_classes)} classes disabled"
                color = (0, 165, 255)
            
            # Draw filter status at bottom-right
            (text_w, text_h), baseline = cv2.getTextSize(
                filter_text,
                self.config.ui.text_font,
                0.55,
                1,
            )
            
            panel_x = w - text_w - 15
            panel_y = h - 55
            
            # Background
            overlay = frame.copy()
            cv2.rectangle(
                overlay,
                (panel_x - 5, panel_y - text_h - 8),
                (w - 10, panel_y + 5),
                (30, 30, 30),
                -1,
            )
            frame = cv2.addWeighted(overlay, 0.5, frame, 0.5, 0)
            
            # Text
            cv2.putText(
                frame,
                filter_text,
                (panel_x, panel_y),
                self.config.ui.text_font,
                0.55,
                color,
                1,
                cv2.LINE_AA,
            )
            
            return frame
        except Exception as exc:
            self.logger.debug("Active filters panel failed: %s", exc)
            return frame
