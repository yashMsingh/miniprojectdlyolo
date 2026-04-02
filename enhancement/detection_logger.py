"""Detection logging to CSV and JSON formats."""

from __future__ import annotations

import csv
import json
import time
from pathlib import Path
from typing import Any, Dict, List, Optional


class DetectionLogger:
    """Logs and exports detections in multiple formats."""

    def __init__(self, logger: Any) -> None:
        """Initialize detection logger.

        Args:
            logger: Logger instance.
        """
        self.logger = logger
        self.detections_log: List[Dict[str, Any]] = []
        self.frame_timestamps: List[float] = []
        self.start_time = time.perf_counter()

    def log_detections(
        self,
        frame_number: int,
        detections: List[Dict[str, Any]],
        frame_shape: Tuple[int, int, int],
    ) -> None:
        """Log detections from a single frame.

        Args:
            frame_number: Frame index.
            detections: List of detection dictionaries.
            frame_shape: (height, width, channels) of frame.
        """
        timestamp = time.perf_counter() - self.start_time

        for det in detections:
            box = det.get("box", (0, 0, 0, 0))
            log_entry = {
                "frame_number": frame_number,
                "timestamp": round(timestamp, 3),
                "class_name": str(det.get("class_name", "unknown")),
                "class_id": int(det.get("class_id", -1)),
                "confidence": round(float(det.get("confidence", 0.0)), 4),
                "x1": int(box[0]),
                "y1": int(box[1]),
                "x2": int(box[2]),
                "y2": int(box[3]),
                "box_width": int(box[2] - box[0]),
                "box_height": int(box[3] - box[1]),
                "frame_width": int(frame_shape[1]),
                "frame_height": int(frame_shape[0]),
            }
            self.detections_log.append(log_entry)

        self.frame_timestamps.append(timestamp)

    def export_csv(self, filepath: str) -> bool:
        """Export detections to CSV file.

        Args:
            filepath: Output CSV path.

        Returns:
            True if successful.
        """
        try:
            Path(filepath).parent.mkdir(parents=True, exist_ok=True)

            if not self.detections_log:
                self.logger.warning("No detections to export to CSV")
                return False

            keys = self.detections_log[0].keys()
            with open(filepath, "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=keys)
                writer.writeheader()
                writer.writerows(self.detections_log)

            self.logger.info(f"Detections exported to CSV: {filepath}")
            return True
        except Exception as exc:
            self.logger.exception(f"CSV export failed: {exc}")
            return False

    def export_json(self, filepath: str) -> bool:
        """Export detections to JSON file.

        Args:
            filepath: Output JSON path.

        Returns:
            True if successful.
        """
        try:
            Path(filepath).parent.mkdir(parents=True, exist_ok=True)

            data = {
                "metadata": self.get_log_summary(),
                "detections": self.detections_log,
            }

            with open(filepath, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2)

            self.logger.info(f"Detections exported to JSON: {filepath}")
            return True
        except Exception as exc:
            self.logger.exception(f"JSON export failed: {exc}")
            return False

    def get_log_summary(self) -> Dict[str, Any]:
        """Return summary statistics of logged detections."""
        if not self.detections_log:
            return {
                "total_frames": 0,
                "total_detections": 0,
                "duration": 0.0,
            }

        class_counts = {}
        confidence_sum = {}
        for log_entry in self.detections_log:
            cls = log_entry["class_name"]
            conf = log_entry["confidence"]
            class_counts[cls] = class_counts.get(cls, 0) + 1
            confidence_sum[cls] = confidence_sum.get(cls, 0) + conf

        avg_confidence = {
            cls: round(confidence_sum[cls] / class_counts[cls], 4)
            for cls in class_counts
        }

        return {
            "total_frames": len(self.frame_timestamps),
            "total_detections": len(self.detections_log),
            "duration_seconds": round(self.frame_timestamps[-1] if self.frame_timestamps else 0, 2),
            "class_counts": class_counts,
            "avg_confidence_by_class": avg_confidence,
        }

    def reset_log(self) -> None:
        """Clear all logged detections."""
        self.detections_log.clear()
        self.frame_timestamps.clear()
        self.start_time = time.perf_counter()

    def get_total_detections(self) -> int:
        """Return total detections logged."""
        return len(self.detections_log)
