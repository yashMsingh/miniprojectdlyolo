"""Object counter for tracking detected objects by class."""

from __future__ import annotations

from collections import Counter, defaultdict
from typing import Any, Dict, List


class ObjectCounter:
    """Tracks object detections by class across frames."""

    def __init__(self) -> None:
        """Initialize counter with empty state."""
        self.class_counts: Counter = Counter()
        self.frame_counts: List[int] = []
        self.cumulative_count: int = 0
        self.max_count_frame: int = 0
        self.frame_number: int = 0

    def update_counts(self, detections: List[Dict[str, Any]]) -> None:
        """Update counts based on current frame detections.

        Args:
            detections: List of detection dictionaries.
        """
        self.frame_number += 1
        frame_count = len(detections)
        self.frame_counts.append(frame_count)
        self.cumulative_count += frame_count

        if frame_count > self.max_count_frame:
            self.max_count_frame = frame_count

        for det in detections:
            class_name = str(det.get("class_name", "unknown"))
            self.class_counts[class_name] += 1

    def get_class_count(self, class_name: str) -> int:
        """Get count for specific class."""
        return int(self.class_counts.get(class_name, 0))

    def get_total_count(self) -> int:
        """Get cumulative total detections."""
        return self.cumulative_count

    def get_all_counts(self) -> Dict[str, int]:
        """Return all class counts as dictionary."""
        return dict(self.class_counts)

    def get_current_frame_count(self) -> int:
        """Get count from latest frame."""
        return self.frame_counts[-1] if self.frame_counts else 0

    def get_count_summary(self) -> str:
        """Return formatted summary string."""
        if not self.class_counts:
            return "Total: 0"

        counts_str = " | ".join(
            f"{cls}: {count}" for cls, count in self.class_counts.most_common()
        )
        return f"Total: {self.cumulative_count} | {counts_str}"

    def get_most_detected(self) -> tuple[str, int]:
        """Return class with highest count and its count."""
        if not self.class_counts:
            return "none", 0
        cls, count = self.class_counts.most_common(1)[0]
        return cls, count

    def get_least_detected(self) -> tuple[str, int]:
        """Return class with lowest count and its count."""
        if not self.class_counts:
            return "none", 0
        cls, count = self.class_counts.most_common()[-1]
        return cls, count

    def get_statistics(self) -> Dict[str, Any]:
        """Return comprehensive statistics."""
        if not self.frame_counts:
            return {
                "total_detections": 0,
                "total_frames": self.frame_number,
                "avg_objects_per_frame": 0.0,
                "max_objects_in_frame": 0,
                "class_counts": {},
                "most_detected": "none",
                "least_detected": "none",
            }

        avg_per_frame = self.cumulative_count / len(self.frame_counts)
        most_cls, most_count = self.get_most_detected()
        least_cls, least_count = self.get_least_detected()

        return {
            "total_detections": self.cumulative_count,
            "total_frames": self.frame_number,
            "avg_objects_per_frame": round(avg_per_frame, 2),
            "max_objects_in_frame": self.max_count_frame,
            "class_counts": dict(self.class_counts),
            "most_detected": f"{most_cls} ({most_count})",
            "least_detected": f"{least_cls} ({least_count})",
        }

    def reset_counts(self) -> None:
        """Clear all counters."""
        self.class_counts.clear()
        self.frame_counts.clear()
        self.cumulative_count = 0
        self.max_count_frame = 0
        self.frame_number = 0

    def export_counts(self, filepath: str) -> bool:
        """Export counts to JSON file."""
        import json
        from pathlib import Path

        try:
            path = Path(filepath)
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("w", encoding="utf-8") as f:
                json.dump(self.get_statistics(), f, indent=2)
            return True
        except Exception:
            return False
