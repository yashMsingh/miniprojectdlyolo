"""Detection post-processing utilities for YOLO output."""

from __future__ import annotations

from collections import Counter
from typing import Any, Dict, List, Optional


Detection = Dict[str, Any]


class DetectionProcessor:
    """Parses, filters, and summarizes YOLO detections."""

    def __init__(self) -> None:
        self._last_detections: List[Detection] = []

    def parse_results(self, yolo_results: Any) -> List[Detection]:
        """Extract standardized detection dictionaries from YOLO results."""
        detections: List[Detection] = []
        if not yolo_results:
            self._last_detections = detections
            return detections

        try:
            result = yolo_results[0]
            names = getattr(result, "names", {})
            boxes = getattr(result, "boxes", None)
            if boxes is None:
                self._last_detections = detections
                return detections

            xyxy = boxes.xyxy.cpu().numpy()
            conf = boxes.conf.cpu().numpy()
            cls_ids = boxes.cls.cpu().numpy().astype(int)

            for i in range(len(xyxy)):
                x1, y1, x2, y2 = [int(v) for v in xyxy[i].tolist()]
                class_id = int(cls_ids[i])
                class_name = names.get(class_id, str(class_id)) if isinstance(names, dict) else str(class_id)
                detections.append(
                    {
                        "box": (x1, y1, x2, y2),
                        "confidence": float(conf[i]),
                        "class_id": class_id,
                        "class_name": class_name,
                    }
                )

            self._last_detections = detections
            return detections
        except Exception:
            self._last_detections = []
            return []

    def apply_threshold(self, detections: List[Detection], threshold: float) -> List[Detection]:
        """Filter detections by confidence and sort descending."""
        filtered = [d for d in detections if d.get("confidence", 0.0) >= threshold]
        filtered.sort(key=lambda d: d.get("confidence", 0.0), reverse=True)
        self._last_detections = filtered
        return filtered

    def get_detection_data(self) -> List[Detection]:
        """Return latest processed detections."""
        return list(self._last_detections)

    def count_objects(self, detections: Optional[List[Detection]] = None) -> Dict[str, int]:
        """Count objects grouped by class name."""
        data = detections if detections is not None else self._last_detections
        return dict(Counter(d.get("class_name", "unknown") for d in data))

    def get_statistics(self, detections: Optional[List[Detection]] = None) -> Dict[str, Any]:
        """Return summary statistics for current detections."""
        data = detections if detections is not None else self._last_detections
        if not data:
            return {
                "total_detections": 0,
                "avg_confidence": 0.0,
                "max_confidence": 0.0,
                "object_counts": {},
            }

        confidences = [float(d.get("confidence", 0.0)) for d in data]
        return {
            "total_detections": len(data),
            "avg_confidence": sum(confidences) / len(confidences),
            "max_confidence": max(confidences),
            "object_counts": self.count_objects(data),
        }
