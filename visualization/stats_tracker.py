"""Runtime statistics tracker for detections and performance metrics."""

from __future__ import annotations

from collections import Counter, deque
from typing import Any, Deque, Dict, List


class StatsTracker:
    """Tracks rolling and cumulative stats for the detector runtime."""

    def __init__(self, history_size: int = 180) -> None:
        self.history_size = history_size
        self.total_frames: int = 0
        self.total_detections: int = 0
        self.class_counter: Counter = Counter()
        self.confidence_sum_by_class: Dict[str, float] = {}

        self.fps_history: Deque[float] = deque(maxlen=history_size)
        self.inference_ms_history: Deque[float] = deque(maxlen=history_size)
        self.processing_ms_history: Deque[float] = deque(maxlen=history_size)
        self.objects_per_frame: Deque[int] = deque(maxlen=history_size)

    def update_stats(
        self,
        detections: List[Dict[str, Any]],
        fps: float,
        inference_ms: float,
        processing_ms: float,
    ) -> None:
        """Update tracker with data from one processed frame."""
        self.total_frames += 1
        det_count = len(detections)
        self.total_detections += det_count

        self.fps_history.append(float(fps))
        self.inference_ms_history.append(float(inference_ms))
        self.processing_ms_history.append(float(processing_ms))
        self.objects_per_frame.append(det_count)

        for det in detections:
            cls_name = str(det.get("class_name", "unknown"))
            conf = float(det.get("confidence", 0.0))
            self.class_counter[cls_name] += 1
            self.confidence_sum_by_class[cls_name] = (
                self.confidence_sum_by_class.get(cls_name, 0.0) + conf
            )

    def get_current_stats(self) -> Dict[str, Any]:
        """Return recent snapshot stats for display."""
        return {
            "total_frames": self.total_frames,
            "total_detections": self.total_detections,
            "fps": self.fps_history[-1] if self.fps_history else 0.0,
            "inference_ms": self.inference_ms_history[-1] if self.inference_ms_history else 0.0,
            "processing_ms": self.processing_ms_history[-1] if self.processing_ms_history else 0.0,
            "objects_last_frame": self.objects_per_frame[-1] if self.objects_per_frame else 0,
            "class_counter": dict(self.class_counter),
        }

    def get_average_stats(self) -> Dict[str, Any]:
        """Return average metrics over rolling window and runtime totals."""

        def _avg(values: Deque[float]) -> float:
            return float(sum(values) / len(values)) if values else 0.0

        avg_conf: Dict[str, float] = {}
        for cls_name, count in self.class_counter.items():
            if count > 0:
                avg_conf[cls_name] = self.confidence_sum_by_class.get(cls_name, 0.0) / count

        return {
            "avg_fps": _avg(self.fps_history),
            "avg_inference_ms": _avg(self.inference_ms_history),
            "avg_processing_ms": _avg(self.processing_ms_history),
            "avg_objects_per_frame": _avg(self.objects_per_frame),
            "total_frames": self.total_frames,
            "total_detections": self.total_detections,
            "objects_by_class": dict(self.class_counter),
            "avg_confidence_by_class": avg_conf,
        }

    def reset_stats(self) -> None:
        """Reset all tracked statistics."""
        self.total_frames = 0
        self.total_detections = 0
        self.class_counter.clear()
        self.confidence_sum_by_class.clear()
        self.fps_history.clear()
        self.inference_ms_history.clear()
        self.processing_ms_history.clear()
        self.objects_per_frame.clear()

    def summary_string(self) -> str:
        """Return compact summary string for logs."""
        avg = self.get_average_stats()
        return (
            f"frames={avg['total_frames']} | avg_fps={avg['avg_fps']:.2f} | "
            f"avg_inf_ms={avg['avg_inference_ms']:.2f} | "
            f"avg_obj_frame={avg['avg_objects_per_frame']:.2f}"
        )

    def get_performance_metrics(self) -> Dict[str, Any]:
        """Return detailed performance metrics for monitoring."""
        avg = self.get_average_stats()
        current = self.get_current_stats()
        
        return {
            "current_fps": float(current.get("fps", 0.0)),
            "avg_fps": float(avg.get("avg_fps", 0.0)),
            "current_inference_ms": float(current.get("inference_ms", 0.0)),
            "avg_inference_ms": float(avg.get("avg_inference_ms", 0.0)),
            "current_processing_ms": float(current.get("processing_ms", 0.0)),
            "avg_processing_ms": float(avg.get("avg_processing_ms", 0.0)),
            "total_frames": int(avg.get("total_frames", 0)),
            "total_detections": int(avg.get("total_detections", 0)),
            "avg_objects_per_frame": float(avg.get("avg_objects_per_frame", 0.0)),
        }

    def get_optimization_suggestions(self) -> List[str]:
        """Return list of optimization suggestions based on performance metrics."""
        suggestions = []
        avg = self.get_average_stats()
        
        avg_fps = float(avg.get("avg_fps", 0.0))
        avg_inf_ms = float(avg.get("avg_inference_ms", 0.0))
        avg_proc_ms = float(avg.get("avg_processing_ms", 0.0))
        
        # FPS-based suggestions
        if avg_fps < 15:
            suggestions.append("⚠ Low FPS (<15). Consider reducing resolution or enabling GPU.")
        elif avg_fps < 25:
            suggestions.append("⚠ Moderate FPS (<25). Consider disabling demo mode or logging.")
        
        # Inference-based suggestions
        if avg_inf_ms > 40:
            suggestions.append("⚠ High inference time (>40ms). Use CPU-optimized model or GPU.")
        
        # Processing-based suggestions
        if avg_proc_ms > 20:
            suggestions.append("⚠ High processing time (>20ms). Optimize visualization or reduce features.")
        
        # Positive feedback
        if avg_fps >= 25 and avg_inf_ms < 40:
            suggestions.append("✓ Performance is good!")
        
        return suggestions

    def get_bottleneck(self) -> str:
        """Identify the primary performance bottleneck."""
        current = self.get_current_stats()
        
        inf_ms = float(current.get("inference_ms", 0.0))
        proc_ms = float(current.get("processing_ms", 0.0))
        fps = float(current.get("fps", 0.0))
        frame_time_ms = 1000.0 / fps if fps > 0 else 0.0
        
        if frame_time_ms == 0:
            return "Unknown (data insufficient)"
        
        camera_time = frame_time_ms - inf_ms - proc_ms
        
        if inf_ms >= proc_ms and inf_ms >= camera_time * 0.1:
            return f"Inference ({inf_ms:.1f}ms)"
        elif proc_ms >= inf_ms and proc_ms >= camera_time * 0.1:
            return f"Processing ({proc_ms:.1f}ms)"
        else:
            return f"Balanced ({frame_time_ms:.1f}ms/frame)"
