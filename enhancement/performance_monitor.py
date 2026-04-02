"""Performance monitoring and optimization suggestions."""

from __future__ import annotations

import time
from collections import deque
from typing import Any, Deque, Dict


class PerformanceMonitor:
    """Track and analyze system performance metrics."""

    def __init__(self, history_size: int = 300, logger: Any = None) -> None:
        """Initialize performance monitor.

        Args:
            history_size: Rolling window size for averaging.
            logger: Logger instance.
        """
        self.logger = logger
        self.history_size = history_size

        self.frame_times: Deque[float] = deque(maxlen=history_size)
        self.inference_times: Deque[float] = deque(maxlen=history_size)
        self.processing_times: Deque[float] = deque(maxlen=history_size)

        self.frame_start_time: float = 0
        self.dropped_frames: int = 0

    def start_frame_timer(self) -> None:
        """Mark start of frame processing."""
        self.frame_start_time = time.perf_counter()

    def end_frame_timer(self) -> float:
        """Mark end of frame processing, return elapsed time."""
        elapsed = time.perf_counter() - self.frame_start_time
        self.frame_times.append(elapsed)
        return elapsed

    def record_inference_time(self, inference_ms: float) -> None:
        """Record inference time for one frame.

        Args:
            inference_ms: Inference time in milliseconds.
        """
        self.inference_times.append(inference_ms)

    def record_processing_time(self, processing_ms: float) -> None:
        """Record total processing time for one frame.

        Args:
            processing_ms: Processing time in milliseconds.
        """
        self.processing_times.append(processing_ms)

    def get_current_fps(self) -> float:
        """Return FPS based on latest frame time."""
        if not self.frame_times:
            return 0.0
        latest = self.frame_times[-1]
        return 1.0 / latest if latest > 0 else 0.0

    def get_avg_fps(self) -> float:
        """Return average FPS over history window."""
        if not self.frame_times or not self.frame_times:
            return 0.0
        avg_time = sum(self.frame_times) / len(self.frame_times)
        return 1.0 / avg_time if avg_time > 0 else 0.0

    def get_avg_inference_ms(self) -> float:
        """Return average inference time."""
        if not self.inference_times:
            return 0.0
        return sum(self.inference_times) / len(self.inference_times)

    def get_avg_processing_ms(self) -> float:
        """Return average total processing time."""
        if not self.processing_times:
            return 0.0
        return sum(self.processing_times) / len(self.processing_times)

    def get_performance_summary(self) -> Dict[str, float]:
        """Return comprehensive performance metrics."""
        return {
            "current_fps": round(self.get_current_fps(), 2),
            "avg_fps": round(self.get_avg_fps(), 2),
            "avg_inference_ms": round(self.get_avg_inference_ms(), 2),
            "avg_processing_ms": round(self.get_avg_processing_ms(), 2),
            "min_frame_time_ms": round(min(self.frame_times) * 1000, 2) if self.frame_times else 0.0,
            "max_frame_time_ms": round(max(self.frame_times) * 1000, 2) if self.frame_times else 0.0,
            "dropped_frames": self.dropped_frames,
        }

    def get_optimization_suggestions(self) -> list[str]:
        """Analyze performance and suggest optimizations."""
        suggestions = []
        avg_fps = self.get_avg_fps()

        if avg_fps < 15:
            suggestions.append("FPS is very low. Consider lowering resolution or model size.")
        elif avg_fps < 24:
            suggestions.append("FPS below 24. Try reducing frame resolution or disabling features.")

        if self.get_avg_inference_ms() > 40:
            suggestions.append("Inference time is high. Try using smaller model (yolov8n) or GPU.")

        if self.dropped_frames > (len(self.frame_times) * 0.05):
            suggestions.append("High frame drop rate. Close background applications.")

        if not suggestions:
            suggestions.append("Performance is good!")

        return suggestions

    def record_dropped_frame(self) -> None:
        """Increment dropped frame counter."""
        self.dropped_frames += 1

    def reset_stats(self) -> None:
        """Clear all performance statistics."""
        self.frame_times.clear()
        self.inference_times.clear()
        self.processing_times.clear()
        self.dropped_frames = 0

    def performance_report(self) -> str:
        """Return formatted performance report."""
        summary = self.get_performance_summary()
        lines = [
            "═" * 40,
            "PERFORMANCE REPORT",
            "═" * 40,
            f"FPS (current): {summary['current_fps']:.2f}",
            f"FPS (average): {summary['avg_fps']:.2f}",
            f"Inference Time (avg): {summary['avg_inference_ms']:.2f} ms",
            f"Processing Time (avg): {summary['avg_processing_ms']:.2f} ms",
            f"Frame Time (min): {summary['min_frame_time_ms']:.2f} ms",
            f"Frame Time (max): {summary['max_frame_time_ms']:.2f} ms",
            f"Dropped Frames: {summary['dropped_frames']}",
        ]

        suggestions = self.get_optimization_suggestions()
        lines.append("─" * 40)
        lines.append("SUGGESTIONS:")
        for suggestion in suggestions:
            lines.append(f"• {suggestion}")

        lines.append("═" * 40)
        return "\n".join(lines)
