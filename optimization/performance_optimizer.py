"""Performance analysis and optimization recommendations."""

from __future__ import annotations

from typing import Callable, Dict, List

import numpy as np

from config import AppConfig


class PerformanceOptimizer:
    """Analyze bottlenecks and suggest optimization actions."""

    def __init__(self, config: AppConfig) -> None:
        self.config = config
        self.frame_times: List[float] = []
        self.inference_times: List[float] = []

    def analyze_bottlenecks(self) -> Dict[str, float | str]:
        """Identify bottleneck category and timings.

        Returns:
            Dictionary containing frame, inference, processing metrics.
        """
        if not self.frame_times or not self.inference_times:
            return {"error": "Not enough data"}

        avg_frame_time = float(np.mean(self.frame_times))
        avg_inference_time = float(np.mean(self.inference_times))
        processing_time = max(0.0, avg_frame_time - avg_inference_time)

        bottleneck = "inference" if avg_inference_time > processing_time else "processing"
        bottleneck_pct = (
            max(avg_inference_time, processing_time) / avg_frame_time * 100.0
            if avg_frame_time > 0
            else 0.0
        )

        return {
            "avg_frame_time_ms": avg_frame_time * 1000.0,
            "avg_inference_time_ms": avg_inference_time * 1000.0,
            "avg_processing_time_ms": processing_time * 1000.0,
            "bottleneck": bottleneck,
            "bottleneck_percentage": bottleneck_pct,
        }

    def get_optimization_recommendations(self) -> List[str]:
        """Return recommendations based on bottleneck analysis."""
        analysis = self.analyze_bottlenecks()
        if "error" in analysis:
            return ["Collect more runtime data before optimization."]

        recommendations: List[str] = []
        if analysis.get("bottleneck") == "inference":
            recommendations.extend(
                [
                    "Use yolov8n.pt for real-time throughput.",
                    "Enable GPU acceleration if CUDA is available.",
                    "Reduce input frame resolution.",
                    "Raise confidence threshold to reduce post-processing load.",
                ]
            )
        else:
            recommendations.extend(
                [
                    "Disable non-essential overlays.",
                    "Reduce frame resolution.",
                    "Disable recording during benchmark runs.",
                    "Reduce logging verbosity for runtime sessions.",
                ]
            )
        return recommendations

    def test_optimization_impact(self, optimization_func: Callable[[], None]) -> Dict[str, float]:
        """Compare performance before and after optimization callback."""
        before = self.analyze_bottlenecks()
        if "error" in before:
            return {"error": "Not enough baseline data"}  # type: ignore[return-value]

        optimization_func()
        after = self.analyze_bottlenecks()
        if "error" in after:
            return {"error": "Not enough post-optimization data"}  # type: ignore[return-value]

        before_ms = float(before.get("avg_frame_time_ms", 0.0))
        after_ms = float(after.get("avg_frame_time_ms", 0.0))
        before_fps = 1000.0 / before_ms if before_ms > 0 else 0.0
        after_fps = 1000.0 / after_ms if after_ms > 0 else 0.0
        improvement = ((before_ms - after_ms) / before_ms * 100.0) if before_ms > 0 else 0.0

        return {
            "before_fps": before_fps,
            "after_fps": after_fps,
            "improvement_percentage": improvement,
        }

    def record_frame_time(self, elapsed_time: float) -> None:
        """Store frame processing time in seconds."""
        self.frame_times.append(float(elapsed_time))

    def record_inference_time(self, elapsed_time: float) -> None:
        """Store inference processing time in seconds."""
        self.inference_times.append(float(elapsed_time))
