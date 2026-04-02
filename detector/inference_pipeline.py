"""Inference pipeline that performs YOLO prediction and result processing."""

from __future__ import annotations

import time
from collections import deque
from typing import Any, Deque, Dict, List, Tuple

import numpy as np

from config import AppConfig
from .detection_processor import DetectionProcessor, Detection


class InferencePipeline:
    """Runs end-to-end per-frame inference and post-processing."""

    def __init__(self, model: Any, config: AppConfig, logger) -> None:
        self.model = model
        self.config = config
        self.logger = logger
        self.processor = DetectionProcessor()
        self._inference_history_ms: Deque[float] = deque(maxlen=180)

    def process_frame_pipeline(self, frame: np.ndarray) -> Tuple[List[Detection], Dict[str, float]]:
        """Run full inference + parsing pipeline for a frame.

        Args:
            frame: Input frame from webcam.

        Returns:
            Tuple of (processed_detections, metrics).
        """
        start = time.perf_counter()
        try:
            results = self.model.predict(
                source=frame,
                conf=self.config.model.confidence_threshold,
                iou=self.config.model.iou_threshold,
                device=self.config.model.device,
                verbose=False,
            )
            inference_ms = (time.perf_counter() - start) * 1000.0
            self._inference_history_ms.append(inference_ms)

            detections = self.processor.parse_results(results)
            detections = self.processor.apply_threshold(
                detections,
                self.config.detection.min_confidence,
            )

            metrics = {
                "inference_ms": round(inference_ms, 3),
                "inference_fps": round(1000.0 / inference_ms, 2) if inference_ms > 0 else 0.0,
                "avg_inference_ms": round(
                    sum(self._inference_history_ms) / len(self._inference_history_ms), 3
                )
                if self._inference_history_ms
                else 0.0,
            }
            return detections, metrics
        except Exception as exc:
            self.logger.exception("Inference pipeline failed: %s", exc)
            return [], {"inference_ms": 0.0, "inference_fps": 0.0, "avg_inference_ms": 0.0}

    def get_inference_metrics(self) -> Dict[str, float]:
        """Return rolling inference performance metrics."""
        if not self._inference_history_ms:
            return {"avg_inference_ms": 0.0, "avg_inference_fps": 0.0}
        avg_ms = sum(self._inference_history_ms) / len(self._inference_history_ms)
        return {
            "avg_inference_ms": round(avg_ms, 3),
            "avg_inference_fps": round(1000.0 / avg_ms, 2) if avg_ms > 0 else 0.0,
        }

    def log_detections(self, detections: List[Detection]) -> None:
        """Log concise detection summary for debugging."""
        if not detections:
            self.logger.debug("No detections in current frame.")
            return
        summary = ", ".join(
            f"{d['class_name']}:{d['confidence']:.2f}" for d in detections[:5]
        )
        self.logger.debug("Detections (%d): %s", len(detections), summary)
