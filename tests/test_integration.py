"""Integration tests for full pipeline."""

from __future__ import annotations

import unittest
from unittest.mock import patch

import numpy as np

from config import AppConfig
from detector.main_detector import MainDetector


class _FakeCap:
    def __init__(self) -> None:
        self._opened = True

    def isOpened(self) -> bool:
        return self._opened

    def read(self):
        frame = np.zeros((240, 320, 3), dtype=np.uint8)
        return True, frame

    def set(self, *_args, **_kwargs) -> bool:
        return True

    def get(self, *_args, **_kwargs) -> float:
        return 30.0

    def release(self) -> None:
        self._opened = False


def _fake_load_model(self) -> None:
    self.model = type(
        "FakeModel",
        (),
        {
            "names": {0: "person", 1: "car"},
            "model": type(
                "FakeNet",
                (),
                {
                    "parameters": lambda _s: [],
                    "modules": lambda _s: [],
                },
            )(),
        },
    )()


def _fake_process_frame_pipeline(_self, _frame):
    return [
        {
            "box": (10, 10, 50, 50),
            "confidence": 0.9,
            "class_id": 0,
            "class_name": "person",
        }
    ], {"inference_ms": 8.0}


class TestIntegration(unittest.TestCase):
    """Integration tests for full pipeline."""

    def setUp(self) -> None:
        self.config = AppConfig.default()
        self.config.webcam.frame_width = 320
        self.config.webcam.frame_height = 240
        self.config.output.save_video = False

        self.p1 = patch("detector.main_detector.BaseDetector._load_model", _fake_load_model)
        self.p2 = patch("detector.main_detector.MainDetector.initialize_camera", lambda _s: _FakeCap())
        self.p3 = patch("detector.main_detector.InferencePipeline.process_frame_pipeline", _fake_process_frame_pipeline)
        self.p1.start()
        self.p2.start()
        self.p3.start()

    def tearDown(self) -> None:
        self.p1.stop()
        self.p2.stop()
        self.p3.stop()

    def test_detector_initialization(self) -> None:
        detector = MainDetector(config=self.config)
        detector.shutdown()
        self.assertTrue(True)

    def test_single_frame_inference(self) -> None:
        detector = MainDetector(config=self.config)
        frame = np.zeros((240, 320, 3), dtype=np.uint8)
        detections, inference_ms = detector.run_inference(frame)
        detector.shutdown()
        self.assertIsInstance(detections, list)
        self.assertGreaterEqual(inference_ms, 0.0)

    def test_frame_visualization(self) -> None:
        detector = MainDetector(config=self.config)
        frame = np.zeros((240, 320, 3), dtype=np.uint8)
        detections, _ = detector.run_inference(frame)
        annotated = detector.visualizer.annotate_frame(
            frame,
            detections,
            {"fps": 30.0, "inference_ms": 8.0, "total_detections": len(detections), "frame_count": 1},
        )
        detector.shutdown()
        self.assertIsNotNone(annotated)

    def test_performance_metrics(self) -> None:
        detector = MainDetector(config=self.config)
        for _ in range(5):
            frame = np.random.randint(0, 255, (240, 320, 3), dtype=np.uint8)
            detector.process_and_visualize(frame)
        metrics = detector.stats_tracker.get_current_stats()
        detector.shutdown()
        self.assertIn("total_detections", metrics)


class TestPipelineRobustness(unittest.TestCase):
    """Test pipeline robustness and error handling."""

    def setUp(self) -> None:
        self.config = AppConfig.default()

    def test_extreme_threshold_values(self) -> None:
        from enhancement.threshold_controller import ThresholdController

        controller = ThresholdController()

        controller.set_threshold(0.0)
        self.assertEqual(controller.get_threshold(), 0.0)

        controller.set_threshold(1.0)
        self.assertEqual(controller.get_threshold(), 1.0)

        ok = controller.set_threshold(1.5)
        self.assertFalse(ok)
        self.assertLessEqual(controller.get_threshold(), 1.0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
