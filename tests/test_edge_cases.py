"""Edge cases and error handling tests."""

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
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
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
            "names": {0: "person"},
            "model": type("FakeNet", (), {"parameters": lambda _s: [], "modules": lambda _s: []})(),
        },
    )()


def _fake_process_frame_pipeline(_self, _frame):
    return [], {"inference_ms": 5.0}


class TestEdgeCases(unittest.TestCase):
    """Test edge cases and error handling."""

    def setUp(self) -> None:
        self.config = AppConfig.default()
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

    def _assert_frame_processed(self, frame: np.ndarray) -> None:
        detector = MainDetector(config=self.config)
        detections, metrics = detector.process_frame(frame)
        detector.shutdown()
        self.assertIsInstance(detections, list)
        self.assertIn("inference_ms", metrics)

    def test_empty_frame(self) -> None:
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        self._assert_frame_processed(frame)

    def test_bright_frame(self) -> None:
        frame = np.ones((480, 640, 3), dtype=np.uint8) * 255
        self._assert_frame_processed(frame)

    def test_small_frame(self) -> None:
        frame = np.zeros((32, 32, 3), dtype=np.uint8)
        self._assert_frame_processed(frame)

    def test_large_frame(self) -> None:
        frame = np.zeros((1080, 1920, 3), dtype=np.uint8)
        self._assert_frame_processed(frame)

    def test_invalid_frame_none(self) -> None:
        detector = MainDetector(config=self.config)
        detections, metrics = detector.process_frame(None)
        detector.shutdown()
        self.assertEqual(detections, [])
        self.assertEqual(metrics["inference_ms"], 0.0)

    def test_rapid_threshold_changes(self) -> None:
        from enhancement.threshold_controller import ThresholdController

        controller = ThresholdController()
        for _ in range(100):
            controller.increase_threshold(0.01)
            controller.decrease_threshold(0.01)
        self.assertTrue(True)

    def test_zero_confidence_threshold(self) -> None:
        from enhancement.threshold_controller import ThresholdController

        controller = ThresholdController()
        controller.set_threshold(0.0)
        self.assertEqual(controller.get_threshold(), 0.0)

    def test_maximum_confidence_threshold(self) -> None:
        from enhancement.threshold_controller import ThresholdController

        controller = ThresholdController()
        controller.set_threshold(1.0)
        self.assertEqual(controller.get_threshold(), 1.0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
