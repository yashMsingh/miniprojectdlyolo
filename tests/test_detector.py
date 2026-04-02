"""Unit tests for detector components."""

from __future__ import annotations

import unittest
from unittest.mock import patch

import numpy as np

from config import AppConfig
from detector.base_detector import BaseDetector
from detector.camera_handler import CameraHandler
from utils import setup_logger


class _FakeParam:
    def __init__(self, n: int = 10, requires_grad: bool = True) -> None:
        self._n = n
        self.requires_grad = requires_grad

    def numel(self) -> int:
        return self._n


class _FakeNet:
    def parameters(self):
        return [_FakeParam(10, True), _FakeParam(5, False)]

    def modules(self):
        return [object(), object()]


class _FakeYOLO:
    def __init__(self, *_args, **_kwargs) -> None:
        self.model = _FakeNet()


class _FakeCap:
    def __init__(self, *_args, **_kwargs) -> None:
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


class TestBaseDetector(unittest.TestCase):
    """Test base detector functionality."""

    @patch("detector.base_detector.YOLO", _FakeYOLO)
    @patch("detector.base_detector.cv2.VideoCapture", _FakeCap)
    def setUp(self) -> None:
        self.config = AppConfig.default()
        self.detector = BaseDetector(self.config)

    def test_detector_initialization(self) -> None:
        self.assertIsNotNone(self.detector.model)
        self.assertIsNone(self.detector.cap)

    def test_model_loading(self) -> None:
        model_info = self.detector.get_model_info()
        self.assertTrue(model_info["loaded"])
        self.assertGreater(model_info["parameters"], 0)

    @patch("detector.base_detector.cv2.VideoCapture", _FakeCap)
    def test_camera_availability(self) -> None:
        available = self.detector.check_camera_available()
        self.assertIsInstance(available, bool)

    def tearDown(self) -> None:
        self.detector.release_resources()


class TestCameraHandler(unittest.TestCase):
    """Test camera handler functionality."""

    def setUp(self) -> None:
        self.config = AppConfig.default()
        self.logger = setup_logger("test_camera_handler", enable_logging=False)
        self.cap = _FakeCap()
        self.handler = CameraHandler(self.cap, self.config, self.logger)

    def test_camera_handler_initialization(self) -> None:
        self.assertIsNotNone(self.handler)

    def test_frame_capture(self) -> None:
        ret, frame = self.handler.read_frame()
        self.assertIsInstance(ret, bool)
        if ret:
            self.assertIsNotNone(frame)

    def tearDown(self) -> None:
        self.handler.release()


if __name__ == "__main__":
    unittest.main(verbosity=2)
