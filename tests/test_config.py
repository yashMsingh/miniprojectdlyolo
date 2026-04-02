"""Unit tests for configuration objects."""

from __future__ import annotations

import unittest

from config import AppConfig


class TestConfig(unittest.TestCase):
    """Test configuration classes."""

    def setUp(self) -> None:
        self.config = AppConfig.default()

    def test_config_default_values(self) -> None:
        self.assertEqual(self.config.model.confidence_threshold, 0.5)
        self.assertEqual(self.config.webcam.frame_width, 640)
        self.assertEqual(self.config.webcam.frame_height, 480)

    def test_config_path_creation_values(self) -> None:
        self.assertIsNotNone(self.config.output.output_dir)
        self.assertIsNotNone(self.config.logging.log_dir)

    def test_config_value_ranges(self) -> None:
        self.assertGreaterEqual(self.config.model.confidence_threshold, 0.0)
        self.assertLessEqual(self.config.model.confidence_threshold, 1.0)
        self.assertGreater(self.config.webcam.target_fps, 0)

    def test_default_device(self) -> None:
        self.assertIn(self.config.model.device, ["cpu", "cuda"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
