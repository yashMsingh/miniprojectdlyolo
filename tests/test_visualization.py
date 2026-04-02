"""Unit tests for visualization helpers."""

from __future__ import annotations

import unittest

import numpy as np

from config import AppConfig
from utils import setup_logger
from visualization.stats_tracker import StatsTracker
from visualization.visualizer import Visualizer


class TestVisualizer(unittest.TestCase):
    """Test visualization functionality."""

    def setUp(self) -> None:
        self.config = AppConfig.default()
        self.logger = setup_logger("test_visualizer", enable_logging=False)
        self.visualizer = Visualizer(self.config, self.logger)
        self.frame = np.zeros((480, 640, 3), dtype=np.uint8)

    def test_visualizer_initialization(self) -> None:
        self.assertIsNotNone(self.visualizer)

    def test_draw_box_does_not_crash(self) -> None:
        detection = {
            "box": (10, 10, 100, 100),
            "confidence": 0.95,
            "class_name": "person",
            "class_id": 0,
        }
        out = self.visualizer.draw_box(self.frame.copy(), detection)
        self.assertIsNotNone(out)

    def test_draw_label_does_not_crash(self) -> None:
        detection = {
            "box": (10, 10, 100, 100),
            "confidence": 0.95,
            "class_name": "person",
        }
        out = self.visualizer.draw_label(self.frame.copy(), detection)
        self.assertIsNotNone(out)

    def test_draw_fps_does_not_crash(self) -> None:
        out = self.visualizer.draw_fps(self.frame.copy(), fps=30.5)
        self.assertIsNotNone(out)


class TestStatsTracker(unittest.TestCase):
    """Test statistics tracking."""

    def setUp(self) -> None:
        self.stats = StatsTracker()

    def test_stats_initialization(self) -> None:
        self.assertIsNotNone(self.stats)

    def test_stats_update(self) -> None:
        detections = [
            {"class_name": "person", "confidence": 0.95},
            {"class_name": "car", "confidence": 0.87},
        ]
        self.stats.update_stats(detections, fps=30.0, inference_ms=10.0, processing_ms=14.0)
        current = self.stats.get_current_stats()
        self.assertEqual(current["total_detections"], 2)

    def test_stats_reset(self) -> None:
        self.stats.update_stats([], fps=0.0, inference_ms=0.0, processing_ms=0.0)
        self.stats.reset_stats()
        stats = self.stats.get_current_stats()
        self.assertEqual(stats["total_detections"], 0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
