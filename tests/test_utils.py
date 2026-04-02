"""Unit tests for utility helpers."""

from __future__ import annotations

import unittest

import numpy as np

from utils import (
    calculate_aspect_ratio,
    calculate_box_area,
    calculate_fps,
    check_memory,
    create_output_dirs,
    filter_by_confidence,
    format_inference_time,
    get_class_name,
    get_device,
    get_frame_shape,
    get_inference_stats,
    get_safe_path,
    get_system_info,
    validate_box_coordinates,
    validate_frame,
    yolo_results_to_dict,
)


class _FakeTensor:
    def __init__(self, arr: np.ndarray) -> None:
        self._arr = arr

    def cpu(self) -> "_FakeTensor":
        return self

    def numpy(self) -> np.ndarray:
        return self._arr


class _FakeBoxes:
    def __init__(self) -> None:
        self.xyxy = _FakeTensor(np.array([[10, 10, 100, 100]], dtype=np.float32))
        self.conf = _FakeTensor(np.array([0.9], dtype=np.float32))
        self.cls = _FakeTensor(np.array([0], dtype=np.float32))


class _FakeResult:
    def __init__(self) -> None:
        self.names = {0: "person"}
        self.boxes = _FakeBoxes()


class TestUtilsFunctions(unittest.TestCase):
    """Test utility functions."""

    def test_get_device(self) -> None:
        device = get_device()
        self.assertIn(device, ["cuda", "cpu"])

    def test_get_system_info(self) -> None:
        info = get_system_info()
        self.assertIn("platform", info)
        self.assertIn("python_version", info)

    def test_check_memory(self) -> None:
        memory = check_memory()
        self.assertIn("total_gb", memory)
        self.assertIn("available_gb", memory)

    def test_create_output_dirs(self) -> None:
        paths = create_output_dirs("./outputs/test_utils")
        self.assertIn("output", paths)
        self.assertIn("videos", paths)
        self.assertIn("screenshots", paths)

    def test_get_safe_path(self) -> None:
        safe_path = get_safe_path("./outputs", create=False)
        self.assertTrue(str(safe_path))

    def test_validate_frame(self) -> None:
        valid_frame = np.zeros((480, 640, 3), dtype=np.uint8)
        self.assertTrue(validate_frame(valid_frame))
        self.assertFalse(validate_frame(None))

    def test_get_frame_shape(self) -> None:
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        shape = get_frame_shape(frame)
        self.assertEqual(shape, (480, 640, 3))

    def test_calculate_aspect_ratio(self) -> None:
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        ratio = calculate_aspect_ratio(frame)
        self.assertAlmostEqual(ratio, 640 / 480, places=2)

    def test_format_inference_time(self) -> None:
        formatted = format_inference_time(15.5)
        self.assertIn("ms", formatted)
        self.assertIn("FPS", formatted)

    def test_calculate_fps(self) -> None:
        fps = calculate_fps(process_seconds=1.0 / 30.0)
        self.assertAlmostEqual(fps, 30, places=0)

    def test_get_inference_stats(self) -> None:
        stats = get_inference_stats([10.0, 20.0, 30.0])
        self.assertAlmostEqual(stats["avg_ms"], 20.0)
        self.assertAlmostEqual(stats["min_ms"], 10.0)
        self.assertAlmostEqual(stats["max_ms"], 30.0)

    def test_validate_box_coordinates(self) -> None:
        valid_box = (10, 10, 100, 100)
        self.assertTrue(validate_box_coordinates(valid_box, (480, 640, 3)))

        invalid_box = (100, 100, 10, 10)
        self.assertFalse(validate_box_coordinates(invalid_box, (480, 640, 3)))

    def test_calculate_box_area(self) -> None:
        box = (0, 0, 100, 100)
        area = calculate_box_area(box)
        self.assertEqual(area, 10000)

    def test_yolo_results_to_dict(self) -> None:
        detections = yolo_results_to_dict([_FakeResult()])
        self.assertEqual(len(detections), 1)
        self.assertEqual(detections[0]["class_name"], "person")

    def test_filter_by_confidence(self) -> None:
        detections = [
            {"confidence": 0.9, "class_name": "person"},
            {"confidence": 0.3, "class_name": "car"},
        ]
        out = filter_by_confidence(detections, 0.5)
        self.assertEqual(len(out), 1)

    def test_get_class_name(self) -> None:
        self.assertEqual(get_class_name(0, {0: "person"}), "person")


if __name__ == "__main__":
    unittest.main(verbosity=2)
