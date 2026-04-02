"""Cross-platform compatibility tests."""

from __future__ import annotations

import os
import pathlib
import platform
import shutil
import tempfile
import unittest

from utils import ensure_path_exists, get_safe_path


class TestCrossPlatformCompatibility(unittest.TestCase):
    """Test cross-platform compatibility."""

    def test_platform_detection(self) -> None:
        current_platform = platform.system()
        self.assertIn(current_platform, ["Windows", "Darwin", "Linux"])

    def test_path_handling(self) -> None:
        path = get_safe_path("./test", create=False)
        self.assertIsNotNone(path)

        p = pathlib.Path("./outputs")
        self.assertEqual(p.name, "outputs")

    def test_file_operations(self) -> None:
        temp_dir = tempfile.mkdtemp(prefix="yolo_test_")
        try:
            test_path = os.path.join(temp_dir, "test_dir")
            ensure_path_exists(test_path)
            self.assertTrue(os.path.exists(test_path))
        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)

    def test_opencv_compatibility(self) -> None:
        import cv2

        version = cv2.__version__
        self.assertIsNotNone(version)

    def test_torch_compatibility(self) -> None:
        import torch

        self.assertIsNotNone(torch.__version__)


if __name__ == "__main__":
    print(f"\nTesting on: {platform.platform()}\n")
    unittest.main(verbosity=2)
