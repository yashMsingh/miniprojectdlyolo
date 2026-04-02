"""Run cross-platform compatibility tests."""

from __future__ import annotations

import platform
import unittest

from tests.test_cross_platform import TestCrossPlatformCompatibility


if __name__ == "__main__":
    print(f"\nTesting on: {platform.platform()}\n")
    suite = unittest.defaultTestLoader.loadTestsFromTestCase(TestCrossPlatformCompatibility)
    unittest.TextTestRunner(verbosity=2).run(suite)
