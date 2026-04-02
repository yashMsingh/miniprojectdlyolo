"""Master test runner for all project tests."""

from __future__ import annotations

import sys
import unittest


def run_all_tests() -> int:
    """Discover and run all tests under the tests package."""
    print("=" * 70)
    print("YOLOv8 PROJECT - COMPREHENSIVE TEST SUITE")
    print("=" * 70)

    loader = unittest.TestLoader()
    suite = loader.discover("tests", pattern="test_*.py")

    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)

    print("\n" + "=" * 70)
    print("TEST SUMMARY")
    print("=" * 70)
    print(f"Tests Run: {result.testsRun}")
    print(f"Successes: {result.testsRun - len(result.failures) - len(result.errors)}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    print(f"Skipped: {len(result.skipped)}")

    if result.wasSuccessful():
        print("\nALL TESTS PASSED")
        return 0

    print("\nSOME TESTS FAILED")
    return 1


if __name__ == "__main__":
    sys.exit(run_all_tests())
