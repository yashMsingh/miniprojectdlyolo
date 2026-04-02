"""Interactive demo launcher for YOLOv8 project features."""

from __future__ import annotations

import os
from typing import Dict

from config import AppConfig
from detector.main_detector import MainDetector


def print_section(title: str) -> None:
    """Print section header for demo output."""
    print("\n" + "=" * 70)
    print(f"  {title}")
    print("=" * 70)


def run_detector_session(header: str) -> None:
    """Start detector session and clean up safely."""
    print_section(header)
    config = AppConfig.default()
    detector = MainDetector(config=config)
    try:
        detector.start()
    except KeyboardInterrupt:
        print("Interrupted by user.")
    finally:
        detector.shutdown()


def demo_basic_detection() -> None:
    """Demo 1: Basic object detection."""
    print("Observe bounding boxes, class labels, and FPS.")
    print("Press q to end this demo.")
    run_detector_session("DEMO 1: BASIC OBJECT DETECTION")


def demo_threshold_control() -> None:
    """Demo 2: Threshold control."""
    print("Use + and - to change threshold during runtime.")
    print("Press q to end this demo.")
    run_detector_session("DEMO 2: THRESHOLD CONTROL")


def demo_all_features() -> None:
    """Demo 3: Full features."""
    print("Try r for recording, s for screenshot, p for performance, d for demo mode.")
    print("Press q to end this demo.")
    run_detector_session("DEMO 3: FULL FEATURE SHOWCASE")


def show_generated_files() -> None:
    """Display generated output files by folder."""
    print_section("GENERATED OUTPUT FILES")
    output_dirs: Dict[str, str] = {
        "screenshots": "outputs/screenshots",
        "videos": "outputs/videos",
        "detections": "outputs/detections",
        "reports": "outputs/reports",
    }

    for name, path in output_dirs.items():
        print(f"\n{name.upper()}:")
        if os.path.exists(path):
            files = sorted(os.listdir(path))
            if files:
                for file in files:
                    print(f"  - {file}")
            else:
                print("  (empty)")
        else:
            print("  (directory not found)")


def main() -> None:
    """Run interactive demo menu."""
    print("\n" + "=" * 70)
    print("  YOLOv8 REAL-TIME OBJECT DETECTION - DEMO MENU")
    print("=" * 70)

    while True:
        print("\nAvailable Demos:")
        print("1. Basic Object Detection")
        print("2. Threshold Control")
        print("3. All Features")
        print("4. Show Generated Files")
        print("5. Exit")

        choice = input("\nSelect demo (1-5): ").strip()
        if choice == "1":
            demo_basic_detection()
        elif choice == "2":
            demo_threshold_control()
        elif choice == "3":
            demo_all_features()
        elif choice == "4":
            show_generated_files()
        elif choice == "5":
            print("Exiting demo.")
            break
        else:
            print("Invalid choice. Try again.")


if __name__ == "__main__":
    main()
