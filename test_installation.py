"""Comprehensive environment validation for YOLOv8 Real-Time Object Detection.

Run this script after installing dependencies to verify that your setup is ready.
"""

from __future__ import annotations

import os
import platform
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Tuple


@dataclass
class TestResult:
    """Stores one test result."""

    name: str
    passed: bool
    details: str
    solution: Optional[str] = None


class Console:
    """Simple console formatter with colored symbols when supported."""

    GREEN = "\033[92m"
    RED = "\033[91m"
    YELLOW = "\033[93m"
    BLUE = "\033[94m"
    RESET = "\033[0m"

    @staticmethod
    def supports_ansi() -> bool:
        return sys.stdout.isatty()

    @classmethod
    def color(cls, text: str, code: str) -> str:
        if cls.supports_ansi():
            return f"{code}{text}{cls.RESET}"
        return text

    @classmethod
    def ok(cls, text: str) -> str:
        return cls.color(f"[PASS] {text}", cls.GREEN)

    @classmethod
    def fail(cls, text: str) -> str:
        return cls.color(f"[FAIL] {text}", cls.RED)

    @classmethod
    def warn(cls, text: str) -> str:
        return cls.color(f"[WARN] {text}", cls.YELLOW)

    @classmethod
    def info(cls, text: str) -> str:
        return cls.color(f"[INFO] {text}", cls.BLUE)


def print_header(title: str) -> None:
    print("\n" + "=" * 80)
    print(title)
    print("=" * 80)


def get_total_memory_gb() -> Optional[float]:
    """Get total RAM in GB using stdlib only (best effort)."""
    try:
        if platform.system() == "Windows":
            import ctypes

            class MEMORYSTATUSEX(ctypes.Structure):
                _fields_ = [
                    ("dwLength", ctypes.c_ulong),
                    ("dwMemoryLoad", ctypes.c_ulong),
                    ("ullTotalPhys", ctypes.c_ulonglong),
                    ("ullAvailPhys", ctypes.c_ulonglong),
                    ("ullTotalPageFile", ctypes.c_ulonglong),
                    ("ullAvailPageFile", ctypes.c_ulonglong),
                    ("ullTotalVirtual", ctypes.c_ulonglong),
                    ("ullAvailVirtual", ctypes.c_ulonglong),
                    ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
                ]

            stat = MEMORYSTATUSEX()
            stat.dwLength = ctypes.sizeof(MEMORYSTATUSEX)
            ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(stat))
            return round(stat.ullTotalPhys / (1024**3), 2)

        if hasattr(os, "sysconf"):
            pages = os.sysconf("SC_PHYS_PAGES")
            page_size = os.sysconf("SC_PAGE_SIZE")
            return round((pages * page_size) / (1024**3), 2)
    except Exception:
        return None
    return None


def check_python_version() -> TestResult:
    required = (3, 9)
    current = sys.version_info[:3]
    passed = current >= required
    details = f"Python version detected: {current[0]}.{current[1]}.{current[2]}"
    solution = None if passed else "Install Python 3.9+ from https://www.python.org/downloads/"
    return TestResult("Python Version (3.9+)", passed, details, solution)


def check_imports() -> Tuple[TestResult, Dict[str, str], Dict[str, object]]:
    versions: Dict[str, str] = {}
    modules: Dict[str, object] = {}
    packages = [
        ("numpy", "numpy"),
        ("pandas", "pandas"),
        ("Pillow", "PIL"),
        ("OpenCV", "cv2"),
        ("PyTorch", "torch"),
        ("torchvision", "torchvision"),
        ("Ultralytics", "ultralytics"),
    ]

    errors = []
    for display_name, module_name in packages:
        try:
            mod = __import__(module_name)
            modules[module_name] = mod
            version = getattr(mod, "__version__", "unknown")
            versions[display_name] = str(version)
        except Exception as exc:
            errors.append(f"{display_name}: {exc}")

    if errors:
        return (
            TestResult(
                name="Library Imports",
                passed=False,
                details="; ".join(errors),
                solution="Re-run: pip install -r requirements.txt",
            ),
            versions,
            modules,
        )

    detail_lines = [f"{k}={v}" for k, v in versions.items()]
    return (
        TestResult(
            name="Library Imports",
            passed=True,
            details=", ".join(detail_lines),
        ),
        versions,
        modules,
    )


def check_torch_cuda(torch_module: object) -> TestResult:
    try:
        torch = torch_module
        torch_version = getattr(torch, "__version__", "unknown")
        cuda_available = torch.cuda.is_available()
        device_count = torch.cuda.device_count() if cuda_available else 0

        lines = [
            f"PyTorch version: {torch_version}",
            f"CUDA available: {cuda_available}",
            f"CUDA device count: {device_count}",
        ]

        if cuda_available:
            names = [torch.cuda.get_device_name(i) for i in range(device_count)]
            lines.append(f"Devices: {names}")

        return TestResult("PyTorch/CUDA", True, " | ".join(lines))
    except Exception as exc:
        return TestResult(
            "PyTorch/CUDA",
            False,
            f"CUDA check failed: {exc}",
            "If you need GPU, install CUDA-enabled torch build from pytorch.org.",
        )


def check_yolov8_model() -> Tuple[TestResult, Optional[object]]:
    try:
        from ultralytics import YOLO

        model_path = "yolov8n.pt"
        local_path = Path(model_path)

        # This will load local file if present, otherwise Ultralytics downloads pretrained weights.
        model = YOLO(model_path)

        resolved = None
        ckpt_path = getattr(model, "ckpt_path", None)
        if ckpt_path and Path(str(ckpt_path)).exists():
            resolved = str(ckpt_path)
        elif local_path.exists():
            resolved = str(local_path.resolve())

        if resolved:
            details = f"Model loaded successfully. Weights path: {resolved}"
        else:
            details = "Model loaded via Ultralytics cache, but local path could not be resolved."

        return TestResult("YOLOv8 Model Load", True, details), model
    except Exception as exc:
        return (
            TestResult(
                "YOLOv8 Model Load",
                False,
                f"Could not load yolov8n.pt: {exc}",
                "Check internet once for first-time model download and verify ultralytics installation.",
            ),
            None,
        )


def check_webcam_and_inference(model: Optional[object]) -> TestResult:
    if model is None:
        return TestResult(
            "Webcam + Inference",
            False,
            "Skipped because model did not load.",
            "Fix model loading first, then rerun this test.",
        )

    try:
        import cv2

        cap = cv2.VideoCapture(0)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

        if not cap.isOpened():
            return TestResult(
                "Webcam + Inference",
                False,
                "Webcam is not accessible (camera index 0).",
                "Close other apps using webcam and verify camera permissions.",
            )

        ok, frame = cap.read()
        if not ok or frame is None:
            cap.release()
            return TestResult(
                "Webcam + Inference",
                False,
                "Could not capture frame from webcam.",
                "Try a different camera index (1/2) or reconnect webcam.",
            )

        start = time.time()
        results = model(frame, verbose=False)
        elapsed_ms = (time.time() - start) * 1000

        rendered = results[0].plot()
        detections = len(results[0].boxes) if results and results[0].boxes is not None else 0

        # Display test output for one second when GUI backend is available.
        try:
            cv2.imshow("YOLOv8 Setup Test", rendered)
            cv2.waitKey(1000)
            cv2.destroyAllWindows()
            display_note = "Display test: shown for 1 second"
        except Exception:
            display_note = "Display test skipped (headless/no GUI backend)"

        cap.release()

        return TestResult(
            "Webcam + Inference",
            True,
            f"Frame captured and inference ran ({detections} detections, {elapsed_ms:.2f} ms). {display_note}",
        )
    except Exception as exc:
        return TestResult(
            "Webcam + Inference",
            False,
            f"Webcam/inference test failed: {exc}",
            "Update OpenCV, check webcam permissions, and ensure model loads successfully.",
        )


def check_system_information() -> TestResult:
    try:
        os_name = platform.system()
        os_version = platform.version()
        machine = platform.machine()
        processor = platform.processor() or "unknown"
        cpu_count = os.cpu_count() or 0
        memory_gb = get_total_memory_gb()

        details = (
            f"OS={os_name} ({os_version}), Arch={machine}, CPU={processor}, "
            f"Cores={cpu_count}, RAM={memory_gb if memory_gb is not None else 'unknown'} GB"
        )
        return TestResult("System Information", True, details)
    except Exception as exc:
        return TestResult("System Information", False, f"System info check failed: {exc}")


def print_result(result: TestResult) -> None:
    symbol_line = Console.ok(result.name) if result.passed else Console.fail(result.name)
    print(symbol_line)
    print(f"  Details: {result.details}")
    if result.solution:
        print(f"  Suggested fix: {result.solution}")


def main() -> None:
    print_header("YOLOv8 Environment Installation Test")

    results = []

    # 1) Python version check
    py_result = check_python_version()
    results.append(py_result)
    print_result(py_result)

    # 2) Imports + versions
    import_result, versions, modules = check_imports()
    results.append(import_result)
    print_result(import_result)

    if versions:
        print(Console.info("Detected package versions:"))
        for pkg, ver in versions.items():
            print(f"  - {pkg}: {ver}")

    # 3) Torch/CUDA check
    torch_mod = modules.get("torch")
    if torch_mod is not None:
        cuda_result = check_torch_cuda(torch_mod)
    else:
        cuda_result = TestResult(
            "PyTorch/CUDA",
            False,
            "Skipped because torch import failed.",
            "Install torch and torchvision correctly.",
        )
    results.append(cuda_result)
    print_result(cuda_result)

    # 4) YOLO model check
    model_result, model = check_yolov8_model()
    results.append(model_result)
    print_result(model_result)

    # 5) Webcam + one-frame inference
    webcam_result = check_webcam_and_inference(model)
    results.append(webcam_result)
    print_result(webcam_result)

    # 6) System information
    sys_result = check_system_information()
    results.append(sys_result)
    print_result(sys_result)

    # 7) Final summary
    print_header("Final Summary")
    passed = [r for r in results if r.passed]
    failed = [r for r in results if not r.passed]

    for item in results:
        marker = "CHECK" if item.passed else "X"
        print(f"{marker} {item.name}")

    print(f"\nTotal: {len(results)} | Passed: {len(passed)} | Failed: {len(failed)}")

    if not failed:
        print(Console.ok("Environment Status: READY"))
        print("Next steps:")
        print("  1) Implement real-time detection loop in main.py")
        print("  2) Tune thresholds in config.py")
        print("  3) Start project with: python main.py")
    else:
        print(Console.fail("Environment Status: NOT READY"))
        print("Next steps:")
        print("  1) Resolve failed checks shown above")
        print("  2) Re-run: python test_installation.py")
        print("  3) Continue only after all critical checks pass")


if __name__ == "__main__":
    main()
