"""Memory usage profiler for stress testing and leak checks."""

from __future__ import annotations

import os
from datetime import datetime
from typing import Dict, List

try:
    import psutil
except Exception:  # pragma: no cover
    psutil = None


class MemoryProfiler:
    """Profile process memory usage over time."""

    def __init__(self) -> None:
        if psutil is None:
            raise RuntimeError("psutil is required for MemoryProfiler")
        self.process = psutil.Process(os.getpid())
        self.memory_snapshots: List[Dict[str, object]] = []

    def get_memory_usage(self) -> float:
        """Return current RSS memory in MB."""
        return self.process.memory_info().rss / 1024.0 / 1024.0

    def get_memory_info(self) -> Dict[str, float]:
        """Return detailed memory metrics in MB and percent."""
        mem = self.process.memory_info()
        return {
            "rss_mb": mem.rss / 1024.0 / 1024.0,
            "vms_mb": mem.vms / 1024.0 / 1024.0,
            "percent": float(self.process.memory_percent()),
        }

    def record_snapshot(self, label: str = "") -> Dict[str, object]:
        """Record memory snapshot with timestamp and label."""
        snapshot = {
            "timestamp": datetime.now(),
            "label": label,
            "memory_mb": self.get_memory_usage(),
        }
        self.memory_snapshots.append(snapshot)
        return snapshot

    def get_memory_growth(self) -> float:
        """Return memory growth from first to last snapshot in MB."""
        if len(self.memory_snapshots) < 2:
            return 0.0
        first = float(self.memory_snapshots[0]["memory_mb"])
        last = float(self.memory_snapshots[-1]["memory_mb"])
        return last - first

    def check_memory_leak(self, threshold_mb: float = 100.0) -> Dict[str, float | bool | int]:
        """Evaluate whether observed memory growth indicates potential leak."""
        growth = self.get_memory_growth()
        return {
            "has_leak": growth > threshold_mb,
            "growth_mb": growth,
            "threshold_mb": threshold_mb,
            "snapshots_count": len(self.memory_snapshots),
        }

    def print_memory_report(self) -> None:
        """Print memory usage timeline report."""
        print("\n" + "=" * 50)
        print("MEMORY USAGE REPORT")
        print("=" * 50)

        for snapshot in self.memory_snapshots:
            ts = snapshot["timestamp"]
            label = snapshot["label"]
            mem = snapshot["memory_mb"]
            print(f"{ts.strftime('%H:%M:%S')} [{label}]: {mem:.2f} MB")

        leak_analysis = self.check_memory_leak()
        print(f"\nTotal Growth: {leak_analysis['growth_mb']:.2f} MB")
        print("Leak Status: POTENTIAL LEAK" if leak_analysis["has_leak"] else "Leak Status: OK")
