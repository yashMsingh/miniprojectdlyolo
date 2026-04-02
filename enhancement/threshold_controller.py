"""Dynamic confidence threshold controller."""

from __future__ import annotations

from typing import Tuple


class ThresholdController:
    """Manage real-time confidence threshold adjustments."""

    def __init__(self, initial_threshold: float = 0.5, step: float = 0.05) -> None:
        """Initialize threshold controller.

        Args:
            initial_threshold: Starting threshold (0.0-1.0).
            step: Increment/decrement step size.
        """
        self.current_threshold = max(0.0, min(1.0, initial_threshold))
        self.default_threshold = initial_threshold
        self.step = max(0.01, min(0.1, step))
        self.min_threshold = 0.0
        self.max_threshold = 1.0

    def increase_threshold(self, step: float | None = None) -> float:
        """Increase threshold by step amount.

        Args:
            step: Custom step size, or use default.

        Returns:
            New threshold value.
        """
        delta = step if step is not None else self.step
        self.current_threshold = min(self.max_threshold, self.current_threshold + delta)
        return self.current_threshold

    def decrease_threshold(self, step: float | None = None) -> float:
        """Decrease threshold by step amount.

        Args:
            step: Custom step size, or use default.

        Returns:
            New threshold value.
        """
        delta = step if step is not None else self.step
        self.current_threshold = max(self.min_threshold, self.current_threshold - delta)
        return self.current_threshold

    def set_threshold(self, value: float) -> bool:
        """Set threshold to specific value.

        Args:
            value: Target threshold (0.0-1.0).

        Returns:
            True if valid and set, False if out of bounds.
        """
        if not (0.0 <= value <= 1.0):
            return False
        self.current_threshold = value
        return True

    def get_threshold(self) -> float:
        """Return current threshold."""
        return round(self.current_threshold, 2)

    def get_threshold_range(self) -> Tuple[float, float]:
        """Return min and max threshold bounds."""
        return (self.min_threshold, self.max_threshold)

    def get_threshold_info(self) -> str:
        """Return formatted threshold info string."""
        pct = int(self.current_threshold * 100)
        return f"Confidence: {pct}%"

    def reset_to_default(self) -> float:
        """Reset to initial default threshold.

        Returns:
            Reset threshold value.
        """
        self.current_threshold = self.default_threshold
        return self.current_threshold

    def is_valid_threshold(self, value: float) -> bool:
        """Check if value is valid threshold."""
        return isinstance(value, (int, float)) and 0.0 <= value <= 1.0
