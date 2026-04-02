"""Class-based detection filtering."""

from __future__ import annotations

from typing import Any, Dict, List, Set


class ClassFilter:
    """Filter detections by enabled object classes."""

    def __init__(self, all_classes: List[str] | None = None, logger: Any = None) -> None:
        """Initialize class filter.

        Args:
            all_classes: List of all available class names.
            logger: Logger instance.
        """
        self.all_classes = set(all_classes or [])
        self.enabled_classes: Set[str] = set(self.all_classes)
        self.logger = logger

    def set_enabled_classes(self, class_list: List[str]) -> bool:
        """Set which classes to detect.

        Args:
            class_list: List of class names to enable.

        Returns:
            True if all classes are valid.
        """
        invalid = set(class_list) - self.all_classes
        if invalid and self.logger:
            self.logger.warning(f"Invalid class names: {invalid}")

        self.enabled_classes = set(class_list) & self.all_classes
        return len(invalid) == 0

    def get_enabled_classes(self) -> List[str]:
        """Return list of enabled class names."""
        return sorted(self.enabled_classes)

    def add_class(self, class_name: str) -> bool:
        """Enable single class.

        Args:
            class_name: Class to enable.

        Returns:
            True if valid class.
        """
        if class_name not in self.all_classes:
            if self.logger:
                self.logger.warning(f"Unknown class: {class_name}")
            return False
        self.enabled_classes.add(class_name)
        return True

    def remove_class(self, class_name: str) -> bool:
        """Disable single class.

        Args:
            class_name: Class to disable.

        Returns:
            True if class was enabled.
        """
        was_enabled = class_name in self.enabled_classes
        self.enabled_classes.discard(class_name)
        return was_enabled

    def toggle_class(self, class_name: str) -> bool:
        """Toggle class enabled/disabled state.

        Args:
            class_name: Class to toggle.

        Returns:
            True if now enabled, False if now disabled.
        """
        if class_name in self.enabled_classes:
            self.enabled_classes.remove(class_name)
            return False
        else:
            if class_name in self.all_classes:
                self.enabled_classes.add(class_name)
                return True
            return False

    def filter_detections(self, detections: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Filter detections by enabled classes.

        Args:
            detections: List of detection dictionaries.

        Returns:
            Filtered list of detections.
        """
        return [
            d for d in detections
            if str(d.get("class_name", "")).lower() in
            {c.lower() for c in self.enabled_classes}
        ]

    def enable_all_classes(self) -> None:
        """Enable all available classes."""
        self.enabled_classes = set(self.all_classes)

    def disable_all_classes(self) -> None:
        """Disable all classes."""
        self.enabled_classes.clear()

    def get_filter_info(self) -> str:
        """Return formatted filter status string."""
        if not self.enabled_classes:
            return "Classes: NONE"
        if len(self.enabled_classes) == len(self.all_classes):
            return "Classes: ALL"
        return f"Classes: {', '.join(sorted(self.enabled_classes))}"

    def get_filter_summary(self) -> Dict[str, Any]:
        """Return detailed filter status."""
        return {
            "enabled_classes": self.get_enabled_classes(),
            "disabled_classes": sorted(self.all_classes - self.enabled_classes),
            "enabled_count": len(self.enabled_classes),
            "total_available": len(self.all_classes),
        }
