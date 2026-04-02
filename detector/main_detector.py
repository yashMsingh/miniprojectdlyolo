"""Main detector orchestration class for real-time detection pipeline."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np

from config import AppConfig
from enhancement.class_filter import ClassFilter
from enhancement.counter import ObjectCounter
from enhancement.demo_mode import DemoMode
from enhancement.detection_logger import DetectionLogger
from enhancement.performance_monitor import PerformanceMonitor
from enhancement.report_generator import AssignmentReportGenerator
from enhancement.threshold_controller import ThresholdController
from enhancement.video_recorder import VideoRecorder
from utils import (
    calculate_fps,
    create_output_dirs,
    ensure_path_exists,
    format_inference_time,
    setup_logger,
    validate_frame,
)
from visualization.frame_decorator import FrameDecorator
from visualization.stats_tracker import StatsTracker
from visualization.visualizer import Visualizer

from .base_detector import BaseDetector
from .camera_handler import CameraHandler
from .detection_processor import DetectionProcessor
from .inference_pipeline import InferencePipeline


class MainDetector(BaseDetector):
    """Coordinates camera, inference, processing, and visualization."""

    def __init__(self, config_path: Optional[str] = None, config: Optional[AppConfig] = None) -> None:
        """Initialize main detector resources.

        Args:
            config_path: Optional JSON config path.
            config: Optional prebuilt configuration object.
        """
        loaded_config = config
        if loaded_config is None:
            loaded_config = AppConfig.load_json(config_path) if config_path else AppConfig.default()

        super().__init__(loaded_config)

        self.logger = setup_logger(
            name=self.__class__.__name__,
            log_dir=self.config.logging.log_dir,
            log_level=self.config.logging.log_level,
            enable_logging=self.config.logging.enable_logging,
        )

        self.output_paths = create_output_dirs(self.config.output.output_dir)
        self.output_paths["detections"] = str(
            ensure_path_exists(str(Path(self.output_paths["output"]) / "detections"))
        )
        self.output_paths["reports"] = str(
            ensure_path_exists(str(Path(self.output_paths["output"]) / "reports"))
        )
        self.cap = self.initialize_camera()
        self.camera_handler = CameraHandler(self.cap, self.config, self.logger)
        self.processor = DetectionProcessor()
        self.inference_pipeline = InferencePipeline(self.model, self.config, self.logger)
        self.visualizer = Visualizer(self.config, self.logger)
        self.stats_tracker = StatsTracker()
        self.frame_decorator = FrameDecorator(self.config, self.logger)

        # Phase 3 Enhancements
        self.object_counter = ObjectCounter()
        self.threshold_controller = ThresholdController()
        self.video_recorder = VideoRecorder(self.logger)
        self._output_paths = self.output_paths
        self.detection_logger = DetectionLogger(self.logger)
        self.class_filter = ClassFilter(list(self.model.names.values()), self.logger)
        self.performance_monitor = PerformanceMonitor(logger=self.logger)
        self.demo_mode = DemoMode(self.config, self.logger)
        self.report_generator = AssignmentReportGenerator(self.logger)

        self.window_name = "YOLOv8 Real-Time Detection"
        self.video_writer: Optional[cv2.VideoWriter] = None
        self.last_frame: Optional[np.ndarray] = None
        self.last_detections: List[Dict[str, Any]] = []

        # UI State
        self.show_help = False
        self.show_performance = False
        self.show_threshold_indicator = True
        self.show_screenshot_notification = False
        self.screenshot_notification_countdown = 0
        self.paused = False

        self._setup_video_writer_if_enabled()
        self.logger.info("MainDetector initialized successfully.")
        self.logger.info(
            "Available YOLO classes: %s",
            ", ".join(self.model.names.values()),
        )

    def _setup_video_writer_if_enabled(self) -> None:
        """Initialize video writer when video saving is enabled."""
        if not self.config.output.save_video:
            return

        try:
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            out_path = Path(self.output_paths["videos"]) / f"detected_{timestamp}.mp4"
            fourcc = cv2.VideoWriter_fourcc(*self.config.output.video_codec)
            self.video_writer = cv2.VideoWriter(
                str(out_path),
                fourcc,
                self.config.output.video_fps,
                (self.config.webcam.frame_width, self.config.webcam.frame_height),
            )
            self.logger.info("Video recording enabled: %s", out_path)
        except Exception as exc:
            self.logger.warning("Video writer init failed: %s", exc)
            self.video_writer = None

    def run_inference(self, frame: np.ndarray) -> Tuple[List[Dict[str, Any]], float]:
        """Run YOLO inference on a frame and return detections plus time.

        Args:
            frame: Input frame from webcam.

        Returns:
            Tuple of detection list and inference time in milliseconds.
        """
        detections, metrics = self.inference_pipeline.process_frame_pipeline(frame)

        # Apply phase 3 enhancements: threshold controller and class filter
        threshold = self.threshold_controller.get_threshold()
        filtered_by_threshold = [d for d in detections if d.get("confidence", 0.0) >= threshold]
        
        filtered_by_class = self.class_filter.filter_detections(filtered_by_threshold)

        return filtered_by_class, float(metrics.get("inference_ms", 0.0))

    def process_frame(self, frame: np.ndarray) -> Tuple[List[Dict[str, Any]], Dict[str, float]]:
        """Process a single frame with inference and detection stats."""
        start = time.perf_counter()

        if not validate_frame(frame):
            return [], {"inference_ms": 0.0, "processing_ms": 0.0, "fps": 0.0}

        detections, inference_ms = self.run_inference(frame)
        processing_ms = (time.perf_counter() - start) * 1000.0
        fps = calculate_fps(processing_ms / 1000.0)

        stats = {
            "inference_ms": inference_ms,
            "processing_ms": processing_ms,
            "fps": fps,
        }
        return detections, stats

    def process_and_visualize(self, frame: np.ndarray) -> np.ndarray:
        """Run inference + visualization pipeline and return display-ready frame."""
        self.performance_monitor.start_frame_timer()
        
        detections, perf = self.process_frame(frame)
        self.last_detections = detections

        self.frame_count += 1
        self.stats_tracker.update_stats(
            detections=detections,
            fps=perf["fps"],
            inference_ms=perf["inference_ms"],
            processing_ms=perf["processing_ms"],
        )

        # Track performance metrics
        self.performance_monitor.end_frame_timer()
        self.performance_monitor.record_inference_time(perf["inference_ms"])
        self.performance_monitor.record_processing_time(perf["processing_ms"])
        
        # Update object counter
        self.object_counter.update_counts(detections)
        
        # Log detections
        self.detection_logger.log_detections(self.frame_count, detections, frame.shape)

        stats = {
            "fps": perf["fps"],
            "inference_ms": perf["inference_ms"],
            "processing_ms": perf["processing_ms"],
            "total_detections": len(detections),
            "frame_count": self.frame_count,
        }

        # Main visualization
        annotated = self.visualizer.annotate_frame(frame, detections, stats)
        
        # Phase 3 overlay enhancements
        if self.show_threshold_indicator:
            annotated = self.visualizer.draw_threshold_indicator(
                annotated,
                self.threshold_controller.get_threshold(),
            )
        
        if self.video_recorder.is_recording:
            duration = time.time() - self.video_recorder.recording_start_time
            annotated = self.visualizer.draw_recording_indicator(annotated, True, duration)
        
        if self.show_screenshot_notification and self.screenshot_notification_countdown > 0:
            annotated = self.visualizer.draw_screenshot_notification(
                annotated,
                True,
                self.screenshot_notification_countdown,
            )
            self.screenshot_notification_countdown -= 1
        
        # Object count and filter status
        annotated = self.frame_decorator.add_object_counts(annotated, dict(self.object_counter.class_counts))
        enabled_classes = list(self.class_filter.enabled_classes)
        all_classes = list(self.model.names.values())
        annotated = self.frame_decorator.add_active_filters_panel(annotated, enabled_classes, all_classes)
        
        # Demo mode enhancements
        if self.demo_mode.enabled:
            annotated = self.demo_mode.apply_demo_enhancements(annotated, detections)
        
        # Performance panel
        if self.show_performance:
            perf_metrics = self.stats_tracker.get_performance_metrics()
            perf_text = f"FPS: {perf_metrics['current_fps']:.1f} | Inf: {perf_metrics['current_inference_ms']:.1f}ms"
            suggestions = self.stats_tracker.get_optimization_suggestions()
            cv2.putText(
                annotated,
                perf_text,
                (10, annotated.shape[0] - 50),
                self.config.ui.text_font,
                0.6,
                (255, 255, 0),
                1,
                cv2.LINE_AA,
            )
            if suggestions:
                for idx, suggestion in enumerate(suggestions[:2]):  # Show top 2 suggestions
                    cv2.putText(
                        annotated,
                        suggestion,
                        (10, annotated.shape[0] - 30 + (idx * 20)),
                        self.config.ui.text_font,
                        0.5,
                        (255, 165, 0),
                        1,
                        cv2.LINE_AA,
                    )
        
        # Help text
        if self.show_help:
            self._draw_help_overlay(annotated)
        
        annotated = self.frame_decorator.format_all_info(annotated, stats)
        self.last_frame = annotated

        return annotated

    def _draw_help_overlay(self, frame: np.ndarray) -> None:
        """Draw help text overlay on frame."""
        help_text = [
            "KEYBOARD CONTROLS",
            "q/ESC=Quit  h=Help  SPACE=Pause",
            "+=Threshold+ -=Threshold-  r=Record  s=Screenshot",
            "d=Demo  p=Performance  f=Filter  l=Log",
        ]
        
        y_pos = 100
        for text in help_text:
            cv2.putText(
                frame,
                text,
                (20, y_pos),
                self.config.ui.text_font,
                0.5,
                (255, 255, 0),
                1,
                cv2.LINE_AA,
            )
            y_pos += 25

    def display_frame(self, frame: np.ndarray) -> None:
        """Display frame in OpenCV window."""
        cv2.imshow(self.window_name, frame)

    def save_frame(self, frame: np.ndarray) -> Optional[Path]:
        """Save screenshot when enabled and requested.

        Returns:
            Path to saved screenshot, if any.
        """
        if not self.config.output.save_screenshots:
            return None

        try:
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            path = Path(self.output_paths["screenshots"]) / f"frame_{timestamp}.jpg"
            cv2.imwrite(str(path), frame)
            self.logger.info("Screenshot saved: %s", path)
            return path
        except Exception as exc:
            self.logger.error("Screenshot save failed: %s", exc)
            return None

    def _handle_key_input(self) -> bool:
        """Process keyboard events.

        Returns:
            True to continue loop, False to stop.
        """
        key = cv2.waitKey(1) & 0xFF
        
        # General controls
        if key in (ord("q"), 27):  # q or ESC
            self.logger.info("Shutdown requested by user.")
            return False

        if key == ord("h"):  # Help
            self.show_help = True
            self.logger.info("Help requested. Displaying control reference...")
        
        if key == ord(" "):  # Space - pause/resume
            self.paused = not self.paused
            status = "PAUSED" if self.paused else "RESUMED"
            self.logger.info("Detection %s", status)
            return True

        if self.paused:
            return True

        # Threshold control (+/- keys)
        if key in (ord("+"), ord("=")):
            self.threshold_controller.increase_threshold()
            self.logger.info(
                "Confidence threshold increased to %.2f",
                self.threshold_controller.get_threshold(),
            )
        elif key in (ord("-"), ord("_")):
            self.threshold_controller.decrease_threshold()
            self.logger.info(
                "Confidence threshold decreased to %.2f",
                self.threshold_controller.get_threshold(),
            )
        elif key == ord("c"):
            self.threshold_controller.reset_to_default()
            self.logger.info(
                "Confidence threshold reset to %.2f",
                self.threshold_controller.get_threshold(),
            )

        # Screenshot (s)
        if key == ord("s"):
            if self.last_frame is not None:
                self.video_recorder.save_screenshot(self.last_frame, str(self.output_paths["screenshots"]))
                self.show_screenshot_notification = True
                self.screenshot_notification_countdown = 15
                self.logger.info("Screenshot captured")

        # Recording toggle (r)
        if key == ord("r"):
            if self.video_recorder.is_recording:
                self.video_recorder.stop_recording()
                self.logger.info("Video recording stopped")
            else:
                self.video_recorder.start_recording(
                    output_path=str(Path(self.output_paths["videos"]) / f"recorded_{time.strftime('%Y%m%d_%H%M%S')}.mp4"),
                    fps=self.config.output.video_fps,
                    frame_width=self.config.webcam.frame_width,
                    frame_height=self.config.webcam.frame_height,
                    codec=self.config.output.video_codec,
                )
                self.logger.info("Video recording started")

        # Demo mode toggle (d)
        if key == ord("d"):
            if self.demo_mode.enabled:
                self.demo_mode.disable()
                self.logger.info("Demo mode disabled")
            else:
                self.demo_mode.enable()
                self.logger.info("Demo mode enabled")

        # Performance panel toggle (p)
        if key == ord("p"):
            self.show_performance = not self.show_performance
            status = "enabled" if self.show_performance else "disabled"
            self.logger.info("Performance panel %s", status)

        # Threshold indicator toggle (t)
        if key == ord("t"):
            self.show_threshold_indicator = not self.show_threshold_indicator
            status = "enabled" if self.show_threshold_indicator else "disabled"
            self.logger.info("Threshold indicator %s", status)

        # Class filter menu (f)
        if key == ord("f"):
            self._show_class_filter_menu()

        # Detection log summary (l)
        if key == ord("l"):
            summary = self.detection_logger.get_log_summary()
            self.logger.info("Detection log summary: %s", summary)

        # Export detections (CTRL+E) - NOTE: This may not work perfectly with cv2.waitKey
        if key == 5:  # CTRL+E hex code
            stamp = time.strftime("%Y%m%d_%H%M%S")
            self.detection_logger.export_csv(
                str(Path(self.output_paths["detections"]) / f"detections_{stamp}.csv")
            )
            self.detection_logger.export_json(
                str(Path(self.output_paths["detections"]) / f"detections_{stamp}.json")
            )
            self.logger.info("Detection logs exported")

        return True

    def _show_class_filter_menu(self) -> None:
        """Display class filter menu and handle user selection."""
        print("\n" + "=" * 60)
        print("CLASS FILTER MENU")
        print("=" * 60)
        available_classes = list(self.model.names.values())
        for i, cls_name in enumerate(available_classes, 1):
            enabled = "✓" if cls_name in self.class_filter.enabled_classes else "✗"
            print(f"{i:2d}. [{enabled}] {cls_name}")
        print("\nPress class first letter to toggle, ESC to return, 'a' for all, 'n' for none")
        print("=" * 60 + "\n")

    def start(self) -> None:
        """Start main real-time detection loop."""
        self.logger.info("Starting detection loop...")
        self.is_running = True

        while self.is_running:
            ok, frame = self.camera_handler.read_frame()
            if not ok or frame is None:
                self.logger.warning("Failed to read frame. Retrying...")
                continue

            try:
                if not self.paused:
                    output = self.process_and_visualize(frame)
                    
                    # Write to video recorder if enabled
                    if self.video_recorder.is_recording:
                        self.video_recorder.write_frame(output)
                else:
                    # Display last frame when paused
                    output = self.last_frame if self.last_frame is not None else frame

                self.display_frame(output)

                if not self._handle_key_input():
                    break
            except RuntimeError as exc:
                self.logger.error("Runtime processing error: %s", exc)
            except Exception as exc:
                self.logger.exception("Unexpected loop error: %s", exc)

        self.shutdown()

    def run(self) -> None:
        """Alias for start() for a cleaner external API."""
        self.start()

    def shutdown(self) -> None:
        """Cleanly release all resources."""
        self.logger.info("Shutting down detector...")
        self.is_running = False

        # Stop recording if active
        if self.video_recorder.is_recording:
            self.video_recorder.stop_recording()

        if self.video_writer is not None:
            self.video_writer.release()
            self.video_writer = None

        if self.camera_handler is not None:
            self.camera_handler.release()

        # Export detection logs
        stamp = time.strftime("%Y%m%d_%H%M%S")
        self.detection_logger.export_csv(
            str(Path(self.output_paths["detections"]) / f"detections_{stamp}.csv")
        )
        self.detection_logger.export_json(
            str(Path(self.output_paths["detections"]) / f"detections_{stamp}.json")
        )
        self.logger.info("Detection logs exported")

        # Generate and save assignment report
        avg_stats = self.stats_tracker.get_average_stats()
        report_path = str(Path(self.output_paths["reports"]) / f"assignment_report_{stamp}.md")
        stats_payload = {
            "total_frames": int(avg_stats.get("total_frames", 0)),
            "total_detections": int(avg_stats.get("total_detections", 0)),
            "avg_fps": float(avg_stats.get("avg_fps", 0.0)),
            "avg_inference_ms": float(avg_stats.get("avg_inference_ms", 0.0)),
            "class_detections": dict(self.object_counter.class_counts),
            "performance_notes": self.stats_tracker.get_optimization_suggestions(),
        }

        self.report_generator.generate_full_report(
            output_path=report_path,
            stats=stats_payload,
            detections_log=self.detection_logger.detections_log,
        )
        self.logger.info("Assignment report generated")

        self.release_resources()

        self.logger.info(
            "Session summary: frames=%d avg_fps=%.2f avg_inf=%s total_detections=%d",
            int(avg_stats.get("total_frames", 0)),
            float(avg_stats.get("avg_fps", 0.0)),
            format_inference_time(float(avg_stats.get("avg_inference_ms", 0.0))),
            int(avg_stats.get("total_detections", 0)),
        )
