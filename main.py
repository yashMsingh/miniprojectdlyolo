"""Main entry point for YOLOv8 real-time object detection (Phase 1-3 Complete)."""

from __future__ import annotations

from config import AppConfig
from detector.main_detector import MainDetector
from utils import setup_logger


def main() -> None:
    """Initialize detector and run the real-time processing loop."""
    config = AppConfig.default()
    logger = setup_logger(
        name="main",
        log_dir=config.logging.log_dir,
        log_level=config.logging.log_level,
        enable_logging=config.logging.enable_logging,
    )

    detector = MainDetector(config=config)

    try:
        logger.info("=" * 70)
        logger.info("YOLOv8 REAL-TIME OBJECT DETECTION (Phase 1-3 Complete)")
        logger.info("=" * 70)
        logger.info("Press 'h' during execution for keyboard controls")
        logger.info("All detections logged to outputs/detections/")
        logger.info("Videos saved to outputs/videos/")
        logger.info("Screenshots saved to outputs/screenshots/")
        logger.info("Reports saved to outputs/reports/")
        logger.info("Session starts now...")
        logger.info("=" * 70)
        detector.start()
    except KeyboardInterrupt:
        logger.info("KeyboardInterrupt received. Shutting down gracefully...")
    except Exception as exc:
        logger.exception("Fatal error in main loop: %s", exc)
    finally:
        logger.info("Cleanup and resource release in progress...")
        detector.shutdown()
        logger.info("=" * 70)
        logger.info("YOLOv8 Detection session completed successfully")
        logger.info("=" * 70)


if __name__ == "__main__":
    main()
