"""Viva preparation guide and Q&A."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List


class VivaGuide:
    """Comprehensive viva preparation guide."""

    def __init__(self, logger: Any = None) -> None:
        """Initialize viva guide.

        Args:
            logger: Logger instance.
        """
        self.logger = logger
        self.questions_db = self._load_questions()

    def _load_questions(self) -> Dict[str, List[Dict[str, Any]]]:
        """Load questions from JSON file if available."""
        try:
            qfile = Path(__file__).parent.parent / "data" / "viva_questions.json"
            if qfile.exists():
                with open(qfile, "r", encoding="utf-8") as f:
                    return json.load(f)
        except Exception as exc:
            if self.logger:
                self.logger.warning(f"Could not load viva questions: {exc}")
        return self._default_questions()

    def _default_questions(self) -> Dict[str, List[Dict[str, Any]]]:
        """Return default Q&A database."""
        return {
            "deep_learning": [
                {
                    "question": "Explain the OODA loop (Observe-Orient-Decide-Act) in your YOLOv8 project",
                    "answer": (
                        "OODA stands for Observe-Orient-Decide-Act. In our project:\n"
                        "- OBSERVE: Webcam captures real-time frames\n"
                        "- ORIENT: YOLOv8 CNN analyzes features through convolutional layers\n"
                        "- DECIDE: Model predicts objects above confidence threshold\n"
                        "- ACT: System visualizes detections and logs results"
                    ),
                    "difficulty": "easy",
                },
                {
                    "question": "What is transfer learning and why use it?",
                    "answer": (
                        "Transfer Learning uses a pretrained model (learned on massive datasets like COCO) "
                        "instead of training from scratch. Benefits:\n"
                        "- 1000x faster deployment (seconds vs weeks)\n"
                        "- Better accuracy (billions of examples already learned)\n"
                        "- Works on consumer hardware\n"
                        "- Minimal labeled data needed\n"
                        "YOLOv8n is pretrained on 80 object classes and immediately usable."
                    ),
                    "difficulty": "easy",
                },
                {
                    "question": "Explain how CNNs extract features",
                    "answer": (
                        "Convolutional Layers progressively extract features:\n"
                        "LAYER 1: Detects edges (vertical, horizontal, diagonal)\n"
                        "LAYER 2: Combines edges into textures and patterns\n"
                        "LAYER 3: Recognizes object parts (wheels, eyes, faces)\n"
                        "LAYER 4: Identifies whole objects (car, person, dog)\n"
                        "Each layer's output feeds into the next, building abstract understanding."
                    ),
                    "difficulty": "medium",
                },
            ],
            "yolo": [
                {
                    "question": "What does YOLO stand for and how does it differ from Faster R-CNN?",
                    "answer": (
                        "YOLO = 'You Only Look Once'\n\n"
                        "YOLO: Splits image into grid, predicts all bboxes+classes in ONE pass (fast)\n"
                        "Faster R-CNN: Proposes regions first, then classifies each (slower but more accurate)\n\n"
                        "Trade-off:\n"
                        "YOLO: ~30 FPS, 90% mAP - Great for real-time\n"
                        "R-CNN: ~5 FPS, 95% mAP - Better for accuracy-critical tasks"
                    ),
                    "difficulty": "medium",
                },
                {
                    "question": "Explain Non-Maximum Suppression (NMS)",
                    "answer": (
                        "When YOLO detects an object, multiple overlapping bboxes might be predicted.\n"
                        "NMS removes duplicate detections:\n"
                        "1. Sort predictions by confidence\n"
                        "2. Keep highest confidence bbox\n"
                        "3. Remove all overlapping boxes (IoU > threshold)\n"
                        "4. Repeat until done\n\n"
                        "Result: One clean bbox per object, no duplicates."
                    ),
                    "difficulty": "hard",
                },
            ],
            "implementation": [
                {
                    "question": "Describe your system architecture",
                    "answer": (
                        "Three-package modular design:\n\n"
                        "1. DETECTOR (capture + inference):\n"
                        "   - base_detector: Model loading, device management\n"
                        "   - camera_handler: Robust frame capture with reconnection\n"
                        "   - main_detector: Orchestrates all components\n"
                        "   - inference_pipeline: YOLO prediction + parsing\n\n"
                        "2. VISUALIZATION (display):\n"
                        "   - visualizer: Draws boxes and labels\n"
                        "   - stats_tracker: Performance metrics\n"
                        "   - frame_decorator: Overlays (FPS, timestamp)\n\n"
                        "3. ENHANCEMENT (Phase 3 features):\n"
                        "   - counter: Object counting\n"
                        "   - threshold_controller: Dynamic confidence\n"
                        "   - video_recorder: MP4 recording\n"
                        "   - detection_logger: CSV/JSON export\n"
                        "   - class_filter: Class-based filtering\n"
                        "   - performance_monitor: FPS/memory tracking"
                    ),
                    "difficulty": "medium",
                },
                {
                    "question": "How do you handle camera disconnections?",
                    "answer": (
                        "CameraHandler implements robust reconnection:\n"
                        "1. Detect dropped frames (cv2.read() returns False)\n"
                        "2. Trigger reconnection attempt (3 retries)\n"
                        "3. For each retry:\n"
                        "   - Release current camera\n"
                        "   - Re-open VideoCapture(camera_id)\n"
                        "   - Reset resolution and FPS\n"
                        "4. If reconnect fails, log error and skip frame\n"
                        "5. Continue processing next frame\n\n"
                        "User doesn't notice gaps, system stays resilient."
                    ),
                    "difficulty": "hard",
                },
            ],
            "performance": [
                {
                    "question": "How do you achieve real-time performance (30+ FPS)?",
                    "answer": (
                        "Multiple optimization strategies:\n\n"
                        "1. MODEL SELECTION: yolov8n (nano) is 10x faster than yolov8x\n"
                        "2. HARDWARE: Use GPU (CUDA) when available\n"
                        "3. RESOLUTION: 640x480 instead of 1080p\n"
                        "4. BATCHING: Process frames individually (no batching overhead)\n"
                        "5. EFFICIENT CODE:\n"
                        "   - NumPy vectorization (no Python loops)\n"
                        "   - OpenCV optimized operations\n"
                        "   - Minimal memory copies\n"
                        "6. OPTIONAL FEATURES: Can disable fancy overlays if needed"
                    ),
                    "difficulty": "medium",
                },
            ],
        }

    def get_all_topics(self) -> List[str]:
        """Return list of all question topics."""
        return list(self.questions_db.keys())

    def get_questions_by_topic(self, topic: str) -> List[Dict[str, Any]]:
        """Get all Q&A for a specific topic."""
        return self.questions_db.get(topic, [])

    def get_question(self, topic: str, index: int = 0) -> Dict[str, Any] | None:
        """Get specific question by topic and index."""
        questions = self.get_questions_by_topic(topic)
        if 0 <= index < len(questions):
            return questions[index]
        return None

    def print_viva_guide(self) -> str:
        """Generate formatted viva guide text."""
        lines = ["=" * 80, "YOLO REAL-TIME DETECTION - VIVA PREPARATION GUIDE", "=" * 80, ""]

        for topic in self.get_all_topics():
            lines.append(f"\n{'═' * 80}")
            lines.append(f"TOPIC: {topic.upper()}")
            lines.append(f"{'═' * 80}\n")

            questions = self.get_questions_by_topic(topic)
            for i, qa in enumerate(questions, 1):
                lines.append(f"Q{i}. {qa['question']}")
                lines.append(f"\nA: {qa['answer']}")
                lines.append(f"\nDifficulty: {qa.get('difficulty', 'unknown')}")
                lines.append("-" * 80 + "\n")

        return "\n".join(lines)

    def export_to_file(self, filepath: str) -> bool:
        """Export guide to text file."""
        try:
            Path(filepath).parent.mkdir(parents=True, exist_ok=True)
            with open(filepath, "w", encoding="utf-8") as f:
                f.write(self.print_viva_guide())
            if self.logger:
                self.logger.info(f"Viva guide exported: {filepath}")
            return True
        except Exception as exc:
            if self.logger:
                self.logger.exception(f"Export failed: {exc}")
            return False
