"""Interactive viva practice simulator."""

import json
import random
import sys
from pathlib import Path

from enhancement.viva_guide import VivaGuide


def print_header(text: str, width: int = 80) -> None:
    """Print formatted header."""
    print("\n" + "═" * width)
    print(f" {text.center(width - 2)} ")
    print("═" * width + "\n")


def print_question(question: str, number: int = 1) -> None:
    """Print question with formatting."""
    print(f"\n📋 QUESTION {number}:")
    print(f"{question}\n")


def print_answer(answer: str) -> None:
    """Print answer with formatting."""
    print("📝 ANSWER:")
    print(answer)
    print()


def get_difficulty_stars(difficulty: str) -> str:
    """Return star rating for difficulty."""
    star_map = {
        "easy": "⭐",
        "medium": "⭐⭐",
        "hard": "⭐⭐⭐",
    }
    return star_map.get(difficulty, "?")


def main() -> None:
    """Run interactive viva practice."""
    print_header("YOLO VIVA PRACTICE SESSION")

    guide = VivaGuide()
    topics = guide.get_all_topics()

    if not topics:
        print("❌ No viva questions loaded. Please check data/viva_questions.json")
        return

    print("Available topics:")
    for i, topic in enumerate(topics, 1):
        questions = guide.get_questions_by_topic(topic)
        print(f"  {i}. {topic.upper()} ({len(questions)} questions)")

    while True:
        print("\n" + "─" * 80)
        choice = input(
            "\nEnter topic number ('list' to show all, 'random' for random quiz, 'exit' to quit): "
        ).strip().lower()

        if choice == "exit":
            print("\n✅ Thanks for practicing! Good luck with your viva! 🚀\n")
            break

        if choice == "list":
            print("\nAll Questions:\n")
            for i, topic in enumerate(topics, 1):
                questions = guide.get_questions_by_topic(topic)
                print(f"\n{topic.upper()}")
                for j, qa in enumerate(questions, 1):
                    difficulty = get_difficulty_stars(qa.get("difficulty", "?"))
                    print(f"  {j}. [{difficulty}] {qa['question'][:60]}...")
            continue

        if choice == "random":
            num = input("How many random questions? (default 5): ").strip()
            try:
                num_questions = int(num) if num else 5
            except ValueError:
                num_questions = 5

            all_questions = []
            for topic in topics:
                questions = guide.get_questions_by_topic(topic)
                for qa in questions:
                    qa["topic"] = topic
                    all_questions.append(qa)

            selected = random.sample(all_questions, min(num_questions, len(all_questions)))

            print_header(f"RANDOM QUIZ - {len(selected)} QUESTIONS")

            for idx, qa in enumerate(selected, 1):
                print_question(qa["question"], idx)
                print(f"Topic: {qa['topic'].upper()}")
                print(f"Difficulty: {get_difficulty_stars(qa.get('difficulty', '?'))}")

                input("Press Enter to see answer...")
                print_answer(qa["answer"])

                if qa.get("key_points"):
                    print("🔑 KEY POINTS:")
                    for point in qa["key_points"]:
                        print(f"   • {point}")
                    print()

                if idx < len(selected):
                    input("Press Enter for next question...")

            print("\n✅ Quiz complete!\n")
            continue

        try:
            topic_idx = int(choice) - 1
            if not (0 <= topic_idx < len(topics)):
                print("❌ Invalid topic number")
                continue

            topic = topics[topic_idx]
            questions = guide.get_questions_by_topic(topic)

            print_header(f"TOPIC: {topic.upper()} ({len(questions)} questions)")

            for q_idx, qa in enumerate(questions, 1):
                print_question(qa["question"], q_idx)
                print(f"Difficulty: {get_difficulty_stars(qa.get('difficulty', '?'))}\n")

                input("Press Enter to see answer...")
                print_answer(qa["answer"])

                if qa.get("key_points"):
                    print("🔑 KEY POINTS:")
                    for point in qa["key_points"]:
                        print(f"   • {point}")
                    print()

                if q_idx < len(questions):
                    input("Press Enter for next question...")

            print("\n✅ Topic complete!\n")

        except ValueError:
            print("❌ Invalid input. Please enter a number or 'random'")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n👋 Practice session interrupted. Good luck with your viva!\n")
