"""Code quality assessment utility."""

from __future__ import annotations

import ast
import os
from pathlib import Path
from typing import Dict, List


class CodeQualityChecker:
    """Check code quality metrics for project Python files."""

    def __init__(self, project_path: str = ".") -> None:
        self.project_path = project_path
        self.python_files = self._find_python_files()

    def _find_python_files(self) -> List[str]:
        files: List[str] = []
        for root, dirs, filenames in os.walk(self.project_path):
            dirs[:] = [d for d in dirs if d not in ["venv", "__pycache__", ".venv", ".git"]]
            for filename in filenames:
                if filename.endswith(".py"):
                    files.append(os.path.join(root, filename))
        return files

    def check_docstrings(self) -> Dict[str, int]:
        results = {
            "total_functions": 0,
            "documented_functions": 0,
            "total_classes": 0,
            "documented_classes": 0,
        }

        for filepath in self.python_files:
            try:
                source = Path(filepath).read_text(encoding="utf-8", errors="ignore")
                tree = ast.parse(source)
            except Exception:
                continue

            for node in ast.walk(tree):
                if isinstance(node, ast.FunctionDef):
                    results["total_functions"] += 1
                    if ast.get_docstring(node):
                        results["documented_functions"] += 1
                elif isinstance(node, ast.ClassDef):
                    results["total_classes"] += 1
                    if ast.get_docstring(node):
                        results["documented_classes"] += 1

        return results

    def check_type_hints(self) -> Dict[str, int]:
        results = {
            "total_functions": 0,
            "functions_with_hints": 0,
        }

        for filepath in self.python_files:
            try:
                source = Path(filepath).read_text(encoding="utf-8", errors="ignore")
                tree = ast.parse(source)
            except Exception:
                continue

            for node in ast.walk(tree):
                if isinstance(node, ast.FunctionDef):
                    results["total_functions"] += 1
                    has_return = node.returns is not None
                    has_arg_hint = any(arg.annotation is not None for arg in node.args.args)
                    if has_return or has_arg_hint:
                        results["functions_with_hints"] += 1

        return results

    def check_line_length(self, max_length: int = 100) -> List[Dict[str, int | str]]:
        violations: List[Dict[str, int | str]] = []
        for filepath in self.python_files:
            try:
                with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
                    for i, line in enumerate(f, 1):
                        if len(line.rstrip()) > max_length:
                            violations.append({"file": filepath, "line": i, "length": len(line.rstrip())})
            except Exception:
                continue
        return violations

    def generate_report(self) -> None:
        print("\n" + "=" * 70)
        print("CODE QUALITY ASSESSMENT REPORT")
        print("=" * 70)
        print(f"\nPython Files Found: {len(self.python_files)}")

        print("\nDOCSTRING COVERAGE")
        print("-" * 70)
        doc_results = self.check_docstrings()
        total_doc_items = doc_results["total_functions"] + doc_results["total_classes"]
        documented_items = doc_results["documented_functions"] + doc_results["documented_classes"]
        doc_percentage = (documented_items / max(total_doc_items, 1)) * 100.0
        print(f"Functions: {doc_results['documented_functions']}/{doc_results['total_functions']}")
        print(f"Classes: {doc_results['documented_classes']}/{doc_results['total_classes']}")
        print(f"Coverage: {doc_percentage:.1f}%")
        print("Status: GOOD" if doc_percentage > 80 else "Status: NEEDS IMPROVEMENT")

        print("\nTYPE HINT COVERAGE")
        print("-" * 70)
        hint_results = self.check_type_hints()
        hint_percentage = (
            hint_results["functions_with_hints"] / max(hint_results["total_functions"], 1) * 100.0
        )
        print(f"Functions with hints: {hint_results['functions_with_hints']}/{hint_results['total_functions']}")
        print(f"Coverage: {hint_percentage:.1f}%")
        print("Status: GOOD" if hint_percentage > 70 else "Status: NEEDS IMPROVEMENT")

        print("\nLINE LENGTH COMPLIANCE")
        print("-" * 70)
        violations = self.check_line_length()
        print("Max Recommended: 100 characters")
        print(f"Violations: {len(violations)}")
        print("Status: GOOD" if len(violations) < 10 else "Status: NEEDS IMPROVEMENT")

        for v in violations[:5]:
            print(f"- {v['file']}:{v['line']} ({v['length']} chars)")

        print("\n" + "=" * 70)


if __name__ == "__main__":
    checker = CodeQualityChecker()
    checker.generate_report()
