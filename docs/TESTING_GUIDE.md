# Testing Guide

## Test Layout
All automated tests are under tests:
- test_utils.py
- test_config.py
- test_detector.py
- test_visualization.py
- test_integration.py
- test_edge_cases.py
- test_cross_platform.py

## Run Individual Suites
- python -m unittest tests.test_utils -v
- python -m unittest tests.test_config -v
- python -m unittest tests.test_detector -v
- python -m unittest tests.test_visualization -v
- python -m unittest tests.test_integration -v
- python -m unittest tests.test_edge_cases -v
- python -m unittest tests.test_cross_platform -v

## Run Full Suite
- python test_runner.py

## Test Philosophy
- Unit tests validate utility and component behavior.
- Integration tests validate end-to-end interactions.
- Edge-case tests validate unusual inputs and robustness.
- Cross-platform tests validate portability assumptions.

## Performance and Profiling
- python optimization/profile_performance.py
- python code_quality_checker.py

## Notes
- Some tests use mocks to avoid mandatory hardware dependencies.
- Tests are designed to be non-blocking and CI-friendly.
- Keep runtime under five minutes for full suite on typical hardware.
