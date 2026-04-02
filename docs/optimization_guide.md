# Optimization Guide (Documentation Copy)

This document mirrors optimization/optimization_guide.md for docs-focused navigation.

## Key Actions
- Prefer yolov8n for maximum real-time throughput.
- Lower input resolution when FPS is below target.
- Disable demo overlays and recording for benchmark sessions.
- Use CUDA when available.

## Run Profiling
- python optimization/profile_performance.py

## Evaluate Memory
- Use optimization/memory_profiler.py in long-run sessions.

## Validate Results
- Compare baseline and optimized FPS.
- Keep average FPS above your assignment target.
