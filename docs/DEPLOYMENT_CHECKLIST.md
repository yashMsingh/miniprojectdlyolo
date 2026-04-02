# Deployment Readiness Checklist

## Environment
- [ ] Python 3.9+ installed
- [ ] Virtual environment configured
- [ ] Dependencies installed from requirements.txt
- [ ] Model file available (yolov8n.pt minimum)
- [ ] test_installation.py passes

## Core Functionality
- [ ] Camera feed opens
- [ ] Inference runs on frames
- [ ] Boxes, labels, confidence drawn
- [ ] FPS visible and reasonable
- [ ] Quit path works (q or ESC)

## Enhanced Features
- [ ] Object counting updates
- [ ] Threshold keys work (+, -, c)
- [ ] Screenshot key works (s)
- [ ] Recording toggle works (r)
- [ ] Detection logs export on shutdown
- [ ] Class filtering menu opens (f)
- [ ] Performance panel toggles (p)
- [ ] Demo mode toggles (d)
- [ ] Report generated at shutdown

## Testing and Quality
- [ ] Unit tests pass
- [ ] Integration tests pass
- [ ] Edge case tests pass
- [ ] Cross-platform tests pass
- [ ] code_quality_checker.py report reviewed

## Performance Targets
- [ ] CPU average FPS >= 25 (target)
- [ ] GPU average FPS >= 30 (target)
- [ ] Inference latency stable
- [ ] No obvious memory leak over long run

## Documentation
- [ ] README is current
- [ ] USER_GUIDE.md complete
- [ ] TESTING_GUIDE.md complete
- [ ] controls_reference.txt available

## Submission Steps
1. Run python test_runner.py
2. Run python optimization/profile_performance.py
3. Run python main.py for demo capture
4. Verify outputs folder artifacts
5. Archive project for submission

## Sign-Off
- Reviewer:
- Date:
- Status:
