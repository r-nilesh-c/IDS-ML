# Lesson 9: Live CSV Window Monitoring

## Learning Objectives

By the end of this lesson, you will:

- Implement real-time folder/file monitoring for IDS inference
- Process rolling CSV windows continuously
- Log anomaly events in JSONL format
- Optionally bootstrap scaler from training data

## Build Tasks

1. Implement watcher mode (`--watch-dir`) and single-file mode (`--input-file`).
2. Load cascaded models and selected-feature metadata.
3. Apply preprocessing + scaling safely to incoming windows.
4. Emit anomaly logs with timestamp and prediction metadata.
5. Handle malformed files gracefully without crashing monitor loop.

## Exercise

### Exercise 1 (Medium)

Add a `processed/` folder move-after-read workflow.

### Exercise 2 (Medium)

Add file deduping by filename hash or seen set.

### Exercise 3 (Hard)

Integrate optional multimodal post-validation in live mode.

## Verification

```bash
python live_monitor_cascaded.py --watch-dir data/live --poll-seconds 5
```

## Solution Reference

- `live_monitor_cascaded.py`
- `logs/live_anomalies.jsonl`
- `src/cascaded_detector.py`
- `src/multimodal_validation.py`
- Rebuild map: `learning_module/PROJECT_REBUILD_MODULES.md`

## Self-Check

- I can run both single-file and watch-directory modes.
- I can trace one incoming file from read -> predict -> log.
- I can recover cleanly from a bad input file.
