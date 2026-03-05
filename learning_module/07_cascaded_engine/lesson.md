# Lesson 7: Cascaded Inference Engine

## Learning Objectives

By the end of this lesson, you will:

- Implement two-stage decision flow in `CascadedDetector`
- Handle Stage 1 fast-path benign decisions
- Route suspicious samples to Stage 2 classification
- Track latency and decision statistics

## Decision Flow

1. Compute anomaly score (Stage 1).
2. If score < threshold -> BENIGN (fast path).
3. Else pass to Stage 2 classifier.
4. If Stage 2 says BENIGN -> correct false positive.
5. Else -> ATTACK with attack type metadata.

## Build Tasks

1. Implement `load_stage1()` and `load_stage2()`.
2. Implement `predict_single()` and `predict_batch()`.
3. Track stats:
   - total samples
   - Stage 1 benign/suspicious counts
   - Stage 2 benign/attack counts
   - per-stage latencies
4. Return rich result dictionaries.

## Exercise

### Exercise 1 (Medium)

Create unit tests for all branch paths in `predict_single()`.

### Exercise 2 (Hard)

Simulate mixed input and verify stat counters exactly match outcomes.

### Exercise 3 (Hard)

Measure mean Stage 1 vs Stage 2 latency and explain bottlenecks.

## Verification

```bash
python quick_cascaded_demo.py
python -m pytest tests/test_zero_day_detection.py -v
```

## Solution Reference

- `src/cascaded_detector.py`
- `quick_cascaded_demo.py`
- `train_cascaded.py`
- `train_cascaded_full.py`
- Rebuild map: `learning_module/PROJECT_REBUILD_MODULES.md`

## Self-Check

- I can explain when a prediction is finalized in Stage 1 vs Stage 2.
- I can interpret latency metrics for real-time deployment concerns.
- I can verify stage-wise counters after a batch run.
