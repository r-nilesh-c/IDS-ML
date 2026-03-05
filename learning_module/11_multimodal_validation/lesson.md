# Lesson 11: Multimodal Validation (Network + Medical)

## Learning Objectives

By the end of this lesson, you will:

- Add medical plausibility checks to network IDS output
- Detect cross-modal inconsistencies
- Compute combined risk for final alert escalation
- Evaluate network-only vs multimodal performance

## Build Tasks

1. Implement signal alias detection (HR, SpO2, temperature, BP, respiration).
2. Define per-signal rules:
   - value range checks
   - abrupt delta checks
3. Compute medical risk score and combine with network score.
4. Raise alerts on:
   - network attack
   - combined risk threshold
   - cross-modal mismatch

## Exercise

### Exercise 1 (Medium)

Add aliases for your custom medical column names.

### Exercise 2 (Hard)

Tune weights/thresholds and measure recall-FPR tradeoff.

### Exercise 3 (Hard)

Run a side-by-side comparison and produce a markdown summary.

## Verification

```bash
python -m pytest tests/test_multimodal_validation.py -v
python train_cascaded_multimodal.py --train-data data/multimodal/multimodal_train.csv --holdout-data data/multimodal/multimodal_holdout.csv
python evaluate_multimodal_comparison.py --data data/multimodal/multimodal_holdout.csv --model-dir models/multimodal
```

## Solution Reference

- `src/multimodal_validation.py`
- `train_cascaded_multimodal.py`
- `evaluate_multimodal_comparison.py`
- `tests/test_multimodal_validation.py`
- Rebuild map: `learning_module/PROJECT_REBUILD_MODULES.md`

## Self-Check

- I can explain why multimodal validation changes alert decisions.
- I can justify selected weights and thresholds.
- I can quantify delta between network-only and multimodal metrics.
