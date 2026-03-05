# Lesson 8: Batch Inference + Evaluation Reporting

## Learning Objectives

By the end of this lesson, you will:

- Build batch inference that loads saved artifacts safely
- Run predictions on unseen data windows
- Generate metrics reports for healthcare deployment criteria
- Log outputs for reproducible validation

## Build Tasks

1. Implement artifact loading checks:
   - Autoencoder model
   - Isolation Forest model
   - Fusion params
   - Scaler / selected feature metadata
2. Build batch inference pipeline that outputs:
   - anomaly score
   - binary prediction
   - latency summary
3. Implement evaluation report generation:
   - accuracy, precision, recall, F1
   - false positive rate
   - ROC-AUC
   - confusion matrix + plots

## Exercise

### Exercise 1 (Medium)

Add explicit feature-dimension mismatch error messages.

### Exercise 2 (Medium)

Save both JSON and human-readable text reports.

### Exercise 3 (Hard)

Compare performance on known vs temporally unseen attack windows.

## Verification

```bash
python inference.py --input <your_csv> --output reports/inference_results.json
python evaluate.py --test-data <your_test_csv>
python -m pytest tests/test_alert_system.py -v
```

## Solution Reference

- `inference.py`
- `evaluate.py`
- `src/alert_system.py`
- `tests/test_alert_system.py`
- Rebuild map: `learning_module/PROJECT_REBUILD_MODULES.md`

## Self-Check

- I can explain every artifact loaded during inference.
- I can compute and interpret false positive rate for deployment readiness.
- I can reproduce evaluation results from saved outputs.
