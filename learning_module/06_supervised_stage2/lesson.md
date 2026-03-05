# Lesson 6: Supervised Stage 2 Classifier

## Learning Objectives

By the end of this lesson, you will:

- Train a Random Forest classifier on full labeled traffic
- Use Stage 2 to reduce Stage 1 false positives
- Produce class probabilities and feature importance
- Save and reload classifier artifacts

## Why Stage 2 Exists

Stage 1 is optimized for anomaly recall. Stage 2 improves precision by re-checking suspicious samples and assigning attack classes.

## Build Tasks

1. Implement `SupervisedClassifier`.
2. Train with class imbalance handling (`class_weight='balanced'`).
3. Add optional hyperparameter optimization.
4. Return detailed prediction output:
   - `class_label`
   - `confidence`
   - `probabilities`
   - `top_features`
5. Implement `save()` and `load()`.

## Exercise

### Exercise 1 (Easy)

Train on synthetic imbalanced labels and inspect class counts.

### Exercise 2 (Medium)

Add `get_feature_importance(n=20)` and print top features.

### Exercise 3 (Hard)

Run with and without hyperparameter optimization, compare macro-F1.

## Verification

```bash
python -m pytest tests/test_inference.py -v
python train_cascaded.py --config config/default_config.yaml
```

## Solution Reference

- `src/supervised_classifier.py`
- `train_cascaded.py`
- `tests/test_inference.py`
- Rebuild map: `learning_module/PROJECT_REBUILD_MODULES.md`

## Self-Check

- I can explain how Stage 2 reduces false positives.
- I can map model probabilities to class labels correctly.
- I can inspect and communicate top contributing features.
