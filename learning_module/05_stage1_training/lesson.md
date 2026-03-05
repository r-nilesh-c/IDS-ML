# Lesson 5: Stage 1 Training Pipeline (AE + IF + Fusion)

## Learning Objectives

By the end of this lesson, you will:

- Build an end-to-end Stage 1 training script
- Connect preprocessing, autoencoder, isolation forest, and fusion
- Save reusable model artifacts for inference
- Ensure reproducibility with fixed seeds

## What You Build

A script equivalent to `train.py` that:

1. Loads YAML config
2. Preprocesses data
3. Trains AE + IF
4. Fits fusion threshold on benign validation
5. Saves model artifacts

## Build Tasks

1. Implement config loading + logger setup.
2. Add deterministic seed setup (`random`, `numpy`, and TensorFlow).
3. Wire preprocessing outputs into AE and IF training.
4. Fit fusion threshold using validation benign scores.
5. Save artifacts:
   - `autoencoder_best.keras`
   - `isolation_forest.pkl`
   - `fusion_params.pkl`
   - `scaler.pkl`

## Exercise

### Exercise 1 (Medium)

Add a fallback synthetic-data path when dataset files are missing.

### Exercise 2 (Hard)

Add training summary output (epoch count, final losses, threshold).

### Exercise 3 (Hard)

Make save/load metadata explicit to prevent feature mismatch during inference.

## Verification

```bash
python train.py --config config/default_config.yaml
python -m pytest tests/test_training_pipeline.py -v
```

## Solution Reference

- `train.py`
- `src/preprocessing.py`
- `src/autoencoder.py`
- `src/isolation_forest.py`
- `src/fusion.py`
- `tests/test_training_pipeline.py`
- Rebuild map: `learning_module/PROJECT_REBUILD_MODULES.md`

## Self-Check

- I can rerun training and get consistent behavior with the same seed.
- I understand which artifact is used by each inference stage.
- I know where threshold and normalization stats are persisted.
