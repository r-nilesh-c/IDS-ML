# Lesson 4: Score Fusion + Dynamic Thresholding

## Learning Objectives

By the end of this lesson, you will:

- Normalize heterogeneous anomaly scores to a common scale
- Combine Autoencoder and Isolation Forest scores with configurable weights
- Fit a percentile threshold on benign validation samples
- Generate final anomaly predictions from fused scores

## Why Fusion Matters

Autoencoder and Isolation Forest produce different score distributions. Fusion improves robustness by combining:

- Representation-based anomaly signal (autoencoder)
- Isolation-based outlier signal (forest)

## Build Tasks

1. Implement `FusionModule` in your own project.
2. Validate weights sum to 1.0.
3. Store benign min/max stats for both score streams.
4. Normalize incoming scores with min-max scaling and clipping.
5. Compute weighted score and fit threshold from benign validation set.

## Core APIs to Implement

- `fit_threshold(recon_errors_benign, iso_scores_benign)`
- `normalize_scores(recon_errors, iso_scores)`
- `compute_combined_score(recon_errors, iso_scores)`
- `predict(recon_errors, iso_scores)`

## Hands-On Exercise

### Exercise 1 (Easy)

Create a function that checks weight validity:

- Input: `weight_autoencoder`, `weight_isolation`
- Output: raise error if sum != 1.0

### Exercise 2 (Medium)

Use synthetic benign scores to compute a threshold at the 95th percentile.

### Exercise 3 (Medium)

Normalize two score arrays and verify all outputs are within [0, 1].

### Exercise 4 (Hard)

Run a fusion ablation:

- AE only
- IF only
- 70/30 weighted fusion
  Compare FPR and recall.

## Verification

Run:

```bash
python -m pytest tests/test_fusion.py -v
```

Optional quick check:

```bash
python quick_cascaded_demo.py
```

## Solution Reference

- `src/fusion.py`
- `tests/test_fusion.py`
- Rebuild map: `learning_module/PROJECT_REBUILD_MODULES.md`

## Self-Check

- I can explain why percentile thresholds are fitted on benign validation data.
- I can describe the effect of changing AE/IF weights.
- I can debug score-range mismatches before prediction.
