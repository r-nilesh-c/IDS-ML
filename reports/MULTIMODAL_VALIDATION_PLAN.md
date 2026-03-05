# Multimodal Validation Plan (Full Training)

## Scope
- Goal: validate a healthcare IDS prototype with full multimodal training (network + medical signals).
- Evidence level: academic prototype (controlled benchmark + holdout comparison).

## Dataset Strategy
1. Base network data: CIC-IDS (public).
2. Medical channels added per sample/window: `hr`, `spo2`, `temp`, `sys`, `dia`, `rr`.
3. Scenario generation:
   - `normal`
   - `network_attack`
   - `medical_tamper`
   - `combined_attack`
4. Holdout split: tail-based holdout to simulate temporal generalization.

## Pipeline
1. Build benchmark dataset:
   - `python scripts/build_multimodal_benchmark.py`
2. Train full multimodal cascaded model:
   - `python train_cascaded_multimodal.py`
3. Compare network-only vs multimodal on holdout:
   - `python evaluate_multimodal_comparison.py`

## Outputs
- Dataset files: `data/multimodal/multimodal_{full,train,holdout}.csv`
- Model artifacts: `models/multimodal/*`
- Training summary: `reports/multimodal_training_summary.json`
- Comparison metrics: `reports/multimodal_comparison_metrics.json`
- Comparison report: `reports/MULTIMODAL_COMPARISON_REPORT.md`

## Notes
- This benchmark is public-data-derived + controlled medical tampering.
- Suitable claim: validated multimodal healthcare IDS prototype (not clinical deployment certification).
