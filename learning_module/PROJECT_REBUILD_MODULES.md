# Hands-On Project Rebuild Modules (Build This Project Yourself)

This guide is a step-by-step rebuild path so you can create your own copy of this IDS project by implementing each module yourself.

## How to Use This

For each module:
1. Create your own implementation first.
2. Run the module verification command(s).
3. Compare with the solution reference files listed in that module.
4. Refactor your version until behavior matches.

Recommended workflow:
- Keep your work in a separate folder (for example: `my_hybrid_ids/`).
- Implement each module in order.
- Use this repository only as the reference solution.

---

## Module 0 - Environment + Project Skeleton

### Build
- Create folder layout: `src/`, `config/`, `models/`, `logs/`, `reports/`, `tests/`.
- Install dependencies from `requirements.txt`.
- Create minimal logger and config loader.

### Verification
- `python -m pytest tests/ -q` (will fail initially, but should run)

### Solution Reference
- `requirements.txt`
- `config/default_config.yaml`
- `config/logging_config.yaml`
- `src/utils.py`

---

## Module 1 - Data Preprocessing Pipeline

### Build
- Implement `PreprocessingPipeline` with:
  - robust CSV/parquet loading
  - label-column normalization
  - NaN/inf/duplicate cleanup
  - benign/attack split
  - normalization + train/val/test split

### Verification
- Run lesson exercises from the learning module.
- `python -m pytest tests/test_preprocessing.py -v`

### Solution Reference
- `src/preprocessing.py`
- `learning_module/01_preprocessing/lesson.md`
- `learning_module/solutions/01_preprocessing_solutions.py`

---

## Module 2 - Autoencoder Anomaly Detector

### Build
- Implement `AutoencoderDetector`:
  - model builder
  - train loop with early stopping/checkpointing
  - reconstruction error scoring
  - optional GPU + mixed precision support

### Verification
- `python -m pytest tests/test_autoencoder.py -v`

### Solution Reference
- `src/autoencoder.py`
- `learning_module/02_autoencoder/lesson.md`
- `learning_module/solutions/02_autoencoder_solutions.py`

---

## Module 3 - Isolation Forest Detector

### Build
- Implement `IsolationForestDetector`:
  - config validation
  - benign-only training
  - anomaly score computation
  - prediction helper

### Verification
- `python -m pytest tests/test_isolation_forest.py -v`

### Solution Reference
- `src/isolation_forest.py`
- `learning_module/03_isolation_forest/lesson.md`

---

## Module 4 - Score Fusion + Dynamic Thresholding

### Build
- Implement `FusionModule`:
  - min-max normalization for AE and IF scores
  - weighted score fusion
  - percentile-based threshold fit on benign validation

### Verification
- `python -m pytest tests/test_fusion.py -v`

### Solution Reference
- `src/fusion.py`

---

## Module 5 - Stage 1 Training Script

### Build
- Implement a full training pipeline script that:
  - loads config
  - preprocesses data
  - trains AE + IF
  - fits fusion threshold
  - saves artifacts to `models/`

### Verification
- `python train.py --config config/default_config.yaml`
- `python -m pytest tests/test_training_pipeline.py -v`

### Solution Reference
- `train.py`

---

## Module 6 - Supervised Stage 2 Classifier

### Build
- Implement `SupervisedClassifier`:
  - random forest training on full labeled data
  - optional hyperparameter tuning
  - inference probabilities + feature importance
  - model save/load

### Verification
- `python -m pytest tests/test_inference.py -v`

### Solution Reference
- `src/supervised_classifier.py`

---

## Module 7 - Cascaded Inference Engine (Stage 1 + Stage 2)

### Build
- Implement `CascadedDetector` with logic:
  - Stage 1 anomaly gate
  - Stage 2 benign correction / attack confirmation
  - per-sample latency + stats tracking

### Verification
- `python quick_cascaded_demo.py`
- `python -m pytest tests/test_zero_day_detection.py -v`

### Solution Reference
- `src/cascaded_detector.py`
- `train_cascaded.py`
- `train_cascaded_full.py`

---

## Module 8 - Batch Inference + Evaluation Reports

### Build
- Implement:
  - model loading for inference
  - batch prediction export
  - evaluation metrics and report generation

### Verification
- `python inference.py --input <your_csv> --output reports/inference_results.json`
- `python evaluate.py --test-data <your_test_csv>`

### Solution Reference
- `inference.py`
- `evaluate.py`
- `src/alert_system.py`

---

## Module 9 - Live CSV Window Monitoring

### Build
- Implement folder/file based live monitor:
  - watch directory for new windows
  - run cascaded predictions
  - write JSONL alert stream

### Verification
- `python live_monitor_cascaded.py --watch-dir data/live --poll-seconds 5`

### Solution Reference
- `live_monitor_cascaded.py`
- `logs/live_anomalies.jsonl`

---

## Module 10 - Live Packet Monitoring (Interface Capture)

### Build
- Implement packet capture monitor:
  - sniff packets from interface
  - aggregate flow features per window
  - run cascaded detection on generated flow rows

### Verification
- `python live_packet_monitor.py --interface "Wi-Fi" --window-seconds 5`

### Solution Reference
- `live_packet_monitor.py`
- `logs/live_packet_anomalies.jsonl`

---

## Module 11 - Multimodal Validation (Network + Medical Signals)

### Build
- Implement multimodal rule layer:
  - medical signal range/change checks
  - cross-modal mismatch detection
  - combined risk scoring and final alert decision

### Verification
- `python -m pytest tests/test_multimodal_validation.py -v`
- `python train_cascaded_multimodal.py --train-data data/multimodal/multimodal_train.csv --holdout-data data/multimodal/multimodal_holdout.csv`
- `python evaluate_multimodal_comparison.py --data data/multimodal/multimodal_holdout.csv --model-dir models/multimodal`

### Solution Reference
- `src/multimodal_validation.py`
- `train_cascaded_multimodal.py`
- `evaluate_multimodal_comparison.py`

---

## Module 12 - Hardening, Testing, and Final Submission Pack

### Build
- Add/complete tests for all major modules.
- Validate reproducibility and model artifact consistency.
- Generate final reports and documentation bundle.

### Verification
- `python -m pytest tests/ -v`
- `python quick_cascaded_demo.py`

### Solution Reference
- `tests/`
- `PROJECT_SETUP_AND_USAGE_GUIDE.md`
- `COMPLETE_PROJECT_EXPLANATION.md`
- `CLEANUP_AND_SUBMISSION_SUMMARY.md`

---

## Module Completion Template (Use For Every Module)

Copy this checklist into your notes for each module:

- [ ] I implemented it myself first (no copy/paste).
- [ ] My script/class runs without runtime errors.
- [ ] I ran the listed verification command(s).
- [ ] I compared behavior with the solution reference files.
- [ ] I can explain design decisions in my own words.

---

## Suggested 4-Week Plan

- Week 1: Modules 0-3
- Week 2: Modules 4-7
- Week 3: Modules 8-10
- Week 4: Modules 11-12 + final polish

If you follow this sequence and implement each module yourself before checking solutions, you will end up with your own full project implementation that matches this repository in architecture and behavior.
