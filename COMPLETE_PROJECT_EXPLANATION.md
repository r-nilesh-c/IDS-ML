# Complete Project Explanation

## Abstract

Network intrusion attacks such as DDoS, infiltration, web attacks, and port scans pose serious threats to healthcare networks. These attacks can disrupt critical services and compromise sensitive patient data. Existing intrusion detection systems suffer from high false positives due to limited detection strategies or inability to detect zero-day attacks without prior signatures. To overcome these issues, this project proposes an efficient cascaded hybrid intrusion detection system using deep learning and ensemble machine learning techniques. A deep autoencoder is employed for unsupervised anomaly detection through reconstruction error analysis, trained exclusively on benign traffic to detect novel threats. An Isolation Forest provides complementary anomaly scoring through isolation-based statistical modeling. Both Stage 1 detectors are fused using weighted score combination with dynamic percentile thresholding to achieve high recall. Stage 2 applies a Random Forest classifier on suspicious samples to reduce false alarms and provide precise attack-type classification. The proposed cascaded architecture leverages both unsupervised zero-day detection and supervised refinement advantages. Trained and validated on CIC-IDS2017 and CIC-IDS2018 datasets, experimental results demonstrate healthcare-grade performance with false positive rates below 5% and recall exceeding 90%, making the system suitable for production deployment in critical infrastructure environments.

---

## 1) What this project is

This project is a **hybrid Intrusion Detection System (IDS)** for network-flow data, targeted at healthcare-grade requirements (high recall, low false positives).

It combines:
- **Unsupervised anomaly detection** (to catch novel/zero-day behavior)
- **Supervised classification** (to reduce false positives and label known attacks)

The implementation supports:
- Offline training
- Offline evaluation/reporting
- Real-time (live folder) monitoring

---

## 2) Core idea in one line

Train on historical labeled network-flow datasets, learn normal behavior, then classify incoming traffic as **BENIGN** or **ATTACK** (and often attack type), with logging and reporting.

---

## 3) End-to-end system flow

## A. Training flow
1. Load CIC-IDS datasets (CSV/Parquet).
2. Clean data (deduplicate, remove NaN/Inf, normalize labels, keep numeric features).
3. Split into benign vs attack subsets.
4. Select a fixed feature subset (current training script uses 12 features).
5. Build train/validation/test splits and scaler using benign-first logic.
6. Train Stage 1 models:
   - Autoencoder on benign only
   - Isolation Forest on benign only
7. Fit fusion threshold from benign validation score distribution.
8. Train Stage 2 Random Forest on labeled train data.
9. Save artifacts: models, fusion params, scaler, selected features.

## B. Evaluation flow
1. Load saved artifacts.
2. Load test dataset and apply same cleaning/feature alignment.
3. Apply saved scaler (or derive with training data fallback).
4. Run Stage 1 scoring + threshold classification.
5. Generate metrics, plots, and deployment readiness assessment.

## C. Live monitoring flow
1. Watch folder or read single CSV window.
2. Clean input and enforce selected feature set.
3. Apply saved scaler.
4. Run cascaded inference.
5. Save prediction CSV + append anomaly JSONL logs.

---

## 4) Architecture: why two stages

## Stage 1: Anomaly detection (high sensitivity)
- **Autoencoder** learns to reconstruct benign traffic.
- **Isolation Forest** isolates outliers statistically.
- **Fusion module** combines both scores and applies threshold.

Why: catches unknown behaviors that supervised labels may miss.

## Stage 2: Supervised classifier (precision/refinement)
- **Random Forest classifier** evaluates suspicious samples.

Why: reduces false positives and gives better attack categorization/confidence.

---

## 5) Module-by-module explanation (what happens and why)

## `src/preprocessing.py` — `PreprocessingPipeline`

### What it does
- Loads multiple datasets with robust encoding/file-format handling.
- Standardizes label column names and label values.
- Cleans rows (duplicates, NaN, Inf).
- Removes non-numeric columns for model compatibility.
- Splits benign vs attack.
- Selects top features (variance or statistical F-test).
- Normalizes and creates train/val/test splits.

### Why it exists
- Ensures reproducible, consistent feature space across training/evaluation/live inference.
- Prevents training-serving skew by saving/using the same selected features and scaler.

### Key design decisions
- Train normalization primarily from benign training distribution.
- Supports `RobustScaler` for outlier-heavy traffic.
- Returns both Stage 1 and Stage 2 training arrays from one pipeline.

---

## `src/autoencoder.py` — `AutoencoderDetector`

### What it does
- Builds a dense encoder-decoder neural network.
- Trains on benign traffic only.
- Computes reconstruction error per sample.

### Why it exists
- If a sample differs from normal behavior, reconstruction error rises.
- This enables detection of novel attacks not explicitly seen in labels.

### Key design decisions
- Early stopping + checkpointing for stable training.
- Optional mixed precision/GPU use.
- Input-dimension checks to prevent artifact mismatch.

---

## `src/isolation_forest.py` — `IsolationForestDetector`

### What it does
- Trains Isolation Forest on benign samples.
- Produces anomaly score (higher = more anomalous).

### Why it exists
- Provides a second, classical anomaly signal independent of neural reconstruction.
- Improves robustness when one detector is weak on a pattern.

### Key design decisions
- Strict input validation.
- Feature-dimension checks against trained model.
- Parallel training/inference support via `n_jobs`.

---

## `src/fusion.py` — `FusionModule`

### What it does
- Learns min/max normalization stats from benign validation scores.
- Normalizes autoencoder and IF scores to comparable ranges.
- Computes weighted combined score.
- Sets dynamic threshold via percentile of benign distribution.
- Outputs binary anomaly decision.

### Why it exists
- Raw scores from different detectors are not directly comparable.
- Weighted fusion reduces single-model bias.
- Percentile thresholding lets you tune sensitivity vs false positives.

### Key design decisions
- Weight sum enforced to 1.0.
- Threshold learned from benign validation behavior.
- Clips normalized scores to `[0,1]` for stability.

---

## `src/supervised_classifier.py` — `SupervisedClassifier`

### What it does
- Trains Random Forest on labeled data (benign + attack classes).
- Produces class predictions and probabilities.
- Reports global feature importance and per-sample top features.

### Why it exists
- Converts suspicious detections into precise class decisions.
- Reduces false positives from anomaly-only detection.
- Improves explainability through feature importance outputs.

### Key design decisions
- Class imbalance handling (`class_weight='balanced'`).
- Optional grid search for hyperparameter tuning.
- Supports multi-class attack labeling.

---

## `src/cascaded_detector.py` — `CascadedDetector`

### What it does
- Orchestrates two-stage runtime inference.
- Stage 1 computes anomaly score and checks threshold.
- Benign fast-path exits early.
- Suspicious samples go to Stage 2 classifier.
- Tracks latency/statistics.

### Why it exists
- Encapsulates deployment-time control flow for speed + precision.
- Makes the “high recall first, precision second” strategy explicit.

### Key design decisions
- Stage-wise timing statistics.
- Configurable stage enable/disable.
- Detailed output including confidence, probabilities, and top features.

---

## `src/alert_system.py` — `HealthcareAlertSystem`

### What it does
- Logs anomalies in JSONL format.
- Computes metrics (accuracy, precision, recall, F1, macro-F1, FPR, ROC-AUC).
- Generates ROC, PR, and confusion matrix plots.
- Produces deployment-readiness assessment.

### Why it exists
- IDS without quality reporting is unsafe for production decisions.
- Healthcare use cases require explicit FPR/recall visibility and auditability.

---

## `src/utils.py`

### What it does
- Loads YAML config.
- Configures logging.
- Sets random seeds for reproducibility.
- Ensures required directories exist.

### Why it exists
- Keeps training/inference scripts consistent and reproducible.

---

## 6) Script-level responsibilities

## `train_cascaded_full.py`
- Full training pipeline over available CIC-IDS2017/2018 files.
- Performs feature selection, trains both stages, evaluates classifier, and saves artifacts.
- Saves critical metadata:
  - `autoencoder_best.keras`
  - `isolation_forest.pkl`
  - `fusion_params.pkl`
  - `supervised_classifier.pkl`
  - `selected_features.pkl`
  - `scaler.pkl`

## `evaluate.py`
- Loads trained artifacts and test data.
- Applies feature alignment and scaling.
- Runs evaluation and writes JSON/text reports + plots.

## `live_monitor_cascaded.py`
- Real-time folder watcher/single-file inference for cascaded detection.
- Enforces selected-feature consistency and logs ATTACK events.

## `inference.py`
- Batch inference utility focused on Stage 1 pipeline outputs and latency logging.

---

## 7) Data contracts and artifact contracts

## Input expectations
- Network flow files with label column (variant names tolerated and normalized).
- Numeric model features after cleaning.

## Output expectations
- Binary decision: benign vs attack.
- In cascaded mode: attack type/confidence for suspicious samples.
- Logs/reports in JSONL, JSON, TXT, PNG.

---

## 8) Multimodal validation (healthcare-specific enhancement)

To better align with healthcare IDS research that combines network behavior with medical context, the project now includes an optional **multimodal post-validation** layer.

### What it adds
- Uses medical signal plausibility checks (HR, SpO2, temperature, systolic/diastolic BP, respiration rate).
- Detects abrupt physiological changes between consecutive samples.
- Performs cross-modal consistency checks:
  - If network model says **BENIGN** but medical risk is high, sample is flagged as **cross-modal mismatch**.
- Produces additional outputs in live monitoring reports:
  - `medical_risk_score`
  - `combined_risk_score`
  - `cross_modal_mismatch`
  - `multimodal_prediction`
  - `multimodal_reason`

### Where it is integrated
- `src/multimodal_validation.py` (`MultimodalValidator`)
- `live_monitor_cascaded.py` (post-validation after cascaded inference)
- `config/default_config.yaml` (`multimodal_validation` section)

### Why this matters
- Network-only IDS can miss semantically invalid but protocol-normal data tampering.
- Medical validation adds patient-state plausibility as a second detection signal.
- This improves healthcare realism without changing the core cascaded ML architecture.

### Demo
- Run: `python demo_multimodal_validation.py`
- This script demonstrates benign network predictions escalated by medical anomaly checks.

## Critical consistency rule
The same selected feature set and scaler from training must be reused during evaluation/live inference. This project enforces that through `selected_features.pkl` and `scaler.pkl`.

---

## 8) Why each major design choice was made

- **Benign-only anomaly training**: enables zero-day detection.
- **Dual-detector fusion**: reduces dependence on one anomaly signal.
- **Percentile thresholding**: direct control over false-positive behavior.
- **Cascaded stage-2 classification**: improves precision and interpretability.
- **Saved preprocessing artifacts**: prevents feature mismatch errors in production.
- **Comprehensive reporting**: supports deployment decisions and compliance-style review.

---

## 9) Typical execution sequence

1. Train: `python train_cascaded_full.py`
2. Evaluate: `python evaluate.py --test-data <path>`
3. Live monitor: `python live_monitor_cascaded.py --watch-dir data/live --poll-seconds 5`

---

## 10) Practical interpretation of “Performance Tuning” in this project

In this codebase, performance tuning means iterative validation-time optimization of:
- feature count/selection method,
- autoencoder architecture and training hyperparameters,
- isolation forest parameters,
- fusion weights and percentile threshold,
- classifier hyperparameters,
to improve recall/FPR trade-off before deployment.

It is a **development-time loop**, not a runtime prediction step.
