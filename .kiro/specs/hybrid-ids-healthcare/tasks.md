# Implementation Plan: Hybrid Anomaly-Based IDS for Healthcare Networks

## Overview

This implementation plan breaks down the development of the Hybrid Anomaly-Based Intrusion Detection System into discrete, incremental coding tasks. The system will be implemented in Python using TensorFlow/Keras for the autoencoder, scikit-learn for the Isolation Forest, and standard data science libraries (pandas, numpy) for preprocessing.

The implementation follows a bottom-up approach: first establishing the data pipeline and preprocessing, then implementing each detection module independently, followed by the fusion logic, and finally the alert system and evaluation framework. Each major component includes property-based tests to validate correctness properties from the design document.

## Tasks

- [x] 1. Set up project structure and dependencies
  - Create directory structure: `src/`, `tests/`, `data/`, `models/`, `logs/`, `reports/`, `config/`
  - Create `requirements.txt` with dependencies: tensorflow, scikit-learn, pandas, numpy, matplotlib, seaborn, hypothesis, pytest
  - Create `config/default_config.yaml` for hyperparameters and paths
  - Set up logging configuration with rotating file handlers
  - Create `src/__init__.py` and module structure
  - _Requirements: 9.2, 9.3, 10.3_

- [ ] 2. Implement preprocessing pipeline
  - [x] 2.1 Create PreprocessingPipeline class with dataset loading
    - Implement `load_datasets()` to read and merge CIC-IDS CSV files
    - Handle common label column name variations (" Label", "Label", "label")
    - Add error handling for missing files and corrupted data
    - _Requirements: 2.1, 2.2_
  
  - [x] 2.2 Implement data cleaning methods
    - Implement `clean_data()` to remove duplicates, NaN, inf values
    - Implement feature type detection and non-numeric feature removal
    - Add validation to ensure sufficient samples remain after cleaning
    - _Requirements: 2.3, 2.4, 2.5_
  
  - [x] 2.3 Implement benign/attack separation
    - Implement `split_benign_attack()` to separate samples by label
    - Validate that both benign and attack samples exist
    - Log sample counts for each category
    - _Requirements: 1.1, 1.5_
  
  - [x] 2.4 Implement normalization and stratified splitting
    - Implement `normalize_and_split()` with StandardScaler
    - Use stratified train/test split to maintain class distribution
    - Create separate benign-only validation set (20% of benign training data)
    - Return dictionary with all splits and fitted scaler
    - _Requirements: 2.6, 2.7, 2.8_
  
  - [x] 2.5 Write property tests for preprocessing
    - **Property 1: Benign-Only Training Data** - verify training/validation sets contain only benign samples
    - **Property 2: Label Preservation During Merge** - verify labels unchanged after merge
    - **Property 3: Duplicate Removal** - verify no duplicates in output
    - **Property 4: NaN and Infinity Removal** - verify no NaN/inf in output
    - **Property 5: Numeric Features Only** - verify only numeric features remain
    - **Property 6: StandardScaler Normalization** - verify mean≈0, std≈1
    - **Property 7: Stratified Sampling Preservation** - verify class proportions maintained
    - **Validates: Requirements 1.1-1.5, 2.1-2.8**

- [x] 3. Checkpoint - Verify preprocessing pipeline
  - Test preprocessing with small synthetic dataset
  - Verify all data quality checks pass
  - Ensure all tests pass, ask the user if questions arise

- [ ] 4. Implement Autoencoder detection module
  - [x] 4.1 Create AutoencoderDetector class
    - Implement `__init__()` with configuration loading
    - Set random seeds for TensorFlow reproducibility
    - Configure GPU/CPU execution and mixed precision if available
    - _Requirements: 3.1, 3.6, 3.7, 9.4_
  
  - [x] 4.2 Implement autoencoder architecture
    - Implement `build_model()` with encoder-decoder structure
    - Encoder: Input → Dense(encoding_dim*2, relu) → Dense(encoding_dim, relu)
    - Decoder: Dense(encoding_dim*2, relu) → Dense(input_dim, sigmoid)
    - Compile with MSE loss and Adam optimizer
    - Add optional dropout layers (0.2) for regularization
    - _Requirements: 3.1, 3.2, 3.4_
  
  - [x] 4.3 Implement training method
    - Implement `train()` with early stopping callback
    - Use batch processing for memory efficiency
    - Log training progress (loss per epoch)
    - Save model checkpoints
    - Validate that only benign samples are used for training
    - _Requirements: 3.3, 3.8_
  
  - [x] 4.4 Implement reconstruction error computation
    - Implement `compute_reconstruction_error()` using MSE
    - Process data in batches for efficiency
    - Return array of per-sample reconstruction errors
    - _Requirements: 3.5_
  
  - [x] 4.5 Write property tests for autoencoder
    - **Property 8: Autoencoder Reconstruction Error Output** - verify output is non-negative scalar per sample
    - **Property 21: Reproducibility with Fixed Seeds** - verify identical outputs with same seed
    - **Validates: Requirements 3.5, 9.4, 10.1**
  
  - [x] 4.6 Write unit tests for autoencoder
    - Test model architecture (layer dimensions, activation functions)
    - Test GPU fallback to CPU when GPU unavailable
    - Test early stopping behavior
    - Test batch processing with various batch sizes
    - _Requirements: 3.1, 3.6, 3.8_

- [ ] 5. Implement Isolation Forest detection module
  - [x] 5.1 Create IsolationForestDetector class
    - Implement `__init__()` with configuration loading
    - Configure n_estimators, max_samples, contamination, random_state
    - Set n_jobs=-1 for parallel processing
    - _Requirements: 4.1_
  
  - [x] 5.2 Implement training method
    - Implement `train()` to fit Isolation Forest on benign samples only
    - Validate that only benign samples are used
    - Log training completion
    - _Requirements: 4.3_
  
  - [x] 5.3 Implement anomaly score computation
    - Implement `compute_anomaly_score()` using decision_function
    - Negate scores to make higher values more anomalous
    - Return array of per-sample anomaly scores
    - _Requirements: 4.4_
  
  - [x] 5.4 Write property tests for Isolation Forest
    - **Property 9: Isolation Forest Anomaly Score Output** - verify output is scalar per sample
    - **Validates: Requirements 4.4**
  
  - [x] 5.5 Write unit tests for Isolation Forest
    - Test configuration parameters
    - Test training on benign-only data
    - Test anomaly score computation
    - _Requirements: 4.1, 4.3, 4.4_

- [x] 6. Checkpoint - Verify detection modules
  - Test both modules independently with synthetic data
  - Verify outputs are in expected format
  - Ensure all tests pass, ask the user if questions arise

- [ ] 7. Implement Fusion Module
  - [x] 7.1 Create FusionModule class
    - Implement `__init__()` with weight configuration
    - Validate weights sum to 1.0
    - Initialize storage for normalization statistics
    - _Requirements: 5.3_
  
  - [x] 7.2 Implement score normalization
    - Implement `fit_threshold()` to compute min/max from benign validation set
    - Implement `normalize_scores()` using min-max scaling to [0,1]
    - Clip test scores to [0,1] range
    - Store normalization parameters
    - _Requirements: 5.1, 5.2_
  
  - [x] 7.3 Implement score combination and thresholding
    - Implement `compute_combined_score()` using weighted average
    - Implement threshold computation using percentile (95th or 99th)
    - Implement `classify()` to apply threshold
    - _Requirements: 5.3, 5.4, 5.5, 5.6_
  
  - [x] 7.4 Write property tests for fusion module
    - **Property 10: Score Normalization to Unit Range** - verify normalized scores in [0,1]
    - **Property 11: Weighted Average Combination** - verify combined score equals weighted average
    - **Property 12: Percentile-Based Threshold** - verify threshold equals specified percentile
    - **Property 13: Threshold-Based Classification** - verify classification follows threshold rule
    - **Validates: Requirements 5.1-5.6**
  
  - [x] 7.5 Write unit tests for fusion module
    - Test weight validation (must sum to 1.0)
    - Test normalization with edge cases (all same values, single value)
    - Test threshold computation with various percentiles
    - _Requirements: 5.1-5.6_

- [ ] 8. Implement Healthcare Alert System
  - [x] 8.1 Create HealthcareAlertSystem class
    - Implement `__init__()` with log and report path configuration
    - Set up JSON Lines logging for anomalies
    - Create report directory if not exists
    - _Requirements: 6.5_
  
  - [x] 8.2 Implement anomaly logging
    - Implement `log_anomaly()` to write JSON Lines format
    - Include timestamp, flow features, anomaly score, prediction
    - Handle log file write failures gracefully
    - _Requirements: 6.5_
  
  - [x] 8.3 Implement evaluation metrics computation
    - Implement `generate_evaluation_report()` with all required metrics
    - Compute accuracy, macro F1-score, FPR, recall, precision
    - Compute per-class metrics for each attack category
    - Generate confusion matrix
    - Compute ROC-AUC score
    - Generate ROC curve and Precision-Recall curve visualizations
    - Save visualizations as PNG files
    - _Requirements: 6.6, 6.7, 6.8, 6.9, 8.1, 8.2, 8.3, 8.4_
  
  - [x] 8.4 Implement deployment readiness assessment
    - Implement `assess_deployment_readiness()` based on metrics
    - Check FPR < 5%, Recall > 90%
    - Generate readiness report with justification
    - _Requirements: 8.6_
  
  - [x] 8.5 Write property tests for alert system
    - **Property 14: Anomaly Logging Completeness** - verify all anomalies have complete log entries
    - **Property 15: ROC-AUC Computation Correctness** - verify ROC-AUC matches sklearn
    - **Property 16: Confusion Matrix Correctness** - verify confusion matrix counts
    - **Property 18: Accuracy Computation Correctness** - verify accuracy formula
    - **Property 19: Macro F1-Score Computation Correctness** - verify F1 calculation
    - **Property 20: False Positive Rate Computation Correctness** - verify FPR formula
    - **Validates: Requirements 6.5-6.9, 8.1-8.3**
  
  - [x] 8.6 Write unit tests for alert system
    - Test log file creation and writing
    - Test report generation with known metrics
    - Test deployment readiness assessment logic
    - Test visualization generation
    - **Validates: Requirements 6.5-6.9, 8.6**
    - **Property 18: Accuracy Computation Correctness** - verify accuracy formula
    - **Property 19: Macro F1-Score Computation Correctness** - verify F1 calculation
    - **Property 20: False Positive Rate Computation Correctness** - verify FPR formula
    - **Validates: Requirements 6.5-6.9, 8.1-8.3**
  
  - [ ] 8.6 Write unit tests for alert system
    - Test log file creation and writing
    - Test report generation with known metrics
    - Test deployment readiness assessment logic
    - Test visualization generation
    - _Requirements: 6.5-6.9, 8.6_

- [x] 9. Checkpoint - Verify complete pipeline
  - Test end-to-end pipeline with synthetic data
  - Verify all components integrate correctly
  - Ensure all tests pass, ask the user if questions arise

- [x] 10. Implement main training pipeline
  - [x] 10.1 Create training script
    - Implement `train.py` as main entry point
    - Load configuration from YAML file
    - Initialize all components (preprocessing, autoencoder, isolation forest, fusion)
    - Execute training pipeline: load data → preprocess → train models → fit threshold
    - Save trained models and normalization parameters
    - Log training summary
    - _Requirements: 9.1, 9.6_
  
  - [x] 10.2 Add reproducibility controls
    - Set random seeds for NumPy, TensorFlow, scikit-learn, Python random
    - Document seed values in configuration
    - Verify reproducibility in training script
    - _Requirements: 9.4, 10.1_
  
  - [x] 10.3 Write integration tests for training pipeline
    - Test complete training flow with small synthetic dataset
    - Verify models are saved correctly
    - Verify reproducibility with fixed seeds
    - _Requirements: 9.4, 10.1_

- [x] 11. Implement inference pipeline
  - [x] 11.1 Create inference script
    - Implement `inference.py` for batch processing
    - Load trained models and normalization parameters
    - Process input data through complete pipeline
    - Generate predictions and anomaly scores
    - Measure and log inference latency
    - _Requirements: 6.3, 10.4_
  
  - [x] 11.2 Implement streaming inference support
    - Implement `stream_inference.py` for real-time detection
    - Process samples one at a time or in mini-batches
    - Optimize for low latency (<100ms per sample)
    - _Requirements: 6.3, 10.5_
  
  - [x] 11.3 Write unit tests for inference pipelines
    - Test batch inference with various batch sizes
    - Test streaming inference latency
    - Test error handling for invalid inputs
    - _Requirements: 6.3, 10.4, 10.5_

- [-] 12. Implement evaluation script
  - [x] 12.1 Create evaluation script
    - Implement `evaluate.py` to test trained system
    - Load test data (benign + attacks)
    - Run inference on all test samples
    - Generate comprehensive evaluation report
    - Call HealthcareAlertSystem to produce metrics and visualizations
    - Save report as JSON and human-readable text
    - _Requirements: 8.1-8.6_
  
  - [x] 12.2 Add zero-day evaluation mode
    - Implement cross-dataset evaluation (train on 2017, test on 2018)
    - Report per-attack-category detection rates
    - Highlight novel attack detection capability
    - _Requirements: 7.1, 7.2, 7.4, 7.5_
  
  - [x] 12.3 Write property test for zero-day detection
    - **Property 17: Zero-Day Attack Detection** - verify novel attacks get high anomaly scores
    - **Validates: Requirements 7.2, 7.5**

- [x] 13. Create configuration and documentation
  - [x] 13.1 Create comprehensive configuration file
    - Document all hyperparameters with descriptions
    - Provide default values optimized for healthcare use case
    - Include paths for datasets, models, logs, reports
    - _Requirements: 9.2_
  
  - [x] 13.2 Write README with installation and usage instructions
    - Document system requirements (Python version, GPU optional)
    - Provide installation steps
    - Explain how to run training, inference, and evaluation
    - Include example commands
    - _Requirements: 10.2_
  
  - [x] 13.3 Create requirements.txt with pinned versions
    - List all dependencies with version constraints
    - Separate optional dependencies (GPU support)
    - _Requirements: 10.3_

- [-] 14. Final integration and testing
  - [x] 14.1 Run complete test suite
    - Execute all unit tests
    - Execute all property tests (100 iterations each)
    - Verify >80% code coverage
    - _Requirements: All_
  
  - [x] 14.2 Test with real CIC-IDS datasets
    - Download CIC-IDS2017 and CIC-IDS2018 datasets
    - Run complete training pipeline
    - Evaluate on test set
    - Verify FPR < 5% and Recall > 90%
    - Generate deployment readiness report
    - _Requirements: 6.1, 6.2, 7.1-7.5, 8.6_
  
  - [ ] 14.3 Performance optimization
    - Profile inference latency
    - Optimize bottlenecks if latency > 100ms
    - Test GPU acceleration effectiveness
    - _Requirements: 6.3, 3.6, 10.6_

- [ ] 15. Final checkpoint - System validation
  - Verify all requirements are met
  - Review deployment readiness assessment
  - Ensure all tests pass, ask the user if questions arise

## Notes

- Tasks marked with `*` are optional property-based and unit tests that can be skipped for faster MVP development
- Each task references specific requirements for traceability
- Checkpoints ensure incremental validation and provide opportunities for user feedback
- Property tests validate universal correctness properties from the design document
- Unit tests validate specific examples, edge cases, and integration points
- The implementation uses Python with TensorFlow, scikit-learn, pandas, and Hypothesis for property testing
- GPU acceleration is optional; the system falls back to CPU automatically
- All random operations use fixed seeds for reproducibility
- The system prioritizes low false positive rate for healthcare deployment
