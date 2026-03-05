# Requirements Document

## Introduction

This document specifies the requirements for a Hybrid Anomaly-Based Intrusion Detection System (IDS) designed specifically for healthcare networks. The system combines deep learning (Autoencoder) and classical machine learning (Isolation Forest) to detect known attacks, variations of known attacks, and zero-day attacks without relying on signature-based or supervised attack classification. The system is trained exclusively on benign traffic to ensure true anomaly detection capability.

## Glossary

- **System**: The Hybrid Anomaly-Based Intrusion Detection System
- **Autoencoder**: Deep learning neural network component that learns to reconstruct benign traffic patterns
- **Isolation_Forest**: Classical machine learning component that identifies anomalies through isolation
- **Reconstruction_Error**: Measure of how poorly the Autoencoder reconstructs input data
- **Anomaly_Score**: Combined metric from both detection modules indicating likelihood of malicious activity
- **Benign_Traffic**: Normal, non-malicious network traffic used exclusively for training
- **Attack_Traffic**: Malicious network traffic used only for evaluation and testing
- **Fusion_Module**: Component that combines outputs from Autoencoder and Isolation Forest
- **Threshold**: Dynamic cutoff value derived from benign validation distribution
- **Zero_Day_Attack**: Previously unseen attack pattern not present in training data
- **CIC_IDS2017**: Intrusion detection dataset containing labeled network flows from 2017
- **CIC_IDS2018**: Intrusion detection dataset containing labeled network flows from 2018
- **False_Positive_Rate**: Proportion of benign traffic incorrectly classified as malicious
- **Preprocessing_Pipeline**: Data transformation component handling cleaning, normalization, and feature preparation
- **Healthcare_Alert_System**: Output component that logs and reports detected anomalies

## Requirements

### Requirement 1: Benign-Only Training Strategy

**User Story:** As a security researcher, I want the system to train exclusively on benign traffic, so that it can detect truly novel zero-day attacks without bias from known attack patterns.

#### Acceptance Criteria

1. WHEN the training pipeline is executed, THE System SHALL use only samples labeled as BENIGN from the dataset
2. WHEN the training pipeline encounters attack-labeled samples, THE System SHALL exclude them from the training set
3. WHEN model training completes, THE System SHALL validate that no attack samples were included in training data
4. WHEN creating validation sets, THE System SHALL use only benign traffic samples
5. THE System SHALL reserve all attack samples exclusively for evaluation and testing phases

### Requirement 2: Dataset Acquisition and Preprocessing

**User Story:** As a data engineer, I want the system to properly load and preprocess CIC-IDS2017 and CIC-IDS2018 datasets, so that the data is clean and suitable for anomaly detection.

#### Acceptance Criteria

1. WHEN loading datasets, THE Preprocessing_Pipeline SHALL read both CIC-IDS2017 and CIC-IDS2018 datasets
2. WHEN merging datasets, THE Preprocessing_Pipeline SHALL combine them while preserving label information
3. WHEN duplicate records are detected, THE Preprocessing_Pipeline SHALL remove them
4. WHEN NaN or infinite values are detected, THE Preprocessing_Pipeline SHALL remove affected rows
5. WHEN non-numeric features are encountered, THE Preprocessing_Pipeline SHALL drop them from the feature set
6. WHEN preparing features for training, THE Preprocessing_Pipeline SHALL apply StandardScaler normalization
7. WHEN splitting data for testing, THE Preprocessing_Pipeline SHALL use stratified sampling to maintain class distribution
8. THE Preprocessing_Pipeline SHALL maintain a separate validation set composed only of benign samples

### Requirement 3: Autoencoder Architecture and Training

**User Story:** As a machine learning engineer, I want to implement a deep learning autoencoder trained on benign traffic, so that it can detect anomalies through reconstruction error.

#### Acceptance Criteria

1. THE Autoencoder SHALL implement an encoder-decoder architecture using TensorFlow or Keras
2. THE Autoencoder SHALL accept all numeric flow features as input
3. WHEN training the Autoencoder, THE System SHALL use only benign traffic samples
4. THE Autoencoder SHALL use Mean Squared Error (MSE) as the loss function
5. WHEN processing input samples, THE Autoencoder SHALL output reconstruction error as the anomaly indicator
6. THE Autoencoder SHALL utilize GPU acceleration when available
7. WHERE mixed precision training is supported, THE Autoencoder SHALL use it for efficiency
8. THE Autoencoder SHALL process data in batches for memory efficiency

### Requirement 4: Isolation Forest Implementation

**User Story:** As a machine learning engineer, I want to implement an Isolation Forest trained on benign traffic, so that it provides a complementary anomaly detection mechanism.

#### Acceptance Criteria

1. THE Isolation_Forest SHALL be implemented using scikit-learn or equivalent library
2. THE Isolation_Forest SHALL accept all numeric flow features as input
3. WHEN training the Isolation Forest, THE System SHALL use only benign traffic samples
4. WHEN processing input samples, THE Isolation_Forest SHALL output an anomaly score
5. THE Isolation_Forest SHALL be configured for efficient inference on streaming data

### Requirement 5: Hybrid Fusion Strategy

**User Story:** As a system architect, I want to combine outputs from both detection modules intelligently, so that the system leverages strengths of both approaches.

#### Acceptance Criteria

1. THE Fusion_Module SHALL normalize the reconstruction error from the Autoencoder
2. THE Fusion_Module SHALL normalize the anomaly score from the Isolation Forest
3. THE Fusion_Module SHALL combine normalized scores using a weighted average or learnable weight parameter
4. THE Fusion_Module SHALL derive a dynamic threshold from the benign validation distribution
5. THE Fusion_Module SHALL use percentile-based cutoff at the 95th or 99th percentile
6. WHEN the combined score exceeds the threshold, THE Fusion_Module SHALL classify the sample as anomalous

### Requirement 6: Healthcare-Specific Performance Constraints

**User Story:** As a healthcare IT administrator, I want the system to minimize false alarms while maintaining high attack detection rates, so that it is practical for deployment in clinical environments.

#### Acceptance Criteria

1. THE System SHALL prioritize achieving a low False Positive Rate on benign traffic
2. THE System SHALL maintain high Recall (sensitivity) for detecting attack traffic
3. THE System SHALL perform inference with latency suitable for near real-time detection
4. THE System SHALL be lightweight enough to run on standard healthcare network infrastructure
5. WHEN an anomaly is detected, THE Healthcare_Alert_System SHALL log the event with timestamp and flow details
6. THE System SHALL generate reports including False Positive Rate per attack class
7. THE System SHALL compute and report ROC-AUC scores
8. THE System SHALL generate Precision-Recall curves for evaluation
9. THE System SHALL produce confusion matrices showing classification performance

### Requirement 7: Zero-Day Attack Detection Capability

**User Story:** As a cybersecurity analyst, I want the system to detect previously unseen attack patterns, so that healthcare networks are protected against zero-day threats.

#### Acceptance Criteria

1. WHEN evaluating zero-day capability, THE System SHALL be tested on all attack categories from both datasets
2. THE System SHALL detect attacks that were not present during training
3. WHERE cross-dataset generalization is tested, THE System SHALL train on 2017 benign traffic and test on 2018 attacks
4. THE System SHALL report detection rates for each attack category separately
5. THE System SHALL demonstrate capability to detect variations of known attacks

### Requirement 8: Model Evaluation and Reporting

**User Story:** As a project stakeholder, I want comprehensive evaluation metrics and visualizations, so that I can assess system performance and deployment readiness.

#### Acceptance Criteria

1. WHEN evaluation completes, THE System SHALL report overall accuracy
2. THE System SHALL compute and report macro-averaged F1-score
3. THE System SHALL report False Positive Rate across all classes
4. THE System SHALL generate ROC curves with AUC scores
5. THE System SHALL report the threshold value used for classification
6. THE System SHALL provide a healthcare deployment readiness assessment based on performance metrics

### Requirement 9: System Architecture and Modularity

**User Story:** As a software engineer, I want a modular and maintainable codebase with clear separation of concerns, so that the system is easy to extend and debug.

#### Acceptance Criteria

1. THE System SHALL implement a modular architecture with separate components for preprocessing, detection, fusion, and alerting
2. THE System SHALL use a configuration file for all hyperparameters
3. THE System SHALL implement a logging system for tracking execution and debugging
4. THE System SHALL use reproducible random seeds for all stochastic operations
5. THE System SHALL maintain clear separation between training and inference pipelines
6. THE System SHALL organize code following the flow: Data → Preprocessing → Detection Modules → Fusion → Thresholding → Alerting

### Requirement 10: Reproducibility and Deployment

**User Story:** As a DevOps engineer, I want the system to be reproducible and deployment-ready, so that it can be reliably deployed in production healthcare environments.

#### Acceptance Criteria

1. THE System SHALL set random seeds for NumPy, TensorFlow, and scikit-learn to ensure reproducibility
2. THE System SHALL provide clear documentation for installation and setup
3. THE System SHALL include requirements specification for all dependencies
4. THE System SHALL support batch processing for offline analysis
5. THE System SHALL support streaming inference for real-time detection
6. WHERE GPU resources are available, THE System SHALL utilize them for training
7. THE System SHALL gracefully handle absence of GPU by falling back to CPU execution
