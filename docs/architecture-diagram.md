# System Architecture Diagram

## Hybrid Anomaly-Based IDS for Healthcare Networks

This document contains the system architecture diagrams for the Hybrid IDS project.

## High-Level System Architecture

```mermaid
flowchart LR
    TR[(Training Dataset)]
    TS[(Test Dataset)]

    FEF[Preprocessing<br/>Fit + Transform]
    FEFS[Feature Selection<br/>Fit + Transform]
    FET[Preprocessing<br/>Transform Only]
    FSTS[Feature Selection<br/>Transform Only]

    subgraph DM[Detection Mechanism]
        direction TB
        PT[Performance Tuning]
        AE[Autoencoder]
        CL[Random Forest Classifier]
    end

    TM[Trained Models]
    OUT[Intrusion Classification<br/>Attack / Benign]

    TR --> FEF --> FEFS --> AE --> CL --> TM
    TS --> FET --> FSTS --> TM
    TM --> OUT

    PT -. tune .-> AE
    PT -. tune .-> CL

    style TR fill:#f2f2f2,stroke:#333,stroke-width:1px
    style TS fill:#f2f2f2,stroke:#333,stroke-width:1px
    style FEF fill:#f7f7f7,stroke:#333,stroke-width:1px
    style FEFS fill:#f7f7f7,stroke:#333,stroke-width:1px
    style FET fill:#f7f7f7,stroke:#333,stroke-width:1px
    style FSTS fill:#f7f7f7,stroke:#333,stroke-width:1px
    style DM fill:#ffffff,stroke:#333,stroke-width:1.5px
    style PT fill:#f7f7f7,stroke:#333,stroke-width:1px
    style AE fill:#f7f7f7,stroke:#333,stroke-width:1px
    style CL fill:#f7f7f7,stroke:#333,stroke-width:1px
    style TM fill:#f2f2f2,stroke:#333,stroke-width:1px
    style OUT fill:#f2f2f2,stroke:#333,stroke-width:1px
```

### High-Level Notes

- **Training vs test flow**: Training data uses **fit + transform** for preprocessing/feature selection, while test data uses **transform only** (same fitted pipeline, no re-fitting).
- **Model usage on test data**: Test data is transformed with the same fitted preprocessing/feature pipeline, then uses the **Trained Models** artifact for inference-only prediction.
- **Classifier naming**: `Random Forest Classifier` is used as an explicit label; `Classifier` is also valid, but this is clearer for reports.
- **Performance Tuning role**: The dashed links indicate an iterative development loop where validation feedback is used to tune feature count, autoencoder architecture, classifier hyperparameters, and thresholds before final evaluation/deployment.

## Detailed Component Architecture

```mermaid
graph LR
    subgraph "PreprocessingPipeline"
        P1[load_datasets]
        P2[clean_data]
        P3[split_benign_attack]
        P4[normalize_and_split]
        P1 --> P2 --> P3 --> P4
    end

    subgraph "AutoencoderDetector"
        A1[build_model]
        A2[train]
        A3[compute_reconstruction_error]
        A1 --> A2 --> A3
    end

    subgraph "IsolationForestDetector"
        I1[train]
        I2[compute_anomaly_score]
        I1 --> I2
    end

    subgraph "FusionModule"
        F1[fit_threshold]
        F2[normalize_scores]
        F3[compute_combined_score]
        F4[classify]
        F1 --> F2 --> F3 --> F4
    end

    subgraph "HealthcareAlertSystem"
        H1[log_anomaly]
        H2[generate_evaluation_report]
        H3[assess_deployment_readiness]
    end

    P4 --> A2
    P4 --> I1
    A3 --> F2
    I2 --> F2
    F4 --> H1
    F4 --> H2
    H2 --> H3

    style P4 fill:#e1f5ff
    style A2 fill:#ffe1e1
    style I1 fill:#ffe1e1
    style F4 fill:#fff4e1
    style H2 fill:#e1ffe1
```

## Data Flow Architecture

```mermaid
flowchart TD
    Start([Raw Network Flow Data]) --> Load[Load & Merge Datasets]
    Load --> Clean[Clean Data<br/>Remove Duplicates/NaN/Inf]
    Clean --> Filter[Filter Numeric Features]
    Filter --> Split1{Split by Label}
    
    Split1 -->|BENIGN| Benign[Benign Samples]
    Split1 -->|ATTACK| Attack[Attack Samples]
    
    Benign --> Norm[StandardScaler<br/>Fit on Benign]
    Attack --> Norm
    
    Norm --> Split2{Stratified Split}
    
    Split2 -->|70% Benign| Train[Training Set<br/>Benign Only]
    Split2 -->|20% Benign| Val[Validation Set<br/>Benign Only]
    Split2 -->|30% Mixed| Test[Test Set<br/>Benign + Attacks]
    
    Train --> AE_Train[Train Autoencoder]
    Train --> IF_Train[Train Isolation Forest]
    Val --> AE_Train
    Val --> IF_Train
    
    AE_Train --> AE_Model[Autoencoder Model]
    IF_Train --> IF_Model[Isolation Forest Model]
    
    Test --> AE_Infer[Compute Reconstruction Error]
    Test --> IF_Infer[Compute Anomaly Score]
    
    AE_Model --> AE_Infer
    IF_Model --> IF_Infer
    
    Val --> Threshold[Compute Dynamic Threshold<br/>95th/99th Percentile]
    
    AE_Infer --> Normalize[Normalize Scores<br/>Min-Max [0,1]]
    IF_Infer --> Normalize
    
    Normalize --> Combine[Weighted Combination<br/>w_ae × recon + w_if × iso]
    Threshold --> Classify{Score > Threshold?}
    Combine --> Classify
    
    Classify -->|Yes| Anomaly[Anomaly: 1]
    Classify -->|No| Normal[Benign: 0]
    
    Anomaly --> Alert[Healthcare Alert System]
    Normal --> Alert
    
    Alert --> Metrics[Compute Metrics<br/>FPR, Recall, F1, ROC-AUC]
    Alert --> Log[Log Anomalies]
    Alert --> Report[Generate Reports]
    
    Metrics --> End([Deployment Assessment])
    
    style Train fill:#ffe1e1
    style Val fill:#ffe1e1
    style Test fill:#fff4e1
    style AE_Model fill:#ffcccc
    style IF_Model fill:#ffcccc
    style Alert fill:#e1ffe1
```

## Module Interaction Sequence

```mermaid
sequenceDiagram
    participant User
    participant Config
    participant Preprocessing
    participant Autoencoder
    participant IsolationForest
    participant Fusion
    participant AlertSystem

    User->>Config: Load Configuration
    Config-->>User: Return Config Dict
    
    User->>Preprocessing: Initialize Pipeline
    User->>Preprocessing: load_datasets(paths)
    Preprocessing-->>User: Merged DataFrame
    
    User->>Preprocessing: clean_data(df)
    Preprocessing-->>User: Cleaned DataFrame
    
    User->>Preprocessing: split_benign_attack(df)
    Preprocessing-->>User: Benign & Attack DataFrames
    
    User->>Preprocessing: normalize_and_split()
    Preprocessing-->>User: Train/Val/Test Sets + Scaler
    
    par Training Phase
        User->>Autoencoder: train(X_train_benign, X_val_benign)
        Autoencoder-->>User: Training History
    and
        User->>IsolationForest: train(X_train_benign)
        IsolationForest-->>User: Trained Model
    end
    
    User->>Autoencoder: compute_reconstruction_error(X_val_benign)
    Autoencoder-->>User: Reconstruction Errors
    
    User->>IsolationForest: compute_anomaly_score(X_val_benign)
    IsolationForest-->>User: Anomaly Scores
    
    User->>Fusion: fit_threshold(recon_errors, iso_scores)
    Fusion-->>User: Threshold Computed
    
    User->>Autoencoder: compute_reconstruction_error(X_test)
    Autoencoder-->>User: Test Reconstruction Errors
    
    User->>IsolationForest: compute_anomaly_score(X_test)
    IsolationForest-->>User: Test Anomaly Scores
    
    User->>Fusion: normalize_scores(recon_errors, iso_scores)
    Fusion-->>User: Normalized Scores
    
    User->>Fusion: compute_combined_score()
    Fusion-->>User: Combined Scores
    
    User->>Fusion: classify(combined_scores)
    Fusion-->>User: Binary Predictions
    
    User->>AlertSystem: log_anomaly(predictions)
    AlertSystem-->>User: Logged
    
    User->>AlertSystem: generate_evaluation_report()
    AlertSystem-->>User: Metrics & Visualizations
    
    User->>AlertSystem: assess_deployment_readiness()
    AlertSystem-->>User: Deployment Assessment
```

## Technology Stack

```mermaid
graph TB
    subgraph "Data Processing"
        Pandas[Pandas 2.0+<br/>Data Manipulation]
        NumPy[NumPy 1.24+<br/>Numerical Computing]
    end

    subgraph "Machine Learning"
        TF[TensorFlow 2.13+<br/>Deep Learning]
        Keras[Keras API<br/>Autoencoder]
        SKL[scikit-learn 1.3+<br/>Isolation Forest]
    end

    subgraph "Visualization & Reporting"
        MPL[Matplotlib<br/>Plotting]
        Seaborn[Seaborn<br/>Statistical Viz]
    end

    subgraph "Configuration & Logging"
        YAML[PyYAML<br/>Config Files]
        Logging[Python Logging<br/>Event Tracking]
    end

    subgraph "Testing"
        Pytest[pytest<br/>Unit Tests]
        Hypothesis[Hypothesis<br/>Property Tests]
    end

    subgraph "Hardware Acceleration"
        GPU[CUDA/cuDNN<br/>GPU Support]
        Mixed[Mixed Precision<br/>Performance]
    end

    Pandas --> SKL
    NumPy --> TF
    NumPy --> SKL
    TF --> Keras
    TF --> GPU
    TF --> Mixed
    SKL --> MPL
    Keras --> MPL

    style TF fill:#ff6b6b
    style Keras fill:#ff6b6b
    style SKL fill:#4ecdc4
    style GPU fill:#ffe66d
```

## Deployment Architecture

```mermaid
graph TB
    subgraph "Healthcare Network"
        Traffic[Network Traffic<br/>Real-time Flows]
    end

    subgraph "IDS Deployment"
        Capture[Traffic Capture<br/>Flow Extraction]
        Preprocess[Preprocessing<br/>Feature Engineering]
        
        subgraph "Detection Engine"
            AE[Autoencoder<br/>GPU Accelerated]
            IF[Isolation Forest<br/>CPU Parallel]
        end
        
        Fusion[Fusion & Classification]
    end

    subgraph "Alert Management"
        SIEM[SIEM Integration]
        Dashboard[Monitoring Dashboard]
        Alerts[Alert Notifications]
    end

    subgraph "Storage & Audit"
        Logs[Anomaly Logs<br/>JSONL Format]
        Reports[Periodic Reports<br/>Metrics & Trends]
        Models[Model Storage<br/>Versioning]
    end

    Traffic --> Capture
    Capture --> Preprocess
    Preprocess --> AE
    Preprocess --> IF
    AE --> Fusion
    IF --> Fusion
    
    Fusion --> SIEM
    Fusion --> Dashboard
    Fusion --> Alerts
    Fusion --> Logs
    
    Logs --> Reports
    
    AE -.Model Updates.-> Models
    IF -.Model Updates.-> Models

    style Traffic fill:#e1f5ff
    style AE fill:#ffe1e1
    style IF fill:#ffe1e1
    style Fusion fill:#fff4e1
    style SIEM fill:#e1ffe1
    style Dashboard fill:#e1ffe1
```

## Key Design Principles

1. **Benign-Only Training**: All models train exclusively on normal traffic to enable true anomaly detection
2. **Hybrid Detection**: Combines reconstruction-based (Autoencoder) and isolation-based (Isolation Forest) approaches
3. **Dynamic Thresholding**: Uses percentile-based thresholds from benign validation distribution
4. **Healthcare Optimization**: Prioritizes low false positive rates (<5%) for clinical deployment
5. **Modularity**: Clear separation of concerns for testing, debugging, and extension
6. **Scalability**: Supports both batch processing and streaming inference
7. **Reproducibility**: Fixed random seeds and deterministic operations
8. **Zero-Day Detection**: Capable of detecting novel attack patterns not seen during training
