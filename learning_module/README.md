# Interactive Learning Module: Hybrid IDS Implementation

Welcome! This module will teach you how to build a hybrid intrusion detection system from scratch. You'll learn by reimplementing each component with guided exercises and explanations.

## New: Full Project Rebuild Path

- **[PROJECT_REBUILD_MODULES.md](PROJECT_REBUILD_MODULES.md)** - End-to-end hands-on module plan to recreate this full project yourself (with solution references per module).

## What You'll Build

A production-ready hybrid IDS that:
- Detects network intrusions using deep learning + classical ML
- Trains only on benign traffic (zero-day detection capable)
- Achieves <5% false positive rate for healthcare deployment
- Processes network flows in real-time

## Learning Path

### Phase 1: Foundations (Weeks 1-2)
1. **Data Preprocessing Pipeline** - Learn data cleaning, normalization, and splitting
2. **Understanding Network Flow Features** - Explore CIC-IDS datasets

### Phase 2: Deep Learning Detection (Weeks 3-4)
3. **Autoencoder Architecture** - Build reconstruction-based anomaly detection
4. **Training Strategies** - Master benign-only training and early stopping

### Phase 3: Classical ML Detection (Week 5)
5. **Isolation Forest** - Implement tree-based anomaly detection
6. **Comparative Analysis** - Understand strengths of each approach

### Phase 4: Fusion & Deployment (Week 6)
7. **Score Fusion** - Combine multiple detectors effectively
8. **Healthcare Optimization** - Tune for low false positives

## Prerequisites

- Python programming (intermediate level)
- Basic machine learning concepts
- Understanding of neural networks (helpful but not required)
- Familiarity with pandas and numpy

## How to Use This Module

Each lesson includes:
- **Concept Explanation**: Why this component matters
- **Code Walkthrough**: Line-by-line explanation of the implementation
- **Exercises**: Hands-on coding challenges
- **Quiz**: Test your understanding
- **Project**: Build the component yourself

Start with `01_preprocessing/lesson.md` and work through sequentially.

## Estimated Time

- **Fast Track**: 20-30 hours (experienced developers)
- **Standard**: 40-60 hours (learning as you go)
- **Deep Dive**: 80-100 hours (with all exercises and projects)

## Support

- Check `solutions/` folder for reference implementations
- Review `common_mistakes.md` for debugging help
- See `resources.md` for additional learning materials

Let's begin! 🚀
