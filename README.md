# NLP Uncertainty & Calibration Analysis

This repository provides a framework for evaluating model uncertainty and calibration in NLP classification tasks, based on the Disaster Tweets dataset. The project implements a complete pipeline to diagnose, calibrate, and filter model predictions.

## Project Scope & Implementation

### 1. Baseline & Diagnostics
Evaluation of four architectures: **TF-IDF + Logistic Regression**, **MLP**, **LSTM**, and **BERT-Tiny**.
- **Metrics:** F1-score, Precision, Recall, Accuracy, and confusion matrices.
- **Uncertainty Analysis:** Investigation of the relationship between model confidence and prediction accuracy, including the identification of high-confidence errors and calculation of *Expected Calibration Error (ECE)* and *Reliability Diagrams*.

### 2. Model Calibration
Implementation and comparison of two techniques to align confidence scores with true probabilities, as discussed in *Guo et al. (2017)*:
- **Temperature Scaling:** Parametric optimization of the temperature parameter $T$ (minimizing Negative Log-Likelihood).
- **Isotonic Regression:** Non-parametric probability mapping using validation data.

### 3. Rejection Mechanism ("I Don't Know")
A threshold-based strategy ($\tau$) to handle model uncertainty. The system evaluates the performance trade-offs by analyzing:
- **Accuracy vs. Coverage:** How filtering low-confidence predictions impacts overall system precision.
- **Ensemble Evaluation:** Using ensemble probabilities to enhance the robustness of the rejection mechanism.

## References
- *On Calibration of Modern Neural Networks*  (Guo et al., 2017)