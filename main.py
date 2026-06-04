import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

from src.data_loader import DisasterDataLoader
from src.metrics import CalibrationAnalyzer
from src.models import (run_logistic_regression, run_mlp_embeddings, run_lstm, run_transformer)
from src.calibration import TemperatureScaler, IsotonicCalibrator
from src.rejection import RejectionMechanism

def main():

    X_train_raw, X_test_raw, y_train, y_test = DisasterDataLoader('data/train.csv').load_splits()

    X_m_train, X_c_val, y_m_train, y_c_val = train_test_split(
        X_train_raw, y_train, test_size=0.2, random_state=42, stratify=y_train
    )

    X_eval_raw = np.concatenate([X_c_val, X_test_raw])

    models = {
        "Logistic Regression TF-IDF": run_logistic_regression,
        "MLP Embeddings": run_mlp_embeddings,
        "LSTM Recurrent Net": run_lstm,
        "Transformer BERT-Tiny": run_transformer
    }

    analyzer = CalibrationAnalyzer(n_bins=10)
    rejector = RejectionMechanism()

    all_uncal_test_probs = []

    print("STARTING PIPELINE: TRAINING & INDIVIDUAL CALIBRATION")

    for name, func in models.items():
        print(f"\n-> Processing Model: {name} <-\n")

        probs_eval = func(X_m_train, X_eval_raw, y_m_train)

        probs_val = probs_eval[:len(X_c_val)]
        probs_test = probs_eval[len(X_c_val):]
        
        all_uncal_test_probs.append(probs_test)

        preds_test_uncal = np.argmax(probs_test, axis=1)
        confs_test_uncal = np.max(probs_test, axis=1)

        print(f"\n[BASELINE] Evaluating uncalibrated {name}")
        analyzer.print_standard_metrics(y_test, preds_test_uncal, model_name=f"{name} (Base)")
        analyzer.calculate_ece(y_test, confs_test_uncal, preds_test_uncal, model_name=f"{name} (Base)")
        analyzer.plot_reliability_diagram(y_test, confs_test_uncal, preds_test_uncal, model_name=f"{name} (Base)")
        analyzer.analyze_highly_confident_errors(X_test_raw, y_test, confs_test_uncal, preds_test_uncal, model_name=f"{name} (Base)", top_k=3)

        print(f"\n[CALIBRATION] Fitting Temperature Scaling & Isotonic Regression...")
        
        t_scaler = TemperatureScaler().fit(probs_val, y_c_val)
        i_scaler = IsotonicCalibrator().fit(probs_val, y_c_val)

        probs_test_ts = t_scaler.predict_proba(probs_test)
        probs_test_iso = i_scaler.predict_proba(probs_test)

        print(f"\n[CALIBRATION RESULT] Comparing ECE values for {name}:")
        analyzer.calculate_ece(y_test, np.max(probs_test_ts, axis=1), np.argmax(probs_test_ts, axis=1), model_name=f"{name} (Temp Scaling)")
        analyzer.calculate_ece(y_test, np.max(probs_test_iso, axis=1), np.argmax(probs_test_iso, axis=1), model_name=f"{name} (Isotonic)")

        analyzer.plot_reliability_diagram(y_test, np.max(probs_test_ts, axis=1), np.argmax(probs_test_ts, axis=1), model_name=f"{name} (Temp Scaling)")
        analyzer.plot_reliability_diagram(y_test, np.max(probs_test_iso, axis=1), np.argmax(probs_test_iso, axis=1), model_name=f"{name} (Isotonic)")
        
        print(f"Individual reliability diagrams saved to figures/ folder.")
        print("-" * 60)


    print("STAGE 2: METHOD 3 — MODEL ENSEMBLE EVALUATION")
    
    ensemble_probs = np.mean(all_uncal_test_probs, axis=0)
    ensemble_preds = np.argmax(ensemble_probs, axis=1)
    ensemble_confs = np.max(ensemble_probs, axis=1)
    
    print("[ENSEMBLE] Evaluating Model Ensemble (Average of all 4 architectures)")
    analyzer.print_standard_metrics(y_test, ensemble_preds, model_name="Model Ensemble")
    ece_ensemble = analyzer.calculate_ece(y_test, ensemble_confs, ensemble_preds, model_name="Model Ensemble")
    analyzer.plot_reliability_diagram(y_test, ensemble_confs, ensemble_preds, model_name="Model Ensemble")


    print("STAGE 3: 'I DON'T KNOW' MECHANISM USING ENSEMBLE PROBABILITIES")
    
    thresholds = np.array([0.5, 0.6, 0.7, 0.8, 0.85, 0.9, 0.95])
    rejection_df = rejector.evaluate_thresholds(y_test, ensemble_probs, thresholds)
    
    print("Rejection table for Model Ensemble:")
    print(rejection_df.to_string(index=False, formatters={
        'threshold': '{:.2f}'.format, 'accuracy': '{:.2%}'.format, 
        'coverage': '{:.2%}'.format, 'rejected_count': '{:d}'.format
    }))
    
    rejector.plot_tradeoff(rejection_df, model_name="Model Ensemble")
    print(f"\nSaved ensemble trade-off plot to figures/ directory.")
    print("Pipeline finished successfully!")


if __name__ == '__main__':
    main()