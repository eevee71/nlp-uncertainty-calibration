import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix


class CalibrationAnalyzer:
    """Computes classification metrics and uncertainty calibration diagnostics."""

    def __init__(self, n_bins: int = 10, output_dir: str = 'figures'):
        self.n_bins = n_bins
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)

    def _get_filename(self, model_name: str, suffix: str) -> str:
        clean_name = model_name.lower().replace(" ", "_").replace("(", "").replace(")", "")
        return os.path.join(self.output_dir, f"{clean_name}_{suffix}.png")

    def print_standard_metrics(self, y_true: np.ndarray, y_pred: np.ndarray, model_name: str) -> None:
        """Calculates and prints standard metrics and displays Confusion Matrix."""

        acc = accuracy_score(y_true, y_pred)
        precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='binary')

        print(f"\n=== {model_name} - Standard Metrics ===")
        print(f"Accuracy:  {acc:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall:    {recall:.4f}")
        print(f"F1 Score:  {f1:.4f}")

        cm = confusion_matrix(y_true, y_pred)
        plt.figure(figsize=(5, 4))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title(f"{model_name} - Confusion Matrix")
        plt.xlabel("Predicted Label")
        plt.ylabel("True Label")

        filepath = self._get_filename(model_name, "confusion_matrix")
        plt.savefig(filepath, bbox_inches='tight', dpi=300)

        plt.show()

    def calculate_ece(self, y_true: np.ndarray, confidences: np.ndarray, predictions: np.ndarray,
                      model_name: str) -> float:
        """Calculates Expected Calibration Error across defined bins."""

        bin_boundaries = np.linspace(0, 1, self.n_bins + 1)
        ece = 0.0

        for i in range(self.n_bins):
            bin_lower, bin_upper = bin_boundaries[i], bin_boundaries[i + 1]
            in_bin = (confidences > bin_lower) & (confidences <= bin_upper)
            prop_in_bin = in_bin.mean()

            if prop_in_bin > 0:
                accuracy_in_bin = (y_true[in_bin] == predictions[in_bin]).mean()
                avg_confidence_in_bin = confidences[in_bin].mean()
                ece += np.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin

        print(f"[{model_name}] Expected Calibration Error (ECE): {ece:.4f}")
        return ece

    def plot_reliability_diagram(self, y_true: np.ndarray, confidences: np.ndarray, predictions: np.ndarray,
                                 model_name: str) -> None:
        """Plots the Reliability Diagram comparing model confidence vs empirical accuracy."""

        bin_boundaries = np.linspace(0, 1, self.n_bins + 1)
        accuracies, avg_confs = [], []

        for i in range(self.n_bins):
            bin_lower, bin_upper = bin_boundaries[i], bin_boundaries[i + 1]
            in_bin = (confidences > bin_lower) & (confidences <= bin_upper)

            if in_bin.any():
                accuracies.append((y_true[in_bin] == predictions[in_bin]).mean())
                avg_confs.append(confidences[in_bin].mean())
            else:
                accuracies.append(0.0)
                avg_confs.append(0.0)

        plt.figure(figsize=(6, 6))
        plt.plot([0, 1], [0, 1], linestyle='--', color='gray', label='Perfect Calibration')
        plt.plot(avg_confs, accuracies, marker='o', color='red', label=f'{model_name}')
        plt.bar(bin_boundaries[:-1], accuracies, width=1 / self.n_bins, alpha=0.3, align='edge', edgecolor='black',
                color='blue')

        plt.ylabel("Empirical Accuracy")
        plt.xlabel("Model Confidence")
        plt.title(f"{model_name} - Reliability Diagram")
        plt.legend()
        plt.grid(True, alpha=0.3)

        filepath = self._get_filename(model_name, "reliability_diagram")
        plt.savefig(filepath, bbox_inches='tight', dpi=300)

        plt.show()

    def analyze_highly_confident_errors(self, texts: np.ndarray, y_true: np.ndarray, confidences: np.ndarray,
                                        predictions: np.ndarray, model_name: str, top_k: int = 3) -> None:
        """Finds and prints cases where the model was extremely confident but wrong."""

        errors_mask = (y_true != predictions)
        error_texts = texts[errors_mask]
        error_confs = confidences[errors_mask]
        error_preds = predictions[errors_mask]
        error_trues = y_true[errors_mask]

        sorted_idx = np.argsort(error_confs)[::-1]

        print(f"\n--- {model_name} - Top {top_k} Highly Confident Errors ---")
        for idx in sorted_idx[:top_k]:
            print(
                f"Confidence: {error_confs[idx]:.4f} | True Label: {error_trues[idx]} | Predicted: {error_preds[idx]}")
            print(f"Tweet: {error_texts[idx]}\n")