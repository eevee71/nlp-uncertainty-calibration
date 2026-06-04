# src/rejection.py
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score


class RejectionMechanism:
    """Handles the 'I don't know' decision strategy based on a confidence threshold."""

    def __init__(self, output_dir: str = 'figures'):
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)

    def evaluate_thresholds(self, y_true: np.ndarray, probs: np.ndarray, thresholds: np.ndarray) -> pd.DataFrame:
        """Evaluates model performance and data coverage for various confidence thresholds."""
        confidences = np.max(probs, axis=1)
        predictions = np.argmax(probs, axis=1)
        total_samples = len(y_true)

        results = []
        for tau in thresholds:
            accepted_mask = confidences >= tau
            rejected_count = np.sum(~accepted_mask)
            coverage = np.sum(accepted_mask) / total_samples

            if np.sum(accepted_mask) > 0:
                acc = accuracy_score(y_true[accepted_mask], predictions[accepted_mask])
            else:
                acc = np.nan  # Model said 'I don't know' to everything

            results.append({
                'threshold': tau,
                'accuracy': acc,
                'coverage': coverage,
                'rejected_count': rejected_count
            })

        return pd.DataFrame(results)

    def plot_tradeoff(self, df_results: pd.DataFrame, model_name: str) -> None:
        """Plots and saves the dual-axis trade-off graph between Accuracy and Coverage."""
        fig, ax1 = plt.subplots(figsize=(7, 5))

        color = 'tab:blue'
        ax1.set_xlabel('Confidence Threshold (tau)')
        ax1.set_ylabel('Accuracy on Accepted Samples', color=color)
        ax1.plot(df_results['threshold'], df_results['accuracy'], marker='o', color=color, linewidth=2, label='Accuracy')
        ax1.tick_params(axis='y', labelcolor=color)
        ax1.grid(True, alpha=0.3)

        ax2 = ax1.twinx()
        color = 'tab:orange'
        ax2.set_ylabel('Data Coverage', color=color)
        ax2.plot(df_results['threshold'], df_results['coverage'], marker='s', linestyle='--', color=color, linewidth=2, label='Coverage')
        ax2.tick_params(axis='y', labelcolor=color)

        plt.title(f'{model_name} - "I Don\'t Know" Mechanism Trade-off')
        fig.tight_layout()

        clean_name = model_name.lower().replace(" ", "_").replace("(", "").replace(")", "")
        filepath = os.path.join(self.output_dir, f"{clean_name}_rejection_tradeoff.png")
        plt.savefig(filepath, bbox_inches='tight', dpi=300)
        plt.close()