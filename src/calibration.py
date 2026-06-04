import numpy as np
from scipy.optimize import minimize
from sklearn.isotonic import IsotonicRegression


class TemperatureScaler:
    """Parametric calibration using Temperature Scaling on pseudo-logits."""

    def __init__(self):
        self.T = 1.0

    def _probs_to_logits(self, probs: np.ndarray, eps: float = 1e-7) -> np.ndarray:
        """Converts probabilities of class 1 into pseudo-logits with clipping."""
        clipped_probs = np.clip(probs[:, 1], eps, 1.0 - eps)
        return np.log(clipped_probs / (1.0 - clipped_probs))

    def fit(self, val_probs: np.ndarray, y_val: np.ndarray) -> 'TemperatureScaler':
        """Optimizes temperature T using Negative Log-Likelihood on validation set."""
        logits = self._probs_to_logits(val_probs)

        def nll_loss(T_val):
            t = T_val[0]
            if t <= 0:
                return 1e5
            scaled_logits = logits / t
            scaled_probs = 1.0 / (1.0 + np.exp(-scaled_logits))
            scaled_probs = np.clip(scaled_probs, 1e-15, 1.0 - 1.0e-15)
            loss = -np.mean(y_val * np.log(scaled_probs) + (1.0 - y_val) * np.log(1.0 - scaled_probs))
            return loss

        # Optimize T starting from 1.0
        res = minimize(nll_loss, x0=[1.0], bounds=[(1e-3, 10.0)], method='L-BFGS-B')
        self.T = res.x[0]
        return self

    def predict_proba(self, probs: np.ndarray) -> np.ndarray:
        """Applies calibrated temperature to input probabilities."""
        logits = self._probs_to_logits(probs)
        scaled_logits = logits / self.T
        p1 = 1.0 / (1.0 + np.exp(-scaled_logits))
        return np.vstack([1.0 - p1, p1]).T


class IsotonicCalibrator:
    """Non-parametric calibration using Isotonic Regression."""

    def __init__(self):
        self.iso = IsotonicRegression(out_of_bounds='clip')

    def fit(self, val_probs: np.ndarray, y_val: np.ndarray) -> 'IsotonicCalibrator':
        """Fits isotonic mapping on validation probabilities."""
        self.iso.fit(val_probs[:, 1], y_val)
        return self

    def predict_proba(self, probs: np.ndarray) -> np.ndarray:
        """Maps probabilities through the fitted isotonic function."""
        p1 = self.iso.predict(probs[:, 1])
        return np.vstack([1.0 - p1, p1]).T