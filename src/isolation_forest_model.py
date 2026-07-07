# src/isolation_forest_model.py
import numpy as np
import joblib
from pathlib import Path
from sklearn.ensemble import IsolationForest
from typing import Tuple, Optional

class TelemetryAnomalyTriage:

    def __init__(self, 
                 n_estimators: int = 100, 
                 max_samples: int = 1024, 
                 max_features: float = 0.7, 
                 contamination: float = 0.001, 
                 random_state: int = 42):
        
        self.n_estimators = n_estimators
        self.max_samples = max_samples
        self.max_features = max_features
        self.contamination = contamination
        self.random_state = random_state
        
        # Initialize Scikit-Learn Isolation Forest with engineering overrides
        self.model = IsolationForest(
            n_estimators=self.n_estimators,
            max_samples=self.max_samples,
            max_features=self.max_features,
            contamination=self.contamination,
            bootstrap=False,            # Keep path-length math pure
            n_jobs=-1,                  # Utilize all CPU cores
            random_state=self.random_state
        )
        self.is_fitted = False
        self.calibrated_threshold = None

    def fit(self, X_train: np.ndarray):
        """
        Trains the Isolation Forest exclusively on clean, anomaly-free statistical windows.
        Expected shape: (Batch, 35)
        """
        print(f"Training Stage 1 Isolation Forest on {X_train.shape[0]} windows...")
        print(f" -> Hyperparameters: max_samples={self.max_samples}, max_features={self.max_features}")
        self.model.fit(X_train)
        self.is_fitted = True
        print(" -> Model fitted successfully!")

    def score_windows(self, X: np.ndarray) -> np.ndarray:
        """
        Returns continuous anomaly scores using .score_samples().
        We invert the sign so that HIGHER scores = MORE ANOMALOUS (easier for dashboard thresholds).
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before scoring windows!")
            
        raw_scores = self.model.score_samples(X)
        # Invert scores: normal driving ~ low positive numbers, anomalies ~ high positive spikes
        inverted_scores = -raw_scores 
        return inverted_scores

    def triage_windows(self, X: np.ndarray, custom_threshold: Optional[float] = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Executes real-time triage.
        Returns:
            - scores: Array of continuous anomaly scores
            - flags: Boolean array where True indicates an anomaly crossing threshold
        """
        scores = self.score_windows(X)
        threshold = custom_threshold if custom_threshold is not None else self.calibrated_threshold
        
        if threshold is None:
            # Fallback threshold: 99.9th percentile of training score distribution
            threshold = np.percentile(scores, 99.9)
            
        flags = scores > threshold
        return scores, flags

    def calibrate_threshold_from_validation(self, validation_scores: np.ndarray, target_fpr: float = 0.01) -> float:
        """
        Dynamically calibrates the triage hand-off threshold based on an acceptable False Positive Rate (FPR).
        """
        # Sort validation scores (from clean baseline testing data)
        self.calibrated_threshold = np.percentile(validation_scores, (1.0 - target_fpr) * 100)
        print(f"Calibrated Triage Threshold set to: {self.calibrated_threshold:.4f} (at {target_fpr*100}% FPR)")
        return self.calibrated_threshold

    def save_model(self, filepath: Path):
        """Persists trained weights and calibrated threshold to disk."""
        filepath.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump({'model': self.model, 'threshold': self.calibrated_threshold}, filepath)
        print(f"Stage 1 model successfully saved to: {filepath}")

    def load_model(self, filepath: Path):
        """Loads trained weights and threshold from disk."""
        if not filepath.exists():
            raise FileNotFoundError(f"No saved model found at {filepath}")
        data = joblib.load(filepath)
        self.model = data['model']
        self.calibrated_threshold = data['threshold']
        self.is_fitted = True
        print(f"Stage 1 model loaded from: {filepath}")