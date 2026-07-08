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


class PerChannelAnomalyTriage:
    """
    Trains one Isolation Forest per sensor channel (5 forests, each on 7 statistical features).
    
    Why: A single-channel fault (e.g., Speed dropout) only corrupts 7/35 features.
    A single joint forest averages over all 35 features, diluting the anomaly signal.
    Per-channel forests detect anomalies in ANY individual channel independently,
    making single-sensor faults clearly separable from clean data.
    """
    
    STATS_PER_CHANNEL = 7  # mean, var, min, max, q25, q50, q75
    
    def __init__(self, channel_names, n_estimators=100, max_samples=512,
                 max_features=1.0, random_state=42):
        self.channel_names = list(channel_names)
        self.n_channels = len(self.channel_names)
        self.random_state = random_state
        
        self.models = {}
        for ch in self.channel_names:
            self.models[ch] = IsolationForest(
                n_estimators=n_estimators,
                max_samples=max_samples,
                max_features=max_features,
                contamination='auto',
                bootstrap=False,
                n_jobs=-1,
                random_state=random_state
            )
        self.thresholds = {ch: None for ch in self.channel_names}
        self.is_fitted = False

    def _split_features(self, X: np.ndarray) -> dict:
        """Split (N, 35) -> dict of {channel_name: (N, 7)}."""
        return {
            ch: X[:, i * self.STATS_PER_CHANNEL : (i + 1) * self.STATS_PER_CHANNEL]
            for i, ch in enumerate(self.channel_names)
        }

    def fit(self, X_train: np.ndarray):
        """Train one Isolation Forest per channel. X_train shape: (N, 35)."""
        print(f"Training {self.n_channels} Per-Channel Isolation Forests...")
        channel_data = self._split_features(X_train)
        for ch in self.channel_names:
            self.models[ch].fit(channel_data[ch])
            print(f"  -> {ch} forest fitted on {channel_data[ch].shape}")
        self.is_fitted = True
        print(" -> All per-channel forests fitted successfully!")

    def score_per_channel(self, X: np.ndarray) -> dict:
        """Returns {channel_name: anomaly_scores_array}. Higher = more anomalous."""
        channel_data = self._split_features(X)
        return {
            ch: -self.models[ch].score_samples(channel_data[ch])
            for ch in self.channel_names
        }

    def calibrate_thresholds(self, X_clean: np.ndarray, target_fpr_per_channel: float = 0.01):
        """Set per-channel thresholds from clean data at a given FPR per channel."""
        scores = self.score_per_channel(X_clean)
        percentile = (1.0 - target_fpr_per_channel) * 100
        print(f"Calibrating per-channel thresholds at {target_fpr_per_channel*100:.1f}% FPR each...")
        for ch in self.channel_names:
            self.thresholds[ch] = float(np.percentile(scores[ch], percentile))
            print(f"  -> {ch} threshold: {self.thresholds[ch]:.4f}")

    def triage_window(self, X_single: np.ndarray):
        """
        Score a single window of features (1, 35).
        Returns: (is_anomalous, per_channel_scores_dict, suspect_channel_name)
        """
        scores = self.score_per_channel(X_single)
        suspect = None
        max_excess = -float('inf')
        is_anomalous = False

        for ch in self.channel_names:
            score = float(scores[ch][0])
            threshold = self.thresholds[ch]
            if threshold is not None and score > threshold:
                is_anomalous = True
                excess = score - threshold
                if excess > max_excess:
                    max_excess = excess
                    suspect = ch

        per_ch = {ch: float(scores[ch][0]) for ch in self.channel_names}
        return is_anomalous, per_ch, suspect

    def save_model(self, filepath: Path):
        """Persist all per-channel forests and thresholds."""
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump({
            'models': self.models,
            'thresholds': self.thresholds,
            'channel_names': self.channel_names,
            'type': 'per_channel'
        }, filepath)
        print(f"Per-Channel Stage 1 model saved to: {filepath}")

    def load_model(self, filepath: Path):
        """Load per-channel forests and thresholds from disk."""
        data = joblib.load(filepath)
        self.models = data['models']
        self.thresholds = data['thresholds']
        self.channel_names = data['channel_names']
        self.n_channels = len(self.channel_names)
        self.is_fitted = True
        print(f"Per-Channel Stage 1 model loaded from: {filepath}")