# src/train_isolation_forest.py
import sys
from pathlib import Path
import numpy as np

# Ensure project root is in Python path
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

from src.config import DATA_DIR
from src.isolation_forest_model import TelemetryAnomalyTriage

def train_and_evaluate_stage1():
    
    # 1. Load pre-computed clean training matrix
    stage1_data_path = DATA_DIR / "X_stage1_train.npy"
    if not stage1_data_path.exists():
        raise FileNotFoundError(f"Missing {stage1_data_path}! Run 'python -m src.build_dataset' first.")
        
    X_train = np.load(stage1_data_path)
    print(f"Loaded Clean Baseline Matrix: {X_train.shape} (Windows, Features)")
    
    # 2. Initialize and Train Model
    triage_model = TelemetryAnomalyTriage(
        n_estimators=100,
        max_samples=1024,      # Deep variance mapping
        max_features=0.7,      # Feature subspace diversification
        contamination=0.001
    )
    triage_model.fit(X_train)
    
    # 3. Score the training data to inspect continuous score distribution
    print("\nCalculating clean baseline score distribution...")
    clean_scores = triage_model.score_windows(X_train)
    
    min_score = np.min(clean_scores)
    mean_score = np.mean(clean_scores)
    max_score = np.max(clean_scores)
    p99 = np.percentile(clean_scores, 99.0)
    p999 = np.percentile(clean_scores, 99.9)
    
    print(" -> Clean Baseline Anomaly Score Summary:")
    print(f"    Min Score:        {min_score:.4f}")
    print(f"    Mean Score:       {mean_score:.4f}")
    print(f"    99th Percentile:  {p99:.4f}")
    print(f"    99.9th Percentile:{p999:.4f} (Recommended Baseline Triage Threshold)")
    print(f"    Max Score:        {max_score:.4f}")
    
    # Calibrate a baseline threshold assuming 0.1% false positive tolerance on clean data
    triage_model.calibrate_threshold_from_validation(clean_scores, target_fpr=0.001)
    
    # 4. Save model to disk
    model_save_path = Path("models") / "stage1_iforest.pkl"
    triage_model.save_model(model_save_path)
    print("\n=== STAGE 1 TRAINING COMPLETE ===")

if __name__ == "__main__":
    train_and_evaluate_stage1()