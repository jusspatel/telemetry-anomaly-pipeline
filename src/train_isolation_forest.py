from pathlib import Path
import sys
import numpy as np

# Ensure project root is in Python path
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
  sys.path.append(str(project_root))

from src.config import CHANNELS, DATA_DIR
from src.isolation_forest_model import PerChannelAnomalyTriage


def train_and_evaluate_stage1():
  # 1. Load pre-computed clean training matrix
  stage1_data_path = DATA_DIR / "X_stage1_train.npy"
  if not stage1_data_path.exists():
    raise FileNotFoundError(
        f"Missing {stage1_data_path}! Run 'python -m src.build_dataset' first."
    )

  X_train = np.load(stage1_data_path)
  print(f"Loaded Clean Baseline Matrix: {X_train.shape} (Windows, Features)")

  # 2. Initialize and Train Per-Channel & Physics Isolation Forests
  triage_model = PerChannelAnomalyTriage(
      channel_names=CHANNELS,
      n_estimators=200,
      max_samples=1024,
      max_features=0.85,  # <-- Explicitly enforce subspace diversity!
      random_state=42,
  )
  triage_model.fit(X_train)

  # 3. Score training data and show distributions across ALL 6 MONITORED ENTITIES!
  print("\nCalculating clean baseline score distributions...")
  scores = triage_model.score_per_channel(X_train)
  for entity in triage_model.monitored_entities:  # <-- UPGRADED: Includes Physics!
    ch_scores = scores[entity]
    print(
        f" -> {entity:<10}:  Mean={np.mean(ch_scores):.4f}  "
        f"P95={np.percentile(ch_scores, 95):.4f}  "
        f"P99={np.percentile(ch_scores, 99):.4f}  "
        f"Max={np.max(ch_scores):.4f}"
    )

  # 4. Calibrate per-channel thresholds at 1.5% FPR for a Stage 1 Recall boost
  triage_model.calibrate_thresholds(X_train, target_fpr_per_channel=0.015)

  # 5. Save model to disk
  model_save_path = Path("models") / "stage1_iforest.pkl"
  triage_model.save_model(model_save_path)
  print("\n=== STAGE 1 PER-CHANNEL & PHYSICS TRAINING COMPLETE ===")


if __name__ == "__main__":
  train_and_evaluate_stage1()