import sys
from pathlib import Path


project_root = Path(__file__).resolve().parents[1] 
sys.path.append(str(project_root))
from src.ingestion import load_clean_session_laps
from src.preprocessing import process_all_laps
from src.features import generate_training_datasets

# 1. Pull clean laps (e.g., Verstappen at 2023 Bahrain Grand Prix Qualifying)
raw_laps = load_clean_session_laps(year=2023, gp='Bahrain', session_type='Q', driver='VER')

# 2. Linearly interpolate all laps to 10Hz
resampled_laps = process_all_laps(raw_laps)

# 3. Generate Stage 1 & Stage 2 anomaly-free training matrices
X_train_iForest, X_train_TCN = generate_training_datasets(resampled_laps)

