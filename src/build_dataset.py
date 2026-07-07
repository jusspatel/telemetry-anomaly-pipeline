# src/build_dataset.py
import sys
from pathlib import Path
import numpy as np

# Ensure project root is in Python path
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

from src.config import DATA_DIR
from src.ingestion import load_clean_session_laps
from src.preprocessing import process_all_laps
from src.features import generate_training_datasets

def build_and_save_master_dataset():
    
    # We use Qualifying ('Q') sessions because drivers push the cars to physical limits without traffic
    tracks_to_sample = ['Bahrain', 'Monza', 'Silverstone']
    drivers_to_sample = ['VER', 'PER', 'HAM', 'LEC', 'NOR']
    year = 2023
    
    master_clean_laps = []
    
    for track in tracks_to_sample:
        for driver in drivers_to_sample:
            try:
                print(f"\nHarvesting: {year} {track} - Driver: {driver}")
                laps = load_clean_session_laps(year=year, gp=track, session_type='Q', driver=driver)
                
                # Linearly interpolate to 10Hz immediately
                resampled = process_all_laps(laps)
                master_clean_laps.extend(resampled)
            except Exception as e:
                print(f" -> Could not load {driver} at {track}: {e}")
                
    print(f"\nTotal Clean Laps Harvested Across All Tracks: {len(master_clean_laps)}")
    
    # Generate the Stage 1 and Stage 2 training matrices
    print("\nExtracting sliding windows and engineering features...")
    X_stage1, X_stage2 = generate_training_datasets(master_clean_laps)
    
    # Save directly to disk as .npy files!
    stage1_path = DATA_DIR / "X_stage1_train.npy"
    stage2_path = DATA_DIR / "X_stage2_train.npy"
    
    np.save(stage1_path, X_stage1)
    np.save(stage2_path, X_stage2)
    
    print("\n--- DATASET SUCCESSFULLY SAVED TO DISK ---")
    print(f"Saved Stage 1 (iForest) Matrix: {stage1_path} | Shape: {X_stage1.shape}")
    print(f"Saved Stage 2 (TCN AE) Matrix:  {stage2_path} | Shape: {X_stage2.shape}")

if __name__ == "__main__":
    build_and_save_master_dataset()