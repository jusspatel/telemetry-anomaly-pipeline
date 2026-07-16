# src/calibrate_stage2_difficulty.py
import sys
from pathlib import Path
import numpy as np
import torch

# Ensure project root is in Python path
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

from src.config import DATA_DIR, CHANNELS
from src.autoencoder_model import TCNAutoencoder

def calibrate_clean_baseline_difficulty():
    print("=== STARTING STAGE 2 BASELINE DIFFICULTY CALIBRATION ===")
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    
    # 1. Load Pre-Computed Clean Training Matrix (X_stage2_train.npy)
    stage2_data_path = DATA_DIR / "X_stage2_train.npy"
    if not stage2_data_path.exists():
        raise FileNotFoundError(f"Missing {stage2_data_path}! Run 'python -m src.build_dataset' first.")
        
    clean_windows = np.load(stage2_data_path)
    print(f"Loaded Clean Baseline Matrix: {clean_windows.shape} (Windows, Channels, Time)")
    
    # 2. Load Trained Stage 2 TCN Autoencoder & Normalizers
    stage2_model_path = Path("models") / "stage2_tcn_ae.pth"
    if not stage2_model_path.exists():
        raise FileNotFoundError(f"Missing {stage2_model_path}! Run 'python -m src.train_autoencoder' first.")
        
    payload = torch.load(stage2_model_path, map_location=device, weights_only=False)
    
    config = payload['architecture_config']
    tcn = TCNAutoencoder(
        num_channels=config['num_channels'],
        latent_dim=config['latent_dim'],
        kernel_size=config['kernel_size']
    ).to(device)
    
    tcn.load_state_dict(payload['model_state_dict'])
    tcn.eval()
    
    means = payload['channel_means']  # Shape: (1, 5, 1)
    stds = payload['channel_stds']    # Shape: (1, 5, 1)
    
    # 3. Batch Process Clean Windows in Z-Score Space
    print("Streaming clean windows through TCN Autoencoder to map natural noise ceilings...")
    
    # Normalize input into pure Z-score space
    scaled_clean = (clean_windows - means) / stds
    
    # To prevent GPU out-of-memory on 16k windows, process in chunks of 1024
    batch_size = 1024
    all_peak_errors = []
    
    with torch.no_grad():
        for i in range(0, len(scaled_clean), batch_size):
            batch_slice = scaled_clean[i : i + batch_size]
            tensor_in = torch.tensor(batch_slice, dtype=torch.float32).to(device)
            
            # Reconstruct
            reconstructed_tensor, _, _ = tcn(tensor_in)
            reconstructed_scaled = reconstructed_tensor.cpu().numpy()
            
            # Calculate absolute Z-score errors for this batch: Shape (Batch, 5, 20)
            abs_errors = np.abs(batch_slice - reconstructed_scaled)
            
            # Extract peak error across time (axis=-1) using 95th percentile
# CHANGE THIS: Don't take the 95th percentile across time! Take the Mean Absolute Error!
            # abs_errors shape: (Batch, 5, 20)
            mean_errors = np.mean(abs_errors, axis=-1)  # Shape: (Batch, 5)
            all_peak_errors.append(mean_errors)

    master_mean_errors = np.concatenate(all_peak_errors, axis=0)

    # Calculate the TRUE expected baseline mean across the entire clean dataset!
    clean_baseline_difficulty = np.mean(master_mean_errors, axis=0)  # Shape: (5,)# Shape: (5,)
    
    print("\n--- CALIBRATED CLEAN BASELINE DIFFICULTY VECTOR ---")
    for idx, ch in enumerate(CHANNELS):
        print(f" -> {ch:<10}: {clean_baseline_difficulty[idx]:.4f} Z-scores")
        
    # 5. Save the vector to disk for evaluate_pipeline.py to consume
    save_path = Path("models") / "clean_baseline_difficulty.npy"
    np.save(save_path, clean_baseline_difficulty)
    print(f"\nSuccessfully saved calibrated vector to: {save_path}")
    print("=== CALIBRATION COMPLETE ===")

if __name__ == "__main__":
    calibrate_clean_baseline_difficulty()