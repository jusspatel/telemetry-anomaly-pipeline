# # src/train_autoencoder.py
# import sys
# from pathlib import Path
# import numpy as np
# import torch
# import torch.nn as nn
# import torch.optim as optim
# from torch.utils.data import TensorDataset, DataLoader

# # Ensure project root is in Python path
# project_root = Path(__file__).resolve().parent.parent
# if str(project_root) not in sys.path:
#     sys.path.append(str(project_root))

# from src.config import DATA_DIR
# from src.autoencoder_model import TCNAutoencoder

# def train_stage2_autoencoder():
#     print("=== STARTING STAGE 2: TCN AUTOENCODER TRAINING ===")
    
#     # 1. Hardware Selection (GPU if available, else CPU)
#     device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
#     print(f"Training on device: {device}")
    
#     # 2. Load Pre-Computed Stage 2 Tensor Data
#     stage2_data_path = DATA_DIR / "X_stage2_train.npy"
#     if not stage2_data_path.exists():
#         raise FileNotFoundError(f"Missing {stage2_data_path}! Run 'python -m src.build_dataset' first.")
        
#     # Shape: (Batch, Channels=5, Time=20)
#     X_train_raw = np.load(stage2_data_path)
#     print(f"Loaded Raw Tensor Matrix: {X_train_raw.shape}")
    
#     # 3. Channel-Wise Normalization (CRITICAL STEP)
#     # We must scale each channel independently so RPM (12000) doesn't dominate Brake (1.0).
#     # We calculate Mean and Std across the Batch (axis=0) and Time (axis=2) dimensions.
#     channel_means = np.mean(X_train_raw, axis=(0, 2), keepdims=True)
#     channel_stds = np.std(X_train_raw, axis=(0, 2), keepdims=True)
#     channel_stds[channel_stds == 0] = 1e-8  # Prevent division by zero for flat channels
    
#     X_train_scaled = (X_train_raw - channel_means) / channel_stds
#     print("Dataset Normalized successfully (Zero Mean, Unit Variance per Channel).")
    
#     # 4. Create PyTorch DataLoader
#     # Autoencoders are self-supervised: the input IS the target (X, X)
#     tensor_x = torch.tensor(X_train_scaled, dtype=torch.float32)
#     dataset = TensorDataset(tensor_x, tensor_x) 
    
#     batch_size = 256
#     dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=False)
    
#     # 5. Initialize the Dilated Causal TCN Autoencoder
#     model = TCNAutoencoder(num_channels=5, latent_dim=3, kernel_size=3).to(device)
    
#     # Optimizer and Loss Function
#     # We use Mean Squared Error (MSE) to penalize large reconstruction failures
#     optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
#     criterion = nn.HuberLoss(delta=1.0)
#     epochs = 40
#     print(f"\nBeginning training loop for {epochs} epochs...")
    
#     # 6. The Training Loop
#     model.train()
#     for epoch in range(1, epochs + 1):
#         epoch_loss = 0.0
        
#         for batch_x, target_x in dataloader:
#             batch_x = batch_x.to(device)
#             target_x = target_x.to(device)
            
#             # Forward pass
#             optimizer.zero_grad()
#             reconstructed, latent = model(batch_x)
            
#             # Calculate loss (Reconstruction vs Original)
#             loss = criterion(reconstructed, target_x)
            
#             # Backward pass & Optimize
#             loss.backward()
#             optimizer.step()
            
#             epoch_loss += loss.item() * batch_x.size(0)
            
#         avg_loss = epoch_loss / len(dataset)
        
#         # Print progress every 5 epochs
#         if epoch % 5 == 0 or epoch == 1:
#             print(f" -> Epoch [{epoch}/{epochs}] | Train Loss (MSE): {avg_loss:.6f}")
            
#     print("\nTraining Converged! Reconstructions are tight.")
    
#     # 7. Save the Model AND the Scaling Parameters
#     model_save_path = Path("models") / "stage2_tcn_ae.pth"
#     model_save_path.parent.mkdir(parents=True, exist_ok=True)
    
#     # We save a dictionary containing weights + normalizers so inference knows how to scale data
#     save_payload = {
#         'model_state_dict': model.state_dict(),
#         'channel_means': channel_means,
#         'channel_stds': channel_stds,
#         'architecture_config': {
#             'num_channels': 5,
#             'latent_dim': 3,
#             'kernel_size': 3
#         }
#     }
    
#     torch.save(save_payload, model_save_path)
#     print(f"=== STAGE 2 MODEL SAVED: {model_save_path} ===")

# if __name__ == "__main__":
#     train_stage2_autoencoder()
from pathlib import Path
import sys
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

# Ensure project root is in Python path
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
  sys.path.append(str(project_root))

from src.autoencoder_model import TCNAutoencoder
from src.config import DATA_DIR, CHANNELS
from src.fault_injection import TelemetryFaultInjector


def _inject_faults_for_training(
    clean_windows: np.ndarray, fault_probability: float = 0.50, seed: int = 123
) -> tuple:
  """Augments clean windows with synthetic faults for multi-task training.

  Args:
      clean_windows: Shape (N, 5, 20) — clean Z-scored windows.
      fault_probability: Fraction of windows to corrupt.
      seed: Random seed for reproducibility.

  Returns:
      augmented_windows: Shape (N, 5, 20) — mix of clean and corrupted.
      fault_labels: Shape (N,) — -1 for clean, 0-4 for fault channel index.
  """
  rng = np.random.default_rng(seed)
  injector = TelemetryFaultInjector(seed=seed)
  N = len(clean_windows)

  augmented = clean_windows.copy()
  labels = np.full(N, -1, dtype=np.int64)  # -1 = clean (no fault)

  fault_types = ['dropout', 'stuck_value', 'drift', 'noise']

  for i in range(N):
    if rng.random() < fault_probability:
      ch_idx = rng.integers(0, len(CHANNELS))
      fault_type = rng.choice(fault_types)

      # Extract the 1D time-series for the chosen channel: Shape (20,)
      series = augmented[i, ch_idx, :].copy()

      # Inject fault across the entire 20-step window
      if fault_type == 'dropout':
        series = injector.inject_dropout(series, start_idx=0, duration_idx=len(series))
      elif fault_type == 'stuck_value':
        series = injector.inject_stuck_value(series, start_idx=0, duration_idx=len(series))
      elif fault_type == 'drift':
        series = injector.inject_drift(series, start_idx=0, duration_idx=len(series), channel_name=CHANNELS[ch_idx])
      else:
        series = injector.inject_noise_burst(series, start_idx=0, duration_idx=len(series))

      augmented[i, ch_idx, :] = series
      labels[i] = ch_idx

  n_faults = np.sum(labels >= 0)
  print(f"  Fault augmentation: {n_faults}/{N} windows corrupted ({n_faults/N*100:.1f}%)")
  return augmented, labels


def train_stage2_autoencoder():
  print("=== STARTING STAGE 2: TCN AUTOENCODER TRAINING (FAULT-AWARE) ===")

  # 1. Hardware Selection
  device = torch.device(
      "cuda"
      if torch.cuda.is_available()
      else "mps" if torch.backends.mps.is_available() else "cpu"
  )
  print(f"Training on device: {device}")

  # 2. Load Pre-Computed Stage 2 Tensor Data
  stage2_data_path = DATA_DIR / "X_stage2_train.npy"
  if not stage2_data_path.exists():
    raise FileNotFoundError(
        f"Missing {stage2_data_path}! Run 'python -m src.build_dataset' first."
    )

  # Shape: (Batch, Channels=5, Time=20)
  X_train_raw = np.load(stage2_data_path)
  print(f"Loaded Raw Tensor Matrix: {X_train_raw.shape}")

  # 3. Channel-Wise Normalization
  channel_means = np.mean(X_train_raw, axis=(0, 2), keepdims=True)
  channel_stds = np.std(X_train_raw, axis=(0, 2), keepdims=True)
  channel_stds[channel_stds == 0] = 1e-8

  X_train_scaled = (X_train_raw - channel_means) / channel_stds
  print("Dataset Normalized successfully (Zero Mean, Unit Variance per Channel).")

  # 4. Fault-Aware Data Augmentation (Suggestion 6)
  print("\nInjecting synthetic faults for multi-task training...")
  X_augmented, fault_labels = _inject_faults_for_training(
      X_train_scaled, fault_probability=0.50, seed=123
  )

  # 5. Create PyTorch DataLoader with fault labels
  tensor_x_aug = torch.tensor(X_augmented, dtype=torch.float32)
  tensor_x_clean = torch.tensor(X_train_scaled, dtype=torch.float32)  # Reconstruction target is always clean!
  tensor_labels = torch.tensor(fault_labels, dtype=torch.long)

  dataset = TensorDataset(tensor_x_aug, tensor_x_clean, tensor_labels)

  batch_size = 256
  dataloader = DataLoader(
      dataset, batch_size=batch_size, shuffle=True, drop_last=False
  )

  # 6. Initialize the Dilated Causal TCN Autoencoder
  # Suggestion 5: Widened latent_dim to 6 for better channel-specific preservation
  model = TCNAutoencoder(num_channels=5, latent_dim=6, kernel_size=3).to(
      device
  )

  # Optimizer, Loss Functions, and Learning Rate Scheduler
  optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
  recon_criterion = nn.HuberLoss(delta=1.0)
  class_criterion = nn.CrossEntropyLoss()
  classification_weight = 0.5  # Safe to increase: error_head is detached from reconstruction

  # Suggestion 4: Increased epochs and lower LR floor
  epochs = 100
  scheduler = optim.lr_scheduler.CosineAnnealingLR(
      optimizer, T_max=epochs, eta_min=1e-7
  )

  print(f"\nBeginning fault-aware training loop for {epochs} epochs...")
  print(f"  Multi-task loss: L_recon + {classification_weight} * L_classify")

  # 7. The Training Loop (Multi-Task)
  model.train()
  for epoch in range(1, epochs + 1):
    epoch_recon_loss = 0.0
    epoch_class_loss = 0.0
    epoch_total_loss = 0.0

    for batch_aug, batch_clean, batch_labels in dataloader:
      batch_aug = batch_aug.to(device)
      batch_clean = batch_clean.to(device)
      batch_labels = batch_labels.to(device)

      optimizer.zero_grad()

      # Forward pass returns (reconstructed, latent, fault_logits)
      reconstructed, latent, fault_logits = model(batch_aug)

      # Reconstruction loss: compare reconstruction to CLEAN target
      loss_recon = recon_criterion(reconstructed, batch_clean)

      # Classification loss: only for fault-injected windows (label >= 0)
      fault_mask = batch_labels >= 0
      if fault_mask.any():
        loss_class = class_criterion(
            fault_logits[fault_mask], batch_labels[fault_mask]
        )
      else:
        loss_class = torch.tensor(0.0, device=device)

      total_loss = loss_recon + classification_weight * loss_class

      total_loss.backward()
      optimizer.step()

      bs = batch_aug.size(0)
      epoch_recon_loss += loss_recon.item() * bs
      epoch_class_loss += loss_class.item() * bs
      epoch_total_loss += total_loss.item() * bs

    scheduler.step()

    avg_recon = epoch_recon_loss / len(dataset)
    avg_class = epoch_class_loss / len(dataset)
    avg_total = epoch_total_loss / len(dataset)
    current_lr = scheduler.get_last_lr()[0]

    if epoch % 10 == 0 or epoch == 1:
      print(
          f" -> Epoch [{epoch:03d}/{epochs}] | Recon: {avg_recon:.6f}"
          f" | Class: {avg_class:.6f} | Total: {avg_total:.6f}"
          f" | LR: {current_lr:.7f}"
      )

  print("\nTraining Converged! Fault-aware reconstructions are tight.")

  # 8. Save the Model AND the Scaling Parameters
  model_save_path = Path("models") / "stage2_tcn_ae.pth"
  model_save_path.parent.mkdir(parents=True, exist_ok=True)

  save_payload = {
      "model_state_dict": model.state_dict(),
      "channel_means": channel_means,
      "channel_stds": channel_stds,
      "architecture_config": {
          "num_channels": 5,
          "latent_dim": 6,
          "kernel_size": 3,
      },
  }

  torch.save(save_payload, model_save_path)
  print(f"=== STAGE 2 MODEL SAVED: {model_save_path} ===")


if __name__ == "__main__":
  train_stage2_autoencoder()