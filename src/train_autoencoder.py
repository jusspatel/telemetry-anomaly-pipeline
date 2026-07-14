# src/train_autoencoder.py
import sys
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

# Ensure project root is in Python path
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

from src.config import DATA_DIR
from src.autoencoder_model import TCNAutoencoder

def train_stage2_autoencoder():
    print("=== STARTING STAGE 2: TCN AUTOENCODER TRAINING ===")
    
    # 1. Hardware Selection (GPU if available, else CPU)
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Training on device: {device}")
    
    # 2. Load Pre-Computed Stage 2 Tensor Data
    stage2_data_path = DATA_DIR / "X_stage2_train.npy"
    if not stage2_data_path.exists():
        raise FileNotFoundError(f"Missing {stage2_data_path}! Run 'python -m src.build_dataset' first.")
        
    # Shape: (Batch, Channels=5, Time=20)
    X_train_raw = np.load(stage2_data_path)
    print(f"Loaded Raw Tensor Matrix: {X_train_raw.shape}")
    
    # 3. Channel-Wise Normalization (CRITICAL STEP)
    # We must scale each channel independently so RPM (12000) doesn't dominate Brake (1.0).
    # We calculate Mean and Std across the Batch (axis=0) and Time (axis=2) dimensions.
    channel_means = np.mean(X_train_raw, axis=(0, 2), keepdims=True)
    channel_stds = np.std(X_train_raw, axis=(0, 2), keepdims=True)
    channel_stds[channel_stds == 0] = 1e-8  # Prevent division by zero for flat channels
    
    X_train_scaled = (X_train_raw - channel_means) / channel_stds
    print("Dataset Normalized successfully (Zero Mean, Unit Variance per Channel).")
    
    # 4. Create PyTorch DataLoader
    # Autoencoders are self-supervised: the input IS the target (X, X)
    tensor_x = torch.tensor(X_train_scaled, dtype=torch.float32)
    dataset = TensorDataset(tensor_x, tensor_x) 
    
    batch_size = 256
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=False)
    
    # 5. Initialize the Dilated Causal TCN Autoencoder
    model = TCNAutoencoder(num_channels=5, latent_dim=3, kernel_size=3).to(device)
    
    # Optimizer and Loss Function
    # We use Mean Squared Error (MSE) to penalize large reconstruction failures
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
    criterion = nn.HuberLoss(delta=1.0)
    
    epochs = 40
    print(f"\nBeginning training loop for {epochs} epochs...")
    
    # 6. The Training Loop
    model.train()
    for epoch in range(1, epochs + 1):
        epoch_loss = 0.0
        
        for batch_x, target_x in dataloader:
            batch_x = batch_x.to(device)
            target_x = target_x.to(device)
            
            # Forward pass
            optimizer.zero_grad()
            reconstructed, latent = model(batch_x)
            
            # Calculate loss (Reconstruction vs Original)
            loss = criterion(reconstructed, target_x)
            
            # Backward pass & Optimize
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item() * batch_x.size(0)
            
        avg_loss = epoch_loss / len(dataset)
        
        # Print progress every 5 epochs
        if epoch % 5 == 0 or epoch == 1:
            print(f" -> Epoch [{epoch}/{epochs}] | Train Loss (MSE): {avg_loss:.6f}")
            
    print("\nTraining Converged! Reconstructions are tight.")
    
    # 7. Save the Model AND the Scaling Parameters
    model_save_path = Path("models") / "stage2_tcn_ae.pth"
    model_save_path.parent.mkdir(parents=True, exist_ok=True)
    
    # We save a dictionary containing weights + normalizers so inference knows how to scale data
    save_payload = {
        'model_state_dict': model.state_dict(),
        'channel_means': channel_means,
        'channel_stds': channel_stds,
        'architecture_config': {
            'num_channels': 5,
            'latent_dim': 3,
            'kernel_size': 3
        }
    }
    
    torch.save(save_payload, model_save_path)
    print(f"=== STAGE 2 MODEL SAVED: {model_save_path} ===")

if __name__ == "__main__":
    train_stage2_autoencoder()