import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import sys
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
  sys.path.append(str(project_root))

# Try importing seaborn for nicer heatmaps, fallback to matplotlib if missing
try:
    import seaborn as sns
    HAS_SEABORN = True
except ImportError:
    HAS_SEABORN = False

from sklearn.metrics import confusion_matrix
from src.autoencoder_model import TCNAutoencoder
from src.fault_injection import TelemetryFaultInjector, CHANNELS

def visualize_thesis_claims():
    print("Generating Thesis Visualizations...")
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    
    # 1. Load Model and Config
    model_path = Path("models") / "stage2_tcn_ae.pth"
    if not model_path.exists():
        print("Model not found. Please train it first.")
        return
        
    payload = torch.load(model_path, map_location=device, weights_only=False)
    means = payload['channel_means']
    stds = payload['channel_stds']
    config = payload['architecture_config']
    
    model = TCNAutoencoder(
        num_channels=config['num_channels'],
        latent_dim=config['latent_dim'],
        kernel_size=config['kernel_size']
    ).to(device)
    model.load_state_dict(payload['model_state_dict'])
    model.eval()

    # 2. Load Clean Test Data
    test_data_path = Path("data") / "X_stage2_train.npy"
    if not test_data_path.exists():
         print("Test data not found.")
         return
    X_clean_raw = np.load(test_data_path)
    X_clean = (X_clean_raw - means) / stds
    
    injector = TelemetryFaultInjector(seed=42)
    output_dir = Path("exploration")
    output_dir.mkdir(exist_ok=True)

    # =====================================================================
    # CLAIM 1: Denoising "Healing" Overlay
    # Prove the model recovers the lost physical ground truth
    # =====================================================================
    print("Generating Figure 1: Healing Overlay...")
    window_idx = 50
    clean_window = X_clean[window_idx].copy()
    corrupted_window = clean_window.copy()
    # Inject a severe dropout on Speed (Channel 0)
    corrupted_window[0, :] = injector.inject_dropout(corrupted_window[0, :], 5, 10)
    
    with torch.no_grad():
        x_tensor = torch.tensor(corrupted_window, dtype=torch.float32).unsqueeze(0).to(device)
        recon_tensor, _, _ = model(x_tensor)
        recon_window = recon_tensor.squeeze(0).cpu().numpy()

    plt.figure(figsize=(10, 6))
    plt.plot(clean_window[0], 'k--', label='Clean Ground Truth', linewidth=2)
    plt.plot(corrupted_window[0], 'r-', label='Corrupted Input (Dropout)', alpha=0.5, linewidth=2)
    plt.plot(recon_window[0], 'b-', label='TCN Reconstruction', linewidth=2)
    plt.title("Denoising Proof: Recovering Lost Telemetry", fontsize=16)
    plt.xlabel("Time Step (20-step window)", fontsize=12)
    plt.ylabel("Z-Score Amplitude (Speed)", fontsize=12)
    plt.legend(fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / "claim_1_healing_overlay.png", dpi=300)
    plt.close()

    # =====================================================================
    # CLAIM 2: 3D Latent Space Projection
    # Prove the bottleneck forms a healthy manifold
    # =====================================================================
    print("Generating Figure 2: 3D Latent Space...")
    num_samples = min(2000, len(X_clean))
    X_subset_clean = X_clean[:num_samples].copy()
    X_subset_corrupt = X_clean[:num_samples].copy()
    
    labels = []
    rng = np.random.default_rng(42)
    for i in range(num_samples):
        if rng.random() < 0.5:
            # Keep clean
            labels.append(-1)
        else:
            # Corrupt
            ch_idx = rng.integers(0, 5)
            X_subset_corrupt[i, ch_idx, :] = injector.inject_stuck_value(X_subset_corrupt[i, ch_idx, :], 0, 20)
            labels.append(ch_idx)
            
    with torch.no_grad():
        tensor_in = torch.tensor(X_subset_corrupt, dtype=torch.float32).to(device)
        _, latent_out, _ = model(tensor_in)
        # Latent shape: (N, 3, 20). Take the mean over time to get a single 3D point per window
        latent_points = latent_out.mean(dim=2).cpu().numpy()
        
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    labels = np.array(labels)
    
    # Plot clean data
    clean_mask = labels == -1
    ax.scatter(latent_points[clean_mask, 0], latent_points[clean_mask, 1], latent_points[clean_mask, 2], 
               c='blue', label='Healthy Driving', alpha=0.6, s=20)
    
    # Plot anomalies
    anomaly_mask = labels >= 0
    ax.scatter(latent_points[anomaly_mask, 0], latent_points[anomaly_mask, 1], latent_points[anomaly_mask, 2], 
               c='red', label='Sensor Anomalies', alpha=0.6, s=20)
               
    ax.set_title("3D Latent Space: The Healthy Manifold", fontsize=16)
    ax.set_xlabel("Latent Neuron 1")
    ax.set_ylabel("Latent Neuron 2")
    ax.set_zlabel("Latent Neuron 3")
    ax.legend(fontsize=12)
    plt.savefig(output_dir / "claim_2_latent_space.png", dpi=300)
    plt.close()

    # =====================================================================
    # CLAIM 3: Diagnostic Confusion Matrix
    # Prove the Multi-Stat head eliminates cross-talk
    # =====================================================================
    print("Generating Figure 3: Confusion Matrix...")
    # Evaluate only on the corrupted ones
    corrupt_idx = labels >= 0
    X_eval = X_subset_corrupt[corrupt_idx]
    y_true = labels[corrupt_idx]
    
    with torch.no_grad():
        tensor_eval = torch.tensor(X_eval, dtype=torch.float32).to(device)
        _, _, logits = model(tensor_eval)
        y_pred = torch.argmax(logits, dim=1).cpu().numpy()
        
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1, 2, 3, 4])
    
    plt.figure(figsize=(8, 6))
    if HAS_SEABORN:
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=CHANNELS, yticklabels=CHANNELS)
    else:
        plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
        plt.colorbar()
        tick_marks = np.arange(len(CHANNELS))
        plt.xticks(tick_marks, CHANNELS, rotation=45)
        plt.yticks(tick_marks, CHANNELS)
        for i in range(5):
            for j in range(5):
                plt.text(j, i, format(cm[i, j], 'd'), horizontalalignment="center", color="white" if cm[i, j] > cm.max()/2 else "black")
                
    plt.title("Stage 2 Fault Attribution: Confusion Matrix", fontsize=16)
    plt.ylabel('True Broken Sensor', fontsize=12)
    plt.xlabel('Predicted Broken Sensor', fontsize=12)
    plt.tight_layout()
    plt.savefig(output_dir / "claim_3_confusion_matrix.png", dpi=300)
    plt.close()

    # =====================================================================
    # CLAIM 4: Error Tensor Heatmap
    # Visually explain how the model localizes the fault
    # =====================================================================
    print("Generating Figure 4: Error Tensor Heatmap...")
    # Using the corrupted window from Claim 1 (Speed Dropout)
    abs_error = np.abs(corrupted_window - recon_window)
    
    plt.figure(figsize=(10, 4))
    if HAS_SEABORN:
        sns.heatmap(abs_error, cmap='Reds', yticklabels=CHANNELS, cbar_kws={'label': 'Absolute Error Magnitude'})
    else:
        plt.imshow(abs_error, aspect='auto', cmap='Reds')
        plt.colorbar(label='Absolute Error Magnitude')
        plt.yticks(np.arange(len(CHANNELS)), CHANNELS)
        
    plt.title("Error Tensor Hotspot (Speed Sensor Dropout)", fontsize=16)
    plt.xlabel("Time Step (20-step window)", fontsize=12)
    plt.ylabel("Sensor Channel", fontsize=12)
    plt.tight_layout()
    plt.savefig(output_dir / "claim_4_error_heatmap.png", dpi=300)
    plt.close()

    print(f"All visualizations saved successfully to {output_dir.absolute()}!")

if __name__ == "__main__":
    visualize_thesis_claims()
