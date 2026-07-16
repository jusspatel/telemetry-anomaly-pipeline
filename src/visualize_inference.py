import sys
import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Ensure project root is in Python path
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.autoencoder_model import TCNAutoencoder
from src.fault_injection import TelemetryFaultInjector
from src.config import CHANNELS, DATA_DIR

def visualize_inference():
    print("Loading model and data for visualization...")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    
    # Load model payload
    model_path = project_root / "models" / "stage2_tcn_ae.pth"
    if not model_path.exists():
        print(f"Model not found at {model_path}. Please train it first.")
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
    
    # Load clean data
    data_path = DATA_DIR / "X_stage2_train.npy"
    clean_data = np.load(data_path)
    
    # Randomly select 4 clean windows
    rng = np.random.default_rng(42)
    indices = rng.choice(len(clean_data), 4, replace=False)
    clean_windows = clean_data[indices]
    
    # Z-score scale the clean windows
    clean_windows_scaled = (clean_windows - means) / stds
    
    # Initialize fault injector
    injector = TelemetryFaultInjector(seed=42)
    fault_types = ['dropout', 'stuck_value', 'drift', 'noise']
    
    # Inject one of each fault type into the 4 windows
    corrupted_windows = clean_windows_scaled.copy()
    labels = []
    
    for i, f_type in enumerate(fault_types):
        ch_idx = i % len(CHANNELS)
        labels.append((f_type, CHANNELS[ch_idx]))
        series = corrupted_windows[i, ch_idx, :].copy()
        duration = len(series)
        
        if f_type == 'dropout':
            series = injector.inject_dropout(series, 0, duration)
        elif f_type == 'stuck_value':
            series = injector.inject_stuck_value(series, 0, duration)
        elif f_type == 'drift':
            series = injector.inject_drift(series, 0, duration, CHANNELS[ch_idx])
        else:
            series = injector.inject_noise_burst(series, 0, duration)
            
        corrupted_windows[i, ch_idx, :] = series
        
    # Run Inference
    with torch.no_grad():
        tensor_in = torch.tensor(corrupted_windows, dtype=torch.float32).to(device)
        reconstructed, latent, fault_logits = model(tensor_in)
        
        # Calculate error
        error_tensor = torch.abs(tensor_in - reconstructed).cpu().numpy()
        reconstructed = reconstructed.cpu().numpy()
        fault_logits = fault_logits.cpu().numpy()
        probs = torch.softmax(torch.tensor(fault_logits), dim=1).numpy()

    # Plotting
    fig, axes = plt.subplots(4, 4, figsize=(20, 16))
    fig.suptitle("Stage 2 TCN Inference Visualization", fontsize=20, fontweight='bold')
    
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
    
    for i in range(4):
        true_fault_type, true_fault_ch = labels[i]
        true_ch_idx = CHANNELS.index(true_fault_ch)
        
        # Row 1: Raw Signal (Corrupted)
        ax_raw = axes[0, i]
        for c in range(5):
            lw = 3 if c == true_ch_idx else 1
            alpha = 1.0 if c == true_ch_idx else 0.3
            ax_raw.plot(corrupted_windows[i, c, :], label=CHANNELS[c], color=colors[c], linewidth=lw, alpha=alpha)
        ax_raw.set_title(f"Input ({true_fault_type.upper()} on {true_fault_ch})")
        if i == 0: ax_raw.set_ylabel("Z-Score Amplitude")
        if i == 3: ax_raw.legend(loc='upper right', fontsize='small')
        
        # Row 2: Reconstructed
        ax_recon = axes[1, i]
        for c in range(5):
            lw = 3 if c == true_ch_idx else 1
            alpha = 1.0 if c == true_ch_idx else 0.3
            ax_recon.plot(reconstructed[i, c, :], color=colors[c], linewidth=lw, alpha=alpha)
        ax_recon.set_title("TCN Reconstruction")
        if i == 0: ax_recon.set_ylabel("Z-Score Amplitude")
        
        # Row 3: Absolute Error
        ax_err = axes[2, i]
        for c in range(5):
            lw = 3 if c == true_ch_idx else 1
            alpha = 1.0 if c == true_ch_idx else 0.3
            ax_err.plot(error_tensor[i, c, :], color=colors[c], linewidth=lw, alpha=alpha)
        ax_err.set_title("Absolute Error |X - X_hat|")
        if i == 0: ax_err.set_ylabel("Error Magnitude")
        
        # Row 4: Fault Logits / Probs
        ax_prob = axes[3, i]
        bars = ax_prob.bar(CHANNELS, probs[i] * 100, color=colors)
        ax_prob.set_ylim(0, 100)
        ax_prob.set_title(f"Model Prediction")
        if i == 0: ax_prob.set_ylabel("Confidence (%)")
        
        # Highlight highest probability
        pred_idx = np.argmax(probs[i])
        bars[pred_idx].set_edgecolor('black')
        bars[pred_idx].set_linewidth(2)
        
        # Indicate if correct
        is_correct = pred_idx == true_ch_idx
        result_text = "CORRECT" if is_correct else "INCORRECT"
        result_color = 'green' if is_correct else 'red'
        ax_prob.text(0.5, 0.8, result_text, transform=ax_prob.transAxes, ha='center', color=result_color, fontweight='bold', fontsize=14)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    
    out_path = project_root / "exploration" / "inference_visualization.png"
    out_path.parent.mkdir(exist_ok=True)
    plt.savefig(out_path, dpi=150)
    print(f"Visualization saved to {out_path}")

if __name__ == "__main__":
    visualize_inference()
