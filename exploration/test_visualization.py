import matplotlib.pyplot as plt
import numpy as np
import sys
from pathlib import Path


project_root = Path(__file__).resolve().parents[1] 
sys.path.append(str(project_root))
from src.config import CHANNELS, TARGET_FREQ_HZ
from test_pipeline import resampled_laps , X_train_TCN

def visualize_telemetry_and_window(resampled_laps, X_stage2, lap_idx=0, window_idx=100):
    """
    Plots a full resampled lap and zooms in on a single 2-second ML tensor window.
    """
    # Set style for a clean, modern dashboard look
    plt.style.use('seaborn-v0_8-darkgrid' if 'seaborn-v0_8-darkgrid' in plt.style.available else 'default')
    
    # =========================================================================
    # PLOT 1: The Full Lap "Pit Wall" Telemetry View
    # =========================================================================
    lap_df = resampled_laps[lap_idx]
    time_sec = lap_df['Time_Sec'].values
    
    fig, axes = plt.subplots(len(CHANNELS), 1, figsize=(14, 10), sharex=True)
    fig.suptitle(f"Continuous 10Hz Resampled Telemetry (Lap {lap_idx + 1})", fontsize=16, fontweight='bold')
    
    # Colors for different F1 channels
    colors = ['#1f77b4', '#d62728', '#2ca02c', '#ff7f0e', '#9467bd']
    
    for i, channel in enumerate(CHANNELS):
        ax = axes[i]
        ax.plot(time_sec, lap_df[channel], label=channel, color=colors[i % len(colors)], linewidth=1.5)
        ax.set_ylabel(channel, fontweight='bold')
        ax.legend(loc="upper right")
        
        # Highlight where our sample 2-second window comes from (e.g., around window_idx)
        # Each window steps by STEP_SIZE timestamps (from config)
        sample_start_time = time_sec[0] + (window_idx * 0.5) # Approximate mapping for visual highlighting
        ax.axvspan(sample_start_time, sample_start_time + 2.0, color='red', alpha=0.2, label="ML Window" if i==0 else "")
        
    axes[-1].set_xlabel("Time (Seconds)", fontweight='bold', fontsize=12)
    plt.tight_layout()
    plt.show()

    # =========================================================================
    # PLOT 2: What the Autoencoder Actually Sees (Single 2-Second Tensor)
    # =========================================================================
    # Check shape: PyTorch tensors are (Batch, Channels, Time), some NumPy setups are (Batch, Time, Channels)
    sample_window = X_stage2[window_idx]
    
    # If shape is (Channels, Time) i.e. (5, 20), we transpose for easier plotting to (20, 5)
    if sample_window.shape[0] == len(CHANNELS):
        sample_window = sample_window.T 
        
    window_time_grid = np.arange(0, 2.0, 1.0 / TARGET_FREQ_HZ) # 0.0s to 1.9s (20 timestamps)
    
    fig, ax = plt.subplots(figsize=(10, 5))
    fig.suptitle(f"Stage 2 Autoencoder Input Tensor (Window Index: {window_idx})", fontsize=14, fontweight='bold')
    
    for i, channel in enumerate(CHANNELS):
        # Normalize the channel just for visual comparison on a single Y-axis
        raw_vals = sample_window[:, i]
        norm_vals = (raw_vals - np.min(raw_vals)) / (np.max(raw_vals) - np.min(raw_vals) + 1e-8)
        
        ax.plot(window_time_grid, norm_vals, marker='o', label=f"{channel} (Normalized)", color=colors[i % len(colors)], linewidth=2)
        
    ax.set_title("20 Timestamps of Simultaneous Multi-Sensor Dynamics (Normalized to 0-1 Scale)", fontsize=11, fontstyle='italic')
    ax.set_xlabel("Window Elapsed Time (Seconds)", fontweight='bold')
    ax.set_ylabel("Normalized Amplitude", fontweight='bold')
    ax.set_xticks(window_time_grid)
    ax.legend(loc="upper left", bbox_to_anchor=(1, 1))
    plt.tight_layout()
    plt.show()

# Run the visualization on your generated data!
visualize_telemetry_and_window(resampled_laps, X_train_TCN, lap_idx=0, window_idx=50)