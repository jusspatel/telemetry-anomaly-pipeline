import sys
import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from numpy.lib.stride_tricks import sliding_window_view
import seaborn as sns

# Ensure project root is in Python path
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

from src.config import CHANNELS, WINDOW_SIZE
from src.ingestion import load_clean_session_laps
from src.preprocessing import process_all_laps
from src.evaluate_pipeline import TelemetryAnomalyOrchestrator

def run_visual_live_inference():
    print("Generating Live Inference Visualization...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    stage1_path = Path("models") / "stage1_iforest.pkl"
    stage2_path = Path("models") / "stage2_tcn_ae.pth"
    orchestrator = TelemetryAnomalyOrchestrator(stage1_path, stage2_path, device)

    # Load Melbourne qualifying lap
    raw_laps = load_clean_session_laps(year=2023, gp='Melbourne', session_type='Q', driver='VER')
    clean_laps = process_all_laps([raw_laps[0]]) 
    live_lap = clean_laps[0]

    raw_matrix = live_lap[CHANNELS].values.copy()
    time_array = live_lap['Time_Sec'].values
    
    # Inject fault at T=40.0s for 3 seconds (Dropout on Speed)
    fault_start_idx = np.searchsorted(time_array, 40.0)
    fault_duration = 30 # 3 seconds at 10Hz
    raw_matrix[fault_start_idx : fault_start_idx + fault_duration, 0] = 0.0 # Speed drops to 0
    
    windows = sliding_window_view(raw_matrix, window_shape=WINDOW_SIZE, axis=0)
    
    # Calibrate on first 20 seconds
    calib_end_idx = np.searchsorted(time_array, 20.0)
    calib_windows = sliding_window_view(raw_matrix[:calib_end_idx], window_shape=WINDOW_SIZE, axis=0)
    orchestrator.calibrate_from_clean_windows(calib_windows, target_fpr=0.005)

    # Storage for plotting
    plot_times = []
    alerts = []
    diagnoses = []
    
    # Run through the whole lap
    for w_idx in range(len(windows)):
        current_time = time_array[w_idx + WINDOW_SIZE - 1]
        current_window = windows[w_idx]
        
        result = orchestrator.evaluate_window_streaming(current_window)
        
        plot_times.append(current_time)
        alerts.append(1 if result['triage_alert'] else 0)
        
        if result['triage_alert']:
            diagnoses.append(result['diagnosed_culprit'])
        else:
            diagnoses.append("None")

    # Offset to align plot times with raw matrix
    plot_times = np.array(plot_times)
    
    # Setup the plot
    plt.style.use('dark_background')
    fig, (ax1, ax_gear, ax_gear_zoomed, ax2, ax3) = plt.subplots(5, 1, figsize=(15, 17), gridspec_kw={'height_ratios': [3, 1, 1, 1, 1]})
    
    # Plot 1: Sensor Data (Speed, RPM, Brake)
    ax1.plot(time_array, raw_matrix[:, 0], color='cyan', label='Speed (km/h)', linewidth=2)
    ax1.plot(time_array, raw_matrix[:, 1] / 100, color='orange', label='RPM (x100)', alpha=0.5)
    ax1.plot(time_array, raw_matrix[:, 3] * 100, color='red', label='Brake', alpha=0.5)
    
    # Highlight the true fault region
    ax1.axvspan(40.0, 43.0, color='white', alpha=0.2, label='True Injected Fault (Speed=0)')
    ax1.axvspan(56.8, 57.5, color='yellow', alpha=0.1, label='FastF1 Interpolation Glitch')
    ax1.set_ylabel("Sensor Values")
    ax1.set_title("Live F1 Telemetry Stream (Melbourne)", fontsize=14)
    ax1.legend(loc='upper right')
    ax1.grid(True, alpha=0.2)

    # Plot 1.5: Dedicated Gear Plot (Wide)
    ax_gear.plot(time_array, raw_matrix[:, 4], color='purple', label='Gear (Full View)', linewidth=2, linestyle='--')
    ax_gear.set_ylabel("Gear")
    ax_gear.legend(loc='upper right')
    ax_gear.grid(True, alpha=0.2)
    
    # Plot 1.75: Dedicated Gear Plot (Zoomed 55-60)
    ax_gear_zoomed.plot(time_array, raw_matrix[:, 4], color='magenta', label='Gear (ZOOMED 55s-60s)', linewidth=2, linestyle='-')
    ax_gear_zoomed.set_ylabel("Gear (Zoomed)")
    ax_gear_zoomed.legend(loc='upper right')
    ax_gear_zoomed.grid(True, alpha=0.2)
    ax_gear_zoomed.set_xlim(55.0, 60.0) # ONLY THIS ONE IS ZOOMED
    
    # Plot 2: Stage 1 Alarms
    ax2.plot(plot_times, alerts, color='red', drawstyle='steps-pre', linewidth=2)
    ax2.fill_between(plot_times, 0, alerts, color='red', alpha=0.3, step='pre')
    ax2.set_ylabel("Stage 1 Alert")
    ax2.set_yticks([0, 1])
    ax2.set_yticklabels(['Normal', 'ANOMALY'])
    ax2.grid(True, alpha=0.2)

    # Plot 3: Stage 2 Diagnoses
    ch_y = {ch: i for i, ch in enumerate(CHANNELS)}
    y_vals = []
    x_vals = []
    colors = []
    
    for t, diag in zip(plot_times, diagnoses):
        if diag != "None":
            x_vals.append(t)
            y_vals.append(ch_y[diag])
            colors.append('red' if diag == 'Speed' else 'yellow')

    if x_vals:
        ax3.scatter(x_vals, y_vals, c=colors, s=50, marker='x')
        
    ax3.set_yticks(range(len(CHANNELS)))
    ax3.set_yticklabels(CHANNELS)
    ax3.set_ylabel("Stage 2 Diagnosis")
    ax3.set_xlabel("Time (Seconds)")
    ax3.grid(True, alpha=0.2)
    ax3.set_ylim(-0.5, 4.5)

    # Align X-axes (Restore to 30-60s wide view for everything except the zoomed plot)
    for ax in [ax1, ax_gear, ax2, ax3]:
        ax.set_xlim(30.0, 60.0)

    plt.tight_layout()
    output_path = Path("exploration") / "live_inference_timeline.png"
    plt.savefig(output_path, dpi=300)
    plt.close()
    
    print(f"Visualization saved to {output_path}")

if __name__ == "__main__":
    run_visual_live_inference()
