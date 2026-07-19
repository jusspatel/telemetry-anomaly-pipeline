import sys
import time
import torch
import numpy as np
from pathlib import Path
from numpy.lib.stride_tricks import sliding_window_view

# Ensure project root is in Python path
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

from src.config import CHANNELS, WINDOW_SIZE
from src.ingestion import load_clean_session_laps
from src.preprocessing import process_all_laps
from src.evaluate_pipeline import TelemetryAnomalyOrchestrator

def run_live_inference():
    print("\n" + "="*60)
    print("=== LIVE F1 TELEMETRY INFERENCE SIMULATOR ===")
    print("="*60)

    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")

    # 1. Initialize Master Orchestrator
    stage1_path = Path("models") / "stage1_iforest.pkl"
    stage2_path = Path("models") / "stage2_tcn_ae.pth"
    print("\n[SYSTEM] Booting AI Diagnostics Engine...")
    
    try:
        orchestrator = TelemetryAnomalyOrchestrator(stage1_path, stage2_path, device)
    except FileNotFoundError:
        print("ERROR: Pipeline models not found! Run the training scripts first.")
        return

    # 2. Load Unseen Track Data (Melbourne is NOT in the training set!)
    print("\n[SYSTEM] Connecting to live car telemetry stream...")
    print("         Track: Melbourne (Albert Park), Driver: VER")
    try:
        # We load a single lap of Qualifying at Melbourne
        raw_laps = load_clean_session_laps(year=2023, gp='Melbourne', session_type='Q', driver='VER')
        if not raw_laps:
            print("ERROR: No laps found. Please check network/cache.")
            return
        # Process and resample to 10Hz
        clean_laps = process_all_laps([raw_laps[0]]) 
        live_lap = clean_laps[0]
    except Exception as e:
        print(f"ERROR loading telemetry: {e}")
        return

    # Extract raw numpy arrays
    raw_matrix = live_lap[CHANNELS].values
    time_array = live_lap['Time_Sec'].values
    
    # 3. Inject a catastrophic failure at 40.0 seconds!
    fault_start_idx = np.searchsorted(time_array, 40.0)
    fault_duration = 20 # 2 seconds at 10Hz
    
    # We simulate the Speed sensor suddenly dying (dropping to 0 km/h) while the car is at high speed
    print("\n[WARNING] System operator has primed a SPEED SENSOR FAILURE at T=40.0s!")
    raw_matrix[fault_start_idx : fault_start_idx + fault_duration, 0] = 0.0
    
    # Extract sliding windows for streaming
    windows = sliding_window_view(raw_matrix, window_shape=WINDOW_SIZE, axis=0)
    
    # 4. Calibrate thresholds (optional, but good practice for an unseen track)
    # Using the first 20 seconds of the lap as "clean calibration data"
    calib_end_idx = np.searchsorted(time_array, 20.0)
    calib_windows = sliding_window_view(raw_matrix[:calib_end_idx], window_shape=WINDOW_SIZE, axis=0)
    orchestrator.calibrate_from_clean_windows(calib_windows, target_fpr=0.01)

    print("\n" + "="*60)
    print("="*60 + "\n")
    
    # 5. Start Live Streaming Loop
    time.sleep(2) # Dramatic pause
    
    for w_idx in range(len(windows)):
        current_time = time_array[w_idx + WINDOW_SIZE - 1]
        current_window = windows[w_idx]
        
        # Get the latest raw values for UI display
        latest_vals = current_window[:, -1] 
        speed, rpm, throttle, brake, gear = latest_vals
        
        # 🏎️ RUN INFERENCE ON THIS WINDOW
        result = orchestrator.evaluate_window_streaming(current_window)
        
        # Format the UI
        ui_string = f"[{current_time:05.1f}s] SPD: {int(speed):3} | RPM: {int(rpm):5} | THR: {int(throttle):3} | BRK: {int(brake):3} | GR: {int(gear)}  --> "
        
        if result['triage_alert']:
            ui_string += f"ALARM! (Culprit: {result['diagnosed_culprit']})"
            sys.stdout.write('\r' + ui_string + '\n')
            
            # Print diagnostic breakdown
            print(f"      >> Stage 2 Breakdown: ", end="")
            for ch, score in result['channel_residuals'].items():
                print(f"{ch}: {score:.2f} | ", end="")
            print("\n")
            time.sleep(0.5) # Pause briefly when an anomaly is detected so we can read it
        else:
            ui_string += f" NORMAL "
            sys.stdout.write('\r' + ui_string)
            sys.stdout.flush()
            
        # Simulate real-time 10Hz streaming (0.1s per step), but slightly faster so we don't wait forever
        time.sleep(0.02) 

    print("\n\n" + "="*60)
    print("="*60)

if __name__ == "__main__":
    run_live_inference()
