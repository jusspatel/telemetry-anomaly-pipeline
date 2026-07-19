import sys
from pathlib import Path
import numpy as np
import torch

project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

from src.config import DATA_DIR, CHANNELS, WINDOW_SIZE
from src.evaluate_pipeline import TelemetryAnomalyOrchestrator
from src.ingestion import load_clean_session_laps
from src.preprocessing import process_all_laps

def load_track_data(track_name):
    # Mimic the load_track_data from app.py
    raw_laps = load_clean_session_laps(year=2023, gp=track_name, session_type='Q', driver='VER')
    clean_laps = process_all_laps([raw_laps[0]]) 
    return clean_laps[0]

def main():
    print("Loading Orchestrator...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    stage1_path = project_root / "models" / "stage1_iforest.pkl"
    stage2_path = project_root / "models" / "stage2_tcn_ae.pth"
    orchestrator = TelemetryAnomalyOrchestrator(stage1_path, stage2_path, device)
    orchestrator.tcn.eval()
    
    tracks = ["Melbourne", "Jeddah", "Miami"]
    fault_types = ['Dropout', 'Stuck Value', 'Drift', 'Noise']
    
    # 20.0 to 74.0 is 55 steps (1.0 step size)
    sim_times = np.arange(20.0, 75.0, 1.0)
    
    total_combinations_per_track = len(sim_times) * len(CHANNELS) * len(fault_types)
    total_combinations_all = total_combinations_per_track * len(tracks)
    print(f"Starting sweep of {total_combinations_all} total combinations across {len(tracks)} tracks...")
    
    # Ensure random seed is fixed for deterministic noise
    np.random.seed(42)
    
    # We will save the results in the exploration folder
    output_path = project_root / "exploration" / "diagnostic_sweep_results.txt"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    overall_caught = 0
    overall_missed = 0
    
    with open(output_path, "w", encoding="utf-8") as f:
        f.write("=== MULTI-TRACK DIAGNOSTIC ENGINE SWEEP ===\n\n")
        
        for track in tracks:
            print(f"--> Sweeping {track}...")
            f.write(f"--- TRACK: {track} ---\n")
            
            live_lap = load_track_data(track)
            raw_matrix = live_lap[CHANNELS].values
            time_array = live_lap['Time_Sec'].values
            
            caught = 0
            missed = 0
            caught_list = []
            
            for sim_time in sim_times:
                start_idx = np.searchsorted(time_array, sim_time)
                clean_window = raw_matrix[start_idx : start_idx + WINDOW_SIZE].copy()
                clean_window = clean_window.T # Shape: (5, 20)
                
                for target_sensor in CHANNELS:
                    ch_idx = CHANNELS.index(target_sensor)
                    
                    for fault_type in fault_types:
                        corrupted_window = clean_window.copy()
                        series = corrupted_window[ch_idx, :]
                        
                        # --- FAULT INJECTION ---
                        if fault_type == 'Dropout':
                            series[:] = 0.0 
                        elif fault_type == 'Stuck Value':
                            series[:] = series[0] 
                        elif fault_type == 'Drift':
                            mean_val = np.mean(series)
                            drift_mag = mean_val * 0.3 + 5.0
                            
                            if (target_sensor in ['Throttle', 'Brake'] and mean_val > 50.0) or \
                               (target_sensor == 'Speed' and mean_val > 250.0) or \
                               (target_sensor == 'nGear' and mean_val > 4):
                                drift_dir = -1.0 
                            else:
                                drift_dir = 1.0  
                                
                            drift_amount = np.linspace(0, drift_dir * drift_mag, len(series))
                            series[:] = series + drift_amount
                        else: # Noise
                            global_sigma = orchestrator.stds[0][ch_idx][0]
                            noise_scale = global_sigma * 0.4 
                            t = np.arange(len(series))
                            vibration = np.cos(t * np.pi) * noise_scale
                            random_amps = (np.random.default_rng().uniform(0.0, 1.3, len(series))) ** 2
                            series[:] = series + (vibration * random_amps)
                            
                        # Hard-clip physical bounds
                        if target_sensor == 'Speed':
                            series[:] = np.clip(series, 0, 360)
                        elif target_sensor == 'nGear':
                            series[:] = np.clip(np.round(series), 0, 8)
                        elif target_sensor in ['Throttle', 'Brake']:
                            series[:] = np.clip(series, 0, 100)
                        elif target_sensor == 'RPM':
                            series[:] = np.clip(series, 0, 13000)
                            
                        corrupted_window[ch_idx, :] = series
                        
                        # --- INFERENCE ---
                        corrupted_scaled = (corrupted_window - orchestrator.means[0]) / orchestrator.stds[0]
                        with torch.no_grad():
                            tensor_in = torch.tensor(np.expand_dims(corrupted_scaled, 0), dtype=torch.float32).to(orchestrator.device)
                            _, _, fault_logits = orchestrator.tcn(tensor_in)
                            probs = torch.nn.functional.softmax(fault_logits[0], dim=0).cpu().numpy()
                            pred_idx = np.argmax(probs)
                            pred_sensor = CHANNELS[pred_idx]
                        
                        # --- EVALUATE ---
                        if pred_sensor == target_sensor:
                            caught += 1
                            overall_caught += 1
                            caught_list.append(f"✅ CAUGHT | Time: {sim_time:4.1f}s | Sensor: {target_sensor:<8} | Fault: {fault_type:<12} | Conf: {probs[pred_idx]*100:5.1f}%")
                        else:
                            missed += 1
                            overall_missed += 1
                            
            f.write(f"Track Caught: {caught}/{total_combinations_per_track} ({caught/total_combinations_per_track*100:.2f}%)\n")
            f.write("\n--- CAUGHT COMBINATIONS ---\n")
            for line in caught_list:
                f.write(line + "\n")
            f.write("\n" + "="*50 + "\n\n")
            
        f.write("=== FINAL GLOBAL RESULTS ===\n")
        f.write(f"Total Combinations Tested: {total_combinations_all}\n")
        f.write(f"Total Anomalies Caught:    {overall_caught}\n")
        f.write(f"Total Anomalies Missed:    {overall_missed}\n")
        f.write(f"Overall Accuracy:          {overall_caught/total_combinations_all*100:.2f}%\n")
        
    print(f"\nMulti-Track Sweep Complete! AI caught {overall_caught}/{total_combinations_all} ({overall_caught/total_combinations_all*100:.2f}%).")
    print(f"Full breakdown written to '{output_path}'.")

if __name__ == "__main__":
    main()
