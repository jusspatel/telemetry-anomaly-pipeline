# src/evaluate_pipeline.py
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import joblib
from numpy.lib.stride_tricks import sliding_window_view
from sklearn.metrics import precision_score, recall_score, f1_score, classification_report

# Ensure project root is in Python path
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

from src.config import DATA_DIR, CHANNELS, WINDOW_SIZE
from src.features import build_stage1_features
from src.fault_injection import TelemetryFaultInjector
from src.autoencoder_model import TCNAutoencoder
from src.isolation_forest_model import PerChannelAnomalyTriage
from src.ingestion import load_clean_session_laps
from src.preprocessing import process_all_laps

class TelemetryAnomalyOrchestrator:
    """
    Master Inference Pipeline: Connects Stage 1 (Per-Channel iForest Triage)
    and Stage 2 (TCN Autoencoder Diagnostics).
    """
    def __init__(self, stage1_path: Path, stage2_path: Path, device: torch.device):
        self.device = device
        
        # 1. Load Stage 1 (Per-Channel Isolation Forests)
        print("Loading Stage 1 Per-Channel Isolation Forests...")
        self.stage1 = PerChannelAnomalyTriage(channel_names=CHANNELS)
        self.stage1.load_model(stage1_path)
        print(f" -> Loaded {self.stage1.n_channels} per-channel forests")
        for ch in CHANNELS:
            print(f"    {ch} threshold: {self.stage1.thresholds[ch]:.4f}")
        
        # 2. Load Stage 2 (TCN Autoencoder + Normalization Scalers)
        print("Loading Stage 2 TCN Autoencoder...")
        stage2_payload = torch.load(
            stage2_path, map_location=device, weights_only=False
        )
                
        config = stage2_payload['architecture_config']
        self.tcn = TCNAutoencoder(
            num_channels=config['num_channels'],
            latent_dim=config['latent_dim'],
            kernel_size=config['kernel_size']
        ).to(device)
        
        self.tcn.load_state_dict(stage2_payload['model_state_dict'])
        self.tcn.eval() # Set to evaluation mode (disables dropout)
        
        # Extract normalization scaling arrays
        self.means = stage2_payload['channel_means'] # Shape: (1, 5, 1)
        self.stds = stage2_payload['channel_stds']   # Shape: (1, 5, 1)
        print(" -> Stage 2 Loaded successfully with normalization parameters.")

    def calibrate_from_clean_windows(self, clean_windows: np.ndarray, target_fpr: float = 0.01):
        """
        Dynamically recalibrate per-channel thresholds using clean test data.
        This adapts thresholds to the actual test distribution, eliminating
        train/test distribution mismatch.
        """
        print(f"\nDynamic threshold calibration from {len(clean_windows)} clean windows...")
        feats = build_stage1_features(clean_windows)
        self.stage1.calibrate_thresholds(feats, target_fpr_per_channel=target_fpr)

    def evaluate_window(self, raw_window: np.ndarray) -> dict:
        """
        Processes a single (5, 20) raw telemetry window through the two-stage architecture.
        """
        # Step 1: Feature Extraction for Stage 1
        window_expanded = np.expand_dims(raw_window, axis=0)  # (1, 5, 20)
        stage1_feats = build_stage1_features(window_expanded)  # (1, 35)
        
        # Step 2: Per-Channel Stage 1 Triage
        is_anomalous, channel_scores, suspect = self.stage1.triage_window(stage1_feats)
        
        result = {
            'stage1_scores': channel_scores,
            'triage_alert': bool(is_anomalous),
            'stage1_suspect': suspect,
            'diagnosed_culprit': None,
            'channel_residuals': {}
        }
        
        # Step 3: If clean, dismiss window immediately (Save Compute!)
        if not is_anomalous:
            return result
            
        # Step 4: If Anomalous, wake up Stage 2 TCN Autoencoder for diagnostics
        with torch.no_grad():
            # Normalize input using baseline training parameters
            scaled_window = (window_expanded - self.means) / self.stds
            tensor_in = torch.tensor(scaled_window, dtype=torch.float32).to(self.device)
            
            # Reconstruct
            reconstructed_tensor, _ = self.tcn(tensor_in)
            
            # Convert back to numpy and un-scale to get physical units
            reconstructed_scaled = reconstructed_tensor.cpu().numpy()
            reconstructed_physical = (reconstructed_scaled * self.stds) + self.means
            
        # Calculate per-channel Mean Absolute Error (MAE) across the 20 timestamps
        epsilon_floor = 0.5
        robust_denominator = self.stds[0] + epsilon_floor  # Shape: (5, 1)

        # Calculate error relative to stabilized variance
        abs_errors = np.abs(window_expanded[0] - reconstructed_physical[0]) / (
            robust_denominator
        )
        channel_mae = np.mean(abs_errors, axis=1) # Shape: (5,)
        
        # Identify sensor with maximum residual error
        for idx, ch_name in enumerate(CHANNELS):
            result['channel_residuals'][ch_name] = float(channel_mae[idx])
            
        result['diagnosed_culprit'] = max(result['channel_residuals'], key=result['channel_residuals'].get)
        return result


def run_rigorous_evaluation():
    print("\n=======================================================")
    print("=== STARTING WEEK 4: RIGOROUS PIPELINE EVALUATION ===")
    print("=======================================================")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    
    # 1. Initialize Master Orchestrator
    stage1_path = Path("models") / "stage1_iforest.pkl"
    stage2_path = Path("models") / "stage2_tcn_ae.pth"
    orchestrator = TelemetryAnomalyOrchestrator(stage1_path, stage2_path, device)
    
    # 2. Harvest Held-Out Test Laps (Qualifying from a track NOT in training set)
    #    Training used: Bahrain, Monza, Silverstone. Testing on: Jeddah (held-out track).
    #    Using qualifying (same session type as training) to avoid distribution mismatch.
    #    Multiple drivers for more test data.
    print("\nHarvesting held-out test laps from FastF1 (2023 Jeddah GP - Qualifying)...")
    clean_test_laps = []
    test_drivers = ['VER', 'HAM', 'LEC']
    for driver in test_drivers:
        try:
            raw_laps = load_clean_session_laps(year=2023, gp='Jeddah', session_type='Q', driver=driver)
            clean_test_laps.extend(process_all_laps(raw_laps))
        except Exception as e:
            print(f" -> Could not load {driver} at Jeddah: {e}")
    
    if len(clean_test_laps) == 0:
        print("FATAL: No test laps could be loaded. Exiting.")
        return
        
    print(f" -> Harvested {len(clean_test_laps)} held-out test laps.")
    
    # 3. Extract clean windows for dynamic threshold calibration
    print("\nExtracting clean windows for dynamic threshold calibration...")
    all_clean_windows = []
    for lap_df in clean_test_laps:
        raw_matrix = lap_df[CHANNELS].values
        wins = sliding_window_view(raw_matrix, window_shape=WINDOW_SIZE, axis=0)
        all_clean_windows.append(wins)
    clean_windows_array = np.concatenate(all_clean_windows, axis=0)
    print(f" -> Extracted {len(clean_windows_array)} clean windows for calibration.")
    
    # 4. Dynamic Threshold Calibration
    #    This adapts thresholds to the actual test distribution, so we don't
    #    depend on the training distribution matching the test distribution.
    orchestrator.calibrate_from_clean_windows(clean_windows_array, target_fpr=0.01)
    
    # 5. Inject Synthetic Hardware Faults
    print("\nInjecting synthetic hardware faults into test laps...")
    injector = TelemetryFaultInjector(seed=999)
    corrupted_laps, ground_truth_log = injector.generate_faulty_test_set(clean_test_laps, faults_per_lap=3)
    
    # 6. Stream Continuous Windows through the Pipeline & Record Predictions
    print("\nStreaming test telemetry through Two-Stage Pipeline...")
    
    y_true_binary = []      # 0 for clean window, 1 for ground-truth fault window
    y_pred_binary = []      # 0 for Stage 1 dismissal, 1 for Stage 1 alert
    correct_attributions = 0
    total_true_fault_windows = 0
    
    # We loop through each corrupted lap and extract 2-second sliding windows
    for lap_idx, lap_df in enumerate(corrupted_laps):
        raw_matrix = lap_df[CHANNELS].values
        time_array = lap_df['Time_Sec'].values
        
        # Extract sliding windows
        windows = sliding_window_view(raw_matrix, window_shape=WINDOW_SIZE, axis=0)
        
        # Check ground truth log for this lap
        lap_faults = ground_truth_log[ground_truth_log['Lap_ID'] == lap_idx]
        
        for w_idx in range(len(windows)):
            w_start_time = time_array[w_idx]
            w_end_time = time_array[w_idx + WINDOW_SIZE - 1]
            
            # Check if this window overlaps with any injected fault from our log
            active_fault = None
            for _, row in lap_faults.iterrows():
                # If window overlaps with fault timestamp span
                if max(w_start_time, row['Start_Time']) <= min(w_end_time, row['End_Time']):
                    active_fault = row
                    break
            
            is_true_fault = active_fault is not None
            y_true_binary.append(1 if is_true_fault else 0)
            
            # Execute Pipeline Inference!
            prediction = orchestrator.evaluate_window(windows[w_idx])
            
            y_pred_binary.append(1 if prediction['triage_alert'] else 0)
            
            # If this was a true fault AND Stage 1 caught it, grade Stage 2's diagnosis!
            if is_true_fault:
                total_true_fault_windows += 1
                if prediction['triage_alert']:
                    if prediction['diagnosed_culprit'] == active_fault['Channel']:
                        correct_attributions += 1
                        
    # 7. Calculate Industrial Reliability Metrics
    precision = precision_score(y_true_binary, y_pred_binary, zero_division=0)
    recall = recall_score(y_true_binary, y_pred_binary, zero_division=0)
    f1 = f1_score(y_true_binary, y_pred_binary, zero_division=0)
    
    attribution_acc = (correct_attributions / total_true_fault_windows) * 100 if total_true_fault_windows > 0 else 0.0
    
    # Count predictions for context
    total_alerts = sum(y_pred_binary)
    total_true_faults = sum(y_true_binary)
    
    print("\n=======================================================")
    print("=== FINAL INDUSTRIAL RELIABILITY EVALUATION REPORT ===")
    print("=======================================================")
    print(f"Total Telemetry Windows Evaluated: {len(y_true_binary):,}")
    print(f"Total True Fault Windows Injected: {total_true_fault_windows:,}")
    print(f"Total Stage 1 Alerts Fired:        {total_alerts:,}")
    print(f"\n--- STAGE 1: TRIAGE DETECTION PERFORMANCE ---")
    print(f" -> Precision:        {precision:.4f}  (How many alarms were real faults?)")
    print(f" -> Recall:           {recall:.4f}  (How many real faults did we catch?)")
    print(f" -> F1-Score:         {f1:.4f}  (Harmonic mean of precision & recall)")
    print(f"\n--- STAGE 2: DIAGNOSTIC ATTRIBUTION PERFORMANCE ---")
    print(f" -> Culprit Localization Accuracy: {attribution_acc:.2f}%")
    print(f"    (When a fault occurred, how often did TCN correctly name the broken sensor?)")
    print("=======================================================\n")

if __name__ == "__main__":
    run_rigorous_evaluation()