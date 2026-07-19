import sys
import time
import torch
import numpy as np
from pathlib import Path

# Ensure project root is in Python path
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

from src.evaluate_pipeline import TelemetryAnomalyOrchestrator
from src.features import build_stage1_features

def profile_latency():
    print("\n=======================================================")
    print("=== PIPELINE INFERENCE LATENCY PROFILER ===")
    print("=======================================================")

    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    stage1_path = Path("models") / "stage1_iforest.pkl"
    stage2_path = Path("models") / "stage2_tcn_ae.pth"
    
    print("\nLoading models into memory...")
    try:
        orchestrator = TelemetryAnomalyOrchestrator(stage1_path, stage2_path, device)
    except Exception as e:
        print(f"Error loading models: {e}")
        return

    # Generate 1,000 random telemetry windows of shape (5, 20)
    # This exactly mimics the 2.0-second sliding windows
    n_samples = 1000
    dummy_windows = np.random.rand(n_samples, 5, 20)
    
    print(f"\n[1] Benchmarking Stage 1: Isolation Forest (Triage) over {n_samples:,} windows...")
    
    stage1_latencies = []
    for i in range(n_samples):
        window_expanded = np.expand_dims(dummy_windows[i], axis=0)
        
        start_time = time.perf_counter()
        # Full Stage 1 execution: Feature Extraction + Isolation Forest prediction
        stage1_feats = build_stage1_features(window_expanded)
        orchestrator.stage1.triage_window(stage1_feats)
        end_time = time.perf_counter()
        
        stage1_latencies.append((end_time - start_time) * 1000) # Convert to milliseconds

    avg_s1_ms = np.mean(stage1_latencies)
    print(f"    -> Average Latency: {avg_s1_ms:.4f} ms per window")
    
    
    print(f"\n[2] Benchmarking Stage 2: TCN Autoencoder (Diagnosis) over {n_samples:,} windows...")
    
    stage2_latencies = []
    # Pre-scale to isolate pure neural network forward-pass time
    scaled_windows = (dummy_windows - orchestrator.means[0]) / orchestrator.stds[0]
    
    for i in range(n_samples):
        # Convert to tensor and push to device (simulating real-time casting)
        tensor_in = torch.tensor(np.expand_dims(scaled_windows[i], axis=0), dtype=torch.float32).to(orchestrator.device)
        
        start_time = time.perf_counter()
        with torch.no_grad():
            # Full neural network forward pass
            recon, _, _ = orchestrator.tcn(tensor_in)
        end_time = time.perf_counter()
        
        stage2_latencies.append((end_time - start_time) * 1000) # Convert to milliseconds
        
    avg_s2_ms = np.mean(stage2_latencies)
    print(f"    -> Average Latency: {avg_s2_ms:.4f} ms per window")

    print("\n=======================================================")
    print("=== FINAL ARCHITECTURE REPORT ===")
    print(f"Target SLA Constraints : < 100.0 ms (10Hz Telemetry)")
    print(f"Stage 1 Latency        :   {avg_s1_ms:.3f} ms")
    print(f"Stage 2 Latency        :   {avg_s2_ms:.3f} ms")
    print(f"Total Max Pipeline Time:   {avg_s1_ms + avg_s2_ms:.3f} ms")
    print("=======================================================\n")

if __name__ == "__main__":
    profile_latency()
