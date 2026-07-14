import numpy as np
import pandas as pd
from typing import List, Tuple
from src.config import CHANNELS, WINDOW_SIZE, STEP_SIZE

def extract_raw_windows(df_10hz: pd.DataFrame, window_size: int = WINDOW_SIZE, step: int = STEP_SIZE) -> np.ndarray:
    """
    Slides a window across a 10Hz DataFrame.
    Returns array of shape: (N_windows, Time=20, Channels=5)
    """
    matrix = df_10hz[CHANNELS].values
    if len(matrix) < window_size:
        return np.empty((0, window_size, len(CHANNELS)))
        
    # Instantaneous O(1) sliding window view via NumPy strides
    windows = np.lib.stride_tricks.sliding_window_view(matrix, window_shape=window_size, axis=0)
    
    # Step slice to reduce overlap redundancy during training
    return windows[::step]

def extract_domain_physics_features(raw_windows: np.ndarray) -> np.ndarray:
    """
    Calculates domain-specific F1 mechanical features directly from 3D NumPy windows.
    Input Shape:  (Batch, Channels=5, Time=20)
    Output Shape: (Batch, 6 physics summary features)
    
    Assumed Channel Index Order in CHANNELS:
    [0: Speed, 1: RPM, 2: Throttle, 3: Brake, 4: nGear]
    """
    if raw_windows.size == 0:
        return np.empty((0, 6))
        
    # 1. Slice individual sensor channels across all windows -> Shape: (Batch, Time=20)
    speed    = raw_windows[:, 0, :]
    rpm      = raw_windows[:, 1, :]
    throttle = raw_windows[:, 2, :]
    brake    = raw_windows[:, 3, :]
    
    # --- FEATURE A: Rolling Speed Gradients (Acceleration / Deceleration in G-forces) ---
    # np.diff calculates (Speed_t - Speed_{t-1}) across the time axis (axis=1).
    # Multiplied by (1000/3600) to convert km/h to m/s, divided by 0.1s (10Hz step), divided by 9.81m/s^2 for Gs.
    speed_diff_ms = np.diff(speed, axis=1) * (1000.0 / 3600.0) # Shape: (Batch, 19)
    g_forces = (speed_diff_ms / 0.10) / 9.81
    
    max_accel_g = np.max(g_forces, axis=1, keepdims=True) # Shape: (Batch, 1)
    max_decel_g = np.min(g_forces, axis=1, keepdims=True) # Shape: (Batch, 1)
    
    # --- FEATURE B: Drivetrain Ratio (RPM / Speed) ---
    # We floor Speed at 10.0 km/h to prevent Division by Zero when stationary in grid/pits.
    safe_speed = np.maximum(speed, 10.0)
    drivetrain_ratio = rpm / safe_speed                   # Shape: (Batch, 20)
    
    mean_ratio = np.mean(drivetrain_ratio, axis=1, keepdims=True) # Shape: (Batch, 1)
    std_ratio  = np.std(drivetrain_ratio, axis=1, keepdims=True)  # Shape: (Batch, 1)
    
    # --- FEATURE C: Pedal Conflict Ratio (Throttle & Brake pressed simultaneously) ---
    # Normal racing rarely sees Throttle > 15% and Brake > 15% active together for long periods.
    active_throttle = throttle > 15.0
    active_brake    = brake > 15.0
    conflict_mask   = np.logical_and(active_throttle, active_brake) # Shape: (Batch, 20)
    
    # Proportion of the 20-timestamp window spent in pedal conflict (0.0 to 1.0)
    conflict_ratio = np.mean(conflict_mask.astype(float), axis=1, keepdims=True) # Shape: (Batch, 1)
    
    # --- FEATURE D: Zero-Speed Engine Rev Flag ---
    # High RPM (>6000) while vehicle speed is near zero (<15 km/h) indicates wheelspin or clutch drop
    rev_mask = np.logical_and(rpm > 6000.0, speed < 15.0)
    rev_ratio = np.mean(rev_mask.astype(float), axis=1, keepdims=True) # Shape: (Batch, 1)
    
    # Horizontally stack all 6 physics features -> Shape: (Batch, 6)
    physics_matrix = np.hstack([
        max_accel_g, 
        max_decel_g, 
        mean_ratio, 
        std_ratio, 
        conflict_ratio,
        rev_ratio
    ])
    
    return physics_matrix

def build_stage1_features(raw_windows: np.ndarray) -> np.ndarray:
    """
    Converts (Batch, Channels=5, Time=20) -> (Batch, 41 flat features).
    Combines 35 statistical features (Mean, Var, Min, Max, Q25, Q50, Q75) with 6 Domain Physics features.
    """
    if raw_windows.size == 0:
        return np.empty((0, len(CHANNELS) * 7 + 6))
        
    # 1. Calculate original 35 statistical features across time (axis=2)
    mean_val = np.mean(raw_windows, axis=2)        # Shape: (Batch, 5)
    var_val  = np.var(raw_windows, axis=2)         # Shape: (Batch, 5)
    min_val  = np.min(raw_windows, axis=2)         # Shape: (Batch, 5)
    max_val  = np.max(raw_windows, axis=2)         # Shape: (Batch, 5)
    q25      = np.percentile(raw_windows, 25, axis=2)
    q50      = np.percentile(raw_windows, 50, axis=2)
    q75      = np.percentile(raw_windows, 75, axis=2)
    
    # Stack and flatten -> Shape: (Batch, 35)
    stats_stacked = np.stack([mean_val, var_val, min_val, max_val, q25, q50, q75], axis=2)
    stat_features = stats_stacked.reshape(raw_windows.shape[0], -1)
    
    # 2. Calculate the 6 domain physics features -> Shape: (Batch, 6)
    physics_features = extract_domain_physics_features(raw_windows)
    
    # 3. Combine into master feature matrix -> Shape: (Batch, 41)
    return np.hstack([stat_features, physics_features])

def build_stage2_tensors(raw_windows: np.ndarray) -> np.ndarray:
    """
    Returns (Batch, Channels=5, Time=20).
    sliding_window_view already outputs this exact PyTorch 1D Conv alignment!
    Stage 2 receives clean raw data completely untouched by Stage 1 physics features!
    """
    if raw_windows.size == 0:
        return np.empty((0, len(CHANNELS), WINDOW_SIZE))
    return raw_windows

def generate_training_datasets(resampled_laps: List[pd.DataFrame]) -> Tuple[np.ndarray, np.ndarray]:
    """
    Master function: Takes resampled clean laps and outputs training sets for both stages.
    Returns: (stage1_tabular_X, stage2_tensor_X)
    """
    all_raw_windows = []
    for lap_df in resampled_laps:
        win = extract_raw_windows(lap_df)
        if win.size > 0:
            all_raw_windows.append(win)
            
    master_windows = np.concatenate(all_raw_windows, axis=0)
    
    X_stage1 = build_stage1_features(master_windows)
    X_stage2 = build_stage2_tensors(master_windows)
    
    print(f"Dataset Built Successfully!")
    print(f" -> Stage 1 (iForest) Shape:     {X_stage1.shape} (Batch, Features)")
    print(f" -> Stage 2 (TCN AE) Shape:      {X_stage2.shape} (Batch, Channels, Time)")
    return X_stage1, X_stage2