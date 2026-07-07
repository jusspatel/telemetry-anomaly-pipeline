# src/features.py
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

    ##eg:
    #     [280, 100],  # 0.0s - Full throttle down the straight
    # [285, 100],  # 0.1s
    # [290, 100],  # 0.2s - Peak speed
    # [270,   0],  # 0.3s - Driver slams brakes, throttle drops to 0%
    # [240,   0],  # 0.4s - Speed drops rapidly
    # [210,   0],  # 0.5s
    # [190,  10]   # 0.6s - Apex of corner, driver gently gets back on throttle

#     [[[320 322 324 280]
#   [100 100 100   0]]

#  [[322 324 280 210]
#   [100 100   0   0]]

#  [[324 280 210 150]
#   [100   0   0   0]]

#  [[280 210 150 115]
#   [  0   0   0  15]]]

    ##this is for window size of 4. 3d matrix , where axis 1 is the window number and axis 2 is window size of 4 of speed and windows size of 4 of throttle

    
    # Step slice to reduce overlap redundancy during training
    return windows[::step]

def build_stage1_features(raw_windows: np.ndarray) -> np.ndarray:
    """
    Converts (Batch, Channels=5, Time=20) -> (Batch, 35 flat statistical features).
    Stats: Mean, Var, Min, Max, Q25, Q50, Q75 calculated OVER TIME (axis=2) per channel.
    """
    if raw_windows.size == 0:
        return np.empty((0, len(CHANNELS) * 7))
        
    # Notice we changed axis=1 to axis=2 so we calculate stats across the 20 timestamps!
    mean_val = np.mean(raw_windows, axis=2)        # Shape: (Batch, 5)
    var_val  = np.var(raw_windows, axis=2)         # Shape: (Batch, 5)
    min_val  = np.min(raw_windows, axis=2)         # Shape: (Batch, 5)
    max_val  = np.max(raw_windows, axis=2)         # Shape: (Batch, 5)
    q25      = np.percentile(raw_windows, 25, axis=2)
    q50      = np.percentile(raw_windows, 50, axis=2)
    q75      = np.percentile(raw_windows, 75, axis=2)
    
    # Stack along third axis -> Shape: (Batch, 5 channels, 7 stats)
    stats_stacked = np.stack([mean_val, var_val, min_val, max_val, q25, q50, q75], axis=2)
    
    # Flatten channels and stats together -> Shape: (Batch, 35)
    return stats_stacked.reshape(raw_windows.shape[0], -1)

def build_stage2_tensors(raw_windows: np.ndarray) -> np.ndarray:
    """
    Returns (Batch, Channels=5, Time=20).
    sliding_window_view already outputs this exact PyTorch 1D Conv alignment!
    """
    if raw_windows.size == 0:
        return np.empty((0, len(CHANNELS), WINDOW_SIZE))
    return raw_windows  # Removed the unnecessary transpose!

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