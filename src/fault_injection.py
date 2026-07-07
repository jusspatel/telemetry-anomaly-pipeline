# src/fault_injection.py
import numpy as np
import pandas as pd
from typing import List, Tuple, Dict
from .config import CHANNELS, TARGET_FREQ_HZ

class TelemetryFaultInjector:
    def __init__(self, seed: int = 42):
        self.seed = seed
        self.rng = np.random.default_rng(seed)
        self.ground_truth_log = []
        self.fault_types = ['dropout', 'stuck_value', 'drift', 'noise']

    def _log_fault(self, lap_idx: int, channel: str, fault_type: str, 
                   start_idx: int, end_idx: int, start_time: float, end_time: float):
        """Records injected fault details for Stage 1 & Stage 2 evaluation metrics."""
        self.ground_truth_log.append({
            'Lap_ID': lap_idx,
            'Channel': channel,
            'Fault_Type': fault_type,
            'Start_Idx': start_idx,
            'End_Idx': end_idx,
            'Start_Time': start_time,
            'End_Time': end_time
        })

    def inject_dropout(self, series: np.ndarray, start_idx: int, duration_idx: int) -> np.ndarray:
        """Simulates sensor electrical cut-out by zeroing out the signal."""
        corrupted = series.copy()
        end_idx = min(start_idx + duration_idx, len(corrupted))
        corrupted[start_idx:end_idx] = 0.0
        return corrupted

    def inject_stuck_value(self, series: np.ndarray, start_idx: int, duration_idx: int) -> np.ndarray:
        """Simulates a frozen data buffer by holding the channel fixed at its last valid numerical reading."""
        corrupted = series.copy()
        end_idx = min(start_idx + duration_idx, len(corrupted))
        # Freeze at the reading just before the fault occurs
        freeze_value = corrupted[max(0, start_idx - 1)]
        corrupted[start_idx:end_idx] = freeze_value
        return corrupted

    def inject_drift(self, series: np.ndarray, start_idx: int, duration_idx: int) -> np.ndarray:
        """Simulates thermal drift or calibration slip by slowly adding a linearly increasing offset over time."""
        corrupted = series.copy()
        end_idx = min(start_idx + duration_idx, len(corrupted))
        actual_duration = end_idx - start_idx
        
        # Scale max drift relative to channel magnitude (e.g., 30% of max channel reading)
        max_drift = np.max(np.abs(series)) * 0.30
        drift_slope = np.linspace(0.0, max_drift, actual_duration)
        
        corrupted[start_idx:end_idx] += drift_slope
        return corrupted

    def inject_noise_burst(self, series: np.ndarray, start_idx: int, duration_idx: int) -> np.ndarray:
        """Simulates RF or electrical interference by adding high-variance Gaussian noise across a short window."""
        corrupted = series.copy()
        end_idx = min(start_idx + duration_idx, len(corrupted))
        actual_duration = end_idx - start_idx
        
        # Noise standard deviation set to 15% of channel variance
        noise_std = np.std(series) * 0.15 if np.std(series) > 0 else 5.0
        noise = self.rng.normal(loc=0.0, scale=noise_std, size=actual_duration)
        
        corrupted[start_idx:end_idx] += noise
        return corrupted

    def corrupt_lap(self, lap_df: pd.DataFrame, lap_idx: int, num_faults: int = 3) -> pd.DataFrame:
        """
        Injects multiple non-overlapping synthetic faults into a clean lap DataFrame.
        """
        corrupted_df = lap_df.copy()
        n_timestamps = len(corrupted_df)
        
        # Ensure faults last between 1.5 and 4.0 seconds (15 to 40 timestamps at 10Hz)
        min_duration = int(1.5 * TARGET_FREQ_HZ)
        max_duration = int(4.0 * TARGET_FREQ_HZ)
        
        # Divide lap into safe segments to prevent faults from overlapping
        segment_size = n_timestamps // (num_faults + 1)
        
        for i in range(num_faults):
            fault_type = self.rng.choice(self.fault_types)
            channel = self.rng.choice(CHANNELS)
            
            # Pick a random start index within this segment
            seg_start = (i * segment_size) + int(TARGET_FREQ_HZ * 2) # Leave 2s buffer at start
            seg_end = ((i + 1) * segment_size) - max_duration
            
            if seg_start >= seg_end:
                continue
                
            start_idx = self.rng.integers(seg_start, seg_end)
            duration_idx = self.rng.integers(min_duration, max_duration)
            end_idx = start_idx + duration_idx
            
            # Get exact timestamps for the log
            start_time = corrupted_df['Time_Sec'].iloc[start_idx]
            end_time = corrupted_df['Time_Sec'].iloc[end_idx - 1]
            
            # Apply corruption
            raw_series = corrupted_df[channel].values
            if fault_type == 'dropout':
                mod_series = self.inject_dropout(raw_series, start_idx, duration_idx)
            elif fault_type == 'stuck_value':
                mod_series = self.inject_stuck_value(raw_series, start_idx, duration_idx)
            elif fault_type == 'drift':
                mod_series = self.inject_drift(raw_series, start_idx, duration_idx)
            else: # noise
                mod_series = self.inject_noise_burst(raw_series, start_idx, duration_idx)
                
            corrupted_df[channel] = mod_series
            self._log_fault(lap_idx, channel, fault_type, start_idx, end_idx, start_time, end_time)
            
        return corrupted_df

    def generate_faulty_test_set(self, clean_test_laps: List[pd.DataFrame], faults_per_lap: int = 3) -> Tuple[List[pd.DataFrame], pd.DataFrame]:
        """
        Master function: Injects faults across all held-out test laps and returns the ground-truth log.
        """
        self.ground_truth_log = [] # Reset log
        corrupted_laps = []
        
        for idx, lap in enumerate(clean_test_laps):
            if not lap.empty:
                corrupted_laps.append(self.corrupt_lap(lap, lap_idx=idx, num_faults=faults_per_lap))
                
        log_df = pd.DataFrame(self.ground_truth_log)
        print(f"Successfully injected {len(log_df)} synthetic faults across {len(corrupted_laps)} test laps!")
        return corrupted_laps, log_df