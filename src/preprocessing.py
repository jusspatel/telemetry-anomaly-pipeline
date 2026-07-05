# src/preprocessing.py
import numpy as np
import pandas as pd
from scipy.interpolate import interp1d
from typing import List
from src.config import CHANNELS, TARGET_FREQ_HZ

def resample_lap_to_10hz(lap_df: pd.DataFrame) -> pd.DataFrame:
    """
    
    telemetry reports asynchronously with irregular, messy timestamps (e.g., 0.012s, 
    0.098s, 0.205s), and different sensors often report at slightly different milliseconds. 
    Machine learning models (like sliding windows and 1D Convolutional networks) require a rigid, 
    mathematically perfect grid where every single timestamp is spaced identically apart.
    
    This function takes those irregular timestamps and forces them onto a perfect 10Hz ruler: 
    exactly 0.0s, 0.1s, 0.2s, 0.3s, 0.4s... (10 ticks per second).
    """
    # FastF1 stores time as a "time-delta" stopwatch object (e.g., '0 days 00:01:23.456'). 
    # We cannot easily do math on clock text. .dt.total_seconds() converts that clock text 
    # into a simple decimal number representing elapsed seconds from the start of the lap.
    # Example Output: [0.032, 0.095, 0.188, 0.291, ...]
    time_sec = lap_df['Time'].dt.total_seconds().values
    
    # We round up the start and round down the end to create clean 0.1-second boundaries.
    
    # grid_start: Rounding up to the first clean tick.
    # If telemetry starts late at 0.032s -> multiply by 10 (0.32) -> ceil rounds up to 1.0 -> 
    # divide by 10 (0.1). Our clean ruler officially starts at 0.1 seconds.
    grid_start = np.ceil(time_sec[0] * TARGET_FREQ_HZ) / TARGET_FREQ_HZ
    
    # grid_end: Rounding down to the last clean tick.
    # If lap ends at 85.964s -> floor rounds down to the nearest tenth -> 85.9 seconds.
    grid_end = np.floor(time_sec[-1] * TARGET_FREQ_HZ) / TARGET_FREQ_HZ
    
    # Generate the new, perfect time array from start to end, stepping by exactly 0.1s (1/10Hz)
    # New Clean Grid Output: [0.1, 0.2, 0.3, 0.4, ... 85.9]
    uniform_time_grid = np.arange(grid_start, grid_end, 1.0 / TARGET_FREQ_HZ)
    
    resampled_data = {'Time_Sec': uniform_time_grid}
    
    # =========================================================================
    # PHASE 3: "Connect the Dots" (Linear Interpolation)
    # =========================================================================
    # Now that we have our perfect time grid (0.1s, 0.2s...), we have a problem: we don't have 
    # real sensor readings at exactly those timestamps! We solve this by playing "connect the dots" 
    # with a straight line between the real sensor readings.
    #
    # Analogy: If Speed was 100 km/h at timestamp 0.08s, and 110 km/h at 0.12s, linear 
    # interpolation draws a line between them and calculates that at our new clean timestamp 
    # of exactly 0.10s (halfway between), the speed was 105 km/h.
    for channel in CHANNELS:
        # 1. Build a mathematical blueprint of the line connecting messy timestamps to messy values
        interp_func = interp1d(
            time_sec, 
            lap_df[channel].values, 
            kind='linear', 
            fill_value="extrapolate"
        )
        # 2. Feed our new, perfect 0.1s timestamps into that blueprint to get aligned sensor values!
        resampled_data[channel] = interp_func(uniform_time_grid)
        
    return pd.DataFrame(resampled_data)

def process_all_laps(lap_list: List[pd.DataFrame]) -> List[pd.DataFrame]:
    """
    Applies 10Hz resampling across a list of raw lap DataFrames.
    Filters out any empty DataFrames that may have resulted from ingestion glitches.
    """
    return [resample_lap_to_10hz(lap) for lap in lap_list if not lap.empty]