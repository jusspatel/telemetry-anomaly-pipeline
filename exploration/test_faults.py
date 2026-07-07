import sys
from pathlib import Path


project_root = Path(__file__).resolve().parents[1] 
sys.path.append(str(project_root))
# test_faults.py
import matplotlib.pyplot as plt
from src.fault_injection import TelemetryFaultInjector
from test_pipeline import resampled_laps

# 1. Initialize the injector
injector = TelemetryFaultInjector(seed=101)

# 2. Take your first resampled lap and treat it as a "Held-Out Test Lap"
test_lap = resampled_laps[0]

# 3. Inject 4 distinct faults into this lap
corrupted_laps, log_df = injector.generate_faulty_test_set([test_lap], faults_per_lap=4)
corrupted_lap = corrupted_laps[0]

# Print the unambiguous ground truth log!
print("\n--- GROUND TRUTH ANOMALY LOG ---")
print(log_df[['Channel', 'Fault_Type', 'Start_Time', 'End_Time']])

# 4. Visualize a corrupted channel against the clean baseline
def plot_corrupted_channel(clean_df, corrupt_df, log_df, channel_to_plot='Speed'):
    plt.figure(figsize=(14, 5))
    plt.plot(clean_df['Time_Sec'], clean_df[channel_to_plot], label="Clean Baseline (Normal)", color='black', linestyle='--', alpha=0.6)
    plt.plot(corrupt_df['Time_Sec'], corrupt_df[channel_to_plot], label="Corrupted Telemetry (Injected)", color='#d62728', linewidth=1.5)
    
    # Highlight the exact injected fault windows from our ground truth log
    channel_faults = log_df[log_df['Channel'] == channel_to_plot]
    for _, row in channel_faults.iterrows():
        plt.axvspan(row['Start_Time'], row['End_Time'], color='yellow', alpha=0.3, label=f"Fault: {row['Fault_Type'].upper()}")
        
    plt.title(f"Synthetic Fault Injection Verification: {channel_to_plot} Channel", fontsize=14, fontweight='bold')
    plt.xlabel("Time (Seconds)", fontweight='bold')
    plt.ylabel(channel_to_plot, fontweight='bold')
    
    # Deduplicate legend items
    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    plt.legend(by_label.values(), by_label.keys(), loc="upper right")
    
    plt.tight_layout()
    plt.show()

# Plot the channel that received the first fault in the log
first_fault_channel = log_df.iloc[0]['Channel']
plot_corrupted_channel(test_lap, corrupted_lap, log_df, channel_to_plot=first_fault_channel)