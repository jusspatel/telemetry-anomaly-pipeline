from pathlib import Path
import sys
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Ensure project root is in path
project_root = Path(__file__).resolve().parents[1]
sys.path.append(str(project_root))

from src.config import CHANNELS, TARGET_FREQ_HZ
from src.fault_injection import TelemetryFaultInjector
from src.features import generate_training_datasets
from src.ingestion import load_clean_session_laps
from src.preprocessing import process_all_laps


def visualize_multi_fault_audit(
    clean_lap_df: pd.DataFrame,
    corrupted_lap_df: pd.DataFrame,
    fault_log: pd.DataFrame,
    lap_idx: int = 0,
):
  """Plots a full lap overview with colored fault banners, then generates a multi-instance grid comparing clean vs.

  corrupted telemetry side-by-side.
  """
  plt.style.use(
      "seaborn-v0_8-darkgrid"
      if "seaborn-v0_8-darkgrid" in plt.style.available
      else "default"
  )

  lap_faults = fault_log[fault_log["Lap_ID"] == lap_idx].reset_index(drop=True)
  time_sec = clean_lap_df["Time_Sec"].values

  fault_colors = {
      "drift": "#e67e22",  # Orange
      "stuck_value": "#9b59b6",  # Purple
      "noise": "#e74c3c",  # Red
      "dropout": "#34495e",  # Dark Gray
  }

  # =========================================================================
  # PLOT 1: Full Pit Wall Overview (All Channels + Fault Regions)
  # =========================================================================
  fig, axes = plt.subplots(len(CHANNELS), 1, figsize=(16, 10), sharex=True)
  fig.suptitle(
      f"Pit Wall Telemetry Audit (Lap {lap_idx + 1}) — {len(lap_faults)} Active Faults Injected",
      fontsize=16,
      fontweight="bold",
  )

  for i, channel in enumerate(CHANNELS):
    ax = axes[i]
    ax.plot(
        time_sec,
        clean_lap_df[channel],
        label="Clean Baseline",
        color="#bdc3c7",
        linestyle="--",
        linewidth=1.2,
        alpha=0.8,
    )
    ax.plot(
        time_sec,
        corrupted_lap_df[channel],
        label="Corrupted Telemetry",
        color="#2c3e50",
        linewidth=1.5,
    )
    ax.set_ylabel(channel, fontweight="bold", fontsize=11)
    if i == 0:
      ax.legend(loc="upper right", frameon=True)

    # Highlight injected spans
    for _, row in lap_faults.iterrows():
      if row["Channel"] == channel:
        f_type = row["Fault_Type"]
        f_color = fault_colors.get(f_type, "blue")
        ax.axvspan(
            row["Start_Time"], row["End_Time"], color=f_color, alpha=0.25
        )
        ax.text(
            row["Start_Time"],
            ax.get_ylim()[1] * 0.85,
            f" {f_type.upper()}",
            color=f_color,
            fontweight="bold",
            fontsize=9,
        )

  axes[-1].set_xlabel("Lap Elapsed Time (Seconds)", fontweight="bold")
  plt.tight_layout()
  plt.show()

  # =========================================================================
  # PLOT 2: Multi-Instance Zoomed-In Grid (One Subplot Per Fault)
  # =========================================================================
  num_faults = len(lap_faults)
  if num_faults == 0:
    return

  fig, axes = plt.subplots(1, num_faults, figsize=(5 * num_faults, 5))
  if num_faults == 1:
    axes = [axes]

  fig.suptitle(
      f"Stage 2 Autoencoder Input Audit: Clean vs. Corrupted Profiles (Lap {lap_idx + 1})",
      fontsize=14,
      fontweight="bold",
  )

  for idx, row in lap_faults.iterrows():
    ax = axes[idx]
    ch = row["Channel"]
    f_type = row["Fault_Type"]
    s_idx, e_idx = row["Start_Idx"], row["End_Idx"]

    # Add a 5-timestamp buffer on both sides to visualize the transition
    view_start = max(0, s_idx - 5)
    view_end = min(len(clean_lap_df), e_idx + 5)

    t_grid = time_sec[view_start:view_end]
    clean_vals = clean_lap_df[ch].iloc[view_start:view_end].values
    corrupt_vals = corrupted_lap_df[ch].iloc[view_start:view_end].values

    ax.plot(
        t_grid,
        clean_vals,
        label="Expected Physics (Clean)",
        color="green",
        linestyle="--",
        marker="o",
        markersize=4,
        linewidth=1.8,
    )
    ax.plot(
        t_grid,
        corrupt_vals,
        label="Actual Input (Corrupted)",
        color="red",
        marker="x",
        markersize=5,
        linewidth=2.0,
    )
    ax.axvspan(
        row["Start_Time"],
        row["End_Time"],
        color=fault_colors.get(f_type, "gray"),
        alpha=0.15,
        label="Fault Span",
    )

    ax.set_title(
        f"Fault #{idx+1}: {ch} ({f_type.upper()})",
        fontsize=12,
        fontweight="bold",
    )
    ax.set_xlabel("Time (s)", fontweight="bold")
    ax.set_ylabel(f"{ch} Amplitude", fontweight="bold")
    ax.legend(loc="best", fontsize=8)
    ax.grid(True, linestyle=":", alpha=0.6)

  plt.tight_layout()
  plt.show()


# =========================================================================
# EXECUTION: Pull Clean Lap & Inject All 4 Fault Types Deterministically
# =========================================================================
if __name__ == "__main__":
  print("1. Pulling clean qualifying lap from FastF1...")
  raw_laps = load_clean_session_laps(
      year=2023, gp="Bahrain", session_type="Q", driver="VER"
  )
  resampled_laps = process_all_laps(raw_laps)

  if not resampled_laps:
    print("FATAL: No clean laps loaded.")
    sys.exit(1)

  clean_lap = resampled_laps[0]
  print(
      f" -> Loaded clean lap with {len(clean_lap)} timestamps ({len(clean_lap)/TARGET_FREQ_HZ:.1f}s)."
  )

  print("2. Injecting all 4 distinct fault types deterministically...")
  injector = TelemetryFaultInjector(seed=42)

  # Force exactly 4 faults onto this lap so we can audit every failure mode
  corrupted_lap = clean_lap.copy()
  n_time = len(corrupted_lap)
  seg_size = n_time // 5  # Divide lap into 5 safe segments

  # Target 4 specific channels and fault types
  audit_plan = [
      ("Brake", "drift"),
      ("Throttle", "stuck_value"),
      ("RPM", "noise"),
      ("Speed", "dropout"),
  ]

  injector.ground_truth_log = []

  for i, (channel, f_type) in enumerate(audit_plan):
    start_idx = (i * seg_size) + int(TARGET_FREQ_HZ * 3)
    duration_idx = int(TARGET_FREQ_HZ * 2.5)  # 2.5 second duration
    end_idx = start_idx + duration_idx

    start_time = corrupted_lap["Time_Sec"].iloc[start_idx]
    end_time = corrupted_lap["Time_Sec"].iloc[end_idx - 1]

    raw_series = corrupted_lap[channel].values
    if f_type == "dropout":
      mod_series = injector.inject_dropout(raw_series, start_idx, duration_idx)
    elif f_type == "stuck_value":
      mod_series = injector.inject_stuck_value(
          raw_series, start_idx, duration_idx
      )
    elif f_type == "drift":
      mod_series = injector.inject_drift(raw_series, start_idx, duration_idx)
    else:
      mod_series = injector.inject_noise_burst(
          raw_series, start_idx, duration_idx
      )

    corrupted_lap[channel] = mod_series
    injector._log_fault(
        0, channel, f_type, start_idx, end_idx, start_time, end_time
    )

  log_df = pd.DataFrame(injector.ground_truth_log)
  print(f" -> Injected {len(log_df)} faults. Launching visualizer...")

  # 3. Render the multi-instance audit grids!
  visualize_multi_fault_audit(
      clean_lap, corrupted_lap, log_df, lap_idx=0
  )