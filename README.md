# F1 Telemetry Anamoly Pipeline

## 1.Project Overview

- **The Problem**: Telemetry failures in sensors of F1 cars are difficult to isolate and diagnose because the sensors are deeply intterconnected. A single fault can create cascading alarms across all sensors.
- **The Solution**: This pipeline utilises a 2 stage architecture to monitor telemetry in real-time. By injecting a wide variety of synthetic faults during training, and then passing it through a highly compressed TCN , we can isolate the fault.

## 2. Architecture & The "2-Stage" Approach

**Stage 1: The Gatekeeper (Isolation Forest)**
* *The Goal (Detection):* Designed for extreme computational speed (sub-millisecond triage). Its only job is to act as a binary alarm bell ("Is there an anomaly?").
* *The "Big Net" Strategy:* Tuned heavily for **Recall (85.5%)** over Precision (20.4%). It intentionally lets a significant amount of clean telemetry slip through to ensure it never misses a true physical fault, leaving the precise diagnostic work to the neural network.
* *The Diagnostic Limitation (14% Accuracy):* While excellent at *detecting* faults, Stage 1 is incapable of *diagnosing* them. If forced to identify the broken sensor using naive math (e.g., picking the sensor with the largest Z-score deviation), it fails 86% of the time. This is because a dropped Throttle causes Speed to drop; mathematically, the Speed sensor may register a larger deviation than the Throttle itself, confusing the Isolation Forest.

**Stage 2: The Diagnostic Engine (TCN Autoencoder)**
* *The Goal (Diagnosis):* Deep physical attribution. Activated only when the Stage 1 Gatekeeper sounds the alarm.
* *The Mechanism (Physics Engine):* Takes a 20-timestep window of all 5 sensors and forces it through a massive bottleneck (`latent_dim = 2`). By compressing 100 data points into 2 spatial dimensions, it organically learns the physical laws of the car (e.g., *Throttle=100% means Speed must be rising*).
* *The Classification Head:* When a fault occurs, the Autoencoder understands the physical chain reaction. It knows that if the Speed drops while the Throttle reads 0%, the Speed sensor is telling the truth and the Throttle sensor is lying. This physical reasoning allows the neural classification head to achieve an **83.67% conditional diagnostic accuracy**.
* *Current Limitations (Variance Spillover):* Because the system only operates on 5 highly correlated sensors, the AI does not have a complete map of the car's physical environment. Consequently, even with an extreme 2D bottleneck, the network is sometimes forced to mathematically encode injected noise, causing partial reconstruction of erratic faults and slight degradation in diagnosis certainty.

## 3. Project Structure

This repository is organized to separate the data pipeline, the core neural architecture, and the visual dashboard. 

```text
├── app.py                            # The interactive Streamlit Dashboard (Live Injections & Visuals)
├── pipeline_metrics.json             # Dynamic evaluation metrics loaded by the dashboard
├── .gitignore                        # Prevents tracking of large datasets and model weights
├── data/                             # [IGNORED] Stores raw, fetched FastF1 telemetry data
├── models/                           # [IGNORED] Stores saved weights (e.g., Stage 1 .pkl models)
├── *.pth                             # [IGNORED] Stage 2 TCN PyTorch weights (e.g., stage2_tcn_ae.pth)
│
├── exploration/                      
│   └── diagnostic_sweep_results.txt  # Benchmark logs tracking AI performance across tracks
│   └── *.png                         # [IGNORED] Auto-generated ablation and latent space charts
│
└── src/
    ├── Data Pipeline:
    │   ├── ingestion.py              # Fetches live telemetry from the FastF1 API
    │   ├── preprocessing.py          # Cleans missing values and interpolates erratic sensors
    │   ├── features.py               # Computes rolling volatility and delta features for Stage 1
    │   └── build_dataset.py          # Generates the foundational training matrices
    │
    ├── Neural Architecture:
    │   ├── isolation_forest_model.py # Stage 1 Gatekeeper definition and logic
    │   ├── autoencoder_model.py      # Stage 2 TCN PyTorch architecture (2D Bottleneck)
    │   └── fault_injection.py        # Simulates physical faults (Drift, Dropouts, Noise, Stuck)
    │
    ├── Training:
    │   ├── train_isolation_forest.py # Trains the Stage 1 triage model
    │   └── train_autoencoder.py      # Trains the Stage 2 TCN using generalized fault physics
    │
    └── Evaluation & Benchmarking:
        ├── evaluate_pipeline.py         # Runs the full end-to-end 2-stage pipeline and outputs metrics
        ├── evaluate_all_combinations.py # Master benchmark (simulates 3,300 physical permutations)
        ├── evaluate_ablation.py         # Generates precision/recall metrics for the ablation charts
        └── profile_latency.py           # Evaluates the sub-millisecond inference speed of the models
```
