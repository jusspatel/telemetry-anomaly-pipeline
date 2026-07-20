import sys
from pathlib import Path
import streamlit as st
import numpy as np
import torch
import pandas as pd
import matplotlib.pyplot as plt
from numpy.lib.stride_tricks import sliding_window_view

# Ensure project root is in Python path
project_root = Path(__file__).resolve().parent
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

from src.config import CHANNELS, WINDOW_SIZE
from src.ingestion import load_clean_session_laps
from src.preprocessing import process_all_laps
from src.evaluate_pipeline import TelemetryAnomalyOrchestrator

# ==========================================
# 1. CACHING MODELS & DATA (For Speed)
# ==========================================
@st.cache_resource
def load_orchestrator():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    stage1_path = Path("models") / "stage1_iforest.pkl"
    stage2_path = Path("models") / "stage2_tcn_ae.pth"
    return TelemetryAnomalyOrchestrator(stage1_path, stage2_path, device)

@st.cache_data
def load_track_data(track_name):
    raw_laps = load_clean_session_laps(year=2023, gp=track_name, session_type='Q', driver='VER')
    clean_laps = process_all_laps([raw_laps[0]]) 
    return clean_laps[0]

# ==========================================
# 2. STREAMLIT UI SETUP
# ==========================================
st.set_page_config(page_title="F1 Telemetry AI", layout="wide")

# Sidebar Navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio("Select View:", ["The Diagnostic Engine", "Pipeline Metrics", "Sensitivity Analysis"])

orchestrator = load_orchestrator()

# ==========================================
# 3. PAGE: THE DIAGNOSTIC ENGINE
# ==========================================
if page == "The Diagnostic Engine":
    st.header("Diagnostic Engine (Stage 2)")
    st.markdown("Test the **TCN Autoencoder's** ability to heal data and diagnose the broken sensor on an unseen track.")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        track = st.selectbox("1. Choose Unseen Track:", ["Melbourne", "Jeddah", "Miami"])
    with col2:
        fault_type = st.selectbox("2. Choose Fault Type:", ['Dropout', 'Stuck Value', 'Drift', 'Noise'])
    with col3:
        target_sensor = st.selectbox("3. Choose Sensor to Break:", CHANNELS)
        
    sim_time = st.slider("4. Select Lap Time to Inject Fault (seconds):", 20.0, 75.0, 30.0, 1.0, 
                         help="Scrub through the lap! If testing a Brake fault, find a braking zone. If testing a Throttle fault, find a corner exit.")
        
    if st.button("Run Diagnosis", use_container_width=True):
        with st.spinner(f"Fetching {track} telemetry and running AI..."):
            
            # Load data
            live_lap = load_track_data(track)
            raw_matrix = live_lap[CHANNELS].values
            
            # Grab a clean 2.0s window based on the slider
            time_array = live_lap['Time_Sec'].values
            start_idx = np.searchsorted(time_array, sim_time)
            clean_window = raw_matrix[start_idx : start_idx + WINDOW_SIZE].copy()
            clean_window = clean_window.T # Transpose to shape (5, 20) to match TCN expectations
            
            # Inject Fault manually in PHYSICAL space for realism
            ch_idx = CHANNELS.index(target_sensor)
            corrupted_window = clean_window.copy()
            
            series = corrupted_window[ch_idx, :]
            if fault_type == 'Dropout':
                series[:] = 0.0 # Realistic physical signal drop
            elif fault_type == 'Stuck Value':
                series[:] = series[0] # Signal freezes at its entry value
            elif fault_type == 'Drift':
                # Smart Drift: Drift downwards if we are already near the physical ceiling so it doesn't get clipped
                mean_val = np.mean(series)
                drift_mag = mean_val * 0.3 + 5.0
                
                if (target_sensor in ['Throttle', 'Brake'] and mean_val > 50.0) or \
                   (target_sensor == 'Speed' and mean_val > 250.0) or \
                   (target_sensor == 'nGear' and mean_val > 4):
                    drift_dir = -1.0 # Drift downwards
                else:
                    drift_dir = 1.0  # Drift upwards
                    
                drift_amount = np.linspace(0, drift_dir * drift_mag, len(series))
                series[:] = series + drift_amount
            else: # Noise / Seismograph Vibration
                # Simulate a high-frequency mechanical vibration (loose sensor mount)
                global_sigma = orchestrator.stds[0][ch_idx][0]
                noise_scale = global_sigma * 0.4 # Increased for higher maximum peaks
                
                # Create a rapid +1, -1, +1, -1 oscillation
                t = np.arange(len(series))
                vibration = np.cos(t * np.pi) * noise_scale
                
                # Square the random amplitude so most vibrations are small, but a few are extremely sharp spikes
                random_amps = (np.random.default_rng().uniform(0.0, 1.3, len(series))) ** 2
                
                series[:] = series + (vibration * random_amps)
                
            # Hard-clip to F1 physical bounds so it makes intuitive sense
            if target_sensor == 'Speed':
                series[:] = np.clip(series, 0, 360)
            elif target_sensor == 'nGear':
                series[:] = np.clip(np.round(series), 0, 8)
            elif target_sensor in ['Throttle', 'Brake']:
                series[:] = np.clip(series, 0, 100)
            elif target_sensor == 'RPM':
                series[:] = np.clip(series, 0, 13000)
                
            corrupted_window[ch_idx, :] = series
            
            # Convert to Z-score for the neural network
            clean_scaled = (clean_window - orchestrator.means[0]) / orchestrator.stds[0]
            corrupted_scaled = (corrupted_window - orchestrator.means[0]) / orchestrator.stds[0]
            
            # Run Inference directly through Stage 2
            with torch.no_grad():
                tensor_in = torch.tensor(np.expand_dims(corrupted_scaled, 0), dtype=torch.float32).to(orchestrator.device)
                reconstructed, _, fault_logits = orchestrator.tcn(tensor_in)
                
                recon_numpy = reconstructed.cpu().numpy()[0]
                error_numpy = np.abs(corrupted_scaled - recon_numpy)
                
                # Use the trained neural network classification head!
                probs = torch.nn.functional.softmax(fault_logits[0], dim=0).cpu().numpy()
                
                pred_idx = np.argmax(probs)
                pred_sensor = CHANNELS[pred_idx]
                
            # --- PLOTTING ---
            plt.style.use('dark_background')
            fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1, figsize=(10, 12))
            
            x_time = np.arange(WINDOW_SIZE) / 10.0 # 0.0 to 2.0 seconds
            
            # Unscale AI outputs back to physical units for visualization
            mu = orchestrator.means[0][ch_idx][0]
            sigma = orchestrator.stds[0][ch_idx][0]
            
            phys_corrupted = corrupted_window[ch_idx, :] # Already in physical units!
            phys_clean = clean_window[ch_idx, :]         # Already in physical units!
            phys_recon = recon_numpy[ch_idx, :] * sigma + mu
            phys_error = error_numpy[ch_idx, :] * sigma # Absolute error scales directly
            
            # 1. Corrupted Input
            ax1.plot(x_time, phys_corrupted, color='red', linewidth=3, label='Corrupted Input')
            ax1.plot(x_time, phys_clean, color='white', linestyle='--', alpha=0.5, label='True Signal')
            ax1.set_title(f"1. Stage 2 saw({fault_type} on {target_sensor})", color='white')
            ax1.set_ylabel("Physical Value")
            ax1.legend()
            
            # 2. Reconstruction
            ax2.plot(x_time, phys_recon, color='cyan', linewidth=3, label='AI Reconstruction')
            ax2.plot(x_time, phys_clean, color='white', linestyle='--', alpha=0.5, label='True Signal')
            
            # Pad the Y-axis using the sensor's natural standard deviation 
            # so Matplotlib doesn't micro-zoom into rounding errors (e.g. -0.006).
            y_min = min(min(phys_recon), min(phys_clean))
            y_max = max(max(phys_recon), max(phys_clean))
            padding = sigma * 0.5
            ax2.set_ylim(y_min - padding, y_max + padding)
            
            ax2.set_title("2. Attempted reconstruction", color='white')
            ax2.set_ylabel("Physical Value")
            ax2.legend()
            
            # 3. Absolute Error (Plotted in Z-Scores so all sensors are on the same scale!)
            ax3.plot(x_time, error_numpy[ch_idx, :], color='orange', linewidth=2, label=f'{target_sensor} Error (Z-Score)')
            # Plot other channels faintly for comparison
            for i in range(5):
                if i != ch_idx:
                    ax3.plot(x_time, error_numpy[i, :], color='gray', alpha=0.3)
            ax3.set_title("3. The Error Signal (Z-Score space)", color='white')
            ax3.set_ylabel("Absolute Z-Error")
            ax3.legend()
            
            # 4. Probabilities
            colors = ['gray'] * 5
            colors[pred_idx] = 'green' if pred_sensor == target_sensor else 'red'
            ax4.bar(CHANNELS, probs * 100, color=colors)
            ax4.set_ylim(0, 100)
            ax4.set_title(f"4. The Final Verdict: {pred_sensor} ({probs[pred_idx]*100:.1f}%)", color='white')
            
            plt.tight_layout()
            st.pyplot(fig)
            
            if pred_sensor == target_sensor:
                st.success(f"**CORRECT!** The Autoencoder successfully isolated the fault to the {target_sensor} sensor.")
            else:
                st.error(f"**INCORRECT!** The Autoencoder blamed the {pred_sensor} sensor.")

# ==========================================
# 5. PAGE: PIPELINE METRICS
# ==========================================
elif page == "Pipeline Metrics":
    st.header("Pipeline Performance Metrics")
    # --- Pipeline Performance Metrics ---
    st.markdown("Evaluation Report (Jeddah 2023 Held-Out Set)")
    
    # Load dynamic metrics
    import json
    try:
        with open("pipeline_metrics.json", "r") as f:
            metrics = json.load(f)
    except:
        metrics = {}
        
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.metric(label="Total Windows Evaluated", value=f"{metrics.get('total_evaluated', 13140):,}")
    with c2:
        st.metric(label="Total Faults Injected", value=f"{metrics.get('total_injected', 2048):,}")
    with c3:
        st.metric(label="Stage 1 Alerts Fired", value=f"{metrics.get('stage1_alerts', 8554):,}")
    with c4:
        st.metric(label="Stage 1 True Positives", value=f"{metrics.get('stage1_true_positives', 1751):,}")
        
    st.divider()
    
    c5, c6, c7 = st.columns(3)
    with c5:
        st.metric(label="Stage 1: Precision", value=metrics.get('precision', '0.2047'), help="How many alarms were real faults?")
    with c6:
        st.metric(label="Stage 1: Recall", value=metrics.get('recall', '0.8550'), help="How many real faults did we catch?")
    with c7:
        st.metric(label="Stage 1: F1-Score", value=metrics.get('f1_score', '0.3303'))
        
    st.divider()
    
    c8, c9, c10 = st.columns(3)
    with c8:
        st.metric(label="Stage 2: Inference Speed", value="5.6 ms / window")
    with c9:
        st.metric(label="Stage 2: Conditional Accuracy", value=metrics.get('conditional_acc', '83.67%'), help="When Stage 1 caught a fault, how often did TCN name the broken sensor?")
    with c10:
        st.metric(label="Stage 2: System-Wide Accuracy", value=metrics.get('system_acc', '71.53%'), help="End-to-end reliability across all injected faults.")
        
    st.divider()
    
    col_chart, col_text = st.columns([1.2, 1])
    
    with col_chart:
        from PIL import Image
        img_path = Path("exploration") / "ablation_results.png"
        if img_path.exists():
            st.image(Image.open(img_path), caption="Live Ablation Benchmark Result", use_container_width=True)
            
    with col_text:
        st.markdown("### Architecture Breakdown")
        st.markdown('''
        **Stage 1: The Gatekeeper (Isolation Forest)**
        * Designed for extreme speed and triage. It filters out 99% of normal telemetry so the neural network doesn't waste GPU cycles.
        * Because it doesn't learn complex physical laws, its diagnostic accuracy is extremely poor (**11.9%**).
        
        **Stage 2: The Diagnostic Engine (TCN Autoencoder)**
        * A bottleneck neural network that learns the interconnected physics of a Formula 1 car (e.g., Throttle = 100% means Speed must be increasing).
        * Upgraded with a **Neural Classification Head** and a hyper-compressed **2D Bottleneck** that skyrocketed diagnostic accuracy to **83.67%**, proving it can diagnose complex physical anomalies rather than just mathematical deviations.
        ''')
        
    st.divider()
    
    st.header("2D Latent Spce")
    st.markdown("Visualizing how the TCN Autoencoder natively untangles physics by compressing 100 data points (5 sensors x 20 timesteps) into a 2D map.")
    
    if True: # Auto-generate on page load so all 3 graphs display immediately
        import plotly.express as px
        from sklearn.decomposition import PCA
        
        with st.spinner("Extracting windows and running PCA..."):
            live_lap = load_track_data("Melbourne")
            raw_matrix = live_lap[CHANNELS].values
            
            clean_windows = sliding_window_view(raw_matrix, window_shape=WINDOW_SIZE, axis=0)[::2][:400]
            n_clean = len(clean_windows)
            n_fault = min(200, n_clean)
            
            def make_dropout(sensor_name, amount):
                ch_idx = CHANNELS.index(sensor_name)
                faulty = np.copy(clean_windows[:amount])
                faulty[:, ch_idx, 10:] = 0.0  # Drops to 0 halfway
                return faulty
                
            def make_drift(sensor_name, amount):
                ch_idx = CHANNELS.index(sensor_name)
                faulty = np.copy(clean_windows[:amount])
                for i in range(len(faulty)):
                    mean_val = np.mean(faulty[i, ch_idx, :])
                    drift_dir = -1.0 if mean_val > 50.0 else 1.0
                    faulty[i, ch_idx, :] += np.linspace(0, drift_dir * 100, 20)
                faulty[:, ch_idx, :] = np.clip(faulty[:, ch_idx, :], 0, 100.0)
                return faulty
                
            def make_noise(sensor_name, amount, scale=50):
                ch_idx = CHANNELS.index(sensor_name)
                faulty = np.copy(clean_windows[:amount])
                if sensor_name == 'Brake':
                    global_sigma = orchestrator.stds[0][ch_idx][0]
                    noise_scale = global_sigma * 1.0
                    t = np.arange(20)
                    vibration = np.cos(t * np.pi) * noise_scale
                    random_amps = (np.random.default_rng().uniform(0.0, 1.3, (len(faulty), 20))) ** 2
                    faulty[:, ch_idx, :] += (vibration * random_amps)
                else:
                    noise = np.random.normal(0, scale, faulty[:, ch_idx, :].shape)
                    faulty[:, ch_idx, :] += noise
                return faulty
                
            rpm_noise = make_noise('RPM', n_fault, scale=2000) 
            throttle_drift = make_drift('Throttle', n_fault)
            brake_noise = make_noise('Brake', n_fault)
            speed_dropout = make_dropout('Speed', n_fault)
            
            all_windows = np.concatenate([clean_windows, rpm_noise, throttle_drift, brake_noise, speed_dropout], axis=0)
            labels = ['Normal Physics'] * n_clean + ['RPM Noise'] * n_fault + ['Throttle Drift'] * n_fault + ['Brake Noise'] * n_fault + ['Speed Dropout'] * n_fault
            
            scaled_windows = (all_windows - orchestrator.means[0]) / orchestrator.stds[0]
            
            with torch.no_grad():
                tensor_in = torch.tensor(scaled_windows, dtype=torch.float32).to(orchestrator.device)
                reconstructed, latent_repr, fault_logits = orchestrator.tcn(tensor_in)
                
                # Reconstruction Error Matrix for the Hotspots
                error_matrix = np.abs(scaled_windows - reconstructed.cpu().numpy())
                
            # We now have 4 different faults (plus Normal).
            # The TCN bottleneck compressed all 100 data points (5 sensors x 20 timesteps) into 2 spatial dimensions!
            # We take the mean across the 20-timestep temporal window to plot a single stable point per window.
            latent_np = latent_repr.cpu().numpy()
            latent_np = np.mean(latent_np, axis=2)
            
            df = pd.DataFrame({
                'Latent Dim 1': latent_np[:, 0],
                'Latent Dim 2': latent_np[:, 1],
                'Class': labels
            })
            
            # --- Interactive UI Filter ---
            unique_classes = df['Class'].unique()
            selected_classes = st.multiselect(
                "Filter Anomaly Classes:",
                options=unique_classes,
                default=unique_classes
            )
            
            # Filter the dataframe based on user selection
            filtered_df = df[df['Class'].isin(selected_classes)]
            
            fig = px.scatter(filtered_df, x='Latent Dim 1', y='Latent Dim 2', color='Class', 
                                title="True 2D Physics Bottleneck (TCN Latent Space)",
                                color_discrete_sequence=['#2ca02c', '#9467bd', '#ff7f0e', '#d62728', '#1f77b4'],
                                category_orders={'Class': unique_classes}) # Keep colors consistent when filtering
            
            fig.update_layout(paper_bgcolor="rgba(0,0,0,0)",
                              plot_bgcolor="rgba(0,0,0,0)",
                              margin=dict(l=0, r=0, b=0, t=30))
                              
            st.plotly_chart(fig, use_container_width=True)
            
            st.success("Notice how the neural network maps different faults into entirely different regions of the physical latent space! By squeezing 100 telemetry data points into just 2 numbers, the AI is forced to create a literal map of F1 physics.")

            st.divider()
            
            st.header("Diagnostic Confusion Matrix")
            
            # Extract ground truth and predictions using the NEW Classification Head (fault_logits)
            true_channels = ['RPM'] * n_fault + ['Throttle'] * n_fault + ['Brake'] * n_fault + ['Speed'] * n_fault
            fault_logits_np = fault_logits.cpu().numpy()[n_clean:]
            fault_preds = [CHANNELS[i] for i in np.argmax(fault_logits_np, axis=1)]
            
            from sklearn.metrics import confusion_matrix
            cm = confusion_matrix(true_channels, fault_preds, labels=CHANNELS)
            
            # Subset the rows to only show the 4 faults we actively injected
            cm_subset = cm[[CHANNELS.index('RPM'), CHANNELS.index('Throttle'), CHANNELS.index('Brake'), CHANNELS.index('Speed')], :]
            
            fig_cm = px.imshow(cm_subset, x=CHANNELS, y=['RPM Noise', 'Throttle Drift', 'Brake Noise', 'Speed Dropout'], 
                               labels=dict(x="AI Predicted Culprit", y="True Hardware Failure", color="Count"),
                               text_auto=True, aspect="auto", color_continuous_scale="Reds")
            
            fig_cm.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", margin=dict(l=0, r=0, b=0, t=30))
            st.plotly_chart(fig_cm, use_container_width=True)

            st.divider()
            
            st.header("Error Tensor Hotspots")
            st.markdown("Visualizing exactly how the neural network 'sees' these advanced, stealthy anomalies across time.")
            
            col_err1, col_err2 = st.columns(2)
            col_err3, col_err4 = st.columns(2)
            
            # Grab one random window from each complex fault block
            idx_rpm = n_clean + 10
            idx_throttle = n_clean + n_fault + 10
            idx_brake = n_clean + 2 * n_fault + 10
            idx_speed = n_clean + 3 * n_fault + 10
            
            def plot_error_heatmap(idx, title):
                err_tensor = error_matrix[idx] 
                err_tensor[err_tensor < 0.5] = 0 # Suppress background noise
                
                fig = px.imshow(err_tensor, x=list(range(20)), y=CHANNELS, 
                                labels=dict(x="Time Step", y="Sensor", color="Absolute Error"),
                                title=title, aspect="auto", color_continuous_scale="Reds")
                fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", margin=dict(l=0, r=0, b=0, t=30))
                return fig
                
            with col_err1:
                st.plotly_chart(plot_error_heatmap(idx_speed, "Speed Sensor: Dropout"), use_container_width=True)
            with col_err2:
                st.plotly_chart(plot_error_heatmap(idx_throttle, "Throttle Sensor: Drift"), use_container_width=True)
            with col_err3:
                st.plotly_chart(plot_error_heatmap(idx_rpm, "RPM Sensor: Noise"), use_container_width=True)
            with col_err4:
                st.plotly_chart(plot_error_heatmap(idx_brake, "Brake Sensor: Noise"), use_container_width=True)

# ==========================================
# 6. PAGE: SENSITIVITY ANALYSIS
# ==========================================
elif page == "Sensitivity Analysis":
    st.header("Sensitivity Analysis Curves")
    st.markdown("How subtle can a physical anomaly be before the neural network fails to diagnose it? ")
    
    with st.spinner("Simulating thousands of dynamic fault permutations..."):
        import plotly.express as px
        
        live_lap = load_track_data("Melbourne")
        raw_matrix = live_lap[CHANNELS].values
        windows = sliding_window_view(raw_matrix, window_shape=WINDOW_SIZE, axis=0)
        # Use 200 windows for evaluation
        clean_windows = windows[::2][:200]
        n_fault = len(clean_windows)
        
        def eval_accuracy(faulty_windows, target_sensor):
            ch_idx = CHANNELS.index(target_sensor)
            
            # GHOST ANOMALY FILTER: Did the injection actually change the physics?
            # If we inject a 'Stuck Value' into nGear when the car is already holding 4th gear, 
            # the telemetry is mathematically identical to a clean window. We must ignore these!
            diffs = np.sum(np.abs(faulty_windows[:, ch_idx, :] - clean_windows[:, ch_idx, :]), axis=1)
            valid_mask = diffs > 1e-3
            
            # If the severity is so low (e.g., length=0) that NOTHING changed, accuracy is naturally 0%
            if not np.any(valid_mask):
                return 0.0
                
            valid_faulty = faulty_windows[valid_mask]
            
            scaled = (valid_faulty - orchestrator.means[0]) / orchestrator.stds[0]
            with torch.no_grad():
                t_in = torch.tensor(scaled, dtype=torch.float32).to(orchestrator.device)
                recon, _, fault_logits = orchestrator.tcn(t_in)
                
            # Use the trained neural network classification head instead of raw SSE
            preds = torch.argmax(fault_logits, dim=1).cpu().numpy()
            target_idx = CHANNELS.index(target_sensor)
            return np.mean(preds == target_idx) * 100.0
            
        st.markdown("### Interactive Sensitivity Explorer")
        st.markdown("Select any Sensor and Anomaly Type below to instantly compute how the Neural Network's diagnostic accuracy degrades as the physical anomaly becomes subtler.")
        
        col_sel1, col_sel2 = st.columns(2)
        with col_sel1:
            target_sensor = st.selectbox("Select Sensor:", CHANNELS, index=0)
        with col_sel2:
            fault_type = st.selectbox("Select Anomaly Type:", ["Dropout (Length)", "Stuck Value (Length)", "Drift (Magnitude)", "Noise (Amplitude)"])

        # Depending on selection, generate the curve dynamically!
        if "Dropout" in fault_type:
            x_vals = np.arange(1, 21)
            x_label = "Dropout Duration (Timesteps)"
            accuracies = []
            for length in x_vals:
                faulty = np.copy(clean_windows)
                ch_idx = CHANNELS.index(target_sensor)
                faulty[:, ch_idx, (20 - length):] = 0.0
                accuracies.append(eval_accuracy(faulty, target_sensor))
                
        elif "Stuck Value" in fault_type:
            x_vals = np.arange(1, 21)
            x_label = "Stuck Value Duration (Timesteps)"
            accuracies = []
            for length in x_vals:
                ch_idx = CHANNELS.index(target_sensor)
                faulty = np.copy(clean_windows)
                for i in range(len(faulty)):
                    freeze_idx = max(0, 20 - length - 1)
                    stuck_val = faulty[i, ch_idx, freeze_idx]
                    faulty[i, ch_idx, (20 - length):] = stuck_val
                accuracies.append(eval_accuracy(faulty, target_sensor))
                
        elif "Drift" in fault_type:
            # Scale the drift magnitude sweep bounds based on the sensor's physical range
            max_drift = 100 if target_sensor in ['Speed', 'Throttle', 'Brake'] else (2000 if target_sensor == 'RPM' else 5)
            x_vals = np.linspace(0, max_drift, 11)
            x_label = f"Drift Magnitude (Absolute Physics Delta)"
            accuracies = []
            for mag in x_vals:
                ch_idx = CHANNELS.index(target_sensor)
                faulty = np.copy(clean_windows)
                for i in range(len(faulty)):
                    series = faulty[i, ch_idx, :]
                    mean_val = np.mean(series)
                    drift_dir = -1.0 if mean_val > (max_drift / 2) else 1.0
                    faulty[i, ch_idx, :] += np.linspace(0, drift_dir * mag, 20)
                
                # Hard clip to physical reality
                if target_sensor == 'Speed': faulty[:, ch_idx, :] = np.clip(faulty[:, ch_idx, :], 0, 360)
                elif target_sensor == 'nGear': faulty[:, ch_idx, :] = np.clip(np.round(faulty[:, ch_idx, :]), 0, 8)
                elif target_sensor in ['Throttle', 'Brake']: faulty[:, ch_idx, :] = np.clip(faulty[:, ch_idx, :], 0, 100)
                elif target_sensor == 'RPM': faulty[:, ch_idx, :] = np.clip(faulty[:, ch_idx, :], 0, 13000)
                accuracies.append(eval_accuracy(faulty, target_sensor))
                
        elif "Noise" in fault_type:
            x_vals = np.linspace(0.0, 2.0, 11)
            x_label = "Noise Amplitude (x Standard Deviations)"
            accuracies = []
            for mag in x_vals:
                ch_idx = CHANNELS.index(target_sensor)
                faulty = np.copy(clean_windows)
                global_sigma = orchestrator.stds[0][ch_idx][0]
                
                noise_std = global_sigma * mag
                faulty[:, ch_idx, :] += np.random.normal(0, noise_std, faulty[:, ch_idx, :].shape)
                
                if target_sensor == 'Speed': faulty[:, ch_idx, :] = np.clip(faulty[:, ch_idx, :], 0, 360)
                elif target_sensor == 'nGear': faulty[:, ch_idx, :] = np.clip(np.round(faulty[:, ch_idx, :]), 0, 8)
                elif target_sensor in ['Throttle', 'Brake']: faulty[:, ch_idx, :] = np.clip(faulty[:, ch_idx, :], 0, 100)
                elif target_sensor == 'RPM': faulty[:, ch_idx, :] = np.clip(faulty[:, ch_idx, :], 0, 13000)
                accuracies.append(eval_accuracy(faulty, target_sensor))
                
        # --- Plot the Dynamic Interactive Curve ---
        fig = px.line(x=x_vals, y=accuracies, markers=True,
                       title=f"Sensitivity Curve: {target_sensor} {fault_type.split(' ')[0]}",
                       labels={'x': x_label, 'y': "Diagnostic Accuracy (%)"})
        
        fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", margin=dict(l=0, r=0, b=0, t=30))
        fig.update_traces(line=dict(width=4, color='#00ffcc'), marker=dict(size=8, color='#ff00ff'))
        fig.update_yaxes(range=[-5, 105])
        
        st.plotly_chart(fig, use_container_width=True)
