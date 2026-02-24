import streamlit as st
import pandas as pd
import plotly.express as px
import requests
import time
import numpy as np
from datetime import datetime
from sklearn.ensemble import IsolationForest

# --- Page Config ---
st.set_page_config(page_title="Live ML Anomaly Detector", layout="wide")
st.title("🤖 Live AI/ML Website Anomaly Detection")
st.markdown("This tool uses an **Isolation Forest** model that *retrains in real-time* on a sliding window of data.")

# --- 1. Session State (The Short-Term Memory) ---
# We store the last N data points to train the model
if 'data_buffer' not in st.session_state:
    st.session_state.data_buffer = []

# --- 2. Sidebar Controls ---
with st.sidebar:
    st.header("ML Configuration")
    target_url = st.text_input("Target URL", "https://www.google.com")
    interval = st.slider("Ping Interval (seconds)", 0.5, 5.0, 1.0)
    
    # ML Hyperparameters
    window_size = st.slider("Training Window Size", 20, 200, 50, 
                           help="How many recent data points the AI learns from.")
    contamination = st.slider("Anomaly Sensitivity", 0.01, 0.5, 0.1,
                             help="The % of data the model expects to be anomalous.")
    
    is_running = st.checkbox("🟢 Start AI Monitoring")
    
    if st.button("Clear Memory"):
        st.session_state.data_buffer = []

# --- 3. The ML Loop ---
chart_placeholder = st.empty()
stats_placeholder = st.empty()

if is_running:
    while True:
        # A. Collect Real Data (Ping)
        try:
            start_time = time.time()
            requests.get(target_url, timeout=3)
            latency = (time.time() - start_time) * 1000  # ms
        except:
            latency = 3000  # High penalty for timeout
        
        timestamp = datetime.now()
        
        # B. Update Buffer (Sliding Window)
        st.session_state.data_buffer.append({'time': timestamp, 'latency': latency})
        
        # Keep buffer size fixed (Remove oldest if too big)
        if len(st.session_state.data_buffer) > window_size:
            st.session_state.data_buffer.pop(0)
            
        # C. The "Brain" - Train ML Model on the fly
        df = pd.DataFrame(st.session_state.data_buffer)
        current_status = "Insufficient Data"
        
        # We need at least 10 points to train a decent model
        if len(df) >= 10:
            # 1. Prepare features (Isolation Forest needs 2D array)
            X = df[['latency']].values
            
            # 2. Train Model (This happens every loop!)
            model = IsolationForest(contamination=contamination, random_state=42)
            model.fit(X)
            
            # 3. Predict Anomalies (-1 = Anomaly, 1 = Normal)
            df['anomaly'] = model.predict(X)
            
            # Check the status of the *latest* point we just added
            latest_is_anomaly = df.iloc[-1]['anomaly'] == -1
            current_status = "⚠️ ANOMALY DETECTED" if latest_is_anomaly else "Normal"
            
            # Define colors for the graph based on ML results
            df['color'] = df['anomaly'].apply(lambda x: 'red' if x == -1 else '#00CC96')
        else:
            # Not enough data yet, show neutral colors
            df['color'] = 'gray'
            latest_is_anomaly = False

        # D. Visualization
        with chart_placeholder.container():
            fig = px.bar(
                df, 
                x='time', 
                y='latency', 
                title=f"Real-Time Isolation Forest Analysis (Window: {len(df)})",
                template="plotly_dark"
            )
            # Apply the colors determined by the ML model
            fig.update_traces(marker_color=df['color'])
            st.plotly_chart(fig, use_container_width=True)

        # E. Stats Display
        with stats_placeholder.container():
            col1, col2, col3 = st.columns(3)
            col1.metric("Live Latency", f"{latency:.0f} ms")
            col2.metric("ML Confidence", f"{len(df)}/{window_size} samples")
            
            if latest_is_anomaly:
                col3.error(f"{current_status}")
            else:
                col3.success(f"{current_status}")

        time.sleep(interval)

else:
    st.info("Start the monitor to let the AI learn the network patterns.")
