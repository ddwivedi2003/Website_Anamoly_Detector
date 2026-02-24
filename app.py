import streamlit as st
import pandas as pd
import plotly.express as px
import requests
import time
from datetime import datetime

# --- Page Config ---
st.set_page_config(page_title="Live Site Monitor", layout="wide")
st.title("⚡ Real-Time Website Anomaly Monitor")

# --- 1. Session State (The App's Memory) ---
# We need to store data between "refreshes" or it will disappear.
if 'monitor_data' not in st.session_state:
    st.session_state.monitor_data = []

# --- 2. Sidebar Controls ---
with st.sidebar:
    st.header("Settings")
    target_url = st.text_input("Target URL", "https://www.google.com")
    interval = st.slider("Ping Interval (seconds)", 1, 10, 2)
    threshold_multiplier = st.slider("Anomaly Sensitivity", 1.5, 5.0, 2.5,
                                     help="Mark anomaly if latency is X times higher than average.")
    
    # The "Switch" to turn the loop on/off
    is_running = st.checkbox("🔴 Start Live Monitoring")
    
    if st.button("Clear History"):
        st.session_state.monitor_data = []

# --- 3. The Monitoring Loop ---
# This placeholder allows us to update the chart without reloading the whole page
chart_placeholder = st.empty()
stats_placeholder = st.empty()

if is_running:
    while True:
        # A. Ping the Site
        try:
            start_time = time.time()
            response = requests.get(target_url, timeout=5)
            latency = (time.time() - start_time) * 1000  # Convert to milliseconds
            status_code = response.status_code
        except Exception as e:
            latency = 0
            status_code = 500  # Error
        
        # B. Append to History
        timestamp = datetime.now()
        st.session_state.monitor_data.append({
            'Timestamp': timestamp,
            'Latency (ms)': latency,
            'Status': status_code
        })
        
        # Keep only the last 100 points to keep it fast
        if len(st.session_state.monitor_data) > 100:
            st.session_state.monitor_data.pop(0)
            
        # C. Calculate Anomalies (Rolling Z-Score Lite)
        df = pd.DataFrame(st.session_state.monitor_data)
        
        # Calculate dynamic average (Rolling Mean of last 10 points)
        if len(df) > 5:
            rolling_avg = df['Latency (ms)'].rolling(window=10).mean().iloc[-1]
            # If current latency is > Average * Multiplier -> It's an Anomaly
            is_anomaly = latency > (rolling_avg * threshold_multiplier)
        else:
            is_anomaly = False
            rolling_avg = latency

        # D. Update the Chart
        # We assign colors based on the anomaly status we just calculated
        # Note: We rebuild the color list every frame for the chart
        colors = ['red' if (x > rolling_avg * threshold_multiplier and i > 5) else 'blue' 
                  for i, x in enumerate(df['Latency (ms)'])]

        with chart_placeholder.container():
            fig = px.bar(
                df, 
                x='Timestamp', 
                y='Latency (ms)', 
                title=f"Live Latency for {target_url}",
                template="plotly_dark"
            )
            
            # Update bar colors manually to show anomalies in Red
            fig.update_traces(marker_color=colors)
            
            # Add a threshold line
            fig.add_hline(y=rolling_avg * threshold_multiplier, line_dash="dot", 
                          annotation_text="Anomaly Threshold", annotation_position="top right")
            
            st.plotly_chart(fig, use_container_width=True)

        # E. Update Stats
        with stats_placeholder.container():
            col1, col2, col3 = st.columns(3)
            col1.metric("Current Latency", f"{latency:.0f} ms", 
                        delta=f"{latency - rolling_avg:.0f} ms" if len(df) > 1 else 0,
                        delta_color="inverse")
            col2.metric("Status Code", status_code)
            
            if is_anomaly:
                st.error(f"⚠️ Anomaly Detected! Latency spiked to {latency:.0f} ms")
            else:
                st.success("System Normal")

        # F. Sleep before next ping
        time.sleep(interval)
        
        # G. Stop logic: If user unchecks the box, Streamlit will rerun and stop the loop.
else:
    st.info("Check 'Start Live Monitoring' in the sidebar to begin.")
    if len(st.session_state.monitor_data) > 0:
        # Show the static chart if we stopped
        df = pd.DataFrame(st.session_state.monitor_data)
        st.dataframe(df.tail())
