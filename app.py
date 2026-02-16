import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.ensemble import IsolationForest
from datetime import datetime, timedelta

# --- Page Config ---
st.set_page_config(page_title="Site Anomaly Detector", layout="wide")

# --- Helper Function: Generate Mock Data ---
def generate_site_data(days=30):
    """
    Generates synthetic server response time data for demonstration.
    Returns a DataFrame with timestamps and response times (ms).
    """
    # Create date range
    end_date = datetime.now()
    start_date = end_date - timedelta(days=days)
    dates = pd.date_range(start=start_date, end=end_date, freq='H') # Hourly data
    
    # Generate 'Normal' Traffic (Sine wave pattern + random noise)
    # Simulating daily cycles (24 hour period)
    base_latency = 120  # Base ms
    daily_cycle = 20 * np.sin(2 * np.pi * dates.hour / 24)
    noise = np.random.normal(0, 10, size=len(dates))
    response_times = base_latency + daily_cycle + noise
    
    # Inject Anomalies (Random spikes)
    # We'll make about 2% of the data anomalous
    n_anomalies = int(len(dates) * 0.02)
    anomaly_indices = np.random.choice(len(dates), n_anomalies, replace=False)
    
    # Anomalies will be significantly higher (e.g., server load spike)
    response_times[anomaly_indices] += np.random.randint(100, 400, size=n_anomalies)
    
    # Ensure no negative latencies
    response_times = np.maximum(response_times, 10)
    
    return pd.DataFrame({'timestamp': dates, 'response_time': response_times})

# --- Main App Interface ---
st.title("🔎 Website Anomaly Detection System")
st.markdown("""
This tool uses an **Isolation Forest (Unsupervised Learning)** model to detect 
unusual spikes in website latency (response time).
""")

# Sidebar for controls
with st.sidebar:
    st.header("Configuration")
    target_url = st.text_input("Enter Website URL", "https://google.com")
    contamination = st.slider("Anomaly Sensitivity", 0.01, 0.1, 0.02, 
                              help="The proportion of outliers in the data set.")
    days_to_analyze = st.slider("Days of History", 7, 90, 30)
    run_btn = st.button("Analyze Site Health")

# --- Application Logic ---
if run_btn and target_url:
    with st.spinner(f"Fetching and analyzing data for {target_url}..."):
        
        # 1. Get Data (Simulated for this demo)
        df = generate_site_data(days=days_to_analyze)
        
        # 2. Prepare Data for ML
        # Isolation Forest expects a 2D array
        X = df[['response_time']]
        
        # 3. Initialize and Train Model
        # contamination determines the threshold for what is considered an anomaly
        model = IsolationForest(contamination=contamination, random_state=42)
        model.fit(X)
        
        # 4. Predict Anomalies
        # -1 is anomaly, 1 is normal
        df['anomaly_score'] = model.predict(X)
        df['status'] = df['anomaly_score'].apply(lambda x: 'Anomaly' if x == -1 else 'Normal')
        
        # Calculate stats
        total_anomalies = df[df['status'] == 'Anomaly'].shape[0]
        avg_latency = df['response_time'].mean()
        
        # --- Display Results ---
        
        # Metrics Row
        col1, col2, col3 = st.columns(3)
        col1.metric("Avg Response Time", f"{avg_latency:.0f} ms")
        col2.metric("Total Data Points", len(df))
        col3.metric("Anomalies Detected", total_anomalies, delta_color="inverse")
        
        st.divider()
        
        # Plotting with Plotly
        st.subheader("Latency Analysis Graph")
        
        # We assign colors: Normal = Blue, Anomaly = Red
        fig = px.scatter(
            df, 
            x='timestamp', 
            y='response_time', 
            color='status',
            color_discrete_map={'Normal': '#00B4D8', 'Anomaly': '#FF4B4B'},
            title=f"Response Time History for {target_url}",
            labels={'response_time': 'Latency (ms)', 'timestamp': 'Date'},
            hover_data=['response_time']
        )
        
        # Add a line for the trend
        fig.add_scatter(x=df['timestamp'], y=df['response_time'], mode='lines', 
                        line=dict(color='gray', width=1), opacity=0.3, showlegend=False)
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Show the raw anomaly data
        if total_anomalies > 0:
            st.warning("⚠️ Detailed Anomaly Log")
            st.dataframe(
                df[df['status'] == 'Anomaly'][['timestamp', 'response_time']]
                .sort_values(by='response_time', ascending=False),
                use_container_width=True
            )
        else:
            st.success("No significant anomalies detected in the generated timeframe.")

elif not target_url:
    st.info("Please enter a URL in the sidebar to begin.")
