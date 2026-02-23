import streamlit as st
import pandas as pd
import plotly.express as px
from sklearn.ensemble import IsolationForest

# --- Page Config ---
st.set_page_config(page_title="Real-Time Anomaly Detector", layout="wide")

st.title("🔎 Anomaly Detection on Real Data")
st.markdown("""
Upload your CSV file (e.g., server logs, sales data, sensor readings). 
The app will automatically detect spikes and irregularities using **Isolation Forest**.
""")

# --- 1. File Upload Section ---
uploaded_file = st.sidebar.file_uploader("Upload CSV", type=["csv"])

if uploaded_file is not None:
    # Read the file
    try:
        df = pd.read_csv(uploaded_file)
        
        # Check if data loaded correctly
        st.write("### Raw Data Preview")
        st.dataframe(df.head())

        # --- 2. Column Selection ---
        # We need the user to tell us which column is Time and which is the Value
        st.sidebar.header("Data Mapping")
        
        # Try to guess the date column (looks for 'date', 'time', 'timestamp' in name)
        date_candidates = [col for col in df.columns if 'date' in col.lower() or 'time' in col.lower()]
        default_date = date_candidates[0] if date_candidates else df.columns[0]
        
        time_col = st.sidebar.selectbox("Select Time Column", df.columns, index=df.columns.get_loc(default_date))
        value_col = st.sidebar.selectbox("Select Value Column (Metric)", [c for c in df.columns if c != time_col])
        
        contamination = st.sidebar.slider("Anomaly Sensitivity", 0.01, 0.20, 0.02, 
                                          help="Higher = more points flagged as anomalies.")

        # --- 3. Data Preprocessing ---
        # Convert time column to datetime objects
        df[time_col] = pd.to_datetime(df[time_col], errors='coerce')
        
        # Drop rows where date parsing failed
        df = df.dropna(subset=[time_col])
        
        # Sort by time (critical for plotting)
        df = df.sort_values(by=time_col)

        # --- 4. The AI (Isolation Forest) ---
        # The model requires a 2D array, so we reshape the value column
        X = df[[value_col]].values
        
        # Train model
        model = IsolationForest(contamination=contamination, random_state=42)
        model.fit(X)
        
        # Predict (-1 = Anomaly, 1 = Normal)
        df['anomaly_score'] = model.predict(X)
        df['status'] = df['anomaly_score'].apply(lambda x: 'Anomaly' if x == -1 else 'Normal')
        
        # --- 5. Visualization ---
        st.divider()
        st.subheader(f"Anomaly Analysis: {value_col}")

        # Metrics
        anomalies = df[df['status'] == 'Anomaly']
        col1, col2, col3 = st.columns(3)
        col1.metric("Total Data Points", len(df))
        col2.metric("Anomalies Found", len(anomalies), delta_color="inverse")
        col3.metric("Avg Value", f"{df[value_col].mean():.2f}")

        # Plotly Chart
        fig = px.scatter(
            df, 
            x=time_col, 
            y=value_col, 
            color='status',
            color_discrete_map={'Normal': '#00B4D8', 'Anomaly': '#FF4B4B'},
            title=f"{value_col} over Time (Red = Anomaly)",
            hover_data=[value_col]
        )
        
        # Add a line connecting the dots (makes time series easier to read)
        fig.add_scatter(x=df[time_col], y=df[value_col], mode='lines', 
                        line=dict(color='gray', width=1), opacity=0.3, showlegend=False)

        st.plotly_chart(fig, use_container_width=True)

        # Show the actual bad data points
        if not anomalies.empty:
            st.warning("⚠️ Detected Anomalies List")
            st.dataframe(anomalies[[time_col, value_col]].sort_values(by=value_col, ascending=False))

    except Exception as e:
        st.error(f"Error parsing file: {e}")

else:
    st.info("Waiting for CSV upload...")
    st.write("No data? Download this sample CSV to test:")
    
    # Create a small sample CSV for them to download if they have none
    sample_data = pd.DataFrame({
        'timestamp': pd.date_range(start='2024-01-01', periods=20, freq='H'),
        'cpu_usage': [12, 15, 14, 13, 98, 14, 15, 12, 13, 11, 14, 15, 99, 12, 14, 13, 15, 14, 12, 13]
    })
    st.download_button("Download Sample CSV", sample_data.to_csv(index=False), "sample_data.csv", "text/csv")
