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
    
    # FIX: Convert dates.hour to a numpy array first to avoid creating an immutable Index
    hours = dates.hour.to_numpy()
    daily_cycle = 20 * np.sin(2 * np.pi * hours / 24)
    noise = np.random.normal(0, 10, size=len(dates))
    
    response_times = base_latency + daily_cycle + noise
    
    # FIX: Ensure response_times is a modifiable numpy array
    response_times = np.array(response_times)
    
    # Inject Anomalies (Random spikes)
    # We'll make about 2% of the data anomalous
    n_anomalies = int(len(dates) * 0.02)
    anomaly_indices = np.random.choice(len(dates), n_anomalies, replace=False)
    
    # Anomalies will be significantly higher (e.g., server load spike)
    response_times[anomaly_indices] += np.random.randint(100, 400, size=n_anomalies)
    
    # Ensure no negative latencies
    response_times = np.maximum(response_times, 10)
    
    return pd.DataFrame({'timestamp': dates, 'response_time': response_times})
