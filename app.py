import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import seasonal_decompose


st.subheader("Time Series Decomposition")


df = df.set_index('timestamp') 

result = seasonal_decompose(df['response_time'], model='additive', period=24)

# 3. Plotting
fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1, figsize=(10, 8))
result.observed.plot(ax=ax1, title='Observed (Raw Data)')
result.trend.plot(ax=ax2, title='Trend (Direction)')
result.seasonal.plot(ax=ax3, title='Seasonality (Repeating Pattern)')
result.resid.plot(ax=ax4, title='Residuals (Noise/Anomalies)')

plt.tight_layout()
st.pyplot(fig)
