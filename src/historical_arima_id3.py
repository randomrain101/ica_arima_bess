#%%
def warn(*args, **kwargs):
    pass
import warnings
warnings.warn = warn

warnings.filterwarnings("ignore", category=DeprecationWarning)

import pandas as pd
import numpy as np
import glob
import os
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from statsforecast.models import AutoARIMA
from joblib import Parallel, delayed
import warnings
warnings.filterwarnings('ignore', category=FutureWarning)

#%%
df_prices = pd.read_csv(os.path.join(os.path.dirname(__file__), '..', 'data', 'id3_prices.csv'), index_col=0, parse_dates=True)

#%%
# 1 year rolling window (356 days)
window = 356

def process_window(window_start, df_prices, window):
    """
    Process a single window for ARIMA prediction
    For each hour, use the last 356 values of that specific hour to predict the next day's price
    """
    # Get the window of data
    price_window = df_prices.iloc[window_start:window_start + window]
    
    # Predict next day's prices for each hour
    next_day_predictions = []
    
    for hour_col in price_window.columns:
        # Get time series for this specific hour
        hour_series = price_window[hour_col].dropna()
        
        if len(hour_series) < 10:  # Need minimum data points
            next_day_predictions.append(np.nan)
            continue
            
        try:
            # Fit ARIMA model to this hour's historical prices
            arima_model = AutoARIMA().fit(hour_series.values)
            
            # Predict next value
            forecast = arima_model.predict(h=1)['mean'][0]
            next_day_predictions.append(forecast)
            
        except Exception as e:
            # If ARIMA fails, use simple mean as fallback
            next_day_predictions.append(hour_series.mean())
    
    # Create result dataframe
    next_date = price_window.index[-1] + pd.Timedelta(days=1)
    return pd.DataFrame([next_day_predictions],
                        columns=df_prices.columns,
                        index=[next_date])

#%%
# Process all windows in parallel
print(f"Processing {len(df_prices) - window} windows with {window}-day lookback...")
window_results = Parallel(n_jobs=-1)(
    delayed(process_window)(window_start, df_prices, window)
    for window_start in tqdm(range(len(df_prices) - window))
)

#%%
# Combine results
df_arima_predictions = pd.concat(window_results, axis=0)
df_arima_predictions = df_arima_predictions.sort_index()

# Save results
output_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'arima_predictions.csv')
df_results.to_csv(output_path)
print(f"Saved ARIMA predictions to {output_path}")

#%%
# Quick validation
print(f"Predictions shape: {df_results.shape}")
print(f"Date range: {df_results.index.min()} to {df_results.index.max()}")
print(f"Sample predictions:\n{df_results.head()}")

#%%
