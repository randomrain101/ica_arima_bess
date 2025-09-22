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
from pmdarima.arima import auto_arima
from joblib import Parallel, delayed
import warnings
warnings.filterwarnings('ignore', category=FutureWarning)

#%%
df = pd.read_csv(os.path.join(os.path.dirname(__file__), '..', 'data', 'id3_prices.csv'), index_col=0, parse_dates=True)

#%%
# 1 year rolling window (356 days)
window = 356
df_pred = pd.DataFrame(columns=df.columns)

def process_window(i, df, window):
    """
    Process a single window for ARIMA prediction
    For each hour, use the last 356 values of that specific hour to predict the next day's price
    """
    # Get the window of data
    df_window = df.iloc[i:i + window]
    
    # Predict next day's prices for each hour
    predictions = []
    
    for hour_col in df_window.columns:
        # Get time series for this specific hour
        hour_series = df_window[hour_col].dropna()
        
        if len(hour_series) < 10:  # Need minimum data points
            predictions.append(np.nan)
            continue
            
        try:
            # Fit ARIMA model to this hour's historical prices
            model = auto_arima(
                hour_series, 
                suppress_warnings=True, 
                error_action='ignore'
            )
            
            # Predict next value
            forecast = model.predict(n_periods=1)[0]
            predictions.append(forecast)
            
        except Exception as e:
            # If ARIMA fails, use simple mean as fallback
            predictions.append(hour_series.mean())
    
    # Create result dataframe
    next_date = df_window.index[-1] + pd.Timedelta(days=1)
    result = pd.DataFrame([predictions], 
                         columns=df.columns, 
                         index=[next_date])
    
    print(f"Finished window {i} of {len(df) - window}")
    return result

#%%
# Process all windows in parallel
print(f"Processing {len(df) - window} windows with {window}-day lookback...")
results = Parallel(n_jobs=-1)(
    delayed(process_window)(i, df, window) 
    for i in tqdm(range(len(df) - window))
)

#%%
# Combine results
df_results = pd.concat(results, axis=0)
df_results = df_results.sort_index()

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
