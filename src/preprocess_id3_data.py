#%%
import glob
import pandas as pd
import os

# Load all CSV files containing 'id3' in their filename
id3_files = glob.glob('../data/id3/*.csv')

#%%
yearly_dfs = []
# Process each file
for file_path in id3_files:
    # Read the CSV file
    df_year = pd.read_csv(file_path)
    
    # Determine year from file path or data - adjust as needed
    year = int(file_path.split("_")[-1].split(".")[0])
    # Create hourly date range for the year (handles leap years automatically)
    start_date = f'{year}-01-01'
    end_date = f'{year}-12-31 23:00:00'
    hourly_index = pd.date_range(start=start_date, end=end_date, freq='H')
    
    # Set the hourly index
    df_year.index = hourly_index[:len(df_year)]
    
    yearly_dfs.append(df_year)

    print(f"Processed {file_path}: {len(df_year)} rows with hourly index")

#%%
# Concatenate all dataframes
id3_series = pd.concat(yearly_dfs).sort_index().rename(columns={'Value': 'price'})["price"]

#%%
# Create pivot table with hour of day as columns
df_daily_prices = id3_series.to_frame().reset_index()
df_daily_prices['hour'] = df_daily_prices['index'].dt.hour
df_daily_prices['date'] = df_daily_prices['index'].dt.date
df_daily_prices = df_daily_prices.pivot(index='date', columns='hour', values='price').ffill().dropna()
df_daily_prices.index = pd.to_datetime(df_daily_prices.index)

#%%
df_daily_prices.to_csv(os.path.join(os.path.dirname(__file__), '..', 'data', 'id3_prices.csv'))

# %%
