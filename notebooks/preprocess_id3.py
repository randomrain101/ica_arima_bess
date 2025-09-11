#%%
import glob
import pandas as pd
import os

# Load all CSV files containing 'id3' in their filename
id3_files = glob.glob('../data/id3/*.csv')

#%%
df_list = []
# Process each file
for file_path in id3_files:
    # Read the CSV file
    df = pd.read_csv(file_path)
    
    # Determine year from file path or data - adjust as needed
    year = int(file_path.split("_")[-1].split(".")[0])  # You may want to extract this from filename or data
    #year = 2024
    # Create hourly date range for the year (handles leap years automatically)
    start_date = f'{year}-01-01'
    end_date = f'{year}-12-31 23:00:00'
    hourly_index = pd.date_range(start=start_date, end=end_date, freq='H')
    
    # Set the hourly index
    df.index = hourly_index[:len(df)]
    
    df_list.append(df)

    print(f"Processed {file_path}: {len(df)} rows with hourly index")

#%%
# Concatenate all dataframes
df_id3 = pd.concat(df_list).sort_index().rename(columns={'Value': 'price'})["price"]#.resample('h').last().ffill()

#%%
# Create pivot table with hour of day as columns
df_pivot = df_id3.to_frame().reset_index()
df_pivot['hour'] = df_pivot['index'].dt.hour
df_pivot['date'] = df_pivot['index'].dt.date
df_pivot = df_pivot.pivot(index='date', columns='hour', values='price').ffill().dropna()
df_pivot.index = pd.to_datetime(df_pivot.index)

#%%
df_pivot.to_csv(os.path.join(os.path.dirname(__file__), '..', 'data', 'id3_prices.csv'))

# %%
