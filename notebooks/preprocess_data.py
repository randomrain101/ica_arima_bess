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
from sklearn.decomposition import FastICA
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, r2_score, cohen_kappa_score, root_mean_squared_error, median_absolute_error, explained_variance_score
from pmdarima.arima import auto_arima
from joblib import Parallel, delayed
import warnings
warnings.filterwarnings('ignore', category=FutureWarning)



#%%
# Get all CSV files from the ../data directory
data_path = os.path.join(os.path.dirname(__file__), '..', 'data')
csv_files = glob.glob(os.path.join(data_path, 'ENERGY*.csv'))

# Import all CSV files into a dictionary of DataFrames
df = pd.concat(map(pd.read_csv, csv_files), ignore_index=False)
datetimes = df["MTU (CET/CEST)"].str.replace(" (CEST)", "").str.replace(" (CET)", "")
df["delivery_start"] = pd.to_datetime(datetimes.str.split(" - ").str[0], format="%d/%m/%Y %H:%M:%S")
df["delivery_end"] = pd.to_datetime(datetimes.str.split(" - ").str[1], format="%d/%m/%Y %H:%M:%S")

# drop quaterly prices, only keep hourly prices
df = df.loc[df["Sequence"] != "Sequence Sequence 2"]
df = df[["delivery_start", "Day-ahead Price (EUR/MWh)"]].set_index("delivery_start").sort_index()
# drop duplicates from quaterly prices
df = df.iloc[::4]

# %%
df = df.pivot_table(index=df.index.date, columns=df.index.hour, values="Day-ahead Price (EUR/MWh)")
df.index = pd.to_datetime(df.index)

#%%
plt.figure(figsize=(12, 8))
sns.heatmap(df.isna().T, cmap='viridis', cbar_kws={'label': 'Missing Values'})
plt.title('Missing Values Heatmap')
plt.xlabel('Hour of Day')
plt.ylabel('Date')
plt.tight_layout()
plt.show()

# use last days price to fill missing values
df = df.ffill()

# %%
plt.figure(figsize=(12, 8))
sns.heatmap(df.T, cmap='viridis', cbar_kws={'label': 'Day-ahead Price (EUR/MWh)'})
plt.title('Day-ahead Electricity Prices Heatmap')
plt.xlabel('Hour of Day')
plt.ylabel('Date')
plt.tight_layout()
plt.show()

#%%
# Filter out the year 2022
df_filtered = df.loc[df.index.year != 2022]

# Create heatmap without 2022 data
plt.figure(figsize=(12, 8))
sns.heatmap(df_filtered.T, cmap='viridis', cbar_kws={'label': 'Day-ahead Price (EUR/MWh)'})
plt.title('Day-ahead Electricity Prices Heatmap (Excluding 2022)')
plt.xlabel('Hour of Day')
plt.ylabel('Date')
plt.tight_layout()
plt.show()

df.to_csv(os.path.join(data_path, 'id3_prices.csv'))

