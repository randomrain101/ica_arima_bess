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
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, r2_score, cohen_kappa_score, root_mean_squared_error, median_absolute_error, explained_variance_score
from pmdarima.arima import auto_arima
from joblib import Parallel, delayed
import warnings
warnings.filterwarnings('ignore', category=FutureWarning)

#%%
df = pd.read_csv(os.path.join(os.path.dirname(__file__), '..', 'data', 'id3_prices.csv'), index_col=0, parse_dates=True)

# Calculate price differences
df_price_diff = df.diff().dropna()

#%%
# 2 year rolling window
window = 356
df_pred = pd.DataFrame(columns=df.columns)
pca_list = []

def process_window(i, df_returns, window):
    df_window = df_returns.iloc[i:i + window]
    sc = StandardScaler().fit(df_window)
    df_scaled = sc.transform(df_window)
    
    # Use PCA with explained variance ratio of 95%
    pca = PCA().fit(df_scaled)
    df_pca = pd.DataFrame(pca.transform(df_scaled), index=df_window.index, 
                         columns=[f"pca_{j}" for j in range(pca.n_components_)])

    df_pred_pca = pd.DataFrame()

    # predict in PCA latent space
    for j in range(df_pca.shape[1]):
        # Fit ARIMA model
        model = auto_arima(df_pca.iloc[:, j], suppress_warnings=True, error_action='ignore').fit(df_pca.iloc[:, j])
        df_pred_pca[j] = model.predict(n_periods=1)

    # Transform back to original space (price differences)
    pred_price_diff = sc.inverse_transform(pca.inverse_transform(df_pred_pca.values))
    
    # Convert price differences back to prices
    last_prices = df_window.iloc[-1]  # Last prices from the window
    pred_prices = last_prices + pred_price_diff[0]
    
    print(f"finished window {i} of {len(df_returns) - window}")
    return pd.DataFrame([pred_prices], columns=df.columns, 
                       index=[df_returns.index[i + window]])

results = Parallel(n_jobs=-1)(delayed(process_window)(i, df_price_diff, window) 
                             for i in tqdm(range(len(df_price_diff) - window)))

#%%
df_results = pd.concat(results, axis=0)
df_results.to_csv("../data/pca_predictions.csv")

#%%
