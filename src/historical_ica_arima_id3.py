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
from statsforecast.models import AutoARIMA
from joblib import Parallel, delayed
import warnings
warnings.filterwarnings('ignore', category=FutureWarning)

#%%
df_prices = pd.read_csv(os.path.join(os.path.dirname(__file__), '..', 'data', 'id3_prices.csv'), index_col=0, parse_dates=True)

#%%
# 1 year rolling window
window = 365

def process_window(window_start, df_prices, window):
    price_window = df_prices.iloc[window_start:window_start + window]
    scaler = StandardScaler().fit(price_window)
    prices_scaled = scaler.transform(price_window)
    ica = FastICA(fun="exp", max_iter=200, algorithm="deflation").fit(prices_scaled)
    ica_components = pd.DataFrame(ica.transform(prices_scaled), index=price_window.index, columns=[f"ica_{component_idx}" for component_idx in range(prices_scaled.shape[1])])

    next_day_components = pd.DataFrame()

    # predict in ica latent space
    for component_idx in range(ica_components.shape[1]):
        # Fit ARIMA model
        next_day_components[component_idx] = AutoARIMA() \
            .fit(ica_components.iloc[:, component_idx].values) \
                .predict(h=1)['mean']

    next_day_prices = scaler.inverse_transform(ica.inverse_transform(next_day_components))
    return pd.DataFrame(next_day_prices, index=[price_window.index[-1] + pd.Timedelta(days=1)])

window_results = Parallel(n_jobs=-1)(delayed(process_window)(window_start, df_prices, window) for window_start in tqdm(range(len(df_prices) - window)))

#%%
df_ica_predictions = pd.concat(window_results, axis=0)
df_ica_predictions.to_csv("../data/ica_predictions.csv")

#%%
