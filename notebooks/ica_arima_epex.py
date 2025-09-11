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
df = pd.read_csv(os.path.join(os.path.dirname(__file__), '..', 'data', 'id3_prices.csv'), index_col=0, parse_dates=True)

#%%
# 2 year rolling window
window = 356
df_pred = pd.DataFrame(columns=df.columns)
ica_list = []

def process_window(i, df, window):
    df_window = df.iloc[i:i + window]
    sc = StandardScaler().fit(df_window)
    df_scaled = sc.transform(df_window)
    ica = FastICA(fun="exp", max_iter=200, algorithm="deflation").fit(df_scaled)
    df_ica = pd.DataFrame(ica.transform(df_scaled), index=df_window.index, columns=[f"ica_{j}" for j in range(df_scaled.shape[1])])

    df_pred_ica = pd.DataFrame()

    # predict in ica latent space
    for j in range(df_ica.shape[1]):
        # Fit ARIMA model
        df_pred_ica[j] = auto_arima(df_ica.iloc[:, j], suppress_warnings=True, error_action='ignore') \
            .fit(df_ica.iloc[:, j]) \
                .predict(n_periods=1)

    pred_values = sc.inverse_transform(ica.inverse_transform(df_pred_ica))
    print("finished window", i, "of", len(df) - window)
    return pd.DataFrame(pred_values, index=[df_window.index[-1] + pd.Timedelta(days=1)])

results = Parallel(n_jobs=-1)(delayed(process_window)(i, df, window) for i in tqdm(range(len(df) - window)))

#%%
df_results = pd.concat(results, axis=0)
#df_results.index = pd.date_range(start=df_results.index[0], end=df_results.index[-1], freq='D')
#df_results.index = pd.DatetimeIndex(df_results.index)
df_results.to_csv("../data/predictions.csv")

#%%
