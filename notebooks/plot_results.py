#%%
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from sklearn.metrics import mean_absolute_error, mean_squared_error
from scipy.stats import spearmanr
import warnings
from mpl_toolkits.mplot3d import Axes3D
from plotly.subplots import make_subplots
warnings.filterwarnings('ignore', category=FutureWarning)

#%%
df = pd.read_csv(os.path.join(os.path.dirname(__file__), '..', 'data', 'id3_prices.csv'), index_col=0, parse_dates=True)
df_results = pd.read_csv(os.path.join(os.path.dirname(__file__), '..', 'data', 'predictions.csv'), index_col=0, parse_dates=True)
df_pca_results = pd.read_csv(os.path.join(os.path.dirname(__file__), '..', 'data', 'pca_predictions.csv'), index_col=0, parse_dates=True)

#%%
# Align data - get common indices
common_index = df.index.intersection(df_results.index).intersection(df_pca_results.index)
df_actual = df.loc[common_index]
df_ica_pred = df_results.loc[common_index]
df_pca_pred = df_pca_results.loc[common_index]

# Naive prediction (previous day's prices)
df_naive_pred = df_actual.shift(1).dropna()
common_index_naive = df_actual.index.intersection(df_naive_pred.index)
df_actual_naive = df_actual.loc[common_index_naive]
df_naive_pred = df_naive_pred.loc[common_index_naive]

#%%
# Create interactive 3D surface plot
import plotly.graph_objects as go
import plotly.express as px

# Prepare data for surface plot
#days = np.arange(len(df_results))
days = df_results.index
hours = np.arange(24)
Z = df_results.T.values

# Create interactive 3D surface plot
fig = go.Figure(data=[go.Surface(
    z=Z,
    x=days,
    y=hours,
    colorscale='viridis',
    name='ICA Predictions'
)])

# Update layout
fig.update_layout(
    title='Interactive 3D Surface Plot of Price Predictions',
    scene=dict(
        xaxis_title='Day',
        yaxis_title='Hour (0-23)',
        zaxis_title='Predicted Price',
        camera=dict(
            eye=dict(x=1.2, y=1.2, z=1.2)
        )
    ),
    width=900,
    height=700
)

fig.show()

# Create additional interactive comparison plot
fig_comp = make_subplots(
    rows=1, cols=2,
    subplot_titles=('ICA Predictions', 'PCA Predictions'),
    specs=[[{'type': 'surface'}, {'type': 'surface'}]]
)

# Add ICA surface
fig_comp.add_trace(
    go.Surface(z=df_results.T.values, x=days, y=hours, colorscale='viridis', name='ICA'),
    row=1, col=1
)

# Add PCA surface
fig_comp.add_trace(
    go.Surface(z=df_pca_results.T.values, x=days, y=hours, colorscale='plasma', name='PCA'),
    row=1, col=2
)

fig_comp.update_layout(
    title='Interactive Comparison: ICA vs PCA Predictions',
    scene=dict(xaxis_title='Day', yaxis_title='Hour', zaxis_title='Price'),
    scene2=dict(xaxis_title='Day', yaxis_title='Hour', zaxis_title='Price'),
    height=700
)

fig_comp.show()

#%%