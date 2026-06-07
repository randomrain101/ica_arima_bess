#%%
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
from joblib import Parallel, delayed
from scipy.stats import spearmanr
import warnings
warnings.filterwarnings('ignore', category=FutureWarning)

#%%
df_prices = pd.read_csv(os.path.join(os.path.dirname(__file__), '..', 'data', 'id3_prices.csv'), index_col=0, parse_dates=True)
df_ica_predictions = pd.read_csv(os.path.join(os.path.dirname(__file__), '..', 'data', 'predictions.csv'), index_col=0, parse_dates=True)
df_arima_predictions = pd.read_csv(os.path.join(os.path.dirname(__file__), '..', 'data', 'arima_predictions.csv'), index_col=0, parse_dates=True)

#%%
df_ica_predicted_returns = (df_ica_predictions - df_prices.shift(1)).dropna()
df_arima_predicted_returns = (df_arima_predictions - df_prices.shift(1)).dropna()
df_actual_returns = df_prices.diff().reindex(df_ica_predicted_returns.index)
naive_predicted_returns = df_prices.shift(1).diff().reindex(df_ica_predicted_returns.index)

#%%
# Ensure all DataFrames have the same index
common_index = (df_ica_predicted_returns.index
                .intersection(df_actual_returns.index)
                .intersection(df_arima_predicted_returns.index)
                .intersection(naive_predicted_returns.index))
df_ica_predicted_returns = df_ica_predicted_returns.loc[common_index]
df_arima_predicted_returns = df_arima_predicted_returns.loc[common_index]
df_actual_returns = df_actual_returns.loc[common_index]
naive_predicted_returns = naive_predicted_returns.loc[common_index]

#%%
# Flatten the returns DataFrames for metric calculation
ica_returns_flat = df_ica_predicted_returns.values.flatten()
actual_returns_flat = df_actual_returns.values.flatten()

# Remove NaN values for metric calculation
valid_mask = ~(np.isnan(ica_returns_flat) | np.isnan(actual_returns_flat))
print("Dropping", np.sum(~valid_mask), "NaN values from the returns data.")
ica_pred_clean = ica_returns_flat[valid_mask]
actual_clean_ica = actual_returns_flat[valid_mask]

# Calculate R² score
r2_ica = r2_score(actual_clean_ica, ica_pred_clean)

# Mean Absolute Error
mae_ica = np.mean(np.abs(ica_pred_clean - actual_clean_ica))
print(f"Mean Absolute Error: {mae_ica:.4f}")

# For classification metrics, convert returns to directional predictions
ica_pred_dir = np.sign(ica_pred_clean)
actual_dir_ica = np.sign(actual_clean_ica)

# Calculate accuracy
accuracy_ica = accuracy_score(actual_dir_ica, ica_pred_dir)
print(f"Direction Accuracy: {accuracy_ica:.4f}")

# Calculate Cohen's Kappa
kappa_ica = cohen_kappa_score(actual_dir_ica, ica_pred_dir)
print(f"Cohen's Kappa: {kappa_ica:.4f}")

# Hit Rate (percentage of predictions within 1 standard deviation)
std_actual_ica = np.std(actual_clean_ica)
hit_rate_ica = np.mean(np.abs(ica_pred_clean - actual_clean_ica) <= std_actual_ica)
print(f"Hit Rate (within 1 std): {hit_rate_ica:.4f}")

# Calculate RMSE
rmse_ica = root_mean_squared_error(actual_clean_ica, ica_pred_clean)
print(f"RMSE: {rmse_ica:.4f}")

# Calculate Median Absolute Error
medae_ica = median_absolute_error(actual_clean_ica, ica_pred_clean)
print(f"Median Absolute Error: {medae_ica:.4f}")

# Calculate Spearman rank correlation coefficient
spearman_ica, _ = spearmanr(actual_clean_ica, ica_pred_clean)
print(f"Spearman Correlation: {spearman_ica:.4f}")

# Calculate metrics for naive predictions
naive_returns_flat = naive_predicted_returns.values.flatten()
naive_mask = ~(np.isnan(naive_returns_flat) | np.isnan(actual_returns_flat))
naive_clean = naive_returns_flat[naive_mask]
actual_clean_naive = actual_returns_flat[naive_mask]

# Naive metrics
r2_naive = r2_score(actual_clean_naive, naive_clean)
mae_naive = np.mean(np.abs(naive_clean - actual_clean_naive))
rmse_naive = root_mean_squared_error(actual_clean_naive, naive_clean)
medae_naive = median_absolute_error(actual_clean_naive, naive_clean)
spearman_naive, _ = spearmanr(actual_clean_naive, naive_clean)

naive_direction = np.sign(naive_clean)
actual_direction_naive = np.sign(actual_clean_naive)
accuracy_naive = accuracy_score(actual_direction_naive, naive_direction)
kappa_naive = cohen_kappa_score(actual_direction_naive, naive_direction)

std_actual_naive = np.std(actual_clean_naive)
hit_rate_naive = np.mean(np.abs(naive_clean - actual_clean_naive) <= std_actual_naive)

# Calculate metrics for ARIMA predictions
arima_returns_flat = df_arima_predicted_returns.values.flatten()
arima_mask = ~(np.isnan(arima_returns_flat) | np.isnan(actual_returns_flat))
arima_clean = arima_returns_flat[arima_mask]
actual_clean_arima = actual_returns_flat[arima_mask]

# ARIMA metrics
r2_arima = r2_score(actual_clean_arima, arima_clean)
mae_arima = np.mean(np.abs(arima_clean - actual_clean_arima))
rmse_arima = root_mean_squared_error(actual_clean_arima, arima_clean)
medae_arima = median_absolute_error(actual_clean_arima, arima_clean)
spearman_arima, _ = spearmanr(actual_clean_arima, arima_clean)

arima_direction = np.sign(arima_clean)
actual_direction_arima = np.sign(actual_clean_arima)
accuracy_arima = accuracy_score(actual_direction_arima, arima_direction)
kappa_arima = cohen_kappa_score(actual_direction_arima, arima_direction)

std_actual_arima = np.std(actual_clean_arima)
hit_rate_arima = np.mean(np.abs(arima_clean - actual_clean_arima) <= std_actual_arima)

# Create comparison table including ARIMA vs Naive benchmark
print("\n" + "="*90)
print("MODEL vs ARIMA vs NAIVE COMPARISON")
print("="*90)
comparison_metrics = pd.DataFrame({
    'ICA Model': [r2_ica, mae_ica, rmse_ica, medae_ica, accuracy_ica, kappa_ica, hit_rate_ica, spearman_ica],
    'ARIMA Model': [r2_arima, mae_arima, rmse_arima, medae_arima, accuracy_arima, kappa_arima, hit_rate_arima, spearman_arima],
    'Naive': [r2_naive, mae_naive, rmse_naive, medae_naive, accuracy_naive, kappa_naive, hit_rate_naive, spearman_naive],
}, index=['R²', 'MAE', 'RMSE', 'Median AE', 'Direction Accuracy', 'Cohen\'s Kappa', 'Hit Rate', 'Spearman Corr'])

# Define which metrics are "higher is better" vs "lower is better"
higher_is_better = ['R²', 'Direction Accuracy', 'Cohen\'s Kappa', 'Hit Rate', 'Spearman Corr']
lower_is_better = ['MAE', 'RMSE', 'Median AE']

# Calculate improvements
for model in ['ICA Model', 'ARIMA Model']:
    comparison_metrics[f'{model} vs Naive (%)'] = 0.0
    
    for metric in comparison_metrics.index:
        model_val = comparison_metrics.loc[metric, model]
        naive_val = comparison_metrics.loc[metric, 'Naive']
        
        if metric in higher_is_better:
            rel_improvement = ((model_val - naive_val) / abs(naive_val)) * 100 if naive_val != 0 else 0
        else:
            rel_improvement = ((naive_val - model_val) / abs(naive_val)) * 100 if naive_val != 0 else 0
        
        comparison_metrics.loc[metric, f'{model} vs Naive (%)'] = round(rel_improvement, 2)

print(comparison_metrics.round(4))

#%%
# Yearly analysis
def calculate_yearly_metrics(predicted_df, actual_df, label):
    """Calculate metrics for yearly intervals"""
    yearly_results = []
    
    # Align indices
    common_index = predicted_df.index.intersection(actual_df.index)
    predicted_aligned = predicted_df.loc[common_index]
    actual_aligned = actual_df.loc[common_index]
    
    # Group by years
    years = predicted_aligned.groupby(pd.Grouper(freq='YE'))
    
    for year_end, year_data in years:
        if len(year_data) == 0:
            continue
            
        year_actual = actual_aligned.loc[year_data.index]
        
        # Flatten and clean data
        pred_flat = year_data.values.flatten()
        actual_flat = year_actual.values.flatten()
        mask = ~(np.isnan(pred_flat) | np.isnan(actual_flat))
        
        if np.sum(mask) < 10:  # Skip if too few data points
            continue
            
        pred_clean = pred_flat[mask]
        actual_clean = actual_flat[mask]
        
        # Calculate metrics
        r2_y = r2_score(actual_clean, pred_clean)
        mae_y = np.mean(np.abs(pred_clean - actual_clean))
        rmse_y = root_mean_squared_error(actual_clean, pred_clean)
        spearman_y, _ = spearmanr(actual_clean, pred_clean)
        
        pred_dir = np.sign(pred_clean)
        actual_dir = np.sign(actual_clean)
        accuracy_y = accuracy_score(actual_dir, pred_dir)
        
        yearly_results.append({
            'Year': year_end,
            'Model': label,
            'R²': r2_y,
            'MAE': mae_y,
            'RMSE': rmse_y,
            'Direction_Accuracy': accuracy_y,
            'Spearman': spearman_y,
            'N_Observations': len(pred_clean)
        })
    
    return yearly_results

# Calculate yearly metrics for all models
yearly_metrics = []
yearly_metrics.extend(calculate_yearly_metrics(df_ica_predicted_returns, df_actual_returns, 'ICA'))
yearly_metrics.extend(calculate_yearly_metrics(df_arima_predicted_returns, df_actual_returns, 'ARIMA'))
yearly_metrics.extend(calculate_yearly_metrics(naive_predicted_returns, df_actual_returns, 'Naive'))

yearly_df = pd.DataFrame(yearly_metrics)

print("\nYearly Performance:")
print(yearly_df.pivot_table(index='Year', columns='Model', values=['R²', 'MAE', 'Direction_Accuracy', 'Spearman']).round(4))

#%%
# Plot yearly progression
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
axes = axes.flatten()

metrics_to_plot = ['R²', 'MAE', 'RMSE', 'Direction_Accuracy']
for i, metric in enumerate(metrics_to_plot):
    ax = axes[i]
    
    for model in ['ICA', 'ARIMA', 'Naive']:
        model_data = yearly_df[yearly_df['Model'] == model]
        ax.plot(model_data['Year'], model_data[metric], marker='o', label=model, linewidth=2)
    
    ax.set_title(f'{metric} by Year')
    ax.set_xlabel('Year')
    ax.set_ylabel(metric)
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.tick_params(axis='x', rotation=45)

plt.tight_layout()
plt.show()

#%% 
# Plot comparison of all models
fig, axes = plt.subplots(2, 2, figsize=(18, 12))

# Scatter plot comparison
axes[0,0].scatter(actual_clean, predicted_clean, alpha=0.3, label='ICA', s=5)
axes[0,0].scatter(actual_clean_arima, arima_clean, alpha=0.3, label='ARIMA', s=5)
axes[0,0].scatter(actual_clean_naive, naive_clean, alpha=0.3, label='Naive', s=5)
axes[0,0].plot([-10, 10], [-10, 10], 'r--', alpha=0.8)
axes[0,0].set_xlabel('Actual Returns')
axes[0,0].set_ylabel('Predicted Returns')
axes[0,0].set_title('All Models: Predicted vs Actual')
axes[0,0].legend()
axes[0,0].grid(True)

# Bar chart of key metrics
metrics_names = ['R²', 'MAE', 'Direction Accuracy', 'Spearman Corr']
ica_values = [r2, mae_returns, accuracy, spearman_corr]
arima_values = [r2_arima, mae_arima, accuracy_arima, spearman_arima]
naive_values = [r2_naive, mae_naive, accuracy_naive, spearman_naive]

x = np.arange(len(metrics_names))
width = 0.25

axes[0,1].bar(x - width, ica_values, width, label='ICA', alpha=0.8)
axes[0,1].bar(x, arima_values, width, label='ARIMA', alpha=0.8)
axes[0,1].bar(x + width, naive_values, width, label='Naive', alpha=0.8)
axes[0,1].set_xlabel('Metrics')
axes[0,1].set_ylabel('Value')
axes[0,1].set_title('All Models: Key Metrics Comparison')
axes[0,1].set_xticks(x)
axes[0,1].set_xticklabels(metrics_names)
axes[0,1].legend()
axes[0,1].grid(True, alpha=0.3)

# Residuals comparison
ica_residuals = predicted_clean - actual_clean
arima_residuals = arima_clean - actual_clean_arima
naive_residuals = naive_clean - actual_clean_naive

axes[1,0].hist(ica_residuals, bins=50, alpha=0.6, label='ICA Residuals', density=True)
axes[1,0].hist(arima_residuals, bins=50, alpha=0.6, label='ARIMA Residuals', density=True)
axes[1,0].hist(naive_residuals, bins=50, alpha=0.6, label='Naive Residuals', density=True)
axes[1,0].set_xlabel('Residuals')
axes[1,0].set_ylabel('Density')
axes[1,0].set_title('Residuals Distribution')
axes[1,0].legend()
axes[1,0].grid(True, alpha=0.3)

# Yearly improvement vs naive
yearly_pivot = yearly_df.pivot_table(index='Year', columns='Model', values='R²')
improvement_vs_naive = yearly_pivot.subtract(yearly_pivot['Naive'], axis=0)

axes[1,1].plot(improvement_vs_naive.index, improvement_vs_naive['ICA'], marker='o', label='ICA vs Naive', linewidth=2)
axes[1,1].plot(improvement_vs_naive.index, improvement_vs_naive['ARIMA'], marker='^', label='ARIMA vs Naive', linewidth=2)
axes[1,1].axhline(y=0, color='red', linestyle='--', alpha=0.7)
axes[1,1].set_xlabel('Year')
axes[1,1].set_ylabel('R² Improvement vs Naive')
axes[1,1].set_title('Model Performance Improvement Over Time (vs Naive Benchmark)')
axes[1,1].legend()
axes[1,1].grid(True, alpha=0.3)
axes[1,1].tick_params(axis='x', rotation=45)

plt.tight_layout()
plt.show()

#%%