#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Enhanced Visualization of Uncertainty Quantification for Pre-trained AlphaPeptDeep Models

This script provides improved visualizations for uncertainty estimates from pre-trained models.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import norm

def plot_rt_predictions_with_uncertainty(results_file, output_dir, method_name):
    """Plot RT predictions with enhanced uncertainty visualization
    
    Parameters
    ----------
    results_file : str
        Path to the results CSV file
    output_dir : str
        Directory to save the output plots
    method_name : str
        Name of the method (e.g., 'MC Dropout', 'Ensemble')
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Load results
    results = pd.read_csv(results_file)
    
    # Plot predictions with uncertainty
    plt.figure(figsize=(12, 8))
    
    # Plot error bars
    plt.errorbar(results['rt_norm'], results['rt_pred_mean'], 
                 yerr=1.96*results['rt_pred_std'], fmt='o', alpha=0.7,
                 ecolor='red', capsize=5, capthick=2, elinewidth=2)
    
    # Plot identity line
    plt.plot([min(results['rt_norm']), max(results['rt_norm'])], 
             [min(results['rt_norm']), max(results['rt_norm'])], 'k--')
    
    plt.xlabel('True RT', fontsize=14)
    plt.ylabel('Predicted RT', fontsize=14)
    plt.title(f'RT Prediction with Uncertainty ({method_name})', fontsize=16)
    plt.grid(True)
    
    # Add a text box with statistics
    mean_std = results['rt_pred_std'].mean()
    max_std = results['rt_pred_std'].max()
    min_std = results['rt_pred_std'].min()
    
    stats_text = f"Mean std: {mean_std:.4f}\nMax std: {max_std:.4f}\nMin std: {min_std:.4f}"
    plt.figtext(0.15, 0.15, stats_text, fontsize=12, 
                bbox=dict(facecolor='white', alpha=0.8))
    
    # Save figure
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'{method_name.lower().replace(" ", "_")}_rt_predictions_enhanced.png'), dpi=300)
    
    # Create a zoomed-in plot to better see the error bars
    plt.figure(figsize=(12, 8))
    
    # Sort by standard deviation to highlight points with largest uncertainty
    results_sorted = results.sort_values('rt_pred_std', ascending=False)
    top_n = min(10, len(results_sorted))  # Top 10 or all if less than 10
    
    # Plot top N points with highest uncertainty
    plt.errorbar(results_sorted['rt_norm'].iloc[:top_n], 
                 results_sorted['rt_pred_mean'].iloc[:top_n], 
                 yerr=1.96*results_sorted['rt_pred_std'].iloc[:top_n], 
                 fmt='o', alpha=0.7, ecolor='red', capsize=5, capthick=2, elinewidth=2,
                 label=f'Top {top_n} points with highest uncertainty')
    
    # Plot identity line
    min_val = min(results_sorted['rt_norm'].iloc[:top_n].min(), results_sorted['rt_pred_mean'].iloc[:top_n].min() - 2*results_sorted['rt_pred_std'].iloc[:top_n].max())
    max_val = max(results_sorted['rt_norm'].iloc[:top_n].max(), results_sorted['rt_pred_mean'].iloc[:top_n].max() + 2*results_sorted['rt_pred_std'].iloc[:top_n].max())
    plt.plot([min_val, max_val], [min_val, max_val], 'k--')
    
    plt.xlabel('True RT', fontsize=14)
    plt.ylabel('Predicted RT', fontsize=14)
    plt.title(f'RT Prediction with Uncertainty - Top {top_n} Uncertain Points ({method_name})', fontsize=16)
    plt.grid(True)
    plt.legend()
    
    # Add a text box with statistics for these points
    mean_std = results_sorted['rt_pred_std'].iloc[:top_n].mean()
    max_std = results_sorted['rt_pred_std'].iloc[:top_n].max()
    min_std = results_sorted['rt_pred_std'].iloc[:top_n].min()
    
    stats_text = f"Mean std: {mean_std:.4f}\nMax std: {max_std:.4f}\nMin std: {min_std:.4f}"
    plt.figtext(0.15, 0.15, stats_text, fontsize=12, 
                bbox=dict(facecolor='white', alpha=0.8))
    
    # Save figure
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'{method_name.lower().replace(" ", "_")}_rt_predictions_zoomed.png'), dpi=300)
    
    # Create a plot showing the relationship between RT and uncertainty
    plt.figure(figsize=(12, 8))
    plt.scatter(results['rt_norm'], results['rt_pred_std'], alpha=0.7, s=50)
    
    # Add trend line
    z = np.polyfit(results['rt_norm'], results['rt_pred_std'], 1)
    p = np.poly1d(z)
    plt.plot(results['rt_norm'], p(results['rt_norm']), "r--", linewidth=2)
    
    plt.xlabel('True RT', fontsize=14)
    plt.ylabel('Standard Deviation', fontsize=14)
    plt.title(f'Relationship between RT and Uncertainty ({method_name})', fontsize=16)
    plt.grid(True)
    
    # Add correlation coefficient
    corr = np.corrcoef(results['rt_norm'], results['rt_pred_std'])[0, 1]
    plt.figtext(0.15, 0.85, f"Correlation: {corr:.4f}", fontsize=12, 
                bbox=dict(facecolor='white', alpha=0.8))
    
    # Save figure
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'{method_name.lower().replace(" ", "_")}_rt_uncertainty_correlation.png'), dpi=300)


def plot_calibration_curve_enhanced(results_file, output_dir, method_name, n_bins=10):
    """Plot enhanced calibration curve for uncertainty estimates
    
    Parameters
    ----------
    results_file : str
        Path to the results CSV file
    output_dir : str
        Directory to save the output plots
    method_name : str
        Name of the method (e.g., 'MC Dropout', 'Ensemble')
    n_bins : int
        Number of bins for the calibration curve
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Load results
    results = pd.read_csv(results_file)
    
    # Extract data
    y_true = results['rt_norm'].values
    y_pred = results['rt_pred_mean'].values
    y_std = results['rt_pred_std'].values
    
    # Filter out zero standard deviations to avoid division by zero
    mask = y_std > 0
    y_true_filtered = y_true[mask]
    y_pred_filtered = y_pred[mask]
    y_std_filtered = y_std[mask]
    
    if len(y_std_filtered) == 0:
        print("Warning: All standard deviations are zero. Cannot create calibration curve.")
        return None
    
    # Calculate normalized errors
    normalized_errors = np.abs(y_true_filtered - y_pred_filtered) / y_std_filtered
    
    # Create bins and calculate frequencies
    bins = np.linspace(0, 3, n_bins+1)  # Up to 3 standard deviations
    bin_centers = 0.5 * (bins[:-1] + bins[1:])
    observed_freq = np.zeros(n_bins)
    
    for i in range(n_bins):
        observed_freq[i] = np.mean(normalized_errors <= bins[i+1])
    
    # Expected frequencies for a Gaussian distribution
    expected_freq = np.array([norm.cdf(b) for b in bins[1:]])
    
    # Plot
    plt.figure(figsize=(12, 8))
    plt.plot(expected_freq, observed_freq, 'o-', linewidth=2, markersize=8, label='Calibration curve')
    plt.plot([0, 1], [0, 1], 'k--', linewidth=2, label='Ideal calibration')
    plt.xlabel('Expected cumulative probability', fontsize=14)
    plt.ylabel('Observed cumulative probability', fontsize=14)
    plt.title(f'Calibration Curve for Uncertainty Estimates ({method_name})', fontsize=16)
    plt.legend(fontsize=12)
    plt.grid(True)
    
    # Calculate calibration error
    calibration_error = np.mean(np.abs(observed_freq - expected_freq))
    plt.figtext(0.15, 0.15, f"Calibration Error: {calibration_error:.4f}", fontsize=12, 
                bbox=dict(facecolor='white', alpha=0.8))
    
    # Save figure
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'{method_name.lower().replace(" ", "_")}_calibration_enhanced.png'), dpi=300)


def compare_methods_enhanced(mc_results_file, ensemble_results_file, output_dir):
    """Create enhanced comparison plots between different uncertainty quantification methods
    
    Parameters
    ----------
    mc_results_file : str
        Path to the MC Dropout results CSV file
    ensemble_results_file : str
        Path to the Ensemble results CSV file
    output_dir : str
        Directory to save the output plots
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Load results
    mc_results = pd.read_csv(mc_results_file)
    ensemble_results = pd.read_csv(ensemble_results_file)
    
    # Compare standard deviations
    plt.figure(figsize=(12, 8))
    plt.scatter(mc_results['rt_norm'], mc_results['rt_pred_std'], alpha=0.7, s=50, label='Simulated MC Dropout')
    plt.scatter(ensemble_results['rt_norm'], ensemble_results['rt_pred_std'], alpha=0.7, s=50, label='Ensemble')
    
    plt.xlabel('True RT', fontsize=14)
    plt.ylabel('Prediction Standard Deviation', fontsize=14)
    plt.title('Uncertainty Comparison: Simulated MC Dropout vs Ensemble', fontsize=16)
    plt.legend(fontsize=12)
    plt.grid(True)
    
    # Add statistics
    mc_mean_std = mc_results['rt_pred_std'].mean()
    ensemble_mean_std = ensemble_results['rt_pred_std'].mean()
    
    stats_text = f"MC Dropout mean std: {mc_mean_std:.4f}\nEnsemble mean std: {ensemble_mean_std:.4f}\nRatio (Ensemble/MC): {ensemble_mean_std/mc_mean_std:.2f}x"
    plt.figtext(0.15, 0.15, stats_text, fontsize=12, 
                bbox=dict(facecolor='white', alpha=0.8))
    
    # Save figure
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'uncertainty_comparison_enhanced.png'), dpi=300)
    
    # Compare prediction intervals
    plt.figure(figsize=(12, 8))
    
    # Select a subset of points for clarity
    n_points = min(10, len(mc_results))
    indices = np.linspace(0, len(mc_results)-1, n_points, dtype=int)
    
    x = np.arange(n_points)
    width = 0.35
    
    mc_intervals = mc_results['rt_pred_upper'].iloc[indices] - mc_results['rt_pred_lower'].iloc[indices]
    ensemble_intervals = ensemble_results['rt_pred_upper'].iloc[indices] - ensemble_results['rt_pred_lower'].iloc[indices]
    
    plt.bar(x - width/2, mc_intervals, width, label='Simulated MC Dropout')
    plt.bar(x + width/2, ensemble_intervals, width, label='Ensemble')
    
    plt.xlabel('Sample Index', fontsize=14)
    plt.ylabel('Prediction Interval Width', fontsize=14)
    plt.title('Comparison of Prediction Interval Widths', fontsize=16)
    plt.xticks(x, [f"{i}" for i in indices])
    plt.legend(fontsize=12)
    plt.grid(True, axis='y')
    
    # Save figure
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'interval_width_comparison.png'), dpi=300)


if __name__ == "__main__":
    # Define input and output directories
    input_dir = 'simple_uncertainty_results/rt'
    output_dir = 'enhanced_visualizations'
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Plot enhanced visualizations for MC Dropout results
    mc_results_file = os.path.join(input_dir, 'mc_dropout_results.csv')
    plot_rt_predictions_with_uncertainty(mc_results_file, output_dir, 'MC Dropout')
    plot_calibration_curve_enhanced(mc_results_file, output_dir, 'MC Dropout')
    
    # Plot enhanced visualizations for Ensemble results
    ensemble_results_file = os.path.join(input_dir, 'ensemble_results.csv')
    plot_rt_predictions_with_uncertainty(ensemble_results_file, output_dir, 'Ensemble')
    plot_calibration_curve_enhanced(ensemble_results_file, output_dir, 'Ensemble')
    
    # Compare methods
    compare_methods_enhanced(mc_results_file, ensemble_results_file, output_dir)
    
    print(f"Enhanced visualizations saved to {output_dir}")