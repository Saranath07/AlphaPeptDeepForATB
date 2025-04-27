#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Uncertainty Quantification Analysis for Modification-Specific Peptides

This script analyzes a subset of modification-specific peptides using:
1. Simulated Monte Carlo Dropout (by adding noise to predictions)
2. Deep Ensembles (using different pre-trained models)

It evaluates the uncertainty estimates using Prediction Interval Coverage Probability (PICP),
Mean Prediction Interval Width (MPIW), and calibration curves.
"""

import os
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
from scipy.stats import norm
from tqdm import tqdm

from peptdeep.pretrained_models import ModelManager
from peptdeep.model.rt import AlphaRTModel
from peptdeep.utils import evaluate_linear_regression, evaluate_linear_regression_plot
from alphabase.peptide.fragment import init_fragment_by_precursor_dataframe

# Import the uncertainty quantification classes from the existing script
from simple_pretrained_uncertainty_quantification import (
    SimulatedMCDropoutPredictor,
    PretrainedEnsemblePredictor,
    calculate_picp,
    calculate_mpiw,
    plot_calibration_curve
)


def load_modification_specific_peptides(file_path, sample_size=50):
    """
    Load peptide data from the modification-specific peptides file
    
    Parameters
    ----------
    file_path : str
        Path to the modification-specific peptides file
    sample_size : int
        Number of peptides to sample
        
    Returns
    -------
    pd.DataFrame
        DataFrame with peptide data
    """
    # Load the data
    df = pd.read_csv(file_path, sep='\t')
    
    # Print basic information about the dataset
    print(f"Loaded {len(df)} peptides from {file_path}")
    print(f"Columns: {df.columns.tolist()}")
    
    # Filter to only include unmodified peptides for simplicity
    df_unmodified = df[df['Modifications'] == 'Unmodified']
    print(f"Filtered to {len(df_unmodified)} unmodified peptides")
    
    # Create a subset of the data
    if len(df_unmodified) > sample_size:
        # Sample randomly but with a fixed seed for reproducibility
        df_subset = df_unmodified.sample(sample_size, random_state=42)
    else:
        df_subset = df_unmodified
    
    # Prepare the data for AlphaPeptDeep
    # Create the necessary columns expected by the models
    peptide_df = pd.DataFrame({
        'sequence': df_subset['Sequence'],
        'mods': [''] * len(df_subset),  # Empty string for unmodified peptides
        'mod_sites': [''] * len(df_subset),  # Default empty string if not available
        'charge': df_subset['Charges'].apply(lambda x: int(str(x).split(';')[0]) if pd.notna(x) else 2),  # Take first charge state
        'rt': df_subset['Calibrated retention time'],  # Use calibrated RT if available
        'nce': [30] * len(df_subset),  # Default NCE value
        'instrument': ['QE'] * len(df_subset)  # Default instrument
    })
    
    # Add nAA column (peptide length)
    peptide_df['nAA'] = peptide_df.sequence.str.len()
    
    return peptide_df


def analyze_rt_uncertainty(peptide_df, output_dir='modification_specific_results'):
    """
    Analyze RT prediction uncertainty for the peptide dataset
    
    Parameters
    ----------
    peptide_df : pd.DataFrame
        DataFrame with peptide data
    output_dir : str
        Directory to save results
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    print("Loading models...")
    # Load the models
    model_mgr = ModelManager(mask_modloss=False, device='cuda')
    model_mgr.load_installed_models()
    
    # Make a copy of the input data to avoid modifying it
    df = peptide_df.copy()
    
    # Normalize RT values to be in a similar range as the model's training data
    # This is important for proper uncertainty quantification
    rt_min = df['rt'].min()
    rt_max = df['rt'].max()
    df['rt_norm'] = (df['rt'] - rt_min) / (rt_max - rt_min) * 100
    
    print("Running Simulated Monte Carlo Dropout experiment...")
    # Create MC Dropout predictor for RT model
    mc_dropout_rt = SimulatedMCDropoutPredictor(model_mgr.rt_model, n_samples=30, noise_scale=0.05)
    
    # Predict with uncertainty
    mc_results = mc_dropout_rt.predict(df)
    
    # Evaluate
    picp = calculate_picp(mc_results['rt_norm'], mc_results['rt_pred_lower'], mc_results['rt_pred_upper'])
    mpiw = calculate_mpiw(mc_results['rt_pred_lower'], mc_results['rt_pred_upper'])
    
    print(f"Simulated Monte Carlo Dropout - PICP: {picp:.4f}, MPIW: {mpiw:.4f}")
    
    # Calculate error metrics
    mc_results['abs_error'] = np.abs(mc_results['rt_norm'] - mc_results['rt_pred_mean'])
    mc_results['rel_error'] = mc_results['abs_error'] / mc_results['rt_norm'] * 100
    
    print(f"MC Dropout - Mean Absolute Error: {mc_results['abs_error'].mean():.4f}")
    print(f"MC Dropout - Mean Relative Error: {mc_results['rel_error'].mean():.4f}%")
    
    # Plot calibration curve
    fig = plot_calibration_curve(
        mc_results['rt_norm'], 
        mc_results['rt_pred_mean'], 
        mc_results['rt_pred_std'],
        save_path=os.path.join(output_dir, 'mc_dropout_calibration.png')
    )
    
    # Plot predictions with uncertainty
    plt.figure(figsize=(10, 6))
    plt.errorbar(mc_results['rt_norm'], mc_results['rt_pred_mean'], 
                 yerr=1.96*mc_results['rt_pred_std'], fmt='o', alpha=0.5)
    plt.plot([min(mc_results['rt_norm']), max(mc_results['rt_norm'])], 
             [min(mc_results['rt_norm']), max(mc_results['rt_norm'])], 'k--')
    plt.xlabel('True RT')
    plt.ylabel('Predicted RT')
    plt.title('RT Prediction with Uncertainty (Simulated Monte Carlo Dropout)')
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, 'mc_dropout_rt_predictions.png'))
    
    # Plot error vs uncertainty
    plt.figure(figsize=(10, 6))
    plt.scatter(mc_results['rt_pred_std'], mc_results['abs_error'], alpha=0.7)
    plt.xlabel('Prediction Standard Deviation (Uncertainty)')
    plt.ylabel('Absolute Error')
    plt.title('Error vs. Uncertainty (MC Dropout)')
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, 'mc_dropout_error_vs_uncertainty.png'))
    
    print("Running Ensemble experiment...")
    # Create Ensemble predictor for RT model
    ensemble_rt = PretrainedEnsemblePredictor(model_types=['generic', 'phospho', 'digly'])
    
    # Predict with uncertainty
    ensemble_results = ensemble_rt.predict_rt(df)
    
    # Evaluate
    picp = calculate_picp(ensemble_results['rt_norm'], ensemble_results['rt_pred_lower'], ensemble_results['rt_pred_upper'])
    mpiw = calculate_mpiw(ensemble_results['rt_pred_lower'], ensemble_results['rt_pred_upper'])
    
    print(f"Ensemble - PICP: {picp:.4f}, MPIW: {mpiw:.4f}")
    
    # Calculate error metrics
    ensemble_results['abs_error'] = np.abs(ensemble_results['rt_norm'] - ensemble_results['rt_pred_mean'])
    ensemble_results['rel_error'] = ensemble_results['abs_error'] / ensemble_results['rt_norm'] * 100
    
    print(f"Ensemble - Mean Absolute Error: {ensemble_results['abs_error'].mean():.4f}")
    print(f"Ensemble - Mean Relative Error: {ensemble_results['rel_error'].mean():.4f}%")
    
    # Plot calibration curve
    fig = plot_calibration_curve(
        ensemble_results['rt_norm'], 
        ensemble_results['rt_pred_mean'], 
        ensemble_results['rt_pred_std'],
        save_path=os.path.join(output_dir, 'ensemble_calibration.png')
    )
    
    # Plot predictions with uncertainty
    plt.figure(figsize=(10, 6))
    plt.errorbar(ensemble_results['rt_norm'], ensemble_results['rt_pred_mean'], 
                 yerr=1.96*ensemble_results['rt_pred_std'], fmt='o', alpha=0.5)
    plt.plot([min(ensemble_results['rt_norm']), max(ensemble_results['rt_norm'])], 
             [min(ensemble_results['rt_norm']), max(ensemble_results['rt_norm'])], 'k--')
    plt.xlabel('True RT')
    plt.ylabel('Predicted RT')
    plt.title('RT Prediction with Uncertainty (Ensemble)')
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, 'ensemble_rt_predictions.png'))
    
    # Plot error vs uncertainty
    plt.figure(figsize=(10, 6))
    plt.scatter(ensemble_results['rt_pred_std'], ensemble_results['abs_error'], alpha=0.7)
    plt.xlabel('Prediction Standard Deviation (Uncertainty)')
    plt.ylabel('Absolute Error')
    plt.title('Error vs. Uncertainty (Ensemble)')
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, 'ensemble_error_vs_uncertainty.png'))
    
    # Compare methods
    plt.figure(figsize=(12, 6))
    plt.scatter(mc_results['rt_norm'], mc_results['rt_pred_std'], alpha=0.5, label='Simulated MC Dropout')
    plt.scatter(ensemble_results['rt_norm'], ensemble_results['rt_pred_std'], alpha=0.5, label='Ensemble')
    plt.xlabel('True RT')
    plt.ylabel('Prediction Standard Deviation')
    plt.title('Uncertainty Comparison: Simulated MC Dropout vs Ensemble')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, 'uncertainty_comparison.png'))
    
    # Compare error distributions
    plt.figure(figsize=(12, 6))
    plt.hist(mc_results['abs_error'], alpha=0.5, bins=20, label='MC Dropout')
    plt.hist(ensemble_results['abs_error'], alpha=0.5, bins=20, label='Ensemble')
    plt.xlabel('Absolute Error')
    plt.ylabel('Frequency')
    plt.title('Error Distribution Comparison')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, 'error_distribution_comparison.png'))
    
    # Save results to CSV
    mc_results.to_csv(os.path.join(output_dir, 'mc_dropout_results.csv'), index=False)
    ensemble_results.to_csv(os.path.join(output_dir, 'ensemble_results.csv'), index=False)
    
    # Create a summary DataFrame for comparison
    summary = pd.DataFrame({
        'Method': ['MC Dropout', 'Ensemble'],
        'PICP': [calculate_picp(mc_results['rt_norm'], mc_results['rt_pred_lower'], mc_results['rt_pred_upper']),
                 calculate_picp(ensemble_results['rt_norm'], ensemble_results['rt_pred_lower'], ensemble_results['rt_pred_upper'])],
        'MPIW': [calculate_mpiw(mc_results['rt_pred_lower'], mc_results['rt_pred_upper']),
                 calculate_mpiw(ensemble_results['rt_pred_lower'], ensemble_results['rt_pred_upper'])],
        'Mean Absolute Error': [mc_results['abs_error'].mean(), ensemble_results['abs_error'].mean()],
        'Mean Relative Error (%)': [mc_results['rel_error'].mean(), ensemble_results['rel_error'].mean()],
        'Median Absolute Error': [mc_results['abs_error'].median(), ensemble_results['abs_error'].median()],
        'Max Absolute Error': [mc_results['abs_error'].max(), ensemble_results['abs_error'].max()],
        'Mean Uncertainty (Std)': [mc_results['rt_pred_std'].mean(), ensemble_results['rt_pred_std'].mean()]
    })
    
    summary.to_csv(os.path.join(output_dir, 'uncertainty_summary.csv'), index=False)
    
    print(f"Results saved to {output_dir}")
    
    return mc_results, ensemble_results


def analyze_peptide_properties(mc_results, ensemble_results, output_dir='modification_specific_results'):
    """
    Analyze the impact of peptide properties on prediction uncertainty
    
    Parameters
    ----------
    mc_results : pd.DataFrame
        Results from MC Dropout
    ensemble_results : pd.DataFrame
        Results from Ensemble
    output_dir : str
        Directory to save results
    """
    # Since we're only using unmodified peptides, let's analyze by peptide length instead
    
    # Add peptide length column if not already present
    if 'nAA' not in mc_results.columns:
        mc_results['nAA'] = mc_results['sequence'].str.len()
    if 'nAA' not in ensemble_results.columns:
        ensemble_results['nAA'] = ensemble_results['sequence'].str.len()
    
    # Create length categories for easier analysis
    mc_results['length_category'] = pd.cut(mc_results['nAA'],
                                          bins=[0, 8, 12, 16, 100],
                                          labels=['Short (≤8)', 'Medium (9-12)', 'Long (13-16)', 'Very Long (>16)'])
    ensemble_results['length_category'] = pd.cut(ensemble_results['nAA'],
                                               bins=[0, 8, 12, 16, 100],
                                               labels=['Short (≤8)', 'Medium (9-12)', 'Long (13-16)', 'Very Long (>16)'])
    
    # Analyze MC Dropout results by peptide length
    mc_length_summary = mc_results.groupby('length_category').agg({
        'abs_error': ['mean', 'median', 'std', 'max'],
        'rt_pred_std': ['mean', 'median', 'max']
    })
    
    # Analyze Ensemble results by peptide length
    ensemble_length_summary = ensemble_results.groupby('length_category').agg({
        'abs_error': ['mean', 'median', 'std', 'max'],
        'rt_pred_std': ['mean', 'median', 'max']
    })
    
    # Save summaries
    mc_length_summary.to_csv(os.path.join(output_dir, 'mc_dropout_length_impact.csv'))
    ensemble_length_summary.to_csv(os.path.join(output_dir, 'ensemble_length_impact.csv'))
    
    # Plot comparison of error by peptide length
    plt.figure(figsize=(14, 8))
    
    # Extract data for different length categories
    categories = mc_length_summary.index.tolist()
    x = np.arange(len(categories))
    width = 0.35
    
    # Get mean absolute errors for each category
    mc_errors = [mc_length_summary.loc[cat, ('abs_error', 'mean')] for cat in categories]
    ensemble_errors = [ensemble_length_summary.loc[cat, ('abs_error', 'mean')] for cat in categories]
    
    # Create the bar plot
    plt.bar(x - width/2, mc_errors, width, label='MC Dropout')
    plt.bar(x + width/2, ensemble_errors, width, label='Ensemble')
    
    plt.xlabel('Peptide Length')
    plt.ylabel('Mean Absolute Error')
    plt.title('Impact of Peptide Length on Prediction Error')
    plt.xticks(x, categories)
    plt.legend()
    plt.grid(True, axis='y')
    plt.savefig(os.path.join(output_dir, 'length_impact_on_error.png'))
    
    # Plot comparison of uncertainty by peptide length
    plt.figure(figsize=(14, 8))
    
    # Get mean uncertainty for each category
    mc_uncertainty = [mc_length_summary.loc[cat, ('rt_pred_std', 'mean')] for cat in categories]
    ensemble_uncertainty = [ensemble_length_summary.loc[cat, ('rt_pred_std', 'mean')] for cat in categories]
    
    # Create the bar plot
    plt.bar(x - width/2, mc_uncertainty, width, label='MC Dropout')
    plt.bar(x + width/2, ensemble_uncertainty, width, label='Ensemble')
    
    plt.xlabel('Peptide Length')
    plt.ylabel('Mean Uncertainty (Std)')
    plt.title('Impact of Peptide Length on Prediction Uncertainty')
    plt.xticks(x, categories)
    plt.legend()
    plt.grid(True, axis='y')
    plt.savefig(os.path.join(output_dir, 'length_impact_on_uncertainty.png'))
    
    # Analyze correlation between peptide length and uncertainty/error
    plt.figure(figsize=(12, 6))
    plt.scatter(mc_results['nAA'], mc_results['rt_pred_std'], alpha=0.7, label='Uncertainty')
    plt.scatter(mc_results['nAA'], mc_results['abs_error'], alpha=0.7, label='Error')
    plt.xlabel('Peptide Length')
    plt.ylabel('Value')
    plt.title('Correlation between Peptide Length and Uncertainty/Error (MC Dropout)')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, 'length_correlation_mc_dropout.png'))


def run_modification_specific_analysis(file_path='modificationSpecificPeptides.txt', sample_size=50):
    """
    Run the complete analysis for modification-specific peptides
    
    Parameters
    ----------
    file_path : str
        Path to the modification-specific peptides file
    sample_size : int
        Number of peptides to sample
    """
    output_dir = 'modification_specific_results'
    os.makedirs(output_dir, exist_ok=True)
    
    # Load the peptide data
    peptide_df = load_modification_specific_peptides(file_path, sample_size)
    
    # Save the sampled dataset for reference
    peptide_df.to_csv(os.path.join(output_dir, 'sampled_peptides.csv'), index=False)
    
    # Analyze RT uncertainty
    mc_results, ensemble_results = analyze_rt_uncertainty(peptide_df, output_dir)
    
    # Analyze the impact of peptide properties
    analyze_peptide_properties(mc_results, ensemble_results, output_dir)
    
    print(f"Analysis complete. Results saved to {output_dir}")


if __name__ == "__main__":
    # Run the analysis
    run_modification_specific_analysis()