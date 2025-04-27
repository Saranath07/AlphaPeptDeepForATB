#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Combined Uncertainty Quantification Analysis for Modification-Specific Peptides

This script performs a comprehensive uncertainty quantification analysis for both
retention time (RT) and MS2 fragment ion intensity predictions using:
1. Simulated Monte Carlo Dropout
2. Deep Ensembles (using different pre-trained models)

It evaluates the uncertainty estimates and visualizes the results.
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
    plot_calibration_curve,
    predict_ms2_with_uncertainty,
    plot_ms2_with_uncertainty
)


def load_modification_specific_peptides(file_path, sample_size=None):
    """
    Load peptide data from the modification-specific peptides file
    
    Parameters
    ----------
    file_path : str
        Path to the modification-specific peptides file
    sample_size : int, optional
        Number of peptides to sample. If None, use all peptides.
        
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
    
    # Create a subset of the data if requested
    if sample_size is not None and len(df_unmodified) > sample_size:
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


def analyze_rt_uncertainty(peptide_df, model_mgr, output_dir='combined_results/rt'):
    """
    Analyze RT prediction uncertainty for the peptide dataset
    
    Parameters
    ----------
    peptide_df : pd.DataFrame
        DataFrame with peptide data
    model_mgr : ModelManager
        Model manager with loaded models
    output_dir : str
        Directory to save results
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Make a copy of the input data to avoid modifying it
    df = peptide_df.copy()
    
    # Normalize RT values to be in a similar range as the model's training data
    rt_min = df['rt'].min()
    rt_max = df['rt'].max()
    df['rt_norm'] = (df['rt'] - rt_min) / (rt_max - rt_min) * 100
    
    print("Running Simulated Monte Carlo Dropout experiment for RT...")
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
    plt.close()
    
    # Plot error vs uncertainty
    plt.figure(figsize=(10, 6))
    plt.scatter(mc_results['rt_pred_std'], mc_results['abs_error'], alpha=0.7)
    plt.xlabel('Prediction Standard Deviation (Uncertainty)')
    plt.ylabel('Absolute Error')
    plt.title('Error vs. Uncertainty (MC Dropout)')
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, 'mc_dropout_error_vs_uncertainty.png'))
    plt.close()
    
    print("Running Ensemble experiment for RT...")
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
    plt.close()
    
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
    plt.close()
    
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
    
    summary.to_csv(os.path.join(output_dir, 'rt_uncertainty_summary.csv'), index=False)
    
    print(f"RT uncertainty analysis complete. Results saved to {output_dir}")
    
    return mc_results, ensemble_results


def analyze_ms2_uncertainty(peptide_df, model_mgr, output_dir='combined_results/ms2'):
    """
    Analyze MS2 prediction uncertainty for the peptide dataset
    
    Parameters
    ----------
    peptide_df : pd.DataFrame
        DataFrame with peptide data
    model_mgr : ModelManager
        Model manager with loaded models
    output_dir : str
        Directory to save results
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Make a copy of the input data to avoid modifying it
    df = peptide_df.copy()
    
    print("Running Simulated Monte Carlo Dropout for MS2 prediction...")
    # Predict MS2 with uncertainty using MC Dropout
    mc_ms2_results = predict_ms2_with_uncertainty(model_mgr, df, n_samples=30, noise_scale=0.05)
    
    # Plot MS2 spectra with uncertainty for a subset of peptides
    print("Generating MS2 spectra plots with uncertainty...")
    max_plots = min(len(df), 5)  # Limit to 5 peptides to avoid too many plots
    for i in range(max_plots):
        peptide_seq = df.iloc[i]['sequence']
        print(f"Plotting MS2 spectrum for peptide {i+1}/{max_plots}: {peptide_seq}")
        
        fig = plot_ms2_with_uncertainty(
            mc_ms2_results, 
            peptide_idx=i,
            save_path=os.path.join(output_dir, f'mc_dropout_ms2_uncertainty_{peptide_seq}.png')
        )
        plt.close(fig)
    
    print("Running Ensemble for MS2 prediction...")
    # Create Ensemble predictor for MS2 model
    ensemble = PretrainedEnsemblePredictor(model_types=['generic', 'phospho', 'digly'])
    
    # Predict with uncertainty
    ensemble_ms2_results = ensemble.predict_ms2(df)
    
    # Plot MS2 spectra with uncertainty for a subset of peptides
    print("Generating MS2 spectra plots with uncertainty (Ensemble)...")
    for i in range(max_plots):
        peptide_seq = df.iloc[i]['sequence']
        print(f"Plotting MS2 spectrum for peptide {i+1}/{max_plots}: {peptide_seq}")
        
        fig = plot_ms2_with_uncertainty(
            ensemble_ms2_results, 
            peptide_idx=i,
            save_path=os.path.join(output_dir, f'ensemble_ms2_uncertainty_{peptide_seq}.png')
        )
        plt.close(fig)
    
    # Calculate and compare uncertainty metrics for MS2 predictions
    print("Calculating MS2 uncertainty metrics...")
    
    # Extract fragment intensities for analysis
    mc_intensities = mc_ms2_results['fragment_intensity_mean_df'].values
    mc_uncertainties = mc_ms2_results['fragment_intensity_std_df'].values
    
    ensemble_intensities = ensemble_ms2_results['fragment_intensity_mean_df'].values
    ensemble_uncertainties = ensemble_ms2_results['fragment_intensity_std_df'].values
    
    # Calculate average uncertainty (standard deviation) for each method
    mc_avg_uncertainty = np.mean(mc_uncertainties)
    ensemble_avg_uncertainty = np.mean(ensemble_uncertainties)
    
    print(f"MC Dropout - Average MS2 intensity uncertainty: {mc_avg_uncertainty:.4f}")
    print(f"Ensemble - Average MS2 intensity uncertainty: {ensemble_avg_uncertainty:.10f}")
    
    # Compare the distribution of uncertainties
    plt.figure(figsize=(12, 6))
    plt.hist(mc_uncertainties.flatten(), bins=50, alpha=0.5, label='MC Dropout')
    plt.hist(ensemble_uncertainties.flatten(), bins=50, alpha=0.5, label='Ensemble')
    plt.xlabel('Fragment Intensity Uncertainty (Std)')
    plt.ylabel('Frequency')
    plt.title('Distribution of MS2 Intensity Uncertainties')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, 'ms2_uncertainty_distribution.png'))
    plt.close()
    
    # Analyze uncertainty vs intensity relationship
    plt.figure(figsize=(12, 6))
    plt.scatter(mc_intensities.flatten(), mc_uncertainties.flatten(), alpha=0.3, label='MC Dropout')
    plt.scatter(ensemble_intensities.flatten(), ensemble_uncertainties.flatten(), alpha=0.3, label='Ensemble')
    plt.xlabel('Predicted Fragment Intensity')
    plt.ylabel('Uncertainty (Std)')
    plt.title('MS2 Intensity vs Uncertainty')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, 'ms2_intensity_vs_uncertainty.png'))
    plt.close()
    
    # Analyze by ion type
    # Separate b and y ions
    b_ion_cols = [col for col in mc_ms2_results['fragment_intensity_mean_df'].columns if col.startswith('b_')]
    y_ion_cols = [col for col in mc_ms2_results['fragment_intensity_mean_df'].columns if col.startswith('y_')]
    
    # Calculate average uncertainty for each ion type
    mc_b_uncertainty = mc_ms2_results['fragment_intensity_std_df'][b_ion_cols].values.mean()
    mc_y_uncertainty = mc_ms2_results['fragment_intensity_std_df'][y_ion_cols].values.mean()
    
    ensemble_b_uncertainty = ensemble_ms2_results['fragment_intensity_std_df'][b_ion_cols].values.mean()
    ensemble_y_uncertainty = ensemble_ms2_results['fragment_intensity_std_df'][y_ion_cols].values.mean()
    
    print(f"MC Dropout - Average b-ion uncertainty: {mc_b_uncertainty:.4f}")
    print(f"MC Dropout - Average y-ion uncertainty: {mc_y_uncertainty:.4f}")
    print(f"Ensemble - Average b-ion uncertainty: {ensemble_b_uncertainty:.10f}")
    print(f"Ensemble - Average y-ion uncertainty: {ensemble_y_uncertainty:.10f}")
    
    # Plot comparison of ion type uncertainties
    plt.figure(figsize=(10, 6))
    
    x = np.arange(2)
    width = 0.35
    
    plt.bar(x - width/2, [mc_b_uncertainty, mc_y_uncertainty], width, label='MC Dropout')
    plt.bar(x + width/2, [ensemble_b_uncertainty, ensemble_y_uncertainty], width, label='Ensemble')
    
    plt.xlabel('Ion Type')
    plt.ylabel('Average Uncertainty (Std)')
    plt.title('MS2 Uncertainty by Ion Type')
    plt.xticks(x, ['b-ions', 'y-ions'])
    plt.legend()
    plt.grid(True, axis='y')
    plt.savefig(os.path.join(output_dir, 'ms2_uncertainty_by_ion_type.png'))
    plt.close()
    
    # Create summary DataFrame
    summary = pd.DataFrame({
        'Method': ['MC Dropout', 'Ensemble'],
        'Average MS2 Uncertainty': [mc_avg_uncertainty, ensemble_avg_uncertainty],
        'Max MS2 Uncertainty': [np.max(mc_uncertainties), np.max(ensemble_uncertainties)],
        'Min MS2 Uncertainty': [np.min(mc_uncertainties[mc_uncertainties > 0]), 
                               np.min(ensemble_uncertainties[ensemble_uncertainties > 0])],
        'b-ion Uncertainty': [mc_b_uncertainty, ensemble_b_uncertainty],
        'y-ion Uncertainty': [mc_y_uncertainty, ensemble_y_uncertainty]
    })
    
    summary.to_csv(os.path.join(output_dir, 'ms2_uncertainty_summary.csv'), index=False)
    
    print(f"MS2 uncertainty analysis complete. Results saved to {output_dir}")
    
    return mc_ms2_results, ensemble_ms2_results


def run_combined_analysis(file_path='modificationSpecificPeptides.txt', rt_sample_size=None, ms2_sample_size=20):
    """
    Run the combined RT and MS2 uncertainty analysis
    
    Parameters
    ----------
    file_path : str
        Path to the modification-specific peptides file
    rt_sample_size : int, optional
        Number of peptides to sample for RT analysis. If None, use all peptides.
    ms2_sample_size : int
        Number of peptides to sample for MS2 analysis (smaller due to complexity)
    """
    base_output_dir = 'combined_results'
    os.makedirs(base_output_dir, exist_ok=True)
    
    # Load the peptide data for RT analysis
    rt_peptide_df = load_modification_specific_peptides(file_path, sample_size=rt_sample_size)
    
    # Save the sampled dataset for reference
    rt_peptide_df.to_csv(os.path.join(base_output_dir, 'rt_sampled_peptides.csv'), index=False)
    
    # Load the models (shared between RT and MS2 analysis)
    print("Loading models...")
    model_mgr = ModelManager(mask_modloss=False, device='cuda')
    model_mgr.load_installed_models()
    
    # Analyze RT uncertainty
    print("\n=== Starting RT Uncertainty Analysis ===\n")
    mc_rt_results, ensemble_rt_results = analyze_rt_uncertainty(rt_peptide_df, model_mgr, 
                                                              output_dir=os.path.join(base_output_dir, 'rt'))
    
    # Load a smaller subset for MS2 analysis (more computationally intensive)
    ms2_peptide_df = load_modification_specific_peptides(file_path, sample_size=ms2_sample_size)
    
    # Save the sampled dataset for reference
    ms2_peptide_df.to_csv(os.path.join(base_output_dir, 'ms2_sampled_peptides.csv'), index=False)
    
    # Analyze MS2 uncertainty
    print("\n=== Starting MS2 Uncertainty Analysis ===\n")
    mc_ms2_results, ensemble_ms2_results = analyze_ms2_uncertainty(ms2_peptide_df, model_mgr,
                                                                output_dir=os.path.join(base_output_dir, 'ms2'))
    
    # Create a combined summary
    print("\n=== Creating Combined Summary ===\n")
    
    # Extract key metrics
    rt_summary = pd.read_csv(os.path.join(base_output_dir, 'rt', 'rt_uncertainty_summary.csv'))
    ms2_summary = pd.read_csv(os.path.join(base_output_dir, 'ms2', 'ms2_uncertainty_summary.csv'))
    
    # Create a combined summary table
    combined_summary = pd.DataFrame({
        'Method': ['MC Dropout', 'Ensemble'],
        'RT PICP': rt_summary['PICP'].values,
        'RT MPIW': rt_summary['MPIW'].values,
        'RT Mean Absolute Error': rt_summary['Mean Absolute Error'].values,
        'RT Mean Uncertainty': rt_summary['Mean Uncertainty (Std)'].values,
        'MS2 Mean Uncertainty': ms2_summary['Average MS2 Uncertainty'].values,
        'MS2 b-ion Uncertainty': ms2_summary['b-ion Uncertainty'].values,
        'MS2 y-ion Uncertainty': ms2_summary['y-ion Uncertainty'].values
    })
    
    combined_summary.to_csv(os.path.join(base_output_dir, 'combined_summary.csv'), index=False)
    
    # Create a combined visualization
    plt.figure(figsize=(12, 8))
    
    # Set up the bar positions
    x = np.arange(2)
    width = 0.2
    
    # Plot RT metrics
    plt.bar(x - 1.5*width, rt_summary['PICP'].values, width, label='RT PICP')
    plt.bar(x - 0.5*width, rt_summary['MPIW'].values, width, label='RT MPIW')
    plt.bar(x + 0.5*width, rt_summary['Mean Uncertainty (Std)'].values, width, label='RT Uncertainty')
    
    # Plot MS2 metrics (scaled for visibility)
    ms2_uncertainty_scaled = ms2_summary['Average MS2 Uncertainty'].values * 10  # Scale for visibility
    plt.bar(x + 1.5*width, ms2_uncertainty_scaled, width, label='MS2 Uncertainty (×10)')
    
    plt.xlabel('Method')
    plt.ylabel('Value')
    plt.title('Combined RT and MS2 Uncertainty Metrics')
    plt.xticks(x, ['MC Dropout', 'Ensemble'])
    plt.legend()
    plt.grid(True, axis='y')
    plt.savefig(os.path.join(base_output_dir, 'combined_metrics_comparison.png'))
    plt.close()
    
    print(f"Combined analysis complete. Results saved to {base_output_dir}")
    print("\nKey findings:")
    print(f"1. RT Prediction - MC Dropout PICP: {rt_summary['PICP'].values[0]:.4f}, Ensemble PICP: {rt_summary['PICP'].values[1]:.4f}")
    print(f"2. MS2 Prediction - MC Dropout Uncertainty: {ms2_summary['Average MS2 Uncertainty'].values[0]:.4f}, Ensemble Uncertainty: {ms2_summary['Average MS2 Uncertainty'].values[1]:.10f}")
    print("\nSee the combined_summary.csv file for more details.")


if __name__ == "__main__":
    # Run the combined analysis
    run_combined_analysis()