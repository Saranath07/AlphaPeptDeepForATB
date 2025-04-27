#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
MS2 Uncertainty Quantification Analysis for Modification-Specific Peptides

This script extends the uncertainty quantification analysis to include MS2 predictions:
1. Simulated Monte Carlo Dropout for MS2 intensity predictions
2. Deep Ensembles for MS2 intensity predictions

It evaluates the uncertainty in fragment ion intensity predictions and visualizes the results.
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
    predict_ms2_with_uncertainty,
    plot_ms2_with_uncertainty
)

# Import the peptide loading function from our RT analysis
from modification_specific_uncertainty_analysis import load_modification_specific_peptides


def analyze_ms2_uncertainty(peptide_df, output_dir='ms2_uncertainty_results'):
    """
    Analyze MS2 prediction uncertainty for the peptide dataset
    
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
    
    print("Running Simulated Monte Carlo Dropout for MS2 prediction...")
    # Predict MS2 with uncertainty using MC Dropout
    mc_ms2_results = predict_ms2_with_uncertainty(model_mgr, df, n_samples=30, noise_scale=0.05)
    
    # Plot MS2 spectra with uncertainty for each peptide
    print("Generating MS2 spectra plots with uncertainty...")
    for i in range(min(len(df), 10)):  # Limit to first 10 peptides to avoid too many plots
        peptide_seq = df.iloc[i]['sequence']
        print(f"Plotting MS2 spectrum for peptide {i+1}/{min(len(df), 10)}: {peptide_seq}")
        
        fig = plot_ms2_with_uncertainty(
            mc_ms2_results, 
            peptide_idx=i,
            save_path=os.path.join(output_dir, f'mc_dropout_ms2_uncertainty_{peptide_seq}.png')
        )
    
    print("Running Ensemble for MS2 prediction...")
    # Create Ensemble predictor for MS2 model
    ensemble = PretrainedEnsemblePredictor(model_types=['generic', 'phospho', 'digly'])
    
    # Predict with uncertainty
    ensemble_ms2_results = ensemble.predict_ms2(df)
    
    # Plot MS2 spectra with uncertainty for each peptide
    print("Generating MS2 spectra plots with uncertainty (Ensemble)...")
    for i in range(min(len(df), 10)):  # Limit to first 10 peptides
        peptide_seq = df.iloc[i]['sequence']
        print(f"Plotting MS2 spectrum for peptide {i+1}/{min(len(df), 10)}: {peptide_seq}")
        
        fig = plot_ms2_with_uncertainty(
            ensemble_ms2_results, 
            peptide_idx=i,
            save_path=os.path.join(output_dir, f'ensemble_ms2_uncertainty_{peptide_seq}.png')
        )
    
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
    print(f"Ensemble - Average MS2 intensity uncertainty: {ensemble_avg_uncertainty:.4f}")
    
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
    
    # Create summary DataFrame
    summary = pd.DataFrame({
        'Method': ['MC Dropout', 'Ensemble'],
        'Average MS2 Uncertainty': [mc_avg_uncertainty, ensemble_avg_uncertainty],
        'Max MS2 Uncertainty': [np.max(mc_uncertainties), np.max(ensemble_uncertainties)],
        'Min MS2 Uncertainty': [np.min(mc_uncertainties[mc_uncertainties > 0]), 
                               np.min(ensemble_uncertainties[ensemble_uncertainties > 0])],
    })
    
    summary.to_csv(os.path.join(output_dir, 'ms2_uncertainty_summary.csv'), index=False)
    
    print(f"MS2 uncertainty analysis complete. Results saved to {output_dir}")
    
    return mc_ms2_results, ensemble_ms2_results


def analyze_ms2_by_ion_type(mc_ms2_results, ensemble_ms2_results, output_dir='ms2_uncertainty_results'):
    """
    Analyze MS2 uncertainty by ion type (b and y ions)
    
    Parameters
    ----------
    mc_ms2_results : dict
        Results from MC Dropout
    ensemble_ms2_results : dict
        Results from Ensemble
    output_dir : str
        Directory to save results
    """
    # Extract fragment intensity and uncertainty DataFrames
    mc_intensities = mc_ms2_results['fragment_intensity_mean_df']
    mc_uncertainties = mc_ms2_results['fragment_intensity_std_df']
    
    ensemble_intensities = ensemble_ms2_results['fragment_intensity_mean_df']
    ensemble_uncertainties = ensemble_ms2_results['fragment_intensity_std_df']
    
    # Separate b and y ions
    b_ion_cols = [col for col in mc_intensities.columns if col.startswith('b_')]
    y_ion_cols = [col for col in mc_intensities.columns if col.startswith('y_')]
    
    # Calculate average uncertainty for each ion type
    mc_b_uncertainty = mc_uncertainties[b_ion_cols].values.mean()
    mc_y_uncertainty = mc_uncertainties[y_ion_cols].values.mean()
    
    ensemble_b_uncertainty = ensemble_uncertainties[b_ion_cols].values.mean()
    ensemble_y_uncertainty = ensemble_uncertainties[y_ion_cols].values.mean()
    
    print(f"MC Dropout - Average b-ion uncertainty: {mc_b_uncertainty:.4f}")
    print(f"MC Dropout - Average y-ion uncertainty: {mc_y_uncertainty:.4f}")
    print(f"Ensemble - Average b-ion uncertainty: {ensemble_b_uncertainty:.4f}")
    print(f"Ensemble - Average y-ion uncertainty: {ensemble_y_uncertainty:.4f}")
    
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
    
    # Create summary DataFrame
    summary = pd.DataFrame({
        'Method': ['MC Dropout', 'MC Dropout', 'Ensemble', 'Ensemble'],
        'Ion Type': ['b-ions', 'y-ions', 'b-ions', 'y-ions'],
        'Average Uncertainty': [mc_b_uncertainty, mc_y_uncertainty, 
                               ensemble_b_uncertainty, ensemble_y_uncertainty]
    })
    
    summary.to_csv(os.path.join(output_dir, 'ms2_uncertainty_by_ion_type.csv'), index=False)


def analyze_ms2_by_peptide_length(peptide_df, mc_ms2_results, ensemble_ms2_results, output_dir='ms2_uncertainty_results'):
    """
    Analyze MS2 uncertainty by peptide length
    
    Parameters
    ----------
    peptide_df : pd.DataFrame
        DataFrame with peptide data
    mc_ms2_results : dict
        Results from MC Dropout
    ensemble_ms2_results : dict
        Results from Ensemble
    output_dir : str
        Directory to save results
    """
    # Extract peptide sequences and lengths
    sequences = peptide_df['sequence'].tolist()
    lengths = peptide_df['nAA'].tolist()
    
    # Create length categories
    length_categories = pd.cut(lengths, bins=[0, 8, 12, 16, 100], 
                              labels=['Short (≤8)', 'Medium (9-12)', 'Long (13-16)', 'Very Long (>16)'])
    
    # Initialize arrays to store average uncertainties by peptide
    mc_uncertainties = []
    ensemble_uncertainties = []
    
    # Calculate average uncertainty for each peptide
    for i in range(len(peptide_df)):
        start_idx = peptide_df.iloc[i]['frag_start_idx'] if 'frag_start_idx' in peptide_df.columns else None
        end_idx = peptide_df.iloc[i]['frag_stop_idx'] if 'frag_stop_idx' in peptide_df.columns else None
        
        if start_idx is None or end_idx is None:
            # Skip if fragment indices are not available
            continue
        
        # Extract fragment uncertainties for this peptide
        mc_peptide_uncertainty = mc_ms2_results['fragment_intensity_std_df'].iloc[start_idx:end_idx].values.mean()
        ensemble_peptide_uncertainty = ensemble_ms2_results['fragment_intensity_std_df'].iloc[start_idx:end_idx].values.mean()
        
        mc_uncertainties.append(mc_peptide_uncertainty)
        ensemble_uncertainties.append(ensemble_peptide_uncertainty)
    
    # Create DataFrame for analysis
    analysis_df = pd.DataFrame({
        'sequence': sequences[:len(mc_uncertainties)],
        'length': lengths[:len(mc_uncertainties)],
        'length_category': length_categories[:len(mc_uncertainties)],
        'mc_uncertainty': mc_uncertainties,
        'ensemble_uncertainty': ensemble_uncertainties
    })
    
    # Group by length category
    length_summary = analysis_df.groupby('length_category').agg({
        'mc_uncertainty': ['mean', 'std'],
        'ensemble_uncertainty': ['mean', 'std']
    })
    
    # Save summary
    length_summary.to_csv(os.path.join(output_dir, 'ms2_uncertainty_by_length.csv'))
    
    # Plot comparison of uncertainties by length category
    plt.figure(figsize=(14, 8))
    
    # Extract data for different length categories
    categories = length_summary.index.tolist()
    x = np.arange(len(categories))
    width = 0.35
    
    # Get mean uncertainties for each category
    mc_means = [length_summary.loc[cat, ('mc_uncertainty', 'mean')] for cat in categories]
    ensemble_means = [length_summary.loc[cat, ('ensemble_uncertainty', 'mean')] for cat in categories]
    
    # Create the bar plot
    plt.bar(x - width/2, mc_means, width, label='MC Dropout')
    plt.bar(x + width/2, ensemble_means, width, label='Ensemble')
    
    plt.xlabel('Peptide Length')
    plt.ylabel('Average MS2 Uncertainty')
    plt.title('Impact of Peptide Length on MS2 Prediction Uncertainty')
    plt.xticks(x, categories)
    plt.legend()
    plt.grid(True, axis='y')
    plt.savefig(os.path.join(output_dir, 'ms2_uncertainty_by_length.png'))
    
    # Plot correlation between peptide length and uncertainty
    plt.figure(figsize=(12, 6))
    plt.scatter(analysis_df['length'], analysis_df['mc_uncertainty'], alpha=0.7, label='MC Dropout')
    plt.scatter(analysis_df['length'], analysis_df['ensemble_uncertainty'], alpha=0.7, label='Ensemble')
    plt.xlabel('Peptide Length')
    plt.ylabel('Average MS2 Uncertainty')
    plt.title('Correlation between Peptide Length and MS2 Uncertainty')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, 'ms2_length_correlation.png'))


def run_ms2_uncertainty_analysis(file_path='modificationSpecificPeptides.txt', sample_size=20):
    """
    Run the complete MS2 uncertainty analysis for modification-specific peptides
    
    Parameters
    ----------
    file_path : str
        Path to the modification-specific peptides file
    sample_size : int
        Number of peptides to sample (smaller than RT analysis due to MS2 complexity)
    """
    output_dir = 'ms2_uncertainty_results'
    os.makedirs(output_dir, exist_ok=True)
    
    # Load the peptide data
    peptide_df = load_modification_specific_peptides(file_path, sample_size)
    
    # Save the sampled dataset for reference
    peptide_df.to_csv(os.path.join(output_dir, 'sampled_peptides.csv'), index=False)
    
    # Analyze MS2 uncertainty
    mc_ms2_results, ensemble_ms2_results = analyze_ms2_uncertainty(peptide_df, output_dir)
    
    # Analyze MS2 uncertainty by ion type
    analyze_ms2_by_ion_type(mc_ms2_results, ensemble_ms2_results, output_dir)
    
    # Analyze MS2 uncertainty by peptide length
    analyze_ms2_by_peptide_length(peptide_df, mc_ms2_results, ensemble_ms2_results, output_dir)
    
    print(f"MS2 uncertainty analysis complete. Results saved to {output_dir}")


if __name__ == "__main__":
    # Run the MS2 uncertainty analysis
    run_ms2_uncertainty_analysis()