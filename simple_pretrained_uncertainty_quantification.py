#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Simple Uncertainty Quantification for Pre-trained AlphaPeptDeep Models

This script implements uncertainty quantification for pre-trained AlphaPeptDeep models using:
1. Simulated Monte Carlo Dropout (by adding noise to predictions)
2. Deep Ensembles (using different pre-trained models)

It evaluates the uncertainty estimates using Prediction Interval Coverage Probability (PICP)
and calibration curves.
"""

import os
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
from scipy.stats import norm
from tqdm import tqdm

from peptdeep.pretrained_models import ModelManager
from peptdeep.model.rt import IRT_PEPTIDE_DF, AlphaRTModel
from peptdeep.utils import evaluate_linear_regression, evaluate_linear_regression_plot
from alphabase.peptide.fragment import init_fragment_by_precursor_dataframe


class SimulatedMCDropoutPredictor:
    """Simulated Monte Carlo Dropout predictor for uncertainty quantification"""
    
    def __init__(self, model, n_samples=30, noise_scale=0.05):
        """Initialize the MC Dropout predictor
        
        Parameters
        ----------
        model : ModelInterface
            The model to use for prediction
        n_samples : int
            Number of Monte Carlo samples to draw
        noise_scale : float
            Scale of the Gaussian noise to add to predictions
        """
        self.model = model
        self.n_samples = n_samples
        self.noise_scale = noise_scale
        
    def predict(self, df, batch_size=1024):
        """Predict with uncertainty using simulated Monte Carlo Dropout
        
        Parameters
        ----------
        df : pd.DataFrame
            DataFrame with peptide sequences
        batch_size : int
            Batch size for prediction
            
        Returns
        -------
        pd.DataFrame
            DataFrame with predictions and uncertainty estimates
        """
        # Get base prediction
        pred_df = self.model.predict(df, batch_size=batch_size)
        
        # Get the prediction column
        if hasattr(self.model, 'target_column_to_predict'):
            pred_col = self.model.target_column_to_predict
        else:
            # For RT model
            pred_col = 'rt_pred'
        
        # Generate multiple predictions with added noise to simulate dropout effect
        predictions = []
        base_preds = pred_df[pred_col].values
        
        for _ in tqdm(range(self.n_samples), desc="Simulated MC Dropout Samples"):
            # Add noise to simulate dropout effect
            noise = np.random.normal(0, self.noise_scale, size=len(pred_df))
            noisy_preds = base_preds + noise
            predictions.append(noisy_preds)
        
        # Calculate statistics
        predictions = np.array(predictions)
        mean = np.mean(predictions, axis=0)
        std = np.std(predictions, axis=0)
        
        # Create result DataFrame
        result_df = df.copy()
        result_df[f'{pred_col}_mean'] = mean
        result_df[f'{pred_col}_std'] = std
        result_df[f'{pred_col}_lower'] = mean - 1.96 * std  # 95% confidence interval
        result_df[f'{pred_col}_upper'] = mean + 1.96 * std  # 95% confidence interval
        
        return result_df


class PretrainedEnsemblePredictor:
    """Ensemble predictor using different pre-trained models for uncertainty quantification"""
    
    def __init__(self, model_types=None, n_models=None):
        """Initialize the Ensemble predictor
        
        Parameters
        ----------
        model_types : list
            List of model types to use in the ensemble (e.g., ['generic', 'phospho', 'digly'])
            If None, will use all available model types
        n_models : int
            Number of models to use in the ensemble
            If None, will use all available models
        """
        if model_types is None:
            # Use all available model types
            self.model_types = ['generic', 'phospho', 'digly']
        else:
            self.model_types = model_types
            
        # Initialize model managers
        self.model_managers = []
        for model_type in self.model_types:
            model_mgr = ModelManager(mask_modloss=False, device='cuda')
            model_mgr.load_installed_models(model_type=model_type)
            self.model_managers.append(model_mgr)
            
        self.n_models = len(self.model_managers)
        
    def predict_rt(self, df, batch_size=1024):
        """Predict RT with uncertainty using the ensemble
        
        Parameters
        ----------
        df : pd.DataFrame
            DataFrame with peptide sequences
        batch_size : int
            Batch size for prediction
            
        Returns
        -------
        pd.DataFrame
            DataFrame with predictions and uncertainty estimates
        """
        # Collect predictions from all models
        predictions = []
        for i, model_mgr in enumerate(self.model_managers):
            print(f"Predicting with model {i+1}/{self.n_models} ({self.model_types[i]})")
            pred_df = model_mgr.rt_model.predict(df, batch_size=batch_size)
            predictions.append(pred_df['rt_pred'].values)
        
        # Calculate statistics
        predictions = np.array(predictions)
        mean = np.mean(predictions, axis=0)
        std = np.std(predictions, axis=0)
        
        # Create result DataFrame
        result_df = df.copy()
        result_df['rt_pred_mean'] = mean
        result_df['rt_pred_std'] = std
        result_df['rt_pred_lower'] = mean - 1.96 * std  # 95% confidence interval
        result_df['rt_pred_upper'] = mean + 1.96 * std  # 95% confidence interval
        
        return result_df
    
    def predict_ms2(self, df, batch_size=1024):
        """Predict MS2 with uncertainty using the ensemble
        
        Parameters
        ----------
        df : pd.DataFrame
            DataFrame with peptide sequences
        batch_size : int
            Batch size for prediction
            
        Returns
        -------
        dict
            Dictionary with prediction results
        """
        # Prepare data for prediction
        init_fragment_by_precursor_dataframe(df, self.model_managers[0].ms2_model.charged_frag_types)
        
        # Collect predictions from all models
        all_intensities = []
        for i, model_mgr in enumerate(self.model_managers):
            print(f"Predicting with model {i+1}/{self.n_models} ({self.model_types[i]})")
            result = model_mgr.predict_all(df.copy(), predict_items=['ms2'])
            all_intensities.append(result['fragment_intensity_df'].values)
        
        # Calculate statistics
        all_intensities = np.array(all_intensities)
        mean_intensities = np.mean(all_intensities, axis=0)
        std_intensities = np.std(all_intensities, axis=0)
        
        # Create result DataFrames
        mean_df = pd.DataFrame(mean_intensities, columns=result['fragment_intensity_df'].columns)
        std_df = pd.DataFrame(std_intensities, columns=result['fragment_intensity_df'].columns)
        
        return {
            'precursor_df': df,
            'fragment_mz_df': result['fragment_mz_df'],
            'fragment_intensity_mean_df': mean_df,
            'fragment_intensity_std_df': std_df
        }


def calculate_picp(y_true, y_lower, y_upper):
    """Calculate Prediction Interval Coverage Probability (PICP)
    
    Parameters
    ----------
    y_true : array-like
        True values
    y_lower : array-like
        Lower bounds of prediction intervals
    y_upper : array-like
        Upper bounds of prediction intervals
        
    Returns
    -------
    float
        PICP value (between 0 and 1)
    """
    covered = np.logical_and(y_true >= y_lower, y_true <= y_upper)
    return np.mean(covered)


def calculate_mpiw(y_lower, y_upper):
    """Calculate Mean Prediction Interval Width (MPIW)
    
    Parameters
    ----------
    y_lower : array-like
        Lower bounds of prediction intervals
    y_upper : array-like
        Upper bounds of prediction intervals
        
    Returns
    -------
    float
        MPIW value
    """
    return np.mean(y_upper - y_lower)


def plot_calibration_curve(y_true, y_pred, y_std, n_bins=10, save_path=None):
    """Plot calibration curve for uncertainty estimates
    
    Parameters
    ----------
    y_true : array-like
        True values
    y_pred : array-like
        Predicted values
    y_std : array-like
        Standard deviations of predictions
    n_bins : int
        Number of bins for the calibration curve
    save_path : str, optional
        Path to save the figure
        
    Returns
    -------
    matplotlib.figure.Figure
        Figure with calibration curve
    """
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
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(expected_freq, observed_freq, 'o-', label='Calibration curve')
    ax.plot([0, 1], [0, 1], 'k--', label='Ideal calibration')
    ax.set_xlabel('Expected cumulative probability')
    ax.set_ylabel('Observed cumulative probability')
    ax.set_title('Calibration Curve for Uncertainty Estimates')
    ax.legend()
    ax.grid(True)
    
    if save_path:
        plt.savefig(save_path)
    
    return fig


def predict_ms2_with_uncertainty(model_mgr, df, n_samples=30, noise_scale=0.05):
    """Predict MS2 spectra with uncertainty using simulated Monte Carlo Dropout
    
    Parameters
    ----------
    model_mgr : ModelManager or EnhancedModelManager
        Model manager
    df : pd.DataFrame
        DataFrame with peptide sequences
    n_samples : int
        Number of Monte Carlo samples
    noise_scale : float
        Scale of the Gaussian noise to add to predictions
        
    Returns
    -------
    dict
        Dictionary with prediction results
    """
    # Prepare data for prediction
    init_fragment_by_precursor_dataframe(df, model_mgr.ms2_model.charged_frag_types)
    
    # Get base prediction
    # Check if the model manager is an EnhancedModelManager
    if hasattr(model_mgr, 'predict_ms2') and callable(getattr(model_mgr, 'predict_ms2')):
        # For EnhancedModelManager
        result = model_mgr.predict_ms2(df.copy())
    else:
        # For standard ModelManager
        result = model_mgr.predict_all(df.copy(), predict_items=['ms2'])
    
    print(f"Available keys in result dictionary: {list(result.keys())}")
    
    # Handle different result formats
    if 'fragment_intensity_df' in result:
        # Standard format
        base_intensities = result['fragment_intensity_df'].values
        fragment_mz_df = result['fragment_mz_df']
    elif 'intensities_df' in result:
        # Alternative format
        base_intensities = result['intensities_df'].values
        fragment_mz_df = result.get('mz_df', pd.DataFrame())
    else:
        # Handle the case where result contains ion-specific DataFrames
        ion_keys = [k for k in result.keys() if k.startswith('b_') or k.startswith('y_')]
        
        if ion_keys:
            print(f"Found ion-specific keys: {ion_keys}")
            
            # Create a combined DataFrame for intensities
            combined_df = pd.DataFrame()
            
            # Process each peptide
            for i in range(len(df)):
                start_idx = df.iloc[i].get('frag_start_idx', i)
                end_idx = df.iloc[i].get('frag_stop_idx', i+1)
                
                # Combine all ion types for this peptide
                for ion_key in ion_keys:
                    if ion_key in result:
                        combined_df[ion_key] = result[ion_key]
            
            base_intensities = combined_df.values if not combined_df.empty else np.array([[0.0]])
            fragment_mz_df = pd.DataFrame(index=combined_df.index, columns=combined_df.columns)
            
            # Create a result dictionary in the expected format
            result = {
                'fragment_intensity_df': combined_df,
                'fragment_mz_df': fragment_mz_df
            }
        else:
            raise KeyError("Could not find fragment intensity data in the result")
    
    # Collect predictions with added noise
    all_intensities = []
    for _ in tqdm(range(n_samples), desc="Simulated MC Dropout Samples"):
        # Add noise to simulate dropout effect
        noise = np.random.normal(0, noise_scale, size=base_intensities.shape)
        noisy_intensities = base_intensities + noise
        # Clip to valid range [0, 1]
        noisy_intensities = np.clip(noisy_intensities, 0, 1)
        all_intensities.append(noisy_intensities)
    
    # Calculate statistics
    all_intensities = np.array(all_intensities)
    mean_intensities = np.mean(all_intensities, axis=0)
    std_intensities = np.std(all_intensities, axis=0)
    
    # Create result DataFrames
    mean_df = pd.DataFrame(mean_intensities, columns=result['fragment_intensity_df'].columns)
    std_df = pd.DataFrame(std_intensities, columns=result['fragment_intensity_df'].columns)
    
    return {
        'precursor_df': df,
        'fragment_mz_df': result['fragment_mz_df'],
        'fragment_intensity_mean_df': mean_df,
        'fragment_intensity_std_df': std_df
    }


def plot_ms2_with_uncertainty(ms2_results, peptide_idx=0, save_path=None):
    """Plot MS2 spectrum with uncertainty
    
    Parameters
    ----------
    ms2_results : dict
        MS2 prediction results
    peptide_idx : int
        Index of peptide to plot
    save_path : str, optional
        Path to save the figure
        
    Returns
    -------
    matplotlib.figure.Figure
        Figure with MS2 spectrum
    """
    # Get peptide info
    df = ms2_results['precursor_df']
    
    # Handle the case where frag_start_idx and frag_stop_idx might not be available
    if 'frag_start_idx' in df.columns and 'frag_stop_idx' in df.columns:
        start_idx = df.iloc[peptide_idx]['frag_start_idx']
        end_idx = df.iloc[peptide_idx]['frag_stop_idx']
        
        # Get fragment m/z and intensities for this peptide
        frag_mz = ms2_results['fragment_mz_df'].iloc[start_idx:end_idx]
        frag_intensity_mean = ms2_results['fragment_intensity_mean_df'].iloc[start_idx:end_idx]
        frag_intensity_std = ms2_results['fragment_intensity_std_df'].iloc[start_idx:end_idx]
    else:
        # If fragment indices are not available, use the entire DataFrame
        # This assumes we're working with a single peptide or a small batch
        print(f"Warning: frag_start_idx and frag_stop_idx not found. Using all data for peptide {peptide_idx}.")
        frag_mz = ms2_results['fragment_mz_df']
        frag_intensity_mean = ms2_results['fragment_intensity_mean_df']
        frag_intensity_std = ms2_results['fragment_intensity_std_df']
    
    # Plot for b and y ions
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Check if columns are strings or integers
    if frag_mz.columns.dtype == 'object':
        # String columns - filter by prefix
        b_cols = [col for col in frag_mz.columns if isinstance(col, str) and col.startswith('b_')]
        y_cols = [col for col in frag_mz.columns if isinstance(col, str) and col.startswith('y_')]
    else:
        # Integer columns - assume first half are b ions, second half are y ions
        num_cols = len(frag_mz.columns)
        b_cols = frag_mz.columns[:num_cols//2]
        y_cols = frag_mz.columns[num_cols//2:]
        print(f"Using integer columns: {num_cols//2} b-ions, {num_cols - num_cols//2} y-ions")
    
    # Plot b ions
    for col in b_cols:
        try:
            mz_values = frag_mz[col].values
            intensity_values = frag_intensity_mean[col].values
            std_values = frag_intensity_std[col].values
            
            # Filter out zero m/z values
            mask = np.isfinite(mz_values) & (mz_values > 0)
            mz_values = mz_values[mask]
            intensity_values = intensity_values[mask]
            std_values = std_values[mask]
            
            if len(mz_values) > 0:
                ax.errorbar(mz_values, intensity_values, yerr=1.96*std_values,
                          fmt='o', alpha=0.7, label=f"b-ion {col}")
        except Exception as e:
            print(f"Error plotting b-ion {col}: {e}")
    
    # Plot y ions
    for col in y_cols:
        try:
            mz_values = frag_mz[col].values
            intensity_values = frag_intensity_mean[col].values
            std_values = frag_intensity_std[col].values
            
            # Filter out zero m/z values
            mask = np.isfinite(mz_values) & (mz_values > 0)
            mz_values = mz_values[mask]
            intensity_values = intensity_values[mask]
            std_values = std_values[mask]
            
            if len(mz_values) > 0:
                ax.errorbar(mz_values, intensity_values, yerr=1.96*std_values,
                          fmt='s', alpha=0.7, label=f"y-ion {col}")
        except Exception as e:
            print(f"Error plotting y-ion {col}: {e}")
    
    ax.set_xlabel('m/z')
    ax.set_ylabel('Relative Intensity')
    ax.set_title(f'MS2 Spectrum with Uncertainty for {df.iloc[peptide_idx]["sequence"]}')
    ax.legend()
    ax.grid(True)
    
    if save_path:
        plt.savefig(save_path)
    
    return fig


def run_rt_uncertainty_experiment(output_dir='simple_uncertainty_results'):
    """Run RT uncertainty experiment with pre-trained models
    
    Parameters
    ----------
    output_dir : str
        Directory to save results
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    print("Loading models...")
    # Load the models
    model_mgr = ModelManager(mask_modloss=False, device='cuda')
    model_mgr.load_installed_models()
    
    print("Preparing dataset...")
    # Load a dataset for RT prediction
    df = IRT_PEPTIDE_DF.copy()
    df['rt_norm'] = df['irt']  # Use iRT values as normalized RT
    
    print("Running Simulated Monte Carlo Dropout experiment...")
    # Create MC Dropout predictor for RT model
    mc_dropout_rt = SimulatedMCDropoutPredictor(model_mgr.rt_model, n_samples=30, noise_scale=0.05)
    
    # Predict with uncertainty
    mc_results = mc_dropout_rt.predict(df)
    
    # Evaluate
    picp = calculate_picp(mc_results['rt_norm'], mc_results['rt_pred_lower'], mc_results['rt_pred_upper'])
    mpiw = calculate_mpiw(mc_results['rt_pred_lower'], mc_results['rt_pred_upper'])
    
    print(f"Simulated Monte Carlo Dropout - PICP: {picp:.4f}, MPIW: {mpiw:.4f}")
    
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
    
    print("Running Ensemble experiment...")
    # Create Ensemble predictor for RT model
    ensemble_rt = PretrainedEnsemblePredictor(model_types=['generic', 'phospho', 'digly'])
    
    # Predict with uncertainty
    ensemble_results = ensemble_rt.predict_rt(df)
    
    # Evaluate
    picp = calculate_picp(ensemble_results['rt_norm'], ensemble_results['rt_pred_lower'], ensemble_results['rt_pred_upper'])
    mpiw = calculate_mpiw(ensemble_results['rt_pred_lower'], ensemble_results['rt_pred_upper'])
    
    print(f"Ensemble - PICP: {picp:.4f}, MPIW: {mpiw:.4f}")
    
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
    
    # Save results to CSV
    mc_results.to_csv(os.path.join(output_dir, 'mc_dropout_results.csv'), index=False)
    ensemble_results.to_csv(os.path.join(output_dir, 'ensemble_results.csv'), index=False)
    
    print(f"Results saved to {output_dir}")


def run_ms2_uncertainty_experiment(output_dir='simple_uncertainty_results/ms2'):
    """Run MS2 uncertainty experiment with pre-trained models
    
    Parameters
    ----------
    output_dir : str
        Directory to save results
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    print("Loading models...")
    # Load the models
    model_mgr = ModelManager(mask_modloss=False, device='cuda')
    model_mgr.load_installed_models()
    
    print("Preparing dataset...")
    # Create a small dataset for MS2 prediction
    ms2_df = pd.DataFrame({
        'sequence': ['LGGNEQVTR', 'GAGSSEPVTGLDAK', 'VEATFGVDESNAK', 'YILAGVENSK'],
        'mods': ['', '', '', ''],
        'mod_sites': ['', '', '', ''],
        'charge': [2, 2, 2, 2],
        'nce': [30, 30, 30, 30],
        'instrument': ['QE', 'QE', 'QE', 'QE']
    })
    ms2_df['nAA'] = ms2_df.sequence.str.len()
    
    print("Running Simulated Monte Carlo Dropout for MS2 prediction...")
    # Predict MS2 with uncertainty
    ms2_results = predict_ms2_with_uncertainty(model_mgr, ms2_df, n_samples=10, noise_scale=0.05)
    
    # Plot MS2 spectrum with uncertainty for each peptide
    for i in range(len(ms2_df)):
        fig = plot_ms2_with_uncertainty(
            ms2_results, 
            peptide_idx=i,
            save_path=os.path.join(output_dir, f'ms2_uncertainty_{ms2_df.iloc[i]["sequence"]}.png')
        )
    
    print("Running Ensemble for MS2 prediction...")
    # Create Ensemble predictor for MS2 model
    ensemble_ms2 = PretrainedEnsemblePredictor(model_types=['generic', 'phospho', 'digly'])
    
    # Predict with uncertainty
    ensemble_results = ensemble_ms2.predict_ms2(ms2_df)
    
    # Plot MS2 spectrum with uncertainty for each peptide
    for i in range(len(ms2_df)):
        fig = plot_ms2_with_uncertainty(
            ensemble_results, 
            peptide_idx=i,
            save_path=os.path.join(output_dir, f'ensemble_ms2_uncertainty_{ms2_df.iloc[i]["sequence"]}.png')
        )
    
    print(f"Results saved to {output_dir}")


if __name__ == "__main__":
    # Create results directory
    output_dir = 'simple_uncertainty_results'
    os.makedirs(output_dir, exist_ok=True)
    
    # Run RT uncertainty experiment
    print("Running RT uncertainty experiment...")
    run_rt_uncertainty_experiment(output_dir=os.path.join(output_dir, 'rt'))
    
    # Run MS2 uncertainty experiment
    print("Running MS2 uncertainty experiment...")
    run_ms2_uncertainty_experiment(output_dir=os.path.join(output_dir, 'ms2'))