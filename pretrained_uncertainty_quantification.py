#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Uncertainty Quantification for Pre-trained AlphaPeptDeep Models

This script implements uncertainty quantification for pre-trained AlphaPeptDeep models using:
1. Monte Carlo Dropout
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


class MCDropoutPredictor:
    """Monte Carlo Dropout predictor for uncertainty quantification"""
    
    def __init__(self, model, n_samples=30):
        """Initialize the MC Dropout predictor
        
        Parameters
        ----------
        model : ModelInterface
            The model to use for prediction
        n_samples : int
            Number of Monte Carlo samples to draw
        """
        self.model = model
        self.n_samples = n_samples
        
    def enable_dropout(self, model):
        """Enable dropout during inference"""
        for module in model.modules():
            if isinstance(module, torch.nn.Dropout):
                module.train()
                
    def predict(self, df, batch_size=1024):
        """Predict with uncertainty using Monte Carlo Dropout
        
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
        # Store original model state
        was_training = self.model.model.training
        
        # Set model to eval mode but enable dropout
        self.model.model.eval()
        self.enable_dropout(self.model.model)
        
        # Collect predictions
        predictions = []
        for _ in tqdm(range(self.n_samples), desc="MC Dropout Samples"):
            pred_df = self.model.predict(df, batch_size=batch_size)
            if hasattr(self.model, 'target_column_to_predict'):
                pred_col = self.model.target_column_to_predict
            else:
                # For RT model
                pred_col = 'rt_pred'
            predictions.append(pred_df[pred_col].values)
        
        # Restore original model state
        self.model.model.train(was_training)
        
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
            model_mgr = ModelManager(mask_modloss=False, device='cpu')
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
    # Calculate normalized errors
    normalized_errors = np.abs(y_true - y_pred) / y_std
    
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


def predict_ms2_with_uncertainty(model_mgr, df, n_samples=30):
    """Predict MS2 spectra with uncertainty using Monte Carlo Dropout
    
    Parameters
    ----------
    model_mgr : ModelManager
        Model manager
    df : pd.DataFrame
        DataFrame with peptide sequences
    n_samples : int
        Number of Monte Carlo samples
        
    Returns
    -------
    dict
        Dictionary with prediction results
    """
    # Store original model state
    was_training = model_mgr.ms2_model.model.training
    
    # Set model to eval mode but enable dropout
    model_mgr.ms2_model.model.eval()
    for module in model_mgr.ms2_model.model.modules():
        if isinstance(module, torch.nn.Dropout):
            module.train()
    
    # Prepare data for prediction
    init_fragment_by_precursor_dataframe(df, model_mgr.ms2_model.charged_frag_types)
    
    # Collect predictions
    all_intensities = []
    for _ in tqdm(range(n_samples), desc="MC Dropout Samples"):
        result = model_mgr.predict_all(df.copy(), predict_items=['ms2'])
        all_intensities.append(result['fragment_intensity_df'].values)
    
    # Restore original model state
    model_mgr.ms2_model.model.train(was_training)
    
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
    start_idx = df.iloc[peptide_idx]['frag_start_idx']
    end_idx = df.iloc[peptide_idx]['frag_stop_idx']
    
    # Get fragment m/z and intensities for this peptide
    frag_mz = ms2_results['fragment_mz_df'].iloc[start_idx:end_idx]
    frag_intensity_mean = ms2_results['fragment_intensity_mean_df'].iloc[start_idx:end_idx]
    frag_intensity_std = ms2_results['fragment_intensity_std_df'].iloc[start_idx:end_idx]
    
    # Plot for b and y ions
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Plot b ions
    b_cols = [col for col in frag_mz.columns if col.startswith('b_')]
    for col in b_cols:
        mz_values = frag_mz[col].values
        intensity_values = frag_intensity_mean[col].values
        std_values = frag_intensity_std[col].values
        
        # Filter out zero m/z values
        mask = mz_values > 0
        mz_values = mz_values[mask]
        intensity_values = intensity_values[mask]
        std_values = std_values[mask]
        
        if len(mz_values) > 0:
            ax.errorbar(mz_values, intensity_values, yerr=1.96*std_values, 
                      fmt='o', alpha=0.7, label=col)
    
    # Plot y ions
    y_cols = [col for col in frag_mz.columns if col.startswith('y_')]
    for col in y_cols:
        mz_values = frag_mz[col].values
        intensity_values = frag_intensity_mean[col].values
        std_values = frag_intensity_std[col].values
        
        # Filter out zero m/z values
        mask = mz_values > 0
        mz_values = mz_values[mask]
        intensity_values = intensity_values[mask]
        std_values = std_values[mask]
        
        if len(mz_values) > 0:
            ax.errorbar(mz_values, intensity_values, yerr=1.96*std_values, 
                      fmt='s', alpha=0.7, label=col)
    
    ax.set_xlabel('m/z')
    ax.set_ylabel('Relative Intensity')
    ax.set_title(f'MS2 Spectrum with Uncertainty for {df.iloc[peptide_idx]["sequence"]}')
    ax.legend()
    ax.grid(True)
    
    if save_path:
        plt.savefig(save_path)
    
    return fig


def run_rt_uncertainty_experiment(output_dir='pretrained_uncertainty_results'):
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
    model_mgr = ModelManager(mask_modloss=False, device='cpu')
    model_mgr.load_installed_models()
    
    print("Preparing dataset...")
    # Load a dataset for RT prediction
    df = IRT_PEPTIDE_DF.copy()
    df['rt_norm'] = df['irt']  # Use iRT values as normalized RT
    
    print("Running Monte Carlo Dropout experiment...")
    # Create MC Dropout predictor for RT model
    mc_dropout_rt = MCDropoutPredictor(model_mgr.rt_model, n_samples=30)
    
    # Predict with uncertainty
    mc_results = mc_dropout_rt.predict(df)
    
    # Evaluate
    picp = calculate_picp(mc_results['rt_norm'], mc_results['rt_pred_lower'], mc_results['rt_pred_upper'])
    mpiw = calculate_mpiw(mc_results['rt_pred_lower'], mc_results['rt_pred_upper'])
    
    print(f"Monte Carlo Dropout - PICP: {picp:.4f}, MPIW: {mpiw:.4f}")
    
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
    plt.title('RT Prediction with Uncertainty (Monte Carlo Dropout)')
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
    plt.scatter(mc_results['rt_norm'], mc_results['rt_pred_std'], alpha=0.5, label='MC Dropout')
    plt.scatter(ensemble_results['rt_norm'], ensemble_results['rt_pred_std'], alpha=0.5, label='Ensemble')
    plt.xlabel('True RT')
    plt.ylabel('Prediction Standard Deviation')
    plt.title('Uncertainty Comparison: MC Dropout vs Ensemble')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, 'uncertainty_comparison.png'))
    
    # Save results to CSV
    mc_results.to_csv(os.path.join(output_dir, 'mc_dropout_results.csv'), index=False)
    ensemble_results.to_csv(os.path.join(output_dir, 'ensemble_results.csv'), index=False)
    
    print(f"Results saved to {output_dir}")


def run_ms2_uncertainty_experiment(output_dir='pretrained_uncertainty_results/ms2'):
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
    model_mgr = ModelManager(mask_modloss=False, device='cpu')
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
    
    print("Running Monte Carlo Dropout for MS2 prediction...")
    # Predict MS2 with uncertainty
    ms2_results = predict_ms2_with_uncertainty(model_mgr, ms2_df, n_samples=10)
    
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
    output_dir = 'pretrained_uncertainty_results'
    os.makedirs(output_dir, exist_ok=True)
    
    # Run RT uncertainty experiment
    print("Running RT uncertainty experiment...")
    run_rt_uncertainty_experiment(output_dir=os.path.join(output_dir, 'rt'))
    
    # Run MS2 uncertainty experiment
    print("Running MS2 uncertainty experiment...")
    run_ms2_uncertainty_experiment(output_dir=os.path.join(output_dir, 'ms2'))