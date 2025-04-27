#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Improved Uncertainty Quantification Analysis for Modification-Specific Peptides

This script implements improvements to the uncertainty quantification analysis:
1. Alternative RT normalization approaches
2. Post-hoc uncertainty calibration
3. Cross-validation for more robust evaluation
4. Feature engineering with additional peptide properties
"""

import os
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
from scipy.stats import norm
from tqdm import tqdm
from sklearn.model_selection import KFold
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import mean_absolute_error

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
    
    # Add additional peptide features
    peptide_df = add_peptide_features(peptide_df)
    
    return peptide_df


def add_peptide_features(df):
    """
    Add additional peptide features that might improve prediction
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with peptide data
        
    Returns
    -------
    pd.DataFrame
        DataFrame with additional features
    """
    # Define amino acid properties
    aa_hydrophobicity = {
        'A': 1.8, 'R': -4.5, 'N': -3.5, 'D': -3.5, 'C': 2.5,
        'Q': -3.5, 'E': -3.5, 'G': -0.4, 'H': -3.2, 'I': 4.5,
        'L': 3.8, 'K': -3.9, 'M': 1.9, 'F': 2.8, 'P': -1.6,
        'S': -0.8, 'T': -0.7, 'W': -0.9, 'Y': -1.3, 'V': 4.2
    }
    
    # Calculate average hydrophobicity
    df['avg_hydrophobicity'] = df['sequence'].apply(
        lambda seq: sum(aa_hydrophobicity.get(aa, 0) for aa in seq) / len(seq)
    )
    
    # Calculate N-terminal and C-terminal amino acid features
    df['n_term_aa'] = df['sequence'].str[0]
    df['c_term_aa'] = df['sequence'].str[-1]
    
    # One-hot encode N-terminal and C-terminal amino acids
    n_term_dummies = pd.get_dummies(df['n_term_aa'], prefix='n_term')
    c_term_dummies = pd.get_dummies(df['c_term_aa'], prefix='c_term')
    
    # Add to dataframe
    df = pd.concat([df, n_term_dummies, c_term_dummies], axis=1)
    
    return df


def normalize_rt_values(df, method='minmax'):
    """
    Normalize RT values using different methods
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with peptide data
    method : str
        Normalization method: 'minmax', 'zscore', 'robust', or 'none'
        
    Returns
    -------
    pd.DataFrame
        DataFrame with normalized RT values
    """
    # Make a copy to avoid modifying the original
    result_df = df.copy()
    
    if method == 'none':
        # Use raw RT values
        result_df['rt_norm'] = result_df['rt']
    elif method == 'minmax':
        # Min-max normalization to [0, 100] range
        rt_min = result_df['rt'].min()
        rt_max = result_df['rt'].max()
        result_df['rt_norm'] = (result_df['rt'] - rt_min) / (rt_max - rt_min) * 100
    elif method == 'zscore':
        # Z-score normalization
        rt_mean = result_df['rt'].mean()
        rt_std = result_df['rt'].std()
        result_df['rt_norm'] = (result_df['rt'] - rt_mean) / rt_std
    elif method == 'robust':
        # Robust scaling using median and IQR
        rt_median = result_df['rt'].median()
        rt_iqr = result_df['rt'].quantile(0.75) - result_df['rt'].quantile(0.25)
        result_df['rt_norm'] = (result_df['rt'] - rt_median) / rt_iqr
    else:
        raise ValueError(f"Unknown normalization method: {method}")
    
    return result_df


class UncertaintyCalibrator:
    """Class for post-hoc calibration of uncertainty estimates"""
    
    def __init__(self, method='isotonic'):
        """
        Initialize the calibrator
        
        Parameters
        ----------
        method : str
            Calibration method: 'isotonic' or 'temperature'
        """
        self.method = method
        self.calibrator = None
    
    def fit(self, pred_mean, pred_std, true_values):
        """
        Fit the calibrator to the data
        
        Parameters
        ----------
        pred_mean : array-like
            Predicted mean values
        pred_std : array-like
            Predicted standard deviations
        true_values : array-like
            True values
        """
        # Calculate normalized errors
        normalized_errors = np.abs(true_values - pred_mean) / pred_std
        
        if self.method == 'isotonic':
            # Use isotonic regression to calibrate the errors
            # We map from predicted confidence level to empirical confidence level
            confidence_levels = np.linspace(0.01, 0.99, 99)
            pred_quantiles = np.array([norm.ppf((1 + cl) / 2) for cl in confidence_levels])
            emp_quantiles = np.array([
                np.mean(normalized_errors <= q) for q in pred_quantiles
            ])
            
            self.calibrator = IsotonicRegression(out_of_bounds='clip')
            self.calibrator.fit(pred_quantiles, emp_quantiles)
        
        elif self.method == 'temperature':
            # Temperature scaling - find optimal temperature parameter
            # to scale the standard deviations
            from scipy.optimize import minimize_scalar
            
            def neg_log_likelihood(T):
                # Negative log likelihood for Gaussian with scaled std
                scaled_std = pred_std * T
                nll = np.mean(
                    0.5 * np.log(2 * np.pi * scaled_std**2) + 
                    0.5 * ((true_values - pred_mean) / scaled_std)**2
                )
                return nll
            
            result = minimize_scalar(neg_log_likelihood, bounds=(0.1, 10), method='bounded')
            self.calibrator = result.x
        
        else:
            raise ValueError(f"Unknown calibration method: {self.method}")
    
    def calibrate(self, pred_mean, pred_std):
        """
        Calibrate the uncertainty estimates
        
        Parameters
        ----------
        pred_mean : array-like
            Predicted mean values
        pred_std : array-like
            Predicted standard deviations
            
        Returns
        -------
        tuple
            Calibrated mean and standard deviation
        """
        if self.calibrator is None:
            raise ValueError("Calibrator not fitted. Call fit() first.")
        
        if self.method == 'isotonic':
            # Apply isotonic regression to get calibrated confidence levels
            # We don't change the mean, only the standard deviation
            calibrated_std = pred_std.copy()
            
            # For each prediction, calculate the calibrated standard deviation
            for i in range(len(pred_mean)):
                # Calculate z-scores for different confidence levels
                confidence_levels = np.linspace(0.01, 0.99, 99)
                z_scores = np.array([norm.ppf((1 + cl) / 2) for cl in confidence_levels])
                
                # Calibrate the z-scores
                calibrated_z = z_scores / self.calibrator.predict(z_scores)
                
                # Use the median calibrated z-score to adjust the standard deviation
                median_calib_z = np.median(calibrated_z)
                calibrated_std[i] = pred_std[i] * median_calib_z
            
            return pred_mean, calibrated_std
        
        elif self.method == 'temperature':
            # Scale the standard deviations by the temperature parameter
            return pred_mean, pred_std * self.calibrator


def cross_validate_uncertainty(peptide_df, n_splits=5, norm_method='minmax', calib_method='isotonic'):
    """
    Perform cross-validation to evaluate uncertainty quantification
    
    Parameters
    ----------
    peptide_df : pd.DataFrame
        DataFrame with peptide data
    n_splits : int
        Number of cross-validation folds
    norm_method : str
        RT normalization method
    calib_method : str
        Uncertainty calibration method
        
    Returns
    -------
    dict
        Dictionary with cross-validation results
    """
    # Initialize cross-validation
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    
    # Initialize results
    results = {
        'mc_dropout': {
            'picp': [], 'mpiw': [], 'mae': [],
            'picp_calibrated': [], 'mpiw_calibrated': []
        },
        'ensemble': {
            'picp': [], 'mpiw': [], 'mae': [],
            'picp_calibrated': [], 'mpiw_calibrated': []
        }
    }
    
    # Perform cross-validation
    for fold, (train_idx, test_idx) in enumerate(kf.split(peptide_df)):
        print(f"Processing fold {fold+1}/{n_splits}")
        
        # Split data
        train_df = peptide_df.iloc[train_idx].copy()
        test_df = peptide_df.iloc[test_idx].copy()
        
        # Normalize RT values
        train_df = normalize_rt_values(train_df, method=norm_method)
        test_df = normalize_rt_values(test_df, method=norm_method)
        
        # Load models
        model_mgr = ModelManager(mask_modloss=False, device='cuda')
        model_mgr.load_installed_models()
        
        # MC Dropout
        print("Running MC Dropout...")
        mc_dropout = SimulatedMCDropoutPredictor(model_mgr.rt_model, n_samples=30, noise_scale=0.05)
        mc_results = mc_dropout.predict(test_df)
        
        # Calculate metrics
        picp = calculate_picp(mc_results['rt_norm'], mc_results['rt_pred_lower'], mc_results['rt_pred_upper'])
        mpiw = calculate_mpiw(mc_results['rt_pred_lower'], mc_results['rt_pred_upper'])
        mae = mean_absolute_error(mc_results['rt_norm'], mc_results['rt_pred_mean'])
        
        results['mc_dropout']['picp'].append(picp)
        results['mc_dropout']['mpiw'].append(mpiw)
        results['mc_dropout']['mae'].append(mae)
        
        # Calibrate uncertainty
        calibrator = UncertaintyCalibrator(method=calib_method)
        calibrator.fit(mc_results['rt_pred_mean'].values, mc_results['rt_pred_std'].values, 
                      mc_results['rt_norm'].values)
        
        calib_mean, calib_std = calibrator.calibrate(
            mc_results['rt_pred_mean'].values, mc_results['rt_pred_std'].values
        )
        
        # Calculate calibrated confidence intervals
        mc_results['rt_pred_calib_std'] = calib_std
        mc_results['rt_pred_calib_lower'] = calib_mean - 1.96 * calib_std
        mc_results['rt_pred_calib_upper'] = calib_mean + 1.96 * calib_std
        
        # Calculate calibrated metrics
        picp_calib = calculate_picp(mc_results['rt_norm'], mc_results['rt_pred_calib_lower'], 
                                   mc_results['rt_pred_calib_upper'])
        mpiw_calib = calculate_mpiw(mc_results['rt_pred_calib_lower'], mc_results['rt_pred_calib_upper'])
        
        results['mc_dropout']['picp_calibrated'].append(picp_calib)
        results['mc_dropout']['mpiw_calibrated'].append(mpiw_calib)
        
        # Ensemble
        print("Running Ensemble...")
        ensemble = PretrainedEnsemblePredictor(model_types=['generic', 'phospho', 'digly'])
        ensemble_results = ensemble.predict_rt(test_df)
        
        # Calculate metrics
        picp = calculate_picp(ensemble_results['rt_norm'], ensemble_results['rt_pred_lower'], 
                             ensemble_results['rt_pred_upper'])
        mpiw = calculate_mpiw(ensemble_results['rt_pred_lower'], ensemble_results['rt_pred_upper'])
        mae = mean_absolute_error(ensemble_results['rt_norm'], ensemble_results['rt_pred_mean'])
        
        results['ensemble']['picp'].append(picp)
        results['ensemble']['mpiw'].append(mpiw)
        results['ensemble']['mae'].append(mae)
        
        # Calibrate uncertainty
        calibrator = UncertaintyCalibrator(method=calib_method)
        calibrator.fit(ensemble_results['rt_pred_mean'].values, ensemble_results['rt_pred_std'].values, 
                      ensemble_results['rt_norm'].values)
        
        calib_mean, calib_std = calibrator.calibrate(
            ensemble_results['rt_pred_mean'].values, ensemble_results['rt_pred_std'].values
        )
        
        # Calculate calibrated confidence intervals
        ensemble_results['rt_pred_calib_std'] = calib_std
        ensemble_results['rt_pred_calib_lower'] = calib_mean - 1.96 * calib_std
        ensemble_results['rt_pred_calib_upper'] = calib_mean + 1.96 * calib_std
        
        # Calculate calibrated metrics
        picp_calib = calculate_picp(ensemble_results['rt_norm'], ensemble_results['rt_pred_calib_lower'], 
                                   ensemble_results['rt_pred_calib_upper'])
        mpiw_calib = calculate_mpiw(ensemble_results['rt_pred_calib_lower'], ensemble_results['rt_pred_calib_upper'])
        
        results['ensemble']['picp_calibrated'].append(picp_calib)
        results['ensemble']['mpiw_calibrated'].append(mpiw_calib)
    
    # Calculate average metrics
    for method in results:
        for metric in results[method]:
            results[method][f'{metric}_mean'] = np.mean(results[method][metric])
            results[method][f'{metric}_std'] = np.std(results[method][metric])
    
    return results


def compare_normalization_methods(peptide_df, output_dir='improved_results'):
    """
    Compare different RT normalization methods
    
    Parameters
    ----------
    peptide_df : pd.DataFrame
        DataFrame with peptide data
    output_dir : str
        Directory to save results
        
    Returns
    -------
    pd.DataFrame
        DataFrame with comparison results
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Normalization methods to compare
    methods = ['minmax', 'zscore', 'robust', 'none']
    
    # Initialize results
    results = []
    
    # Load models
    model_mgr = ModelManager(mask_modloss=False, device='cuda')
    model_mgr.load_installed_models()
    
    # Compare methods
    for method in methods:
        print(f"Testing normalization method: {method}")
        
        # Normalize RT values
        df = normalize_rt_values(peptide_df, method=method)
        
        # MC Dropout
        mc_dropout = SimulatedMCDropoutPredictor(model_mgr.rt_model, n_samples=30, noise_scale=0.05)
        mc_results = mc_dropout.predict(df)
        
        # Calculate metrics
        picp = calculate_picp(mc_results['rt_norm'], mc_results['rt_pred_lower'], mc_results['rt_pred_upper'])
        mpiw = calculate_mpiw(mc_results['rt_pred_lower'], mc_results['rt_pred_upper'])
        mae = mean_absolute_error(mc_results['rt_norm'], mc_results['rt_pred_mean'])
        
        # Store results
        results.append({
            'Method': f'MC Dropout - {method}',
            'PICP': picp,
            'MPIW': mpiw,
            'MAE': mae
        })
        
        # Ensemble
        ensemble = PretrainedEnsemblePredictor(model_types=['generic', 'phospho', 'digly'])
        ensemble_results = ensemble.predict_rt(df)
        
        # Calculate metrics
        picp = calculate_picp(ensemble_results['rt_norm'], ensemble_results['rt_pred_lower'], 
                             ensemble_results['rt_pred_upper'])
        mpiw = calculate_mpiw(ensemble_results['rt_pred_lower'], ensemble_results['rt_pred_upper'])
        mae = mean_absolute_error(ensemble_results['rt_norm'], ensemble_results['rt_pred_mean'])
        
        # Store results
        results.append({
            'Method': f'Ensemble - {method}',
            'PICP': picp,
            'MPIW': mpiw,
            'MAE': mae
        })
    
    # Create results DataFrame
    results_df = pd.DataFrame(results)
    
    # Save results
    results_df.to_csv(os.path.join(output_dir, 'normalization_comparison.csv'), index=False)
    
    # Plot comparison
    plt.figure(figsize=(12, 6))
    
    # Group by normalization method
    for i, method in enumerate(methods):
        mc_row = results_df[results_df['Method'] == f'MC Dropout - {method}']
        ensemble_row = results_df[results_df['Method'] == f'Ensemble - {method}']
        
        x = i * 3
        width = 0.35
        
        plt.bar(x - width/2, mc_row['PICP'], width, label=f'MC Dropout - {method}' if i == 0 else '')
        plt.bar(x + width/2, ensemble_row['PICP'], width, label=f'Ensemble - {method}' if i == 0 else '')
    
    plt.xlabel('Normalization Method')
    plt.ylabel('PICP')
    plt.title('Impact of Normalization Method on PICP')
    plt.xticks(np.arange(len(methods)) * 3, methods)
    plt.legend()
    plt.grid(True, axis='y')
    plt.savefig(os.path.join(output_dir, 'normalization_comparison_picp.png'))
    
    return results_df


def run_improved_analysis(file_path='modificationSpecificPeptides.txt'):
    """
    Run the improved analysis for modification-specific peptides
    
    Parameters
    ----------
    file_path : str
        Path to the modification-specific peptides file
    """
    output_dir = 'improved_results'
    os.makedirs(output_dir, exist_ok=True)
    
    # Load the peptide data (use all available unmodified peptides)
    peptide_df = load_modification_specific_peptides(file_path, sample_size=None)
    
    # Save the dataset for reference
    peptide_df.to_csv(os.path.join(output_dir, 'peptides_with_features.csv'), index=False)
    
    # Compare normalization methods
    norm_results = compare_normalization_methods(peptide_df, output_dir)
    
    # Determine best normalization method
    best_method = norm_results.loc[norm_results['PICP'].idxmax()]['Method'].split(' - ')[1]
    print(f"Best normalization method: {best_method}")
    
    # Run cross-validation with the best normalization method
    cv_results = cross_validate_uncertainty(
        peptide_df, n_splits=5, norm_method=best_method, calib_method='isotonic'
    )
    
    # Save cross-validation results
    cv_summary = {
        'Method': ['MC Dropout', 'MC Dropout (Calibrated)', 'Ensemble', 'Ensemble (Calibrated)'],
        'PICP': [
            cv_results['mc_dropout']['picp_mean'],
            cv_results['mc_dropout']['picp_calibrated_mean'],
            cv_results['ensemble']['picp_mean'],
            cv_results['ensemble']['picp_calibrated_mean']
        ],
        'MPIW': [
            cv_results['mc_dropout']['mpiw_mean'],
            cv_results['mc_dropout']['mpiw_calibrated_mean'],
            cv_results['ensemble']['mpiw_mean'],
            cv_results['ensemble']['mpiw_calibrated_mean']
        ],
        'MAE': [
            cv_results['mc_dropout']['mae_mean'],
            cv_results['mc_dropout']['mae_mean'],  # Same MAE for calibrated and uncalibrated
            cv_results['ensemble']['mae_mean'],
            cv_results['ensemble']['mae_mean']  # Same MAE for calibrated and uncalibrated
        ]
    }
    
    cv_summary_df = pd.DataFrame(cv_summary)
    cv_summary_df.to_csv(os.path.join(output_dir, 'cross_validation_summary.csv'), index=False)
    
    # Plot cross-validation results
    plt.figure(figsize=(10, 6))
    x = np.arange(len(cv_summary['Method']))
    width = 0.35
    
    plt.bar(x - width/2, cv_summary['PICP'], width, label='PICP')
    plt.bar(x + width/2, cv_summary['MPIW'], width, label='MPIW')
    
    plt.xlabel('Method')
    plt.ylabel('Value')
    plt.title('Cross-Validation Results: PICP and MPIW')
    plt.xticks(x, cv_summary['Method'], rotation=45, ha='right')
    plt.legend()
    plt.tight_layout()
    plt.grid(True, axis='y')
    plt.savefig(os.path.join(output_dir, 'cross_validation_results.png'))
    
    print(f"Analysis complete. Results saved to {output_dir}")


if __name__ == "__main__":
    # Run the improved analysis
    run_improved_analysis()