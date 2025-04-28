#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Analyze and Compare Model Results

This script analyzes and compares the results of the standard AlphaPeptDeep model
and the enhanced model with improved PTM representation.
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_absolute_error, r2_score
import torch

from peptdeep.pretrained_models import ModelManager
from peptdeep.model.enhanced_model import EnhancedModelManager


def load_models(standard_model_dir, enhanced_model_dir):
    """
    Load trained models
    
    Parameters
    ----------
    standard_model_dir : str
        Directory with standard model files
    enhanced_model_dir : str
        Directory with enhanced model files
        
    Returns
    -------
    tuple
        (standard_model_mgr, enhanced_model_mgr)
    """
    print("Loading models...")
    
    # Load standard model
    standard_model_mgr = ModelManager(mask_modloss=False, device='cuda' if torch.cuda.is_available() else 'cpu')
    standard_model_mgr.rt_model.load_state_dict(torch.load(os.path.join(standard_model_dir, 'rt.pth')))
    standard_model_mgr.ms2_model.load_state_dict(torch.load(os.path.join(standard_model_dir, 'ms2.pth')))
    
    # Load enhanced model
    enhanced_model_mgr = EnhancedModelManager(mask_modloss=False, device='cuda' if torch.cuda.is_available() else 'cpu')
    enhanced_model_mgr.load_models(enhanced_model_dir)
    
    return standard_model_mgr, enhanced_model_mgr


def load_test_data(file_path, sample_size=None):
    """
    Load test data
    
    Parameters
    ----------
    file_path : str
        Path to the dataset file
    sample_size : int, optional
        Number of samples to use
        
    Returns
    -------
    pd.DataFrame
        DataFrame with peptide data
    """
    print(f"Loading test data from {file_path}...")
    df = pd.read_csv(file_path, sep='\t')
    
    # Sample if requested
    if sample_size is not None and sample_size < len(df):
        df = df.sample(sample_size, random_state=42)
    
    # Prepare data for AlphaPeptDeep
    peptide_df = pd.DataFrame({
        'sequence': df['sequence'],
        'mods': df['mods'],
        'mod_sites': df['mod_sites'] if 'mod_sites' in df.columns else [''] * len(df),
        'charge': df['charge'],
        'rt': df['rt'],
        'nce': df['nce'] if 'nce' in df.columns else [30] * len(df),
        'instrument': df['instrument'] if 'instrument' in df.columns else ['QE'] * len(df)
    })
    
    # Add nAA column (peptide length)
    peptide_df['nAA'] = peptide_df.sequence.str.len()
    
    return peptide_df


def analyze_rt_predictions(test_df, standard_model_mgr, enhanced_model_mgr, output_dir):
    """
    Analyze RT predictions
    
    Parameters
    ----------
    test_df : pd.DataFrame
        Test data
    standard_model_mgr : ModelManager
        Standard model manager
    enhanced_model_mgr : EnhancedModelManager
        Enhanced model manager
    output_dir : str
        Directory to save output files
    """
    print("Analyzing RT predictions...")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Predict with standard model
    standard_rt_df = standard_model_mgr.rt_model.predict(test_df)
    
    # Predict with enhanced model
    enhanced_rt_df = enhanced_model_mgr.predict_rt(test_df)
    
    # Calculate metrics
    standard_mae = mean_absolute_error(standard_rt_df['rt'], standard_rt_df['rt_pred'])
    enhanced_mae = mean_absolute_error(enhanced_rt_df['rt'], enhanced_rt_df['rt_pred'])
    
    standard_r2 = r2_score(standard_rt_df['rt'], standard_rt_df['rt_pred'])
    enhanced_r2 = r2_score(enhanced_rt_df['rt'], enhanced_rt_df['rt_pred'])
    
    # Print metrics
    print(f"Standard Model RT MAE: {standard_mae:.4f}")
    print(f"Enhanced Model RT MAE: {enhanced_mae:.4f}")
    print(f"Standard Model RT R²: {standard_r2:.4f}")
    print(f"Enhanced Model RT R²: {enhanced_r2:.4f}")
    
    # Calculate improvement percentages
    mae_improvement = (standard_mae - enhanced_mae) / standard_mae * 100
    r2_improvement = (enhanced_r2 - standard_r2) / standard_r2 * 100
    
    print(f"RT MAE Improvement: {mae_improvement:.2f}%")
    print(f"RT R² Improvement: {r2_improvement:.2f}%")
    
    # Create a combined DataFrame for plotting
    plot_df = pd.DataFrame({
        'True RT': standard_rt_df['rt'],
        'Standard Predicted RT': standard_rt_df['rt_pred'],
        'Enhanced Predicted RT': enhanced_rt_df['rt_pred']
    })
    
    # Plot RT predictions
    plt.figure(figsize=(12, 10))
    
    # Standard model
    plt.subplot(2, 2, 1)
    plt.scatter(plot_df['True RT'], plot_df['Standard Predicted RT'], alpha=0.5)
    plt.plot([plot_df['True RT'].min(), plot_df['True RT'].max()], 
             [plot_df['True RT'].min(), plot_df['True RT'].max()], 'r--')
    plt.title(f'Standard Model RT Predictions (R² = {standard_r2:.4f})')
    plt.xlabel('True RT')
    plt.ylabel('Predicted RT')
    
    # Enhanced model
    plt.subplot(2, 2, 2)
    plt.scatter(plot_df['True RT'], plot_df['Enhanced Predicted RT'], alpha=0.5)
    plt.plot([plot_df['True RT'].min(), plot_df['True RT'].max()], 
             [plot_df['True RT'].min(), plot_df['True RT'].max()], 'r--')
    plt.title(f'Enhanced Model RT Predictions (R² = {enhanced_r2:.4f})')
    plt.xlabel('True RT')
    plt.ylabel('Predicted RT')
    
    # Error distribution - Standard model
    plt.subplot(2, 2, 3)
    errors = plot_df['True RT'] - plot_df['Standard Predicted RT']
    plt.hist(errors, bins=50, alpha=0.7)
    plt.axvline(x=0, color='r', linestyle='--')
    plt.title(f'Standard Model Error Distribution (MAE = {standard_mae:.4f})')
    plt.xlabel('Error (True - Predicted)')
    plt.ylabel('Frequency')
    
    # Error distribution - Enhanced model
    plt.subplot(2, 2, 4)
    errors = plot_df['True RT'] - plot_df['Enhanced Predicted RT']
    plt.hist(errors, bins=50, alpha=0.7)
    plt.axvline(x=0, color='r', linestyle='--')
    plt.title(f'Enhanced Model Error Distribution (MAE = {enhanced_mae:.4f})')
    plt.xlabel('Error (True - Predicted)')
    plt.ylabel('Frequency')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'rt_predictions_comparison.png'))
    
    # Analyze RT predictions by peptide length
    plt.figure(figsize=(12, 6))
    
    # Add peptide length to the DataFrame
    plot_df['Peptide Length'] = test_df['nAA']
    
    # Calculate errors
    plot_df['Standard Error'] = np.abs(plot_df['True RT'] - plot_df['Standard Predicted RT'])
    plot_df['Enhanced Error'] = np.abs(plot_df['True RT'] - plot_df['Enhanced Predicted RT'])
    
    # Group by peptide length
    length_groups = plot_df.groupby('Peptide Length').agg({
        'Standard Error': 'mean',
        'Enhanced Error': 'mean'
    }).reset_index()
    
    # Plot errors by peptide length
    plt.plot(length_groups['Peptide Length'], length_groups['Standard Error'], 'o-', label='Standard Model')
    plt.plot(length_groups['Peptide Length'], length_groups['Enhanced Error'], 'o-', label='Enhanced Model')
    plt.xlabel('Peptide Length')
    plt.ylabel('Mean Absolute Error')
    plt.title('RT Prediction Error by Peptide Length')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(output_dir, 'rt_error_by_length.png'))
    
    # Analyze RT predictions by modification count
    plt.figure(figsize=(12, 6))
    
    # Add modification count to the DataFrame
    plot_df['Modification Count'] = test_df['mods'].apply(
        lambda x: 0 if pd.isna(x) or x == '' else len(x.split(';'))
    )
    
    # Group by modification count
    mod_groups = plot_df.groupby('Modification Count').agg({
        'Standard Error': 'mean',
        'Enhanced Error': 'mean'
    }).reset_index()
    
    # Plot errors by modification count
    plt.plot(mod_groups['Modification Count'], mod_groups['Standard Error'], 'o-', label='Standard Model')
    plt.plot(mod_groups['Modification Count'], mod_groups['Enhanced Error'], 'o-', label='Enhanced Model')
    plt.xlabel('Modification Count')
    plt.ylabel('Mean Absolute Error')
    plt.title('RT Prediction Error by Modification Count')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(output_dir, 'rt_error_by_mod_count.png'))
    
    # Save results
    results = {
        'standard_mae': standard_mae,
        'enhanced_mae': enhanced_mae,
        'standard_r2': standard_r2,
        'enhanced_r2': enhanced_r2,
        'mae_improvement': mae_improvement,
        'r2_improvement': r2_improvement
    }
    
    pd.DataFrame([results]).to_csv(os.path.join(output_dir, 'rt_metrics.csv'), index=False)
    
    return results


def analyze_ms2_predictions(test_df, standard_model_mgr, enhanced_model_mgr, output_dir):
    """
    Analyze MS2 predictions
    
    Parameters
    ----------
    test_df : pd.DataFrame
        Test data
    standard_model_mgr : ModelManager
        Standard model manager
    enhanced_model_mgr : EnhancedModelManager
        Enhanced model manager
    output_dir : str
        Directory to save output files
    """
    print("Analyzing MS2 predictions...")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Predict with standard model
    standard_ms2_results = standard_model_mgr.ms2_model.predict(test_df)
    
    # Predict with enhanced model
    enhanced_ms2_results = enhanced_model_mgr.predict_ms2(test_df)
    
    # Extract fragment intensities
    standard_intensities = standard_ms2_results['fragment_intensity_df'].values
    enhanced_intensities = enhanced_ms2_results['fragment_intensity_df'].values
    
    # Calculate cosine similarity between the two models' predictions
    # Normalize intensities
    standard_intensities_norm = standard_intensities / np.linalg.norm(standard_intensities, axis=1, keepdims=True)
    enhanced_intensities_norm = enhanced_intensities / np.linalg.norm(enhanced_intensities, axis=1, keepdims=True)
    
    # Calculate cosine similarity
    cosine_sim = np.sum(standard_intensities_norm * enhanced_intensities_norm, axis=1)
    avg_cosine_sim = np.mean(cosine_sim)
    
    print(f"Average Cosine Similarity between MS2 predictions: {avg_cosine_sim:.4f}")
    
    # Plot cosine similarity distribution
    plt.figure(figsize=(10, 6))
    plt.hist(cosine_sim, bins=50, alpha=0.7)
    plt.axvline(x=avg_cosine_sim, color='r', linestyle='--', 
                label=f'Average: {avg_cosine_sim:.4f}')
    plt.title('Cosine Similarity Distribution between Standard and Enhanced MS2 Predictions')
    plt.xlabel('Cosine Similarity')
    plt.ylabel('Frequency')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(output_dir, 'ms2_cosine_similarity.png'))
    
    # Analyze MS2 predictions by peptide length
    plt.figure(figsize=(12, 6))
    
    # Create DataFrame with peptide lengths and cosine similarities
    ms2_df = pd.DataFrame({
        'Peptide Length': test_df['nAA'],
        'Cosine Similarity': cosine_sim
    })
    
    # Group by peptide length
    length_groups = ms2_df.groupby('Peptide Length').agg({
        'Cosine Similarity': 'mean'
    }).reset_index()
    
    # Plot cosine similarity by peptide length
    plt.plot(length_groups['Peptide Length'], length_groups['Cosine Similarity'], 'o-')
    plt.xlabel('Peptide Length')
    plt.ylabel('Mean Cosine Similarity')
    plt.title('MS2 Prediction Similarity by Peptide Length')
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(output_dir, 'ms2_similarity_by_length.png'))
    
    # Analyze MS2 predictions by modification count
    plt.figure(figsize=(12, 6))
    
    # Add modification count to the DataFrame
    ms2_df['Modification Count'] = test_df['mods'].apply(
        lambda x: 0 if pd.isna(x) or x == '' else len(x.split(';'))
    )
    
    # Group by modification count
    mod_groups = ms2_df.groupby('Modification Count').agg({
        'Cosine Similarity': 'mean'
    }).reset_index()
    
    # Plot cosine similarity by modification count
    plt.plot(mod_groups['Modification Count'], mod_groups['Cosine Similarity'], 'o-')
    plt.xlabel('Modification Count')
    plt.ylabel('Mean Cosine Similarity')
    plt.title('MS2 Prediction Similarity by Modification Count')
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(output_dir, 'ms2_similarity_by_mod_count.png'))
    
    # Save results
    results = {
        'avg_cosine_sim': avg_cosine_sim
    }
    
    pd.DataFrame([results]).to_csv(os.path.join(output_dir, 'ms2_metrics.csv'), index=False)
    
    # Visualize example spectra
    visualize_example_spectra(test_df, standard_ms2_results, enhanced_ms2_results, output_dir)
    
    return results


def visualize_example_spectra(test_df, standard_ms2_results, enhanced_ms2_results, output_dir):
    """
    Visualize example MS2 spectra
    
    Parameters
    ----------
    test_df : pd.DataFrame
        Test data
    standard_ms2_results : dict
        Standard model MS2 results
    enhanced_ms2_results : dict
        Enhanced model MS2 results
    output_dir : str
        Directory to save output files
    """
    print("Visualizing example MS2 spectra...")
    
    # Create output directory for spectra
    spectra_dir = os.path.join(output_dir, 'example_spectra')
    os.makedirs(spectra_dir, exist_ok=True)
    
    # Find peptides with modifications
    modified_peptides = test_df[test_df['mods'].notna() & (test_df['mods'] != '')].index.tolist()
    
    # If no modified peptides, use any peptides
    if not modified_peptides:
        modified_peptides = test_df.index.tolist()[:5]
    else:
        # Limit to 5 examples
        modified_peptides = modified_peptides[:5]
    
    # Visualize spectra for each peptide
    for idx in modified_peptides:
        peptide = test_df.iloc[idx]['sequence']
        mods = test_df.iloc[idx]['mods'] if pd.notna(test_df.iloc[idx]['mods']) else ''
        
        print(f"Visualizing spectrum for peptide: {peptide} (Mods: {mods})")
        
        # Get fragment indices
        start_idx = test_df.iloc[idx]['frag_start_idx'] if 'frag_start_idx' in test_df.columns else None
        end_idx = test_df.iloc[idx]['frag_stop_idx'] if 'frag_stop_idx' in test_df.columns else None
        
        if start_idx is None or end_idx is None:
            print(f"  Skipping peptide {peptide} - fragment indices not available")
            continue
        
        # Get fragment m/z and intensities
        frag_mz = standard_ms2_results['fragment_mz_df'].iloc[start_idx:end_idx]
        standard_intensities = standard_ms2_results['fragment_intensity_df'].iloc[start_idx:end_idx]
        enhanced_intensities = enhanced_ms2_results['fragment_intensity_df'].iloc[start_idx:end_idx]
        
        # Plot spectra
        plt.figure(figsize=(12, 8))
        
        # Standard model spectrum
        plt.subplot(2, 1, 1)
        for col in standard_intensities.columns:
            mz_values = frag_mz[col].values
            intensity_values = standard_intensities[col].values
            
            # Filter out zero m/z values
            mask = mz_values > 0
            mz_values = mz_values[mask]
            intensity_values = intensity_values[mask]
            
            if len(mz_values) > 0:
                plt.stem(mz_values, intensity_values, markerfmt=' ', basefmt=' ', 
                         linefmt='-' if col.startswith('b_') else '--')
        
        plt.title(f'Standard Model MS2 Spectrum for {peptide} {mods}')
        plt.xlabel('m/z')
        plt.ylabel('Relative Intensity')
        plt.ylim(0, 1)
        
        # Enhanced model spectrum
        plt.subplot(2, 1, 2)
        for col in enhanced_intensities.columns:
            mz_values = frag_mz[col].values
            intensity_values = enhanced_intensities[col].values
            
            # Filter out zero m/z values
            mask = mz_values > 0
            mz_values = mz_values[mask]
            intensity_values = intensity_values[mask]
            
            if len(mz_values) > 0:
                plt.stem(mz_values, intensity_values, markerfmt=' ', basefmt=' ', 
                         linefmt='-' if col.startswith('b_') else '--')
        
        plt.title(f'Enhanced Model MS2 Spectrum for {peptide} {mods}')
        plt.xlabel('m/z')
        plt.ylabel('Relative Intensity')
        plt.ylim(0, 1)
        
        plt.tight_layout()
        plt.savefig(os.path.join(spectra_dir, f'ms2_spectrum_{peptide}.png'))
        plt.close()


def main():
    """Main function"""
    # Set random seed for reproducibility
    np.random.seed(42)
    torch.manual_seed(42)
    
    # Parameters
    data_file = 'HLA_DDA_Rescore/AD_pFind_MSV000084172_fdr.tsv'
    standard_model_dir = 'models/standard'
    enhanced_model_dir = 'models/enhanced'
    output_dir = 'analysis_results'
    sample_size = 1000  # Set to None to use all data
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Load models
    standard_model_mgr, enhanced_model_mgr = load_models(standard_model_dir, enhanced_model_dir)
    
    # Load test data
    test_df = load_test_data(data_file, sample_size)
    
    # Analyze RT predictions
    rt_results = analyze_rt_predictions(
        test_df, standard_model_mgr, enhanced_model_mgr, 
        os.path.join(output_dir, 'rt')
    )
    
    # Analyze MS2 predictions
    ms2_results = analyze_ms2_predictions(
        test_df, standard_model_mgr, enhanced_model_mgr, 
        os.path.join(output_dir, 'ms2')
    )
    
    # Combine results
    combined_results = {**rt_results, **ms2_results}
    pd.DataFrame([combined_results]).to_csv(os.path.join(output_dir, 'combined_metrics.csv'), index=False)
    
    print("\nAnalysis complete!")
    print(f"Results saved to {output_dir}")


if __name__ == "__main__":
    main()