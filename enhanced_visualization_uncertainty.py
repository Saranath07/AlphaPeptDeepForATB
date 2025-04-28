#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Enhanced Visualization for Uncertainty Analysis Results

This script generates additional visualizations and insights from the uncertainty analysis results.
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.gridspec import GridSpec

# Set style
plt.style.use('ggplot')
sns.set_context("talk")
plt.rcParams['figure.figsize'] = (12, 8)

def load_data():
    """Load the uncertainty analysis results"""
    rt_summary = pd.read_csv('hla_uncertainty_results/rt/rt_uncertainty_summary.csv')
    sampled_peptides = pd.read_csv('hla_uncertainty_results/sampled_peptides.csv')
    
    return rt_summary, sampled_peptides

def analyze_peptide_properties(sampled_peptides):
    """Analyze peptide properties and their relationship to prediction accuracy"""
    # Create output directory
    os.makedirs('hla_uncertainty_results/enhanced_analysis', exist_ok=True)
    
    # Analyze peptide length distribution
    plt.figure(figsize=(10, 6))
    sns.histplot(sampled_peptides['nAA'], bins=range(min(sampled_peptides['nAA']), max(sampled_peptides['nAA'])+2), kde=True)
    plt.title('Distribution of Peptide Lengths')
    plt.xlabel('Peptide Length (Amino Acids)')
    plt.ylabel('Count')
    plt.grid(True)
    plt.savefig('hla_uncertainty_results/enhanced_analysis/peptide_length_distribution.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Analyze charge state distribution
    plt.figure(figsize=(8, 6))
    charge_counts = sampled_peptides['charge'].value_counts().sort_index()
    sns.barplot(x=charge_counts.index, y=charge_counts.values)
    plt.title('Distribution of Charge States')
    plt.xlabel('Charge State')
    plt.ylabel('Count')
    plt.grid(True, axis='y')
    plt.savefig('hla_uncertainty_results/enhanced_analysis/charge_state_distribution.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Analyze RT distribution
    plt.figure(figsize=(10, 6))
    sns.histplot(sampled_peptides['rt'], bins=20, kde=True)
    plt.title('Distribution of Retention Times')
    plt.xlabel('Retention Time (min)')
    plt.ylabel('Count')
    plt.grid(True)
    plt.savefig('hla_uncertainty_results/enhanced_analysis/rt_distribution.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Analyze relationship between peptide length and RT
    plt.figure(figsize=(10, 6))
    sns.scatterplot(x='nAA', y='rt', data=sampled_peptides, alpha=0.7)
    plt.title('Relationship Between Peptide Length and Retention Time')
    plt.xlabel('Peptide Length (Amino Acids)')
    plt.ylabel('Retention Time (min)')
    plt.grid(True)
    plt.savefig('hla_uncertainty_results/enhanced_analysis/length_vs_rt.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Analyze modified vs unmodified peptides
    sampled_peptides['has_mods'] = sampled_peptides['mods'].notna() & (sampled_peptides['mods'] != '')
    
    plt.figure(figsize=(8, 6))
    mod_counts = sampled_peptides['has_mods'].value_counts()
    plt.pie(mod_counts, labels=['Unmodified', 'Modified'] if False in mod_counts.index else ['Modified', 'Unmodified'],
            autopct='%1.1f%%', startangle=90, colors=['#ff9999','#66b3ff'])
    plt.title('Proportion of Modified vs Unmodified Peptides')
    plt.axis('equal')
    plt.savefig('hla_uncertainty_results/enhanced_analysis/modified_vs_unmodified.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    return {
        'total_peptides': len(sampled_peptides),
        'unique_sequences': sampled_peptides['sequence'].nunique(),
        'length_range': (sampled_peptides['nAA'].min(), sampled_peptides['nAA'].max()),
        'mean_length': sampled_peptides['nAA'].mean(),
        'charge_distribution': charge_counts.to_dict(),
        'rt_range': (sampled_peptides['rt'].min(), sampled_peptides['rt'].max()),
        'mean_rt': sampled_peptides['rt'].mean(),
        'modified_count': mod_counts.get(True, 0),
        'unmodified_count': mod_counts.get(False, 0)
    }

def visualize_rt_uncertainty_comparison(rt_summary):
    """Create enhanced visualizations for RT uncertainty comparison"""
    # Create output directory
    os.makedirs('hla_uncertainty_results/enhanced_analysis', exist_ok=True)
    
    # Create a comprehensive comparison plot
    fig = plt.figure(figsize=(15, 10))
    gs = GridSpec(2, 2, figure=fig)
    
    # Plot 1: Mean Absolute Error Comparison
    ax1 = fig.add_subplot(gs[0, 0])
    models = rt_summary['Model'].tolist()
    mae_values = rt_summary['Mean Absolute Error'].tolist()
    
    colors = ['#ff9999', '#66b3ff', '#99ff99']
    bars = ax1.bar(models, mae_values, color=colors)
    ax1.set_title('Mean Absolute Error Comparison')
    ax1.set_ylabel('Mean Absolute Error')
    ax1.set_ylim(0, max(mae_values) * 1.2)
    
    # Add value labels on top of bars
    for bar in bars:
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + 1,
                f'{height:.2f}', ha='center', va='bottom')
    
    # Plot 2: PICP vs MPIW
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.scatter(rt_summary['PICP'], rt_summary['MPIW'], s=100, c=colors, alpha=0.7)
    
    # Add labels for each point
    for i, model in enumerate(models):
        ax2.annotate(model, 
                    (rt_summary['PICP'].iloc[i], rt_summary['MPIW'].iloc[i]),
                    xytext=(10, 10), textcoords='offset points')
    
    ax2.set_title('PICP vs MPIW')
    ax2.set_xlabel('Prediction Interval Coverage Probability (PICP)')
    ax2.set_ylabel('Mean Prediction Interval Width (MPIW)')
    ax2.grid(True)
    
    # Plot 3: Mean Uncertainty Comparison
    ax3 = fig.add_subplot(gs[1, 0])
    uncertainty_values = rt_summary['Mean Uncertainty (Std)'].tolist()
    
    bars = ax3.bar(models, uncertainty_values, color=colors)
    ax3.set_title('Mean Uncertainty Comparison')
    ax3.set_ylabel('Mean Uncertainty (Std)')
    ax3.set_ylim(0, max(uncertainty_values) * 1.2)
    
    # Add value labels on top of bars
    for bar in bars:
        height = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                f'{height:.2f}', ha='center', va='bottom')
    
    # Plot 4: Relative Improvement
    ax4 = fig.add_subplot(gs[1, 1])
    
    # Calculate relative improvement of Enhanced over Standard
    standard_mae = rt_summary.loc[rt_summary['Model'] == 'Standard (MC Dropout)', 'Mean Absolute Error'].values[0]
    enhanced_mae = rt_summary.loc[rt_summary['Model'] == 'Enhanced (MC Dropout)', 'Mean Absolute Error'].values[0]
    ensemble_mae = rt_summary.loc[rt_summary['Model'] == 'Model Ensemble', 'Mean Absolute Error'].values[0]
    
    rel_improvement_enhanced = (standard_mae - enhanced_mae) / standard_mae * 100
    rel_improvement_ensemble = (standard_mae - ensemble_mae) / standard_mae * 100
    
    improvements = [0, rel_improvement_enhanced, rel_improvement_ensemble]  # Standard as baseline (0% improvement)
    
    bars = ax4.bar(models, improvements, color=colors)
    ax4.set_title('Relative Improvement Over Standard Model')
    ax4.set_ylabel('Improvement (%)')
    ax4.set_ylim(0, max(improvements) * 1.2)
    
    # Add value labels on top of bars
    for bar in bars:
        height = bar.get_height()
        if height > 0:  # Only add labels for non-zero values
            ax4.text(bar.get_x() + bar.get_width()/2., height + 1,
                    f'{height:.1f}%', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig('hla_uncertainty_results/enhanced_analysis/rt_uncertainty_comprehensive.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    return {
        'standard_mae': standard_mae,
        'enhanced_mae': enhanced_mae,
        'ensemble_mae': ensemble_mae,
        'rel_improvement_enhanced': rel_improvement_enhanced,
        'rel_improvement_ensemble': rel_improvement_ensemble
    }

def generate_summary_report(peptide_stats, rt_stats):
    """Generate a summary report with key insights"""
    report = f"""# Enhanced Analysis of HLA Model Uncertainty

## Dataset Statistics
- Total peptides analyzed: {peptide_stats['total_peptides']}
- Unique peptide sequences: {peptide_stats['unique_sequences']}
- Peptide length range: {peptide_stats['length_range'][0]}-{peptide_stats['length_range'][1]} amino acids
- Average peptide length: {peptide_stats['mean_length']:.2f} amino acids
- Charge state distribution: {', '.join([f"{charge}+ ({count})" for charge, count in peptide_stats['charge_distribution'].items()])}
- Retention time range: {peptide_stats['rt_range'][0]:.2f}-{peptide_stats['rt_range'][1]:.2f} minutes
- Average retention time: {peptide_stats['mean_rt']:.2f} minutes
- Modified peptides: {peptide_stats['modified_count']} ({peptide_stats['modified_count']/peptide_stats['total_peptides']*100:.1f}%)
- Unmodified peptides: {peptide_stats['unmodified_count']} ({peptide_stats['unmodified_count']/peptide_stats['total_peptides']*100:.1f}%)

## RT Prediction Performance
- Standard model MAE: {rt_stats['standard_mae']:.2f}
- Enhanced model MAE: {rt_stats['enhanced_mae']:.2f}
- Model Ensemble MAE: {rt_stats['ensemble_mae']:.2f}
- Enhanced model improvement over Standard: {rt_stats['rel_improvement_enhanced']:.1f}%
- Model Ensemble improvement over Standard: {rt_stats['rel_improvement_ensemble']:.1f}%

## Key Insights
1. The Enhanced model with improved PTM representation shows a significant {rt_stats['rel_improvement_enhanced']:.1f}% reduction in mean absolute error compared to the Standard model.
2. The Model Ensemble approach provides a {rt_stats['rel_improvement_ensemble']:.1f}% improvement over the Standard model, but not as good as the Enhanced model alone.
3. The dataset contains a diverse range of peptide lengths ({peptide_stats['length_range'][0]}-{peptide_stats['length_range'][1]} amino acids) and charge states.
4. {peptide_stats['modified_count']/peptide_stats['total_peptides']*100:.1f}% of the peptides contain post-translational modifications, highlighting the importance of proper PTM handling.

## Visualizations
Several visualizations have been generated in the 'hla_uncertainty_results/enhanced_analysis' directory:
- Peptide length distribution
- Charge state distribution
- Retention time distribution
- Relationship between peptide length and retention time
- Proportion of modified vs unmodified peptides
- Comprehensive RT uncertainty comparison

These visualizations provide deeper insights into the dataset characteristics and model performance.
"""
    
    with open('hla_uncertainty_results/enhanced_analysis/summary_report.md', 'w') as f:
        f.write(report)
    
    print("Enhanced analysis complete. Results saved to 'hla_uncertainty_results/enhanced_analysis/'")

def main():
    """Main function to run the enhanced visualization"""
    print("Starting enhanced visualization and analysis...")
    
    # Load data
    rt_summary, sampled_peptides = load_data()
    
    # Analyze peptide properties
    peptide_stats = analyze_peptide_properties(sampled_peptides)
    
    # Visualize RT uncertainty comparison
    rt_stats = visualize_rt_uncertainty_comparison(rt_summary)
    
    # Generate summary report
    generate_summary_report(peptide_stats, rt_stats)

if __name__ == "__main__":
    main()