# Enhanced Analysis of HLA Model Uncertainty

## Dataset Statistics
- Total peptides analyzed: 100
- Unique peptide sequences: 100
- Peptide length range: 8-29 amino acids
- Average peptide length: 9.73 amino acids
- Charge state distribution: 1+ (6), 2+ (76), 3+ (15), 4+ (3)
- Retention time range: 9.51-99.12 minutes
- Average retention time: 42.37 minutes
- Modified peptides: 15 (15.0%)
- Unmodified peptides: 85 (85.0%)

## RT Prediction Performance
- Standard model MAE: 36.26
- Enhanced model MAE: 20.62
- Model Ensemble MAE: 28.44
- Enhanced model improvement over Standard: 43.1%
- Model Ensemble improvement over Standard: 21.6%

## Key Insights
1. The Enhanced model with improved PTM representation shows a significant 43.1% reduction in mean absolute error compared to the Standard model.
2. The Model Ensemble approach provides a 21.6% improvement over the Standard model, but not as good as the Enhanced model alone.
3. The dataset contains a diverse range of peptide lengths (8-29 amino acids) and charge states.
4. 15.0% of the peptides contain post-translational modifications, highlighting the importance of proper PTM handling.

## Visualizations
Several visualizations have been generated in the 'hla_uncertainty_results/enhanced_analysis' directory:
- Peptide length distribution
- Charge state distribution
- Retention time distribution
- Relationship between peptide length and retention time
- Proportion of modified vs unmodified peptides
- Comprehensive RT uncertainty comparison

These visualizations provide deeper insights into the dataset characteristics and model performance.
