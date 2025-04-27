# Uncertainty Quantification Analysis for Modification-Specific Peptides

## Overview

This report presents the results of uncertainty quantification analysis performed on a subset of unmodified peptides from the `modificationSpecificPeptides.txt` dataset. The analysis employed two methods for uncertainty quantification:

1. **Simulated Monte Carlo Dropout**: Adding noise to predictions to simulate the effect of dropout
2. **Deep Ensembles**: Using multiple pre-trained models (generic, phospho, digly) to generate predictions

## Dataset

- Original dataset: 49 peptides from `modificationSpecificPeptides.txt`
- Filtered dataset: 47 unmodified peptides
- Analysis performed on retention time (RT) prediction

## Key Findings

### Performance Metrics

| Method | PICP | MPIW | Mean Absolute Error | Mean Relative Error |
|--------|------|------|---------------------|---------------------|
| MC Dropout | 0.0213 | 0.1918 | 32.6804 | inf% |
| Ensemble | 0.0213 | 0.2464 | 32.6590 | inf% |

### Interpretation

1. **Low PICP (Prediction Interval Coverage Probability)**: 
   - Only about 2.1% of true values fall within the predicted confidence intervals
   - Ideally, PICP should be close to 95% for well-calibrated uncertainty estimates
   - This suggests significant under-estimation of uncertainty

2. **Mean Prediction Interval Width (MPIW)**:
   - MC Dropout: 0.1918
   - Ensemble: 0.2464
   - The ensemble method produces wider prediction intervals than MC Dropout

3. **High Mean Absolute Error**:
   - Both methods show high error rates (>32)
   - This indicates poor prediction accuracy on this specific dataset

4. **Infinite Mean Relative Error**:
   - This suggests division by zero issues, likely due to some normalized RT values being very close to zero

## Potential Issues

1. **Domain Mismatch**: 
   - The pre-trained models may have been trained on different types of peptides or experimental conditions
   - The modification-specific peptides might have different RT characteristics than the training data

2. **Normalization Issues**:
   - The RT normalization approach used might not be appropriate for this dataset
   - The infinite relative error suggests problems with the normalization

3. **Uncertainty Calibration**:
   - The very low PICP indicates that the uncertainty estimates are not well-calibrated
   - The models are overconfident in their predictions

## Peptide Property Analysis

The analysis also examined how peptide properties (particularly length) affect prediction accuracy and uncertainty. The results are saved in:

- `mc_dropout_length_impact.csv` and `ensemble_length_impact.csv`: Summarize error and uncertainty by peptide length category
- `length_impact_on_error.png`: Visualizes how peptide length affects prediction error
- `length_impact_on_uncertainty.png`: Visualizes how peptide length affects prediction uncertainty
- `length_correlation_mc_dropout.png`: Shows correlation between peptide length and uncertainty/error

## Recommendations

1. **Model Fine-tuning**:
   - Fine-tune the pre-trained models on a subset of the modification-specific peptides
   - This could improve prediction accuracy for this specific dataset

2. **Alternative Normalization**:
   - Explore different RT normalization approaches that better suit this dataset
   - Consider using the raw RT values directly if they are already on a comparable scale

3. **Uncertainty Calibration**:
   - Apply post-hoc calibration techniques to improve the reliability of uncertainty estimates
   - Temperature scaling or isotonic regression could help calibrate the confidence intervals

4. **Expanded Dataset**:
   - If possible, increase the dataset size for more robust analysis
   - Include more diverse peptides to better understand the model's performance across different peptide properties

5. **Transfer Learning**:
   - Consider transfer learning approaches that adapt the pre-trained models to this specific dataset
   - This could improve both accuracy and uncertainty estimation

## Conclusion

The uncertainty quantification analysis reveals significant challenges in accurately predicting retention times for this specific peptide dataset. Both the MC Dropout and Ensemble methods show high error rates and poorly calibrated uncertainty estimates. Further investigation into the causes of these issues and implementation of the recommended improvements could lead to better performance.

All results and visualizations are saved in the `modification_specific_results` directory for further examination.