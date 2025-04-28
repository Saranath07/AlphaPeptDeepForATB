# Enhanced PTM Representation for AlphaPeptDeep

This project implements an enhanced version of the AlphaPeptDeep model with improved PTM representation using chemical properties and attention mechanisms. The enhanced model is designed to improve peptide property prediction, particularly for modified peptides.

## Overview

The project consists of the following components:

1. **Enhanced PTM Embedding**: A new embedding layer that incorporates chemical properties of PTMs and uses an attention mechanism to weight these features contextually.
2. **Enhanced Models**: Modified versions of the AlphaPeptDeep RT and MS2 models that use the enhanced PTM embedding.
3. **Training Scripts**: Scripts to train both the standard and enhanced models on the HLA_DDA_Rescore dataset.
4. **Analysis Scripts**: Scripts to analyze and compare the performance of the two models.

## Files

- `peptdeep/model/enhanced_ptm_embedding.py`: Implementation of the enhanced PTM embedding layer and PTM feature extractor.
- `peptdeep/model/enhanced_model.py`: Implementation of the enhanced RT and MS2 models.
- `train_hla_models.py`: Script to train both models on the HLA_DDA_Rescore dataset.
- `analyze_model_results.py`: Script to analyze and compare the performance of the two models.

## Requirements

- Python 3.7+
- PyTorch 1.7+
- pandas
- numpy
- matplotlib
- seaborn
- scikit-learn
- tqdm
- AlphaPeptDeep

## Installation

1. Make sure you have AlphaPeptDeep installed:
   ```
   pip install peptdeep
   ```

2. Clone this repository:
   ```
   git clone https://github.com/yourusername/alphapeptdeep-enhanced-ptm.git
   cd alphapeptdeep-enhanced-ptm
   ```

## Usage

### Training the Models

To train both the standard and enhanced models on the HLA_DDA_Rescore dataset:

```bash
python train_hla_models.py
```

This script will:
1. Load the HLA_DDA_Rescore dataset
2. Split it into training, validation, and test sets
3. Train the standard AlphaPeptDeep model
4. Train the enhanced model with improved PTM representation
5. Evaluate both models on the test set
6. Save the trained models and evaluation metrics

### Analyzing the Results

To analyze and compare the performance of the two models:

```bash
python analyze_model_results.py
```

This script will:
1. Load the trained models
2. Load the test data
3. Generate predictions with both models
4. Calculate evaluation metrics
5. Create visualizations comparing the two models
6. Save the results to the `analysis_results` directory

## Enhanced PTM Representation

The enhanced PTM representation includes the following features:

1. **Chemical Properties**:
   - Molecular weight
   - Hydrophobicity
   - Polarity
   - Charge
   - Size

2. **Attention Mechanism**:
   - Contextually weights the chemical features
   - Combines them with the standard PTM embedding

## Expected Improvements

The enhanced PTM representation is expected to improve:

1. **RT Prediction**:
   - Better accuracy for modified peptides
   - More consistent performance across different modification types

2. **MS2 Prediction**:
   - More accurate fragment ion intensity predictions
   - Better handling of modification-specific fragmentation patterns

## Customization

You can customize the training process by modifying the parameters in the `train_hla_models.py` script:

- `sample_size`: Number of samples to use (set to `None` to use all data)
- `test_size`: Fraction of data to use for testing
- `val_size`: Fraction of training data to use for validation
- `epochs`: Number of training epochs
- `batch_size`: Batch size for training

## Results

After running the analysis script, you can find the results in the `analysis_results` directory:

- `rt/`: RT prediction analysis
  - `rt_predictions_comparison.png`: Scatter plots and error distributions
  - `rt_error_by_length.png`: Error by peptide length
  - `rt_error_by_mod_count.png`: Error by modification count
  - `rt_metrics.csv`: Evaluation metrics

- `ms2/`: MS2 prediction analysis
  - `ms2_cosine_similarity.png`: Cosine similarity distribution
  - `ms2_similarity_by_length.png`: Similarity by peptide length
  - `ms2_similarity_by_mod_count.png`: Similarity by modification count
  - `ms2_metrics.csv`: Evaluation metrics
  - `example_spectra/`: Example MS2 spectra visualizations

- `combined_metrics.csv`: Combined evaluation metrics

## References

1. AlphaPeptDeep: https://github.com/MannLabs/alphapeptdeep
2. HLA_DDA_Rescore dataset: MSV000084172