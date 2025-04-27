#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Thread D: Multi-Modal Integration with Structural Data

This script implements a multi-modal approach that fuses peptide sequence data with 
structural information to enhance RT and MS2 predictions.
"""

import os
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

from peptdeep.pretrained_models import ModelManager
from peptdeep.model.featurize import get_batch_mod_feature
import peptdeep.model.building_block as building_block
from peptdeep.model.rt import AlphaRTModel, IRT_PEPTIDE_DF
from peptdeep.model.ms2 import pDeepModel
from alphabase.peptide.fragment import get_charged_frag_types, init_fragment_by_precursor_dataframe
from peptdeep.settings import global_settings
from peptdeep.utils import evaluate_linear_regression


# Define structural features for amino acids
# These values are based on common structural properties
structural_features = {
    'A': {'secondary_structure': {'helix': 0.7, 'sheet': 0.3, 'coil': 0.0}, 
          'solvent_accessibility': 0.1, 'flexibility': 0.3},
    'R': {'secondary_structure': {'helix': 0.5, 'sheet': 0.3, 'coil': 0.2}, 
          'solvent_accessibility': 0.9, 'flexibility': 0.5},
    'N': {'secondary_structure': {'helix': 0.3, 'sheet': 0.3, 'coil': 0.4}, 
          'solvent_accessibility': 0.7, 'flexibility': 0.6},
    'D': {'secondary_structure': {'helix': 0.4, 'sheet': 0.2, 'coil': 0.4}, 
          'solvent_accessibility': 0.8, 'flexibility': 0.6},
    'C': {'secondary_structure': {'helix': 0.3, 'sheet': 0.6, 'coil': 0.1}, 
          'solvent_accessibility': 0.1, 'flexibility': 0.2},
    'E': {'secondary_structure': {'helix': 0.6, 'sheet': 0.2, 'coil': 0.2}, 
          'solvent_accessibility': 0.7, 'flexibility': 0.5},
    'Q': {'secondary_structure': {'helix': 0.6, 'sheet': 0.2, 'coil': 0.2}, 
          'solvent_accessibility': 0.6, 'flexibility': 0.5},
    'G': {'secondary_structure': {'helix': 0.1, 'sheet': 0.2, 'coil': 0.7}, 
          'solvent_accessibility': 0.3, 'flexibility': 0.9},
    'H': {'secondary_structure': {'helix': 0.5, 'sheet': 0.3, 'coil': 0.2}, 
          'solvent_accessibility': 0.5, 'flexibility': 0.4},
    'I': {'secondary_structure': {'helix': 0.5, 'sheet': 0.5, 'coil': 0.0}, 
          'solvent_accessibility': 0.1, 'flexibility': 0.2},
    'L': {'secondary_structure': {'helix': 0.6, 'sheet': 0.4, 'coil': 0.0}, 
          'solvent_accessibility': 0.1, 'flexibility': 0.2},
    'K': {'secondary_structure': {'helix': 0.5, 'sheet': 0.3, 'coil': 0.2}, 
          'solvent_accessibility': 0.8, 'flexibility': 0.6},
    'M': {'secondary_structure': {'helix': 0.6, 'sheet': 0.3, 'coil': 0.1}, 
          'solvent_accessibility': 0.2, 'flexibility': 0.3},
    'F': {'secondary_structure': {'helix': 0.4, 'sheet': 0.5, 'coil': 0.1}, 
          'solvent_accessibility': 0.2, 'flexibility': 0.3},
    'P': {'secondary_structure': {'helix': 0.1, 'sheet': 0.1, 'coil': 0.8}, 
          'solvent_accessibility': 0.4, 'flexibility': 0.7},
    'S': {'secondary_structure': {'helix': 0.4, 'sheet': 0.3, 'coil': 0.3}, 
          'solvent_accessibility': 0.5, 'flexibility': 0.5},
    'T': {'secondary_structure': {'helix': 0.3, 'sheet': 0.4, 'coil': 0.3}, 
          'solvent_accessibility': 0.4, 'flexibility': 0.4},
    'W': {'secondary_structure': {'helix': 0.4, 'sheet': 0.4, 'coil': 0.2}, 
          'solvent_accessibility': 0.3, 'flexibility': 0.2},
    'Y': {'secondary_structure': {'helix': 0.4, 'sheet': 0.4, 'coil': 0.2}, 
          'solvent_accessibility': 0.4, 'flexibility': 0.3},
    'V': {'secondary_structure': {'helix': 0.4, 'sheet': 0.6, 'coil': 0.0}, 
          'solvent_accessibility': 0.1, 'flexibility': 0.2}
}


def get_default_structural_features():
    """Get default structural features for unknown amino acids"""
    return {
        'secondary_structure': {'helix': 0.33, 'sheet': 0.33, 'coil': 0.34},
        'solvent_accessibility': 0.5,
        'flexibility': 0.5
    }


def get_structural_features_for_sequence(sequence):
    """Get structural features for a peptide sequence"""
    seq_len = len(sequence)
    # 5 features: helix, sheet, coil, solvent_accessibility, flexibility
    features = np.zeros((seq_len, 5))
    
    for i, aa in enumerate(sequence):
        if aa in structural_features:
            props = structural_features[aa]
        else:
            props = get_default_structural_features()
        
        features[i, 0] = props['secondary_structure']['helix']
        features[i, 1] = props['secondary_structure']['sheet']
        features[i, 2] = props['secondary_structure']['coil']
        features[i, 3] = props['solvent_accessibility']
        features[i, 4] = props['flexibility']
    
    return features


def get_batch_structural_features(batch_df):
    """Get batch structural features"""
    batch_size = len(batch_df)
    max_len = max(batch_df.nAA)
    # 5 features: helix, sheet, coil, solvent_accessibility, flexibility
    # Add 2 to max_len for N-term and C-term padding to match sequence features
    features = np.zeros((batch_size, max_len + 2, 5))
    
    for i, (_, row) in enumerate(batch_df.iterrows()):
        seq_len = row.nAA
        seq_features = get_structural_features_for_sequence(row.sequence)
        # Place the features starting at index 1 (after N-term padding)
        features[i, 1:seq_len+1, :] = seq_features
    
    return torch.tensor(features, dtype=torch.float32)


class StructuralFeaturesEmbedding(torch.nn.Module):
    """Embedding layer for structural features"""
    
    def __init__(self, input_dim=5, hidden_dim=64):
        super().__init__()
        self.embedding = torch.nn.Sequential(
            torch.nn.Linear(input_dim, hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dim, hidden_dim)
        )
    
    def forward(self, x):
        return self.embedding(x)


class MultiModalFusion(torch.nn.Module):
    """Fusion module for sequence and structural features"""
    
    def __init__(self, seq_dim, struct_dim, hidden_dim):
        super().__init__()
        self.seq_proj = torch.nn.Linear(seq_dim, hidden_dim)
        self.struct_proj = torch.nn.Linear(struct_dim, hidden_dim)
        self.fusion = torch.nn.Sequential(
            torch.nn.Linear(hidden_dim * 2, hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dim, hidden_dim)
        )
    
    def forward(self, seq_features, struct_features):
        seq_proj = self.seq_proj(seq_features)
        struct_proj = self.struct_proj(struct_features)
        concat = torch.cat([seq_proj, struct_proj], dim=-1)
        fused = self.fusion(concat)
        return fused


class MultiModalRTModel(torch.nn.Module):
    """Multi-modal RT model that fuses sequence and structural features"""
    
    def __init__(self, dropout=0.1, nlayers=4, hidden=128):
        super().__init__()
        
        self.dropout = torch.nn.Dropout(dropout)
        
        # Sequence embedding
        self.seq_embedding = building_block.AATransformerEncoding(hidden)
        
        # Structural features embedding
        self.struct_embedding = StructuralFeaturesEmbedding(input_dim=5, hidden_dim=hidden)
        
        # Fusion module
        self.fusion = MultiModalFusion(seq_dim=hidden, struct_dim=hidden, hidden_dim=hidden)
        
        # Transformer
        self.hidden_nn = building_block.Hidden_HFace_Transformer(hidden, nlayers=nlayers, dropout=dropout)
        
        # Output layer
        self.output_nn = torch.nn.Sequential(
            building_block.SeqAttentionSum(hidden),
            torch.nn.PReLU(),
            self.dropout,
            torch.nn.Linear(hidden, 1)
        )
    
    def forward(self, aa_indices, mod_x, struct_x):
        # Get sequence features
        seq_features = self.seq_embedding(aa_indices, mod_x)
        
        # Get structural features
        struct_features = self.struct_embedding(struct_x)
        
        # Fuse features
        fused_features = self.fusion(seq_features, struct_features)
        
        # Apply dropout
        x = self.dropout(fused_features)
        
        # Apply transformer
        hidden_x = self.hidden_nn(x)[0]
        x = self.dropout(hidden_x + x * 0.2)
        
        # Output layer
        return self.output_nn(x).squeeze(1)


class MultiModalAlphaRTModel(AlphaRTModel):
    """Multi-modal RT model that extends AlphaRTModel"""
    
    def __init__(self, dropout=0.1, device="cpu"):
        super().__init__(dropout=dropout, device=device)
        
        # Replace the model with our multi-modal model
        self.model = MultiModalRTModel(dropout=dropout, nlayers=4, hidden=128)
        
        # Move model to device
        self.model.to(self.device)
    
    def _get_features_from_batch_df(self, batch_df):
        aa_indices = self._get_26aa_indice_features(batch_df)
        mod_x = self._get_mod_features(batch_df)
        struct_x = get_batch_structural_features(batch_df)
        
        return aa_indices, mod_x, struct_x


def compare_rt_models(output_dir='results'):
    """Compare standard and multi-modal RT models"""
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Use IRT peptides as test dataset
    test_df = IRT_PEPTIDE_DF.copy()
    test_df['rt_norm'] = test_df['irt']  # Use iRT values as normalized RT
    
    # Split into train and test sets
    train_df, test_df = train_test_split(test_df, test_size=0.3, random_state=42)
    
    # Load standard model
    print("Loading standard model...")
    standard_model = AlphaRTModel(device='cpu')
    
    # Create multi-modal model
    print("Creating multi-modal model...")
    multimodal_model = MultiModalAlphaRTModel(device='cpu')
    
    # Train standard model
    print("Training standard model...")
    standard_model.train(train_df, epoch=10, batch_size=32, verbose=True)
    
    # Train multi-modal model
    print("Training multi-modal model...")
    multimodal_model.train(train_df, epoch=10, batch_size=32, verbose=True)
    
    # Predict with standard model
    print("Predicting with standard model...")
    standard_pred = standard_model.predict(test_df)
    
    # Predict with multi-modal model
    print("Predicting with multi-modal model...")
    multimodal_pred = multimodal_model.predict(test_df)
    
    # Evaluate standard model
    standard_eval = evaluate_linear_regression(standard_pred, x='rt_pred', y='rt_norm')
    
    # Evaluate multi-modal model
    multimodal_eval = evaluate_linear_regression(multimodal_pred, x='rt_pred', y='rt_norm')
    
    # Print results
    print("Standard model evaluation:")
    print(standard_eval)
    print("\nMulti-modal model evaluation:")
    print(multimodal_eval)
    
    # Plot results
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.scatter(standard_pred['rt_norm'], standard_pred['rt_pred'], alpha=0.5)
    plt.plot([min(standard_pred['rt_norm']), max(standard_pred['rt_norm'])], 
             [min(standard_pred['rt_norm']), max(standard_pred['rt_norm'])], 'k--')
    plt.xlabel('True RT')
    plt.ylabel('Predicted RT')
    plt.title('Standard Model')
    
    plt.subplot(1, 2, 2)
    plt.scatter(multimodal_pred['rt_norm'], multimodal_pred['rt_pred'], alpha=0.5)
    plt.plot([min(multimodal_pred['rt_norm']), max(multimodal_pred['rt_norm'])], 
             [min(multimodal_pred['rt_norm']), max(multimodal_pred['rt_norm'])], 'k--')
    plt.xlabel('True RT')
    plt.ylabel('Predicted RT')
    plt.title('Multi-modal Model')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'rt_model_comparison.png'))
    
    # Save results to CSV
    standard_pred.to_csv(os.path.join(output_dir, 'standard_rt_predictions.csv'), index=False)
    multimodal_pred.to_csv(os.path.join(output_dir, 'multimodal_rt_predictions.csv'), index=False)
    
    print(f"Results saved to {output_dir}")


if __name__ == "__main__":
    # Create results directory
    output_dir = 'multimodal_results'
    os.makedirs(output_dir, exist_ok=True)
    
    # Compare RT models
    print("Comparing RT models...")
    compare_rt_models(output_dir=os.path.join(output_dir, 'rt'))
    
    print("Multi-modal integration experiment completed successfully!")
