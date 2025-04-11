#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Thread B: Enhanced PTM Embedding

This script implements an enhanced PTM embedding strategy that integrates additional 
chemical descriptors to improve spectral prediction accuracy. It also implements an 
attention mechanism to dynamically weight these features in the context of peptide sequences.
"""

import os
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
from tqdm import tqdm
import time

from peptdeep.pretrained_models import ModelManager
from peptdeep.model.featurize import get_batch_mod_feature
from peptdeep.settings import model_const, MOD_DF, MOD_TO_FEATURE
import peptdeep.model.building_block as building_block
from peptdeep.model.ms2 import pDeepModel
from alphabase.peptide.fragment import get_charged_frag_types, init_fragment_by_precursor_dataframe
from peptdeep.settings import global_settings
from peptdeep.utils import evaluate_linear_regression


# Define additional chemical properties for common PTMs
ptm_chemical_properties = {
    'Phospho@S': {
        'molecular_weight': 79.97,  # Da
        'hydrophobicity': -3.5,     # Relative scale
        'polarity': 4.0,            # Relative scale
        'charge': -2.0,             # At pH 7
        'size': 1.2                 # Relative scale
    },
    'Phospho@T': {
        'molecular_weight': 79.97,
        'hydrophobicity': -3.5,
        'polarity': 4.0,
        'charge': -2.0,
        'size': 1.2
    },
    'Phospho@Y': {
        'molecular_weight': 79.97,
        'hydrophobicity': -3.0,
        'polarity': 3.8,
        'charge': -2.0,
        'size': 1.2
    },
    'Carbamidomethyl@C': {
        'molecular_weight': 57.02,
        'hydrophobicity': -0.5,
        'polarity': 1.5,
        'charge': 0.0,
        'size': 0.8
    },
    'Oxidation@M': {
        'molecular_weight': 15.99,
        'hydrophobicity': -1.5,
        'polarity': 2.0,
        'charge': 0.0,
        'size': 0.4
    },
    'Acetyl@Protein_N-term': {
        'molecular_weight': 42.01,
        'hydrophobicity': 0.0,
        'polarity': 1.0,
        'charge': -1.0,
        'size': 0.6
    },
    'GlyGly@K': {
        'molecular_weight': 114.04,
        'hydrophobicity': -0.2,
        'polarity': 1.2,
        'charge': 0.0,
        'size': 1.0
    }
}


def get_default_ptm_properties():
    """Get default PTM properties for PTMs not in our dictionary
    
    Returns
    -------
    dict
        Default PTM properties
    """
    return {
        'molecular_weight': 0.0,
        'hydrophobicity': 0.0,
        'polarity': 0.0,
        'charge': 0.0,
        'size': 0.0
    }


def get_enhanced_mod_feature(mod_name):
    """Get enhanced modification feature vector
    
    Parameters
    ----------
    mod_name : str
        Modification name
        
    Returns
    -------
    np.ndarray
        Enhanced modification feature vector
    """
    # Get the original feature vector
    original_feature = MOD_TO_FEATURE.get(mod_name, np.zeros(len(model_const['mod_elements'])))
    
    # Get chemical properties
    properties = ptm_chemical_properties.get(mod_name, get_default_ptm_properties())
    
    # Normalize properties to [0, 1] range
    mw_norm = min(1.0, properties['molecular_weight'] / 200.0)  # Normalize by max expected MW
    hydro_norm = (properties['hydrophobicity'] + 4.0) / 8.0     # Range from -4 to 4
    polar_norm = properties['polarity'] / 5.0                   # Range from 0 to 5
    charge_norm = (properties['charge'] + 2.0) / 4.0            # Range from -2 to 2
    size_norm = properties['size'] / 2.0                        # Range from 0 to 2
    
    # Create enhanced feature vector
    enhanced_feature = np.concatenate([
        original_feature,
        np.array([mw_norm, hydro_norm, polar_norm, charge_norm, size_norm])
    ])
    
    return enhanced_feature


def create_enhanced_mod_features():
    """Create enhanced feature vectors for all modifications
    
    Returns
    -------
    dict
        Dictionary mapping modification names to enhanced feature vectors
    """
    enhanced_mod_to_feature = {}
    for mod_name in MOD_TO_FEATURE.keys():
        enhanced_mod_to_feature[mod_name] = get_enhanced_mod_feature(mod_name)
    
    return enhanced_mod_to_feature


def get_enhanced_batch_mod_feature(batch_df):
    """Get enhanced batch modification features
    
    Parameters
    ----------
    batch_df : pd.DataFrame
        Batch dataframe
        
    Returns
    -------
    torch.Tensor
        Enhanced batch modification features
    """
    # Get original batch mod features
    original_features = get_batch_mod_feature(batch_df)
    
    # Create enhanced features
    batch_size = len(batch_df)
    max_len = max(batch_df.nAA)
    enhanced_features = np.zeros((batch_size, max_len, len(model_const['mod_elements']) + 5))
    
    for i, (_, row) in enumerate(batch_df.iterrows()):
        seq_len = row.nAA
        
        # Copy original features
        enhanced_features[i, :seq_len, :len(model_const['mod_elements'])] = original_features[i, :seq_len]
        
        # Add chemical properties
        if row.mods and row.mod_sites:
            mods = row.mods.split(';')
            mod_sites = [int(site) for site in row.mod_sites.split(';')]
            
            for mod, site in zip(mods, mod_sites):
                if site < seq_len and mod in ptm_chemical_properties:
                    props = ptm_chemical_properties[mod]
                    
                    # Normalize properties
                    mw_norm = min(1.0, props['molecular_weight'] / 200.0)
                    hydro_norm = (props['hydrophobicity'] + 4.0) / 8.0
                    polar_norm = props['polarity'] / 5.0
                    charge_norm = (props['charge'] + 2.0) / 4.0
                    size_norm = props['size'] / 2.0
                    
                    # Add to features
                    enhanced_features[i, site, len(model_const['mod_elements'])] = mw_norm
                    enhanced_features[i, site, len(model_const['mod_elements'])+1] = hydro_norm
                    enhanced_features[i, site, len(model_const['mod_elements'])+2] = polar_norm
                    enhanced_features[i, site, len(model_const['mod_elements'])+3] = charge_norm
                    enhanced_features[i, site, len(model_const['mod_elements'])+4] = size_norm
    
    return torch.tensor(enhanced_features, dtype=torch.float32)


class PTMAttentionLayer(torch.nn.Module):
    """Attention layer for PTM features"""
    
    def __init__(self, ptm_dim, hidden_dim):
        """Initialize the attention layer
        
        Parameters
        ----------
        ptm_dim : int
            Dimension of PTM features
        hidden_dim : int
            Dimension of hidden layer
        """
        super().__init__()
        self.query = torch.nn.Linear(hidden_dim, ptm_dim)
        self.key = torch.nn.Linear(ptm_dim, ptm_dim)
        self.value = torch.nn.Linear(ptm_dim, ptm_dim)
        self.scale = torch.sqrt(torch.tensor(ptm_dim, dtype=torch.float32))
        
    def forward(self, ptm_features, seq_features):
        """Forward pass
        
        Parameters
        ----------
        ptm_features : torch.Tensor
            PTM features [batch_size, seq_len, ptm_dim]
        seq_features : torch.Tensor
            Sequence features [batch_size, seq_len, hidden_dim]
            
        Returns
        -------
        torch.Tensor
            Attended PTM features [batch_size, seq_len, ptm_dim]
        """
        # Generate query from sequence features
        q = self.query(seq_features)  # [batch_size, seq_len, ptm_dim]
        
        # Generate key and value from PTM features
        k = self.key(ptm_features)    # [batch_size, seq_len, ptm_dim]
        v = self.value(ptm_features)  # [batch_size, seq_len, ptm_dim]
        
        # Calculate attention scores
        scores = torch.matmul(q, k.transpose(-2, -1)) / self.scale  # [batch_size, seq_len, seq_len]
        
        # Apply softmax to get attention weights
        attn_weights = torch.nn.functional.softmax(scores, dim=-1)  # [batch_size, seq_len, seq_len]
        
        # Apply attention weights to values
        attended_features = torch.matmul(attn_weights, v)  # [batch_size, seq_len, ptm_dim]
        
        return attended_features


class EnhancedPTMEmbedding(torch.nn.Module):
    """Enhanced PTM embedding module"""
    
    def __init__(self, hidden_dim):
        """Initialize the enhanced PTM embedding module
        
        Parameters
        ----------
        hidden_dim : int
            Dimension of hidden layer
        """
        super().__init__()
        
        # Original PTM dimension + 5 additional chemical properties
        self.ptm_dim = len(model_const['mod_elements']) + 5
        
        # Embedding layers
        self.ptm_embedding = torch.nn.Linear(self.ptm_dim, hidden_dim)
        
        # Attention mechanism
        self.attention = PTMAttentionLayer(hidden_dim, hidden_dim)
        
    def forward(self, ptm_features, seq_features):
        """Forward pass
        
        Parameters
        ----------
        ptm_features : torch.Tensor
            PTM features [batch_size, seq_len, ptm_dim]
        seq_features : torch.Tensor
            Sequence features [batch_size, seq_len, hidden_dim]
            
        Returns
        -------
        torch.Tensor
            Combined features [batch_size, seq_len, hidden_dim]
        """
        # Embed PTM features
        ptm_embedded = self.ptm_embedding(ptm_features)
        
        # Apply attention
        attended_ptm = self.attention(ptm_embedded, seq_features)
        
        # Combine with sequence features
        combined = seq_features + attended_ptm
        
        return combined


class EnhancedMS2Model(torch.nn.Module):
    """MS2 model with enhanced PTM embedding"""
    
    def __init__(self, num_frag_types, num_modloss_types=0, mask_modloss=True, dropout=0.1, nlayers=4, hidden=256):
        """Initialize the enhanced MS2 model
        
        Parameters
        ----------
        num_frag_types : int
            Number of fragment types
        num_modloss_types : int
            Number of modloss types
        mask_modloss : bool
            Whether to mask modloss
        dropout : float
            Dropout rate
        nlayers : int
            Number of transformer layers
        hidden : int
            Hidden dimension
        """
        super().__init__()
        
        self.dropout = torch.nn.Dropout(dropout)
        
        self._num_modloss_types = num_modloss_types
        self._num_non_modloss = num_frag_types - num_modloss_types
        self._mask_modloss = mask_modloss
        if num_modloss_types == 0:
            self._mask_modloss = True
        
        meta_dim = 8
        self.input_nn = building_block.Input_26AA_Mod_PositionalEncoding(hidden - meta_dim)
        
        # Enhanced PTM embedding
        self.enhanced_ptm_embedding = EnhancedPTMEmbedding(hidden - meta_dim)
        
        self.meta_nn = building_block.Meta_Embedding(meta_dim)
        
        self.hidden_nn = building_block.Hidden_HFace_Transformer(hidden, nlayers=nlayers, dropout=dropout)
        
        self.output_nn = building_block.Decoder_Linear(hidden, self._num_non_modloss)
        
        if num_modloss_types > 0:
            self.modloss_nn = torch.nn.ModuleList([
                building_block.Hidden_HFace_Transformer(hidden, nlayers=1, dropout=dropout),
                building_block.Decoder_Linear(hidden, num_modloss_types)
            ])
        else:
            self.modloss_nn = None
    
    def forward(self, aa_indices, mod_x, charges, NCEs, instrument_indices):
        """Forward pass
        
        Parameters
        ----------
        aa_indices : torch.Tensor
            Amino acid indices
        mod_x : torch.Tensor
            Modification features
        charges : torch.Tensor
            Charges
        NCEs : torch.Tensor
            NCEs
        instrument_indices : torch.Tensor
            Instrument indices
            
        Returns
        -------
        torch.Tensor
            Output tensor
        """
        # Get sequence features
        seq_features = self.input_nn(aa_indices, None)  # Don't pass mod_x yet
        
        # Apply enhanced PTM embedding
        enhanced_features = self.enhanced_ptm_embedding(mod_x, seq_features)
        
        # Apply dropout
        in_x = self.dropout(enhanced_features)
        
        # Add metadata
        meta_x = self.meta_nn(charges, NCEs, instrument_indices).unsqueeze(1).repeat(1, in_x.size(1), 1)
        in_x = torch.cat((in_x, meta_x), 2)
        
        # Apply transformer
        hidden_x = self.hidden_nn(in_x)[0]
        hidden_x = self.dropout(hidden_x + in_x * 0.2)
        
        # Output layer
        out_x = self.output_nn(hidden_x)
        
        # Handle modloss
        if self._num_modloss_types > 0:
            if self._mask_modloss:
                out_x = torch.cat((
                    out_x,
                    torch.zeros(*out_x.size()[:2], self._num_modloss_types, device=in_x.device)
                ), 2)
            else:
                modloss_x = self.modloss_nn[0](in_x)[0] + hidden_x
                modloss_x = self.modloss_nn[-1](modloss_x)
                out_x = torch.cat((out_x, modloss_x), 2)
        
        return out_x[:, 3:, :]


class EnhancedPDeepModel(pDeepModel):
    """pDeepModel with enhanced PTM embedding"""
    
    def __init__(self, charged_frag_types=None, dropout=0.1, mask_modloss=False, device="cpu"):
        """Initialize the enhanced pDeepModel"""
        if charged_frag_types is None:
            frag_types = global_settings["model"]["frag_types"]
            max_frag_charge = global_settings["model"]["max_frag_charge"]
            charged_frag_types = get_charged_frag_types(frag_types, max_frag_charge)
            
        super().__init__(charged_frag_types=charged_frag_types, dropout=dropout, mask_modloss=mask_modloss, device=device)
        
        # Replace the model with our enhanced model
        num_frag_types = len(self.charged_frag_types)
        num_modloss_types = len([frag for frag in self.charged_frag_types if "modloss" in frag])
        
        self.model = EnhancedMS2Model(
            num_frag_types=num_frag_types,
            num_modloss_types=num_modloss_types,
            mask_modloss=mask_modloss,
            dropout=dropout,
            nlayers=4,
            hidden=256
        )
        
        # Move model to device
        self.model.to(self.device)
    
    def _get_features_from_batch_df(self, batch_df):
        """Get features from batch dataframe
        
        Parameters
        ----------
        batch_df : pd.DataFrame
            Batch dataframe
            
        Returns
        -------
        tuple
            Tuple of features
        """
        aa_indices = self._get_26aa_indice_features(batch_df)
        mod_x = get_enhanced_batch_mod_feature(batch_df)
        charges = self._as_tensor(batch_df.charge.values * self.charge_factor, dtype=torch.float32)
        NCEs = self._as_tensor(batch_df.nce.values * self.NCE_factor, dtype=torch.float32)
        instrument_indices = self._as_tensor(
            batch_df.instrument_idx.values if "instrument_idx" in batch_df.columns else np.zeros(len(batch_df)),
            dtype=torch.long,
        )
        
        return aa_indices, mod_x, charges, NCEs, instrument_indices


def create_test_dataset():
    """Create a test dataset with phosphorylated peptides
    
    Returns
    -------
    pd.DataFrame
        Test dataset
    """
    test_df = pd.DataFrame({
        'sequence': ['ANEKTESSSAQQVAVSR', 'ANEKTESSTAQQVAVSR', 'ANEKTESSYAQQVAVSR', 'ANEKTESSSAQQVAVSR'],
        'mods': ['Phospho@S', 'Phospho@T', 'Phospho@Y', ''],
        'mod_sites': ['9', '9', '9', ''],
        'charge': [2, 2, 2, 2],
        'nce': [30, 30, 30, 30],
        'instrument': ['QE', 'QE', 'QE', 'QE']
    })
    test_df['nAA'] = test_df.sequence.str.len()
    
    return test_df


def compare_models(test_df, output_dir='results'):
    """Compare standard and enhanced models
    
    Parameters
    ----------
    test_df : pd.DataFrame
        Test dataset
    output_dir : str
        Directory to save results
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Load standard model
    print("Loading standard model...")
    standard_model_mgr = ModelManager(mask_modloss=False, device='cpu')
    standard_model_mgr.load_installed_models('phos')
    
    # Create enhanced model
    print("Creating enhanced model...")
    enhanced_model = EnhancedPDeepModel(mask_modloss=False, device='cpu')
    
    # Copy weights from standard model to enhanced model
    # This ensures a fair comparison by starting with the same weights
    print("Copying weights from standard model to enhanced model...")
    for name, param in standard_model_mgr.ms2_model.model.named_parameters():
        if name in dict(enhanced_model.model.named_parameters()):
            dict(enhanced_model.model.named_parameters())[name].data.copy_(param.data)
    
    # Prepare data for prediction
    print("Preparing data for prediction...")
    init_fragment_by_precursor_dataframe(test_df, standard_model_mgr.ms2_model.charged_frag_types)
    
    # Predict with standard model
    print("Predicting with standard model...")
    standard_result = standard_model_mgr.predict_all(test_df.copy(), predict_items=['ms2'])
    
    # Create a custom model manager with enhanced model
    enhanced_model_mgr = ModelManager(mask_modloss=False, device='cpu')
    enhanced_model_mgr.ms2_model = enhanced_model
    
    # Predict with enhanced model
    print("Predicting with enhanced model...")
    enhanced_result = enhanced_model_mgr.predict_all(test_df.copy(), predict_items=['ms2'])
    
    # Compare predictions
    print("Comparing predictions...")
    for i, row in test_df.iterrows():
        start_idx = row['frag_start_idx']
        end_idx = row['frag_stop_idx']
        
        # Get fragment intensities for this peptide
        standard_intensities = standard_result['fragment_intensity_df'].iloc[start_idx:end_idx]
        enhanced_intensities = enhanced_result['fragment_intensity_df'].iloc[start_idx:end_idx]
        
        # Plot comparison
        plt.figure(figsize=(12, 6))
        
        # Plot for y ions (most affected by PTMs)
        y_cols = [col for col in standard_intensities.columns if col.startswith('y_')]
        
        # Get fragment m/z values
        frag_mz = standard_result['fragment_mz_df'].iloc[start_idx:end_idx]
        
        for col in y_cols:
            mz_values = frag_mz[col].values
            standard_values = standard_intensities[col].values
            enhanced_values = enhanced_intensities[col].values
            
            # Filter out zero m/z values
            mask = mz_values > 0
            mz_values = mz_values[mask]
            standard_values = standard_values[mask]
            enhanced_values = enhanced_values[mask]
            
            if len(mz_values) > 0:
                plt.plot(mz_values, standard_values, 'o-', alpha=0.7, label=f'Standard {col}')
                plt.plot(mz_values, enhanced_values, 's--', alpha=0.7, label=f'Enhanced {col}')
        
        plt.xlabel('m/z')
        plt.ylabel('Relative Intensity')
        plt.title(f'MS2 Spectrum Comparison for {row["sequence"]} with {row["mods"]}')
        plt.legend()
        plt.grid(True)
        plt.savefig(os.path.join(output_dir, f'comparison_{row["sequence"]}_{row["mods"]}.png'))
    
    # Save results to CSV
    standard_result['fragment_intensity_df'].to_csv(os.path.join(output_dir, 'standard_intensities.csv'), index=False)
    enhanced_result['fragment_intensity_df'].to_csv(os.path.join(output_dir, 'enhanced_intensities.csv'), index=False)
    
    print(f"Results saved to {output_dir}")


def analyze_ptm_attention(output_dir='results'):
    """Analyze PTM attention weights
    
    Parameters
    ----------
    output_dir : str
        Directory to save results
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Create test dataset
    print("Creating test dataset...")
    test_df = create_test_dataset()
    
    # Create enhanced model with attention output
    print("Creating enhanced model...")
    enhanced_model = EnhancedMS2Model(
        num_frag_types=len(get_charged_frag_types(global_settings["model"]["frag_types"], global_settings["model"]["max_frag_charge"])),
        num_modloss_types=0,
        mask_modloss=False,
        dropout=0.1,
        nlayers=4,
        hidden=256
    )
    
    # Get features for visualization
    print("Getting features for visualization...")
    aa_indices = torch.zeros((len(test_df), max(test_df.nAA)), dtype=torch.long)
    for i, row in test_df.iterrows():
        for j, aa in enumerate(row.sequence):
            aa_indices[i, j] = ord(aa) - ord('A')
    
    mod_x = get_enhanced_batch_mod_feature(test_df)
    charges = torch.tensor(test_df.charge.values * 0.1, dtype=torch.float32)
    NCEs = torch.tensor(test_df.nce.values * 0.01, dtype=torch.float32)
    instrument_indices = torch.zeros(len(test_df), dtype=torch.long)
    
    # Get sequence features
    seq_features = enhanced_model.input_nn(aa_indices, None)
    
    # Get PTM embedding
    ptm_embedded = enhanced_model.enhanced_ptm_embedding.ptm_embedding(mod_x)
    
    # Get attention weights
    q = enhanced_model.enhanced_ptm_embedding.attention.query(seq_features)
    k = enhanced_model.enhanced_ptm_embedding.attention.key(ptm_embedded)
    scale = enhanced_model.enhanced_ptm_embedding.attention.scale
    scores = torch.matmul(q, k.transpose(-2, -1)) / scale
    attn_weights = torch.nn.functional.softmax(scores, dim=-1)
    
    # Visualize attention weights
    print("Visualizing attention weights...")
    for i, row in test_df.iterrows():
        plt.figure(figsize=(10, 8))
        plt.imshow(attn_weights[i].detach().numpy(), cmap='viridis')
        plt.colorbar()
        plt.title(f'Attention Weights for {row["sequence"]} with {row["mods"]}')
        plt.xlabel('Position (Key)')
        plt.ylabel('Position (Query)')
        plt.xticks(range(len(row.sequence)), list(row.sequence))
        plt.yticks(range(len(row.sequence)), list(row.sequence))
        plt.savefig(os.path.join(output_dir, f'attention_{row["sequence"]}_{row["mods"]}.png'))
    
    print(f"Results saved to {output_dir}")


if __name__ == "__main__":
    # Create results directory
    output_dir = 'enhanced_ptm_results'
    os.makedirs(output_dir, exist_ok=True)
    
    # Print current PTM feature dimensions
    print(f"Current PTM feature dimensions: {len(model_const['mod_elements'])}")
    print(f"Current mod elements: {model_const['mod_elements']}")
    
    # Create enhanced PTM features
    enhanced_mod_to_feature = create_enhanced_mod_features()
    
    # Print some examples
    print("\nEnhanced PTM features:")
    for mod_name, feature in list(enhanced_mod_to_feature.items())[:5]:
        print(f"{mod_name}: {feature}")
    
    # Print dimensions
    print(f"\nOriginal feature dimension: {len(next(iter(MOD_TO_FEATURE.values())))}")
    print(f"Enhanced feature dimension: {len(next(iter(enhanced_mod_to_feature.values())))}")
    
    # Create test dataset
    test_df = create_test_dataset()
    
    # Compare models
    compare_models(test_df, output_dir=os.path.join(output_dir, 'model_comparison'))
    
    # Analyze PTM attention
    analyze_ptm_attention(output_dir=os.path.join(output_dir, 'attention_analysis'))
    
    print("Enhanced PTM embedding experiment completed successfully!")
