#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Enhanced PTM Embedding Module

This module implements an enhanced PTM embedding layer that incorporates chemical properties
and uses an attention mechanism to weight these features contextually.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple, Optional, Union
import pandas as pd

class PTMChemicalFeatures:
    """
    Class to extract chemical features for post-translational modifications (PTMs)
    
    Features include:
    - Molecular weight
    - Hydrophobicity
    - Polarity
    - Charge
    - Size
    """
    
    def __init__(self):
        # Define chemical properties for common PTMs
        # Format: [molecular_weight, hydrophobicity, polarity, charge, size]
        self.ptm_features = {
            # Standard PTMs
            'Oxidation': [15.9949, -0.8, 0.9, 0, 0.2],
            'Carbamidomethyl': [57.0215, 0.1, 0.5, 0, 0.4],
            'Acetyl': [42.0106, 0.2, 0.3, 0, 0.3],
            'Phospho': [79.9663, -1.0, 1.0, -1, 0.5],
            'Methyl': [14.0157, 0.6, 0.1, 0, 0.1],
            'Deamidated': [0.9840, -0.5, 0.7, 0, 0.0],
            'GlyGly': [114.0429, -0.2, 0.4, 0, 0.6],
            'Pyro-glu': [-17.0265, 0.3, 0.2, 0, -0.1],
            'Pyro-carbamidomethyl': [39.9949, 0.2, 0.3, 0, 0.3],
            'Sulfo': [79.9568, -1.0, 0.9, -1, 0.5],
            'Formyl': [27.9949, 0.1, 0.4, 0, 0.2],
            'Propionyl': [56.0262, 0.4, 0.2, 0, 0.4],
            'Butyryl': [70.0419, 0.6, 0.1, 0, 0.5],
            'Crotonyl': [68.0262, 0.5, 0.2, 0, 0.5],
            'Malonyl': [86.0004, -0.3, 0.6, -1, 0.6],
            'Succinyl': [100.0160, -0.4, 0.7, -1, 0.7],
            'Glutaryl': [114.0317, -0.5, 0.7, -1, 0.8],
            'Methylthio': [45.9877, 0.7, 0.3, 0, 0.3],
            'Carbamyl': [43.0058, -0.1, 0.6, 0, 0.3],
            'Citrulline': [0.9840, -0.3, 0.5, 0, 0.0],
            'Nitro': [44.9851, -0.5, 0.8, 0, 0.3],
            'Dimethyl': [28.0313, 0.8, 0.1, 0, 0.2],
            'Trimethyl': [42.0470, 0.9, 0.0, 1, 0.3],
            'HexNAc': [203.0794, -0.3, 0.7, 0, 1.0],
            'Hex': [162.0528, -0.4, 0.8, 0, 0.9],
            'DeHex': [144.0423, -0.2, 0.6, 0, 0.8],
            'Cysteinyl': [119.1435, 0.1, 0.5, 0, 0.7],
            
            # Default for unknown modifications
            'Unknown': [0.0, 0.0, 0.0, 0, 0.0]
        }
        
        # Normalize features to [0, 1] range
        self._normalize_features()
        
    def _normalize_features(self):
        """Normalize chemical features to [0, 1] range"""
        features = np.array([v for v in self.ptm_features.values()])
        
        # Calculate min and max for each feature
        self.min_values = np.min(features, axis=0)
        self.max_values = np.max(features, axis=0)
        
        # Avoid division by zero
        self.max_values = np.where(self.max_values - self.min_values > 0, 
                                  self.max_values, self.min_values + 1)
        
        # Normalize each PTM's features
        for ptm in self.ptm_features:
            self.ptm_features[ptm] = (np.array(self.ptm_features[ptm]) - self.min_values) / (self.max_values - self.min_values)
    
    def get_features(self, ptm_name: str) -> List[float]:
        """
        Get chemical features for a specific PTM
        
        Parameters
        ----------
        ptm_name : str
            Name of the PTM
            
        Returns
        -------
        List[float]
            Normalized chemical features
        """
        # Extract the base PTM name (remove position information)
        base_ptm = ptm_name.split('@')[0] if '@' in ptm_name else ptm_name
        
        # Return features for the PTM or default to Unknown
        return self.ptm_features.get(base_ptm, self.ptm_features['Unknown'])


class EnhancedPTMEmbedding(nn.Module):
    """
    Enhanced PTM embedding layer that incorporates chemical properties
    and uses an attention mechanism to weight these features contextually
    """
    
    def __init__(self, 
                 ptm_vocab_size: int, 
                 embedding_dim: int = 32, 
                 chemical_feature_dim: int = 5,
                 use_attention: bool = True):
        """
        Initialize the enhanced PTM embedding layer
        
        Parameters
        ----------
        ptm_vocab_size : int
            Size of the PTM vocabulary
        embedding_dim : int
            Dimension of the PTM embedding
        chemical_feature_dim : int
            Dimension of the chemical features
        use_attention : bool
            Whether to use attention mechanism
        """
        super(EnhancedPTMEmbedding, self).__init__()
        
        self.ptm_vocab_size = ptm_vocab_size
        self.embedding_dim = embedding_dim
        self.chemical_feature_dim = chemical_feature_dim
        self.use_attention = use_attention
        
        # Standard embedding layer for PTMs
        self.ptm_embedding = nn.Embedding(ptm_vocab_size, embedding_dim)
        
        # Chemical feature embedding
        self.chemical_embedding = nn.Linear(chemical_feature_dim, embedding_dim)
        
        # Attention mechanism
        if use_attention:
            self.attention_query = nn.Linear(embedding_dim, embedding_dim)
            self.attention_key = nn.Linear(embedding_dim, embedding_dim)
            self.attention_value = nn.Linear(embedding_dim, embedding_dim)
            
        # Final projection
        self.output_projection = nn.Linear(embedding_dim * 2 if use_attention else embedding_dim, embedding_dim)
        
        # Chemical feature extractor
        self.chemical_features = PTMChemicalFeatures()
        
        # PTM vocabulary to map PTM names to indices
        self.ptm_vocab = {}
        
    def set_ptm_vocab(self, ptm_vocab: Dict[str, int]):
        """
        Set the PTM vocabulary
        
        Parameters
        ----------
        ptm_vocab : Dict[str, int]
            Dictionary mapping PTM names to indices
        """
        self.ptm_vocab = ptm_vocab
        
    def get_chemical_features_tensor(self, ptm_names: List[str]) -> torch.Tensor:
        """
        Get chemical features tensor for a list of PTMs
        
        Parameters
        ----------
        ptm_names : List[str]
            List of PTM names
            
        Returns
        -------
        torch.Tensor
            Tensor of chemical features
        """
        features = []
        for ptm in ptm_names:
            features.append(self.chemical_features.get_features(ptm))
        
        return torch.tensor(features, dtype=torch.float32)
    
    def forward(self, ptm_indices: torch.Tensor, chemical_features: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass
        
        Parameters
        ----------
        ptm_indices : torch.Tensor
            Tensor of PTM indices
        chemical_features : torch.Tensor, optional
            Tensor of chemical features. If None, will be derived from ptm_indices
            
        Returns
        -------
        torch.Tensor
            Enhanced PTM embeddings
        """
        # Get standard embeddings
        standard_embeddings = self.ptm_embedding(ptm_indices)
        
        # If chemical features are not provided, use zeros
        if chemical_features is None:
            chemical_features = torch.zeros((ptm_indices.shape[0], self.chemical_feature_dim), 
                                           device=ptm_indices.device)
        
        # Project chemical features to embedding space
        chemical_embeddings = self.chemical_embedding(chemical_features)
        
        if self.use_attention:
            # Compute attention scores
            query = self.attention_query(standard_embeddings)
            key = self.attention_key(chemical_embeddings)
            value = self.attention_value(chemical_embeddings)
            
            # Compute attention weights
            attention_scores = torch.matmul(query, key.transpose(-2, -1)) / (self.embedding_dim ** 0.5)
            attention_weights = F.softmax(attention_scores, dim=-1)
            
            # Apply attention weights
            attended_chemical_embeddings = torch.matmul(attention_weights, value)
            
            # Concatenate standard and attended chemical embeddings
            combined_embeddings = torch.cat([standard_embeddings, attended_chemical_embeddings], dim=-1)
        else:
            # Add standard and chemical embeddings
            combined_embeddings = standard_embeddings + chemical_embeddings
        
        # Final projection
        enhanced_embeddings = self.output_projection(combined_embeddings)
        
        return enhanced_embeddings


class PTMFeatureExtractor:
    """
    Feature extraction pipeline for PTM properties
    """
    
    def __init__(self):
        """Initialize the PTM feature extractor"""
        self.chemical_features = PTMChemicalFeatures()
        
    def extract_features(self, peptide_df):
        """
        Extract PTM features from peptide DataFrame
        
        Parameters
        ----------
        peptide_df : pd.DataFrame
            DataFrame with peptide data
            
        Returns
        -------
        pd.DataFrame
            DataFrame with added PTM features
        """
        # Make a copy to avoid modifying the original
        df = peptide_df.copy()
        
        # Extract PTM features
        ptm_features = []
        
        for _, row in df.iterrows():
            # Get PTMs
            mods = row.get('mods', '')
            if pd.isna(mods) or mods == '':
                # No modifications
                ptm_features.append([0.0, 0.0, 0.0, 0.0, 0.0])
                continue
            
            # Split multiple modifications
            mod_list = mods.split(';')
            
            # Average features for all modifications
            avg_features = np.zeros(5)
            for mod in mod_list:
                if mod:
                    features = self.chemical_features.get_features(mod)
                    avg_features += np.array(features)
            
            if len(mod_list) > 0:
                avg_features /= len(mod_list)
                
            ptm_features.append(avg_features.tolist())
        
        # Add features to DataFrame
        df['ptm_molecular_weight'] = [f[0] for f in ptm_features]
        df['ptm_hydrophobicity'] = [f[1] for f in ptm_features]
        df['ptm_polarity'] = [f[2] for f in ptm_features]
        df['ptm_charge'] = [f[3] for f in ptm_features]
        df['ptm_size'] = [f[4] for f in ptm_features]
        
        return df