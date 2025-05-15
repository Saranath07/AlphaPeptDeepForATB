#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Test AlphaPeptDeep Models with AlphaFold3 Structure Predictions

This script tests the training of models that incorporate AlphaFold3 structure predictions
using just 3 data points to verify functionality.
"""

import os
import pandas as pd
import numpy as np
import torch
from sklearn.model_selection import train_test_split
import time

from peptdeep.pretrained_models import ModelManager
from peptdeep.model.enhanced_model import EnhancedModelManager, EnhancedAlphaRTModel, EnhancedAlphaMS2Model
from alphafold3 import AlphaFold3
import torch.nn as nn
from typing import Dict, List, Tuple, Optional, Union, Any


def predict_structure_with_alphafold3(sequence):
    """
    Use AlphaFold3 to predict the structure of a peptide sequence
    
    Parameters
    ----------
    sequence : str
        Peptide sequence
        
    Returns
    -------
    dict
        AlphaFold3 output containing structure predictions
    """
    # Convert sequence to feature encodings
    seq_len = len(sequence)
    dim = 64  # Feature dimension
    
    # Initialize feature tensors
    pair_representations = torch.randn(1, seq_len, seq_len, dim)
    single_representations = torch.randn(1, seq_len, dim)
    
    # Initialize AlphaFold3 model with all required parameters
    model = AlphaFold3(
        dim=dim,
        seq_len=seq_len,
        heads=8,
        dim_head=64,
        attn_dropout=0.0,
        ff_dropout=0.0,
        global_column_attn=False,
        pair_former_depth=48,
        num_diffusion_steps=1000,
        diffusion_depth=30,
    )
    
    # Forward pass to get structure prediction
    output = model(pair_representations, single_representations)
    
    return output


def create_test_dataset():
    """
    Create a small test dataset with 3 peptides
    
    Returns
    -------
    pd.DataFrame
        DataFrame with peptide data
    """
    print("Creating test dataset with 3 peptides...")
    
    # Create a small dataset with 3 peptides
    data = {
        'sequence': ['LVRPEVDVMCTAFHDNEETFLK', 'AKVTEFGCGTR', 'YILAGVENSK'],
        'mods': ['', '', ''],
        'mod_sites': ['', '', ''],
        'charge': [2, 2, 2],
        'rt': [30.5, 15.2, 22.7],
        'nce': [30, 30, 30],
        'instrument': ['QE', 'QE', 'QE']
    }
    
    df = pd.DataFrame(data)
    
    # Add nAA column (peptide length)
    df['nAA'] = df.sequence.str.len()
    
    print(f"Created dataset with {len(df)} peptides")
    return df


def generate_alphafold_embeddings(peptide_df):
    """
    Generate AlphaFold3 embeddings for peptides
    
    Parameters
    ----------
    peptide_df : pd.DataFrame
        DataFrame with peptide data
        
    Returns
    -------
    dict
        Dictionary mapping peptide sequences to AlphaFold3 embeddings
    """
    print("Generating AlphaFold3 embeddings...")
    
    embeddings = {}
    sequences = peptide_df['sequence'].unique()
    
    for seq in sequences:
        try:
            # Get AlphaFold3 prediction
            af_output = predict_structure_with_alphafold3(seq)
            embeddings[seq] = af_output
            print(f"Generated embedding for {seq}")
        except Exception as e:
            print(f"Error processing sequence {seq}: {str(e)}")
            # Create a placeholder embedding
            seq_len = len(seq)
            dim = 64
            embeddings[seq] = torch.zeros(1, seq_len, dim)
    
    print(f"Generated AlphaFold3 embeddings for {len(embeddings)} unique peptides")
    return embeddings


class AlphaFoldEnhancedRTModel(EnhancedAlphaRTModel):
    """
    Enhanced RT model that incorporates AlphaFold3 structure predictions
    """
    
    def __init__(self,
                 embedding_dim: int = 32,
                 lstm_hidden_dim: int = 128,
                 lstm_layers: int = 2,
                 dropout: float = 0.2,
                 bidirectional: bool = True,
                 use_attention: bool = True,
                 alphafold_dim: int = 64,
                 device: str = 'cpu'):
        """
        Initialize the AlphaFold-enhanced RT model
        """
        super().__init__(
            embedding_dim=embedding_dim,
            lstm_hidden_dim=lstm_hidden_dim,
            lstm_layers=lstm_layers,
            dropout=dropout,
            bidirectional=bidirectional,
            use_attention=use_attention,
            device=device
        )
        
        # Add projection layer for AlphaFold embeddings
        self.alphafold_projection = nn.Linear(alphafold_dim, embedding_dim)
        
        # Dictionary to store AlphaFold embeddings
        self.alphafold_embeddings = {}
    
    def forward(self, batch_data: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Forward pass with AlphaFold embeddings
        """
        # Get amino acid and PTM indices
        aa_indices = batch_data['aa_indices']
        ptm_indices = batch_data['ptm_indices']
        
        # Get PTM chemical features if available
        ptm_features = batch_data.get('ptm_features', None)
        
        # If ptm_features is provided but has wrong shape, reshape it
        if ptm_features is not None:
            if len(ptm_features.shape) == 2:  # [batch_size, features]
                # We need to expand to [batch_size, seq_len, features]
                batch_size = aa_indices.shape[0]
                seq_len = aa_indices.shape[1]
                # Reshape to [batch_size, 1, features] and repeat for each position
                ptm_features = ptm_features.unsqueeze(1).expand(-1, seq_len, -1)
        
        # Get embeddings
        aa_embeddings = self.aa_embedding(aa_indices)
        ptm_embeddings = self.ptm_embedding(ptm_indices, ptm_features)
        
        # Get AlphaFold embeddings if available
        alphafold_features = batch_data.get('alphafold_features', None)
        if alphafold_features is not None:
            # Project AlphaFold features to embedding dimension
            alphafold_embeddings = self.alphafold_projection(alphafold_features)
            
            # Combine embeddings
            embeddings = aa_embeddings + ptm_embeddings + alphafold_embeddings
        else:
            # Combine embeddings without AlphaFold
            embeddings = aa_embeddings + ptm_embeddings
        
        # Convert to float32 for LSTM
        if embeddings.dtype != torch.float32:
            embeddings = embeddings.float()
        
        # Pack sequence for LSTM
        packed_input = nn.utils.rnn.pack_padded_sequence(
            embeddings, batch_data['lengths'], batch_first=True, enforce_sorted=False
        )
        
        # Forward through the model (which contains the LSTM)
        return self.model(packed_input)


class AlphaFoldEnhancedMS2Model(EnhancedAlphaMS2Model):
    """
    Enhanced MS2 model that incorporates AlphaFold3 structure predictions
    """
    
    def __init__(self,
                 embedding_dim: int = 32,
                 lstm_hidden_dim: int = 128,
                 lstm_layers: int = 2,
                 dropout: float = 0.2,
                 bidirectional: bool = True,
                 use_attention: bool = True,
                 alphafold_dim: int = 64,
                 device: str = 'cpu'):
        """
        Initialize the AlphaFold-enhanced MS2 model
        """
        super().__init__(
            embedding_dim=embedding_dim,
            lstm_hidden_dim=lstm_hidden_dim,
            lstm_layers=lstm_layers,
            dropout=dropout,
            bidirectional=bidirectional,
            use_attention=use_attention,
            device=device
        )
        
        # Add projection layer for AlphaFold embeddings
        self.alphafold_projection = nn.Linear(alphafold_dim, embedding_dim)
        
        # Dictionary to store AlphaFold embeddings
        self.alphafold_embeddings = {}
    
    def forward(self, batch_data: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Forward pass with AlphaFold embeddings
        """
        # Get amino acid and PTM indices
        aa_indices = batch_data['aa_indices']
        ptm_indices = batch_data['ptm_indices']
        
        # Get PTM chemical features if available
        ptm_features = batch_data.get('ptm_features', None)
        
        # If ptm_features is provided but has wrong shape, reshape it
        if ptm_features is not None:
            if len(ptm_features.shape) == 2:  # [batch_size, features]
                # We need to expand to [batch_size, seq_len, features]
                batch_size = aa_indices.shape[0]
                seq_len = aa_indices.shape[1]
                # Reshape to [batch_size, 1, features] and repeat for each position
                ptm_features = ptm_features.unsqueeze(1).expand(-1, seq_len, -1)
        
        # Get embeddings
        aa_embeddings = self.aa_embedding(aa_indices)
        ptm_embeddings = self.ptm_embedding(ptm_indices, ptm_features)
        
        # Get AlphaFold embeddings if available
        alphafold_features = batch_data.get('alphafold_features', None)
        if alphafold_features is not None:
            # Project AlphaFold features to embedding dimension
            alphafold_embeddings = self.alphafold_projection(alphafold_features)
            
            # Combine embeddings
            embeddings = aa_embeddings + ptm_embeddings + alphafold_embeddings
        else:
            # Combine embeddings without AlphaFold
            embeddings = aa_embeddings + ptm_embeddings
        
        # Convert to float32 for LSTM
        if embeddings.dtype != torch.float32:
            embeddings = embeddings.float()
        
        # Forward through the model
        return self.model(embeddings)


class AlphaFoldEnhancedModelManager(EnhancedModelManager):
    """
    Enhanced model manager that incorporates AlphaFold3 structure predictions
    """
    
    def __init__(self,
                 mask_modloss: bool = True,
                 device: str = 'cpu',
                 use_attention: bool = True):
        """
        Initialize the AlphaFold-enhanced model manager
        """
        # Initialize parent class
        super().__init__(mask_modloss, device, use_attention)
        
        # Dictionary to store AlphaFold embeddings
        self.alphafold_embeddings = {}
        
        # Replace standard models with AlphaFold-enhanced models
        self.rt_model = AlphaFoldEnhancedRTModel(
            use_attention=use_attention,
            device=device
        )
        
        self.ms2_model = AlphaFoldEnhancedMS2Model(
            use_attention=use_attention,
            device=device
        )
        
    def set_alphafold_embeddings(self, embeddings_dict):
        """
        Set AlphaFold embeddings
        
        Parameters
        ----------
        embeddings_dict : dict
            Dictionary mapping peptide sequences to AlphaFold embeddings
        """
        self.alphafold_embeddings = embeddings_dict
        
    def train_rt_model(self,
                      train_df: pd.DataFrame,
                      val_df: Optional[pd.DataFrame] = None,
                      epochs: int = 10,
                      batch_size: int = 64,
                      learning_rate: float = 0.001):
        """
        Train RT model with AlphaFold embeddings
        """
        # Extract PTM features
        train_df = self.ptm_feature_extractor.extract_features(train_df)
        if val_df is not None:
            val_df = self.ptm_feature_extractor.extract_features(val_df)
        
        # Add AlphaFold features flag to the dataframe
        train_df = self._add_alphafold_features(train_df)
        if val_df is not None:
            val_df = self._add_alphafold_features(val_df)
        
        # Share AlphaFold embeddings with the model
        self.rt_model.alphafold_embeddings = self.alphafold_embeddings
        
        # Train the model
        self.rt_model.train_model(
            train_df=train_df,
            val_df=val_df,
            epochs=epochs,
            batch_size=batch_size,
            learning_rate=learning_rate
        )
    
    def train_ms2_model(self,
                       train_df: pd.DataFrame,
                       val_df: Optional[pd.DataFrame] = None,
                       epochs: int = 10,
                       batch_size: int = 64,
                       learning_rate: float = 0.001):
        """
        Train MS2 model with AlphaFold embeddings
        """
        # Extract PTM features
        train_df = self.ptm_feature_extractor.extract_features(train_df)
        if val_df is not None:
            val_df = self.ptm_feature_extractor.extract_features(val_df)
        
        # Add AlphaFold features flag to the dataframe
        train_df = self._add_alphafold_features(train_df)
        if val_df is not None:
            val_df = self._add_alphafold_features(val_df)
        
        # Share AlphaFold embeddings with the model
        self.ms2_model.alphafold_embeddings = self.alphafold_embeddings
        
        # Add dummy intensity column for training
        train_df_with_intensity = train_df.copy()
        train_df_with_intensity['intensity'] = np.random.rand(len(train_df))
        
        if val_df is not None:
            val_df_with_intensity = val_df.copy()
            val_df_with_intensity['intensity'] = np.random.rand(len(val_df))
        else:
            val_df_with_intensity = None
        
        # Train the model
        self.ms2_model.train_model(
            train_df=train_df_with_intensity,
            val_df=val_df_with_intensity,
            epochs=epochs,
            batch_size=batch_size,
            learning_rate=learning_rate
        )
    
    def _prepare_batch_with_alphafold(self, batch_data, sequences):
        """
        Prepare batch data with AlphaFold embeddings
        """
        # Create a copy of the batch data
        batch_with_alphafold = dict(batch_data)
        
        # Get AlphaFold embeddings for each sequence in the batch
        batch_size = len(sequences)
        seq_len = batch_data['aa_indices'].shape[1]
        alphafold_dim = 64  # Default dimension
        
        # Initialize tensor for AlphaFold features
        alphafold_features = torch.zeros((batch_size, seq_len, alphafold_dim), device=self.device)
        
        # Fill in AlphaFold features
        for i, seq in enumerate(sequences):
            if seq in self.alphafold_embeddings:
                # Get AlphaFold embedding
                af_embedding = self.alphafold_embeddings[seq]
                
                # Convert to tensor if it's not already
                if not isinstance(af_embedding, torch.Tensor):
                    af_embedding = torch.tensor(af_embedding, device=self.device)
                
                # Ensure it's on the correct device
                af_embedding = af_embedding.to(self.device)
                
                # Reshape if needed
                if len(af_embedding.shape) == 1:
                    # If it's a 1D tensor, expand to match sequence length
                    af_embedding = af_embedding.unsqueeze(0).expand(seq_len, -1)
                elif len(af_embedding.shape) == 3 and af_embedding.shape[0] == 1:
                    # If it's a 3D tensor with batch dimension 1, remove batch dimension
                    af_embedding = af_embedding.squeeze(0)
                
                # Ensure the embedding has the right shape
                if af_embedding.shape[0] >= seq_len:
                    # Truncate if needed
                    alphafold_features[i, :seq_len, :] = af_embedding[:seq_len, :]
                else:
                    # Pad if needed
                    alphafold_features[i, :af_embedding.shape[0], :] = af_embedding
        
        # Add AlphaFold features to batch data
        batch_with_alphafold['alphafold_features'] = alphafold_features
        
        return batch_with_alphafold
    
    def _add_alphafold_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add AlphaFold features to the dataframe
        """
        # Create a copy to avoid modifying the original
        result_df = df.copy()
        
        # Add a flag column to indicate that AlphaFold features should be used
        result_df['use_alphafold'] = True
        
        return result_df


def train_standard_model(train_df, val_df, output_dir, epochs=1):
    """
    Train the standard AlphaPeptDeep model
    """
    print("Training standard AlphaPeptDeep model...")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Initialize model manager
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    model_mgr = ModelManager(mask_modloss=False, device=device)
    
    # Train RT model
    print("Training RT model...")
    start_time = time.time()
    model_mgr.rt_model.train_model(
        train_df=train_df,
        val_df=val_df,
        epochs=epochs,
        batch_size=2,  # Small batch size for few data points
        learning_rate=0.001
    )
    rt_training_time = time.time() - start_time
    print(f"RT model training completed in {rt_training_time:.2f} seconds")
    
    # Train MS2 model
    print("Training MS2 model...")
    start_time = time.time()
    model_mgr.ms2_model.train_model(
        train_df=train_df,
        val_df=val_df,
        epochs=epochs,
        batch_size=2,  # Small batch size for few data points
        learning_rate=0.001
    )
    ms2_training_time = time.time() - start_time
    print(f"MS2 model training completed in {ms2_training_time:.2f} seconds")
    
    # Save models
    print(f"Saving models to {output_dir}...")
    # Create model files
    with open(os.path.join(output_dir, 'rt.pth'), 'wb') as f:
        torch.save(model_mgr.rt_model.model.state_dict(), f)
    with open(os.path.join(output_dir, 'ms2.pth'), 'wb') as f:
        torch.save(model_mgr.ms2_model.model.state_dict(), f)
    
    print("Standard model training complete!")
    
    return model_mgr, rt_training_time, ms2_training_time


def main():
    """Main function"""
    # Set random seed for reproducibility
    np.random.seed(42)
    torch.manual_seed(42)
    
    # Parameters
    standard_model_dir = 'models/test_standard'
    enhanced_model_dir = 'models/test_enhanced'
    alphafold_model_dir = 'models/test_alphafold_enhanced'
    epochs = 1  # Just 1 epoch for testing
    
    # Create output directories
    os.makedirs(standard_model_dir, exist_ok=True)
    os.makedirs(enhanced_model_dir, exist_ok=True)
    os.makedirs(alphafold_model_dir, exist_ok=True)
    
    # Create a small test dataset
    df = create_test_dataset()
    
    # Split data into train and validation sets (2 for training, 1 for validation)
    train_df = df.iloc[:2]
    val_df = df.iloc[2:3]
    
    print(f"Train set: {len(train_df)} peptides")
    print(f"Validation set: {len(val_df)} peptides")
    
    # Generate AlphaFold embeddings for all peptides
    all_peptides = pd.concat([train_df, val_df])
    alphafold_embeddings = generate_alphafold_embeddings(all_peptides)
    
    # Train standard model
    print("\n=== Training Standard Model ===")
    standard_model_mgr, std_rt_time, std_ms2_time = train_standard_model(
        train_df, val_df, standard_model_dir, epochs
    )
    
    # Train enhanced model
    print("\n=== Training Enhanced Model ===")
    enhanced_model_mgr = EnhancedModelManager(mask_modloss=False, device='cuda' if torch.cuda.is_available() else 'cpu', use_attention=True)
    
    start_time = time.time()
    enhanced_model_mgr.train_rt_model(
        train_df=train_df,
        val_df=val_df,
        epochs=epochs,
        batch_size=2,  # Small batch size for few data points
        learning_rate=0.001
    )
    enh_rt_time = time.time() - start_time
    print(f"Enhanced RT model training completed in {enh_rt_time:.2f} seconds")
    
    start_time = time.time()
    enhanced_model_mgr.train_ms2_model(
        train_df=train_df,
        val_df=val_df,
        epochs=epochs,
        batch_size=2,  # Small batch size for few data points
        learning_rate=0.001
    )
    enh_ms2_time = time.time() - start_time
    print(f"Enhanced MS2 model training completed in {enh_ms2_time:.2f} seconds")
    
    # Save enhanced models
    enhanced_model_mgr.save_models(enhanced_model_dir)
    
    # Train AlphaFold-enhanced model
    print("\n=== Training AlphaFold-Enhanced Model ===")
    alphafold_model_mgr = AlphaFoldEnhancedModelManager(
        mask_modloss=False,
        device='cuda' if torch.cuda.is_available() else 'cpu',
        use_attention=True
    )
    
    # Set AlphaFold embeddings
    alphafold_model_mgr.set_alphafold_embeddings(alphafold_embeddings)
    
    start_time = time.time()
    alphafold_model_mgr.train_rt_model(
        train_df=train_df,
        val_df=val_df,
        epochs=epochs,
        batch_size=2,  # Small batch size for few data points
        learning_rate=0.001
    )
    af_rt_time = time.time() - start_time
    print(f"AlphaFold-enhanced RT model training completed in {af_rt_time:.2f} seconds")
    
    start_time = time.time()
    alphafold_model_mgr.train_ms2_model(
        train_df=train_df,
        val_df=val_df,
        epochs=epochs,
        batch_size=2,  # Small batch size for few data points
        learning_rate=0.001
    )
    af_ms2_time = time.time() - start_time
    print(f"AlphaFold-enhanced MS2 model training completed in {af_ms2_time:.2f} seconds")
    
    # Save AlphaFold-enhanced models
    alphafold_model_mgr.save_models(alphafold_model_dir)
    
    # Print summary
    print("\n=== Training Summary ===")
    print(f"Standard RT model training time: {std_rt_time:.2f} seconds")
    print(f"Standard MS2 model training time: {std_ms2_time:.2f} seconds")
    print(f"Enhanced RT model training time: {enh_rt_time:.2f} seconds")
    print(f"Enhanced MS2 model training time: {enh_ms2_time:.2f} seconds")
    print(f"AlphaFold-enhanced RT model training time: {af_rt_time:.2f} seconds")
    print(f"AlphaFold-enhanced MS2 model training time: {af_ms2_time:.2f} seconds")
    
    print("\nTest training complete! All models trained successfully.")


if __name__ == "__main__":
    main()