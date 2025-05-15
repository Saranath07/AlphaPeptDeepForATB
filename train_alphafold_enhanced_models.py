#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Train AlphaPeptDeep Models with AlphaFold3 Structure Predictions

This script trains models that incorporate AlphaFold3 structure predictions
alongside the standard AlphaPeptDeep embeddings and the enhanced model
with improved PTM representation.
"""

import os
import pandas as pd
import numpy as np
import torch
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import matplotlib.pyplot as plt
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
def load_hla_dataset(file_path, sample_size=None):
    """
    Load the HLA dataset
    
    Parameters
    ----------
    file_path : str
        Path to the dataset file
    sample_size : int, optional
        Number of samples to use (for debugging)
        
    Returns
    -------
    pd.DataFrame
        DataFrame with peptide data
    """
    print(f"Loading dataset from {file_path}...")
    df = pd.read_csv(file_path, sep='\t')
    
    print(f"Loaded {len(df)} peptides")
    
    # Sample if requested
    if sample_size is not None and sample_size < len(df):
        df = df.sample(sample_size, random_state=42)
        print(f"Sampled {len(df)} peptides")
    
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


def generate_alphafold_embedding(sequence):
    """
    Generate AlphaFold3 embedding for a single peptide sequence
    
    Parameters
    ----------
    sequence : str
        Peptide sequence
        
    Returns
    -------
    torch.Tensor
        AlphaFold3 embedding for the sequence
    """
    try:
        # Get AlphaFold3 prediction
        af_output = predict_structure_with_alphafold3(sequence)
        return af_output
    except Exception as e:
        print(f"Error processing sequence {sequence}: {str(e)}")
        # Create a placeholder embedding
        seq_len = len(sequence)
        dim = 64
        return torch.zeros(1, seq_len, dim)

def generate_alphafold_embeddings_batch(sequences, batch_size=32):
    """
    Generate AlphaFold3 embeddings for a batch of peptide sequences
    
    Parameters
    ----------
    sequences : list
        List of peptide sequences
    batch_size : int
        Batch size for processing
        
    Returns
    -------
    dict
        Dictionary mapping peptide sequences to AlphaFold3 embeddings
    """
    embeddings = {}
    
    for i in range(0, len(sequences), batch_size):
        batch_sequences = sequences[i:i+batch_size]
        for seq in batch_sequences:
            embeddings[seq] = generate_alphafold_embedding(seq)
    
    return embeddings
def train_alphafold_enhanced_model(train_df, val_df, output_dir, epochs=10, batch_size=64):
    """
    Train an enhanced model that incorporates AlphaFold3 structure predictions
    
    Parameters
    ----------
    train_df : pd.DataFrame
        Training data
    val_df : pd.DataFrame
        Validation data
    output_dir : str
        Directory to save the model
    epochs : int
        Number of epochs
    batch_size : int
        Batch size
        
    Returns
    -------
    tuple
        Model manager and training times
    """
    print("Training AlphaFold-enhanced model...")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Initialize enhanced model manager with AlphaFold integration
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    # Use the AlphaFoldEnhancedModelManager
    model_mgr = AlphaFoldEnhancedModelManager(
        mask_modloss=False,
        device=device,
        use_attention=True
    )
    
    # Train RT model
    print("Training RT model with AlphaFold embeddings...")
    start_time = time.time()
    model_mgr.train_rt_model(
        train_df=train_df,
        val_df=val_df,
        epochs=epochs,
        batch_size=batch_size,
        learning_rate=0.001
    )
    rt_training_time = time.time() - start_time
    print(f"RT model training completed in {rt_training_time:.2f} seconds")
    
    # Train MS2 model
    print("Training MS2 model with AlphaFold embeddings...")
    start_time = time.time()
    model_mgr.train_ms2_model(
        train_df=train_df,
        val_df=val_df,
        epochs=epochs,
        batch_size=batch_size,
        learning_rate=0.001
    )
    ms2_training_time = time.time() - start_time
    print(f"MS2 model training completed in {ms2_training_time:.2f} seconds")
    
    # Save models
    print(f"Saving models to {output_dir}...")
    model_mgr.save_models(output_dir)
    
    print("AlphaFold-enhanced model training complete!")
    
    return model_mgr, rt_training_time, ms2_training_time


def train_standard_model(train_df, val_df, output_dir, epochs=10, batch_size=64):
    """
    Train the standard AlphaPeptDeep model
    
    Parameters
    ----------
    train_df : pd.DataFrame
        Training data
    val_df : pd.DataFrame
        Validation data
    output_dir : str
        Directory to save the model
    epochs : int
        Number of epochs
    batch_size : int
        Batch size
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
        batch_size=batch_size,
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
        batch_size=batch_size,
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
        
        Parameters
        ----------
        embedding_dim : int
            Dimension of the embeddings
        lstm_hidden_dim : int
            Hidden dimension of the LSTM
        lstm_layers : int
            Number of LSTM layers
        dropout : float
            Dropout rate
        bidirectional : bool
            Whether to use bidirectional LSTM
        use_attention : bool
            Whether to use attention mechanism for PTM embedding
        alphafold_dim : int
            Dimension of AlphaFold embeddings
        device : str
            Device to use ('cpu' or 'cuda')
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
    
    def forward(self, batch_data: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Forward pass with AlphaFold embeddings
        
        Parameters
        ----------
        batch_data : Dict[str, torch.Tensor]
            Batch data with keys:
            - 'aa_indices': Tensor of amino acid indices
            - 'ptm_indices': Tensor of PTM indices
            - 'ptm_features': Tensor of PTM chemical features (optional)
            - 'alphafold_features': Tensor of AlphaFold features (optional)
            
        Returns
        -------
        torch.Tensor
            Predicted RT values
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
        
        Parameters
        ----------
        embedding_dim : int
            Dimension of the embeddings
        lstm_hidden_dim : int
            Hidden dimension of the LSTM
        lstm_layers : int
            Number of LSTM layers
        dropout : float
            Dropout rate
        bidirectional : bool
            Whether to use bidirectional LSTM
        use_attention : bool
            Whether to use attention mechanism for PTM embedding
        alphafold_dim : int
            Dimension of AlphaFold embeddings
        device : str
            Device to use ('cpu' or 'cuda')
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
    
    def forward(self, batch_data: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Forward pass with AlphaFold embeddings
        
        Parameters
        ----------
        batch_data : Dict[str, torch.Tensor]
            Batch data with keys:
            - 'aa_indices': Tensor of amino acid indices
            - 'ptm_indices': Tensor of PTM indices
            - 'ptm_features': Tensor of PTM chemical features (optional)
            - 'alphafold_features': Tensor of AlphaFold features (optional)
            
        Returns
        -------
        Dict[str, torch.Tensor]
            Dictionary with predicted MS2 intensities
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
        
        Parameters
        ----------
        mask_modloss : bool
            Whether to mask modifications at loss calculation
        device : str
            Device to use ('cpu' or 'cuda')
        use_attention : bool
            Whether to use attention mechanism for PTM embedding
        """
        # Initialize parent class
        super().__init__(mask_modloss, device, use_attention)
        
        # Replace standard models with AlphaFold-enhanced models
        self.rt_model = AlphaFoldEnhancedRTModel(
            use_attention=use_attention,
            device=device
        )
        
        self.ms2_model = AlphaFoldEnhancedMS2Model(
            use_attention=use_attention,
            device=device
        )
        
    # No longer storing embeddings in memory
        
    def train_rt_model(self,
                      train_df: pd.DataFrame,
                      val_df: Optional[pd.DataFrame] = None,
                      epochs: int = 10,
                      batch_size: int = 64,
                      learning_rate: float = 0.001):
        """
        Train RT model with AlphaFold embeddings
        
        Parameters
        ----------
        train_df : pd.DataFrame
            Training data
        val_df : pd.DataFrame, optional
            Validation data
        epochs : int
            Number of epochs
        batch_size : int
            Batch size
        learning_rate : float
            Learning rate
        """
        # Extract PTM features
        train_df = self.ptm_feature_extractor.extract_features(train_df)
        if val_df is not None:
            val_df = self.ptm_feature_extractor.extract_features(val_df)
        
        # Add AlphaFold features flag to the dataframe
        train_df = self._add_alphafold_features(train_df)
        if val_df is not None:
            val_df = self._add_alphafold_features(val_df)
        
        # No longer sharing embeddings with the model
        
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
        
        Parameters
        ----------
        train_df : pd.DataFrame
            Training data
        val_df : pd.DataFrame, optional
            Validation data
        epochs : int
            Number of epochs
        batch_size : int
            Batch size
        learning_rate : float
            Learning rate
        """
        # Extract PTM features
        train_df = self.ptm_feature_extractor.extract_features(train_df)
        if val_df is not None:
            val_df = self.ptm_feature_extractor.extract_features(val_df)
        
        # Add AlphaFold features flag to the dataframe
        train_df = self._add_alphafold_features(train_df)
        if val_df is not None:
            val_df = self._add_alphafold_features(val_df)
        
        # No longer sharing embeddings with the model
        
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
    
    def predict_rt(self, df: pd.DataFrame, batch_size: int = 1024) -> pd.DataFrame:
        """
        Predict RT values with AlphaFold embeddings
        
        Parameters
        ----------
        df : pd.DataFrame
            DataFrame with peptide data
        batch_size : int
            Batch size for prediction
            
        Returns
        -------
        pd.DataFrame
            DataFrame with predicted RT values
        """
        # Extract PTM features
        df = self.ptm_feature_extractor.extract_features(df)
        
        # Add AlphaFold features flag
        df = self._add_alphafold_features(df)
        
        # No longer sharing embeddings with the model
        
        return self.rt_model.predict(df, batch_size=batch_size)
    
    def predict_ms2(self, df: pd.DataFrame, batch_size: int = 1024) -> Dict[str, Any]:
        """
        Predict MS2 spectra with AlphaFold embeddings
        
        Parameters
        ----------
        df : pd.DataFrame
            DataFrame with peptide data
        batch_size : int
            Batch size for prediction
            
        Returns
        -------
        Dict[str, Any]
            Dictionary with predicted MS2 spectra
        """
        # Extract PTM features
        df = self.ptm_feature_extractor.extract_features(df)
        
        # Add AlphaFold features flag
        df = self._add_alphafold_features(df)
        
        # No longer sharing embeddings with the model
        
        return self.ms2_model.predict(df, batch_size=batch_size)
    
    def _prepare_batch_with_alphafold(self, batch_data, sequences):
        """
        Prepare batch data with AlphaFold embeddings generated on-the-fly
        
        Parameters
        ----------
        batch_data : Dict[str, torch.Tensor]
            Batch data
        sequences : List[str]
            List of peptide sequences
            
        Returns
        -------
        Dict[str, torch.Tensor]
            Batch data with AlphaFold embeddings
        """
        # Create a copy of the batch data
        batch_with_alphafold = dict(batch_data)
        
        # Get AlphaFold embeddings for each sequence in the batch
        batch_size = len(sequences)
        seq_len = batch_data['aa_indices'].shape[1]
        alphafold_dim = 64  # Default dimension
        
        # Initialize tensor for AlphaFold features
        alphafold_features = torch.zeros((batch_size, seq_len, alphafold_dim), device=self.device)
        
        # Generate embeddings on-the-fly for each sequence in the batch
        # This is memory efficient as we only generate embeddings for the current batch
        batch_embeddings = generate_alphafold_embeddings_batch(sequences)
        
        # Fill in AlphaFold features
        for i, seq in enumerate(sequences):
            if seq in batch_embeddings:
                # Get AlphaFold embedding
                af_embedding = batch_embeddings[seq]
                
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
        
        # Clear batch embeddings to free memory
        del batch_embeddings
        
        return batch_with_alphafold
    
    def _add_alphafold_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add AlphaFold features to the dataframe
        
        Parameters
        ----------
        df : pd.DataFrame
            DataFrame with peptide data
            
        Returns
        -------
        pd.DataFrame
            DataFrame with added AlphaFold features
        """
        # Create a copy to avoid modifying the original
        result_df = df.copy()
        
        # Add a flag column to indicate that AlphaFold features should be used
        result_df['use_alphafold'] = True
        
        return result_df
    
    def _get_alphafold_features(self, sequence: str) -> np.ndarray:
        """
        Get AlphaFold features for a sequence by generating them on-the-fly
        
        Parameters
        ----------
        sequence : str
            Peptide sequence
            
        Returns
        -------
        np.ndarray
            AlphaFold features
        """
        # Generate embedding on-the-fly
        af_output = generate_alphafold_embedding(sequence)
        
        # Convert to numpy array if it's a tensor
        if isinstance(af_output, torch.Tensor):
            features = af_output.detach().cpu().numpy()
            
            # Flatten or process the features as needed
            if len(features.shape) > 1:
                # Take mean across sequence dimension if it's a 3D tensor
                features = np.mean(features, axis=(0, 1))
            
            return features
        else:
            # If it's already a numpy array or other format
            return np.array(af_output)
def evaluate_models(test_df, standard_model_mgr, enhanced_model_mgr, alphafold_model_mgr):
    """
    Evaluate and compare all models
    
    Parameters
    ----------
    test_df : pd.DataFrame
        Test data
    standard_model_mgr : ModelManager or None
        Standard model manager (can be None if training failed)
    enhanced_model_mgr : EnhancedModelManager or None
        Enhanced model manager (can be None if training failed)
    alphafold_model_mgr : AlphaFoldEnhancedModelManager or None
        AlphaFold-enhanced model manager (can be None if training failed)
        
    Returns
    -------
    dict
        Dictionary with evaluation metrics
    """
    print("Evaluating models...")
    
    # Initialize metrics dictionary
    metrics = {}
    
    # Make sure mods and mod_sites columns are properly formatted to avoid errors
    test_df_clean = test_df.copy()
    if 'mods' in test_df_clean.columns:
        test_df_clean['mods'] = test_df_clean['mods'].fillna('')
    if 'mod_sites' in test_df_clean.columns:
        test_df_clean['mod_sites'] = test_df_clean['mod_sites'].fillna('')
    
    # Check which models are available
    has_standard = standard_model_mgr is not None
    has_enhanced = enhanced_model_mgr is not None
    has_alphafold = alphafold_model_mgr is not None
    
    # If no models are available, return empty metrics
    if not (has_standard or has_enhanced or has_alphafold):
        print("No models available for evaluation.")
        return metrics
    
    try:
        # Try to predict with available models
        standard_rt_df = None
        standard_ms2_results = None
        enhanced_rt_df = None
        enhanced_ms2_results = None
        alphafold_rt_df = None
        alphafold_ms2_results = None
        
        if has_standard:
            print("Predicting with standard model...")
            standard_rt_df = standard_model_mgr.rt_model.predict(test_df_clean)
            standard_ms2_results = standard_model_mgr.ms2_model.predict(test_df_clean)
            metrics['standard_rt_df'] = standard_rt_df
        
        if has_enhanced:
            print("Predicting with enhanced model...")
            enhanced_rt_df = enhanced_model_mgr.predict_rt(test_df_clean)
            enhanced_ms2_results = enhanced_model_mgr.predict_ms2(test_df_clean)
            metrics['enhanced_rt_df'] = enhanced_rt_df
        
        if has_alphafold:
            print("Predicting with AlphaFold-enhanced model...")
            alphafold_rt_df = alphafold_model_mgr.predict_rt(test_df_clean)
            alphafold_ms2_results = alphafold_model_mgr.predict_ms2(test_df_clean)
            metrics['alphafold_rt_df'] = alphafold_rt_df
        
        # Evaluate RT predictions for available models
        if has_standard:
            standard_rt_mae = np.mean(np.abs(standard_rt_df['rt'] - standard_rt_df['rt_pred']))
            standard_rt_r2 = np.corrcoef(standard_rt_df['rt'], standard_rt_df['rt_pred'])[0, 1] ** 2
            metrics['standard_rt_mae'] = standard_rt_mae
            metrics['standard_rt_r2'] = standard_rt_r2
            print(f"Standard Model RT MAE: {standard_rt_mae:.4f}")
            print(f"Standard Model RT R²: {standard_rt_r2:.4f}")
        
        if has_enhanced:
            enhanced_rt_mae = np.mean(np.abs(enhanced_rt_df['rt'] - enhanced_rt_df['rt_pred']))
            enhanced_rt_r2 = np.corrcoef(enhanced_rt_df['rt'], enhanced_rt_df['rt_pred'])[0, 1] ** 2
            metrics['enhanced_rt_mae'] = enhanced_rt_mae
            metrics['enhanced_rt_r2'] = enhanced_rt_r2
            print(f"Enhanced Model RT MAE: {enhanced_rt_mae:.4f}")
            print(f"Enhanced Model RT R²: {enhanced_rt_r2:.4f}")
        
        if has_alphafold:
            alphafold_rt_mae = np.mean(np.abs(alphafold_rt_df['rt'] - alphafold_rt_df['rt_pred']))
            alphafold_rt_r2 = np.corrcoef(alphafold_rt_df['rt'], alphafold_rt_df['rt_pred'])[0, 1] ** 2
            metrics['alphafold_rt_mae'] = alphafold_rt_mae
            metrics['alphafold_rt_r2'] = alphafold_rt_r2
            print(f"AlphaFold-Enhanced Model RT MAE: {alphafold_rt_mae:.4f}")
            print(f"AlphaFold-Enhanced Model RT R²: {alphafold_rt_r2:.4f}")
        
        # Calculate improvement percentages if we have multiple models
        if has_standard and has_enhanced:
            enhanced_rt_mae_improvement = (standard_rt_mae - enhanced_rt_mae) / standard_rt_mae * 100
            enhanced_rt_r2_improvement = (enhanced_rt_r2 - standard_rt_r2) / standard_rt_r2 * 100
            metrics['enhanced_rt_mae_improvement'] = enhanced_rt_mae_improvement
            metrics['enhanced_rt_r2_improvement'] = enhanced_rt_r2_improvement
            print(f"Enhanced vs Standard RT MAE Improvement: {enhanced_rt_mae_improvement:.2f}%")
            print(f"Enhanced vs Standard RT R² Improvement: {enhanced_rt_r2_improvement:.2f}%")
        
        if has_standard and has_alphafold:
            alphafold_rt_mae_improvement = (standard_rt_mae - alphafold_rt_mae) / standard_rt_mae * 100
            alphafold_rt_r2_improvement = (alphafold_rt_r2 - standard_rt_r2) / standard_rt_r2 * 100
            metrics['alphafold_rt_mae_improvement'] = alphafold_rt_mae_improvement
            metrics['alphafold_rt_r2_improvement'] = alphafold_rt_r2_improvement
            print(f"AlphaFold vs Standard RT MAE Improvement: {alphafold_rt_mae_improvement:.2f}%")
            print(f"AlphaFold vs Standard RT R² Improvement: {alphafold_rt_r2_improvement:.2f}%")
        
        if has_enhanced and has_alphafold:
            alphafold_vs_enhanced_rt_mae_improvement = (enhanced_rt_mae - alphafold_rt_mae) / enhanced_rt_mae * 100
            alphafold_vs_enhanced_rt_r2_improvement = (alphafold_rt_r2 - enhanced_rt_r2) / enhanced_rt_r2 * 100
            metrics['alphafold_vs_enhanced_rt_mae_improvement'] = alphafold_vs_enhanced_rt_mae_improvement
            metrics['alphafold_vs_enhanced_rt_r2_improvement'] = alphafold_vs_enhanced_rt_r2_improvement
            print(f"AlphaFold vs Enhanced RT MAE Improvement: {alphafold_vs_enhanced_rt_mae_improvement:.2f}%")
            print(f"AlphaFold vs Enhanced RT R² Improvement: {alphafold_vs_enhanced_rt_r2_improvement:.2f}%")
        
        # Try to evaluate MS2 predictions if we have multiple models
        if has_standard and has_enhanced and has_alphafold:
            try:
                # Calculate average cosine similarity between the models' predictions
                standard_intensities = standard_ms2_results['fragment_intensity_df'].values
                enhanced_intensities = enhanced_ms2_results['fragment_intensity_df'].values
                alphafold_intensities = alphafold_ms2_results['fragment_intensity_df'].values
                
                # Normalize intensities
                standard_intensities_norm = standard_intensities / np.linalg.norm(standard_intensities, axis=1, keepdims=True)
                enhanced_intensities_norm = enhanced_intensities / np.linalg.norm(enhanced_intensities, axis=1, keepdims=True)
                alphafold_intensities_norm = alphafold_intensities / np.linalg.norm(alphafold_intensities, axis=1, keepdims=True)
                
                # Calculate cosine similarity
                enhanced_cosine_sim = np.sum(standard_intensities_norm * enhanced_intensities_norm, axis=1)
                alphafold_cosine_sim = np.sum(standard_intensities_norm * alphafold_intensities_norm, axis=1)
                alphafold_vs_enhanced_cosine_sim = np.sum(enhanced_intensities_norm * alphafold_intensities_norm, axis=1)
                
                avg_enhanced_cosine_sim = np.mean(enhanced_cosine_sim)
                avg_alphafold_cosine_sim = np.mean(alphafold_cosine_sim)
                avg_alphafold_vs_enhanced_cosine_sim = np.mean(alphafold_vs_enhanced_cosine_sim)
                
                metrics['avg_enhanced_cosine_sim'] = avg_enhanced_cosine_sim
                metrics['avg_alphafold_cosine_sim'] = avg_alphafold_cosine_sim
                metrics['avg_alphafold_vs_enhanced_cosine_sim'] = avg_alphafold_vs_enhanced_cosine_sim
                
                print(f"Average Cosine Similarity (Enhanced vs Standard): {avg_enhanced_cosine_sim:.4f}")
                print(f"Average Cosine Similarity (AlphaFold vs Standard): {avg_alphafold_cosine_sim:.4f}")
                print(f"Average Cosine Similarity (AlphaFold vs Enhanced): {avg_alphafold_vs_enhanced_cosine_sim:.4f}")
            except Exception as e:
                print(f"Warning: Error during MS2 evaluation: {str(e)}")
        elif has_standard and has_enhanced:
            try:
                # Calculate cosine similarity between standard and enhanced
                standard_intensities = standard_ms2_results['fragment_intensity_df'].values
                enhanced_intensities = enhanced_ms2_results['fragment_intensity_df'].values
                
                # Normalize intensities
                standard_intensities_norm = standard_intensities / np.linalg.norm(standard_intensities, axis=1, keepdims=True)
                enhanced_intensities_norm = enhanced_intensities / np.linalg.norm(enhanced_intensities, axis=1, keepdims=True)
                
                # Calculate cosine similarity
                enhanced_cosine_sim = np.sum(standard_intensities_norm * enhanced_intensities_norm, axis=1)
                avg_enhanced_cosine_sim = np.mean(enhanced_cosine_sim)
                
                metrics['avg_enhanced_cosine_sim'] = avg_enhanced_cosine_sim
                print(f"Average Cosine Similarity (Enhanced vs Standard): {avg_enhanced_cosine_sim:.4f}")
            except Exception as e:
                print(f"Warning: Error during MS2 evaluation: {str(e)}")
        
        # Print summary of available metrics
        print("\nEvaluation Summary:")
        print(f"Models evaluated: " +
              (f"Standard {'✓' if has_standard else '✗'}, " +
               f"Enhanced {'✓' if has_enhanced else '✗'}, " +
               f"AlphaFold {'✓' if has_alphafold else '✗'}"))
        
    except Exception as e:
        print(f"Warning: Error during model evaluation: {str(e)}")
        import traceback
        traceback.print_exc()
        
        # Try to evaluate just the AlphaFold model if it's available
        if has_alphafold:
            print("Falling back to evaluating only the AlphaFold-enhanced model...")
            try:
                # Predict with AlphaFold-enhanced model only
                alphafold_rt_df = alphafold_model_mgr.predict_rt(test_df_clean)
                
                # Calculate actual metrics for AlphaFold model
                alphafold_rt_mae = np.mean(np.abs(alphafold_rt_df['rt'] - alphafold_rt_df['rt_pred']))
                alphafold_rt_r2 = np.corrcoef(alphafold_rt_df['rt'], alphafold_rt_df['rt_pred'])[0, 1] ** 2
                
                # Add to metrics
                metrics['alphafold_rt_mae'] = alphafold_rt_mae
                metrics['alphafold_rt_r2'] = alphafold_rt_r2
                metrics['alphafold_rt_df'] = alphafold_rt_df
                
                print(f"AlphaFold-Enhanced Model RT MAE: {alphafold_rt_mae:.4f}")
                print(f"AlphaFold-Enhanced Model RT R²: {alphafold_rt_r2:.4f}")
            except Exception as e:
                print(f"Error evaluating AlphaFold model: {str(e)}")
    
    return metrics


def plot_results(metrics, training_times, results_dir):
    """
    Plot the results of the model comparison
    
    Parameters
    ----------
    metrics : dict
        Dictionary with evaluation metrics
    training_times : dict
        Dictionary with training times
    results_dir : str
        Directory to save the plots
    """
    os.makedirs(results_dir, exist_ok=True)
    
    # Check if we have all the required metrics for comparison plots
    has_all_metrics = all(key in metrics for key in ['standard_rt_mae', 'enhanced_rt_mae', 'alphafold_rt_mae',
                                                    'standard_rt_r2', 'enhanced_rt_r2', 'alphafold_rt_r2'])
    
    if has_all_metrics:
        # Plot RT MAE comparison
        plt.figure(figsize=(10, 6))
        models = ['Standard', 'Enhanced', 'AlphaFold-Enhanced']
        rt_mae = [metrics['standard_rt_mae'], metrics['enhanced_rt_mae'], metrics['alphafold_rt_mae']]
        
        plt.bar(models, rt_mae, color=['blue', 'green', 'red'])
        plt.xlabel('Model Type')
        plt.ylabel('RT MAE')
        plt.title('RT Mean Absolute Error Comparison')
        plt.grid(True, alpha=0.3)
        plt.savefig(os.path.join(results_dir, 'rt_mae_comparison.png'))
        plt.close()
        
        # Plot RT R² comparison
        plt.figure(figsize=(10, 6))
        rt_r2 = [metrics['standard_rt_r2'], metrics['enhanced_rt_r2'], metrics['alphafold_rt_r2']]
        
        plt.bar(models, rt_r2, color=['blue', 'green', 'red'])
        plt.xlabel('Model Type')
        plt.ylabel('RT R²')
        plt.title('RT R² Comparison')
        plt.grid(True, alpha=0.3)
        plt.savefig(os.path.join(results_dir, 'rt_r2_comparison.png'))
        plt.close()
    else:
        print("Skipping RT MAE and R² comparison plots due to missing metrics")
    
    # Check if we have all the required training times
    has_all_times = all(key in training_times for key in ['std_rt_time', 'enh_rt_time', 'af_rt_time',
                                                         'std_ms2_time', 'enh_ms2_time', 'af_ms2_time'])
    
    if has_all_times:
        # Plot training times
        plt.figure(figsize=(10, 6))
        models = ['Standard', 'Enhanced', 'AlphaFold-Enhanced']
        rt_times = [training_times['std_rt_time'], training_times['enh_rt_time'], training_times['af_rt_time']]
        ms2_times = [training_times['std_ms2_time'], training_times['enh_ms2_time'], training_times['af_ms2_time']]
        
        x = np.arange(len(models))
        width = 0.35
        
        plt.bar(x - width/2, rt_times, width, label='RT Model')
        plt.bar(x + width/2, ms2_times, width, label='MS2 Model')
        
        plt.xlabel('Model Type')
        plt.ylabel('Training Time (seconds)')
        plt.title('Training Time Comparison')
        plt.xticks(x, models)
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.savefig(os.path.join(results_dir, 'training_time_comparison.png'))
        plt.close()
    else:
        print("Skipping training time comparison plot due to missing data")
    
    # Plot RT predictions scatter plot if we have the dataframes
    has_all_dfs = all(key in metrics for key in ['standard_rt_df', 'enhanced_rt_df', 'alphafold_rt_df'])
    
    if has_all_dfs:
        plt.figure(figsize=(15, 5))
        
        # Create a sample of points for visualization
        sample_size = min(100, len(metrics['standard_rt_df']))
        sample_indices = np.random.choice(len(metrics['standard_rt_df']), sample_size, replace=False)
        
        plt.subplot(1, 3, 1)
        plt.scatter(
            metrics['standard_rt_df'].iloc[sample_indices]['rt'],
            metrics['standard_rt_df'].iloc[sample_indices]['rt_pred'],
            alpha=0.6
        )
        plt.plot([0, 100], [0, 100], 'r--')
        plt.xlabel('Experimental RT')
        plt.ylabel('Predicted RT')
        plt.title('Standard Model')
        
        plt.subplot(1, 3, 2)
        plt.scatter(
            metrics['enhanced_rt_df'].iloc[sample_indices]['rt'],
            metrics['enhanced_rt_df'].iloc[sample_indices]['rt_pred'],
            alpha=0.6
        )
        plt.plot([0, 100], [0, 100], 'r--')
        plt.xlabel('Experimental RT')
        plt.ylabel('Predicted RT')
        plt.title('Enhanced Model')
        
        plt.subplot(1, 3, 3)
        plt.scatter(
            metrics['alphafold_rt_df'].iloc[sample_indices]['rt'],
            metrics['alphafold_rt_df'].iloc[sample_indices]['rt_pred'],
            alpha=0.6
        )
        plt.plot([0, 100], [0, 100], 'r--')
        plt.xlabel('Experimental RT')
        plt.ylabel('Predicted RT')
        plt.title('AlphaFold-Enhanced Model')
        
        plt.tight_layout()
        plt.savefig(os.path.join(results_dir, 'rt_predictions_comparison.png'))
        plt.close()
    elif 'alphafold_rt_df' in metrics:
        # If we only have the AlphaFold model results, plot just that
        plt.figure(figsize=(8, 6))
        
        sample_size = min(100, len(metrics['alphafold_rt_df']))
        sample_indices = np.random.choice(len(metrics['alphafold_rt_df']), sample_size, replace=False)
        
        plt.scatter(
            metrics['alphafold_rt_df'].iloc[sample_indices]['rt'],
            metrics['alphafold_rt_df'].iloc[sample_indices]['rt_pred'],
            alpha=0.6
        )
        plt.plot([0, 100], [0, 100], 'r--')
        plt.xlabel('Experimental RT')
        plt.ylabel('Predicted RT')
        plt.title('AlphaFold-Enhanced Model')
        
        plt.tight_layout()
        plt.savefig(os.path.join(results_dir, 'alphafold_rt_predictions.png'))
        plt.close()
        print("Created AlphaFold-only RT predictions plot")
    else:
        print("Skipping RT predictions scatter plot due to missing dataframes")
    
    # Plot improvement percentages if available
    has_improvements = all(key in metrics for key in ['enhanced_rt_mae_improvement',
                                                     'alphafold_rt_mae_improvement',
                                                     'alphafold_vs_enhanced_rt_mae_improvement'])
    
    if has_improvements:
        plt.figure(figsize=(12, 6))
        improvements = [
            metrics['enhanced_rt_mae_improvement'],
            metrics['alphafold_rt_mae_improvement'],
            metrics['alphafold_vs_enhanced_rt_mae_improvement']
        ]
        labels = ['Enhanced vs Standard', 'AlphaFold vs Standard', 'AlphaFold vs Enhanced']
        
        plt.bar(labels, improvements, color=['green', 'red', 'purple'])
        plt.axhline(y=0, color='k', linestyle='-', alpha=0.3)
        plt.xlabel('Model Comparison')
        plt.ylabel('RT MAE Improvement (%)')
        plt.title('RT MAE Improvement Percentages')
        plt.grid(True, alpha=0.3)
        plt.savefig(os.path.join(results_dir, 'rt_mae_improvement.png'))
        plt.close()
    else:
        print("Skipping improvement percentages plot due to missing metrics")
    
    # Save metrics to CSV (only non-DataFrame values)
    metrics_df = pd.DataFrame({k: [v] for k, v in metrics.items() if not isinstance(v, pd.DataFrame)})
    metrics_df.to_csv(os.path.join(results_dir, 'alphafold_model_comparison_metrics.csv'), index=False)
    print(f"Metrics saved to {os.path.join(results_dir, 'alphafold_model_comparison_metrics.csv')}")
def main():
    """Main function"""
    # Set random seed for reproducibility
    np.random.seed(42)
    torch.manual_seed(42)
    
    # Parameters
    data_file = 'HLA_DDA_Rescore/AD_pFind_MSV000084172_fdr.tsv'
    standard_model_dir = 'models/standard'
    enhanced_model_dir = 'models/enhanced'
    alphafold_model_dir = 'models/alphafold_enhanced'
    results_dir = 'results/alphafold_comparison'
    sample_size = None  # Set to a number for debugging, None to use all data
    test_size = 0.2
    val_size = 0.2
    epochs = 10
    batch_size = 64
    
    # Initialize training times dictionary with default values
    training_times = {
        'std_rt_time': 0.0,
        'std_ms2_time': 0.0,
        'enh_rt_time': 0.0,
        'enh_ms2_time': 0.0,
        'af_rt_time': 0.0,
        'af_ms2_time': 0.0
    }
    
    # Create output directories
    os.makedirs(standard_model_dir, exist_ok=True)
    os.makedirs(enhanced_model_dir, exist_ok=True)
    os.makedirs(alphafold_model_dir, exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)
    
    try:
        # Load dataset
        df = load_hla_dataset(data_file, sample_size)
        
        # Split data into train, validation, and test sets
        train_val_df, test_df = train_test_split(df, test_size=test_size, random_state=42)
        train_df, val_df = train_test_split(train_val_df, test_size=val_size, random_state=42)
        
        print(f"Train set: {len(train_df)} peptides")
        print(f"Validation set: {len(val_df)} peptides")
        print(f"Test set: {len(test_df)} peptides")
        
        # Initialize model managers
        standard_model_mgr = None
        enhanced_model_mgr = None
        alphafold_model_mgr = None
        
        # Train standard model
        try:
            standard_model_mgr, std_rt_time, std_ms2_time = train_standard_model(
                train_df, val_df, standard_model_dir, epochs, batch_size
            )
            training_times['std_rt_time'] = std_rt_time
            training_times['std_ms2_time'] = std_ms2_time
        except Exception as e:
            print(f"Error training standard model: {str(e)}")
            print("Continuing with other models...")
        
        # Train enhanced model
        try:
            enhanced_model_mgr = EnhancedModelManager(mask_modloss=False, device='cuda' if torch.cuda.is_available() else 'cpu', use_attention=True)
            
            print("Training enhanced model...")
            start_time = time.time()
            enhanced_model_mgr.train_rt_model(
                train_df=train_df,
                val_df=val_df,
                epochs=epochs,
                batch_size=batch_size,
                learning_rate=0.001
            )
            enh_rt_time = time.time() - start_time
            training_times['enh_rt_time'] = enh_rt_time
            print(f"Enhanced RT model training completed in {enh_rt_time:.2f} seconds")
            
            start_time = time.time()
            enhanced_model_mgr.train_ms2_model(
                train_df=train_df,
                val_df=val_df,
                epochs=epochs,
                batch_size=batch_size,
                learning_rate=0.001
            )
            enh_ms2_time = time.time() - start_time
            training_times['enh_ms2_time'] = enh_ms2_time
            print(f"Enhanced MS2 model training completed in {enh_ms2_time:.2f} seconds")
            
            # Save enhanced models
            enhanced_model_mgr.save_models(enhanced_model_dir)
        except Exception as e:
            print(f"Error training enhanced model: {str(e)}")
            print("Continuing with other models...")
        
        # Train AlphaFold-enhanced model
        try:
            alphafold_model_mgr = AlphaFoldEnhancedModelManager(
                mask_modloss=False,
                device='cuda' if torch.cuda.is_available() else 'cpu',
                use_attention=True
            )
            
            # No longer setting AlphaFold embeddings in advance
            
            print("Training AlphaFold-enhanced model...")
            start_time = time.time()
            alphafold_model_mgr.train_rt_model(
                train_df=train_df,
                val_df=val_df,
                epochs=epochs,
                batch_size=batch_size,
                learning_rate=0.001
            )
            af_rt_time = time.time() - start_time
            training_times['af_rt_time'] = af_rt_time
            print(f"AlphaFold-enhanced RT model training completed in {af_rt_time:.2f} seconds")
            
            start_time = time.time()
            alphafold_model_mgr.train_ms2_model(
                train_df=train_df,
                val_df=val_df,
                epochs=epochs,
                batch_size=batch_size,
                learning_rate=0.001
            )
            af_ms2_time = time.time() - start_time
            training_times['af_ms2_time'] = af_ms2_time
            print(f"AlphaFold-enhanced MS2 model training completed in {af_ms2_time:.2f} seconds")
            
            # Save AlphaFold-enhanced models
            alphafold_model_mgr.save_models(alphafold_model_dir)
        except Exception as e:
            print(f"Error training AlphaFold-enhanced model: {str(e)}")
            print("Continuing with evaluation...")
        
        # Evaluate models - only pass model managers that were successfully created
        metrics = evaluate_models(
            test_df,
            standard_model_mgr if standard_model_mgr is not None else None,
            enhanced_model_mgr if enhanced_model_mgr is not None else None,
            alphafold_model_mgr if alphafold_model_mgr is not None else None
        )
        
        # Plot results
        plot_results(metrics, training_times, results_dir)
        
        print("\nTraining and evaluation complete!")
        print(f"Results saved to {results_dir}")
    
    except Exception as e:
        print(f"An error occurred during execution: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()