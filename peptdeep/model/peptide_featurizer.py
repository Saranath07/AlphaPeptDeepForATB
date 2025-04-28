import numpy as np
import pandas as pd
import torch
from typing import List, Union, Dict

from peptdeep.model.featurize import (
    get_batch_aa_indices,
    get_batch_mod_feature,
    parse_instrument_indices
)

class PeptideFeaturizer:
    """
    Class for featurizing peptide data for deep learning models
    
    This class provides methods to convert peptide sequences, modifications,
    and other properties into numerical features suitable for deep learning models.
    """
    
    def __init__(self, device='cpu'):
        """
        Initialize the peptide featurizer
        
        Parameters
        ----------
        device : str
            Device to use for tensor operations ('cpu' or 'cuda')
        """
        self.device = device
    
    def featurize(self, df: pd.DataFrame, batch_size: int = 1024) -> dict:
        """
        Convert peptide DataFrame to tensor features
        
        Parameters
        ----------
        df : pd.DataFrame
            DataFrame with peptide data
        batch_size : int
            Batch size for processing
            
        Returns
        -------
        dict
            Dictionary with tensor features
        """
        # Extract features
        aa_indices = self._get_aa_indices(df)
        ptm_indices = self._get_ptm_indices(df)
        lengths = torch.tensor(df['nAA'].values, dtype=torch.long, device=self.device)
        
        # Create batch data
        batch_data = {
            'aa_indices': aa_indices,
            'ptm_indices': ptm_indices,
            'lengths': lengths
        }
        
        # Add charge and NCE if available
        if 'charge' in df.columns:
            batch_data['charge'] = torch.tensor(df['charge'].values, dtype=torch.float32, device=self.device)
        
        if 'nce' in df.columns:
            batch_data['nce'] = torch.tensor(df['nce'].values, dtype=torch.float32, device=self.device)
        
        # Add instrument indices if available
        if 'instrument' in df.columns:
            batch_data['instrument_indices'] = torch.tensor(
                parse_instrument_indices(df['instrument']), 
                dtype=torch.long, 
                device=self.device
            )
        
        return batch_data
    
    def _get_aa_indices(self, df: pd.DataFrame) -> torch.Tensor:
        """
        Get amino acid indices from peptide sequences
        
        Parameters
        ----------
        df : pd.DataFrame
            DataFrame with peptide data
            
        Returns
        -------
        torch.Tensor
            Tensor with amino acid indices
        """
        aa_indices = get_batch_aa_indices(df['sequence'].values)
        return torch.tensor(aa_indices, dtype=torch.long, device=self.device)
    
    def _get_ptm_indices(self, df: pd.DataFrame) -> torch.Tensor:
        """
        Get PTM indices from peptide modifications
        
        Parameters
        ----------
        df : pd.DataFrame
            DataFrame with peptide data
            
        Returns
        -------
        torch.Tensor
            Tensor with PTM indices
        """
        ptm_features = get_batch_mod_feature(df)
        return torch.tensor(ptm_features, dtype=torch.float32, device=self.device)