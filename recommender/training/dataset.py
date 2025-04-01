"""
Dataset implementation for the music recommender system.

This module contains the dataset class for loading and processing music recommendation data
for training and evaluation.
"""

import pandas as pd
import torch
import numpy as np
import logging
from typing import Dict, List, Tuple, Optional

logger = logging.getLogger(__name__)


class MusicRecommenderDataset(torch.utils.data.Dataset):
    """
    Dataset class for the music recommender system.
    
    This class handles loading and preprocessing of data for training and evaluation
    of the music recommender model.
    """
    
    def __init__(self, 
                 data_path: str,
                 user_features: List[str],
                 track_features: List[str],
                 target_col: str = 'liked',
                 categorical_columns: Optional[List[str]] = None,
                 categorical_mappings: Optional[Dict] = None):
        """
        Initialize the dataset.
        
        Args:
            data_path: Path to the preprocessed data file
            user_features: List of user feature column names
            track_features: List of track feature column names
            target_col: Name of the target column
            categorical_columns: List of categorical column names
            categorical_mappings: Dictionary mapping categorical features to their values
        """
        self.data_path = data_path
        self.user_features = user_features
        self.track_features = track_features
        self.target_col = target_col
        self.categorical_columns = categorical_columns or []
        self.categorical_mappings = categorical_mappings or {}
        
        # Load data
        self.data = self._load_data()
        
        # Verify all required columns are present
        self._verify_columns()
        
        # Convert categorical columns to numeric if mappings provided
        self._convert_categorical_columns()
        
        # Log dataset info
        self._log_dataset_info()
    
    def _load_data(self) -> pd.DataFrame:
        """
        Load the dataset from file.
        
        Returns:
            DataFrame containing the dataset
        """
        try:
            logger.info(f"Loading dataset from {self.data_path}")
            data = pd.read_csv(self.data_path)
            return data
        except Exception as e:
            logger.error(f"Failed to load dataset from {self.data_path}: {str(e)}")
            raise
    
    def _verify_columns(self):
        """
        Verify that all required columns are present in the dataset.
        """
        # Check if all required columns are in the dataset
        required_columns = self.user_features + self.track_features + [self.target_col]
        missing_columns = [col for col in required_columns if col not in self.data.columns]
        
        if missing_columns:
            available_columns = self.data.columns.tolist()
            logger.error(f"Missing required columns: {missing_columns}")
            logger.error(f"Available columns: {available_columns}")
            raise ValueError(f"Dataset is missing required columns: {missing_columns}")
        
        logger.info("All required columns are present in the dataset")
    
    def _convert_categorical_columns(self):
        """
        Convert categorical columns to numeric using provided mappings.
        """
        for col in self.categorical_columns:
            if col in self.data.columns:
                if col in self.categorical_mappings:
                    # Use the provided mapping
                    mapping = self.categorical_mappings[col]
                    self.data[col] = self.data[col].map(mapping).fillna(0).astype(int)
                    logger.info(f"Converted categorical column {col} using provided mapping")
                else:
                    # Create a new mapping
                    unique_values = self.data[col].unique()
                    mapping = {val: i+1 for i, val in enumerate(unique_values)}
                    self.categorical_mappings[col] = mapping
                    self.data[col] = self.data[col].map(mapping).fillna(0).astype(int)
                    logger.info(f"Created mapping for categorical column {col}")
    
    def _log_dataset_info(self):
        """
        Log information about the dataset.
        """
        n_samples = len(self.data)
        n_user_features = len(self.user_features)
        n_track_features = len(self.track_features)
        n_positive = self.data[self.target_col].sum()
        n_negative = n_samples - n_positive
        
        logger.info(f"Dataset loaded with {n_samples} samples")
        logger.info(f"User features: {n_user_features}, Track features: {n_track_features}")
        logger.info(f"Positive samples: {n_positive}, Negative samples: {n_negative}")
    
    def __len__(self) -> int:
        """
        Return the number of samples in the dataset.
        
        Returns:
            Number of samples
        """
        return len(self.data)
    
    def __getitem__(self, idx: int) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor], torch.Tensor]:
        """
        Get a sample from the dataset.
        
        Args:
            idx: Index of the sample
            
        Returns:
            Tuple of (user_features, track_features, target)
        """
        if idx >= len(self.data):
            raise IndexError(f"Index {idx} out of range for dataset with {len(self.data)} samples")
        
        # Get the row at the specified index
        row = self.data.iloc[idx]
        
        # Extract user features
        user_features = {}
        for feat in self.user_features:
            # Make sure the feature exists
            if feat in row:
                # Convert to float tensor and handle NaN values
                value = row[feat]
                if pd.isna(value):
                    value = 0.0
                
                # Handle non-numeric values
                if isinstance(value, (str, bool)):
                    if feat in self.categorical_mappings:
                        # Convert using mapping
                        value = self.categorical_mappings[feat].get(value, 0)
                    else:
                        # Skip this feature if it's string and no mapping
                        logger.warning(f"Feature {feat} has string value but no mapping provided")
                        continue
                
                # Add to features dict, ensuring it's a float
                try:
                    user_features[feat] = torch.tensor(float(value), dtype=torch.float32).unsqueeze(0)
                except (ValueError, TypeError) as e:
                    logger.warning(f"Could not convert {feat}={value} ({type(value)}) to tensor: {e}")
                    # Use a default value of 0
                    user_features[feat] = torch.tensor(0.0, dtype=torch.float32).unsqueeze(0)
        
        # Extract track features
        track_features = {}
        for feat in self.track_features:
            # Make sure the feature exists
            if feat in row:
                # Convert to float tensor and handle NaN values
                value = row[feat]
                if pd.isna(value):
                    value = 0.0
                
                # Handle non-numeric values
                if isinstance(value, (str, bool)):
                    if feat in self.categorical_mappings:
                        # Convert using mapping
                        value = self.categorical_mappings[feat].get(value, 0)
                    else:
                        # Skip this feature if it's string and no mapping
                        logger.warning(f"Feature {feat} has string value but no mapping provided")
                        continue
                
                # Add to features dict, ensuring it's a float
                try:
                    track_features[feat] = torch.tensor(float(value), dtype=torch.float32).unsqueeze(0)
                except (ValueError, TypeError) as e:
                    logger.warning(f"Could not convert {feat}={value} ({type(value)}) to tensor: {e}")
                    # Use a default value of 0
                    track_features[feat] = torch.tensor(0.0, dtype=torch.float32).unsqueeze(0)
        
        # Extract target
        target_value = row[self.target_col]
        if pd.isna(target_value):
            target_value = 0.0
        
        # Ensure target is a float tensor with shape [1] for BCEWithLogitsLoss
        target = torch.tensor(float(target_value), dtype=torch.float32).unsqueeze(0)
        
        return user_features, track_features, target
    
    def get_feature_dimensions(self) -> Dict[str, int]:
        """
        Get the dimensions of categorical features for embedding layers.
        
        Returns:
            Dictionary mapping categorical feature names to their cardinality
        """
        feature_dims = {}
        
        for col in self.categorical_columns:
            if col in self.data.columns:
                # Number of unique values + 1 for unknown
                feature_dims[col] = int(self.data[col].max()) + 1
        
        return feature_dims 