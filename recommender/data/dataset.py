"""
Dataset module for the music recommender system.

This module provides dataset classes and functions for preparing data 
for model training and evaluation.
"""

import logging
import torch
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader

logger = logging.getLogger(__name__)

class MusicRecommenderDataset(Dataset):
    """Dataset for training and evaluating the music recommender model."""
    
    def __init__(self, data, user_features, track_features, categorical_mappings=None):
        """
        Initialize the dataset with data and feature lists.
        
        Args:
            data (pandas.DataFrame): DataFrame containing user-track interactions
            user_features (list): List of user feature column names
            track_features (list): List of track feature column names
            categorical_mappings (dict): Dictionary mapping categorical features to their ID mappings
        """
        self.data = data
        self.user_features = user_features
        self.track_features = track_features
        self.categorical_mappings = categorical_mappings or {}
        
        # Verify all required columns are present
        all_features = user_features + track_features + ['target']
        missing_features = [f for f in all_features if f not in data.columns]
        if missing_features:
            logger.warning(f"Missing feature columns: {missing_features}")
            raise ValueError(f"Dataset is missing required features: {missing_features}")
        
        # Identify numerical and categorical features
        self.numerical_features = []
        self.categorical_features = []
        
        for feature in user_features + track_features:
            # If it ends with _id, it's a categorical ID
            if feature.endswith('_id'):
                self.categorical_features.append(feature)
            # Check dtype of the column
            elif pd.api.types.is_numeric_dtype(data[feature]):
                self.numerical_features.append(feature)
            else:
                # Non-numeric columns without _id suffix are treated as categorical
                self.categorical_features.append(feature)
        
        logger.info(f"Created dataset with {len(data)} samples")
        logger.debug(f"Numerical features: {', '.join(self.numerical_features)}")
        logger.debug(f"Categorical features: {', '.join(self.categorical_features)}")
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        """
        Get a single item from the dataset.
        
        Args:
            idx (int): Index of the sample
            
        Returns:
            tuple: Tuple containing user features, track features, and target
        """
        row = self.data.iloc[idx]
        
        # Extract user features 
        user_data = {}
        for feature in self.user_features:
            if feature in self.categorical_features:
                # Handle categorical features as long tensors
                value = row[feature]
                if pd.isna(value):
                    value = 0  # Use 0 for missing categorical values
                elif isinstance(value, str):
                    try:
                        value = int(value)  # Try to convert string to int if possible
                    except ValueError:
                        value = 0  # Use default if conversion fails
                user_data[feature] = torch.tensor(value, dtype=torch.long)
            else:
                # Handle numerical features as float tensors
                value = row[feature]
                if pd.isna(value):
                    value = 0.0  # Use 0.0 for missing numerical values
                elif isinstance(value, str):
                    try:
                        value = float(value)  # Try to convert string to float if possible
                    except ValueError:
                        value = 0.0  # Use default if conversion fails
                user_data[feature] = torch.tensor(value, dtype=torch.float)
        
        # Extract track features
        track_data = {}
        for feature in self.track_features:
            if feature in self.categorical_features:
                # Handle categorical features as long tensors
                value = row[feature]
                if pd.isna(value):
                    value = 0  # Use 0 for missing categorical values
                elif isinstance(value, str):
                    try:
                        value = int(value)  # Try to convert string to int if possible
                    except ValueError:
                        value = 0  # Use default if conversion fails
                track_data[feature] = torch.tensor(value, dtype=torch.long)
            else:
                # Handle numerical features as float tensors
                value = row[feature]
                if pd.isna(value):
                    value = 0.0  # Use 0.0 for missing numerical values
                elif isinstance(value, str):
                    try:
                        value = float(value)  # Try to convert string to float if possible
                    except ValueError:
                        value = 0.0  # Use default if conversion fails
                track_data[feature] = torch.tensor(value, dtype=torch.float)
        
        # Extract target
        target = torch.tensor(row['target'], dtype=torch.float)
        
        return user_data, track_data, target
    
    @classmethod
    def get_feature_dimensions(cls, data, categorical_features):
        """
        Calculate dimensions for categorical features based on their unique values.
        
        Args:
            data (pandas.DataFrame): DataFrame to analyze
            categorical_features (list): List of categorical feature names
            
        Returns:
            dict: Dictionary mapping feature names to their dimensions
        """
        dimensions = {}
        for feature in categorical_features:
            # Check both the original feature and its ID version
            feature_id = f"{feature}_id"
            if feature_id in data.columns:
                dimensions[feature] = int(data[feature_id].max() + 1)
            elif feature in data.columns:
                dimensions[feature] = data[feature].nunique()
            else:
                logger.warning(f"Feature {feature} not found in data")
                
        return dimensions


def create_dataloader(data_path, user_features, track_features, batch_size=128, 
                      shuffle=True, categorical_mappings=None, num_workers=4):
    """
    Create a DataLoader for the music recommender dataset.
    
    Args:
        data_path (str): Path to the data CSV file
        user_features (list): List of user feature column names
        track_features (list): List of track feature column names
        batch_size (int): Batch size for the DataLoader
        shuffle (bool): Whether to shuffle the data
        categorical_mappings (dict): Dictionary mapping categorical features to their ID mappings
        num_workers (int): Number of worker processes for data loading
        
    Returns:
        DataLoader: PyTorch DataLoader for the dataset
    """
    try:
        # Load the data
        logger.info(f"Loading data from {data_path}")
        data = pd.read_csv(data_path)
        
        # Create the dataset
        dataset = MusicRecommenderDataset(
            data=data,
            user_features=user_features,
            track_features=track_features,
            categorical_mappings=categorical_mappings
        )
        
        # Create the DataLoader
        dataloader = DataLoader(
            dataset=dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=True,
        )
        
        logger.info(f"Created DataLoader with batch size {batch_size}")
        return dataloader
        
    except Exception as e:
        logger.error(f"Error creating DataLoader: {str(e)}")
        raise


def collate_batch(batch):
    """
    Custom collate function for batching data.
    
    Args:
        batch (list): List of samples from the dataset
        
    Returns:
        tuple: Tuple containing batched user features, track features, and targets
    """
    user_data, track_data, targets = zip(*batch)
    
    # Batch user features
    user_batch = {}
    for key in user_data[0].keys():
        user_batch[key] = torch.stack([item[key] for item in user_data])
    
    # Batch track features
    track_batch = {}
    for key in track_data[0].keys():
        track_batch[key] = torch.stack([item[key] for item in track_data])
    
    # Batch targets
    targets_batch = torch.stack(targets)
    
    return user_batch, track_batch, targets_batch 