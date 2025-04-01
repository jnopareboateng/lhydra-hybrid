import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset
import os
import sys

# Add root directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.logger import LhydraLogger, log_function

class MusicRecommendationDataset(Dataset):
    """
    PyTorch Dataset for the Lhydra Hybrid Music Recommender System.
    Handles loading and preparing data for training and inference.
    """
    
    def __init__(self, data_frame, user_id_col='user_id', track_id_col='track_id', 
                 target_col='high_engagement', mode='train', logger=None):
        """
        Initialize the dataset.
        
        Args:
            data_frame (pd.DataFrame): DataFrame containing the processed data.
            user_id_col (str): Name of user ID column.
            track_id_col (str): Name of track ID column.
            target_col (str): Name of target column.
            mode (str): 'train', 'val', or 'test'.
            logger (LhydraLogger): Logger instance.
        """
        self.logger = logger or LhydraLogger()
        self.logger.info(f"Initializing MusicRecommendationDataset in {mode} mode")
        
        self.df = data_frame
        self.user_id_col = user_id_col
        self.track_id_col = track_id_col
        self.target_col = target_col
        self.mode = mode
        
        # Get feature columns (all columns except target)
        self.feature_cols = [col for col in self.df.columns 
                           if col != self.target_col]
        
        # Log dataset stats
        self.logger.log_data_stats(f"Dataset ({mode})", self.df.shape,
                                 feature_names=self.feature_cols)
    
    @log_function()
    def __len__(self):
        """Return the number of samples in the dataset."""
        return len(self.df)
    
    @log_function()
    def __getitem__(self, idx):
        """
        Get a sample from the dataset.
        
        Args:
            idx (int): Index of the sample.
            
        Returns:
            dict: Dictionary containing feature tensors and target.
        """
        # Get the row at the specified index
        row = self.df.iloc[idx]
        
        # Extract features
        # User features
        user_features = {}
        if self.user_id_col in self.df.columns:
            user_features['user_id'] = torch.tensor(row[self.user_id_col], dtype=torch.long)
        
        # Demographic features (if available)
        demographic_cols = []
        for col in self.df.columns:
            if col.startswith(('gender_', 'age_', 'region_', 'country_', 'monthly_hours')):
                demographic_cols.append(col)
        
        if demographic_cols:
            user_features['demographics'] = torch.tensor(
                row[demographic_cols].values, dtype=torch.float32
            )
        
        # Audio profile features (average audio preferences)
        audio_profile_cols = [col for col in self.df.columns if col.startswith('avg_')]
        if audio_profile_cols:
            user_features['audio_profile'] = torch.tensor(
                row[audio_profile_cols].values, dtype=torch.float32
            )
        
        # Item features
        item_features = {}
        if self.track_id_col in self.df.columns:
            item_features['track_id'] = torch.tensor(row[self.track_id_col], dtype=torch.long)
        
        # Artist feature (if available)
        if 'artist' in self.df.columns:
            item_features['artist'] = torch.tensor(row['artist'], dtype=torch.long)
        
        # Genre features
        genre_cols = [col for col in self.df.columns if col.startswith('main_genre_')]
        if genre_cols:
            item_features['genre'] = torch.tensor(
                row[genre_cols].values, dtype=torch.float32
            )
        
        # Audio features
        audio_cols = [col for col in self.df.columns 
                     if col in ['danceability', 'energy', 'key', 'loudness', 'mode',
                               'speechiness', 'acousticness', 'instrumentalness',
                               'liveness', 'valence', 'tempo', 'time_signature']]
        if audio_cols:
            item_features['audio_features'] = torch.tensor(
                row[audio_cols].values, dtype=torch.float32
            )
        
        # Temporal features
        temporal_cols = [col for col in self.df.columns 
                        if col in ['year', 'song_age', 'is_recent']]
        if temporal_cols:
            item_features['temporal'] = torch.tensor(
                row[temporal_cols].values, dtype=torch.float32
            )
        
        # Create target tensor if in training mode
        if self.mode != 'inference' and self.target_col in self.df.columns:
            target = torch.tensor(row[self.target_col], dtype=torch.float32)
        else:
            target = torch.tensor(0.0, dtype=torch.float32)  # Dummy value for inference
        
        return {
            'user_features': user_features,
            'item_features': item_features,
            'target': target
        }
    
    @log_function()
    def get_feature_dims(self):
        """
        Get dimensions of each feature group for model architecture.
        
        Returns:
            dict: Dictionary with feature dimensions.
        """
        feature_dims = {
            'user': {},
            'item': {}
        }
        
        # User dimensions
        demographic_cols = [col for col in self.df.columns 
                          if col.startswith(('gender_', 'age_', 'region_', 'country_', 'monthly_hours'))]
        if demographic_cols:
            feature_dims['user']['demographics'] = len(demographic_cols)
        
        audio_profile_cols = [col for col in self.df.columns if col.startswith('avg_')]
        if audio_profile_cols:
            feature_dims['user']['audio_profile'] = len(audio_profile_cols)
        
        # Item dimensions
        genre_cols = [col for col in self.df.columns if col.startswith('main_genre_')]
        if genre_cols:
            feature_dims['item']['genre'] = len(genre_cols)
        
        audio_cols = [col for col in self.df.columns 
                     if col in ['danceability', 'energy', 'key', 'loudness', 'mode',
                               'speechiness', 'acousticness', 'instrumentalness',
                               'liveness', 'valence', 'tempo', 'time_signature']]
        if audio_cols:
            feature_dims['item']['audio_features'] = len(audio_cols)
        
        temporal_cols = [col for col in self.df.columns 
                        if col in ['year', 'song_age', 'is_recent']]
        if temporal_cols:
            feature_dims['item']['temporal'] = len(temporal_cols)
        
        return feature_dims
    
    @log_function()
    def get_id_cardinalities(self):
        """
        Get cardinalities of ID columns for embedding layers.
        
        Returns:
            dict: Dictionary with cardinalities.
        """
        cardinalities = {}
        
        if self.user_id_col in self.df.columns:
            cardinalities['user_id'] = int(self.df[self.user_id_col].max() + 1)
        
        if self.track_id_col in self.df.columns:
            cardinalities['track_id'] = int(self.df[self.track_id_col].max() + 1)
        
        if 'artist' in self.df.columns:
            cardinalities['artist'] = int(self.df['artist'].max() + 1)
        
        return cardinalities


class MusicDataLoader:
    """
    Utility class for creating PyTorch DataLoaders for music recommendation data.
    """
    
    def __init__(self, batch_size=64, num_workers=4, logger=None):
        """
        Initialize the data loader.
        
        Args:
            batch_size (int): Batch size for training.
            num_workers (int): Number of workers for data loading.
            logger (LhydraLogger): Logger instance.
        """
        self.logger = logger or LhydraLogger()
        self.batch_size = batch_size
        self.num_workers = num_workers
        
        self.logger.info(f"Initializing MusicDataLoader with batch_size={batch_size}, num_workers={num_workers}")
    
    @log_function()
    def create_dataloaders(self, train_df, val_df=None, test_df=None, 
                         user_id_col='user_id', track_id_col='track_id', 
                         target_col='high_engagement'):
        """
        Create DataLoaders for training, validation, and test sets.
        
        Args:
            train_df (pd.DataFrame): Training data.
            val_df (pd.DataFrame, optional): Validation data.
            test_df (pd.DataFrame, optional): Test data.
            user_id_col (str): Name of user ID column.
            track_id_col (str): Name of track ID column.
            target_col (str): Name of target column.
            
        Returns:
            dict: Dictionary containing DataLoaders.
        """
        from torch.utils.data import DataLoader
        
        self.logger.info("Creating DataLoaders")
        dataloaders = {}
        
        # Create training dataset and loader
        train_dataset = MusicRecommendationDataset(
            train_df, user_id_col=user_id_col, track_id_col=track_id_col,
            target_col=target_col, mode='train', logger=self.logger
        )
        
        dataloaders['train'] = DataLoader(
            train_dataset, batch_size=self.batch_size, shuffle=True,
            num_workers=self.num_workers, pin_memory=True
        )
        
        # Create validation dataset and loader if provided
        if val_df is not None:
            val_dataset = MusicRecommendationDataset(
                val_df, user_id_col=user_id_col, track_id_col=track_id_col,
                target_col=target_col, mode='val', logger=self.logger
            )
            
            dataloaders['val'] = DataLoader(
                val_dataset, batch_size=self.batch_size, shuffle=False,
                num_workers=self.num_workers, pin_memory=True
            )
        
        # Create test dataset and loader if provided
        if test_df is not None:
            test_dataset = MusicRecommendationDataset(
                test_df, user_id_col=user_id_col, track_id_col=track_id_col,
                target_col=target_col, mode='test', logger=self.logger
            )
            
            dataloaders['test'] = DataLoader(
                test_dataset, batch_size=self.batch_size, shuffle=False,
                num_workers=self.num_workers, pin_memory=True
            )
        
        # Log dataset statistics
        self.logger.info(f"Created DataLoaders with {len(dataloaders['train'])} training batches")
        if 'val' in dataloaders:
            self.logger.info(f"Validation batches: {len(dataloaders['val'])}")
        if 'test' in dataloaders:
            self.logger.info(f"Test batches: {len(dataloaders['test'])}")
        
        # Store feature dimensions for model architecture
        self.feature_dims = train_dataset.get_feature_dims()
        self.id_cardinalities = train_dataset.get_id_cardinalities()
        
        self.logger.debug(f"Feature dimensions: {self.feature_dims}")
        self.logger.debug(f"ID cardinalities: {self.id_cardinalities}")
        
        return dataloaders
    
    @log_function()
    def get_model_config(self):
        """
        Get model configuration based on the data.
        
        Returns:
            dict: Dictionary with model configuration.
        """
        if not hasattr(self, 'feature_dims') or not hasattr(self, 'id_cardinalities'):
            self.logger.error("DataLoaders must be created before getting model config")
            raise ValueError("DataLoaders must be created before getting model config")
        
        config = {
            'feature_dims': self.feature_dims,
            'id_cardinalities': self.id_cardinalities,
            'embedding_dim': 32,  # Default embedding dimension
            'hidden_dims': [128, 64],  # Default hidden dimensions
            'dropout': 0.2,  # Default dropout rate
            'batch_size': self.batch_size
        }
        
        return config


if __name__ == "__main__":
    # Example usage
    import logging
    logger = LhydraLogger(log_dir="logs", log_level=logging.INFO)
    
    logger.info("Testing dataset module")
    
    # Load some test data
    import argparse
    parser = argparse.ArgumentParser(description="Test dataset module")
    parser.add_argument("--data", type=str, required=True, help="Path to processed data file")
    args = parser.parse_args()
    
    try:
        df = pd.read_csv(args.data)
        logger.info(f"Loaded test data from {args.data} with shape {df.shape}")
        
        # Create a small test dataset
        dataset = MusicRecommendationDataset(
            df, user_id_col='user_id', track_id_col='track_id',
            target_col='high_engagement', mode='test', logger=logger
        )
        
        # Test getting an item
        sample = dataset[0]
        logger.info(f"Successfully created dataset with {len(dataset)} samples")
        logger.info(f"Sample keys: {sample.keys()}")
        
        # Test data loader
        loader = MusicDataLoader(batch_size=32, logger=logger)
        dataloaders = loader.create_dataloaders(
            df, user_id_col='user_id', track_id_col='track_id',
            target_col='high_engagement'
        )
        
        logger.info("Dataset module test completed successfully")
        
    except Exception as e:
        logger.error(f"Error testing dataset module: {str(e)}", exc_info=e) 