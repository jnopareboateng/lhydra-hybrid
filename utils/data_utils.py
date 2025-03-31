import pandas as pd
import numpy as np
import os
import yaml
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import logging
import joblib
from typing import Dict, List, Optional, Tuple, Union

logger = logging.getLogger(__name__)

def load_config(config_path):
    """
    Load configuration from YAML file
    
    Args:
        config_path (str): Path to the configuration file
        
    Returns:
        dict: Configuration dictionary
    """
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        logger.info(f"Successfully loaded configuration from {config_path}")
        return config
    except Exception as e:
        logger.error(f"Error loading configuration: {e}")
        raise

def load_data(data_path):
    """
    Load data from CSV file
    
    Args:
        data_path (str): Path to the data file
        
    Returns:
        pandas.DataFrame: Loaded dataframe
    """
    try:
        if not os.path.exists(data_path):
            logger.error(f"Data file not found: {data_path}")
            raise FileNotFoundError(f"Data file not found: {data_path}")
        
        df = pd.read_csv(data_path)
        logger.info(f"Successfully loaded data from {data_path}, shape: {df.shape}")
        return df
    except Exception as e:
        logger.error(f"Error loading data: {e}")
        raise

def load_preprocessed_data(data_dir):
    """
    Load preprocessed data splits and preprocessor object
    
    Args:
        data_dir (str): Directory containing preprocessed data
        
    Returns:
        tuple: (train_df, val_df, test_df, preprocessor)
    """
    train_path = os.path.join(data_dir, "train_data.csv")
    val_path = os.path.join(data_dir, "val_data.csv")
    test_path = os.path.join(data_dir, "test_data.csv")
    preprocessor_path = os.path.join(data_dir, "preprocessor.joblib")
    
    try:
        train_df = pd.read_csv(train_path)
        val_df = pd.read_csv(val_path)
        test_df = pd.read_csv(test_path)
        
        preprocessor = None
        if os.path.exists(preprocessor_path):
            preprocessor = joblib.load(preprocessor_path)
            logger.info(f"Loaded preprocessor from {preprocessor_path}")
        
        logger.info(f"Loaded preprocessed data: train={len(train_df)}, val={len(val_df)}, test={len(test_df)}")
        return train_df, val_df, test_df, preprocessor
    except Exception as e:
        logger.error(f"Error loading preprocessed data: {e}")
        raise

def preprocess_user_features(user_features_df, normalize=True):
    """
    Preprocess user features
    
    Args:
        user_features_df (pandas.DataFrame): DataFrame containing user features
        normalize (bool): Whether to normalize the features
        
    Returns:
        pandas.DataFrame: Preprocessed user features
        dict: Feature preprocessing metadata (for inference)
    """
    # Copy to avoid modifying the original dataframe
    df = user_features_df.copy()
    
    # Handle missing values
    df.fillna(df.mean(), inplace=True)
    
    metadata = {}
    
    # Normalize features if required
    if normalize:
        numerical_cols = df.select_dtypes(include=['int64', 'float64']).columns
        if len(numerical_cols) > 0:
            scaler = StandardScaler()
            df[numerical_cols] = scaler.fit_transform(df[numerical_cols])
            metadata['user_scaler'] = scaler
            metadata['user_numerical_cols'] = numerical_cols
    
    logger.info(f"Preprocessed user features, shape: {df.shape}")
    return df, metadata

def preprocess_item_features(item_features_df, normalize=True):
    """
    Preprocess item features
    
    Args:
        item_features_df (pandas.DataFrame): DataFrame containing item features
        normalize (bool): Whether to normalize the features
        
    Returns:
        pandas.DataFrame: Preprocessed item features
        dict: Feature preprocessing metadata (for inference)
    """
    # Copy to avoid modifying the original dataframe
    df = item_features_df.copy()
    
    # Handle missing values
    df.fillna(df.mean(), inplace=True)
    
    metadata = {}
    
    # Normalize features if required
    if normalize:
        numerical_cols = df.select_dtypes(include=['int64', 'float64']).columns
        if len(numerical_cols) > 0:
            scaler = StandardScaler()
            df[numerical_cols] = scaler.fit_transform(df[numerical_cols])
            metadata['item_scaler'] = scaler
            metadata['item_numerical_cols'] = numerical_cols
    
    logger.info(f"Preprocessed item features, shape: {df.shape}")
    return df, metadata

def prepare_user_item_data(
    interactions_df: pd.DataFrame,
    config: Dict,
    use_high_engagement: bool = True
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Prepare user and item data frames from the interactions dataframe for model training
    
    Args:
        interactions_df (pandas.DataFrame): DataFrame with user-item interactions
        config (dict): Configuration with feature definitions
        use_high_engagement (bool): Whether to use high_engagement or engagement column
        
    Returns:
        tuple: (interactions_df, user_features_df, item_features_df)
    """
    # Create a mapping dictionary from user_id to user features
    user_ids = interactions_df['user_id'].unique()
    logger.info(f"Found {len(user_ids)} unique users in interactions")
    
    # Extract user features
    user_cols = []
    
    # Add demographic features
    user_cols.extend(config.get('user_demographic_features', []))
    
    # Add listening features
    user_cols.extend(config.get('user_listening_features', []))
    
    # Add audio preferences
    user_cols.extend(config.get('user_audio_preferences', []))
    
    # Add engineered user features (if present)
    user_features_list = interactions_df.columns.tolist()
    for col in user_features_list:
        if col.startswith('age_group') or col.startswith('listening_depth'):
            user_cols.append(col)
            
    # Ensure user_id is included
    user_cols.append('user_id')
    
    # Create user features dataframe
    available_user_cols = [col for col in user_cols if col in interactions_df.columns]
    user_features_df = interactions_df[available_user_cols].drop_duplicates('user_id')
    logger.info(f"Created user features dataframe with {len(user_features_df)} users and {len(available_user_cols)} features")
    
    # Extract item features
    item_ids = interactions_df['track_id'].unique()
    logger.info(f"Found {len(item_ids)} unique items in interactions")
    
    # Add track metadata features
    item_cols = []
    item_cols.extend(config.get('track_metadata_features', []))
    
    # Add audio features
    item_cols.extend(config.get('track_audio_features', []))
    
    # Add engineered item features (if present)
    item_features_list = interactions_df.columns.tolist()
    for col in item_features_list:
        if col.startswith('duration_') or col.startswith('song_age') or col.startswith('mood_category'):
            item_cols.append(col)
    
    # Ensure track_id is included (using 'track_id' as the column name)
    item_cols.append('track_id')
    
    # Create item features dataframe
    available_item_cols = [col for col in item_cols if col in interactions_df.columns]
    item_features_df = interactions_df[available_item_cols].drop_duplicates('track_id')
    logger.info(f"Created item features dataframe with {len(item_features_df)} items and {len(available_item_cols)} features")
    
    # Ensure the target column is properly named
    if use_high_engagement and 'high_engagement' in interactions_df.columns:
        if 'engagement' not in interactions_df.columns:
            interactions_df['engagement'] = interactions_df['high_engagement']
            logger.info("Copied 'high_engagement' column to 'engagement' for compatibility")
    
    return interactions_df, user_features_df, item_features_df

def create_interaction_features(interactions_df, user_features_df, item_features_df, playcount_threshold=5):
    """
    Create features from user-item interactions
    
    Args:
        interactions_df (pandas.DataFrame): DataFrame containing user-item interactions
        user_features_df (pandas.DataFrame): DataFrame containing user features
        item_features_df (pandas.DataFrame): DataFrame containing item features
        playcount_threshold (int): Threshold to determine positive engagement
        
    Returns:
        pandas.DataFrame: DataFrame with interaction features
    """
    # Copy to avoid modifying the original dataframe
    df = interactions_df.copy()
    
    # Create binary target based on playcount threshold if it doesn't exist
    if 'engagement' not in df.columns and 'high_engagement' not in df.columns:
        df['engagement'] = (df['playcount'] >= playcount_threshold).astype(int)
        logger.info(f"Created binary 'engagement' target with threshold {playcount_threshold}")
    elif 'high_engagement' in df.columns and 'engagement' not in df.columns:
        df['engagement'] = df['high_engagement']
        logger.info("Using 'high_engagement' as 'engagement' target")
    
    # Calculate user-based aggregated features
    user_agg = df.groupby('user_id').agg({
        'playcount': ['mean', 'sum', 'std', 'count'],
        'engagement': 'mean'
    })
    user_agg.columns = ['user_avg_playcount', 'user_total_playcount', 
                        'user_std_playcount', 'user_interaction_count',
                        'user_engagement_rate']
    user_agg.reset_index(inplace=True)
    
    # Calculate item-based aggregated features
    item_agg = df.groupby('track_id').agg({
        'playcount': ['mean', 'sum', 'std', 'count'],
        'engagement': 'mean'
    })
    item_agg.columns = ['item_avg_playcount', 'item_total_playcount', 
                        'item_std_playcount', 'item_interaction_count',
                        'item_engagement_rate']
    item_agg.reset_index(inplace=True)
    
    # Merge aggregated features back to the interactions
    df = pd.merge(df, user_agg, on='user_id', how='left')
    df = pd.merge(df, item_agg, on='track_id', how='left')
    
    # Create additional interaction features
    if 'timestamp' in df.columns:
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df['day_of_week'] = df['timestamp'].dt.dayofweek
        df['hour_of_day'] = df['timestamp'].dt.hour
    
    logger.info(f"Created interaction features, shape: {df.shape}")
    return df

class HybridRecommenderDataset(Dataset):
    """
    Dataset class for the hybrid recommender system
    """
    def __init__(self, interactions_df, user_features_df, item_features_df, target_col='engagement'):
        """
        Initialize the dataset
        
        Args:
            interactions_df (pandas.DataFrame): DataFrame containing user-item interactions
            user_features_df (pandas.DataFrame): DataFrame containing user features
            item_features_df (pandas.DataFrame): DataFrame containing item features
            target_col (str): Name of the target column in interactions_df
            
        Returns:
            dict: Dictionary containing user_id, item_id, user_features, item_features, and target
        """
        self.interactions = interactions_df
        self.user_features = user_features_df
        self.item_features = item_features_df
        self.target_col = target_col
        
        # Use high_engagement as fallback if engagement is not available
        if target_col not in self.interactions.columns and 'high_engagement' in self.interactions.columns:
            self.target_col = 'high_engagement'
            logger.info(f"Target column '{target_col}' not found, using 'high_engagement' instead")
        
        # Ensure user and item IDs are in the index for faster lookup
        if 'user_id' in self.user_features.columns:
            self.user_features = self.user_features.set_index('user_id')
        if 'track_id' in self.item_features.columns:
            self.item_features = self.item_features.set_index('track_id')
        
        logger.info(f"Created dataset with {len(self.interactions)} interactions")
        logger.info(f"User features: {self.user_features.shape}, Item features: {self.item_features.shape}")
    
    def __len__(self):
        """Return the number of interactions"""
        return len(self.interactions)
    
    def __getitem__(self, idx):
        """Get a single data point"""
        interaction = self.interactions.iloc[idx]
        user_id = interaction['user_id']
        item_id = interaction['track_id']
        
        # Get user features for this user ID
        try:
            user_feats = self.user_features.loc[user_id].values.astype(np.float32)
        except KeyError:
            # If user doesn't exist in features, use zeros
            user_feats = np.zeros(len(self.user_features.columns), dtype=np.float32)
        
        # Get item features for this item ID
        try:
            item_feats = self.item_features.loc[item_id].values.astype(np.float32)
        except KeyError:
            # If item doesn't exist in features, use zeros
            item_feats = np.zeros(len(self.item_features.columns), dtype=np.float32)
        
        # Get the target (engagement)
        target = interaction[self.target_col]
        
        return {
            'user_id': user_id,
            'item_id': item_id,
            'user_features': torch.tensor(user_feats, dtype=torch.float),
            'item_features': torch.tensor(item_feats, dtype=torch.float),
            'target': torch.tensor(target, dtype=torch.float)
        }

def create_data_loaders(train_df, val_df, test_df, user_features_df, item_features_df, batch_size=128, target_col='engagement'):
    """
    Create DataLoader objects for training, validation and testing
    
    Args:
        train_df (pandas.DataFrame): Training interactions
        val_df (pandas.DataFrame): Validation interactions
        test_df (pandas.DataFrame): Test interactions
        user_features_df (pandas.DataFrame): User features
        item_features_df (pandas.DataFrame): Item features
        batch_size (int): Batch size
        target_col (str): Name of the target column
        
    Returns:
        dict: Dictionary containing DataLoader objects
    """
    # Create datasets
    train_dataset = HybridRecommenderDataset(train_df, user_features_df, item_features_df, target_col)
    val_dataset = HybridRecommenderDataset(val_df, user_features_df, item_features_df, target_col)
    test_dataset = HybridRecommenderDataset(test_df, user_features_df, item_features_df, target_col)
    
    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    logger.info(f"Created data loaders with batch size {batch_size}")
    
    return {
        'train': train_loader,
        'validation': val_loader,
        'test': test_loader
    }

def train_test_split_interactions(interactions_df, test_size=0.2, val_size=0.1, random_state=42):
    """
    Split interactions into train, validation and test sets
    
    Args:
        interactions_df (pandas.DataFrame): DataFrame containing interactions
        test_size (float): Proportion of data for testing
        val_size (float): Proportion of data for validation
        random_state (int): Random seed
        
    Returns:
        tuple: (train_df, val_df, test_df)
    """
    np.random.seed(random_state)
    
    # Shuffle the data
    interactions_df = interactions_df.sample(frac=1, random_state=random_state).reset_index(drop=True)
    
    # Calculate split indices
    n = len(interactions_df)
    test_idx = int(n * (1 - test_size))
    val_idx = int(n * (1 - test_size - val_size))
    
    # Split the data
    train_df = interactions_df.iloc[:val_idx].copy()
    val_df = interactions_df.iloc[val_idx:test_idx].copy()
    test_df = interactions_df.iloc[test_idx:].copy()
    
    logger.info(f"Split data: train={len(train_df)}, val={len(val_df)}, test={len(test_df)}")
    
    return train_df, val_df, test_df 