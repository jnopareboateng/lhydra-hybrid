#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Recommendation generator for the Lhydra Hybrid Music Recommender System.
This script is aligned with the preprocessing and training pipeline 
and uses feature alignment for compatibility.
"""

import os
import sys
import argparse
import yaml
import torch
import logging
import pandas as pd
import numpy as np
from datetime import datetime
from tqdm import tqdm
from typing import Dict, List, Optional, Tuple, Union

# Add the root directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.two_tower_model import TwoTowerHybridModel
from data.preprocessor import MusicDataPreprocessor
from utils.data_utils import load_config, load_data
from utils.logger import LhydraLogger, log_function

def setup_logging(name, log_dir='logs/recommendations', log_level=logging.INFO):
    """
    Setup logging configuration
    
    Args:
        name (str): Logger name
        log_dir (str): Directory to store log files
        log_level (int): Logging level
        
    Returns:
        logging.Logger: Configured logger
    """
    # Create logger
    logger = logging.getLogger(name)
    logger.setLevel(log_level)
    logger.handlers = []  # Remove any existing handlers
    
    # Create console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(log_level)
    
    # Create file handler
    os.makedirs(log_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(log_dir, f"{name}_{timestamp}.log")
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(log_level)
    
    # Create formatters
    console_formatter = logging.Formatter('%(levelname)s - %(message)s')
    file_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    
    # Add formatters to handlers
    console_handler.setFormatter(console_formatter)
    file_handler.setFormatter(file_formatter)
    
    # Add handlers to logger
    logger.addHandler(console_handler)
    logger.addHandler(file_handler)
    
    logger.info(f"Logging to {log_file}")
    return logger

class RecommendationEngine:
    """
    Engine for generating recommendations using a trained Two-Tower Hybrid Model.
    
    This class is designed to work with the feature alignment system to ensure
    compatibility between training and inference feature sets.
    """
    
    def __init__(self, model, user_features, item_features, logger, device=None):
        """
        Initialize the recommendation engine
        
        Args:
            model (TwoTowerHybridModel): Trained model
            user_features (pd.DataFrame): User features
            item_features (pd.DataFrame): Item features
            logger (logging.Logger): Logger instance
            device (torch.device, optional): Device to use for inference
        """
        self.logger = logger
        self.user_features = user_features
        self.item_features = item_features
        
        # Print detailed diagnostic information about data
        self.logger.info("\n" + "="*80)
        self.logger.info("RECOMMENDATION ENGINE INITIALIZATION - DATA INSPECTION")
        self.logger.info("="*80)
        
        # User features info
        self.logger.info(f"User features shape: {self.user_features.shape}")
        self.logger.info(f"User features columns: {list(self.user_features.columns)}")
        self.logger.info(f"User features dtypes:\n{self.user_features.dtypes}")
        if not self.user_features.empty:
            self.logger.info(f"User features sample (first 3 rows):\n{self.user_features.head(3).to_string()}")
            
        # Item features info
        self.logger.info(f"Item features shape: {self.item_features.shape}")
        self.logger.info(f"Item features columns: {list(self.item_features.columns)}")
        self.logger.info(f"Item features dtypes:\n{self.item_features.dtypes}")
        if not self.item_features.empty:
            self.logger.info(f"Item features sample (first 3 rows):\n{self.item_features.head(3).to_string()}")
            
        # Model info
        self.logger.info(f"Model type: {type(model).__name__}")
        if hasattr(model, 'user_input_dim') and hasattr(model, 'item_input_dim'):
            self.logger.info(f"Model expected dimensions: user_input_dim={model.user_input_dim}, item_input_dim={model.item_input_dim}")
        self.logger.info("="*80 + "\n")
        
        # Set device
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = device
        
        self.model = model
        self.model.to(self.device)
        self.model.eval()
        
        # Check if model has feature manifest
        self.has_feature_manifest = hasattr(model, 'feature_manifest') and model.feature_manifest is not None
        if self.has_feature_manifest:
            self.logger.info("Model has feature manifest. Feature alignment will be used.")
            manifest_user_dim = model.feature_manifest.get('model_dimensions', {}).get('user_input_dim')
            manifest_item_dim = model.feature_manifest.get('model_dimensions', {}).get('item_input_dim')
            self.logger.info(f"Feature dimensions from manifest: user={manifest_user_dim}, item={manifest_item_dim}")
            
            # Print manifest details for debugging
            self.logger.info("\n" + "="*80)
            self.logger.info("FEATURE MANIFEST DETAILS")
            self.logger.info("="*80)
            if 'user_features' in model.feature_manifest:
                self.logger.info(f"User features in manifest: {model.feature_manifest['user_features']}")
            if 'item_features' in model.feature_manifest:
                self.logger.info(f"Item features in manifest: {model.feature_manifest['item_features']}")
            self.logger.info("="*80 + "\n")
        else:
            self.logger.warning("Model does not have feature manifest. Feature compatibility issues may occur.")
        
        # Extract item IDs
        if 'track_id' in self.item_features.columns:
            self.item_id_field = 'track_id'
            self.item_ids = self.item_features['track_id'].values
        elif 'item_id' in self.item_features.columns:
            self.item_id_field = 'item_id'
            self.item_ids = self.item_features['item_id'].values
        else:
            self.item_id_field = self.item_features.index.name or 'index'
            self.item_ids = self.item_features.index.values
            
        # Extract user IDs
        if 'user_id' in self.user_features.columns:
            self.user_id_field = 'user_id'
            self.user_ids = self.user_features['user_id'].values
        else:
            self.user_id_field = self.user_features.index.name or 'index'
            self.user_ids = self.user_features.index.values
        
        self.logger.info(f"Recommendation engine initialized with {len(self.user_ids)} users and {len(self.item_ids)} items")
        self.logger.info(f"User ID field: {self.user_id_field}, Item ID field: {self.item_id_field}")
    
    @log_function()
    def _prepare_features_for_prediction(self, user_df=None, item_df=None):
        """
        Prepare features for prediction
        
        Args:
            user_df (pd.DataFrame, optional): User features DataFrame
            item_df (pd.DataFrame, optional): Item features DataFrame
            
        Returns:
            tuple: Tuple of user and item tensors
        """
        self.logger.debug("\n" + "="*80)
        self.logger.debug("FEATURE PREPARATION - INPUT")
        self.logger.debug("="*80)
        
        # Validate user features
        if user_df is not None:
            self.logger.debug(f"User features shape: {user_df.shape}")
            if len(user_df) == 0:
                raise ValueError("User DataFrame is empty")
            elif len(user_df) > 1:
                self.logger.warning(f"Multiple user rows found ({len(user_df)}), using the first one")
                user_df = user_df.iloc[[0]]
        else:
            raise ValueError("User features are required for prediction")
            
        # Validate item features
        if item_df is not None:
            self.logger.debug(f"Item features shape: {item_df.shape}")
            if len(item_df) == 0:
                raise ValueError("Item DataFrame is empty")
        else:
            self.logger.debug("No item features provided, using all item features")
            item_df = self.item_features
            
        self.logger.debug("="*80)
        
        # Extract user_id and item_id
        user_id = None
        item_ids = []
        
        # Get user ID
        if user_df is not None and self.user_id_field in user_df.columns:
            user_id = user_df[self.user_id_field].values[0]
        
        # Get item IDs
        if item_df is not None and self.item_id_field in item_df.columns:
            item_ids = item_df[self.item_id_field].values.tolist()
        
        # Create tensors from features
        self.logger.debug("\n" + "="*80)
        self.logger.debug("RAW TENSORS BEFORE ALIGNMENT")
        self.logger.debug("="*80)
        
        # Create user tensors
        user_tensors = {}
        
        # First, handle the ID column separately
        if self.user_id_field in user_df.columns:
            user_id_value = user_df[self.user_id_field].iloc[0]
            if isinstance(user_id_value, (int, float, np.number)):
                user_tensors[self.user_id_field] = torch.tensor([user_id_value], dtype=torch.float32, device=self.device)
            else:
                # For non-numeric IDs, we'll just keep track of it but not include in tensor
                user_id = user_id_value
        
        # Then process feature columns
        user_feature_cols = [col for col in user_df.columns if col != self.user_id_field]
        
        # Extract non-ID features for user
        user_features = user_df[user_feature_cols].values.astype(np.float32)
        self.logger.debug(f"User features array shape: {user_features.shape}")
        
        # Create item tensors
        item_tensors = {}
        
        # First, handle the ID column separately
        if self.item_id_field in item_df.columns:
            # Convert to tensor if numeric, otherwise just keep track
            item_id_values = item_df[self.item_id_field].values
            if np.issubdtype(item_id_values.dtype, np.number):
                item_tensors[self.item_id_field] = torch.tensor(item_id_values, dtype=torch.float32, device=self.device)
        
        # Process feature columns
        item_feature_cols = [col for col in item_df.columns if col != self.item_id_field]
        
        # Extract non-ID features for items
        item_features = item_df[item_feature_cols].values.astype(np.float32)
        self.logger.debug(f"Item features array shape: {item_features.shape}")
        
        # Convert to tensors
        user_tensor = torch.tensor(user_features, dtype=torch.float32, device=self.device)
        item_tensor = torch.tensor(item_features, dtype=torch.float32, device=self.device)
        
        # Log raw tensors
        self.logger.debug(f"Raw user tensor shape: {user_tensor.shape}")
        self.logger.debug(f"Raw item tensor shape: {item_tensor.shape}")
        
        # Check if model has feature manifest for alignment
        if hasattr(self.model, 'feature_manifest') and self.model.feature_manifest is not None:
            self.logger.debug("\n" + "="*80)
            self.logger.debug("FEATURE ALIGNMENT BASED ON MANIFEST")
            self.logger.debug("="*80)
            
            # Try to align features using model's method if available
            if hasattr(self.model, 'align_features'):
                try:
                    self.logger.debug("Using model's align_features method")
                    aligned_user_tensor, aligned_item_tensor = self.model.align_features(
                        user_features={self.user_id_field: user_tensors.get(self.user_id_field), 'features': user_tensor},
                        item_features={self.item_id_field: item_tensors.get(self.item_id_field), 'features': item_tensor}
                    )
                    
                    self.logger.debug(f"Aligned user tensor shape: {aligned_user_tensor.shape}")
                    self.logger.debug(f"Aligned item tensor shape: {aligned_item_tensor.shape}")
                    
                    # Ensure user tensor has batch dimension if it doesn't already
                    if len(aligned_user_tensor.shape) == 1:
                        aligned_user_tensor = aligned_user_tensor.unsqueeze(0)
                        self.logger.debug(f"Added batch dimension to user tensor: {aligned_user_tensor.shape}")
                    
                    # Finalize tensors
                    user_tensor = aligned_user_tensor
                    item_tensor = aligned_item_tensor
                    
                except Exception as e:
                    self.logger.error(f"Error in feature alignment: {str(e)}")
                    self.logger.warning("Proceeding with original tensors")
            else:
                self.logger.debug("Model has manifest but no align_features method, performing manual alignment")
                # TODO: Manual alignment logic based on manifest
        else:
            self.logger.debug("No feature manifest available, using raw tensors")
        
        self.logger.debug("="*80)
        
        # Ensure correct tensor shapes
        self.logger.debug("\n" + "="*80)
        self.logger.debug("FINAL TENSORS FOR MODEL INPUT")
        self.logger.debug("="*80)
        
        # Ensure user tensor has batch dimension
        if len(user_tensor.shape) == 1:
            user_tensor = user_tensor.unsqueeze(0)
            self.logger.debug(f"Added batch dimension to user tensor: new shape={user_tensor.shape}")
        
        # Check if dimensions match model expectations
        user_dim = getattr(self.model, 'user_input_dim', None)
        item_dim = getattr(self.model, 'item_input_dim', None)
        
        # Log tensor dimensions
        self.logger.debug(f"Final user tensor shape: {user_tensor.shape}")
        self.logger.debug(f"Final item tensor shape: {item_tensor.shape}")
        
        if user_dim is not None and item_dim is not None:
            # Check if dimensions match
            if user_tensor.shape[1] != user_dim:
                self.logger.warning(f"User tensor dimension mismatch: got {user_tensor.shape[1]}, expected {user_dim}")
                
                # If dimensions dont match, try to truncate or pad
                if user_tensor.shape[1] > user_dim:
                    self.logger.debug(f"Truncating user tensor from {user_tensor.shape[1]} to {user_dim} dimensions")
                    user_tensor = user_tensor[:, :user_dim]
                else:
                    self.logger.debug(f"Padding user tensor from {user_tensor.shape[1]} to {user_dim} dimensions with zeros")
                    padding = torch.zeros((user_tensor.shape[0], user_dim - user_tensor.shape[1]), device=self.device)
                    user_tensor = torch.cat([user_tensor, padding], dim=1)
            
            if item_tensor.shape[1] != item_dim:
                self.logger.warning(f"Item tensor dimension mismatch: got {item_tensor.shape[1]}, expected {item_dim}")
                
                # If dimensions dont match, try to truncate or pad
                if item_tensor.shape[1] > item_dim:
                    self.logger.debug(f"Truncating item tensor from {item_tensor.shape[1]} to {item_dim} dimensions")
                    item_tensor = item_tensor[:, :item_dim]
                else:
                    self.logger.debug(f"Padding item tensor from {item_tensor.shape[1]} to {item_dim} dimensions with zeros")
                    padding = torch.zeros((item_tensor.shape[0], item_dim - item_tensor.shape[1]), device=self.device)
                    item_tensor = torch.cat([item_tensor, padding], dim=1)
        
        # Log final dimensions
        self.logger.debug(f"User tensor count: {user_tensor.shape[1]}")
        self.logger.debug(f"Item tensor count: {item_tensor.shape[1]}")
        self.logger.debug(f"Total user feature dimensions: {user_tensor.shape[1]}")
        self.logger.debug(f"Total item feature dimensions: {item_tensor.shape[1]}")
        self.logger.debug(f"Model expects: user_dim={user_dim}, item_dim={item_dim}")
        self.logger.debug("="*80)
        
        return user_tensor, item_tensor
    
    @log_function()
    def predict_scores(self, user_id, top_k=10, exclude_items=None):
        """
        Predict scores for a single user across all items
        
        Args:
            user_id: User ID to generate recommendations for
            top_k (int): Number of top recommendations to return
            exclude_items (list, optional): List of item IDs to exclude from recommendations
            
        Returns:
            pd.DataFrame: DataFrame with recommendations
        """
        # Log prediction request
        self.logger.info("\n" + "="*80)
        self.logger.info(f"PREDICTION REQUEST FOR USER {user_id}")
        self.logger.info("="*80)
        self.logger.info(f"Request details: user_id={user_id}, top_k={top_k}, exclude_items_count={len(exclude_items) if exclude_items else 0}")
        
        # Get user features
        if self.user_id_field in self.user_features.columns:
            # Try to convert user_id to match the DataFrame dtype if needed
            user_id_value = user_id
            df_dtype = self.user_features[self.user_id_field].dtype
            
            # Convert user_id to match DataFrame type
            if df_dtype == 'int64':
                try:
                    user_id_value = int(user_id)
                except (ValueError, TypeError):
                    self.logger.warning(f"Cannot convert user ID '{user_id}' to integer for lookup")
            
            # Try exact match first
            user_df = self.user_features[self.user_features[self.user_id_field] == user_id_value]
            
            # If no match, try string comparison
            if len(user_df) == 0:
                self.logger.debug(f"User {user_id} not found with exact match, trying string comparison")
                user_df = self.user_features[self.user_features[self.user_id_field].astype(str) == str(user_id)]
            
            if len(user_df) == 0:
                self.logger.warning(f"User {user_id} not found in user features")
                return pd.DataFrame()
            else:
                self.logger.info(f"Found user {user_id} in user features")
                self.logger.debug(f"User data: \n{user_df.to_string()}")
        else:
            try:
                # Try direct index lookup
                user_df = self.user_features.loc[[user_id]]
            except KeyError:
                # Try string comparison if index is not string type
                try:
                    if not isinstance(user_id, str) and self.user_features.index.dtype.kind == 'O':
                        idx = self.user_features.index.astype(str) == str(user_id)
                        if idx.any():
                            user_df = self.user_features[idx]
                        else:
                            raise KeyError(f"User {user_id} not found")
                    else:
                        raise KeyError(f"User {user_id} not found")
                except KeyError:
                    self.logger.warning(f"User {user_id} not found in user features")
                    return pd.DataFrame()
        
        # Create feature tensors for the user
        user_tensors, item_tensors = self._prepare_features_for_prediction(user_df)
        
        # Initialize scores array
        scores = np.zeros(len(self.item_ids))
        
        # Use chunking for large item sets to avoid memory issues
        chunk_size = 100  # Process 100 items at a time
        num_chunks = (len(self.item_ids) + chunk_size - 1) // chunk_size
        
        self.logger.debug(f"Processing predictions in {num_chunks} chunks of {chunk_size} items")
        
        # Make predictions in chunks to avoid memory issues
        with torch.inference_mode():
            try:
                for chunk_idx in range(num_chunks):
                    start_idx = chunk_idx * chunk_size
                    end_idx = min(start_idx + chunk_size, len(self.item_ids))
                    chunk_size_actual = end_idx - start_idx
                    
                    # Log chunk info
                    self.logger.debug(f"Processing chunk {chunk_idx+1}/{num_chunks}: items {start_idx} to {end_idx}")
                    
                    # Get chunk of item tensors
                    if isinstance(item_tensors, dict):
                        # Dictionary of tensors - typical case
                        chunk_item_tensors = {}
                        for key in item_tensors:
                            chunk_item_tensors[key] = item_tensors[key][start_idx:end_idx]
                    else:
                        # Tensor output from alignment
                        chunk_item_tensors = item_tensors[start_idx:end_idx]
                    
                    # No need to repeat user tensors for batching - the model will handle it
                    # This is critical - we were creating incorrect shaped tensors before
                    
                    # Log tensor shapes before forward pass
                    if chunk_idx == 0:  # Only log for first chunk to avoid verbosity
                        self.logger.debug("\n" + "="*80)
                        self.logger.debug("MODEL INPUT TENSOR SHAPES")
                        self.logger.debug("="*80)
                        if isinstance(user_tensors, dict):
                            for k, v in user_tensors.items():
                                self.logger.debug(f"User tensor '{k}': shape={v.shape}, dtype={v.dtype}")
                        else:
                            self.logger.debug(f"User tensor: shape={user_tensors.shape}, dtype={user_tensors.dtype}")
                            
                        if isinstance(chunk_item_tensors, dict):
                            for k, v in chunk_item_tensors.items():
                                self.logger.debug(f"Item tensor '{k}': shape={v.shape}, dtype={v.dtype}")
                        else:
                            self.logger.debug(f"Item tensor: shape={chunk_item_tensors.shape}, dtype={chunk_item_tensors.dtype}")
                        self.logger.debug("="*80 + "\n")
                    
                    # Forward pass with the original shapes - we'll use broadcasting
                    try:
                        outputs = self.model(user_tensors, chunk_item_tensors)
                        
                        # Log output details for first chunk
                        if chunk_idx == 0:
                            self.logger.debug("\n" + "="*80)
                            self.logger.debug("MODEL OUTPUT")
                            self.logger.debug("="*80)
                            self.logger.debug(f"Output shape: {outputs.shape}")
                            self.logger.debug(f"Output dtype: {outputs.dtype}")
                            self.logger.debug(f"Output sample (first 5 values): {outputs[:5].cpu().numpy() if outputs.numel() > 5 else outputs.cpu().numpy()}")
                            self.logger.debug("="*80 + "\n")
                        
                        # Handle different output shapes
                        chunk_scores = outputs.squeeze().cpu().numpy()
                        
                        # Store scores for this chunk
                        scores[start_idx:end_idx] = chunk_scores
                    except Exception as e:
                        self.logger.error(f"Error in chunk {chunk_idx+1}/{num_chunks}: {str(e)}")
                        # Try an alternative approach with explicit batching
                        try:
                            self.logger.debug("Trying alternative approach with explicit user tensor repetition")
                            # Repeat user tensors to match item tensors
                            if isinstance(user_tensors, dict):
                                # For dictionary of tensors
                                batch_user_tensors = {}
                                for key, tensor in user_tensors.items():
                                    # Create a batch dimension of size chunk_size_actual
                                    batch_user_tensors[key] = tensor.expand(chunk_size_actual, *tensor.shape[1:])
                            else:
                                # For tensor output from alignment
                                batch_user_tensors = user_tensors.expand(chunk_size_actual, *user_tensors.shape[1:])
                            
                            # Try again with explicitly batched user tensors
                            outputs = self.model(batch_user_tensors, chunk_item_tensors)
                            chunk_scores = outputs.squeeze().cpu().numpy()
                            scores[start_idx:end_idx] = chunk_scores
                            
                            self.logger.debug("Alternative approach succeeded")
                        except Exception as e2:
                            self.logger.error(f"Alternative approach also failed: {str(e2)}")
                            # Continue with next chunk, treating this one as zeros
                            self.logger.warning(f"Using zero scores for chunk {chunk_idx+1}")
                            scores[start_idx:end_idx] = 0.0
            except RuntimeError as e:
                self.logger.error(f"Error during prediction: {str(e)}")
                return pd.DataFrame()
        
        # Filter out excluded items
        if exclude_items is not None:
            exclude_indices = [i for i, item_id in enumerate(self.item_ids) if item_id in exclude_items]
            for idx in exclude_indices:
                scores[idx] = -np.inf
        
        # Get top-k items
        top_indices = np.argsort(-scores)[:top_k]
        top_items = [self.item_ids[i] for i in top_indices]
        top_scores = scores[top_indices]
        
        # Create recommendations dataframe
        recommendations = pd.DataFrame({
            'user_id': [user_id] * len(top_items),
            'item_id': top_items,
            'score': top_scores,
            'rank': range(1, len(top_items) + 1)
        })
        
        # Log recommendation results
        self.logger.info("\n" + "="*80)
        self.logger.info("RECOMMENDATION RESULTS")
        self.logger.info("="*80)
        self.logger.info(f"Generated {len(recommendations)} recommendations for user {user_id}")
        if not recommendations.empty:
            self.logger.info(f"Top 5 recommendations:\n{recommendations.head(5).to_string()}")
        self.logger.info("="*80 + "\n")
        
        return recommendations
    
    @log_function()
    def generate_recommendations(self, user_ids=None, top_k=10, exclude_history=None, enhance_metadata=False, spotify_sample_path=None):
        """
        Generate recommendations for multiple users
        
        Args:
            user_ids (list, optional): List of user IDs. If None, generate for all users.
            top_k (int): Number of top recommendations to return
            exclude_history (pd.DataFrame, optional): DataFrame with user-item interactions to exclude
            enhance_metadata (bool): Whether to enhance recommendations with metadata
            spotify_sample_path (str, optional): Path to Spotify sample data for additional metadata
            
        Returns:
            pd.DataFrame: DataFrame with recommendations for all users
        """
        if user_ids is None:
            user_ids = self.user_ids
        
        # Create exclude items dictionary if needed
        exclude_items_dict = {}
        if exclude_history is not None:
            for _, row in exclude_history.iterrows():
                user = row['user_id']
                item = row[self.item_id_field]
                if user not in exclude_items_dict:
                    exclude_items_dict[user] = []
                exclude_items_dict[user].append(item)
        
        # Generate recommendations for each user
        all_recommendations = []
        for user_id in tqdm(user_ids, desc="Generating recommendations"):
            exclude_items = exclude_items_dict.get(user_id, None)
            user_recs = self.predict_scores(user_id, top_k=top_k, exclude_items=exclude_items)
            
            if not user_recs.empty:
                all_recommendations.append(user_recs)
        
        # Combine all recommendations
        if all_recommendations:
            recommendations = pd.concat(all_recommendations, ignore_index=True)
            
            # Enhance with metadata if requested
            if enhance_metadata:
                self.logger.info("Enhancing recommendations with metadata")
                recommendations = self.enhance_recommendations_with_metadata(recommendations, self.item_features, spotify_sample_path)
            
            return recommendations
        else:
            return pd.DataFrame()
    
    @log_function()
    def find_similar_items(self, item_id, top_k=10):
        """
        Find items similar to a given item based on embedding similarity
        
        Args:
            item_id: Item ID to find similar items for
            top_k (int): Number of similar items to return
            
        Returns:
            pd.DataFrame: DataFrame with similar items
        """
        # Get item features
        if self.item_id_field in self.item_features.columns:
            # Try to convert item_id to match the DataFrame dtype if needed
            item_id_value = item_id
            df_dtype = self.item_features[self.item_id_field].dtype
            
            # Convert item_id to match DataFrame type
            if df_dtype == 'int64':
                try:
                    item_id_value = int(item_id)
                except (ValueError, TypeError):
                    self.logger.warning(f"Cannot convert item ID '{item_id}' to integer for lookup")
            
            # Try exact match first
            item_df = self.item_features[self.item_features[self.item_id_field] == item_id_value]
            
            # If no match, try string comparison
            if len(item_df) == 0:
                self.logger.debug(f"Item {item_id} not found with exact match, trying string comparison")
                item_df = self.item_features[self.item_features[self.item_id_field].astype(str) == str(item_id)]
            
            if len(item_df) == 0:
                self.logger.warning(f"Item {item_id} not found in item features")
                return pd.DataFrame()
            else:
                self.logger.info(f"Found item {item_id} in item features")
        else:
            try:
                # Try direct index lookup
                item_df = self.item_features.loc[[item_id]]
            except KeyError:
                # Try string comparison if index is not string type
                try:
                    if not isinstance(item_id, str) and self.item_features.index.dtype.kind == 'O':
                        idx = self.item_features.index.astype(str) == str(item_id)
                        if idx.any():
                            item_df = self.item_features[idx]
                        else:
                            raise KeyError(f"Item {item_id} not found")
                    else:
                        raise KeyError(f"Item {item_id} not found")
                except KeyError:
                    self.logger.warning(f"Item {item_id} not found in item features")
                    return pd.DataFrame()
        
        # Create feature tensors
        _, item_tensors = self._prepare_features_for_prediction(pd.DataFrame(), item_df)
        _, all_item_tensors = self._prepare_features_for_prediction(pd.DataFrame(), self.item_features)
        
        # Get embeddings
        with torch.inference_mode():
            # Get target item embedding
            if hasattr(self.model, 'item_embedding') and callable(self.model.item_embedding):
                target_embedding = self.model.item_embedding(item_tensors).cpu().numpy()[0]
                
                # Get all item embeddings
                all_embeddings = self.model.item_embedding(all_item_tensors).cpu().numpy()
            else:
                self.logger.warning("Model doesn't have item_embedding method, similarity search may not work correctly")
                return pd.DataFrame()
        
        # Calculate cosine similarity
        similarities = []
        for i, item_embedding in enumerate(all_embeddings):
            if self.item_ids[i] == item_id:
                continue  # Skip the target item
                
            # Skip if either embedding is all zeros
            if np.allclose(target_embedding, 0) or np.allclose(item_embedding, 0):
                continue
                
            # Calculate cosine similarity
            similarity = np.dot(target_embedding, item_embedding) / (
                np.linalg.norm(target_embedding) * np.linalg.norm(item_embedding)
            )
            
            similarities.append({
                'item_id': self.item_ids[i],
                'similarity': similarity
            })
        
        # Sort by similarity and take top-k
        similarities = sorted(similarities, key=lambda x: x['similarity'], reverse=True)[:top_k]
        
        # Create dataframe
        similar_items = pd.DataFrame(similarities)
        if not similar_items.empty:
            similar_items['rank'] = range(1, len(similar_items) + 1)
            
        return similar_items
    
    def enhance_recommendations_with_metadata(self, recommendations_df, item_df, spotify_sample_path=None):
        """
        Enhance recommendations with item metadata and return a new DataFrame.
        
        Args:
            recommendations_df (pd.DataFrame): Recommendations DataFrame
            item_df (pd.DataFrame): Item features DataFrame
            spotify_sample_path (str, optional): Path to Spotify sample data for additional metadata
        
        Returns:
            pd.DataFrame: Enhanced recommendations with metadata
        """
        logger = logging.getLogger(__name__)
        logger.info("Enhancing recommendations with metadata")
        
        # Create a new DataFrame with additional columns
        enhanced_recs = recommendations_df.copy()
        
        # Check if we have track_id column
        if 'track_id' in item_df.columns:
            track_id_col = 'track_id'
        else:
            # Assume the last column is the track_id
            track_id_col = item_df.columns[-1]
            logger.warning(f"No explicit track_id column found, using the last column: {track_id_col}")
        
        # Create a mapping from item_id to basic metadata
        item_metadata = {}
        for _, row in item_df.iterrows():
            item_id = row[track_id_col]
            
            # Start with basic metadata
            metadata = {
                'track_id': item_id,
                'name': f"Track {item_id}",  # Default name
                'artist': "Unknown Artist",  # Default artist
            }
            
            # Add any additional metadata if available
            for col in item_df.columns:
                if col != track_id_col and not col.startswith('0') and not pd.isna(row[col]):
                    # Skip numerical features but keep metadata columns
                    if col in ['name', 'artist', 'genre', 'year', 'tags']:
                        metadata[col] = row[col]
            
            item_metadata[item_id] = metadata
        
        # Try to enhance with Spotify metadata if available
        if spotify_sample_path and os.path.exists(spotify_sample_path):
            try:
                logger.info(f"Loading Spotify sample from {spotify_sample_path}")
                sample_df = pd.read_csv(spotify_sample_path)
                
                # Extract unique track information
                spotify_metadata = sample_df.drop_duplicates(subset=['track_id'])[
                    ['track_id', 'name', 'artist', 'spotify_id', 'spotify_preview_url', 
                     'tags', 'genre', 'main_genre', 'year']
                ]
                
                enhanced_count = 0
                
                # Try to match based on track_id
                for i, item in item_metadata.items():
                    spotify_match = spotify_metadata[spotify_metadata['track_id'] == str(i)]
                    
                    if len(spotify_match) > 0:
                        # Update metadata with Spotify data
                        for col in spotify_match.columns:
                            if col in ['name', 'artist', 'genre', 'year', 'tags', 'spotify_id', 'spotify_preview_url']:
                                item_metadata[i][col] = spotify_match.iloc[0][col]
                        enhanced_count += 1
                
                logger.info(f"Enhanced {enhanced_count} tracks with Spotify metadata")
            except Exception as e:
                logger.error(f"Error reading Spotify sample: {e}")
        
        # Add metadata columns to recommendations
        enhanced_recs['name'] = enhanced_recs['item_id'].map(lambda x: item_metadata.get(x, {}).get('name', f"Track {x}"))
        enhanced_recs['artist'] = enhanced_recs['item_id'].map(lambda x: item_metadata.get(x, {}).get('artist', "Unknown Artist"))
        
        # Add other available metadata
        all_metadata_fields = set()
        for meta in item_metadata.values():
            all_metadata_fields.update(meta.keys())
        
        for field in all_metadata_fields:
            if field not in ['track_id', 'name', 'artist'] and field not in enhanced_recs.columns:
                enhanced_recs[field] = enhanced_recs['item_id'].map(
                    lambda x: item_metadata.get(x, {}).get(field, None)
                )
        
        logger.info(f"Added metadata columns: {[col for col in enhanced_recs.columns if col not in recommendations_df.columns]}")
        return enhanced_recs
    
    @classmethod
    def from_checkpoint(cls, model_path, user_features_path, item_features_path, 
                        manifest_path=None, device=None, logger=None):
        """
        Create a recommendation engine from a model checkpoint
        
        Args:
            model_path (str): Path to model checkpoint
            user_features_path (str): Path to user features CSV
            item_features_path (str): Path to item features CSV
            manifest_path (str, optional): Path to feature manifest file
            device (torch.device, optional): Device to use for inference
            logger (logging.Logger, optional): Logger instance
            
        Returns:
            RecommendationEngine: Recommendation engine initialized with the model and features
        """
        # Setup logger if not provided
        if logger is None:
            logger = logging.getLogger()
            handler = logging.StreamHandler()
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            logger.addHandler(handler)
            logger.setLevel(logging.INFO)
        
        # Set device
        if device is None:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        logger.info("\n" + "*"*100)
        logger.info("RECOMMENDATION ENGINE INITIALIZATION FROM CHECKPOINT")
        logger.info("*"*100)
        logger.info(f"Model path: {model_path}")
        logger.info(f"User features path: {user_features_path}")
        logger.info(f"Item features path: {item_features_path}")
        logger.info(f"Feature manifest path: {manifest_path}")
        logger.info(f"Device: {device}")
        
        # Load the model
        try:
            from models.two_tower_model import TwoTowerHybridModel
            
            logger.info("Loading model from checkpoint...")
            
            # Try loading model with manifest path
            try:
                model = TwoTowerHybridModel.load(model_path, device=device, manifest_path=manifest_path)
                logger.info("Model loaded successfully with manifest")
                
                # Log model dimensions
                logger.info(f"Model dimensions: user_input_dim={model.user_input_dim}, item_input_dim={model.item_input_dim}")
                
                # Check if model has feature manifest
                if hasattr(model, 'feature_manifest') and model.feature_manifest is not None:
                    logger.info("Model has feature manifest information")
                    if 'model_dimensions' in model.feature_manifest:
                        logger.info(f"Model dimensions from manifest: {model.feature_manifest['model_dimensions']}")
                else:
                    logger.warning("Model does not have feature manifest information")
                
            except Exception as e:
                logger.error(f"Error loading model with manifest: {str(e)}")
                logger.info("Attempting to load model without manifest...")
                model = TwoTowerHybridModel.load(model_path, device=device, manifest_path=None)
                logger.info("Model loaded successfully without manifest")
        
        except (ImportError, ModuleNotFoundError):
            logger.error("Could not import TwoTowerHybridModel. Make sure models/two_tower_model.py exists")
            raise
        
        # Load user features
        try:
            logger.info(f"Loading user features from {user_features_path}...")
            user_features = pd.read_csv(user_features_path)
            logger.info(f"Loaded user features, shape: {user_features.shape}")
            logger.info(f"User features columns: {list(user_features.columns)}")
            logger.info(f"User features types: {user_features.dtypes}")
            
            # Check for missing values
            missing_values = user_features.isnull().sum().sum()
            if missing_values > 0:
                logger.warning(f"User features have {missing_values} missing values")
            
            # Identify potential user_id columns
            potential_id_cols = []
            for col in user_features.columns:
                if 'id' in col.lower() or 'user' in col.lower():
                    potential_id_cols.append(col)
            
            if 'user_id' not in user_features.columns and potential_id_cols:
                logger.warning(f"'user_id' column not found. Potential ID columns: {potential_id_cols}")
            
        except Exception as e:
            logger.error(f"Error loading user features: {str(e)}")
            raise
        
        # Load item features
        try:
            logger.info(f"Loading item features from {item_features_path}...")
            item_features = pd.read_csv(item_features_path)
            logger.info(f"Loaded item features, shape: {item_features.shape}")
            logger.info(f"Item features columns: {list(item_features.columns)}")
            logger.info(f"Item features types: {item_features.dtypes}")
            
            # Check for missing values
            missing_values = item_features.isnull().sum().sum()
            if missing_values > 0:
                logger.warning(f"Item features have {missing_values} missing values")
            
            # Identify potential item_id columns
            potential_id_cols = []
            for col in item_features.columns:
                if 'id' in col.lower() or 'track' in col.lower() or 'item' in col.lower():
                    potential_id_cols.append(col)
            
            if 'track_id' not in item_features.columns and 'item_id' not in item_features.columns and potential_id_cols:
                logger.warning(f"Neither 'track_id' nor 'item_id' column found. Potential ID columns: {potential_id_cols}")
            
        except Exception as e:
            logger.error(f"Error loading item features: {str(e)}")
            raise
        
        # Create recommendation engine
        try:
            logger.info("Creating recommendation engine...")
            engine = cls(model, user_features, item_features, logger, device=device)
            logger.info("Recommendation engine created successfully")
            return engine
        except Exception as e:
            logger.error(f"Error creating recommendation engine: {str(e)}")
            raise

@log_function()
def main():
    """Main function to run recommendation generation from command line"""
    parser = argparse.ArgumentParser(description='Generate recommendations using a trained model')
    parser.add_argument('--model', required=True, help='Path to the model checkpoint')
    parser.add_argument('--users', required=True, help='Path to user features')
    parser.add_argument('--items', required=True, help='Path to item features')
    parser.add_argument('--manifest', help='Path to feature manifest YAML file')
    parser.add_argument('--user', help='Single user ID to generate recommendations for')
    parser.add_argument('--item', help='Generate similar items to this item ID')
    parser.add_argument('--output', help='Output file path')
    parser.add_argument('--top-k', type=int, default=10, help='Number of recommendations to generate')
    parser.add_argument('--exclude-history', help='Path to user-item interactions to exclude from recommendations')
    parser.add_argument('--log-level', choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'], default='INFO', help='Logging level')
    parser.add_argument('--enhance-metadata', action='store_true', help='Enhance recommendations with metadata')
    parser.add_argument('--spotify-sample', help='Path to Spotify sample data for additional metadata')
    
    args = parser.parse_args()
    
    # Set up logging
    setup_logging(args.log_level, "generate_recommendations.log")
    logger = logging.getLogger(__name__)
    
    # Log all arguments
    logger.info("Starting recommendation generation with parameters:")
    for arg, value in vars(args).items():
        logger.info(f"  {arg}: {value}")
    
    # Load model
    logger.info(f"Loading model from {args.model}")
    engine = RecommendationEngine.from_checkpoint(
        args.model,
        user_features_path=args.users,
        item_features_path=args.items,
        feature_manifest_path=args.manifest
    )
    
    # Load exclusions if provided
    exclude_history = None
    if args.exclude_history:
        logger.info(f"Loading exclusion history from {args.exclude_history}")
        exclude_history = pd.read_csv(args.exclude_history)
    
    # Generate recommendations for a single user
    if args.user:
        logger.info(f"Generating recommendations for user {args.user}")
        recommendations = engine.predict_scores(args.user, top_k=args.top_k)
        
        # Enhance with metadata if requested
        if args.enhance_metadata:
            logger.info("Enhancing recommendations with metadata")
            recommendations = engine.enhance_recommendations_with_metadata(
                recommendations, engine.item_features, args.spotify_sample
            )
    
    # Find similar items
    elif args.item:
        logger.info(f"Finding items similar to {args.item}")
        recommendations = engine.find_similar_items(args.item, top_k=args.top_k)
        
        # Enhance with metadata if requested
        if args.enhance_metadata:
            logger.info("Enhancing recommendations with metadata")
            recommendations = engine.enhance_recommendations_with_metadata(
                recommendations, engine.item_features, args.spotify_sample
            )
    
    # Generate recommendations for all users
    else:
        logger.info("Generating recommendations for all users")
        recommendations = engine.generate_recommendations(
            top_k=args.top_k, 
            exclude_history=exclude_history,
            enhance_metadata=args.enhance_metadata,
            spotify_sample_path=args.spotify_sample
        )
    
    # Save recommendations
    if args.output:
        logger.info(f"Saving recommendations to {args.output}")
        os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
        recommendations.to_csv(args.output, index=False)
    
    # Print sample recommendations
    if len(recommendations) > 0:
        print("\nSample recommendations:")
        print(recommendations.head())
        
        if args.enhance_metadata:
            # Show all columns for the first recommendation to demonstrate the metadata
            print("\nDetailed first recommendation with metadata:")
            print(recommendations.iloc[0].to_string())
    else:
        logger.warning("No recommendations generated")
    
    logger.info("Recommendation generation completed")

if __name__ == "__main__":
    main() 