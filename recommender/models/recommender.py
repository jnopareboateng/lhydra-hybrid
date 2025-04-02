"""
Hybrid Recommender model implementation for the music recommender system.

This module defines the complete model architecture combining the user and item towers
for generating music recommendations.
"""

import torch
import torch.nn as nn
import logging
import yaml
import os
from typing import Dict, List, Tuple, Union, Optional
import numpy as np

from .user_tower import UserTower
from .item_tower import ItemTower

# Configure logging
logger = logging.getLogger(__name__)


class HybridRecommender(nn.Module):
    """
    Two-tower hybrid recommender model architecture.
    
    This model combines user and item towers to generate music recommendations
    based on user preferences and item characteristics.
    """
    
    def __init__(self, config_path: str, categorical_mappings: Optional[Dict] = None):
        """
        Initialize the hybrid recommender model.
        
        Args:
            config_path: Path to the model configuration YAML file
            categorical_mappings: Dictionary mapping categorical features to their values
        """
        super(HybridRecommender, self).__init__()
        
        # Load configuration
        if isinstance(config_path, dict):
            # Config already provided as dict
            self.config = config_path
        else:
            # Load config from file
            self.config = self._load_config(config_path)
            
        self.categorical_mappings = categorical_mappings or {}
        
        # Initialize user and item towers
        self.user_tower = UserTower(self.config, self.categorical_mappings)
        self.item_tower = ItemTower(self.config, self.categorical_mappings)
        
        # Initialize prediction layers dynamically after first forward pass
        self.combine_method = self.config['prediction_layers']['combine_method']
        self.prediction_layers = None
        self.is_initialized = False
        
        logger.info(f"Initialized HybridRecommender with combine method: {self.combine_method}")
    
    def _init_prediction_layers(self, user_embedding_size, item_embedding_size):
        """
        Initialize prediction layers based on user and item embedding sizes.
        
        Args:
            user_embedding_size: Size of user embeddings
            item_embedding_size: Size of item embeddings
        """
        # Calculate input size for prediction layers
        if self.combine_method == 'concatenate':
            prediction_input_size = user_embedding_size + item_embedding_size
        elif self.combine_method == 'dot_product':
            # For dot product, we don't need prediction layers as we'll just compute the dot product
            prediction_input_size = 1
            
            # Check that embeddings have compatible dimensions for dot product
            if user_embedding_size != item_embedding_size:
                logger.warning(f"Embedding dimensions don't match for dot product: {user_embedding_size} vs {item_embedding_size}")
                logger.warning("Switching to concatenation method")
                self.combine_method = 'concatenate'
                prediction_input_size = user_embedding_size + item_embedding_size
        else:
            raise ValueError(f"Unsupported combine method: {self.combine_method}")
        
        # Create prediction layers
        self.prediction_layers = nn.ModuleList()
        
        if self.combine_method == 'concatenate':
            input_dim = prediction_input_size
            for layer_config in self.config['prediction_layers']['layers']:
                # Add dense layer
                self.prediction_layers.append(nn.Linear(input_dim, layer_config['units']))
                
                # Add activation (except for the last layer with sigmoid activation)
                if layer_config['activation'] == 'sigmoid':
                    self.prediction_layers.append(nn.Sigmoid())
                elif layer_config['activation'] == 'relu':
                    self.prediction_layers.append(nn.ReLU())
                elif layer_config['activation'] == 'leaky_relu':
                    self.prediction_layers.append(nn.LeakyReLU())
                
                # Add dropout if specified
                if layer_config.get('dropout', 0) > 0:
                    self.prediction_layers.append(nn.Dropout(layer_config['dropout']))
                
                # Update input dim for next layer
                input_dim = layer_config['units']
        
        logger.info(f"Initialized prediction layers with input size: {prediction_input_size}")
        self.is_initialized = True
    
    def _load_config(self, config_path: str) -> Dict:
        """
        Load model configuration from YAML file.
        
        Args:
            config_path: Path to configuration file
            
        Returns:
            Dictionary containing configuration settings
        """
        try:
            with open(config_path, 'r') as file:
                config = yaml.safe_load(file)
            logger.info(f"Successfully loaded model configuration from {config_path}")
            return config
        except Exception as e:
            logger.error(f"Failed to load configuration from {config_path}: {str(e)}")
            raise
    
    def forward(self, user_features: Dict[str, torch.Tensor], 
                track_features: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Forward pass through the recommender model.
        
        Args:
            user_features: Dictionary of user features
            track_features: Dictionary of track features
            
        Returns:
            Recommendation scores tensor
        """
        # Pass through user tower
        user_embedding = self.user_tower(user_features)
        
        # Pass through item tower
        item_embedding = self.item_tower(track_features)
        
        # Initialize prediction layers if this is the first forward pass
        if not self.is_initialized:
            self._init_prediction_layers(user_embedding.shape[1], item_embedding.shape[1])
        
        # Combine embeddings based on combine method
        if self.combine_method == 'concatenate':
            # Concatenate embeddings
            combined = torch.cat([user_embedding, item_embedding], dim=1)
            
            # Pass through prediction layers
            x = combined
            for layer in self.prediction_layers:
                x = layer(x)
            
            return x
        
        elif self.combine_method == 'dot_product':
            # Compute dot product
            # First make sure embeddings have the same dimensions
            assert user_embedding.shape[1] == item_embedding.shape[1], \
                f"Embedding dimensions do not match: {user_embedding.shape[1]} vs {item_embedding.shape[1]}"
            
            # Compute dot product and apply sigmoid
            dot_product = torch.sum(user_embedding * item_embedding, dim=1, keepdim=True)
            return torch.sigmoid(dot_product)
    
    def predict(self, user_features: Dict[str, torch.Tensor],
                track_features: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Generate recommendation scores for user-track pairs.
        
        Args:
            user_features: Dictionary of user features
            track_features: Dictionary of track features
            
        Returns:
            Tensor of recommendation scores
        """
        self.eval()  # Set model to evaluation mode
        with torch.inference_mode():
            scores = self.forward(user_features, track_features)
        return scores
    
    def recommend_tracks(self, user_features: Dict[str, torch.Tensor],
                         all_track_features: List[Dict[str, torch.Tensor]],
                         top_k: int = 10) -> Tuple[List[int], torch.Tensor]:
        """
        Recommend top-k tracks for a user.
        
        Args:
            user_features: Dictionary of user features
            all_track_features: List of dictionaries containing features for all tracks
            top_k: Number of top tracks to recommend
            
        Returns:
            Tuple of (track_indices, scores) for the top-k recommended tracks
        """
        self.eval()  # Set model to evaluation mode
        
        with torch.inference_mode():
            # Get user embedding (only need to compute once)
            user_embedding = self.user_tower(user_features)
            
            # Compute scores for all tracks
            scores = []
            for track_features in all_track_features:
                # Get track embedding
                track_embedding = self.item_tower(track_features)
                
                # Compute score based on combine method
                if self.combine_method == 'concatenate':
                    # Concatenate embeddings
                    combined = torch.cat([user_embedding, track_embedding], dim=1)
                    
                    # Pass through prediction layers
                    x = combined
                    for layer in self.prediction_layers:
                        x = layer(x)
                    
                    scores.append(x)
                
                elif self.combine_method == 'dot_product':
                    # Compute dot product
                    dot_product = torch.sum(user_embedding * track_embedding, dim=1, keepdim=True)
                    scores.append(torch.sigmoid(dot_product))
            
            # Combine scores
            scores = torch.cat(scores, dim=0)
            
            # Get top-k tracks
            top_scores, top_indices = torch.topk(scores, k=min(top_k, len(scores)))
            
            return top_indices.tolist(), top_scores
    
    def save(self, path: str):
        """
        Save the model to a file.
        
        Args:
            path: Path to save the model
        """
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save({
            'state_dict': self.state_dict(),
            'config': self.config,
            'categorical_mappings': self.categorical_mappings
        }, path)
        logger.info(f"Model saved to {path}")
    
    @classmethod
    def load(cls, path: str) -> 'HybridRecommender':
        """
        Load a saved model.
        
        Args:
            path: Path to the saved model
            
        Returns:
            Loaded HybridRecommender instance
        """
        logger.info(f"Loading model from {path}")
        
        # Load saved model
        checkpoint = torch.load(path)
        
        # Create a temporary config file
        import tempfile
        config_file = tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.yaml')
        yaml.dump(checkpoint['config'], config_file)
        config_path = config_file.name
        config_file.close()
        
        # Create new model instance
        model = cls(config_path, checkpoint.get('categorical_mappings'))
        
        # Load state dict
        model.load_state_dict(checkpoint['model_state_dict'])
        
        # Remove temporary config file
        os.unlink(config_path)
        
        logger.info(f"Model loaded successfully from {path}")
        return model
    
    def get_embeddings(self, user_data, track_data):
        """
        Extract user and item embeddings for visualization.
        
        Args:
            user_data (dict): Dictionary of user features
            track_data (dict): Dictionary of track features
            
        Returns:
            tuple: (user_embeddings, item_embeddings)
        """
        self.eval()  # Set model to evaluation mode
        with torch.no_grad():
            # Process user features
            user_embedding = self.user_tower(user_data)
            
            # Process track features
            track_embedding = self.item_tower(track_data)
            
            return user_embedding, track_embedding
    
    def get_feature_importance(self):
        """
        Get feature importance by analyzing weights.
        
        This is a simple approximation of feature importance based on
        the weights in the first layer of each tower.
        
        Returns:
            tuple: (feature_names, importance_scores)
        """
        # Default empty lists in case we can't extract features
        feature_names = []
        importance_scores = np.array([])
        
        try:
            # Try to get feature names from config
            if hasattr(self, 'config') and isinstance(self.config, dict):
                if 'feature_sets' in self.config:
                    # Get from feature_sets structure
                    user_features = self.config.get('feature_sets', {}).get('user_features', [])
                    track_features = self.config.get('feature_sets', {}).get('track_features', [])
                    feature_names = user_features + track_features
                elif 'features' in self.config:
                    # Alternative format: features structure with user_features and track_features
                    user_num = self.config.get('features', {}).get('user_features', {}).get('numerical', [])
                    user_cat = self.config.get('features', {}).get('user_features', {}).get('categorical', [])
                    track_num = self.config.get('features', {}).get('track_features', {}).get('numerical', [])
                    track_cat = self.config.get('features', {}).get('track_features', {}).get('categorical', [])
                    feature_names = user_num + user_cat + track_num + track_cat
            
            if not feature_names:
                # Fallback: create generic feature names
                logger.warning("Could not extract feature names from model config, using generic names")
                user_tower_params = sum(1 for _ in self.user_tower.parameters())
                item_tower_params = sum(1 for _ in self.item_tower.parameters())
                feature_names = [f"user_feature_{i}" for i in range(user_tower_params)] + \
                               [f"item_feature_{i}" for i in range(item_tower_params)]
            
            # Initialize importance scores
            importance_scores = np.ones(len(feature_names)) / len(feature_names)
            
            # Extract weights from user tower (if available)
            if hasattr(self.user_tower, 'dense_layers') and self.user_tower.dense_layers:
                # Model has dense_layers attribute
                first_layer = self.user_tower.dense_layers[0]
                if hasattr(first_layer, 'weight'):
                    user_weights = first_layer.weight.data.abs().mean(dim=0).cpu().numpy()
                    user_count = min(len(user_weights), len(feature_names) // 2)
                    importance_scores[:user_count] = user_weights[:user_count]
            elif hasattr(self.user_tower, 'layers') and self.user_tower.layers:
                # Try with 'layers' attribute instead
                for layer in self.user_tower.layers:
                    if hasattr(layer, 'weight'):
                        user_weights = layer.weight.data.abs().mean(dim=0).cpu().numpy()
                        user_count = min(len(user_weights), len(feature_names) // 2)
                        importance_scores[:user_count] = user_weights[:user_count]
                        break
            
            # Extract weights from item tower (if available)
            if hasattr(self.item_tower, 'dense_layers') and self.item_tower.dense_layers:
                # Model has dense_layers attribute
                first_layer = self.item_tower.dense_layers[0]
                if hasattr(first_layer, 'weight'):
                    item_weights = first_layer.weight.data.abs().mean(dim=0).cpu().numpy()
                    user_count = len(feature_names) // 2
                    item_count = min(len(item_weights), len(feature_names) - user_count)
                    importance_scores[user_count:user_count+item_count] = item_weights[:item_count]
            elif hasattr(self.item_tower, 'layers') and self.item_tower.layers:
                # Try with 'layers' attribute instead
                for layer in self.item_tower.layers:
                    if hasattr(layer, 'weight'):
                        item_weights = layer.weight.data.abs().mean(dim=0).cpu().numpy()
                        user_count = len(feature_names) // 2
                        item_count = min(len(item_weights), len(feature_names) - user_count)
                        importance_scores[user_count:user_count+item_count] = item_weights[:item_count]
                        break
            
            # Normalize importance scores
            if np.sum(importance_scores) > 0:
                importance_scores = importance_scores / np.sum(importance_scores)
            else:
                # If all zeros, use uniform distribution
                importance_scores = np.ones(len(feature_names)) / len(feature_names)
            
            logger.info(f"Generated feature importance for {len(feature_names)} features")
        except Exception as e:
            logger.error(f"Error extracting feature importance: {str(e)}")
            # Return default dummy data if extraction fails
            feature_names = [f"Feature_{i}" for i in range(10)]
            importance_scores = np.ones(10) / 10
        
        return feature_names, importance_scores 