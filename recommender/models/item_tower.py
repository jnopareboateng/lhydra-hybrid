"""
Item Tower implementation for the music recommender system.

This module defines the neural network architecture for processing track features.
"""

import torch
import torch.nn as nn
import logging
from typing import Dict, List, Tuple, Union, Optional

# Configure logging
logger = logging.getLogger(__name__)


class ItemTower(nn.Module):
    """
    Item tower of the two-tower recommender architecture.
    
    This component processes track features (audio characteristics, metadata,
    popularity metrics) and outputs a track embedding vector.
    """
    
    def __init__(self, config: Dict, categorical_mappings: Optional[Dict] = None):
        """
        Initialize the item tower.
        
        Args:
            config: Dictionary containing model configuration
            categorical_mappings: Dictionary mapping categorical features to their values
        """
        super(ItemTower, self).__init__()
        
        self.config = config
        self.categorical_mappings = categorical_mappings or {}
        
        # Define embedding layers for categorical features
        self.embedding_layers = nn.ModuleDict()
        for feature_name, embedding_config in config['item_tower']['embeddings'].items():
            # Update vocab size based on actual data if available
            vocab_size = embedding_config['vocab_size']
            if feature_name in self.categorical_mappings and 'mapping' in self.categorical_mappings[feature_name]:
                vocab_size = len(self.categorical_mappings[feature_name]['mapping'])
            
            # Create embedding layer
            self.embedding_layers[feature_name] = nn.Embedding(
                num_embeddings=vocab_size,
                embedding_dim=embedding_config['embedding_dim'],
                padding_idx=0  # Reserve 0 for padding/unknown
            )
            
            logger.info(f"Created embedding layer for {feature_name} with vocab size {vocab_size}")
        
        # Define layers dynamically after first forward pass
        self.layers = None
        self.input_size = None
        self.output_size = None
        self.is_initialized = False
    
    def _init_dense_layers(self, input_size):
        """
        Initialize dense layers based on input size.
        
        Args:
            input_size: Actual input size based on available features
        """
        self.input_size = input_size
        
        # Create dense layers
        self.layers = nn.ModuleList()
        
        input_dim = self.input_size
        for layer_config in self.config['item_tower']['layers']:
            # Add dense layer
            self.layers.append(nn.Linear(input_dim, layer_config['units']))
            
            # Add activation
            if layer_config['activation'] == 'relu':
                self.layers.append(nn.ReLU())
            elif layer_config['activation'] == 'leaky_relu':
                self.layers.append(nn.LeakyReLU())
            
            # Add batch normalization if specified
            if layer_config.get('batch_norm', False):
                self.layers.append(nn.BatchNorm1d(layer_config['units']))
            
            # Add dropout if specified
            if layer_config.get('dropout', 0) > 0:
                self.layers.append(nn.Dropout(layer_config['dropout']))
            
            # Update input dim for next layer
            input_dim = layer_config['units']
        
        # Final output size
        self.output_size = input_dim
        
        logger.info(f"Initialized ItemTower with input size {self.input_size} and {len(self.layers)} layers")
        self.is_initialized = True
    
    def _calculate_input_size(self, features):
        """
        Calculate the input size for the first dense layer based on available features.
        
        Args:
            features: Dictionary of track features
            
        Returns:
            Total input size
        """
        # Count actual numerical features available
        num_numerical = 0
        for feature_name in self.config['features']['track_features']['numerical']:
            if feature_name in features:
                num_numerical += 1
        
        # Count engineered features if present
        if 'engineered_features' in self.config['features']:
            for feature_name in self.config['features']['engineered_features']:
                if feature_name in features:
                    num_numerical += 1
        
        # Count embedded dimensions from available categorical features
        embedded_dims = 0
        for feature_name, config in self.config['item_tower']['embeddings'].items():
            feature_id = f"{feature_name}_id"
            if feature_id in features:
                embedded_dims += config['embedding_dim']
        
        total_size = num_numerical + embedded_dims
        logger.info(f"Dynamic ItemTower input size: {total_size} (numerical: {num_numerical}, embeddings: {embedded_dims})")
        
        if total_size == 0:
            logger.warning("No track features are available! Using a default input size")
            return 10  # Default size if no features are available
            
        return total_size
    
    def forward(self, track_features: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Forward pass through the item tower.
        
        Args:
            track_features: Dictionary of track features
            
        Returns:
            Track embedding tensor
        """
        # Process numerical features
        numerical_features = []
        
        # Regular numerical features
        for feature_name in self.config['features']['track_features']['numerical']:
            if feature_name in track_features:
                numerical_features.append(track_features[feature_name])
        
        # Engineered features
        if 'engineered_features' in self.config['features']:
            for feature_name in self.config['features']['engineered_features']:
                if feature_name in track_features:
                    numerical_features.append(track_features[feature_name])
        
        # Process categorical features through embeddings
        embedded_features = []
        for feature_name in self.config['features']['track_features']['categorical']:
            if f"{feature_name}_id" in track_features and feature_name in self.embedding_layers:
                # Get feature tensor
                feature_tensor = track_features[f"{feature_name}_id"]
                
                # Ensure tensor is long type for embedding lookup
                if feature_tensor.dtype != torch.long:
                    feature_tensor = feature_tensor.long()
                
                # Get embedding
                embedded = self.embedding_layers[feature_name](feature_tensor)
                embedded_features.append(embedded)
        
        # Combine all features
        combined_features = []
        
        # Add numerical features
        if numerical_features:
            numerical_tensor = torch.stack(numerical_features, dim=1)
            combined_features.append(numerical_tensor)
        
        # Add embedded features
        for embedded in embedded_features:
            combined_features.append(embedded)
        
        # Concatenate all features
        if len(combined_features) > 1:
            x = torch.cat(combined_features, dim=1)
        elif len(combined_features) == 1:
            x = combined_features[0]
        else:
            # No features available, create a small random tensor as a fallback
            logger.warning("No track features available for forward pass")
            batch_size = next(iter(track_features.values())).shape[0]
            x = torch.zeros((batch_size, 1), device=next(iter(track_features.values())).device)
        
        # Initialize dense layers if this is the first forward pass
        if not self.is_initialized:
            input_size = x.shape[1]
            self._init_dense_layers(input_size)
        
        # Pass through dense layers
        for layer in self.layers:
            x = layer(x)
        
        return x 