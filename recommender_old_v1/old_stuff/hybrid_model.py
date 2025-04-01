"""

NO USED IN CURRENT PIPELINE

HABIBI DON'T TOUCH THIS FILE, YOOO

"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import sys
import numpy as np
from typing import Dict, List, Tuple, Optional

# Add root directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.logger import LhydraLogger, log_function

class UserTower(nn.Module):
    """
    User Tower for the Hybrid Music Recommender System.
    Processes user features including demographics and audio preferences.
    """
    
    def __init__(self, config: Dict, logger=None):
        """
        Initialize the User Tower.
        
        Args:
            config (Dict): Configuration dictionary with model parameters.
            logger (LhydraLogger): Logger instance.
        """
        super(UserTower, self).__init__()
        self.logger = logger or LhydraLogger()
        self.logger.info("Initializing UserTower")
        
        self.config = config
        
        # Get dimensions and cardinalities
        self.feature_dims = config['feature_dims']['user'] 
        self.cardinalities = config['id_cardinalities']
        self.embedding_dim = config['embedding_dim']
        self.hidden_dims = config['hidden_dims']
        self.dropout_rate = config['dropout']
        
        # User ID embedding
        if 'user_id' in self.cardinalities:
            self.user_embedding = nn.Embedding(
                self.cardinalities['user_id'], 
                self.embedding_dim,
                padding_idx=0
            )
            self.logger.debug(f"Created user embedding layer with cardinality {self.cardinalities['user_id']}")
        else:
            self.user_embedding = None
            self.logger.warning("No user_id found in cardinalities. User embedding will not be created.")
        
        # Demographic features processing
        if 'demographics' in self.feature_dims:
            self.demographic_dim = self.feature_dims['demographics']
            self.demo_fc = nn.Linear(self.demographic_dim, self.embedding_dim)
            self.logger.debug(f"Created demographic features layer with input dim {self.demographic_dim}")
        else:
            self.demographic_dim = 0
            self.demo_fc = None
            
        # Audio profile processing (user's average audio preferences)
        if 'audio_profile' in self.feature_dims:
            self.audio_profile_dim = self.feature_dims['audio_profile']
            self.audio_profile_fc = nn.Linear(self.audio_profile_dim, self.embedding_dim)
            self.logger.debug(f"Created audio profile layer with input dim {self.audio_profile_dim}")
        else:
            self.audio_profile_dim = 0
            self.audio_profile_fc = None
        
        # Calculate total dimension of all concatenated user features
        self.total_concat_dim = 0
        if self.user_embedding is not None:
            self.total_concat_dim += self.embedding_dim
        if self.demo_fc is not None:
            self.total_concat_dim += self.embedding_dim
        if self.audio_profile_fc is not None:
            self.total_concat_dim += self.embedding_dim
            
        # If no features, create a default representation
        if self.total_concat_dim == 0:
            self.logger.warning("No user features found. Creating default representation.")
            self.total_concat_dim = self.embedding_dim
            self.default_representation = nn.Parameter(torch.zeros(1, self.embedding_dim))
            nn.init.normal_(self.default_representation, mean=0.0, std=0.01)
        
        # Create tower layers
        layers = []
        input_dim = self.total_concat_dim
        
        for hidden_dim in self.hidden_dims:
            layers.append(nn.Linear(input_dim, hidden_dim))
            layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(self.dropout_rate))
            input_dim = hidden_dim
            
        self.tower = nn.Sequential(*layers)
        self.logger.debug(f"Created user tower with architecture: {self.tower}")
        
        # Final output layer for user representation
        self.output_dim = self.hidden_dims[-1] if self.hidden_dims else self.total_concat_dim
        
        self.logger.info(f"UserTower initialized with output dimension {self.output_dim}")
    
    @log_function()
    def forward(self, user_features: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Forward pass through the User Tower.
        
        Args:
            user_features (Dict[str, torch.Tensor]): Dictionary of user features.
            
        Returns:
            torch.Tensor: User representation vector.
        """
        feature_vectors = []
        
        # Process user ID embedding if available
        if self.user_embedding is not None and 'user_id' in user_features:
            user_id = user_features['user_id']
            user_emb = self.user_embedding(user_id)
            feature_vectors.append(user_emb)
        
        # Process demographic features if available
        if self.demo_fc is not None and 'demographics' in user_features:
            demographics = user_features['demographics']
            demo_emb = self.demo_fc(demographics)
            demo_emb = F.relu(demo_emb)
            feature_vectors.append(demo_emb)
        
        # Process audio profile if available
        if self.audio_profile_fc is not None and 'audio_profile' in user_features:
            audio_profile = user_features['audio_profile']
            audio_emb = self.audio_profile_fc(audio_profile)
            audio_emb = F.relu(audio_emb)
            feature_vectors.append(audio_emb)
            
        # If no features were processed, use default representation
        if not feature_vectors and hasattr(self, 'default_representation'):
            batch_size = next(iter(user_features.values())).size(0) if user_features else 1
            feature_vectors.append(self.default_representation.expand(batch_size, -1))
        
        # Concatenate all feature vectors
        if feature_vectors:
            combined = torch.cat(feature_vectors, dim=1)
        else:
            # Handle case with no features
            batch_size = next(iter(user_features.values())).size(0) if user_features else 1
            combined = torch.zeros(batch_size, self.total_concat_dim, device=self._get_device())
            
        # Pass through tower layers
        user_representation = self.tower(combined)
        
        return user_representation
    
    def _get_device(self):
        """Get the device of the model parameters."""
        return next(self.parameters()).device


class ItemTower(nn.Module):
    """
    Item Tower for the Hybrid Music Recommender System.
    Processes item features including track metadata, audio features, and temporal data.
    """
    
    def __init__(self, config: Dict, logger=None):
        """
        Initialize the Item Tower.
        
        Args:
            config (Dict): Configuration dictionary with model parameters.
            logger (LhydraLogger): Logger instance.
        """
        super(ItemTower, self).__init__()
        self.logger = logger or LhydraLogger()
        self.logger.info("Initializing ItemTower")
        
        self.config = config
        
        # Get dimensions and cardinalities
        self.feature_dims = config['feature_dims']['item']
        self.cardinalities = config['id_cardinalities']
        self.embedding_dim = config['embedding_dim']
        self.hidden_dims = config['hidden_dims']
        self.dropout_rate = config['dropout']
        
        # Track ID embedding
        if 'track_id' in self.cardinalities:
            self.track_embedding = nn.Embedding(
                self.cardinalities['track_id'], 
                self.embedding_dim,
                padding_idx=0
            )
            self.logger.debug(f"Created track embedding layer with cardinality {self.cardinalities['track_id']}")
        else:
            self.track_embedding = None
            self.logger.warning("No track_id found in cardinalities. Track embedding will not be created.")
        
        # Artist embedding
        if 'artist' in self.cardinalities:
            self.artist_embedding = nn.Embedding(
                self.cardinalities['artist'], 
                self.embedding_dim,
                padding_idx=0
            )
            self.logger.debug(f"Created artist embedding layer with cardinality {self.cardinalities['artist']}")
        else:
            self.artist_embedding = None
            
        # Genre features processing
        if 'genre' in self.feature_dims:
            self.genre_dim = self.feature_dims['genre']
            self.genre_fc = nn.Linear(self.genre_dim, self.embedding_dim)
            self.logger.debug(f"Created genre features layer with input dim {self.genre_dim}")
        else:
            self.genre_dim = 0
            self.genre_fc = None
            
        # Audio features processing
        if 'audio_features' in self.feature_dims:
            self.audio_dim = self.feature_dims['audio_features']
            self.audio_fc = nn.Linear(self.audio_dim, self.embedding_dim)
            self.logger.debug(f"Created audio features layer with input dim {self.audio_dim}")
        else:
            self.audio_dim = 0
            self.audio_fc = None
            
        # Temporal features processing
        if 'temporal' in self.feature_dims:
            self.temporal_dim = self.feature_dims['temporal']
            self.temporal_fc = nn.Linear(self.temporal_dim, self.embedding_dim)
            self.logger.debug(f"Created temporal features layer with input dim {self.temporal_dim}")
        else:
            self.temporal_dim = 0
            self.temporal_fc = None
        
        # Calculate total dimension of all concatenated item features
        self.total_concat_dim = 0
        if self.track_embedding is not None:
            self.total_concat_dim += self.embedding_dim
        if self.artist_embedding is not None:
            self.total_concat_dim += self.embedding_dim
        if self.genre_fc is not None:
            self.total_concat_dim += self.embedding_dim
        if self.audio_fc is not None:
            self.total_concat_dim += self.embedding_dim
        if self.temporal_fc is not None:
            self.total_concat_dim += self.embedding_dim
            
        # If no features, create a default representation
        if self.total_concat_dim == 0:
            self.logger.warning("No item features found. Creating default representation.")
            self.total_concat_dim = self.embedding_dim
            self.default_representation = nn.Parameter(torch.zeros(1, self.embedding_dim))
            nn.init.normal_(self.default_representation, mean=0.0, std=0.01)
        
        # Create tower layers
        layers = []
        input_dim = self.total_concat_dim
        
        for hidden_dim in self.hidden_dims:
            layers.append(nn.Linear(input_dim, hidden_dim))
            layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(self.dropout_rate))
            input_dim = hidden_dim
            
        self.tower = nn.Sequential(*layers)
        self.logger.debug(f"Created item tower with architecture: {self.tower}")
        
        # Final output dimension
        self.output_dim = self.hidden_dims[-1] if self.hidden_dims else self.total_concat_dim
        
        self.logger.info(f"ItemTower initialized with output dimension {self.output_dim}")
    
    @log_function()
    def forward(self, item_features: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Forward pass through the Item Tower.
        
        Args:
            item_features (Dict[str, torch.Tensor]): Dictionary of item features.
            
        Returns:
            torch.Tensor: Item representation vector.
        """
        feature_vectors = []
        
        # Process track ID embedding if available
        if self.track_embedding is not None and 'track_id' in item_features:
            track_id = item_features['track_id']
            track_emb = self.track_embedding(track_id)
            feature_vectors.append(track_emb)
        
        # Process artist ID embedding if available
        if self.artist_embedding is not None and 'artist' in item_features:
            artist_id = item_features['artist']
            artist_emb = self.artist_embedding(artist_id)
            feature_vectors.append(artist_emb)
        
        # Process genre features if available
        if self.genre_fc is not None and 'genre' in item_features:
            genre = item_features['genre']
            genre_emb = self.genre_fc(genre)
            genre_emb = F.relu(genre_emb)
            feature_vectors.append(genre_emb)
        
        # Process audio features if available
        if self.audio_fc is not None and 'audio_features' in item_features:
            audio = item_features['audio_features']
            audio_emb = self.audio_fc(audio)
            audio_emb = F.relu(audio_emb)
            feature_vectors.append(audio_emb)
        
        # Process temporal features if available
        if self.temporal_fc is not None and 'temporal' in item_features:
            temporal = item_features['temporal']
            temporal_emb = self.temporal_fc(temporal)
            temporal_emb = F.relu(temporal_emb)
            feature_vectors.append(temporal_emb)
            
        # If no features were processed, use default representation
        if not feature_vectors and hasattr(self, 'default_representation'):
            batch_size = next(iter(item_features.values())).size(0) if item_features else 1
            feature_vectors.append(self.default_representation.expand(batch_size, -1))
        
        # Concatenate all feature vectors
        if feature_vectors:
            combined = torch.cat(feature_vectors, dim=1)
        else:
            # Handle case with no features
            batch_size = next(iter(item_features.values())).size(0) if item_features else 1
            combined = torch.zeros(batch_size, self.total_concat_dim, device=self._get_device())
            
        # Pass through tower layers
        item_representation = self.tower(combined)
        
        return item_representation
    
    def _get_device(self):
        """Get the device of the model parameters."""
        return next(self.parameters()).device


class HybridMusicRecommender(nn.Module):
    """
    Hybrid Music Recommender System combining collaborative filtering with content-based features.
    Uses a two-tower architecture to process user and item features separately.
    """
    
    def __init__(self, config: Dict, logger=None):
        """
        Initialize the Hybrid Music Recommender.
        
        Args:
            config (Dict): Configuration dictionary with model parameters.
            logger (LhydraLogger): Logger instance.
        """
        super(HybridMusicRecommender, self).__init__()
        self.logger = logger or LhydraLogger()
        self.logger.info("Initializing HybridMusicRecommender")
        
        self.config = config
        
        # Create the user and item towers
        self.user_tower = UserTower(config, logger=self.logger)
        self.item_tower = ItemTower(config, logger=self.logger)
        
        # Get output dimensions from towers
        self.user_dim = self.user_tower.output_dim
        self.item_dim = self.item_tower.output_dim
        
        # Create prediction layers
        self.prediction_dim = self.user_dim + self.item_dim
        hidden_dims = config.get('prediction_dims', [64, 32])
        
        pred_layers = []
        input_dim = self.prediction_dim
        
        for hidden_dim in hidden_dims:
            pred_layers.append(nn.Linear(input_dim, hidden_dim))
            pred_layers.append(nn.BatchNorm1d(hidden_dim))
            pred_layers.append(nn.ReLU())
            pred_layers.append(nn.Dropout(config['dropout']))
            input_dim = hidden_dim
            
        # Final output layer for binary classification (high engagement or not)
        pred_layers.append(nn.Linear(input_dim, 1))
        
        self.prediction_layers = nn.Sequential(*pred_layers)
        self.logger.debug(f"Created prediction layers: {self.prediction_layers}")
        
        # Initialize weights
        self._init_weights()
        
        self.logger.info("HybridMusicRecommender initialized successfully")
    
    def _init_weights(self):
        """Initialize weights using Xavier initialization."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, mean=0.0, std=0.01)
    
    @log_function()
    def forward(self, batch: Dict[str, Dict[str, torch.Tensor]]) -> torch.Tensor:
        """
        Forward pass through the entire model.
        
        Args:
            batch (Dict): Batch of data containing user_features and item_features.
            
        Returns:
            torch.Tensor: Prediction scores for engagement.
        """
        # Extract features from batch
        user_features = batch['user_features']
        item_features = batch['item_features']
        
        # Get user and item representations
        user_repr = self.user_tower(user_features)
        item_repr = self.item_tower(item_features)
        
        # Concatenate user and item representations
        combined = torch.cat([user_repr, item_repr], dim=1)
        
        # Pass through prediction layers
        logits = self.prediction_layers(combined)
        
        # Apply sigmoid to get probabilities
        predictions = torch.sigmoid(logits)
        
        return predictions.squeeze()
    
    @log_function()
    def get_user_embedding(self, user_features: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Get the user embedding for given user features.
        
        Args:
            user_features (Dict[str, torch.Tensor]): Dictionary of user features.
            
        Returns:
            torch.Tensor: User representation vector.
        """
        return self.user_tower(user_features)
    
    @log_function()
    def get_item_embedding(self, item_features: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Get the item embedding for given item features.
        
        Args:
            item_features (Dict[str, torch.Tensor]): Dictionary of item features.
            
        Returns:
            torch.Tensor: Item representation vector.
        """
        return self.item_tower(item_features)
    
    @log_function()
    def calculate_loss(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Calculate binary cross entropy loss for the predictions.
        
        Args:
            predictions (torch.Tensor): Predicted engagement probabilities.
            targets (torch.Tensor): Ground truth engagement labels.
            
        Returns:
            torch.Tensor: Loss value.
        """
        return F.binary_cross_entropy(predictions, targets)
    
    @log_function()
    def training_step(self, batch: Dict) -> Dict:
        """
        Perform a training step.
        
        Args:
            batch (Dict): Batch of data containing user_features, item_features, and target.
            
        Returns:
            Dict: Dictionary with loss and predictions.
        """
        # Get predictions
        predictions = self(batch)
        
        # Calculate loss
        loss = self.calculate_loss(predictions, batch['target'])
        
        # Calculate accuracy
        binary_preds = (predictions > 0.5).float()
        accuracy = (binary_preds == batch['target']).float().mean()
        
        return {
            'loss': loss,
            'predictions': predictions,
            'accuracy': accuracy
        }
    
    @log_function()
    def validation_step(self, batch: Dict) -> Dict:
        """
        Perform a validation step.
        
        Args:
            batch (Dict): Batch of data containing user_features, item_features, and target.
            
        Returns:
            Dict: Dictionary with loss, predictions, and metrics.
        """
        # Get predictions
        predictions = self(batch)
        
        # Calculate loss
        loss = self.calculate_loss(predictions, batch['target'])
        
        # Calculate metrics
        binary_preds = (predictions > 0.5).float()
        accuracy = (binary_preds == batch['target']).float().mean()
        
        # Calculate precision, recall, and F1
        true_positives = ((binary_preds == 1) & (batch['target'] == 1)).float().sum()
        predicted_positives = (binary_preds == 1).float().sum()
        actual_positives = (batch['target'] == 1).float().sum()
        
        precision = true_positives / (predicted_positives + 1e-8)
        recall = true_positives / (actual_positives + 1e-8)
        f1 = 2 * precision * recall / (precision + recall + 1e-8)
        
        return {
            'val_loss': loss,
            'val_accuracy': accuracy,
            'val_precision': precision,
            'val_recall': recall,
            'val_f1': f1
        }
    
    @log_function()
    def save(self, path: str) -> None:
        """
        Save the model to disk.
        
        Args:
            path (str): Path to save the model.
        """
        os.makedirs(os.path.dirname(path), exist_ok=True)
        self.logger.log_file_access(path, "write")
        
        # Save model state and config
        state_dict = {
            'model_state_dict': self.state_dict(),
            'config': self.config
        }
        torch.save(state_dict, path)
        
        self.logger.info(f"Model saved to {path}")
    
    @classmethod
    @log_function()
    def load(cls, path: str, logger=None) -> 'HybridMusicRecommender':
        """
        Load a model from disk.
        
        Args:
            path (str): Path to load the model from.
            logger (LhydraLogger, optional): Logger instance.
            
        Returns:
            HybridMusicRecommender: Loaded model.
        """
        logger = logger or LhydraLogger()
        logger.log_file_access(path, "read")
        
        # Load model state and config
        state_dict = torch.load(path, map_location=torch.device('cpu'))
        
        # Create model with loaded config
        model = cls(state_dict['config'], logger=logger)
        
        # Load model state
        model.load_state_dict(state_dict['model_state_dict'])
        
        logger.info(f"Model loaded from {path}")
        return model


class HybridRecommenderConfig:
    """
    Configuration class for the Hybrid Music Recommender System.
    """
    
    @staticmethod
    @log_function()
    def get_default_config() -> Dict:
        """
        Get default configuration for the model.
        
        Returns:
            Dict: Default configuration dictionary.
        """
        return {
            'feature_dims': {
                'user': {
                    'demographics': 10,
                    'audio_profile': 12
                },
                'item': {
                    'genre': 10,
                    'audio_features': 12,
                    'temporal': 3
                }
            },
            'id_cardinalities': {
                'user_id': 1000,
                'track_id': 10000,
                'artist': 2000
            },
            'embedding_dim': 32,
            'hidden_dims': [128, 64],
            'prediction_dims': [64, 32],
            'dropout': 0.2,
            'learning_rate': 0.001,
            'weight_decay': 1e-5,
            'batch_size': 64
        }
    
    @staticmethod
    @log_function()
    def merge_with_dataloader_config(default_config: Dict, dataloader_config: Dict) -> Dict:
        """
        Merge default config with dataloader-derived config.
        
        Args:
            default_config (Dict): Default configuration.
            dataloader_config (Dict): Configuration derived from dataloader.
            
        Returns:
            Dict: Merged configuration.
        """
        config = default_config.copy()
        
        # Merge feature dimensions
        config['feature_dims']['user'].update(dataloader_config.get('feature_dims', {}).get('user', {}))
        config['feature_dims']['item'].update(dataloader_config.get('feature_dims', {}).get('item', {}))
        
        # Merge ID cardinalities
        config['id_cardinalities'].update(dataloader_config.get('id_cardinalities', {}))
        
        # Override other parameters
        for key in ['embedding_dim', 'hidden_dims', 'batch_size']:
            if key in dataloader_config:
                config[key] = dataloader_config[key]
                
        return config


if __name__ == "__main__":
    # Example usage
    import logging
    logger = LhydraLogger(log_dir="logs", log_level=logging.INFO)
    
    logger.info("Testing HybridMusicRecommender model")
    
    try:
        # Get default config
        config = HybridRecommenderConfig.get_default_config()
        
        # Create model
        model = HybridMusicRecommender(config, logger=logger)
        
        # Print model summary
        logger.info(f"Model created with architecture:")
        logger.info(f"User Tower: {model.user_tower}")
        logger.info(f"Item Tower: {model.item_tower}")
        logger.info(f"Prediction Layers: {model.prediction_layers}")
        
        # Test with dummy batch
        batch_size = 4
        user_id = torch.randint(0, 1000, (batch_size,))
        demographics = torch.randn(batch_size, config['feature_dims']['user']['demographics'])
        audio_profile = torch.randn(batch_size, config['feature_dims']['user']['audio_profile'])
        
        track_id = torch.randint(0, 10000, (batch_size,))
        genre = torch.randn(batch_size, config['feature_dims']['item']['genre'])
        audio_features = torch.randn(batch_size, config['feature_dims']['item']['audio_features'])
        temporal = torch.randn(batch_size, config['feature_dims']['item']['temporal'])
        
        target = torch.randint(0, 2, (batch_size,)).float()
        
        batch = {
            'user_features': {
                'user_id': user_id,
                'demographics': demographics,
                'audio_profile': audio_profile
            },
            'item_features': {
                'track_id': track_id,
                'genre': genre,
                'audio_features': audio_features,
                'temporal': temporal
            },
            'target': target
        }
        
        # Forward pass
        predictions = model(batch)
        logger.info(f"Model predictions shape: {predictions.shape}")
        
        # Training step
        train_result = model.training_step(batch)
        logger.info(f"Training loss: {train_result['loss'].item():.4f}")
        logger.info(f"Training accuracy: {train_result['accuracy'].item():.4f}")
        
        # Save and load model
        save_path = "models/test_hybrid_model.pt"
        model.save(save_path)
        loaded_model = HybridMusicRecommender.load(save_path, logger=logger)
        logger.info("Model saved and loaded successfully")
        
        logger.info("HybridMusicRecommender test completed successfully")
        
    except Exception as e:
        logger.error(f"Error testing HybridMusicRecommender: {str(e)}", exc_info=e) 