import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import logging
import os

logger = logging.getLogger(__name__)

class TwoTowerHybridModel(nn.Module):
    """
    Two-Tower Hybrid Recommender model architecture
    
    The model consists of two separate neural networks (towers):
    1. User Tower: Processes user features
    2. Item Tower: Processes item features
    
    The two towers produce embeddings which are then combined
    to predict user-item interaction probability.
    """
    
    def __init__(self, 
                 user_input_dim,
                 item_input_dim,
                 user_hidden_dims=(128, 64, 32),
                 item_hidden_dims=(128, 64, 32),
                 embedding_dim=32,
                 final_layer_size=16,
                 dropout=0.2,
                 activation='relu'):
        """
        Initialize the Two-Tower model
        
        Args:
            user_input_dim (int): Dimension of user input features
            item_input_dim (int): Dimension of item input features
            user_hidden_dims (tuple): Dimensions of hidden layers in user tower
            item_hidden_dims (tuple): Dimensions of hidden layers in item tower
            embedding_dim (int): Dimension of the final embedding vectors
            final_layer_size (int): Size of the final MLP layer before prediction
            dropout (float): Dropout rate
            activation (str): Activation function to use ('relu', 'tanh', or 'sigmoid')
        """
        super(TwoTowerHybridModel, self).__init__()
        
        self.user_input_dim = user_input_dim
        self.item_input_dim = item_input_dim
        self.embedding_dim = embedding_dim
        self.final_layer_size = final_layer_size
        
        # Set activation function
        if activation == 'relu':
            self.activation = nn.ReLU()
        elif activation == 'tanh':
            self.activation = nn.Tanh()
        elif activation == 'sigmoid':
            self.activation = nn.Sigmoid()
        else:
            self.activation = nn.ReLU()
            logger.warning(f"Activation function '{activation}' not recognized. Using ReLU as default.")
        
        # Build user tower
        user_layers = []
        prev_dim = user_input_dim
        
        for dim in user_hidden_dims:
            user_layers.append(nn.Linear(prev_dim, dim))
            user_layers.append(self.activation)
            user_layers.append(nn.Dropout(dropout))
            prev_dim = dim
        
        # Final embedding layer for user tower
        user_layers.append(nn.Linear(prev_dim, embedding_dim))
        user_layers.append(self.activation)
        
        self.user_tower = nn.Sequential(*user_layers)
        
        # Build item tower
        item_layers = []
        prev_dim = item_input_dim
        
        for dim in item_hidden_dims:
            item_layers.append(nn.Linear(prev_dim, dim))
            item_layers.append(self.activation)
            item_layers.append(nn.Dropout(dropout))
            prev_dim = dim
        
        # Final embedding layer for item tower
        item_layers.append(nn.Linear(prev_dim, embedding_dim))
        item_layers.append(self.activation)
        
        self.item_tower = nn.Sequential(*item_layers)
        
        # Final layers for prediction
        # The input to the final layer is the concatenation of user and item embeddings
        self.prediction_layers = nn.Sequential(
            nn.Linear(embedding_dim * 2, final_layer_size),
            self.activation,
            nn.Dropout(dropout),
            nn.Linear(final_layer_size, 1),
            nn.Sigmoid()
        )
        
        # Initialize weights
        self._init_weights()
        
        logger.info(f"Initialized Two-Tower model with user input dim: {user_input_dim}, "
                   f"item input dim: {item_input_dim}, embedding dim: {embedding_dim}")
    
    def _init_weights(self):
        """Initialize weights using Xavier initialization"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
        logger.info("Initialized weights")
    
    
    def user_embedding(self, user_features):
        """
        Get user embeddings
        
        Args:
            user_features (dict): Dictionary of user features tensors
            
        Returns:
            torch.Tensor: User embedding
        """
        # Concatenate all user features
        if isinstance(user_features, dict):
            # Get all feature tensors and concatenate them
            feature_tensors = []
            batch_size = None
            
            for key, value in user_features.items():
                if isinstance(value, torch.Tensor):
                    # Ensure tensor is at least 2D
                    if value.dim() == 1:
                        value = value.unsqueeze(0)
                    
                    # Get or set batch size
                    current_batch_size = value.size(0)
                    if batch_size is None:
                        batch_size = current_batch_size
                    elif current_batch_size != batch_size:
                        # Resize tensor to match batch size
                        if current_batch_size == 1:
                            value = value.expand(batch_size, -1)
                        elif batch_size == 1:
                            # Resize all previous tensors
                            feature_tensors = [t.expand(current_batch_size, -1) for t in feature_tensors]
                            batch_size = current_batch_size
                        else:
                            raise ValueError(f"Batch size mismatch: {current_batch_size} vs {batch_size}")
                    
                    feature_tensors.append(value)
            
            if feature_tensors:
                # Concatenate along feature dimension
                user_features = torch.cat(feature_tensors, dim=-1)
            else:
                raise ValueError("No valid feature tensors found in user_features dictionary")
        
        return self.user_tower(user_features)
    
    def item_embedding(self, item_features):
        """
        Get item embeddings
        
        Args:
            item_features (dict): Dictionary of item features tensors
            
        Returns:
            torch.Tensor: Item embedding
        """
        # Concatenate all item features
        if isinstance(item_features, dict):
            # Get all feature tensors and concatenate them
            feature_tensors = []
            batch_size = None
            
            for key, value in item_features.items():
                if isinstance(value, torch.Tensor):
                    # Ensure tensor is at least 2D
                    if value.dim() == 1:
                        value = value.unsqueeze(0)
                    
                    # Get or set batch size
                    current_batch_size = value.size(0)
                    if batch_size is None:
                        batch_size = current_batch_size
                    elif current_batch_size != batch_size:
                        # Resize tensor to match batch size
                        if current_batch_size == 1:
                            value = value.expand(batch_size, -1)
                        elif batch_size == 1:
                            # Resize all previous tensors
                            feature_tensors = [t.expand(current_batch_size, -1) for t in feature_tensors]
                            batch_size = current_batch_size
                        else:
                            raise ValueError(f"Batch size mismatch: {current_batch_size} vs {batch_size}")
                    
                    feature_tensors.append(value)
            
            if feature_tensors:
                # Concatenate along feature dimension
                item_features = torch.cat(feature_tensors, dim=-1)
            else:
                raise ValueError("No valid feature tensors found in item_features dictionary")
        
        return self.item_tower(item_features)
    
    def forward(self, user_features, item_features):
        """
        Forward pass
        
        Args:
            user_features (dict or torch.Tensor): User features dictionary or tensor
            item_features (dict or torch.Tensor): Item features dictionary or tensor
            
        Returns:
            torch.Tensor: Predicted interaction probability
        """
        # Align features with manifest if available
        if hasattr(self, 'feature_manifest') and self.feature_manifest is not None:
            user_features, item_features = self.align_features(user_features, item_features)
        
        # Get embeddings from both towers
        user_embedding = self.user_embedding(user_features)
        item_embedding = self.item_embedding(item_features)
        
        # Concatenate embeddings
        combined = torch.cat([user_embedding, item_embedding], dim=1)
        
        # Final prediction
        prediction = self.prediction_layers(combined)
        
        return prediction.squeeze()
    
    def predict(self, user_features, item_features):
        """
        Make prediction for a batch of user-item pairs
        
        Args:
            user_features (torch.Tensor): User features tensor
            item_features (torch.Tensor): Item features tensor
            
        Returns:
            numpy.ndarray: Predicted probabilities
        """
        self.eval()
        # with torch.no_grad():
        with torch.inference_mode():
            predictions = self.forward(user_features, item_features)
            return predictions.cpu().numpy()
    
    def get_embeddings(self, user_features=None, item_features=None):
        """
        Get embeddings for users and/or items
        
        Args:
            user_features (torch.Tensor, optional): User features tensor
            item_features (torch.Tensor, optional): Item features tensor
            
        Returns:
            dict: Dictionary containing embeddings
        """
        self.eval()
        embeddings = {}
        
        # with torch.no_grad():
        with torch.inference_mode():
            if user_features is not None:
                user_emb = self.user_embedding(user_features)
                embeddings['user_embeddings'] = user_emb.cpu().numpy()
            
            if item_features is not None:
                item_emb = self.item_embedding(item_features)
                embeddings['item_embeddings'] = item_emb.cpu().numpy()
        
        return embeddings
    
    def save(self, path, feature_manifest=None):
        """
        Save model to disk
        
        Args:
            path (str): Path to save the model
            feature_manifest (dict, optional): Feature manifest from preprocessing
        """
        model_info = {
            'model_state_dict': self.state_dict(),
            'user_input_dim': self.user_input_dim,
            'item_input_dim': self.item_input_dim,
            'embedding_dim': self.embedding_dim,
            'final_layer_size': self.final_layer_size,
            'user_hidden_dims': self.user_tower[0].out_features,  # Get hidden dims from first layer
            'item_hidden_dims': self.item_tower[0].out_features,  # Get hidden dims from first layer
            'dropout': self.user_tower[2].p,  # Get dropout rate from first dropout layer
            'feature_manifest': feature_manifest  # Store feature manifest if provided
        }
        
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(path), exist_ok=True)
        
        # Save with weights_only=True to address the FutureWarning
        torch.save(model_info, path)
        logger.info(f"Model saved to {path}")
    
    @classmethod
    def load(cls, path, device=None, manifest_path=None):
        """
        Load model from disk
        
        Args:
            path (str): Path to the saved model
            device (torch.device, optional): Device to load the model to
            manifest_path (str, optional): Path to feature manifest file
            
        Returns:
            TwoTowerHybridModel: Loaded model
        """
        if device is None:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Load with weights_only=True to address the FutureWarning
        model_info = torch.load(path, map_location=device, weights_only=True)
        
        # Extract model parameters
        user_input_dim = model_info['user_input_dim']
        item_input_dim = model_info['item_input_dim']
        embedding_dim = model_info.get('embedding_dim', 32)
        final_layer_size = model_info.get('final_layer_size', 16)
        
        # Load feature manifest if provided or if stored in model
        feature_manifest = None
        if manifest_path is not None and os.path.exists(manifest_path):
            import yaml
            with open(manifest_path, 'r') as f:
                feature_manifest = yaml.safe_load(f)
            logger.info(f"Loaded feature manifest from {manifest_path}")
        elif 'feature_manifest' in model_info and model_info['feature_manifest'] is not None:
            feature_manifest = model_info['feature_manifest']
            logger.info("Using feature manifest from model checkpoint")
        
        # Check if manifest dimensions match model dimensions
        if feature_manifest is not None:
            manifest_user_dim = feature_manifest.get('model_dimensions', {}).get('user_input_dim')
            manifest_item_dim = feature_manifest.get('model_dimensions', {}).get('item_input_dim')
            
            if manifest_user_dim is not None and manifest_user_dim != user_input_dim:
                logger.warning(f"User input dimension mismatch: model={user_input_dim}, manifest={manifest_user_dim}")
            
            if manifest_item_dim is not None and manifest_item_dim != item_input_dim:
                logger.warning(f"Item input dimension mismatch: model={item_input_dim}, manifest={manifest_item_dim}")
        
        # Create model instance
        model = cls(
            user_input_dim=user_input_dim,
            item_input_dim=item_input_dim,
            embedding_dim=embedding_dim,
            final_layer_size=final_layer_size
        )
        
        # Store feature manifest in model
        model.feature_manifest = feature_manifest
        
        # Load state dict with strict=False to handle missing keys
        try:
            model.load_state_dict(model_info['model_state_dict'], strict=False)
            logger.info("Model state dict loaded successfully")
        except Exception as e:
            logger.warning(f"Error loading state dict: {str(e)}")
            logger.warning("Attempting to load with partial state dict...")
            # Try to load with partial state dict
            state_dict = model_info['model_state_dict']
            model_dict = model.state_dict()
            
            # Filter out missing keys
            filtered_state_dict = {k: v for k, v in state_dict.items() if k in model_dict}
            
            # Update model state dict with filtered state dict
            model_dict.update(filtered_state_dict)
            model.load_state_dict(model_dict)
            logger.info("Model loaded with partial state dict")
        
        model.to(device)
        
        logger.info(f"Model loaded from {path}")
        return model 

    def align_features(self, user_features, item_features):
        """
        Align the provided user and item features with the expected features based on the manifest.
        
        Args:
            user_features (dict): User features
            item_features (dict): Item features
            
        Returns:
            tuple: (aligned_user_features, aligned_item_features)
        """
        # If no feature manifest available, just return the original features
        if not hasattr(self, 'feature_manifest') or self.feature_manifest is None:
            logger.warning("No feature manifest available for alignment. Using features as-is.")
            return user_features, item_features
        
        logger.info("Aligning features based on manifest")
        
        # Create aligned features dictionaries
        aligned_user = {}
        aligned_item = {}
        
        # Get expected user feature columns
        expected_user_cols = self.feature_manifest.get('user_features', {}).get('columns', [])
        
        if isinstance(user_features, dict):
            for key, value in user_features.items():
                # Only keep ID fields and features mentioned in the manifest
                if key == 'user_id' or key in expected_user_cols:
                    aligned_user[key] = value
                    
            # Check for missing expected features
            for col in expected_user_cols:
                if col not in aligned_user:
                    logger.warning(f"Missing expected user feature: {col}")
                    # Create zero tensor of appropriate shape for missing features
                    batch_size = next(iter(user_features.values())).shape[0]
                    aligned_user[col] = torch.zeros(batch_size, 1, device=next(iter(user_features.values())).device)
        else:
            # If user_features is already a tensor, we can't align it
            logger.warning("User features is already a tensor, cannot align with manifest")
            aligned_user = user_features
        
        # Get expected item feature columns
        expected_item_cols = self.feature_manifest.get('item_features', {}).get('columns', [])
        
        if isinstance(item_features, dict):
            for key, value in item_features.items():
                # Only keep ID fields and features mentioned in the manifest
                if key == 'track_id' or key == 'item_id' or key in expected_item_cols:
                    aligned_item[key] = value
                    
            # Check for missing expected features
            for col in expected_item_cols:
                if col not in aligned_item:
                    logger.warning(f"Missing expected item feature: {col}")
                    # Create zero tensor of appropriate shape for missing features
                    batch_size = next(iter(item_features.values())).shape[0]
                    aligned_item[col] = torch.zeros(batch_size, 1, device=next(iter(item_features.values())).device)
        else:
            # If item_features is already a tensor, we can't align it
            logger.warning("Item features is already a tensor, cannot align with manifest")
            aligned_item = item_features
        
        logger.info(f"Feature alignment complete: {len(aligned_user)} user features, {len(aligned_item)} item features")
        
        return aligned_user, aligned_item 