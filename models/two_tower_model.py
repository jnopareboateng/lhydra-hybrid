import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import logging

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
            user_features (torch.Tensor): User features tensor
            
        Returns:
            torch.Tensor: User embedding
        """
        return self.user_tower(user_features)
    
    def item_embedding(self, item_features):
        """
        Get item embeddings
        
        Args:
            item_features (torch.Tensor): Item features tensor
            
        Returns:
            torch.Tensor: Item embedding
        """
        return self.item_tower(item_features)
    
    def forward(self, user_features, item_features):
        """
        Forward pass
        
        Args:
            user_features (torch.Tensor): User features tensor
            item_features (torch.Tensor): Item features tensor
            
        Returns:
            torch.Tensor: Predicted interaction probability
        """
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
    
    def save(self, path):
        """
        Save model to disk
        
        Args:
            path (str): Path to save the model
        """
        model_info = {
            'model_state_dict': self.state_dict(),
            'user_input_dim': self.user_input_dim,
            'item_input_dim': self.item_input_dim,
            'embedding_dim': self.embedding_dim,
            'final_layer_size': self.final_layer_size
        }
        torch.save(model_info, path)
        logger.info(f"Model saved to {path}")
    
    @classmethod
    def load(cls, path, device=None):
        """
        Load model from disk
        
        Args:
            path (str): Path to the saved model
            device (torch.device, optional): Device to load the model to
            
        Returns:
            TwoTowerHybridModel: Loaded model
        """
        if device is None:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        model_info = torch.load(path, map_location=device)
        
        # Extract model parameters
        user_input_dim = model_info['user_input_dim']
        item_input_dim = model_info['item_input_dim']
        embedding_dim = model_info.get('embedding_dim', 32)
        final_layer_size = model_info.get('final_layer_size', 16)
        
        # Create model instance
        model = cls(
            user_input_dim=user_input_dim,
            item_input_dim=item_input_dim,
            embedding_dim=embedding_dim,
            final_layer_size=final_layer_size
        )
        
        # Load weights
        model.load_state_dict(model_info['model_state_dict'])
        model.to(device)
        
        logger.info(f"Model loaded from {path}")
        return model 