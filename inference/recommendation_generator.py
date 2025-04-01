import os
import sys
import torch
import numpy as np
import pandas as pd
import logging
from tqdm import tqdm
from collections import defaultdict

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.two_tower_model import TwoTowerHybridModel
from utils.logging_utils import log_inference_results

logger = logging.getLogger(__name__)

class RecommendationGenerator:
    """
    Class for generating recommendations using the trained model
    """
    
    def __init__(self, model, item_features, user_features=None, device=None):
        """
        Initialize the recommendation generator
        
        Args:
            model (TwoTowerHybridModel): Trained model
            item_features (pandas.DataFrame): Item features
            user_features (pandas.DataFrame, optional): User features
            device (torch.device, optional): Device to use for inference
        """
        self.model = model
        self.item_features = item_features
        self.user_features = user_features
        
        # Set device
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = device
        
        self.model.to(self.device)
        self.model.eval()
        
        # Prepare item features tensor
        if 'item_id' in self.item_features.columns:
            self.item_ids = self.item_features['item_id'].values
            self.item_features = self.item_features.drop('item_id', axis=1)
        else:
            self.item_ids = self.item_features.index.values
        
        # Convert item features to tensor format expected by model
        self.item_features_tensor = {}
        for col in self.item_features.columns:
            self.item_features_tensor[col] = torch.tensor(
                self.item_features[col].values,
                dtype=torch.float32,
                device=self.device
            )
        
        logger.info(f"Recommendation generator initialized with {len(self.item_ids)} items")
    
    def get_recommendations_for_user(self, user_id, user_features=None, top_k=10, exclude_items=None):
        """
        Get recommendations for a specific user
        
        Args:
            user_id: User ID
            user_features (numpy.ndarray, optional): User features. If None, look up in user_features DataFrame
            top_k (int): Number of recommendations to return
            exclude_items (list, optional): Items to exclude from recommendations
            
        Returns:
            pandas.DataFrame: DataFrame with recommendations
        """
        # Get user features
        if user_features is None and self.user_features is not None:
            if 'user_id' in self.user_features.columns:
                user_features = self.user_features[self.user_features['user_id'] == user_id].drop('user_id', axis=1)
            else:
                user_features = self.user_features.loc[[user_id]]
        
        if user_features is None or len(user_features) == 0:
            logger.warning(f"User {user_id} not found in user features")
            return pd.DataFrame()
        
        # Convert user features to tensor format expected by model
        user_features_tensor = {}
        for col in user_features.columns:
            user_features_tensor[col] = torch.tensor(
                user_features[col].values,
                dtype=torch.float32,
                device=self.device
            )
        
        # Batch user features to match item features
        for col in user_features_tensor:
            user_features_tensor[col] = user_features_tensor[col].repeat(len(self.item_ids))
        
        # Get predictions
        with torch.no_grad():
            scores = self.model(user_features_tensor, self.item_features_tensor).cpu().numpy()
        
        # Filter out excluded items
        if exclude_items is not None:
            excluded_indices = [i for i, item_id in enumerate(self.item_ids) if item_id in exclude_items]
            scores[excluded_indices] = -np.inf
        
        # Get top-k items
        top_indices = np.argsort(-scores)[:top_k]
        top_items = [self.item_ids[i] for i in top_indices]
        top_scores = scores[top_indices]
        
        # Create recommendations dataframe
        recs = pd.DataFrame({
            'user_id': [user_id] * len(top_items),
            'item_id': top_items,
            'score': top_scores,
            'rank': range(1, len(top_items) + 1)
        })
        
        return recs
    
    def get_recommendations_batch(self, user_ids, user_features_df=None, top_k=10, exclude_history=None):
        """
        Get recommendations for multiple users
        
        Args:
            user_ids (list): List of user IDs
            user_features_df (pandas.DataFrame, optional): User features. If None, use self.user_features
            top_k (int): Number of recommendations to return
            exclude_history (pandas.DataFrame, optional): DataFrame with user-item interactions to exclude
            
        Returns:
            pandas.DataFrame: DataFrame with recommendations
        """
        # Use provided user features or fallback to instance variable
        if user_features_df is None:
            user_features_df = self.user_features
        
        if user_features_df is None:
            logger.error("No user features provided")
            return pd.DataFrame()
        
        # Create dictionary of excluded items for each user
        excluded_items = defaultdict(set)
        if exclude_history is not None:
            for _, row in exclude_history.iterrows():
                excluded_items[row['user_id']].add(row['item_id'])
        
        all_recommendations = []
        
        # Generate recommendations for each user
        for user_id in tqdm(user_ids, desc="Generating recommendations"):
            # Get user features
            if 'user_id' in user_features_df.columns:
                user_features = user_features_df[user_features_df['user_id'] == user_id].drop('user_id', axis=1)
            else:
                try:
                    user_features = user_features_df.loc[[user_id]]
                except KeyError:
                    logger.warning(f"User {user_id} not found in user features")
                    continue
            
            # Get excluded items for this user
            exclude_items = excluded_items.get(user_id, None)
            
            # Get recommendations
            user_recs = self.get_recommendations_for_user(
                user_id, 
                user_features, 
                top_k=top_k, 
                exclude_items=exclude_items
            )
            
            all_recommendations.append(user_recs)
        
        # Combine all recommendations
        if all_recommendations:
            recommendations_df = pd.concat(all_recommendations, ignore_index=True)
            return recommendations_df
        else:
            return pd.DataFrame()
    
    def generate_all_recommendations(self, top_k=10, exclude_history=None, output_file=None):
        """
        Generate recommendations for all users in user_features
        
        Args:
            top_k (int): Number of recommendations to return
            exclude_history (pandas.DataFrame, optional): DataFrame with user-item interactions to exclude
            output_file (str, optional): Path to save recommendations
            
        Returns:
            pandas.DataFrame: DataFrame with recommendations
        """
        if self.user_features is None:
            logger.error("No user features provided")
            return pd.DataFrame()
        
        # Get list of all user IDs
        if 'user_id' in self.user_features.columns:
            user_ids = self.user_features['user_id'].unique()
        else:
            user_ids = self.user_features.index.values
        
        logger.info(f"Generating recommendations for {len(user_ids)} users")
        
        # Generate recommendations
        recommendations_df = self.get_recommendations_batch(
            user_ids, 
            top_k=top_k, 
            exclude_history=exclude_history
        )
        
        # Save to file if specified
        if output_file and not recommendations_df.empty:
            os.makedirs(os.path.dirname(output_file), exist_ok=True)
            recommendations_df.to_csv(output_file, index=False)
            logger.info(f"Saved recommendations to {output_file}")
        
        return recommendations_df
    
    @classmethod
    def from_checkpoint(cls, checkpoint_path, item_features, user_features=None, device=None):
        """
        Create a recommendation generator from a model checkpoint
        
        Args:
            checkpoint_path (str): Path to model checkpoint
            item_features (pandas.DataFrame): Item features
            user_features (pandas.DataFrame, optional): User features
            device (torch.device, optional): Device to use for inference
            
        Returns:
            RecommendationGenerator: Recommendation generator instance
        """
        # Set device
        if device is None:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Load model
        model = TwoTowerHybridModel.load(checkpoint_path, device)
        
        return cls(model, item_features, user_features, device)
    
    def get_user_embeddings(self, user_ids=None):
        """
        Get embeddings for users
        
        Args:
            user_ids (list, optional): List of user IDs. If None, get embeddings for all users.
            
        Returns:
            dict: Dictionary mapping user IDs to embeddings
        """
        if self.user_features is None:
            logger.error("No user features provided")
            return {}
        
        # Get user IDs
        if user_ids is None:
            if 'user_id' in self.user_features.columns:
                user_ids = self.user_features['user_id'].unique()
            else:
                user_ids = self.user_features.index.values
        
        # Get user features
        user_embeddings = {}
        
        for user_id in user_ids:
            # Get user features
            if 'user_id' in self.user_features.columns:
                user_features = self.user_features[self.user_features['user_id'] == user_id].drop('user_id', axis=1)
            else:
                try:
                    user_features = self.user_features.loc[[user_id]]
                except KeyError:
                    logger.warning(f"User {user_id} not found in user features")
                    continue
            
            # Convert to tensor
            user_features_tensor = {}
            for col in user_features.columns:
                user_features_tensor[col] = torch.tensor(
                    user_features[col].values,
                    dtype=torch.float32,
                    device=self.device
                )
            
            # Get embeddings
            with torch.no_grad():
                embedding = self.model.user_embedding(user_features_tensor).cpu().numpy()
            
            user_embeddings[user_id] = embedding[0]
        
        return user_embeddings
    
    def get_item_embeddings(self, item_ids=None):
        """
        Get embeddings for items
        
        Args:
            item_ids (list, optional): List of item IDs. If None, get embeddings for all items.
            
        Returns:
            dict: Dictionary mapping item IDs to embeddings
        """
        # Get item IDs
        if item_ids is None:
            item_ids = self.item_ids
        
        # Get item features
        item_embeddings = {}
        
        # Get embeddings for all items at once
        with torch.no_grad():
            all_embeddings = self.model.item_embedding(self.item_features_tensor).cpu().numpy()
        
        # Create dictionary
        for i, item_id in enumerate(self.item_ids):
            if item_ids is None or item_id in item_ids:
                item_embeddings[item_id] = all_embeddings[i]
        
        return item_embeddings
    
    def find_similar_items(self, item_id, top_k=10):
        """
        Find items similar to a given item based on embedding similarity
        
        Args:
            item_id: Item ID
            top_k (int): Number of similar items to return
            
        Returns:
            pandas.DataFrame: DataFrame with similar items
        """
        # Get all item embeddings
        item_embeddings = self.get_item_embeddings()
        
        if item_id not in item_embeddings:
            logger.warning(f"Item {item_id} not found in item embeddings")
            return pd.DataFrame()
        
        # Get embedding for target item
        target_embedding = item_embeddings[item_id]
        
        # Calculate cosine similarity
        similarities = {}
        for other_id, other_embedding in item_embeddings.items():
            if other_id == item_id:
                continue
            
            # Cosine similarity
            similarity = np.dot(target_embedding, other_embedding) / (
                np.linalg.norm(target_embedding) * np.linalg.norm(other_embedding)
            )
            
            similarities[other_id] = similarity
        
        # Get top-k similar items
        top_items = sorted(similarities.items(), key=lambda x: x[1], reverse=True)[:top_k]
        
        # Create DataFrame
        similar_items = pd.DataFrame({
            'item_id': [item[0] for item in top_items],
            'similarity': [item[1] for item in top_items],
            'rank': range(1, len(top_items) + 1)
        })
        
        return similar_items 