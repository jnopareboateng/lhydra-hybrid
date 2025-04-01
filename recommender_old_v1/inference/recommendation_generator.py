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
        
        # Check if model has feature manifest
        self.has_feature_manifest = hasattr(self.model, 'feature_manifest') and self.model.feature_manifest is not None
        if self.has_feature_manifest:
            logger.info("Model has feature manifest. Feature alignment will be used during inference.")
        else:
            logger.warning("Model does not have feature manifest. Feature compatibility issues may occur.")
        
        # Prepare item features tensor
        if 'item_id' in self.item_features.columns:
            self.item_ids = self.item_features['item_id'].values
            self.item_id_col = 'item_id'
        elif 'track_id' in self.item_features.columns:
            self.item_ids = self.item_features['track_id'].values
            self.item_id_col = 'track_id'
        else:
            self.item_ids = self.item_features.index.values
            self.item_id_col = None
        
        # Convert item features to tensor format expected by model
        self.item_features_tensor = {}
        
        # If we have an ID column, add it to the tensor dict and create a copy without it
        if self.item_id_col is not None:
            self.item_features_tensor[self.item_id_col] = torch.tensor(
                self.item_features[self.item_id_col].values,
                dtype=torch.float32,
                device=self.device
            )
            feature_df = self.item_features.drop(self.item_id_col, axis=1)
        else:
            feature_df = self.item_features
        
        # Add all feature columns to the tensor dict
        for col in feature_df.columns:
            self.item_features_tensor[col] = torch.tensor(
                feature_df[col].values,
                dtype=torch.float32,
                device=self.device
            )
        
        logger.info(f"Recommendation generator initialized with {len(self.item_ids)} items")
        logger.info(f"Item features shape: {len(self.item_features_tensor)} features")
    
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
                user_features = self.user_features[self.user_features['user_id'] == user_id]
                if len(user_features) == 0:
                    logger.warning(f"User {user_id} not found in user features")
                    return pd.DataFrame()
            else:
                try:
                    user_features = self.user_features.loc[[user_id]]
                except KeyError:
                    logger.warning(f"User {user_id} not found in user features")
                    return pd.DataFrame()
        
        if user_features is None or len(user_features) == 0:
            logger.warning(f"User {user_id} not found and no user features provided")
            return pd.DataFrame()
        
        # Convert user features to tensor format expected by model
        user_features_tensor = {}
        
        # Add user_id if present
        if 'user_id' in user_features.columns:
            user_features_tensor['user_id'] = torch.tensor(
                user_features['user_id'].values,
                dtype=torch.float32,
                device=self.device
            )
            feature_df = user_features.drop('user_id', axis=1)
        else:
            feature_df = user_features
        
        # Add all feature columns to the tensor dict
        for col in feature_df.columns:
            user_features_tensor[col] = torch.tensor(
                feature_df[col].values,
                dtype=torch.float32,
                device=self.device
            )
        
        # Clone user features to match item count (for batch prediction)
        user_features_batch = {}
        for key, tensor in user_features_tensor.items():
            # Repeat the tensor to match the number of items
            user_features_batch[key] = tensor.repeat(len(self.item_ids), 1)[0] if tensor.dim() > 1 else tensor.repeat(len(self.item_ids))
        
        # If model has feature manifest, let it handle alignment internally
        # Otherwise, use the existing feature tensors directly
        with torch.no_grad():
            try:
                # Get model predictions
                scores = self.model(user_features_batch, self.item_features_tensor).cpu().numpy()
                
                # If scores is 2D, take the first column (engagement probability)
                if len(scores.shape) > 1 and scores.shape[1] > 1:
                    scores = scores[:, 0]  # Take first column
                
            except Exception as e:
                logger.error(f"Error during prediction: {str(e)}")
                # Try a fallback approach if the dictionaries cause issues
                if isinstance(user_features_batch, dict) and isinstance(self.item_features_tensor, dict):
                    logger.warning("Attempting to use feature concatenation as a fallback")
                    try:
                        # Try to manually concatenate features
                        user_tensor_list = [tensor for key, tensor in user_features_batch.items() 
                                          if key != 'user_id']
                        user_tensor = torch.cat([t.reshape(len(self.item_ids), -1) for t in user_tensor_list], dim=1)
                        
                        item_tensor_list = [tensor for key, tensor in self.item_features_tensor.items() 
                                          if key != self.item_id_col]
                        item_tensor = torch.cat([t.reshape(len(self.item_ids), -1) for t in item_tensor_list], dim=1)
                        
                        scores = self.model(user_tensor, item_tensor).cpu().numpy()
                        
                        # If scores is 2D, take the first column (engagement probability)
                        if len(scores.shape) > 1 and scores.shape[1] > 1:
                            scores = scores[:, 0]  # Take first column
                    except Exception as inner_e:
                        logger.error(f"Fallback prediction also failed: {str(inner_e)}")
                        return pd.DataFrame()
                else:
                    return pd.DataFrame()
        
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
    def from_checkpoint(cls, checkpoint_path, item_features, user_features=None, device=None, manifest_path=None):
        """
        Create a recommendation generator from a model checkpoint
        
        Args:
            checkpoint_path (str): Path to model checkpoint
            item_features (pandas.DataFrame): Item features
            user_features (pandas.DataFrame, optional): User features
            device (torch.device, optional): Device to use for inference
            manifest_path (str, optional): Path to feature manifest file
            
        Returns:
            RecommendationGenerator: Recommendation generator instance
        """
        # Set device
        if device is None:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Load model
        logger.info(f"Loading model from {checkpoint_path}")
        
        # Check if manifest path was provided
        if manifest_path and os.path.exists(manifest_path):
            logger.info(f"Using feature manifest from {manifest_path}")
            model = TwoTowerHybridModel.load(checkpoint_path, device=device, manifest_path=manifest_path)
        else:
            # Try to look for manifest in the same directory as the model
            model_dir = os.path.dirname(checkpoint_path)
            possible_manifest = os.path.join(model_dir, "feature_manifest.yaml")
            if os.path.exists(possible_manifest):
                logger.info(f"Found feature manifest at {possible_manifest}")
                model = TwoTowerHybridModel.load(checkpoint_path, device=device, manifest_path=possible_manifest)
            else:
                logger.warning("No feature manifest found. Feature alignment will not be available.")
                model = TwoTowerHybridModel.load(checkpoint_path, device=device)
        
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
        
        for user_id in tqdm(user_ids, desc="Extracting user embeddings"):
            # Get user features
            if 'user_id' in self.user_features.columns:
                user_features = self.user_features[self.user_features['user_id'] == user_id]
                if len(user_features) == 0:
                    logger.warning(f"User {user_id} not found in user features")
                    continue
            else:
                try:
                    user_features = self.user_features.loc[[user_id]]
                except KeyError:
                    logger.warning(f"User {user_id} not found in user features")
                    continue
            
            # Convert to tensor
            user_features_tensor = {}
            
            # Add user_id if present
            if 'user_id' in user_features.columns:
                user_features_tensor['user_id'] = torch.tensor(
                    user_features['user_id'].values,
                    dtype=torch.float32,
                    device=self.device
                )
                feature_df = user_features.drop('user_id', axis=1)
            else:
                feature_df = user_features
            
            # Add all feature columns to the tensor dict
            for col in feature_df.columns:
                user_features_tensor[col] = torch.tensor(
                    feature_df[col].values,
                    dtype=torch.float32,
                    device=self.device
                )
            
            # Get embeddings
            try:
                with torch.no_grad():
                    if hasattr(self.model, 'user_embedding') and callable(getattr(self.model, 'user_embedding')):
                        # Use model's dedicated embedding method
                        embedding = self.model.user_embedding(user_features_tensor).cpu().numpy()
                    else:
                        # Fallback to internal implementation
                        logger.warning("Model doesn't have a user_embedding method, using internal embedding extraction")
                        if hasattr(self.model, 'user_tower'):
                            embedding = self.model.user_tower(user_features_tensor).cpu().numpy()
                        else:
                            logger.error("Cannot extract user embeddings - incompatible architecture")
                            continue
                
                user_embeddings[user_id] = embedding[0]
            except Exception as e:
                logger.error(f"Error getting embedding for user {user_id}: {str(e)}")
                
                # Try fallback with concatenated features
                try:
                    logger.warning(f"Attempting fallback approach for user {user_id}")
                    user_tensor_list = [tensor for key, tensor in user_features_tensor.items() 
                                      if key != 'user_id']
                    
                    # Ensure all tensors are 2D for concatenation
                    user_tensor_list = [t.reshape(1, -1) if t.dim() == 1 else t for t in user_tensor_list]
                    
                    user_tensor = torch.cat(user_tensor_list, dim=1)
                    
                    if hasattr(self.model, 'user_tower'):
                        embedding = self.model.user_tower(user_tensor).cpu().numpy()
                        user_embeddings[user_id] = embedding[0]
                    else:
                        logger.error("Cannot extract user embeddings even with fallback - incompatible architecture")
                except Exception as inner_e:
                    logger.error(f"Fallback embedding extraction also failed for user {user_id}: {str(inner_e)}")
        
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
        
        # Get embeddings for all items at once
        with torch.no_grad():
            try:
                if hasattr(self.model, 'item_embedding') and callable(getattr(self.model, 'item_embedding')):
                    # Use the model's dedicated embedding method if available
                    all_embeddings = self.model.item_embedding(self.item_features_tensor).cpu().numpy()
                else:
                    # Otherwise, use our internal implementation
                    # This is a fallback in case the model architecture doesn't expose embedding methods
                    logger.warning("Model doesn't have an item_embedding method, using internal embedding extraction")
                    # Get appropriate item towers based on model type
                    if hasattr(self.model, 'item_tower'):
                        # For TwoTowerHybridModel or similar models
                        embeddings = self.model.item_tower(self.item_features_tensor)
                    else:
                        # Generic fallback
                        logger.error("Cannot extract item embeddings from model - incompatible architecture")
                        return {}
                    
                    all_embeddings = embeddings.cpu().numpy()
            except Exception as e:
                logger.error(f"Error getting item embeddings: {str(e)}")
                
                # Try fallback with concatenated features
                try:
                    logger.warning("Attempting fallback approach for item embeddings")
                    item_tensor_list = [tensor for key, tensor in self.item_features_tensor.items() 
                                      if key != self.item_id_col]
                    item_tensor = torch.cat([t.reshape(len(self.item_ids), -1) for t in item_tensor_list], dim=1)
                    
                    if hasattr(self.model, 'item_tower'):
                        embeddings = self.model.item_tower(item_tensor)
                        all_embeddings = embeddings.cpu().numpy()
                    else:
                        logger.error("Cannot extract item embeddings even with fallback - incompatible architecture")
                        return {}
                except Exception as inner_e:
                    logger.error(f"Fallback embedding extraction also failed: {str(inner_e)}")
                    return {}
        
        # Create dictionary
        item_embeddings = {}
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
        
        if not item_embeddings:
            logger.error("Failed to get item embeddings")
            return pd.DataFrame()
        
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
            
            # Skip if either embedding is all zeros
            if np.allclose(target_embedding, 0) or np.allclose(other_embedding, 0):
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