#!/usr/bin/env python
"""
Script to generate music recommendations for users.

This script loads a trained recommender model and generates personalized
track recommendations for users based on their preferences.
"""

import os
import sys
import argparse
import logging
import yaml
import torch
import json
import pandas as pd
import _pickle
import numpy as np
from pathlib import Path

# Add project root to path to allow imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from models.recommender import HybridRecommender
from training.trainer import ModelTrainer
from utils.logger import setup_logger

def load_model_safely(model_path, device):
    """
    Load model with proper handling of PyTorch weights_only security.
    
    Args:
        model_path: Path to the model checkpoint
        device: Device to load the model on
        
    Returns:
        Loaded checkpoint dictionary
    """
    try:
        # First try loading with weights_only=True (most secure)
        try:
            # Add numpy scalar to safe globals to handle common NumPy types
            import torch.serialization
            from numpy.core.multiarray import scalar
            torch.serialization.add_safe_globals([scalar])
            
            # Also add common dtype classes to handle numpy dtype issues
            import numpy as np
            from numpy import dtype
            torch.serialization.add_safe_globals([type(np.dtype(np.float32))])
            torch.serialization.add_safe_globals([dtype])
            
            # Try for numpy >= 1.25 (different class structure)
            try:
                from numpy.dtypes import Float32DType
                torch.serialization.add_safe_globals([Float32DType])
            except ImportError:
                pass  # Older numpy version, ignore
            
            # Load with weights_only=True
            return torch.load(model_path, map_location=device, weights_only=True)
        except (ImportError, RuntimeError, _pickle.UnpicklingError, AttributeError) as e:
            # If adding safe globals didn't work or other issues occur
            logger.warning(f"Secure loading with weights_only=True failed: {str(e)}")
            logger.warning("Falling back to standard loading. Make sure you trust this checkpoint source.")
            return torch.load(model_path, map_location=device, weights_only=False)
    except Exception as e:
        # If all loading attempts failed
        logger.error(f"Failed to load model from {model_path}: {str(e)}")
        raise

def prepare_features(user_row, track_data, user_features, track_features, categorical_mappings):
    """
    Prepare user and track features for model input.
    
    Args:
        user_row (pd.Series): User data as pandas Series
        track_data (pd.DataFrame): Track data
        user_features (list): List of user features to use
        track_features (list): List of track features to use
        categorical_mappings (dict): Mappings for categorical features
        
    Returns:
        tuple: Prepared user and track data for model input
    """
    # Prepare user features
    user_tensors = {}
    for feature in user_features:
        if feature in user_row:
            # Check if this is a categorical feature that needs to be mapped
            if feature in categorical_mappings:
                mapping = categorical_mappings[feature]
                # Map the value or use default (0) if not found
                value = mapping.get(user_row[feature], 0)
                user_tensors[feature] = torch.tensor([value], dtype=torch.long)
            else:
                # Numerical feature
                user_tensors[feature] = torch.tensor([user_row[feature]], dtype=torch.float)
    
    # Prepare track features for all tracks
    track_tensors = {}
    for feature in track_features:
        if feature in track_data.columns:
            # Check if this is a categorical feature
            if feature in categorical_mappings:
                mapping = categorical_mappings[feature]
                # Map values or use default (0) if not found
                values = track_data[feature].map(lambda x: mapping.get(x, 0))
                track_tensors[feature] = torch.tensor(values.values, dtype=torch.long)
            else:
                # Numerical feature
                track_tensors[feature] = torch.tensor(track_data[feature].values, dtype=torch.float)
    
    return user_tensors, track_tensors

def generate_recommendations(model, user_data, track_data, user_features, track_features, top_k=10):
    """
    Generate track recommendations for a user.
    
    Args:
        model (HybridRecommender): Trained recommender model
        user_data (dict): User features
        track_data (dict): Track features
        user_features (list): List of user features used
        track_features (list): List of track features used
        top_k (int): Number of recommendations to generate
        
    Returns:
        list: Recommended track indices and scores
    """
    # Set the model to evaluation mode and disable batch norm updates
    model.eval()
    
    # Fix for batch normalization with single samples
    for module in model.modules():
        if isinstance(module, torch.nn.BatchNorm1d):
            module.eval()
            # Make sure to use running statistics even with a batch size of 1
            module.track_running_stats = True
    
    # Move user data to the same device as the model
    device = next(model.parameters()).device
    user_data = {k: v.to(device) for k, v in user_data.items()}
    track_data = {k: v.to(device) for k, v in track_data.items()}
    
    # Get predictions
    with torch.no_grad():
        try:
            outputs = model(user_data, track_data)
        except ValueError as e:
            if "Expected more than 1 value per channel when training" in str(e):
                # If we still get the batch norm error, try with batch size of 2 by duplicating the input
                logger.warning("Handling batch normalization for single sample by duplicating input")
                # Duplicate user data for each feature
                duplicated_user_data = {
                    k: torch.cat([v, v], dim=0) if len(v.shape) > 1 else v.repeat(2)
                    for k, v in user_data.items()
                }
                # Forward pass with duplicated user data
                duplicated_outputs = model(duplicated_user_data, track_data)
                # Take only the first half (original user)
                outputs = duplicated_outputs[:len(duplicated_outputs)//2]
            else:
                # Re-raise if it's a different error
                raise
        
        # Apply sigmoid to get probabilities
        predictions = torch.sigmoid(outputs).squeeze().cpu().numpy()
    
    # Get top-k recommendations
    if len(predictions.shape) == 0:  # single prediction
        top_indices = [0]
        top_scores = [predictions.item()]
    else:
        top_indices = np.argsort(predictions)[::-1][:top_k]
        top_scores = predictions[top_indices]
    
    return list(zip(top_indices, top_scores))

def main():
    """Main function to generate recommendations."""
    parser = argparse.ArgumentParser(description="Generate music recommendations for users")
    parser.add_argument("--model", type=str, required=True,
                        help="Path to the trained model checkpoint")
    parser.add_argument("--config", type=str, required=True,
                        help="Path to configuration YAML file")
    parser.add_argument("--user-data", type=str, required=True,
                        help="Path to user data CSV file")
    parser.add_argument("--track-data", type=str, required=True,
                        help="Path to track data CSV file")
    parser.add_argument("--user-id", type=str,
                        help="ID of the user to generate recommendations for. If not provided, generates for all users.")
    parser.add_argument("--top-k", type=int, default=10,
                        help="Number of recommendations to generate for each user")
    parser.add_argument("--output", type=str, default="recommendations.csv",
                        help="Path to output CSV file")
    parser.add_argument("--log-level", type=str, choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
                        default="INFO", help="Set the logging level")
    parser.add_argument("--device", type=str, default=None,
                        help="Device to use for inference (cuda or cpu)")
    
    args = parser.parse_args()
    
    # Create logs directory
    log_dir = "logs"
    os.makedirs(log_dir, exist_ok=True)
    
    # Set up logging
    global logger
    logger = setup_logger(
        name="generate_recommendations",
        log_file=os.path.join(log_dir, "generate_recommendations.log"),
        level=getattr(logging, args.log_level)
    )
    
    try:
        logger.info("Starting recommendation generation")
        
        # Set device
        if args.device:
            device = torch.device(args.device)
        else:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        logger.info(f"Using device: {device}")
        
        # Load model
        logger.info(f"Loading model from {args.model}")
        checkpoint = load_model_safely(args.model, device)
        
        # Create model using checkpoint configuration
        config_path = checkpoint.get('config')
        categorical_mappings = checkpoint.get('categorical_mappings', {})
        
        # Create a temporary config file if needed
        if isinstance(config_path, dict):
            import tempfile
            config_file = tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.yaml')
            yaml.dump(config_path, config_file)
            config_path = config_file.name
            config_file.close()
        
        # Create model
        model = HybridRecommender(config_path, categorical_mappings)
        
        try:
            # First try with strict loading
            model.load_state_dict(checkpoint['model_state_dict'])
        except RuntimeError as e:
            logger.warning(f"Strict loading failed: {str(e)}")
            logger.warning("Attempting to load with strict=False to handle model architecture differences")
            # Try again with non-strict loading to handle architecture changes
            model.load_state_dict(checkpoint['model_state_dict'], strict=False)
        
        model.to(device)
        model.eval()
        
        # Clean up temporary file if created
        if isinstance(config_path, str) and hasattr(config_file, 'name') and config_path == config_file.name:
            os.unlink(config_path)
        
        # Load configuration
        with open(args.config, 'r') as f:
            config = yaml.safe_load(f)
        
        # Extract feature lists from model config
        model_config_path = config['model']['config_path']
        with open(model_config_path, 'r') as f:
            model_config = yaml.safe_load(f)
        
        # Get user and track features based on config structure
        user_features = []
        track_features = []
        
        if 'feature_sets' in model_config:
            # Extract from feature_sets structure
            user_features = model_config['feature_sets']['user_features']
            track_features = model_config['feature_sets']['track_features']
        elif 'features' in model_config:
            # Extract from features structure
            user_features = model_config['features']['user_features']['numerical'] + model_config['features']['user_features']['categorical']
            track_features = model_config['features']['track_features']['numerical'] + model_config['features']['track_features']['categorical']
        
        logger.info(f"User features: {user_features}")
        logger.info(f"Track features: {track_features}")
        
        # Load user and track data
        logger.info(f"Loading user data from {args.user_data}")
        user_df = pd.read_csv(args.user_data)
        
        logger.info(f"Loading track data from {args.track_data}")
        track_df = pd.read_csv(args.track_data)
        
        # Create recommendations dataframe
        recommendations_data = []
        
        # Filter to specific user if requested
        if args.user_id:
            user_ids = [args.user_id]
            logger.info(f"Generating recommendations for user {args.user_id}")
        else:
            user_ids = user_df['user_id'].unique()
            logger.info(f"Generating recommendations for {len(user_ids)} users")
        
        # Generate recommendations for each user
        for user_id in user_ids:
            logger.info(f"Processing user {user_id}")
            
            # Get user data
            user_row = user_df[user_df['user_id'] == user_id].iloc[0]
            
            # Prepare features
            user_tensors, track_tensors = prepare_features(
                user_row, track_df, user_features, track_features, categorical_mappings
            )
            
            # Generate recommendations
            recommendations = generate_recommendations(
                model, user_tensors, track_tensors, 
                user_features, track_features, args.top_k
            )
            
            # Add to recommendations dataframe
            for i, (track_idx, score) in enumerate(recommendations):
                track_id = track_df.iloc[track_idx]['track_id']
                track_name = track_df.iloc[track_idx].get('track_name', f"Track {track_id}")
                artist = track_df.iloc[track_idx].get('artist', "Unknown")
                
                recommendations_data.append({
                    'user_id': user_id,
                    'rank': i + 1,
                    'track_id': track_id,
                    'track_name': track_name,
                    'artist': artist,
                    'score': score
                })
        
        # Create recommendations dataframe
        recommendations_df = pd.DataFrame(recommendations_data)
        
        # Save recommendations
        recommendations_df.to_csv(args.output, index=False)
        logger.info(f"Recommendations saved to {args.output}")
        
        return 0
    
    except Exception as e:
        logger.error(f"Error generating recommendations: {str(e)}", exc_info=True)
        return 1

if __name__ == "__main__":
    sys.exit(main()) 