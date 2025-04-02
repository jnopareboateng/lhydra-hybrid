#!/usr/bin/env python
"""
Script for making interactive predictions with the music recommender.

This script allows predicting the likelihood of a user enjoying
specific tracks, which is useful for real-time recommendation scenarios.
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

def prepare_tensors(user_row, track_rows, user_features, track_features, categorical_mappings):
    """
    Prepare tensors for model input from user and track data.
    
    Args:
        user_row (pd.Series): User data
        track_rows (pd.DataFrame): Track data for prediction
        user_features (list): List of user features to use
        track_features (list): List of track features to use
        categorical_mappings (dict): Mappings for categorical features
        
    Returns:
        dict, dict: User and track tensors for model input
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
    
    # Prepare track features
    track_tensors = {}
    for feature in track_features:
        if feature in track_rows.columns:
            # Check if this is a categorical feature
            if feature in categorical_mappings:
                mapping = categorical_mappings[feature]
                # Map values or use default (0) if not found
                values = track_rows[feature].map(lambda x: mapping.get(x, 0))
                track_tensors[feature] = torch.tensor(values.values, dtype=torch.long)
            else:
                # Numerical feature
                track_tensors[feature] = torch.tensor(track_rows[feature].values, dtype=torch.float)
    
    return user_tensors, track_tensors

def predict_user_track_affinity(model, user_tensors, track_tensors):
    """
    Predict the affinity between a user and tracks.
    
    Args:
        model (HybridRecommender): Trained model
        user_tensors (dict): User feature tensors
        track_tensors (dict): Track feature tensors
        
    Returns:
        numpy.ndarray: Predicted affinities
    """
    # Set model to evaluation mode
    model.eval()
    
    # Fix for batch normalization with single samples
    for module in model.modules():
        if isinstance(module, torch.nn.BatchNorm1d):
            module.eval()
            # Make sure to use running statistics even with a batch size of 1
            module.track_running_stats = True
    
    # Move data to the same device as the model
    device = next(model.parameters()).device
    user_tensors = {k: v.to(device) for k, v in user_tensors.items()}
    track_tensors = {k: v.to(device) for k, v in track_tensors.items()}
    
    # Make predictions
    with torch.no_grad():
        try:
            outputs = model(user_tensors, track_tensors)
        except ValueError as e:
            if "Expected more than 1 value per channel when training" in str(e):
                # If we still get the batch norm error, try with batch size of 2 by duplicating the input
                logger.warning("Handling batch normalization for single sample by duplicating input")
                # Duplicate user data for each feature
                duplicated_user_tensors = {
                    k: torch.cat([v, v], dim=0) if len(v.shape) > 1 else v.repeat(2)
                    for k, v in user_tensors.items()
                }
                # Forward pass with duplicated user data
                duplicated_outputs = model(duplicated_user_tensors, track_tensors)
                # Take only the first half (original user)
                outputs = duplicated_outputs[:len(duplicated_outputs)//2]
            else:
                # Re-raise if it's a different error
                raise
                
        predictions = torch.sigmoid(outputs).cpu().numpy()
    
    return predictions

def main():
    """Main function for interactive prediction."""
    parser = argparse.ArgumentParser(description="Predict user affinities for specific tracks")
    parser.add_argument("--model", type=str, required=True,
                        help="Path to the trained model checkpoint")
    parser.add_argument("--user-data", type=str, required=True,
                        help="Path to user data CSV file")
    parser.add_argument("--track-data", type=str, required=True,
                        help="Path to track data CSV file")
    parser.add_argument("--user-id", type=str, required=True,
                        help="ID of the user to make predictions for")
    parser.add_argument("--track-ids", type=str, nargs='+',
                        help="List of track IDs to predict for (if not provided, all tracks will be used)")
    parser.add_argument("--output", type=str, default="track_predictions.json",
                        help="Path to output JSON file")
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
        name="predict_tracks",
        log_file=os.path.join(log_dir, "predict_tracks.log"),
        level=getattr(logging, args.log_level)
    )
    
    try:
        logger.info("Starting track prediction")
        
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
        
        # Extract feature lists from model configuration
        if isinstance(checkpoint.get('config'), dict):
            model_config = checkpoint.get('config')
        else:
            # Load from file
            with open(config_path, 'r') as f:
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
            user_numerical = model_config['features']['user_features'].get('numerical', [])
            user_categorical = model_config['features']['user_features'].get('categorical', [])
            track_numerical = model_config['features']['track_features'].get('numerical', [])
            track_categorical = model_config['features']['track_features'].get('categorical', [])
            
            user_features = user_numerical + user_categorical
            track_features = track_numerical + track_categorical
        
        logger.info(f"User features: {user_features}")
        logger.info(f"Track features: {track_features}")
        
        # Load user and track data
        logger.info(f"Loading user data from {args.user_data}")
        user_df = pd.read_csv(args.user_data)
        
        logger.info(f"Loading track data from {args.track_data}")
        track_df = pd.read_csv(args.track_data)
        
        # Get user data
        user_rows = user_df[user_df['user_id'] == args.user_id]
        if user_rows.empty:
            logger.error(f"User {args.user_id} not found in user data")
            return 1
        
        user_row = user_rows.iloc[0]
        
        # Filter track data if track IDs are provided
        if args.track_ids:
            track_rows = track_df[track_df['track_id'].isin(args.track_ids)]
            if track_rows.empty:
                logger.error(f"No tracks found with the specified IDs")
                return 1
        else:
            track_rows = track_df
        
        logger.info(f"Predicting for {len(track_rows)} tracks")
        
        # Prepare tensors
        user_tensors, track_tensors = prepare_tensors(
            user_row, track_rows, user_features, track_features, categorical_mappings
        )
        
        # Make predictions
        predictions = predict_user_track_affinity(model, user_tensors, track_tensors)
        
        # Create results
        results = []
        for i, (_, track_row) in enumerate(track_rows.iterrows()):
            score = predictions[i].item() if i < len(predictions) else 0.0
            
            track_data = {
                'track_id': track_row['track_id'],
                'score': score
            }
            
            # Add additional track metadata if available
            if 'track_name' in track_row:
                track_data['track_name'] = track_row['track_name']
            
            if 'artist' in track_row:
                track_data['artist'] = track_row['artist']
                
            if 'album' in track_row:
                track_data['album'] = track_row['album']
                
            if 'genre' in track_row:
                track_data['genre'] = track_row['genre']
                
            results.append(track_data)
        
        # Sort results by score in descending order
        results.sort(key=lambda x: x['score'], reverse=True)
        
        # Create final output
        output = {
            'user_id': args.user_id,
            'predictions': results
        }
        
        # Save results
        with open(args.output, 'w') as f:
            json.dump(output, f, indent=4)
        
        logger.info(f"Predictions saved to {args.output}")
        
        # Print top predictions to console
        print(f"\nTop predictions for user {args.user_id}:")
        for i, result in enumerate(results[:10]):
            track_name = result.get('track_name', result['track_id'])
            artist = result.get('artist', 'Unknown')
            print(f"{i+1}. {track_name} by {artist} - Score: {result['score']:.4f}")
        
        return 0
    
    except Exception as e:
        logger.error(f"Error making predictions: {str(e)}", exc_info=True)
        return 1

if __name__ == "__main__":
    sys.exit(main()) 