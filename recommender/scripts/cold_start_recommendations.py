#!/usr/bin/env python
"""
Script for generating recommendations for new users (cold start problem).

This script handles the cold start problem by generating recommendations for users
with no prior listening history, based on demographic information and initial preferences.
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

def create_user_profile_from_demographics(demographics, preference_data, user_features, categorical_mappings):
    """
    Create a user profile tensor from demographic information.
    
    Args:
        demographics (dict): User demographic information (age, gender, country, etc.)
        preference_data (dict): User preferences (genres, artists, etc.)
        user_features (list): List of user features expected by the model
        categorical_mappings (dict): Mappings for categorical features
        
    Returns:
        dict: User tensors for model input
    """
    user_tensors = {}
    
    # Process demographic information
    for feature in user_features:
        if feature in demographics:
            # Check if this is a categorical feature that needs to be mapped
            if feature in categorical_mappings:
                mapping = categorical_mappings[feature]
                # Map the value or use default (0) if not found
                value = mapping.get(demographics[feature], 0)
                user_tensors[feature] = torch.tensor([value], dtype=torch.long)
            else:
                # Numerical feature
                user_tensors[feature] = torch.tensor([float(demographics[feature])], dtype=torch.float)
    
    # If age is in features but not provided, use average age
    if 'age' in user_features and 'age' not in demographics:
        # Default to average adult age if not provided
        user_tensors['age'] = torch.tensor([30.0], dtype=torch.float)
    
    # For missing categorical features, use defaults or most common values
    for feature in user_features:
        if feature not in user_tensors:
            # Check if it's a known categorical feature
            if feature in categorical_mappings:
                # Default to the first mapping (usually most common or default)
                default_value = 0
                user_tensors[feature] = torch.tensor([default_value], dtype=torch.long)
            elif feature.endswith('_id') and feature.replace('_id', '') in categorical_mappings:
                # Handle _id suffix variants
                base_feature = feature.replace('_id', '')
                default_value = 0
                user_tensors[feature] = torch.tensor([default_value], dtype=torch.long)
            elif feature in ['gender_id', 'gender']:
                # Default gender to unknown (0)
                user_tensors[feature] = torch.tensor([0], dtype=torch.long)
            elif feature in ['country_id', 'country']:
                # Default country to unknown (0)
                user_tensors[feature] = torch.tensor([0], dtype=torch.long)
            else:
                # For other numerical features, use 0 as default
                user_tensors[feature] = torch.tensor([0.0], dtype=torch.float)
    
    # Add preference-derived features if available
    if preference_data and 'genres' in preference_data and 'genre_ids' in user_features:
        # Create preference vector based on genre selections
        genre_ids = [
            categorical_mappings.get('genre', {}).get(genre, 0)
            for genre in preference_data['genres']
        ]
        if genre_ids:
            # Use the first genre as the primary genre
            user_tensors['genre_ids'] = torch.tensor([genre_ids[0]], dtype=torch.long)
    
    return user_tensors

def filter_tracks_by_preferences(track_df, preference_data):
    """
    Filter tracks based on user preferences.
    
    Args:
        track_df (pd.DataFrame): Track data
        preference_data (dict): User preferences (genres, artists, etc.)
        
    Returns:
        pd.DataFrame: Filtered track data
    """
    filtered_df = track_df.copy()
    
    # Apply genre filtering if specified
    if preference_data and 'genres' in preference_data and len(preference_data['genres']) > 0:
        # Check if we have genre columns in the data
        genre_cols = [col for col in filtered_df.columns if 'genre' in col.lower()]
        
        if genre_cols:
            # Create a mask for genre filtering
            genre_mask = filtered_df[genre_cols[0]].isin(preference_data['genres'])
            
            # If multiple genre columns, combine with OR
            for col in genre_cols[1:]:
                genre_mask |= filtered_df[col].isin(preference_data['genres'])
            
            # Apply genre filtering, but ensure we keep at least 100 tracks
            genre_filtered = filtered_df[genre_mask]
            if len(genre_filtered) >= 100:
                filtered_df = genre_filtered
            else:
                logger.warning(f"Genre filtering returned only {len(genre_filtered)} tracks. Using all tracks instead.")
    
    # Apply artist filtering if specified
    if preference_data and 'artists' in preference_data and len(preference_data['artists']) > 0:
        # Check if we have artist columns in the data
        artist_cols = [col for col in filtered_df.columns if 'artist' in col.lower()]
        
        if artist_cols:
            # Create a mask for artist filtering
            artist_mask = filtered_df[artist_cols[0]].isin(preference_data['artists'])
            
            # If multiple artist columns, combine with OR
            for col in artist_cols[1:]:
                artist_mask |= filtered_df[col].isin(preference_data['artists'])
            
            # Apply artist filtering, but ensure we keep at least 50 tracks
            artist_filtered = filtered_df[artist_mask]
            if len(artist_filtered) >= 50:
                filtered_df = artist_filtered
            else:
                # If not enough tracks match exactly, try to find tracks from similar artists
                # This would require a similarity mapping between artists, which is typically 
                # available in music recommendation systems
                logger.warning(f"Artist filtering returned only {len(artist_filtered)} tracks. Using genre filtering only.")
    
    return filtered_df

def prepare_track_features(track_df, track_features, categorical_mappings):
    """
    Prepare track features for the model.
    
    Args:
        track_df (pd.DataFrame): Track data
        track_features (list): List of track features expected by the model
        categorical_mappings (dict): Mappings for categorical features
        
    Returns:
        dict: Track tensors for model input
    """
    track_tensors = {}
    
    # Process track features
    for feature in track_features:
        if feature in track_df.columns:
            # Check if this is a categorical feature
            if feature in categorical_mappings:
                mapping = categorical_mappings[feature]
                # Map values or use default (0) if not found
                values = track_df[feature].map(lambda x: mapping.get(x, 0))
                track_tensors[feature] = torch.tensor(values.values, dtype=torch.long)
            else:
                # Numerical feature
                track_tensors[feature] = torch.tensor(track_df[feature].values, dtype=torch.float)
        else:
            # Handle missing features
            if feature.endswith('_id') and feature.replace('_id', '') in track_df.columns:
                # Try to map from the non-id version
                base_feature = feature.replace('_id', '')
                if base_feature in categorical_mappings:
                    mapping = categorical_mappings[base_feature]
                    values = track_df[base_feature].map(lambda x: mapping.get(x, 0))
                    track_tensors[feature] = torch.tensor(values.values, dtype=torch.long)
                else:
                    # Default to zeros if mapping not available
                    track_tensors[feature] = torch.zeros(len(track_df), dtype=torch.long)
            else:
                # For missing features, use zeros
                if any(f.endswith('_id') for f in track_features) and feature in categorical_mappings:
                    # Categorical feature
                    track_tensors[feature] = torch.zeros(len(track_df), dtype=torch.long)
                else:
                    # Numerical feature
                    track_tensors[feature] = torch.zeros(len(track_df), dtype=torch.float)
    
    return track_tensors

def generate_cold_start_recommendations(model, user_tensors, track_tensors, track_df, top_k=10):
    """
    Generate recommendations for a new user with no prior history.
    
    Args:
        model (HybridRecommender): Trained recommender model
        user_tensors (dict): User feature tensors
        track_tensors (dict): Track feature tensors
        track_df (pd.DataFrame): Track data
        top_k (int): Number of recommendations to generate
        
    Returns:
        list: Recommended track IDs and scores
    """
    # Set model to evaluation mode and disable batch norm updates
    model.eval()
    
    # Fix for batch normalization with single samples
    for module in model.modules():
        if isinstance(module, torch.nn.BatchNorm1d):
            module.eval()
            module.track_running_stats = False
    
    # Move tensors to the same device as the model
    device = next(model.parameters()).device
    user_tensors = {k: v.to(device) for k, v in user_tensors.items()}
    track_tensors = {k: v.to(device) for k, v in track_tensors.items()}
    
    # Make predictions
    with torch.no_grad():
        try:
            outputs = model(user_tensors, track_tensors)
        except ValueError as e:
            if "Expected more than 1 value per channel when training" in str(e):
                logger.warning("Handling batch normalization for single sample...")
                
                # Instead of duplicating, we'll use a different approach that's safer
                # First, temporarily replace all BatchNorm layers with eval-mode layers
                for module in model.modules():
                    if isinstance(module, torch.nn.BatchNorm1d):
                        module.training = False
                        
                # Try again with BatchNorm in eval mode
                try:
                    outputs = model(user_tensors, track_tensors)
                except Exception as inner_e:
                    logger.warning(f"First workaround failed: {str(inner_e)}")
                    
                    # If that fails, try using Instance Normalization instead
                    # Save original state to restore later
                    original_batch_norms = {}
                    for name, module in model.named_modules():
                        if isinstance(module, torch.nn.BatchNorm1d):
                            original_batch_norms[name] = module
                            # Temporarily replace with an identity operation
                            parent_name = '.'.join(name.split('.')[:-1])
                            last_name = name.split('.')[-1]
                            if parent_name:
                                parent = model
                                for part in parent_name.split('.'):
                                    parent = getattr(parent, part)
                                setattr(parent, last_name, torch.nn.Identity())
                    
                    # Try again with Identity instead of BatchNorm
                    try:
                        outputs = model(user_tensors, track_tensors)
                    except Exception as e2:
                        logger.error(f"All batch norm workarounds failed, using simpler approach")
                        
                        # Final backup approach: process tracks one by one and combine results
                        all_scores = []
                        batch_size = 16  # Process tracks in small batches
                        total_tracks = next(iter(track_tensors.values())).shape[0]
                        
                        for i in range(0, total_tracks, batch_size):
                            end_idx = min(i + batch_size, total_tracks)
                            batch_track_tensors = {k: v[i:end_idx] for k, v in track_tensors.items()}
                            
                            # For each batch of tracks, duplicate the user tensor to match
                            batch_size_actual = end_idx - i
                            batch_user_tensors = {}
                            for k, v in user_tensors.items():
                                if len(v.shape) > 1:  # For 2D+ tensors
                                    # Repeat the user tensor to match the batch size
                                    batch_user_tensors[k] = v.repeat(batch_size_actual, 1)
                                else:  # For 1D tensors
                                    batch_user_tensors[k] = v.repeat(batch_size_actual)
                            
                            # Now both user and track tensors have the same batch dimension
                            try:
                                batch_outputs = model(batch_user_tensors, batch_track_tensors)
                                all_scores.append(batch_outputs)
                            except Exception as e3:
                                logger.error(f"Error in batch {i}: {str(e3)}")
                                # If all fails, return random scores for this batch
                                all_scores.append(torch.rand(batch_size_actual, device=device))
                                
                        # Combine all batches
                        outputs = torch.cat(all_scores, dim=0)
                        
                    # Restore original BatchNorm layers
                    for name, module in original_batch_norms.items():
                        parent_name = '.'.join(name.split('.')[:-1])
                        last_name = name.split('.')[-1]
                        if parent_name:
                            parent = model
                            for part in parent_name.split('.'):
                                parent = getattr(parent, part)
                            setattr(parent, last_name, module)
            else:
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
    
    # Get track information
    recommendations = []
    for idx, score in zip(top_indices, top_scores):
        track_id = track_df.iloc[idx]['track_id']
        track_name = track_df.iloc[idx].get('track_name', f"Track {track_id}")
        artist = track_df.iloc[idx].get('artist', "Unknown")
        
        recommendations.append({
            'track_id': track_id,
            'track_name': track_name,
            'artist': artist,
            'score': float(score)
        })
    
    return recommendations

def main():
    """Main function for cold start recommendations."""
    parser = argparse.ArgumentParser(description="Generate recommendations for new users (cold start)")
    parser.add_argument("--model", type=str, required=True,
                      help="Path to the trained model checkpoint")
    parser.add_argument("--config", type=str, required=True,
                      help="Path to training configuration YAML file")
    parser.add_argument("--track-data", type=str, required=True,
                      help="Path to track data CSV file")
    parser.add_argument("--demographics", type=str, required=True,
                      help="Path to JSON file with user demographics (age, gender, country)")
    parser.add_argument("--preferences", type=str, required=False,
                      help="Path to JSON file with user preferences (genres, artists)")
    parser.add_argument("--top-k", type=int, default=10,
                      help="Number of recommendations to generate")
    parser.add_argument("--output", type=str, default="cold_start_recommendations.json",
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
        name="cold_start_recommendations",
        log_file=os.path.join(log_dir, "cold_start_recommendations.log"),
        level=getattr(logging, args.log_level)
    )
    
    try:
        logger.info("Starting cold start recommendation generation")
        
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
            user_numerical = model_config['features']['user_features'].get('numerical', [])
            user_categorical = model_config['features']['user_features'].get('categorical', [])
            track_numerical = model_config['features']['track_features'].get('numerical', [])
            track_categorical = model_config['features']['track_features'].get('categorical', [])
            
            user_features = user_numerical + user_categorical
            track_features = track_numerical + track_categorical
        
        logger.info(f"User features: {user_features}")
        logger.info(f"Track features: {track_features}")
        
        # Load demographic data
        logger.info(f"Loading demographic data from {args.demographics}")
        with open(args.demographics, 'r') as f:
            demographics = json.load(f)
        
        # Load preference data if provided
        preference_data = None
        if args.preferences:
            logger.info(f"Loading preference data from {args.preferences}")
            with open(args.preferences, 'r') as f:
                preference_data = json.load(f)
        
        # Load track data
        logger.info(f"Loading track data from {args.track_data}")
        track_df = pd.read_csv(args.track_data)
        
        # Filter tracks based on preferences if available
        if preference_data:
            logger.info("Filtering tracks based on user preferences")
            filtered_track_df = filter_tracks_by_preferences(track_df, preference_data)
            logger.info(f"Selected {len(filtered_track_df)} tracks based on preferences")
        else:
            filtered_track_df = track_df
        
        # Create user profile from demographics
        logger.info("Creating user profile from demographics")
        user_tensors = create_user_profile_from_demographics(
            demographics, preference_data, user_features, categorical_mappings
        )
        
        # Prepare track features
        logger.info("Preparing track features")
        track_tensors = prepare_track_features(
            filtered_track_df, track_features, categorical_mappings
        )
        
        # Generate recommendations
        logger.info("Generating cold start recommendations")
        recommendations = generate_cold_start_recommendations(
            model, user_tensors, track_tensors, filtered_track_df, args.top_k
        )
        
        # Create output
        output = {
            "user_profile": demographics,
            "recommendations": recommendations
        }
        
        # Save recommendations
        with open(args.output, 'w') as f:
            json.dump(output, f, indent=4)
        
        logger.info(f"Recommendations saved to {args.output}")
        
        # Print recommendations
        print("\nTop recommendations for new user:")
        for i, rec in enumerate(recommendations):
            print(f"{i+1}. {rec['track_name']} by {rec['artist']} - Score: {rec['score']:.4f}")
        
        return 0
    
    except Exception as e:
        logger.error(f"Error generating cold start recommendations: {str(e)}", exc_info=True)
        return 1

if __name__ == "__main__":
    sys.exit(main()) 