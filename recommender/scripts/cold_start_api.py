#!/usr/bin/env python
"""
API service for cold start recommendations.

This script provides HTTP API endpoints for generating recommendations
for new users with no prior listening history.
"""

import os
import sys
import argparse
import logging
import yaml
import torch
import json
import numpy as np
from pathlib import Path
import flask
from flask import Flask, request, jsonify

# Add project root to path to allow imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from models.recommender import HybridRecommender
from utils.logger import setup_logger

# Initialize Flask app
app = Flask(__name__)

# Global variables
model = None
device = None
config = None
model_config = None
categorical_mappings = None
user_features = []
track_features = []
track_df = None
logger = None

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

def create_user_profile_from_demographics(demographics, preference_data):
    """
    Create a user profile tensor from demographic information.
    
    Args:
        demographics (dict): User demographic information (age, gender, country, etc.)
        preference_data (dict): User preferences (genres, artists, etc.)
        
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

def filter_tracks_by_preferences(preference_data):
    """
    Filter tracks based on user preferences.
    
    Args:
        preference_data (dict): User preferences (genres, artists, etc.)
        
    Returns:
        pd.DataFrame: Filtered track data
    """
    global track_df
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
                logger.warning(f"Artist filtering returned only {len(artist_filtered)} tracks. Using genre filtering only.")
    
    return filtered_df

def prepare_track_features(filtered_track_df):
    """
    Prepare track features for the model.
    
    Args:
        filtered_track_df (pd.DataFrame): Filtered track data
        
    Returns:
        dict: Track tensors for model input
    """
    track_tensors = {}
    
    # Process track features
    for feature in track_features:
        if feature in filtered_track_df.columns:
            # Check if this is a categorical feature
            if feature in categorical_mappings:
                mapping = categorical_mappings[feature]
                # Map values or use default (0) if not found
                values = filtered_track_df[feature].map(lambda x: mapping.get(x, 0))
                track_tensors[feature] = torch.tensor(values.values, dtype=torch.long)
            else:
                # Numerical feature
                track_tensors[feature] = torch.tensor(filtered_track_df[feature].values, dtype=torch.float)
        else:
            # Handle missing features
            if feature.endswith('_id') and feature.replace('_id', '') in filtered_track_df.columns:
                # Try to map from the non-id version
                base_feature = feature.replace('_id', '')
                if base_feature in categorical_mappings:
                    mapping = categorical_mappings[base_feature]
                    values = filtered_track_df[base_feature].map(lambda x: mapping.get(x, 0))
                    track_tensors[feature] = torch.tensor(values.values, dtype=torch.long)
                else:
                    # Default to zeros if mapping not available
                    track_tensors[feature] = torch.zeros(len(filtered_track_df), dtype=torch.long)
            else:
                # For missing features, use zeros
                if any(f.endswith('_id') for f in track_features) and feature in categorical_mappings:
                    # Categorical feature
                    track_tensors[feature] = torch.zeros(len(filtered_track_df), dtype=torch.long)
                else:
                    # Numerical feature
                    track_tensors[feature] = torch.zeros(len(filtered_track_df), dtype=torch.float)
    
    return track_tensors

def generate_cold_start_recommendations(user_tensors, track_tensors, filtered_track_df, top_k=10):
    """
    Generate recommendations for a new user with no prior history.
    
    Args:
        user_tensors (dict): User feature tensors
        track_tensors (dict): Track feature tensors
        filtered_track_df (pd.DataFrame): Filtered track data
        top_k (int): Number of recommendations to generate
        
    Returns:
        list: Recommended track IDs and scores
    """
    global model, device
    
    # Set model to evaluation mode and disable batch norm updates
    model.eval()
    
    # Fix for batch normalization with single samples
    for module in model.modules():
        if isinstance(module, torch.nn.BatchNorm1d):
            module.eval()
            module.track_running_stats = True
    
    # Move tensors to the same device as the model
    user_tensors = {k: v.to(device) for k, v in user_tensors.items()}
    track_tensors = {k: v.to(device) for k, v in track_tensors.items()}
    
    # Make predictions
    with torch.no_grad():
        try:
            outputs = model(user_tensors, track_tensors)
        except ValueError as e:
            if "Expected more than 1 value per channel when training" in str(e):
                # Handle batch normalization error by duplicating input
                logger.warning("Handling batch normalization for single sample by duplicating input")
                duplicated_user_tensors = {
                    k: torch.cat([v, v], dim=0) if len(v.shape) > 1 else v.repeat(2)
                    for k, v in user_tensors.items()
                }
                # Forward pass with duplicated user data
                duplicated_outputs = model(duplicated_user_tensors, track_tensors)
                # Take only the first half (original user)
                outputs = duplicated_outputs[:len(duplicated_outputs)//2]
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
        track_id = filtered_track_df.iloc[idx]['track_id']
        track_name = filtered_track_df.iloc[idx].get('track_name', f"Track {track_id}")
        artist = filtered_track_df.iloc[idx].get('artist', "Unknown")
        
        track_info = {
            'track_id': track_id,
            'track_name': track_name,
            'artist': artist,
            'score': float(score)
        }
        
        # Add additional metadata if available
        for col in ['album', 'genre', 'year', 'duration', 'popularity']:
            if col in filtered_track_df.columns:
                track_info[col] = filtered_track_df.iloc[idx][col]
        
        recommendations.append(track_info)
    
    return recommendations

@app.route('/api/recommendations/new-user', methods=['POST'])
def get_cold_start_recommendations():
    """API endpoint for cold start recommendations."""
    try:
        # Get request data
        data = request.json
        
        # Validate request data
        if not data:
            return jsonify({
                'error': 'No data provided',
                'status': 'error'
            }), 400
        
        # Extract demographics and preferences
        demographics = data.get('demographics', {})
        preferences = data.get('preferences', {})
        top_k = data.get('top_k', 10)
        
        # Log request
        logger.info(f"Received cold start recommendation request: user_id={data.get('user_id')}")
        
        # Filter tracks based on preferences
        filtered_track_df = filter_tracks_by_preferences(preferences)
        
        # Create user profile from demographics
        user_tensors = create_user_profile_from_demographics(demographics, preferences)
        
        # Prepare track features
        track_tensors = prepare_track_features(filtered_track_df)
        
        # Generate recommendations
        recommendations = generate_cold_start_recommendations(
            user_tensors, track_tensors, filtered_track_df, top_k
        )
        
        # Create response
        response = {
            'user_id': data.get('user_id', 'new_user'),
            'recommendations': recommendations,
            'status': 'success'
        }
        
        logger.info(f"Generated {len(recommendations)} recommendations for user_id={data.get('user_id')}")
        
        return jsonify(response)
    
    except Exception as e:
        logger.error(f"Error generating recommendations: {str(e)}", exc_info=True)
        return jsonify({
            'error': str(e),
            'status': 'error'
        }), 500

@app.route('/api/genres', methods=['GET'])
def get_available_genres():
    """API endpoint to get available genres for user selection."""
    try:
        # Get available genres from track data
        genre_cols = [col for col in track_df.columns if 'genre' in col.lower()]
        
        genres = set()
        if genre_cols:
            # Collect unique genres from all genre columns
            for col in genre_cols:
                genres.update(track_df[col].dropna().unique())
        
        # Filter out None and convert to list
        genres = [g for g in genres if g]
        
        return jsonify({
            'genres': sorted(genres),
            'status': 'success'
        })
    
    except Exception as e:
        logger.error(f"Error getting genres: {str(e)}", exc_info=True)
        return jsonify({
            'error': str(e),
            'status': 'error'
        }), 500

@app.route('/api/artists', methods=['GET'])
def get_available_artists():
    """API endpoint to get available artists for user selection."""
    try:
        # Get query parameter for search
        query = request.args.get('query', '').lower()
        limit = int(request.args.get('limit', 50))
        
        # Get artist column
        artist_cols = [col for col in track_df.columns if 'artist' in col.lower()]
        
        artists = set()
        if artist_cols:
            # Collect unique artists from all artist columns
            for col in artist_cols:
                if query:
                    # Filter by query
                    filtered_artists = track_df[track_df[col].str.lower().str.contains(query, na=False)][col]
                    artists.update(filtered_artists.dropna().unique())
                else:
                    # Get all artists (limited)
                    artists.update(track_df[col].dropna().unique()[:limit*2])  # Get more than needed for filtering
        
        # Filter out None and convert to list
        artists = [a for a in artists if a]
        
        # Sort by relevance if query provided, otherwise alphabetically
        if query:
            # Sort by how closely they match the query
            artists = sorted(artists, key=lambda x: (0 if query in x.lower() else 1, x.lower()))
        else:
            artists = sorted(artists)
        
        # Limit results
        artists = artists[:limit]
        
        return jsonify({
            'artists': artists,
            'status': 'success'
        })
    
    except Exception as e:
        logger.error(f"Error getting artists: {str(e)}", exc_info=True)
        return jsonify({
            'error': str(e),
            'status': 'error'
        }), 500

@app.route('/api/health', methods=['GET'])
def health_check():
    """API endpoint for health check."""
    return jsonify({
        'status': 'healthy',
        'model_loaded': model is not None,
        'tracks_loaded': track_df is not None and len(track_df) > 0
    })

def initialize_api(args):
    """Initialize the API service with the model and data."""
    global model, device, config, model_config, categorical_mappings, user_features, track_features, track_df, logger
    
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
    
    # Load track data
    logger.info(f"Loading track data from {args.track_data}")
    import pandas as pd
    track_df = pd.read_csv(args.track_data)
    logger.info(f"Loaded {len(track_df)} tracks")
    
    logger.info("API initialization complete")

def main():
    """Main function to start the API service."""
    parser = argparse.ArgumentParser(description="API service for cold start recommendations")
    parser.add_argument("--model", type=str, required=True,
                      help="Path to the trained model checkpoint")
    parser.add_argument("--config", type=str, required=True,
                      help="Path to configuration YAML file")
    parser.add_argument("--track-data", type=str, required=True,
                      help="Path to track data CSV file")
    parser.add_argument("--host", type=str, default="0.0.0.0",
                      help="Host to run the API service on")
    parser.add_argument("--port", type=int, default=5000,
                      help="Port to run the API service on")
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
        name="cold_start_api",
        log_file=os.path.join(log_dir, "cold_start_api.log"),
        level=getattr(logging, args.log_level)
    )
    
    try:
        logger.info("Starting cold start recommendation API service")
        
        # Initialize API
        initialize_api(args)
        
        # Start Flask app
        app.run(host=args.host, port=args.port)
        
    except Exception as e:
        logger.error(f"Error starting API service: {str(e)}", exc_info=True)
        return 1

if __name__ == "__main__":
    sys.exit(main()) 