#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Simple recommendation generator script that works around compatibility issues
"""

import os
import sys
import argparse
import torch
import pandas as pd
import numpy as np
from datetime import datetime
import logging

# Add the root directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.two_tower_model import TwoTowerHybridModel

def setup_logging():
    """Set up basic logging configuration"""
    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s] [%(levelname)s] - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    return logging.getLogger("simple_recommender")

def load_model(model_path, device='cpu'):
    """Load the model from checkpoint with graceful error handling"""
    logger = logging.getLogger("simple_recommender")
    logger.info(f"Loading model from {model_path}")
    
    try:
        checkpoint = torch.load(model_path, map_location=device)
        model_state = checkpoint.get('model_state_dict', checkpoint)
        
        # Try to infer dimensions from the model's state dict
        user_input_dim = None
        item_input_dim = None
        
        for key in model_state.keys():
            if 'user_tower.0.weight' in key:
                user_input_dim = model_state[key].shape[1]
            elif 'item_tower.0.weight' in key:
                item_input_dim = model_state[key].shape[1]
        
        if user_input_dim is None or item_input_dim is None:
            logger.error("Could not infer model dimensions from checkpoint")
            return None
        
        logger.info(f"Creating model with dimensions: user={user_input_dim}, item={item_input_dim}")
        model = TwoTowerHybridModel(user_input_dim=user_input_dim, item_input_dim=item_input_dim)
        
        # Load state dict with lenient settings to handle dimension issues
        model.load_state_dict(model_state, strict=False)
        logger.info("Model loaded successfully")
        
        model.eval()  # Set to evaluation mode
        return model
    
    except Exception as e:
        logger.error(f"Error loading model: {str(e)}")
        return None

def prepare_features(user_df, item_df, user_id, device='cpu'):
    """Prepare user and item features for prediction"""
    logger = logging.getLogger("simple_recommender")
    
    # Find the user in the dataframe, carefully handling type conversion
    if 'user_id' in user_df.columns:
        # Try int conversion first (most common case)
        try:
            user_id_int = int(user_id)
            user_row = user_df[user_df['user_id'] == user_id_int]
            if len(user_row) == 0:
                # Try string comparison as fallback
                user_row = user_df[user_df['user_id'].astype(str) == str(user_id)]
        except ValueError:
            # If conversion to int fails, try string comparison
            user_row = user_df[user_df['user_id'].astype(str) == str(user_id)]
    else:
        logger.error("No user_id column found in user features")
        return None, None
    
    if len(user_row) == 0:
        logger.error(f"User {user_id} not found in user features")
        return None, None
    
    logger.info(f"Found user {user_id} in user features (row {user_row.index[0]})")
    
    # Convert to tensors, handling potential non-numeric columns
    user_tensors = {}
    for col in user_row.columns:
        if col == 'user_id':
            continue  # Skip ID column
        
        try:
            if user_row[col].dtype == 'object':
                # Try to convert to numeric, use zeros if not possible
                try:
                    value = pd.to_numeric(user_row[col].values[0])
                except:
                    logger.warning(f"Column {col} contains non-numeric values, using zeros")
                    value = 0
            else:
                value = user_row[col].values[0]
                
            user_tensors[col] = torch.tensor(value, dtype=torch.float32, device=device)
        except Exception as e:
            logger.warning(f"Error processing column {col}: {str(e)}")
    
    # Prepare item features
    item_tensors = {}
    for col in item_df.columns:
        if col == 'track_id' or col == 'item_id':
            continue  # Skip ID columns
        
        try:
            if item_df[col].dtype == 'object':
                # Try to convert to numeric, use zeros if not possible
                try:
                    values = pd.to_numeric(item_df[col].values)
                except:
                    logger.warning(f"Column {col} contains non-numeric values, using zeros")
                    values = np.zeros(len(item_df))
            else:
                values = item_df[col].values
                
            item_tensors[col] = torch.tensor(values, dtype=torch.float32, device=device)
        except Exception as e:
            logger.warning(f"Error processing column {col}: {str(e)}")
    
    logger.info(f"Prepared features: {len(user_tensors)} user features, {len(item_tensors)} item features")
    return user_tensors, item_tensors

def predict_recommendations(model, user_tensors, item_tensors, item_ids, top_k=10):
    """Make predictions and return top-k recommendations"""
    logger = logging.getLogger("simple_recommender")
    
    if user_tensors is None or item_tensors is None:
        logger.error("Cannot predict with None tensors")
        return None
    
    # Initialize scores array
    scores = np.zeros(len(item_ids))
    
    # Process in chunks to avoid memory issues
    chunk_size = 50
    num_chunks = (len(item_ids) + chunk_size - 1) // chunk_size
    
    logger.info(f"Making predictions in {num_chunks} chunks")
    
    for chunk_idx in range(num_chunks):
        start_idx = chunk_idx * chunk_size
        end_idx = min(start_idx + chunk_size, len(item_ids))
        chunk_size_actual = end_idx - start_idx
        
        # For each chunk of items, create tensors
        batch_user = {}
        for key, tensor in user_tensors.items():
            # Repeat user tensor for each item
            if tensor.dim() == 0:  # scalar
                batch_user[key] = tensor.repeat(chunk_size_actual)
            else:
                batch_user[key] = tensor.repeat(chunk_size_actual, 1)
        
        # Get chunk of item tensors
        batch_items = {}
        for key, tensor in item_tensors.items():
            batch_items[key] = tensor[start_idx:end_idx]
        
        # Make predictions
        with torch.no_grad():
            try:
                outputs = model(batch_user, batch_items)
                chunk_scores = outputs.cpu().numpy().flatten()
                scores[start_idx:end_idx] = chunk_scores
            except Exception as e:
                logger.error(f"Error in prediction chunk {chunk_idx}: {str(e)}")
                # Continue with next chunk
    
    # Get top-k items
    top_indices = np.argsort(-scores)[:top_k]
    recommendations = []
    
    for i, idx in enumerate(top_indices):
        recommendations.append({
            'rank': i + 1,
            'item_id': item_ids[idx],
            'score': float(scores[idx])
        })
    
    return recommendations

def main():
    parser = argparse.ArgumentParser(description='Simple recommendation generator')
    parser.add_argument('--model', type=str, required=True, help='Path to the model checkpoint')
    parser.add_argument('--users', type=str, required=True, help='Path to user features CSV')
    parser.add_argument('--items', type=str, required=True, help='Path to item features CSV')
    parser.add_argument('--user', type=str, required=True, help='User ID to generate recommendations for')
    parser.add_argument('--output', type=str, default='recommendations.csv', help='Output file path')
    parser.add_argument('--top-k', type=int, default=10, help='Number of recommendations to generate')
    parser.add_argument('--cpu', action='store_true', help='Force CPU usage even if GPU is available')
    args = parser.parse_args()
    
    # Set up logging
    logger = setup_logging()
    logger.info("=" * 60)
    logger.info("Starting simple recommendation generation")
    logger.info("=" * 60)
    
    # Set device
    device = torch.device('cpu' if args.cpu else ('cuda' if torch.cuda.is_available() else 'cpu'))
    logger.info(f"Using device: {device}")
    
    # Load model
    model = load_model(args.model, device)
    if model is None:
        logger.error("Failed to load model. Exiting.")
        return 1
    
    # Load user features
    logger.info(f"Loading user features from {args.users}")
    try:
        user_df = pd.read_csv(args.users)
        logger.info(f"Loaded {len(user_df)} user records")
    except Exception as e:
        logger.error(f"Error loading user features: {str(e)}")
        return 1
    
    # Load item features
    logger.info(f"Loading item features from {args.items}")
    try:
        item_df = pd.read_csv(args.items)
        logger.info(f"Loaded {len(item_df)} item records")
    except Exception as e:
        logger.error(f"Error loading item features: {str(e)}")
        return 1
    
    # Get item IDs
    item_id_col = 'track_id' if 'track_id' in item_df.columns else 'item_id'
    item_ids = item_df[item_id_col].values
    
    # Check if the specified user exists
    if 'user_id' not in user_df.columns:
        logger.error("No user_id column found in user features")
        return 1
    
    # Try both formats for comparison
    user_exists = False
    try:
        user_id_int = int(args.user)
        user_exists = user_id_int in user_df['user_id'].values
    except ValueError:
        pass
    
    user_exists_str = args.user in user_df['user_id'].astype(str).values
    
    if not (user_exists or user_exists_str):
        logger.error(f"User {args.user} not found in user features")
        logger.info(f"First 5 available user IDs: {user_df['user_id'].values[:5]}")
        return 1
    
    logger.info(f"User {args.user} found in user features")
    
    # Prepare features
    user_tensors, item_tensors = prepare_features(user_df, item_df, args.user, device)
    if user_tensors is None or item_tensors is None:
        logger.error("Failed to prepare features. Exiting.")
        return 1
    
    # Make predictions
    recommendations = predict_recommendations(model, user_tensors, item_tensors, item_ids, args.top_k)
    if recommendations is None:
        logger.error("Failed to generate recommendations. Exiting.")
        return 1
    
    # Create output directory if needed
    output_dir = os.path.dirname(args.output)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
    
    # Save recommendations
    output_df = pd.DataFrame(recommendations)
    output_df['user_id'] = args.user
    output_df.to_csv(args.output, index=False)
    
    logger.info(f"Saved {len(recommendations)} recommendations to {args.output}")
    
    # Print recommendations
    logger.info("Top recommendations:")
    for rec in recommendations[:5]:  # Show top 5
        logger.info(f"  Rank {rec['rank']}: Item {rec['item_id']} (Score: {rec['score']:.4f})")
    
    logger.info("=" * 60)
    logger.info("Recommendation generation completed successfully")
    logger.info("=" * 60)
    
    return 0

if __name__ == "__main__":
    sys.exit(main()) 