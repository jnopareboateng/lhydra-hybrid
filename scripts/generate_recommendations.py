#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import sys
import argparse
import yaml
import torch
import logging
import pandas as pd
import numpy as np
from datetime import datetime

# Add the root directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.two_tower_model import TwoTowerHybridModel
from inference.recommendation_generator import RecommendationGenerator
from utils.data_utils import load_config, load_data, preprocess_user_features, preprocess_item_features
from utils.logging_utils import setup_logging, log_inference_results

def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Generate Recommendations from Hybrid Music Recommender Model')
    
    parser.add_argument('--config', type=str, default='training/configs/config.yaml', 
                        help='Path to configuration file')
    parser.add_argument('--model', type=str, default='models/checkpoints/best_model.pt', 
                        help='Path to trained model checkpoint')
    parser.add_argument('--output', type=str, default='recommendations/recommendations.csv', 
                        help='Path to save recommendations')
    parser.add_argument('--top-k', type=int, default=10, 
                        help='Number of recommendations per user')
    parser.add_argument('--exclude-history', action='store_true', 
                        help='Exclude user history from recommendations')
    parser.add_argument('--user', type=str, default=None, 
                        help='Generate recommendations for specific user ID')
    parser.add_argument('--item', type=str, default=None, 
                        help='Find similar items to this item ID')
    parser.add_argument('--item-similarity', action='store_true', 
                        help='Generate item-item similarity matrix')
    parser.add_argument('--gpu', action='store_true', 
                        help='Use GPU if available')
    parser.add_argument('--debug', action='store_true', 
                        help='Enable debug mode')
    
    return parser.parse_args()

def main():
    """Main function for generating recommendations"""
    # Parse arguments
    args = parse_args()
    
    # Load configuration
    config = load_config(args.config)
    
    # Setup logging
    log_level = logging.DEBUG if args.debug else logging.INFO
    logger = setup_logging(config['logging']['log_dir'], log_level)
    
    # Log the start of recommendation generation
    logger.info("=" * 80)
    logger.info(f"Starting recommendation generation at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info("=" * 80)
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() and args.gpu else 'cpu')
    logger.info(f"Using device: {device}")
    
    # Load data
    logger.info("Loading data...")
    
    try:
        # Load preprocessed user and item features
        user_features_df = load_data(config['data']['user_features_path'])
        item_features_df = load_data(config['data']['item_features_path'])
        
        logger.info(f"Data loaded successfully: {len(user_features_df)} users, {len(item_features_df)} items")
        
        # Load interaction history if excluding history
        if args.exclude_history:
            history_df = load_data(config['data']['train_path'])
            logger.info(f"Loaded interaction history: {len(history_df)} interactions")
        else:
            history_df = None
    except Exception as e:
        logger.error(f"Error loading data: {e}")
        sys.exit(1)
    
    # Initialize recommendation generator
    logger.info(f"Loading model from {args.model}...")
    
    try:
        # Create recommendation generator from checkpoint
        rec_generator = RecommendationGenerator.from_checkpoint(
            args.model,
            item_features_df,
            user_features_df,
            device
        )
        logger.info("Model loaded successfully")
    except Exception as e:
        logger.error(f"Error loading model: {e}")
        sys.exit(1)
    
    # Generate recommendations based on requested operation
    if args.user:
        # Generate recommendations for a specific user
        logger.info(f"Generating recommendations for user {args.user}...")
        
        # Get user's excluded items if needed
        exclude_items = None
        if args.exclude_history and history_df is not None:
            exclude_items = history_df[history_df['user_id'] == args.user]['item_id'].tolist()
        
        # Generate recommendations
        recommendations = rec_generator.get_recommendations_for_user(
            args.user,
            top_k=args.top_k,
            exclude_items=exclude_items
        )
        
        if recommendations.empty:
            logger.warning(f"No recommendations generated for user {args.user}")
        else:
            logger.info(f"Generated {len(recommendations)} recommendations for user {args.user}")
            
            # Save to file
            if args.output:
                output_dir = os.path.dirname(args.output)
                if output_dir:
                    os.makedirs(output_dir, exist_ok=True)
                
                # Modify filename to include user ID
                filename = os.path.basename(args.output)
                basename, ext = os.path.splitext(filename)
                user_output = os.path.join(output_dir, f"{basename}_user_{args.user}{ext}")
                
                recommendations.to_csv(user_output, index=False)
                logger.info(f"Saved recommendations to {user_output}")
            
            # Display recommendations
            logger.info("\nTop recommendations:")
            for i, row in recommendations.iterrows():
                logger.info(f"Rank {row['rank']}: Item {row['item_id']} (Score: {row['score']:.4f})")
    
    elif args.item:
        # Find similar items
        logger.info(f"Finding items similar to {args.item}...")
        
        similar_items = rec_generator.find_similar_items(args.item, top_k=args.top_k)
        
        if similar_items.empty:
            logger.warning(f"No similar items found for item {args.item}")
        else:
            logger.info(f"Found {len(similar_items)} similar items for item {args.item}")
            
            # Save to file
            if args.output:
                output_dir = os.path.dirname(args.output)
                if output_dir:
                    os.makedirs(output_dir, exist_ok=True)
                
                # Modify filename to include item ID
                filename = os.path.basename(args.output)
                basename, ext = os.path.splitext(filename)
                item_output = os.path.join(output_dir, f"{basename}_similar_to_{args.item}{ext}")
                
                similar_items.to_csv(item_output, index=False)
                logger.info(f"Saved similar items to {item_output}")
            
            # Display similar items
            logger.info("\nSimilar items:")
            for i, row in similar_items.iterrows():
                logger.info(f"Rank {row['rank']}: Item {row['item_id']} (Similarity: {row['similarity']:.4f})")
    
    elif args.item_similarity:
        # Generate item-item similarity matrix
        logger.info("Generating item-item similarity matrix...")
        
        # Get item embeddings
        item_embeddings = rec_generator.get_item_embeddings()
        
        # Calculate similarity matrix
        item_ids = list(item_embeddings.keys())
        n_items = len(item_ids)
        
        logger.info(f"Calculating similarities between {n_items} items...")
        
        # Create similarity matrix
        similarity_matrix = []
        
        for i, item1 in enumerate(item_ids):
            if i % 100 == 0:
                logger.info(f"Progress: {i}/{n_items} items processed")
            
            embedding1 = item_embeddings[item1]
            norm1 = np.linalg.norm(embedding1)
            
            for item2 in item_ids[i+1:]:
                embedding2 = item_embeddings[item2]
                norm2 = np.linalg.norm(embedding2)
                
                # Cosine similarity
                similarity = np.dot(embedding1, embedding2) / (norm1 * norm2)
                
                similarity_matrix.append({
                    'item1': item1,
                    'item2': item2,
                    'similarity': similarity
                })
        
        # Create DataFrame
        similarity_df = pd.DataFrame(similarity_matrix)
        
        # Save to file
        if args.output:
            output_dir = os.path.dirname(args.output)
            if output_dir:
                os.makedirs(output_dir, exist_ok=True)
            
            # Modify filename to include indication it's a similarity matrix
            filename = os.path.basename(args.output)
            basename, ext = os.path.splitext(filename)
            sim_output = os.path.join(output_dir, f"{basename}_item_similarity{ext}")
            
            similarity_df.to_csv(sim_output, index=False)
            logger.info(f"Saved item-item similarity matrix to {sim_output}")
    
    else:
        # Generate recommendations for all users
        logger.info(f"Generating top-{args.top_k} recommendations for all users...")
        
        recommendations = rec_generator.generate_all_recommendations(
            top_k=args.top_k,
            exclude_history=history_df if args.exclude_history else None,
            output_file=args.output if args.output else None
        )
        
        if recommendations.empty:
            logger.warning("No recommendations generated")
        else:
            logger.info(f"Generated {len(recommendations)} recommendations for {recommendations['user_id'].nunique()} users")
            
            # Get distribution of scores
            score_stats = recommendations['score'].describe()
            logger.info("\nRecommendation score distribution:")
            logger.info(f"  Mean: {score_stats['mean']:.4f}")
            logger.info(f"  Min: {score_stats['min']:.4f}")
            logger.info(f"  25%: {score_stats['25%']:.4f}")
            logger.info(f"  50%: {score_stats['50%']:.4f}")
            logger.info(f"  75%: {score_stats['75%']:.4f}")
            logger.info(f"  Max: {score_stats['max']:.4f}")
    
    logger.info("=" * 80)
    logger.info(f"Recommendation generation completed at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info("=" * 80)

if __name__ == '__main__':
    main() 