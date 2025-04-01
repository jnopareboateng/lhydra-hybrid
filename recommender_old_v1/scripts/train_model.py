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
from training.trainer import HybridModelTrainer
from utils.data_utils import (
    load_config, 
    load_data, 
    load_preprocessed_data,
    prepare_user_item_data,
    preprocess_user_features, 
    preprocess_item_features,
    create_interaction_features,
    train_test_split_interactions,
    create_data_loaders
)
from utils.logging_utils import setup_logging, log_training_info

def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Train Hybrid Music Recommender Model')
    
    parser.add_argument('--config', type=str, default='training/configs/training_config.yaml', 
                        help='Path to configuration file')
    parser.add_argument('--data_dir', type=str, default='data', 
                        help='Directory containing datasets')
    parser.add_argument('--preprocessed_data', type=str, default=None,
                        help='Directory containing preprocessed data (if provided, skips preprocessing step)')
    parser.add_argument('--seed', type=int, default=42, 
                        help='Random seed')
    parser.add_argument('--gpu', action='store_true', 
                        help='Use GPU if available')
    parser.add_argument('--debug', action='store_true', 
                        help='Enable debug mode')
    
    return parser.parse_args()

def train_model(config, args=None):
    """Train the hybrid music recommender model.
    
    Args:
        config (dict): Configuration dictionary
        args (argparse.Namespace, optional): Command line arguments
    
    Returns:
        dict: Dictionary containing training history and test metrics
    """
    # Setup logging
    logger = setup_logging(config['logging']['log_dir'])
    
    # Print detailed debug header
    logger.info("\n" + "#"*100)
    logger.info("LHYDRA MUSIC RECOMMENDER - DETAILED TRAINING DEBUG INFORMATION")
    logger.info("#"*100 + "\n")
    
    # Log configuration details
    logger.info("="*80)
    logger.info("CONFIGURATION SUMMARY")
    logger.info("="*80)
    logger.info(f"Data config: {config.get('data', {})}")
    logger.info(f"Model config: {config.get('model', {})}")
    logger.info(f"Training config: {config.get('training', {})}")
    logger.info(f"Evaluation config: {config.get('evaluation', {})}")
    logger.info("="*80 + "\n")
    
    # Set random seeds
    torch.manual_seed(config['data']['random_seed'])
    np.random.seed(config['data']['random_seed'])
    
    # Set device
    use_gpu = args.gpu if args else False
    device = torch.device('cuda' if torch.cuda.is_available() and use_gpu else 'cpu')
    logger.info(f"Using device: {device}")
    
    # Load preprocessed data if specified
    preprocessed_data = args.preprocessed_data if args else None
    
    if preprocessed_data:
        # Use preprocessed data
        logger.info(f"Using preprocessed data from {preprocessed_data}")
        train_df, val_df, test_df, preprocessor = load_preprocessed_data(preprocessed_data)
        
        # Extract user and item features from preprocessed data
        interactions_df = pd.concat([train_df, val_df, test_df], ignore_index=True)
        interactions_df, user_features_df, item_features_df = prepare_user_item_data(
            interactions_df, 
            config['data'],
            use_high_engagement=True,
            save_features=True,
            output_dir=os.path.join(config['data']['output_dir'], 'features')
        )
    else:
        # Load and preprocess data
        logger.info("Loading and preprocessing data...")
        try:
            # Load raw data
            data_dir = args.data_dir if args else 'data'
            data_path = os.path.join(data_dir, "interactions.csv")
            if 'train_path' in config.get('data', {}):
                data_path = config['data']['train_path']
            
            interactions_df = load_data(data_path)
            
            # Log detailed data information
            logger.info("\n" + "="*80)
            logger.info("RAW DATA INFORMATION")
            logger.info("="*80)
            logger.info(f"Interactions shape: {interactions_df.shape}")
            logger.info(f"Interactions columns: {list(interactions_df.columns)}")
            logger.info(f"Interactions dtypes:\n{interactions_df.dtypes}")
            logger.info(f"Interactions sample (first 3 rows):\n{interactions_df.head(3).to_string()}")
            logger.info("="*80 + "\n")
            
            # Load user and item features if separate files are specified
            user_features_df = None
            item_features_df = None
            
            if 'user_features_path' in config.get('data', {}) and 'item_features_path' in config.get('data', {}):
                user_features_df = load_data(config['data']['user_features_path'])
                item_features_df = load_data(config['data']['item_features_path'])
                
                # Log user and item features
                logger.info("\n" + "="*80)
                logger.info("LOADED FEATURES INFORMATION")
                logger.info("="*80)
                logger.info(f"User features shape: {user_features_df.shape}")
                logger.info(f"User features columns: {list(user_features_df.columns)}")
                logger.info(f"Item features shape: {item_features_df.shape}")
                logger.info(f"Item features columns: {list(item_features_df.columns)}")
                logger.info("="*80 + "\n")
                
                logger.info(f"Loaded feature data: {len(user_features_df)} users, {len(item_features_df)} items")
            else:
                # Extract features from interactions dataframe
                interactions_df, user_features_df, item_features_df = prepare_user_item_data(
                    interactions_df,
                    config['data'],
                    save_features=True,
                    output_dir=os.path.join(config['data']['output_dir'], 'features')
                )
            
            # Preprocess user and item features
            user_features_df, user_metadata = preprocess_user_features(user_features_df)
            item_features_df, item_metadata = preprocess_item_features(item_features_df)
            
            # Log preprocessed features
            logger.info("\n" + "="*80)
            logger.info("PREPROCESSED FEATURES INFORMATION")
            logger.info("="*80)
            logger.info(f"Preprocessed user features shape: {user_features_df.shape}")
            logger.info(f"Preprocessed user features columns: {list(user_features_df.columns)}")
            logger.info(f"Preprocessed item features shape: {item_features_df.shape}")
            logger.info(f"Preprocessed item features columns: {list(item_features_df.columns)}")
            
            # Log metadata for normalization
            if user_metadata:
                logger.info("\nUser features metadata:")
                for key, value in user_metadata.items():
                    if key == 'user_scaler':
                        logger.info(f"  - User scaler: {type(value).__name__}")
                    elif key == 'user_numerical_cols':
                        logger.info(f"  - User numerical columns: {list(value)}")
            
            if item_metadata:
                logger.info("\nItem features metadata:")
                for key, value in item_metadata.items():
                    if key == 'item_scaler':
                        logger.info(f"  - Item scaler: {type(value).__name__}")
                    elif key == 'item_numerical_cols':
                        logger.info(f"  - Item numerical columns: {list(value)}")
            logger.info("="*80 + "\n")
            
            # Create interaction features and binary labels
            interactions_df = create_interaction_features(
                interactions_df, 
                user_features_df, 
                item_features_df, 
                config['data']['target_threshold']
            )
            
            # Split data
            train_df, val_df, test_df = train_test_split_interactions(
                interactions_df,
                test_size=config['data'].get('test_size', 0.2),
                val_size=config['data'].get('validation_size', 0.1),
                random_state=config['data']['random_seed']
            )
            
        except Exception as e:
            logger.error(f"Error loading/preprocessing data: {e}", exc_info=True)
            raise
    
    logger.info(f"Data split: train={len(train_df)}, val={len(val_df)}, test={len(test_df)}")
    
    # Create data loaders with target column name from config
    target_col = config['data'].get('target_column', 'high_engagement')
    dataloaders = create_data_loaders(
        train_df,
        val_df,
        test_df,
        user_features_df,
        item_features_df,
        batch_size=config['training']['batch_size'],
        target_col=target_col
    )
    
    # Initialize model
    logger.info("Initializing model...")
    
    # Calculate input dimensions excluding ID columns
    user_input_dim = len(user_features_df.columns)
    if not user_features_df.index.name == 'user_id' and 'user_id' in user_features_df.columns:
        user_input_dim -= 1  # Subtract 1 for user_id column
    
    item_input_dim = len(item_features_df.columns)
    if not item_features_df.index.name == 'track_id' and 'track_id' in item_features_df.columns:
        item_input_dim -= 1  # Subtract 1 for track_id/item_id column
    
    logger.info(f"Feature dimensions: user={user_input_dim}, item={item_input_dim}")
    
    # Get model config parameters with defaults
    model_config = config.get('model', {})
    user_hidden_dims = model_config.get('hidden_dims', [128, 64]) 
    item_hidden_dims = model_config.get('hidden_dims', [128, 64])
    embedding_dim = model_config.get('embedding_dim', 32)
    dropout = model_config.get('dropout', 0.2)
    
    # Check if user_tower and item_tower are in config
    if 'user_tower' in model_config and 'hidden_layers' in model_config['user_tower']:
        user_hidden_dims = model_config['user_tower']['hidden_layers']
    if 'item_tower' in model_config and 'hidden_layers' in model_config['item_tower']:
        item_hidden_dims = model_config['item_tower']['hidden_layers']
    
    # Set final layer size
    final_layer_size = model_config.get('final_layer_size', 16)
    
    # Set activation function
    activation = 'relu'
    if 'user_tower' in model_config and 'activation' in model_config['user_tower']:
        activation = model_config['user_tower']['activation']
    
    # Log model architecture before creation
    logger.info("\n" + "="*80)
    logger.info("MODEL ARCHITECTURE INFORMATION")
    logger.info("="*80)
    logger.info(f"User Input Dimension: {user_input_dim}")
    logger.info(f"Item Input Dimension: {item_input_dim}")
    logger.info(f"User Hidden Dimensions: {user_hidden_dims}")
    logger.info(f"Item Hidden Dimensions: {item_hidden_dims}")
    logger.info(f"Embedding Dimension: {embedding_dim}")
    logger.info(f"Final Layer Size: {final_layer_size}")
    logger.info(f"Dropout Rate: {dropout}")
    logger.info(f"Activation Function: {activation}")
    logger.info("="*80 + "\n")
    
    model = TwoTowerHybridModel(
        user_input_dim=user_input_dim,
        item_input_dim=item_input_dim,
        user_hidden_dims=user_hidden_dims,
        item_hidden_dims=item_hidden_dims,
        embedding_dim=embedding_dim,
        final_layer_size=final_layer_size,
        dropout=dropout,
        activation=activation
    )
    
    # Initialize trainer
    training_config = config.get('training', {})
    trainer = HybridModelTrainer(model, config, device)
    
    # Train model
    logger.info("Starting training...")
    history = trainer.train(
        dataloaders['train'],
        dataloaders['validation'],
        num_epochs=training_config.get('num_epochs', 100)
    )
    
    # Save training history
    log_dir = training_config.get('log_dir', 'logs')
    history_path = os.path.join(log_dir, 'training_history.csv')
    history.to_csv(history_path, index=False)
    logger.info(f"Training history saved to {history_path}")
    
    # Evaluate on test set
    logger.info("Evaluating model on test set...")
    test_metrics = trainer.evaluate(dataloaders['test'])
    
    # Log test metrics
    logger.info("Test Set Metrics:")
    for metric_name, metric_value in test_metrics.items():
        logger.info(f"{metric_name}: {metric_value:.4f}")
    
    # Save test metrics to file
    metrics_dir = config.get('evaluation', {}).get('metrics_dir', 'evaluation/metrics')
    os.makedirs(metrics_dir, exist_ok=True)
    metrics_path = os.path.join(metrics_dir, 'test_metrics.csv')
    pd.DataFrame([test_metrics]).to_csv(metrics_path, index=False)
    logger.info(f"Test metrics saved to {metrics_path}")
    
    # Save feature dimensions in the manifest
    feature_manifest = {
        'model_dimensions': {
            'user_input_dim': user_input_dim,
            'item_input_dim': item_input_dim,
            'embedding_dim': embedding_dim
        },
        'user_features': {
            'columns': list(user_features_df.columns)
        },
        'item_features': {
            'columns': list(item_features_df.columns)
        }
    }
    
    # Save the model with feature manifest
    model_path = os.path.join(config['logging']['model_checkpoint_dir'], 'final_model_with_manifest.pt')
    model.save(model_path, feature_manifest=feature_manifest)
    logger.info(f"Model saved with feature manifest to {model_path}")
    
    logger.info("=" * 80)
    logger.info(f"Training completed at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info("=" * 80)
    
    return {
        'history': history,
        'test_metrics': test_metrics,
        'model': model,
        'trainer': trainer
    }

def main():
    """Main function for training the model"""
    # Parse arguments
    args = parse_args()
    
    # Load configuration
    config = load_config(args.config)
    
    # Override config with command-line arguments
    config['data']['random_seed'] = args.seed
    
    # Train the model
    train_model(config, args)

if __name__ == '__main__':
    main() 