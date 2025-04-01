#!/usr/bin/env python
"""
Script to train the hybrid recommender model.

This script loads preprocessed data, creates and trains the model,
and saves checkpoints and evaluation metrics.
"""

import os
import sys
import argparse
import logging
import joblib
import yaml
import torch
import pandas as pd
from pathlib import Path

# Add project root to path to allow imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from data.dataset import create_dataloader
from training.trainer import ModelTrainer
from utils.logger import setup_logger

def main():
    """Main function to train the model."""
    parser = argparse.ArgumentParser(description='Train the hybrid recommender model')
    parser.add_argument('--config', type=str, required=True, 
                        help='Path to training configuration YAML file')
    parser.add_argument('--resume', type=str, default=None, 
                        help='Path to checkpoint to resume training from')
    parser.add_argument('--log-level', type=str, choices=['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'],
                        default='INFO', help='Set the logging level')
    parser.add_argument('--device', type=str, default=None,
                        help='Device to use for training (cuda or cpu)')
    
    args = parser.parse_args()
    
    # Create logs directory
    log_dir = 'logs'
    os.makedirs(log_dir, exist_ok=True)
    
    # Set up logging
    logger = setup_logger(
        name="train_model",
        log_file=os.path.join(log_dir, "train_model.log"),
        level=getattr(logging, args.log_level)
    )
    
    try:
        logger.info("Starting model training")
        
        # Load configuration
        with open(args.config, 'r') as f:
            config = yaml.safe_load(f)
        
        # Set device
        if args.device:
            device = torch.device(args.device)
        else:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        logger.info(f"Using device: {device}")
        
        # Load preprocessor
        preprocessor_path = config['data']['preprocessor_path']
        if os.path.exists(preprocessor_path):
            logger.info(f"Loading preprocessor from {preprocessor_path}")
            preprocessor = joblib.load(preprocessor_path)
            categorical_mappings = preprocessor.get('categorical_mappings', {})
        else:
            logger.warning(f"Preprocessor not found at {preprocessor_path}")
            categorical_mappings = {}
        
        # Extract feature lists from model config
        model_config_path = config['model']['config_path']
        with open(model_config_path, 'r') as f:
            model_config = yaml.safe_load(f)
        
        # Get user and track features
        user_numerical = model_config['features']['user_features']['numerical']
        user_categorical = model_config['features']['user_features']['categorical']
        track_numerical = model_config['features']['track_features']['numerical']
        track_categorical = model_config['features']['track_features']['categorical']
        
        # Check actual available columns in the data
        train_data_path = config['data']['train_data_path']
        logger.info(f"Loading data from {train_data_path} to verify columns")
        try:
            train_df = pd.read_csv(train_data_path)
            available_columns = train_df.columns.tolist()
            logger.info(f"Available columns in data: {available_columns}")
        except Exception as e:
            logger.error(f"Error loading data to verify columns: {str(e)}")
            available_columns = []
        
        # Create list of features, avoiding duplicate _id suffixes
        user_features = []
        for feature in user_numerical:
            if feature in available_columns:
                user_features.append(feature)
        
        for feature in user_categorical:
            if feature in available_columns:
                user_features.append(feature)
            # Add ID version if not already an ID feature
            id_feature = f"{feature}_id"
            if id_feature in available_columns:
                user_features.append(id_feature)
        
        track_features = []
        for feature in track_numerical:
            if feature in available_columns:
                track_features.append(feature)
        
        for feature in track_categorical:
            if feature in available_columns:
                track_features.append(feature)
            # Add ID version if not already an ID feature
            if not feature.endswith('_id'):
                id_feature = f"{feature}_id"
                if id_feature in available_columns:
                    track_features.append(id_feature)
        
        # Add engineered features if included
        if config['features']['include_engineered'] and 'engineered_features' in model_config['features']:
            for feature in model_config['features']['engineered_features']:
                if feature in available_columns:
                    track_features.append(feature)
        
        logger.info(f"Selected user features: {user_features}")
        logger.info(f"Selected track features: {track_features}")
        
        # Create training and validation dataloaders
        logger.info("Creating train dataloader")
        train_loader = create_dataloader(
            data_path=config['data']['train_data_path'],
            user_features=user_features,
            track_features=track_features,
            batch_size=config['data']['batch_size'],
            shuffle=True,
            categorical_mappings=categorical_mappings,
            num_workers=config['data']['num_workers']
        )
        
        logger.info("Creating validation dataloader")
        val_loader = create_dataloader(
            data_path=config['data']['val_data_path'],
            user_features=user_features,
            track_features=track_features,
            batch_size=config['data']['batch_size'],
            shuffle=False,
            categorical_mappings=categorical_mappings,
            num_workers=config['data']['num_workers']
        )
        
        # Create trainer and train the model
        logger.info("Initializing trainer")
        trainer = ModelTrainer(args.config)
        
        # Train the model
        logger.info("Starting training process")
        training_stats = trainer.train(
            train_loader=train_loader,
            val_loader=val_loader,
            categorical_mappings=categorical_mappings
        )
        
        # Log training summary
        logger.info("Training completed:")
        logger.info(f"  Best validation metrics:")
        for metric_name, metric_value in training_stats.items():
            if isinstance(metric_value, (int, float)):
                logger.info(f"    {metric_name}: {metric_value:.4f}")
        
        # Save best model with full config
        best_model_path = config['model']['best_model_path']
        if os.path.exists(best_model_path):
            logger.info(f"Best model saved to {best_model_path}")
        
        return 0
        
    except Exception as e:
        logger.error(f"Error during training: {str(e)}", exc_info=True)
        return 1

if __name__ == "__main__":
    sys.exit(main()) 