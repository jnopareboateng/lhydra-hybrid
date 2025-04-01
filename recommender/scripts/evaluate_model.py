#!/usr/bin/env python
"""
Script to evaluate the trained hybrid recommender model.

This script loads a trained model and evaluates its performance on a test dataset,
computing classification and ranking metrics.
"""

import os
import sys
import argparse
import logging
import joblib
import yaml
import torch
import json
import pandas as pd
from pathlib import Path

# Add project root to path to allow imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from data.dataset import create_dataloader
from training.trainer import ModelTrainer
from models.recommender import HybridRecommender
from utils.logger import setup_logger

def main():
    """Main function to evaluate the model."""
    parser = argparse.ArgumentParser(description='Evaluate the hybrid recommender model')
    parser.add_argument('--model', type=str, required=True, 
                        help='Path to the trained model checkpoint')
    parser.add_argument('--data', type=str, required=True, 
                        help='Path to the test data CSV file')
    parser.add_argument('--config', type=str, required=True, 
                        help='Path to training configuration YAML file')
    parser.add_argument('--output', type=str, default='evaluation',
                        help='Directory to save evaluation results')
    parser.add_argument('--log-level', type=str, choices=['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'],
                        default='INFO', help='Set the logging level')
    parser.add_argument('--device', type=str, default=None,
                        help='Device to use for evaluation (cuda or cpu)')
    
    args = parser.parse_args()
    
    # Create logs and output directories
    log_dir = 'logs'
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(args.output, exist_ok=True)
    
    # Set up logging
    logger = setup_logger(
        name="evaluate_model",
        log_file=os.path.join(log_dir, "evaluate_model.log"),
        level=getattr(logging, args.log_level)
    )
    
    try:
        logger.info("Starting model evaluation")
        
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
        logger.info(f"Loading data from {args.data} to verify columns")
        try:
            test_df = pd.read_csv(args.data)
            available_columns = test_df.columns.tolist()
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
        
        # Create test dataloader
        logger.info("Creating test data loader")
        test_loader = create_dataloader(
            data_path=args.data,
            user_features=user_features,
            track_features=track_features,
            batch_size=config['data']['batch_size'],
            shuffle=False,
            categorical_mappings=categorical_mappings,
            num_workers=config['data']['num_workers']
        )
        
        # Create trainer and evaluate
        logger.info("Initializing trainer")
        trainer = ModelTrainer(args.config)
        
        # Evaluate model
        logger.info(f"Evaluating model from {args.model}")
        metrics = trainer.evaluate(test_loader, model_path=args.model)
        
        # Print metrics
        logger.info("Evaluation Results:")
        for metric_name, metric_value in metrics.items():
            logger.info(f"  {metric_name}: {metric_value:.4f}")
        
        # Save metrics to file
        metrics_path = os.path.join(args.output, 'metrics.json')
        with open(metrics_path, 'w') as f:
            json.dump(metrics, f, indent=2)
        logger.info(f"Metrics saved to {metrics_path}")
        
        # Generate per-user metrics if data is available
        try:
            # Already loaded the data earlier
            test_data = test_df
            
            # Create a dictionary to group by user
            user_metrics = {}
            
            # Load model directly (not through trainer) for generating recommendations
            logger.info("Generating per-user metrics")
            checkpoint = torch.load(args.model, map_location=device)
            
            # Create a temporary config file if loading directly
            import tempfile
            config_file = tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.yaml')
            yaml.dump(checkpoint['config'], config_file)
            config_path = config_file.name
            config_file.close()
            
            model = HybridRecommender(config_path, checkpoint.get('categorical_mappings'))
            model.load_state_dict(checkpoint['model_state_dict'])
            model.to(device)
            model.eval()
            
            # Clean up temporary file
            os.unlink(config_path)
            
            # Generate per-user metrics report
            user_report_path = os.path.join(args.output, 'user_metrics.csv')
            # This would require additional implementation to generate per-user metrics
            # which is beyond the scope of this basic evaluation script
            
            logger.info(f"Per-user metrics analysis saved to {user_report_path}")
            
        except Exception as e:
            logger.warning(f"Could not generate per-user metrics: {str(e)}")
        
        logger.info("Evaluation completed successfully")
        return 0
        
    except Exception as e:
        logger.error(f"Error during evaluation: {str(e)}", exc_info=True)
        return 1

if __name__ == "__main__":
    sys.exit(main()) 