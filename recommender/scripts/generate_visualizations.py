#!/usr/bin/env python3
"""
Script to generate visualizations for model performance analysis.

This script loads a trained model and test data, runs predictions,
and generates various visualizations to analyze model performance.
"""

import os
import sys
import argparse
import logging
import numpy as np
import torch
import yaml
import json
from pathlib import Path

# Add parent directory to path to allow imports from recommender package
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from recommender.models.recommender import HybridRecommender
from recommender.training.dataset import MusicRecommenderDataset
from recommender.visualization.performance_plots import (
    generate_performance_report,
    plot_learning_curves,
    plot_confusion_matrix,
    plot_roc_curve,
    plot_precision_recall_curve,
    plot_prediction_distribution,
    plot_metrics_radar,
    plot_embedding_visualization,
    plot_feature_importance
)

# Configure logging
log_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'logs')
os.makedirs(log_dir, exist_ok=True)
log_file = os.path.join(log_dir, 'visualizations.log')

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(log_file, mode='w')
    ]
)
logger = logging.getLogger('visualize')


def load_model(model_path, device):
    """
    Load a trained model from checkpoint.
    
    Args:
        model_path: Path to the model checkpoint
        device: Device to load the model on
        
    Returns:
        Loaded model
    """
    try:
        checkpoint = torch.load(model_path, map_location=device)
        
        if 'config' in checkpoint:
            config = checkpoint['config']
            categorical_mappings = checkpoint.get('categorical_mappings', {})
            
            model = HybridRecommender(config, categorical_mappings)
            
            # Load state dict
            state_dict_key = 'state_dict' if 'state_dict' in checkpoint else 'model_state_dict'
            model.load_state_dict(checkpoint[state_dict_key])
            model.to(device)
            logger.info(f"Successfully loaded model from {model_path}")
            return model, config, categorical_mappings
        else:
            logger.error(f"Checkpoint at {model_path} does not contain model config")
            return None, None, None
    except Exception as e:
        logger.error(f"Failed to load model from {model_path}: {str(e)}")
        return None, None, None


def load_metrics_history(metrics_path):
    """
    Load metrics history from JSON file.
    
    Args:
        metrics_path: Path to the metrics JSON file
        
    Returns:
        Metrics history as a dictionary
    """
    try:
        with open(metrics_path, 'r') as f:
            metrics = json.load(f)
        logger.info(f"Successfully loaded metrics from {metrics_path}")
        return metrics
    except Exception as e:
        logger.error(f"Failed to load metrics from {metrics_path}: {str(e)}")
        return None


def create_dataloader(data_path, config, categorical_mappings, batch_size=64):
    """
    Create a data loader for the dataset.
    
    Args:
        data_path: Path to the dataset
        config: Model configuration
        categorical_mappings: Categorical feature mappings
        batch_size: Batch size for the data loader
        
    Returns:
        DataLoader object
    """
    try:
        user_features = config['feature_sets']['user_features']
        track_features = config['feature_sets']['track_features']
        categorical_columns = config['feature_sets'].get('categorical_features', [])
        target_col = config['feature_sets'].get('target_column', 'liked')
        
        dataset = MusicRecommenderDataset(
            data_path=data_path,
            user_features=user_features,
            track_features=track_features,
            target_col=target_col,
            categorical_columns=categorical_columns,
            categorical_mappings=categorical_mappings
        )
        
        dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=0
        )
        
        logger.info(f"Created data loader with {len(dataset)} samples")
        return dataloader
    except Exception as e:
        logger.error(f"Failed to create data loader: {str(e)}")
        return None


def run_predictions(model, dataloader, device):
    """
    Run predictions on the dataloader.
    
    Args:
        model: Trained model
        dataloader: DataLoader containing test data
        device: Device to run inference on
        
    Returns:
        Tuple of (targets, predictions, embeddings)
    """
    model.eval()
    all_targets = []
    all_predictions = []
    all_embeddings = []
    
    with torch.no_grad():
        for user_data, track_data, targets in dataloader:
            # Move data to device
            user_data = {k: v.to(device) for k, v in user_data.items()}
            track_data = {k: v.to(device) for k, v in track_data.items()}
            targets = targets.to(device)
            
            # Get predictions
            outputs = model(user_data, track_data)
            predictions = torch.sigmoid(outputs) if outputs.dim() > 1 else torch.sigmoid(outputs.unsqueeze(1))
            
            # Get embeddings for visualization
            user_embeddings, item_embeddings = model.get_embeddings(user_data, track_data)
            combined_embeddings = torch.cat([user_embeddings, item_embeddings], dim=1)
            
            # Store results
            all_targets.extend(targets.squeeze().cpu().numpy())
            all_predictions.extend(predictions.squeeze().cpu().numpy())
            all_embeddings.extend(combined_embeddings.cpu().numpy())
    
    logger.info(f"Made predictions for {len(all_targets)} samples")
    return np.array(all_targets), np.array(all_predictions), np.array(all_embeddings)


def generate_visualizations(args):
    """
    Generate visualizations for the trained model.
    
    Args:
        args: Command line arguments
    """
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() and not args.no_cuda else 'cpu')
    logger.info(f"Using device: {device}")
    
    # Load model
    model, config, categorical_mappings = load_model(args.model_path, device)
    if model is None:
        return
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Create data loader
    dataloader = create_dataloader(args.data_path, config, categorical_mappings, args.batch_size)
    if dataloader is None:
        return
    
    # Run predictions
    targets, predictions, embeddings = run_predictions(model, dataloader, device)
    
    # Load metrics history if available
    train_metrics = {}
    val_metrics = {}
    if args.metrics_path:
        metrics = load_metrics_history(args.metrics_path)
        if metrics and 'train' in metrics and 'val' in metrics:
            train_metrics = metrics['train']
            val_metrics = metrics['val']
    
    # Get feature names and importance
    feature_names, feature_importance = model.get_feature_importance()
    
    # Generate comprehensive performance report
    model_name = os.path.basename(args.model_path).split('.')[0]
    
    logger.info("Generating performance report...")
    plot_paths = generate_performance_report(
        targets,
        predictions,
        train_metrics,
        val_metrics,
        model_name,
        args.output_dir,
        embeddings=embeddings,
        feature_names=feature_names,
        feature_importance=feature_importance
    )
    
    logger.info(f"Visualization generation complete. Output directory: {args.output_dir}/{model_name}_report")


def parse_args():
    """
    Parse command line arguments.
    
    Returns:
        Parsed arguments
    """
    parser = argparse.ArgumentParser(description='Generate visualizations for model performance analysis')
    
    parser.add_argument('--model-path', type=str, required=True,
                        help='Path to the trained model checkpoint')
    parser.add_argument('--data-path', type=str, required=True,
                        help='Path to the test data CSV file')
    parser.add_argument('--metrics-path', type=str, default=None,
                        help='Path to the metrics history JSON file (optional)')
    parser.add_argument('--output-dir', type=str, default='recommender/visualizations',
                        help='Directory to save visualizations')
    parser.add_argument('--batch-size', type=int, default=64,
                        help='Batch size for inference')
    parser.add_argument('--no-cuda', action='store_true',
                        help='Disable CUDA even if available')
    
    return parser.parse_args()


def main():
    """Main entry point."""
    args = parse_args()
    generate_visualizations(args)


if __name__ == '__main__':
    main() 