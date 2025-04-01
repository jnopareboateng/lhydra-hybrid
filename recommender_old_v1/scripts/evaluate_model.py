#!/usr/bin/env python3
"""
Evaluation script for the Lhydra Hybrid Music Recommender System.
Evaluates a trained model on test data and generates evaluation reports.
"""

import os
import sys
import argparse
import yaml
import torch
import logging
import pandas as pd
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_curve, precision_recall_curve, auc

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import project modules
from data.preprocessor import MusicDataPreprocessor
from data.dataset import MusicDataLoader
from models.two_tower_model import TwoTowerHybridModel
from utils.metrics_utils import calculate_all_metrics
from utils.logger import LhydraLogger, log_function

@log_function()
def load_config(config_path):
    """
    Load configuration from YAML file.
    
    Args:
        config_path (str): Path to configuration file.
        
    Returns:
        dict: Configuration dictionary.
    """
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config

@log_function()
def load_model(model_path, device, logger, manifest_path=None):
    """
    Load a trained model from disk.
    
    Args:
        model_path (str): Path to model checkpoint.
        device (torch.device): Device to load model to.
        logger (LhydraLogger): Logger instance.
        manifest_path (str, optional): Path to feature manifest file.
        
    Returns:
        TwoTowerHybridModel: Loaded model.
    """
    logger.info(f"Loading model from {model_path}")
    
    # Check if manifest path was provided
    if manifest_path and os.path.exists(manifest_path):
        logger.info(f"Using feature manifest from {manifest_path}")
        model = TwoTowerHybridModel.load(model_path, device=device, manifest_path=manifest_path)
    else:
        # Try to look for manifest in the same directory as the model
        model_dir = os.path.dirname(model_path)
        possible_manifest = os.path.join(model_dir, "feature_manifest.yaml")
        if os.path.exists(possible_manifest):
            logger.info(f"Found feature manifest at {possible_manifest}")
            model = TwoTowerHybridModel.load(model_path, device=device, manifest_path=possible_manifest)
        else:
            logger.warning("No feature manifest found. Feature alignment will not be available.")
    model = TwoTowerHybridModel.load(model_path, device=device)
    
    # Set model to evaluation mode
    model.eval()
    
    # Log model parameters for debugging
    logger.info(f"Model loaded successfully")
    logger.info(f"Model architecture parameters:")
    logger.info(f"  - User input dimension: {model.user_input_dim}")
    logger.info(f"  - Item input dimension: {model.item_input_dim}")
    logger.info(f"  - Embedding dimension: {model.embedding_dim}")
    logger.info(f"  - Final layer size: {model.final_layer_size}")
    
    # Log if feature manifest is available
    if hasattr(model, 'feature_manifest') and model.feature_manifest is not None:
        logger.info("Feature manifest is loaded and available for feature alignment")
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(f"Feature manifest contents: {model.feature_manifest}")
    
    # Log model parameter count
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Model has {total_params:,} total parameters ({trainable_params:,} trainable)")
    
    return model

@log_function()
def load_data(data_path, logger):
    """
    Load test data for evaluation.
    
    Args:
        data_path (str): Path to test data file.
        logger (LhydraLogger): Logger instance.
        
    Returns:
        pd.DataFrame: Test data.
    """
    logger.info(f"Loading test data from {data_path}")
    
    df = pd.read_csv(data_path)
    
    logger.info(f"Loaded {len(df)} samples")
    
    return df

@log_function()
def create_dataloaders(test_df, config, batch_size, num_workers, logger):
    """
    Create DataLoader for test data.
    
    Args:
        test_df (pd.DataFrame): Test data.
        config (dict): Configuration dictionary.
        batch_size (int): Batch size.
        num_workers (int): Number of workers.
        logger (LhydraLogger): Logger instance.
        
    Returns:
        DataLoader: Test DataLoader.
    """
    logger.info("Creating test DataLoader")
    
    # Ensure user_id and track_id columns are present
    user_id_col = 'user_id'
    track_id_col = 'track_id'
    target_col = config['data']['target_column']
    
    # Check if target column exists, use fallback if not
    if target_col not in test_df.columns:
        if 'high_engagement' in test_df.columns:
            target_col = 'high_engagement'
            logger.warning(f"Target column '{config['data']['target_column']}' not found, using 'high_engagement' instead")
        elif 'engagement' in test_df.columns:
            target_col = 'engagement'
            logger.warning(f"Target column '{config['data']['target_column']}' not found, using 'engagement' instead")
        else:
            logger.error("No valid target column found in the test data!")
            raise ValueError("No valid target column found in the test data!")
    
    # Create data loader
    loader = MusicDataLoader(
        batch_size=batch_size,
        num_workers=num_workers,
        logger=logger
    )
    
    # Create test dataloader
    try:
    dataloaders = loader.create_dataloaders(
        test_df, test_df=test_df,  # Use same df for train and test
            user_id_col=user_id_col,
            track_id_col=track_id_col,
            target_col=target_col
        )
        
        # Validate dataloader to ensure correct batch format
        if 'test' in dataloaders:
            test_loader = dataloaders['test']
            logger.info(f"DataLoader created with {len(test_loader)} batches")
            
            # Inspect first batch to validate format (if possible)
            try:
                batch_iter = iter(test_loader)
                first_batch = next(batch_iter)
                logger.info(f"Batch keys: {list(first_batch.keys())}")
                
                # Check user_features and item_features structure
                if 'user_features' in first_batch and 'item_features' in first_batch:
                    if isinstance(first_batch['user_features'], dict):
                        logger.info(f"User features keys: {list(first_batch['user_features'].keys())}")
                    else:
                        logger.info(f"User features is a tensor with shape: {first_batch['user_features'].shape}")
                    
                    if isinstance(first_batch['item_features'], dict):
                        logger.info(f"Item features keys: {list(first_batch['item_features'].keys())}")
                    else:
                        logger.info(f"Item features is a tensor with shape: {first_batch['item_features'].shape}")
                
            except StopIteration:
                logger.warning("DataLoader is empty, could not inspect batch format")
            except Exception as e:
                logger.warning(f"Error while inspecting first batch: {str(e)}")
            
            return test_loader
        else:
            logger.error("No test loader found in created dataloaders!")
            raise ValueError("No test loader found in created dataloaders!")
    except Exception as e:
        logger.error(f"Error creating dataloaders: {str(e)}", exc_info=True)
        raise

@log_function()
def prepare_features_for_model(model, user_features, item_features, logger, debug=False):
    """
    Prepare and validate feature tensors for model compatibility.
    
    This function ensures that the feature tensors match the dimensions
    expected by the model's towers by handling different feature formats
    and potentially resizing or padding features.
    
    Args:
        model (TwoTowerHybridModel): The model to use for inference
        user_features (dict or torch.Tensor): User features
        item_features (dict or torch.Tensor): Item features
        logger (LhydraLogger): Logger instance
        debug (bool): Whether to log debug information
        
    Returns:
        tuple: (processed_user_features, processed_item_features)
    """
    # Get expected dimensions from model
    user_input_dim = model.user_input_dim
    item_input_dim = model.item_input_dim
    
    # Process user features
    if isinstance(user_features, dict):
        # Log feature keys and dimensions for debugging
        if debug:
            logger.debug("User features dict keys and dimensions:")
            for k, v in user_features.items():
                if isinstance(v, torch.Tensor):
                    logger.debug(f"  - {k}: {v.shape}")
        
        # Extract data from dict and concatenate
        user_tensors = []
        for k, v in user_features.items():
            if k == 'user_id':  # Skip ID fields
                continue
            if isinstance(v, torch.Tensor):
                if v.dim() == 1:
                    v = v.unsqueeze(1)  # Convert to 2D tensor
                user_tensors.append(v)
        
        # Concatenate along feature dimension
        if user_tensors:
            concatenated_user = torch.cat(user_tensors, dim=1)
            if debug:
                logger.debug(f"Concatenated user features shape: {concatenated_user.shape}")
            
            # Check against expected dimension
            if concatenated_user.shape[1] != user_input_dim:
                # Calculate difference percentage
                diff_percentage = abs(concatenated_user.shape[1] - user_input_dim) / user_input_dim * 100
                severity = "significant" if diff_percentage > 30 else "minor"
                logger.warning(f"User feature dimension mismatch: got {concatenated_user.shape[1]}, expected {user_input_dim} ({severity} difference: {diff_percentage:.1f}%)")
                
                # Handle dimension mismatch
                if concatenated_user.shape[1] < user_input_dim:
                    # Pad with zeros if too small
                    padding = torch.zeros(concatenated_user.shape[0], user_input_dim - concatenated_user.shape[1], 
                                        device=concatenated_user.device)
                    concatenated_user = torch.cat([concatenated_user, padding], dim=1)
                    logger.info(f"Padded user features to match expected dimension: {concatenated_user.shape}")
                else:
                    # Truncate if too large
                    if debug:
                        # Log which features would be truncated
                        total_dims = 0
                        for i, tensor in enumerate(user_tensors):
                            feature_size = tensor.shape[1]
                            if total_dims + feature_size > user_input_dim:
                                logger.debug(f"Feature truncation would occur at tensor {i} with shape {tensor.shape}")
                                break
                            total_dims += feature_size
                    
                    concatenated_user = concatenated_user[:, :user_input_dim]
                    logger.info(f"Truncated user features to match expected dimension: {concatenated_user.shape}")
            
            processed_user_features = concatenated_user
        else:
            # Create dummy features if no valid tensors
            logger.warning("No valid user feature tensors found, using zeros")
            batch_size = next(iter(user_features.values())).shape[0] if user_features else 1
            processed_user_features = torch.zeros(batch_size, user_input_dim, device=next(iter(user_features.values())).device if user_features else 'cpu')
    else:
        # Handle case where user_features is already a tensor
        processed_user_features = user_features
        if processed_user_features.shape[1] != user_input_dim:
            # Calculate difference percentage
            diff_percentage = abs(processed_user_features.shape[1] - user_input_dim) / user_input_dim * 100
            severity = "significant" if diff_percentage > 30 else "minor"
            logger.warning(f"User feature dimension mismatch: got {processed_user_features.shape[1]}, expected {user_input_dim} ({severity} difference: {diff_percentage:.1f}%)")
            
            # Handle dimension mismatch
            if processed_user_features.shape[1] < user_input_dim:
                # Pad with zeros if too small
                padding = torch.zeros(processed_user_features.shape[0], user_input_dim - processed_user_features.shape[1], 
                                    device=processed_user_features.device)
                processed_user_features = torch.cat([processed_user_features, padding], dim=1)
                logger.info(f"Padded user features to match expected dimension: {processed_user_features.shape}")
            else:
                # Truncate if too large
                processed_user_features = processed_user_features[:, :user_input_dim]
                logger.info(f"Truncated user features to match expected dimension: {processed_user_features.shape}")
    
    # Process item features (similar approach)
    if isinstance(item_features, dict):
        # Log feature keys and dimensions for debugging
        if debug:
            logger.debug("Item features dict keys and dimensions:")
            for k, v in item_features.items():
                if isinstance(v, torch.Tensor):
                    logger.debug(f"  - {k}: {v.shape}")
        
        # Extract data from dict and concatenate
        item_tensors = []
        for k, v in item_features.items():
            if k == 'track_id' or k == 'item_id':  # Skip ID fields
                continue
            if isinstance(v, torch.Tensor):
                if v.dim() == 1:
                    v = v.unsqueeze(1)  # Convert to 2D tensor
                item_tensors.append(v)
        
        # Concatenate along feature dimension
        if item_tensors:
            concatenated_item = torch.cat(item_tensors, dim=1)
            if debug:
                logger.debug(f"Concatenated item features shape: {concatenated_item.shape}")
            
            # Check against expected dimension
            if concatenated_item.shape[1] != item_input_dim:
                # Calculate difference percentage
                diff_percentage = abs(concatenated_item.shape[1] - item_input_dim) / item_input_dim * 100
                severity = "significant" if diff_percentage > 30 else "minor"
                logger.warning(f"Item feature dimension mismatch: got {concatenated_item.shape[1]}, expected {item_input_dim} ({severity} difference: {diff_percentage:.1f}%)")
                
                # Handle dimension mismatch
                if concatenated_item.shape[1] < item_input_dim:
                    # Pad with zeros if too small
                    padding = torch.zeros(concatenated_item.shape[0], item_input_dim - concatenated_item.shape[1], 
                                        device=concatenated_item.device)
                    concatenated_item = torch.cat([concatenated_item, padding], dim=1)
                    logger.info(f"Padded item features to match expected dimension: {concatenated_item.shape}")
                else:
                    # Truncate if too large
                    if debug:
                        # Log which features would be truncated
                        total_dims = 0
                        for i, tensor in enumerate(item_tensors):
                            feature_size = tensor.shape[1]
                            if total_dims + feature_size > item_input_dim:
                                logger.debug(f"Feature truncation would occur at tensor {i} with shape {tensor.shape}")
                                break
                            total_dims += feature_size
                    
                    concatenated_item = concatenated_item[:, :item_input_dim]
                    logger.info(f"Truncated item features to match expected dimension: {concatenated_item.shape}")
            
            processed_item_features = concatenated_item
        else:
            # Create dummy features if no valid tensors
            logger.warning("No valid item feature tensors found, using zeros")
            batch_size = next(iter(item_features.values())).shape[0] if item_features else 1
            processed_item_features = torch.zeros(batch_size, item_input_dim, device=next(iter(item_features.values())).device if item_features else 'cpu')
    else:
        # Handle case where item_features is already a tensor
        processed_item_features = item_features
        if processed_item_features.shape[1] != item_input_dim:
            # Calculate difference percentage
            diff_percentage = abs(processed_item_features.shape[1] - item_input_dim) / item_input_dim * 100
            severity = "significant" if diff_percentage > 30 else "minor"
            logger.warning(f"Item feature dimension mismatch: got {processed_item_features.shape[1]}, expected {item_input_dim} ({severity} difference: {diff_percentage:.1f}%)")
            
            # Handle dimension mismatch
            if processed_item_features.shape[1] < item_input_dim:
                # Pad with zeros if too small
                padding = torch.zeros(processed_item_features.shape[0], item_input_dim - processed_item_features.shape[1], 
                                    device=processed_item_features.device)
                processed_item_features = torch.cat([processed_item_features, padding], dim=1)
                logger.info(f"Padded item features to match expected dimension: {processed_item_features.shape}")
            else:
                # Truncate if too large
                processed_item_features = processed_item_features[:, :item_input_dim]
                logger.info(f"Truncated item features to match expected dimension: {processed_item_features.shape}")
    
    return processed_user_features, processed_item_features

@log_function()
def evaluate_model(model, test_loader, output_dir, logger, debug=False):
    """
    Evaluate model and generate evaluation reports.
    
    Args:
        model (TwoTowerHybridModel): Trained model.
        test_loader (DataLoader): Test DataLoader.
        output_dir (str): Directory to save evaluation results.
        logger (LhydraLogger): Logger instance.
        debug (bool): Whether to enable debug output.
        
    Returns:
        dict: Evaluation results.
    """
    logger.info("Evaluating model")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Set model to evaluation mode
    model.eval()
    device = next(model.parameters()).device
    
    # Collect predictions and targets
    all_targets = []
    all_predictions = []
    all_probabilities = []
    all_user_ids = []
    
    # Check if feature manifest is available
    has_feature_manifest = hasattr(model, 'feature_manifest') and model.feature_manifest is not None
    if has_feature_manifest:
        logger.info("Using feature alignment based on the model's feature manifest")
    else:
        logger.warning("No feature manifest available. Using legacy feature preparation.")
    
    with torch.inference_mode():
        for batch_idx, batch in enumerate(test_loader):
            # Validate batch structure
            if 'user_features' not in batch:
                logger.error(f"Batch {batch_idx} is missing 'user_features'")
                if debug:
                    logger.error(f"Batch keys: {list(batch.keys())}")
                continue
                
            if 'item_features' not in batch:
                logger.error(f"Batch {batch_idx} is missing 'item_features'")
                if debug:
                    logger.error(f"Batch keys: {list(batch.keys())}")
                continue
                
            if 'target' not in batch:
                logger.error(f"Batch {batch_idx} is missing 'target'")
                if debug:
                    logger.error(f"Batch keys: {list(batch.keys())}")
                continue
            
            # Extract user_id from user_features if available
            user_ids = None
            if 'user_id' in batch:
                user_ids = batch['user_id']
                if isinstance(user_ids, torch.Tensor):
                    user_ids = user_ids.cpu().numpy()
            elif 'user_features' in batch and isinstance(batch['user_features'], dict) and 'user_id' in batch['user_features']:
                user_ids = batch['user_features']['user_id']
                if isinstance(user_ids, torch.Tensor):
                    user_ids = user_ids.cpu().numpy()
            
            # Try to handle other possible user_id locations
            if user_ids is None and hasattr(batch, 'get') and callable(getattr(batch, 'get')):
                for key in ['user', 'userID', 'userId']:
                    if key in batch:
                        user_ids = batch[key]
                        if isinstance(user_ids, torch.Tensor):
                            user_ids = user_ids.cpu().numpy()
                        break
            
            # Move data to device - handle either dictionary of tensors or tensors directly
            try:
                if isinstance(batch['user_features'], dict):
                    logger.debug(f"User features is a dictionary with keys: {list(batch['user_features'].keys())}")
                    user_features = {k: v.to(device) for k, v in batch['user_features'].items()}
                else:
                    logger.debug(f"User features is a tensor with shape: {batch['user_features'].shape}")
                    user_features = batch['user_features'].to(device)
                    
                if isinstance(batch['item_features'], dict):
                    logger.debug(f"Item features is a dictionary with keys: {list(batch['item_features'].keys())}")
                    item_features = {k: v.to(device) for k, v in batch['item_features'].items()}
                else:
                    logger.debug(f"Item features is a tensor with shape: {batch['item_features'].shape}")
                    item_features = batch['item_features'].to(device)
            except Exception as e:
                logger.error(f"Error processing features for batch {batch_idx}: {str(e)}")
                if debug:
                    logger.error(f"Batch contents: {batch}")
                continue
            
            # Ensure target is a tensor
            if not isinstance(batch['target'], torch.Tensor):
                logger.warning(f"Target is not a tensor, converting: {type(batch['target'])}")
                batch['target'] = torch.tensor(batch['target'], dtype=torch.float32)
            
            targets = batch['target'].to(device)
            
            # Forward pass
            try:
                if has_feature_manifest:
                    # The model's forward method will use align_features internally
                    outputs = model(user_features, item_features)
                else:
                    # Use legacy feature preparation
                    processed_user_features, processed_item_features = prepare_features_for_model(model, user_features, item_features, logger, debug)
                    outputs = model(processed_user_features, processed_item_features)
                
                # Handle different output shapes
                if len(outputs.shape) > 1 and outputs.shape[1] > 1:
                    logger.debug(f"Model output has shape {outputs.shape}, taking first column")
                    outputs = outputs[:, 0]  # Take first column if multiple outputs
            
            # Convert to binary predictions
                predictions = (outputs > 0.5).float()
            except Exception as e:
                logger.error(f"Error during model forward pass: {str(e)}")
                if debug:
                    logger.error(f"User features type: {type(user_features)}")
                    logger.error(f"Item features type: {type(item_features)}")
                    if isinstance(user_features, dict):
                        for k, v in user_features.items():
                            logger.error(f"  user_features[{k}] shape: {v.shape}")
                    if isinstance(item_features, dict):
                        for k, v in item_features.items():
                            logger.error(f"  item_features[{k}] shape: {v.shape}")
                    
                    # Try to get more information about the model
                    try:
                        logger.error(f"Model expected dimensions: user={model.user_input_dim}, item={model.item_input_dim}")
                    except:
                        pass
                continue
            
            # Move to CPU and convert to numpy
            targets_np = targets.cpu().numpy()
            predictions_np = predictions.cpu().numpy()
            probabilities_np = outputs.cpu().numpy()
            
            # Append to lists
            all_targets.append(targets_np)
            all_predictions.append(predictions_np)
            all_probabilities.append(probabilities_np)
            
            # Collect user IDs if available
            if user_ids is not None:
                all_user_ids.append(user_ids)
    
    # Concatenate all batches
    if not all_targets:
        logger.error("No successful predictions were made, all batches failed")
        # Return empty metrics
        return {
            'accuracy': 0.0,
            'precision': 0.0,
            'recall': 0.0,
            'f1': 0.0,
            'auc': 0.0,
            'error': 'No successful predictions'
        }
        
    all_targets = np.concatenate(all_targets)
    all_predictions = np.concatenate(all_predictions)
    all_probabilities = np.concatenate(all_probabilities)
    
    if all_user_ids is not None and len(all_user_ids) > 0:
        all_user_ids = np.concatenate(all_user_ids)
    else:
        all_user_ids = None
        logger.warning("No user IDs were collected during evaluation")
    
    # Calculate all metrics
    try:
    metrics = calculate_all_metrics(
        all_targets, 
        all_predictions, 
        all_probabilities,
            user_ids=all_user_ids,
        k_values=[5, 10, 20]
    )
    except Exception as e:
        logger.error(f"Error calculating metrics: {str(e)}")
        # Create basic metrics without ranking metrics 
        from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
        metrics = {
            'accuracy': accuracy_score(all_targets, all_predictions > 0.5),
            'precision': precision_score(all_targets, all_predictions > 0.5, zero_division=0),
            'recall': recall_score(all_targets, all_predictions > 0.5, zero_division=0),
            'f1': f1_score(all_targets, all_predictions > 0.5, zero_division=0),
        }
        try:
            metrics['auc'] = roc_auc_score(all_targets, all_probabilities)
        except:
            metrics['auc'] = 0.0
        
        logger.info("Successfully calculated basic metrics without ranking metrics")
    
    # Save evaluation results
    results_file = os.path.join(output_dir, "metrics.yaml")
    with open(results_file, 'w') as f:
        yaml.dump(metrics, f)
    
    # Save predictions
    predictions_df = pd.DataFrame({
        'target': all_targets,
        'prediction': all_predictions,
        'probability': all_probabilities
    })
    
    if len(all_user_ids) > 0:
        predictions_df['user_id'] = all_user_ids
    
    predictions_file = os.path.join(output_dir, "predictions.csv")
    predictions_df.to_csv(predictions_file, index=False)
    
    # Generate evaluation plots
    generate_evaluation_plots(
        all_targets, all_predictions, all_probabilities, output_dir, logger
    )
    
    logger.info(f"Evaluation results saved to {output_dir}")
    
    return metrics

@log_function()
def generate_evaluation_plots(targets, predictions, probabilities, output_dir, logger):
    """
    Generate evaluation plots.
    
    Args:
        targets (np.ndarray): Ground truth labels.
        predictions (np.ndarray): Predicted binary labels.
        probabilities (np.ndarray): Predicted probabilities.
        output_dir (str): Directory to save plots.
        logger (LhydraLogger): Logger instance.
    """
    logger.info("Generating evaluation plots")
    
    # Plot confusion matrix
    cm = np.array([
        [(targets == 0) & (predictions == 0), (targets == 0) & (predictions == 1)],
        [(targets == 1) & (predictions == 0), (targets == 1) & (predictions == 1)]
    ]).sum(axis=2)
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
               xticklabels=['Low Engagement', 'High Engagement'],
               yticklabels=['Low Engagement', 'High Engagement'])
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "confusion_matrix.png"))
    plt.close()
    
    # Plot ROC curve
    fpr, tpr, _ = roc_curve(targets, probabilities)
    roc_auc = auc(fpr, tpr)
    
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.3f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc="lower right")
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, "roc_curve.png"))
    plt.close()
    
    # Plot Precision-Recall curve
    precision, recall, _ = precision_recall_curve(targets, probabilities)
    pr_auc = auc(recall, precision)
    
    plt.figure(figsize=(8, 6))
    plt.plot(recall, precision, color='darkred', lw=2, label=f'PR curve (AUC = {pr_auc:.3f})')
    plt.fill_between(recall, precision, alpha=0.2, color='lightcoral')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.legend(loc="upper right")
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, "precision_recall_curve.png"))
    plt.close()
    
    # Plot probability distribution
    plt.figure(figsize=(10, 6))
    plt.hist(probabilities[targets == 0], bins=20, alpha=0.5, label='Low Engagement', color='blue')
    plt.hist(probabilities[targets == 1], bins=20, alpha=0.5, label='High Engagement', color='red')
    plt.xlabel('Predicted Probability')
    plt.ylabel('Count')
    plt.title('Probability Distribution')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, "probability_distribution.png"))
    plt.close()
    
    logger.info(f"Evaluation plots saved to {output_dir}")

@log_function()
def analyze_data_model_compatibility(test_df, model, logger):
    """
    Analyze the compatibility between the test data and the model.
    
    This function helps diagnose dimension mismatches between the data 
    and what the model expects.
    
    Args:
        test_df (pd.DataFrame): Test dataframe
        model (TwoTowerHybridModel): The model
        logger (LhydraLogger): Logger instance
    
    Returns:
        dict: Analysis results
    """
    logger.info("Analyzing data-model compatibility")
    
    analysis = {
        "user_features": {
            "expected_dim": model.user_input_dim,
            "actual_features": [],
            "potential_issues": []
        },
        "item_features": {
            "expected_dim": model.item_input_dim,
            "actual_features": [],
            "potential_issues": []
        }
    }
    
    # Analyze user features
    user_feature_count = 0
    user_columns = []
    
    # Group columns by type
    demographic_cols = [col for col in test_df.columns if col.startswith(('gender_', 'age_', 'region_', 'country_'))]
    user_columns.extend(demographic_cols)
    user_feature_count += len(demographic_cols)
    
    listening_cols = [col for col in test_df.columns if col.startswith(('monthly_hours', 'genre_diversity', 'top_genre_'))]
    user_columns.extend(listening_cols)
    user_feature_count += len(listening_cols)
    
    audio_profile_cols = [col for col in test_df.columns if col.startswith('avg_')]
    user_columns.extend(audio_profile_cols)
    user_feature_count += len(audio_profile_cols)
    
    # Add any other engineered user features
    other_user_cols = [col for col in test_df.columns 
                     if any(x in col for x in ['listening_depth', 'age_group_'])]
    user_columns.extend(other_user_cols)
    user_feature_count += len(other_user_cols)
    
    # Analyze item features
    item_feature_count = 0
    item_columns = []
    
    # Audio features
    audio_cols = [col for col in test_df.columns 
                  if col in ['danceability', 'energy', 'key', 'loudness', 'mode',
                            'speechiness', 'acousticness', 'instrumentalness',
                            'liveness', 'valence', 'tempo', 'time_signature']]
    item_columns.extend(audio_cols)
    item_feature_count += len(audio_cols)
    
    # Genre features
    genre_cols = [col for col in test_df.columns if col.startswith('main_genre_')]
    item_columns.extend(genre_cols)
    item_feature_count += len(genre_cols)
    
    # Temporal features
    temporal_cols = [col for col in test_df.columns 
                    if col in ['year', 'song_age', 'is_recent']]
    item_columns.extend(temporal_cols)
    item_feature_count += len(temporal_cols)
    
    # Add any other engineered item features
    other_item_cols = [col for col in test_df.columns 
                      if any(x in col for x in ['duration_', 'mood_category'])]
    item_columns.extend(other_item_cols)
    item_feature_count += len(other_item_cols)
    
    # Store feature information
    analysis["user_features"]["actual_features"] = user_columns
    analysis["user_features"]["feature_count"] = user_feature_count
    analysis["item_features"]["actual_features"] = item_columns
    analysis["item_features"]["feature_count"] = item_feature_count
    
    # Compare with model expectations and identify potential issues
    if user_feature_count != model.user_input_dim:
        diff = abs(user_feature_count - model.user_input_dim)
        severity = "significant" if diff > model.user_input_dim * 0.3 else "minor"
        analysis["user_features"]["potential_issues"].append(
            f"Dimension mismatch: data has {user_feature_count} features, model expects {model.user_input_dim} " +
            f"({severity} difference of {diff} dimensions)"
        )
    
    if item_feature_count != model.item_input_dim:
        diff = abs(item_feature_count - model.item_input_dim)
        severity = "significant" if diff > model.item_input_dim * 0.3 else "minor"
        analysis["item_features"]["potential_issues"].append(
            f"Dimension mismatch: data has {item_feature_count} features, model expects {model.item_input_dim} " +
            f"({severity} difference of {diff} dimensions)"
        )
    
    # Log analysis results
    logger.info("Data-Model Compatibility Analysis:")
    logger.info(f"User features: {user_feature_count} vs model expected {model.user_input_dim}")
    logger.info(f"Item features: {item_feature_count} vs model expected {model.item_input_dim}")
    
    # Log potential issues
    for issue in analysis["user_features"]["potential_issues"]:
        logger.warning(f"User features issue: {issue}")
    for issue in analysis["item_features"]["potential_issues"]:
        logger.warning(f"Item features issue: {issue}")
    
    return analysis

def main():
    """Main function to run the evaluation pipeline."""
    parser = argparse.ArgumentParser(
        description="Evaluate the Lhydra Hybrid Music Recommender System"
    )
    parser.add_argument("--model", type=str, required=True,
                      help="Path to trained model checkpoint")
    parser.add_argument("--data", type=str, required=True,
                      help="Path to test data file")
    parser.add_argument("--config", type=str, default="training/configs/training_config.yaml",
                      help="Path to configuration file")
    parser.add_argument("--output", type=str, default="evaluation/results",
                      help="Directory to save evaluation results")
    parser.add_argument("--batch-size", type=int, default=64,
                      help="Batch size for evaluation")
    parser.add_argument("--num-workers", type=int, default=4,
                      help="Number of workers for data loading")
    parser.add_argument("--log-level", type=str, default="INFO",
                      choices=["DEBUG", "INFO", "WARNING", "ERROR"],
                      help="Logging level")
    parser.add_argument("--debug", action="store_true",
                      help="Enable debug mode with more detailed output")
    parser.add_argument("--manifest", type=str, default=None,
                      help="Path to feature manifest file")
    
    args = parser.parse_args()
    
    # Create timestamp for this run
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Create logger
    log_level = getattr(logging, args.log_level)
    logger = LhydraLogger(log_dir=f"logs/eval_{timestamp}", log_level=log_level)
    logger.info(f"Starting evaluation run at {timestamp}")
    
        # Set output directory with timestamp
        output_dir = os.path.join(args.output, timestamp)
        os.makedirs(output_dir, exist_ok=True)
    
    try:
        # Verify input files exist
        if not os.path.exists(args.model):
            logger.error(f"Model file not found: {args.model}")
            sys.exit(1)
            
        if not os.path.exists(args.data):
            logger.error(f"Data file not found: {args.data}")
            sys.exit(1)
            
        if not os.path.exists(args.config):
            logger.error(f"Config file not found: {args.config}")
            sys.exit(1)
        
        # Set device
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Using device: {device}")
        
        # Load configuration
        try:
        config = load_config(args.config)
        logger.info(f"Loaded configuration from {args.config}")
        except Exception as e:
            logger.error(f"Error loading configuration: {str(e)}", exc_info=args.debug)
            sys.exit(1)
        
        # Load model with manifest if specified
        try:
            model = load_model(args.model, device, logger, manifest_path=args.manifest)
        except Exception as e:
            logger.error(f"Error loading model: {str(e)}", exc_info=args.debug)
            sys.exit(1)
        
        # Load test data
        try:
        test_df = load_data(args.data, logger)
            
            # Log test data columns for debugging
            logger.info(f"Test data columns: {test_df.columns.tolist()}")
            logger.info(f"Test data shape: {test_df.shape}")
            
            # Debug: Check for key columns
            key_columns = ['user_id', 'track_id', 'high_engagement', 'engagement']
            for col in key_columns:
                if col in test_df.columns:
                    logger.info(f"Column '{col}' found in test data")
                    # For categorical columns, log unique values count
                    if test_df[col].dtype == 'object' or test_df[col].dtype == 'int64':
                        logger.info(f"  - Unique values: {test_df[col].nunique()}")
                else:
                    logger.warning(f"Column '{col}' not found in test data")
            
            # Analyze compatibility between data and model
            compatibility_analysis = analyze_data_model_compatibility(test_df, model, logger)
            
        except Exception as e:
            logger.error(f"Error loading test data: {str(e)}", exc_info=args.debug)
            sys.exit(1)
        
        # Create dataloader
        try:
        test_loader = create_dataloaders(
            test_df, config, args.batch_size, args.num_workers, logger
        )
        except Exception as e:
            logger.error(f"Error creating data loader: {str(e)}", exc_info=args.debug)
            sys.exit(1)
        
        # Evaluate model
        try:
            metrics = evaluate_model(model, test_loader, output_dir, logger, args.debug)
        
        # Print summary
        logger.info("Evaluation completed successfully")
            logger.info(f"Accuracy: {metrics.get('accuracy', 'N/A'):.4f}" if 'accuracy' in metrics else "Accuracy: N/A")
            logger.info(f"Precision: {metrics.get('precision', 'N/A'):.4f}" if 'precision' in metrics else "Precision: N/A")
            logger.info(f"Recall: {metrics.get('recall', 'N/A'):.4f}" if 'recall' in metrics else "Recall: N/A")
            logger.info(f"F1 Score: {metrics.get('f1', 'N/A'):.4f}" if 'f1' in metrics else "F1 Score: N/A")
            logger.info(f"ROC AUC: {metrics.get('auc', 'N/A'):.4f}" if 'auc' in metrics else "ROC AUC: N/A")
        logger.info(f"Results saved to {output_dir}")
        except Exception as e:
            logger.error(f"Error during model evaluation: {str(e)}", exc_info=args.debug)
            
            # Try to capture and log more diagnostic information
            if args.debug:
                logger.error("DEBUG INFORMATION:")
                logger.error(f"Model type: {type(model)}")
                logger.error(f"DataLoader type: {type(test_loader)}")
                
                # Try to extract a batch for inspection
                try:
                    batch_iter = iter(test_loader)
                    first_batch = next(batch_iter)
                    logger.error(f"Batch keys: {list(first_batch.keys())}")
                    
                    if 'user_features' in first_batch:
                        logger.error(f"user_features type: {type(first_batch['user_features'])}")
                    if 'item_features' in first_batch:
                        logger.error(f"item_features type: {type(first_batch['item_features'])}")
                except Exception as batch_error:
                    logger.error(f"Could not inspect batch: {str(batch_error)}")
            
            sys.exit(1)
        
    except Exception as e:
        logger.error(f"Unexpected error in evaluation pipeline: {str(e)}", exc_info=args.debug)
        sys.exit(1)

if __name__ == "__main__":
    main() 