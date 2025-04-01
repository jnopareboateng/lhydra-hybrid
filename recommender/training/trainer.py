"""
Trainer for the music recommender model.

This module contains the implementation of the model trainer which handles
the training, validation, and evaluation of the recommender model.
"""

import os
import logging
import yaml
import time
import json
import torch
import numpy as np
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import random
import matplotlib.pyplot as plt
from pathlib import Path

from models.recommender import HybridRecommender
from training.metrics import calculate_metrics
from visualization.performance_plots import (
    plot_learning_curves, plot_confusion_matrix, plot_roc_curve,
    plot_precision_recall_curve, plot_prediction_distribution,
    plot_metrics_radar, plot_embedding_visualization,
    plot_feature_importance, generate_performance_report
)

logger = logging.getLogger(__name__)


class ModelTrainer:
    """
    Trainer for the hybrid recommender model.
    
    This class encapsulates the logic for training, validating, and evaluating
    the hybrid recommender model, including handling of checkpoints and early stopping.
    """
    
    def __init__(self, config_path):
        """
        Initialize the trainer with the given configuration.
        
        Args:
            config_path (str): Path to the training configuration YAML file
        """
        self.config_path = config_path
        self.config = self._load_config(config_path)
        
        # Set random seeds for reproducibility
        self._set_random_seed(self.config.get('misc', {}).get('random_seed', 42))
        
        # Initialize other components
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = None
        self.optimizer = None
        self.scheduler = None
        self.criterion = None
        self.early_stopping_counter = 0
        self.best_metric_value = float('-inf') if self.config['evaluation']['higher_is_better'] else float('inf')
        self.best_epoch = 0
        
        # Ensure checkpoint directory exists
        os.makedirs(self.config['model']['checkpoint_dir'], exist_ok=True)
        
        # Set up tensorboard
        self.tensorboard_dir = self.config['logging']['tensorboard_dir']
        os.makedirs(self.tensorboard_dir, exist_ok=True)
        self.writer = SummaryWriter(log_dir=self.tensorboard_dir)
        
        # Track metrics over time for visualization
        self.train_metrics_history = {}
        self.val_metrics_history = {}
        
        logger.info(f"ModelTrainer initialized with config from {config_path}")
        logger.info(f"Using device: {self.device}")
        
    def _load_config(self, config_path):
        """
        Load the training configuration from a YAML file.
        
        Args:
            config_path (str): Path to the training configuration YAML file
            
        Returns:
            dict: The parsed configuration dictionary
        """
        try:
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
            logger.info(f"Configuration loaded from {config_path}")
            return config
        except Exception as e:
            logger.error(f"Failed to load configuration from {config_path}: {str(e)}")
            raise
    
    def _set_random_seed(self, seed):
        """
        Set random seeds for reproducibility.
        
        Args:
            seed (int): The random seed
        """
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
        logger.info(f"Random seed set to {seed}")
    
    def _setup_model(self, categorical_mappings=None):
        """
        Initialize the recommender model.
        
        Args:
            categorical_mappings (dict, optional): Mappings for categorical features
            
        Returns:
            HybridRecommender: The initialized model
        """
        model_config_path = self.config['model']['config_path']
        model = HybridRecommender(model_config_path, categorical_mappings)
        model.to(self.device)
        logger.info(f"Model initialized with config from {model_config_path}")
        return model
    
    def _setup_training(self, model):
        """
        Set up the training components like loss function, optimizer, and scheduler.
        
        Args:
            model (HybridRecommender): The model to train
        """
        # Set up loss function
        loss_function = self.config['training']['loss_function']
        if loss_function == 'bce':
            self.criterion = torch.nn.BCEWithLogitsLoss()
        elif loss_function == 'mse':
            self.criterion = torch.nn.MSELoss()
        else:
            raise ValueError(f"Unsupported loss function: {loss_function}")
        
        # Set up optimizer
        optimizer_type = self.config['training']['optimizer']
        lr = self.config['training']['learning_rate']
        weight_decay = self.config['training']['weight_decay']
        
        if optimizer_type == 'adam':
            self.optimizer = torch.optim.Adam(
                model.parameters(), lr=lr, weight_decay=weight_decay
            )
        elif optimizer_type == 'sgd':
            self.optimizer = torch.optim.SGD(
                model.parameters(), lr=lr, momentum=0.9, weight_decay=weight_decay
            )
        elif optimizer_type == 'rmsprop':
            self.optimizer = torch.optim.RMSprop(
                model.parameters(), lr=lr, weight_decay=weight_decay
            )
        else:
            raise ValueError(f"Unsupported optimizer: {optimizer_type}")
        
        # Set up learning rate scheduler
        scheduler_type = self.config['training']['scheduler']
        
        if scheduler_type == 'reduce_on_plateau':
            self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer,
                mode='max' if self.config['evaluation']['higher_is_better'] else 'min',
                factor=self.config['training']['scheduler_factor'],
                patience=self.config['training']['scheduler_patience'],
                min_lr=self.config['training']['scheduler_min_lr'],
                verbose=True
            )
        elif scheduler_type == 'cosine_annealing':
            self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=self.config['training']['num_epochs'],
                eta_min=self.config['training']['scheduler_min_lr']
            )
        elif scheduler_type == 'step':
            self.scheduler = torch.optim.lr_scheduler.StepLR(
                self.optimizer,
                step_size=self.config['training']['scheduler_patience'],
                gamma=self.config['training']['scheduler_factor']
            )
        else:
            raise ValueError(f"Unsupported scheduler: {scheduler_type}")
        
        logger.info(f"Training setup complete: {loss_function} loss, {optimizer_type} optimizer, {scheduler_type} scheduler")
    
    def _train_epoch(self, train_loader, epoch):
        """
        Train the model for one epoch.
        
        Args:
            train_loader (DataLoader): DataLoader for training data
            epoch (int): Current epoch number
            
        Returns:
            dict: Dictionary of training metrics
        """
        self.model.train()
        running_loss = 0.0
        
        # For metrics calculation
        all_targets = []
        all_predictions = []
        
        # Progress bar
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{self.config['training']['num_epochs']}")
        
        for i, (user_data, track_data, targets) in enumerate(pbar):
            # Move data to device
            user_data = {k: v.to(self.device) for k, v in user_data.items()}
            track_data = {k: v.to(self.device) for k, v in track_data.items()}
            targets = targets.to(self.device)
            
            # Make sure targets have the right shape
            if targets.dim() == 1 and self.criterion.__class__.__name__ == 'BCEWithLogitsLoss':
                targets = targets.view(-1, 1)
            
            # Forward pass
            self.optimizer.zero_grad()
            outputs = self.model(user_data, track_data)
            
            # Make sure outputs and targets have the same shape
            if outputs.shape != targets.shape:
                logger.warning(f"Shape mismatch - outputs: {outputs.shape}, targets: {targets.shape}")
                if outputs.dim() == 2 and outputs.shape[1] == 1 and targets.dim() == 1:
                    # If outputs are [batch_size, 1] and targets are [batch_size]
                    # Reshape targets to [batch_size, 1]
                    targets = targets.view(-1, 1)
                elif targets.dim() == 2 and targets.shape[1] == 1 and outputs.dim() == 1:
                    # If targets are [batch_size, 1] and outputs are [batch_size]
                    # Reshape outputs to [batch_size, 1]
                    outputs = outputs.view(-1, 1)
            
            # Calculate loss
            loss = self.criterion(outputs, targets)
            
            # Backward pass and optimization
            loss.backward()
            
            # Clip gradients if enabled
            if self.config['training']['clip_gradients']:
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.config['training']['max_gradient_norm']
                )
            
            self.optimizer.step()
            
            # Update running loss
            running_loss += loss.item()
            
            # Update progress bar
            pbar.set_postfix({'loss': running_loss / (i + 1)})
            
            # Collect predictions and targets for metrics
            predictions = torch.sigmoid(outputs) if isinstance(self.criterion, torch.nn.BCEWithLogitsLoss) else outputs
            
            # Ensure both are 1D for metrics
            all_targets.extend(targets.squeeze().cpu().numpy())
            all_predictions.extend(predictions.squeeze().detach().cpu().numpy())
            
            # Log to tensorboard periodically
            if (i + 1) % self.config['logging']['log_interval'] == 0:
                step = epoch * len(train_loader) + i
                self.writer.add_scalar('training/loss', loss.item(), step)
                lr = self.optimizer.param_groups[0]['lr']
                self.writer.add_scalar('training/learning_rate', lr, step)
        
        # Calculate average loss
        epoch_loss = running_loss / len(train_loader)
        
        # Calculate metrics
        metrics = calculate_metrics(
            np.array(all_targets),
            np.array(all_predictions),
            self.config['evaluation']['classification_metrics']
        )
        
        # Log metrics to tensorboard
        self.writer.add_scalar('training/epoch_loss', epoch_loss, epoch)
        for metric_name, metric_value in metrics.items():
            self.writer.add_scalar(f'training/{metric_name}', metric_value, epoch)
        
        # Add loss to metrics
        metrics['loss'] = epoch_loss
        
        # Update metrics history
        for metric_name, metric_value in metrics.items():
            if metric_name not in self.train_metrics_history:
                self.train_metrics_history[metric_name] = []
            self.train_metrics_history[metric_name].append(metric_value)
        
        logger.info(f"Epoch {epoch+1} - Training Loss: {epoch_loss:.4f}")
        return metrics
    
    def _validate_epoch(self, val_loader, epoch):
        """
        Validate the model on the validation data.
        
        Args:
            val_loader (DataLoader): DataLoader for validation data
            epoch (int): Current epoch number
            
        Returns:
            dict: Dictionary of validation metrics
        """
        self.model.eval()
        val_loss = 0.0
        
        # For metrics calculation
        all_targets = []
        all_predictions = []
        user_prediction_map = {}  # For ranking metrics
        
        with torch.no_grad():
            for user_data, track_data, targets in tqdm(val_loader, desc="Validation"):
                # Move data to device
                user_data = {k: v.to(self.device) for k, v in user_data.items()}
                track_data = {k: v.to(self.device) for k, v in track_data.items()}
                targets = targets.to(self.device)
                
                # Make sure targets have the right shape
                if targets.dim() == 1 and self.criterion.__class__.__name__ == 'BCEWithLogitsLoss':
                    targets = targets.view(-1, 1)
                
                # Forward pass
                outputs = self.model(user_data, track_data)
                
                # Make sure outputs and targets have the same shape
                if outputs.shape != targets.shape:
                    if outputs.dim() == 2 and outputs.shape[1] == 1 and targets.dim() == 1:
                        # If outputs are [batch_size, 1] and targets are [batch_size]
                        # Reshape targets to [batch_size, 1]
                        targets = targets.view(-1, 1)
                    elif targets.dim() == 2 and targets.shape[1] == 1 and outputs.dim() == 1:
                        # If targets are [batch_size, 1] and outputs are [batch_size]
                        # Reshape outputs to [batch_size, 1]
                        outputs = outputs.view(-1, 1)
                
                # Calculate loss
                loss = self.criterion(outputs, targets)
                val_loss += loss.item()
                
                # Collect predictions and targets for metrics
                predictions = torch.sigmoid(outputs) if isinstance(self.criterion, torch.nn.BCEWithLogitsLoss) else outputs
                
                # Ensure both are 1D for metrics
                all_targets.extend(targets.squeeze().cpu().numpy())
                all_predictions.extend(predictions.squeeze().cpu().numpy())
        
        # Calculate average loss
        epoch_val_loss = val_loss / len(val_loader)
        
        # Calculate metrics
        metrics = calculate_metrics(
            np.array(all_targets),
            np.array(all_predictions),
            self.config['evaluation']['classification_metrics'],
            self.config['evaluation']['ranking_metrics']
        )
        
        # Log metrics to tensorboard
        self.writer.add_scalar('validation/loss', epoch_val_loss, epoch)
        for metric_name, metric_value in metrics.items():
            self.writer.add_scalar(f'validation/{metric_name}', metric_value, epoch)
        
        # Add loss to metrics
        metrics['loss'] = epoch_val_loss
        
        # Update metrics history
        for metric_name, metric_value in metrics.items():
            if metric_name not in self.val_metrics_history:
                self.val_metrics_history[metric_name] = []
            self.val_metrics_history[metric_name].append(metric_value)
        
        # Log validation results
        logger.info(f"Epoch {epoch+1} - Validation Loss: {epoch_val_loss:.4f}")
        for metric_name, metric_value in metrics.items():
            if metric_name != 'loss':
                logger.info(f"Epoch {epoch+1} - Validation {metric_name}: {metric_value:.4f}")
        
        return metrics
    
    def _save_checkpoint(self, epoch, metrics, is_best=False):
        """
        Save model checkpoint.
        
        Args:
            epoch (int): Current epoch number
            metrics (dict): Dictionary of metrics
            is_best (bool): Whether this is the best model so far
        """
        checkpoint_dir = self.config['model']['checkpoint_dir']
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'metrics': metrics,
            'config': self.model.config,
            'categorical_mappings': self.model.categorical_mappings
        }
        
        # Save regular checkpoint
        checkpoint_path = os.path.join(checkpoint_dir, f"model_epoch_{epoch+1}.pt")
        torch.save(checkpoint, checkpoint_path)
        logger.info(f"Checkpoint saved to {checkpoint_path}")
        
        # Save best model if this is the best so far
        if is_best:
            best_model_path = self.config['model']['best_model_path']
            torch.save(checkpoint, best_model_path)
            logger.info(f"New best model saved to {best_model_path}")
            
            # Save summary of best model metrics
            metrics_summary = {
                'best_epoch': epoch + 1,
                'metrics': metrics
            }
            
            metrics_path = os.path.join(checkpoint_dir, "best_model_metrics.json")
            with open(metrics_path, 'w') as f:
                json.dump(metrics_summary, f, indent=2)
            logger.info(f"Best model metrics saved to {metrics_path}")
    
    def _check_early_stopping(self, metrics):
        """
        Check if training should stop based on validation performance.
        
        Args:
            metrics (dict): Dictionary of validation metrics
            
        Returns:
            bool: True if training should stop, False otherwise
        """
        # Get the primary metric
        primary_metric = self.config['evaluation']['primary_metric']
        current_metric = metrics.get(primary_metric)
        
        if current_metric is None:
            logger.warning(f"Primary metric {primary_metric} not found in validation metrics")
            return False
        
        higher_is_better = self.config['evaluation']['higher_is_better']
        is_improvement = False
        
        if higher_is_better:
            is_improvement = current_metric > self.best_metric_value
        else:
            is_improvement = current_metric < self.best_metric_value
        
        if is_improvement:
            self.best_metric_value = current_metric
            self.early_stopping_counter = 0
            return False
        else:
            self.early_stopping_counter += 1
            logger.info(f"Early stopping counter: {self.early_stopping_counter}/{self.config['training']['early_stopping_patience']}")
            
            if self.early_stopping_counter >= self.config['training']['early_stopping_patience']:
                logger.info(f"Early stopping triggered after {self.early_stopping_counter} epochs without improvement")
                return True
            return False
    
    def _update_learning_rate(self, metrics):
        """
        Update the learning rate based on the scheduler.
        
        Args:
            metrics (dict): Dictionary of validation metrics
        """
        scheduler_type = self.config['training']['scheduler']
        primary_metric = self.config['evaluation']['primary_metric']
        
        if scheduler_type == 'reduce_on_plateau':
            # ReduceLROnPlateau needs the validation metric
            metric_value = metrics.get(primary_metric, metrics.get('loss'))
            self.scheduler.step(metric_value)
        else:
            # Other schedulers just need to be stepped
            self.scheduler.step()
        
        # Log the new learning rate
        new_lr = self.optimizer.param_groups[0]['lr']
        logger.info(f"Learning rate updated to {new_lr:.6f}")
    
    def train(self, train_loader, val_loader, categorical_mappings=None):
        """
        Train the model.
        
        Args:
            train_loader (DataLoader): DataLoader for training data
            val_loader (DataLoader): DataLoader for validation data
            categorical_mappings (dict, optional): Mappings for categorical features
            
        Returns:
            dict: Dictionary of metrics for the best model
        """
        # Setup model and training components
        self.model = self._setup_model(categorical_mappings)
        self._setup_training(self.model)
        
        # Reset metrics history
        self.train_metrics_history = {}
        self.val_metrics_history = {}
        
        # Training loop
        start_time = time.time()
        for epoch in range(self.config['training']['num_epochs']):
            # Train for one epoch
            train_metrics = self._train_epoch(train_loader, epoch)
            
            # Validate the model
            val_metrics = self._validate_epoch(val_loader, epoch)
            
            # Check if this is the best model so far
            primary_metric = self.config['evaluation']['primary_metric']
            current_metric = val_metrics.get(primary_metric, val_metrics.get('loss'))
            
            is_best = False
            if self.config['evaluation']['higher_is_better']:
                is_best = current_metric > self.best_metric_value
            else:
                is_best = current_metric < self.best_metric_value
            
            if is_best:
                self.best_metric_value = current_metric
                self.best_epoch = epoch
                logger.info(f"New best model at epoch {epoch+1} with {primary_metric} = {current_metric:.4f}")
            
            # Save checkpoint
            self._save_checkpoint(epoch, val_metrics, is_best)
            
            # Update learning rate
            self._update_learning_rate(val_metrics)
            
            # Check early stopping
            if self._check_early_stopping(val_metrics):
                logger.info(f"Early stopping at epoch {epoch+1}")
                break
        
        # Generate and save visualizations
        self._generate_training_visualizations()
        
        # Final summary
        total_time = time.time() - start_time
        logger.info(f"Training completed in {total_time:.2f} seconds")
        logger.info(f"Best model achieved at epoch {self.best_epoch+1} with {primary_metric} = {self.best_metric_value:.4f}")
        
        # Load and return the best model
        best_model_path = self.config['model']['best_model_path']
        if os.path.exists(best_model_path):
            checkpoint = torch.load(best_model_path, map_location=self.device)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            logger.info(f"Loaded best model from {best_model_path}")
            
            return checkpoint['metrics']
        else:
            logger.warning(f"Best model not found at {best_model_path}")
            return val_metrics
    
    def _generate_training_visualizations(self):
        """
        Generate and save training visualizations using the visualization module.
        """
        # Create visualization directory
        vis_dir = os.path.join(self.config['logging']['log_dir'], 'visualizations')
        os.makedirs(vis_dir, exist_ok=True)
        
        # Plot learning curves
        if self.train_metrics_history and self.val_metrics_history:
            learning_curves_path = os.path.join(vis_dir, "learning_curves.png")
            plot_learning_curves(
                self.train_metrics_history,
                self.val_metrics_history,
                learning_curves_path
            )
            logger.info(f"Learning curves plot saved to {learning_curves_path}")
    
    def evaluate(self, test_loader, model_path=None):
        """
        Evaluate the model on test data.
        
        Args:
            test_loader (DataLoader): DataLoader for test data
            model_path (str, optional): Path to the model checkpoint
            
        Returns:
            dict: Dictionary of test metrics
        """
        # Load the model if path is provided
        if model_path:
            checkpoint = torch.load(model_path, map_location=self.device)
            
            # Create model using checkpoint configuration
            self.model = HybridRecommender(
                config_path=checkpoint.get('config'),
                categorical_mappings=checkpoint.get('categorical_mappings')
            )
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.model.to(self.device)
            logger.info(f"Model loaded from {model_path}")
        
        # Ensure model exists
        if self.model is None:
            raise ValueError("Model not loaded. Please train the model or provide a model path")
        
        # Set model to evaluation mode
        self.model.eval()
        
        # Initialize loss if not already set up
        if self.criterion is None:
            loss_function = self.config['training']['loss_function']
            if loss_function == 'bce':
                self.criterion = torch.nn.BCEWithLogitsLoss()
            elif loss_function == 'mse':
                self.criterion = torch.nn.MSELoss()
        
        test_loss = 0.0
        all_targets = []
        all_predictions = []
        all_embeddings = []  # For embedding visualization
        
        # Evaluate the model
        with torch.no_grad():
            for user_data, track_data, targets in tqdm(test_loader, desc="Evaluating"):
                # Move data to device
                user_data = {k: v.to(self.device) for k, v in user_data.items()}
                track_data = {k: v.to(self.device) for k, v in track_data.items()}
                targets = targets.to(self.device)
                
                # Make sure targets have the right shape
                if targets.dim() == 1 and self.criterion.__class__.__name__ == 'BCEWithLogitsLoss':
                    targets = targets.view(-1, 1)
                
                # Forward pass
                outputs = self.model(user_data, track_data)
                
                # Optionally get embeddings
                if hasattr(self.model, 'get_embeddings'):
                    user_embeddings, item_embeddings = self.model.get_embeddings(user_data, track_data)
                    # Combine embeddings (simple concatenation for visualization)
                    combined_embeddings = torch.cat([user_embeddings, item_embeddings], dim=1)
                    all_embeddings.extend(combined_embeddings.cpu().numpy())
                
                # Make sure outputs and targets have the same shape
                if outputs.shape != targets.shape:
                    if outputs.dim() == 2 and outputs.shape[1] == 1 and targets.dim() == 1:
                        # If outputs are [batch_size, 1] and targets are [batch_size]
                        # Reshape targets to [batch_size, 1]
                        targets = targets.view(-1, 1)
                    elif targets.dim() == 2 and targets.shape[1] == 1 and outputs.dim() == 1:
                        # If targets are [batch_size, 1] and outputs are [batch_size]
                        # Reshape outputs to [batch_size, 1]
                        outputs = outputs.view(-1, 1)
                
                # Calculate loss
                loss = self.criterion(outputs, targets)
                test_loss += loss.item()
                
                # Collect predictions and targets for metrics
                predictions = torch.sigmoid(outputs) if isinstance(self.criterion, torch.nn.BCEWithLogitsLoss) else outputs
                
                # Ensure both are 1D for metrics
                all_targets.extend(targets.squeeze().cpu().numpy())
                all_predictions.extend(predictions.squeeze().cpu().numpy())
        
        # Calculate average loss
        test_loss = test_loss / len(test_loader)
        
        # Calculate metrics
        metrics = calculate_metrics(
            np.array(all_targets),
            np.array(all_predictions),
            self.config['evaluation']['classification_metrics'],
            self.config['evaluation']['ranking_metrics']
        )
        
        # Add loss to metrics
        metrics['loss'] = test_loss
        
        # Log test results
        logger.info(f"Test Loss: {test_loss:.4f}")
        for metric_name, metric_value in metrics.items():
            if metric_name != 'loss':
                logger.info(f"Test {metric_name}: {metric_value:.4f}")
        
        # Generate comprehensive performance report
        report_dir = os.path.join(self.config['logging']['log_dir'], 'reports')
        model_name = os.path.basename(model_path).split('.')[0] if model_path else 'latest_model'
        
        # Convert train and val metrics history to the format expected by generate_performance_report
        train_metrics_dict = {k: v for k, v in self.train_metrics_history.items()} if hasattr(self, 'train_metrics_history') else {}
        val_metrics_dict = {k: v for k, v in self.val_metrics_history.items()} if hasattr(self, 'val_metrics_history') else {}
        
        # Get feature names and importances if available
        feature_names = None
        feature_importance = None
        if hasattr(self.model, 'get_feature_importance'):
            feature_names, feature_importance = self.model.get_feature_importance()
        
        # Convert embeddings to numpy array if any were collected
        embeddings = np.array(all_embeddings) if all_embeddings else None
        
        # Generate the performance report
        plot_paths = generate_performance_report(
            np.array(all_targets),
            np.array(all_predictions),
            train_metrics_dict,
            val_metrics_dict,
            model_name,
            report_dir,
            embeddings=embeddings,
            feature_names=feature_names,
            feature_importance=feature_importance
        )
        
        logger.info(f"Performance report generated in {report_dir}/{model_name}_report")
        
        return metrics
    
    def predict(self, user_data, track_data, model_path=None):
        """
        Make predictions for a batch of user-track pairs.
        
        Args:
            user_data (dict): Dictionary of user features
            track_data (dict): Dictionary of track features
            model_path (str, optional): Path to the model checkpoint
            
        Returns:
            numpy.ndarray: Predicted scores
        """
        # Load the model if path is provided
        if model_path:
            checkpoint = torch.load(model_path, map_location=self.device)
            
            # Create model using checkpoint configuration
            self.model = HybridRecommender(
                config=checkpoint.get('config'),
                categorical_mappings=checkpoint.get('categorical_mappings')
            )
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.model.to(self.device)
            logger.info(f"Model loaded from {model_path}")
        
        # Ensure model exists
        if self.model is None:
            raise ValueError("Model not loaded. Please train the model or provide a model path")
        
        # Set model to evaluation mode
        self.model.eval()
        
        # Move data to device
        user_data = {k: v.to(self.device) for k, v in user_data.items()}
        track_data = {k: v.to(self.device) for k, v in track_data.items()}
        
        # Make predictions
        with torch.no_grad():
            outputs = self.model(user_data, track_data)
            predictions = torch.sigmoid(outputs).cpu().numpy()
        
        return predictions 