import torch
import os
import sys
import yaml
import time
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from torch.utils.data import DataLoader
import torch.optim as optim
import torch.nn as nn
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm
import pandas as pd
import logging

# Add root directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models.hybrid_model import HybridMusicRecommender
from utils.logger import LhydraLogger, log_function
from utils.metrics_utils import calculate_all_metrics
from utils.logging_utils import log_epoch_metrics, TensorboardLogger

logger = logging.getLogger(__name__)

class HybridModelTrainer:
    """
    Trainer class for the Hybrid Recommender model
    """
    
    def __init__(self, model, config, device=None):
        """
        Initialize the trainer
        
        Args:
            model (nn.Module): The model to train
            config (dict): Training configuration
            device (torch.device, optional): Device to train on
        """
        self.model = model
        self.config = config
        
        # Set device 
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = device
        
        self.model.to(self.device)
        
        # Set optimizer
        lr = config['training']['learning_rate']
        weight_decay = eval(config['training']['weight_decay'])
        optimizer_name = config['training']['optimizer'].lower()
        
        if optimizer_name == 'adam':
            self.optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=(weight_decay))
        elif optimizer_name == 'sgd':
            self.optimizer = optim.SGD(model.parameters(), lr=lr, weight_decay=weight_decay, momentum=0.9)
        elif optimizer_name == 'adagrad':
            self.optimizer = optim.Adagrad(model.parameters(), lr=lr, weight_decay=weight_decay)
        else:
            logger.warning(f"Optimizer {optimizer_name} not recognized. Using Adam as default.")
            self.optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
        
        # Set loss function
        loss_name = config['training']['loss_function'].lower()
        class_weights = config['training'].get('class_weights', None)
        
        if class_weights is not None:
            class_weights = torch.tensor(class_weights, dtype=torch.float32, device=self.device)
        
        if loss_name == 'binary_cross_entropy':
            self.criterion = nn.BCELoss(weight=class_weights)
        elif loss_name == 'mse':
            self.criterion = nn.MSELoss()
        else:
            logger.warning(f"Loss function {loss_name} not recognized. Using BCE as default.")
            self.criterion = nn.BCELoss(weight=class_weights)
        
        # Setup learning rate scheduler if enabled
        if config['training']['lr_scheduler'].get('use', False):
            scheduler_config = config['training']['lr_scheduler']
            self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer,
                mode='min',
                factor=scheduler_config.get('factor', 0.5),
                patience=scheduler_config.get('patience', 3),
            )
            
        else:
            self.scheduler = None
        
        # Setup early stopping
        self.early_stopping_patience = config['training'].get('early_stopping_patience', 5)
        self.best_val_loss = float('inf')
        self.early_stopping_counter = 0
        
        # Setup logging
        self.log_frequency = config['logging'].get('log_frequency', 100)
        log_dir = config['logging']['log_dir']
        
        # Setup tensorboard logging if enabled
        if config['logging'].get('tensorboard', False):
            self.tb_logger = TensorboardLogger(os.path.join(log_dir, 'tensorboard'))
        else:
            self.tb_logger = None
        
        # Setup model checkpointing
        self.checkpoint_dir = config['logging']['model_checkpoint_dir']
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        self.save_best_only = config['logging'].get('save_best_only', True)
        
        logger.info(f"Trainer initialized on device: {self.device}")
    
    def train_epoch(self, train_loader, epoch):
        """
        Train for one epoch
        
        Args:
            train_loader (DataLoader): Training data loader
            epoch (int): Current epoch number
            
        Returns:
            dict: Training metrics
        """
        self.model.train()
        epoch_loss = 0.0
        all_targets = []
        all_predictions = []
        
        start_time = time.time()
        
        for batch_idx, batch in enumerate(tqdm(train_loader, desc=f"Epoch {epoch}")):
            # Get data
            user_features = batch['user_features'].to(self.device)
            item_features = batch['item_features'].to(self.device)
            targets = batch['target'].to(self.device)
            
            # Forward pass
            self.optimizer.zero_grad()
            outputs = self.model(user_features, item_features)
            
            # Calculate loss
            loss = self.criterion(outputs, targets)
            
            # Backward pass and optimization
            loss.backward()
            self.optimizer.step()
            
            # Track loss
            epoch_loss += loss.item()
            
            # Store predictions and targets for metrics calculation
            all_targets.extend(targets.cpu().numpy())
            all_predictions.extend(outputs.detach().cpu().numpy())
            
            # Log batch stats
            if batch_idx % self.log_frequency == 0:
                logger.debug(f"Epoch {epoch} Batch {batch_idx}/{len(train_loader)}: Loss {loss.item():.4f}")
                if self.tb_logger:
                    self.tb_logger.log_scalar('batch/train_loss', loss.item(), epoch * len(train_loader) + batch_idx)
        
        # Convert to numpy arrays
        all_targets = np.array(all_targets)
        all_predictions = np.array(all_predictions)
        
        # Calculate metrics
        metrics = calculate_all_metrics(all_targets, all_predictions, all_predictions)
        metrics['loss'] = epoch_loss / len(train_loader)
        metrics['epoch_time'] = time.time() - start_time
        
        return metrics
    
    def validate(self, val_loader, epoch):
        """
        Validate the model
        
        Args:
            val_loader (DataLoader): Validation data loader
            epoch (int): Current epoch number
            
        Returns:
            dict: Validation metrics
        """
        self.model.eval()
        val_loss = 0.0
        all_targets = []
        all_predictions = []
        all_user_ids = []
        
        # with torch.no_grad():
        with torch.inference_mode():
            for batch in tqdm(val_loader, desc="Validation"):
                # Get data
                user_features = batch['user_features'].to(self.device)
                item_features = batch['item_features'].to(self.device)
                targets = batch['target'].to(self.device)
                user_ids = batch['user_id']
                
                # Forward pass
                outputs = self.model(user_features, item_features)
                
                # Calculate loss
                loss = self.criterion(outputs, targets)
                
                # Track loss
                val_loss += loss.item()
                
                # Store predictions and targets for metrics calculation
                all_targets.extend(targets.cpu().numpy())
                all_predictions.extend(outputs.cpu().numpy())
                all_user_ids.extend(user_ids)
        
        # Convert to numpy arrays
        all_targets = np.array(all_targets)
        all_predictions = np.array(all_predictions)
        all_user_ids = np.array(all_user_ids)
        
        # Calculate metrics
        metrics = calculate_all_metrics(
            all_targets, 
            all_predictions, 
            all_predictions, 
            user_ids=all_user_ids,
            k_values=self.config['evaluation']['k_values']
        )
        metrics['loss'] = val_loss / len(val_loader)
        
        # Log validation set confusion matrix and ROC curve
        if self.tb_logger:
            self.tb_logger.log_confusion_matrix(
                'validation/confusion_matrix',
                all_targets,
                all_predictions,
                epoch
            )
            self.tb_logger.log_roc_curve(
                'validation/roc_curve',
                all_targets,
                all_predictions,
                epoch
            )
            self.tb_logger.log_precision_recall_curve(
                'validation/pr_curve',
                all_targets,
                all_predictions,
                epoch
            )
        
        return metrics
    
    def train(self, train_loader, val_loader, num_epochs=None):
        """
        Train the model
        
        Args:
            train_loader (DataLoader): Training data loader
            val_loader (DataLoader): Validation data loader
            num_epochs (int, optional): Number of epochs to train. If None, use config value.
            
        Returns:
            pandas.DataFrame: Training history
        """
        if num_epochs is None:
            num_epochs = self.config['training']['epochs']
        
        # Create history dataframe
        history = pd.DataFrame()
        
        # Log hyperparameters
        if self.tb_logger:
            hparam_dict = {
                'learning_rate': self.config['training']['learning_rate'],
                'batch_size': self.config['training']['batch_size'],
                'optimizer': self.config['training']['optimizer'],
                'weight_decay': self.config['training']['weight_decay'],
                'user_tower_hidden_layers': str(self.config['model']['user_tower']['hidden_layers']),
                'item_tower_hidden_layers': str(self.config['model']['item_tower']['hidden_layers']),
                'dropout': self.config['model']['user_tower']['dropout'],
                'embedding_dim': self.config['model']['embedding_dim'],
                'final_layer_size': self.config['model']['final_layer_size']
            }
            self.tb_logger.log_hyperparameters(hparam_dict, {})
        
        logger.info(f"Starting training for {num_epochs} epochs")
        
        for epoch in range(1, num_epochs + 1):
            # Train and validate
            train_metrics = self.train_epoch(train_loader, epoch)
            val_metrics = self.validate(val_loader, epoch)
            
            # Log metrics
            log_epoch_metrics(epoch, train_metrics, val_metrics, logger, self.tb_logger)
            
            # Update learning rate scheduler if used
            if self.scheduler is not None:
                self.scheduler.step(val_metrics['loss'])
            
            # Save metrics to history
            epoch_df = pd.DataFrame({
                'epoch': [epoch],
                **{f'train_{k}': [v] for k, v in train_metrics.items()},
                **{f'val_{k}': [v] for k, v in val_metrics.items()}
            })
            history = pd.concat([history, epoch_df], ignore_index=True)
            
            # Save model checkpoint
            self._save_checkpoint(epoch, val_metrics['loss'])
            
            # Check early stopping
            if val_metrics['loss'] < self.best_val_loss:
                self.best_val_loss = val_metrics['loss']
                self.early_stopping_counter = 0
            else:
                self.early_stopping_counter += 1
                logger.info(f"Early stopping counter: {self.early_stopping_counter}/{self.early_stopping_patience}")
                
                if self.early_stopping_counter >= self.early_stopping_patience:
                    logger.info(f"Early stopping triggered after {epoch} epochs")
                    break
        
        logger.info("Training completed")
        
        if self.tb_logger:
            self.tb_logger.close()
        
        return history
    
    def _save_checkpoint(self, epoch, val_loss):
        """
        Save model checkpoint
        
        Args:
            epoch (int): Current epoch number
            val_loss (float): Validation loss
        """
        is_best = val_loss < self.best_val_loss
        
        # Save the model if it's the best so far or if we're saving all checkpoints
        if is_best or not self.save_best_only:
            checkpoint_file = os.path.join(self.checkpoint_dir, f'model_epoch_{epoch}.pt')
            self.model.save(checkpoint_file)
            
            # If it's the best model, save a copy as 'best_model.pt'
            if is_best:
                best_model_file = os.path.join(self.checkpoint_dir, 'best_model.pt')
                self.model.save(best_model_file)
                logger.info(f"Saved new best model with validation loss: {val_loss:.4f}")
    
    def evaluate(self, test_loader):
        """
        Evaluate the model on the test set
        
        Args:
            test_loader (DataLoader): Test data loader
            
        Returns:
            dict: Test metrics
        """
        logger.info("Evaluating model on test set")
        self.model.eval()
        test_loss = 0.0
        all_targets = []
        all_predictions = []
        all_probabilities = []
        all_user_ids = []
        
        # with torch.no_grad():
        with torch.inference_mode():
            for batch in tqdm(test_loader, desc="Testing"):
                # Get data
                user_features = batch['user_features'].to(self.device)
                item_features = batch['item_features'].to(self.device)
                targets = batch['target'].to(self.device)
                user_ids = batch['user_id']
                
                # Forward pass
                outputs = self.model(user_features, item_features)
                
                # Calculate loss
                loss = self.criterion(outputs, targets)
                
                # Track loss
                test_loss += loss.item()
                
                # Store predictions and targets for metrics calculation
                all_targets.extend(targets.cpu().numpy())
                all_predictions.extend((outputs > 0.5).float().cpu().numpy())
                all_probabilities.extend(outputs.cpu().numpy())
                all_user_ids.extend(user_ids)
        
        # Convert to numpy arrays
        all_targets = np.array(all_targets)
        all_predictions = np.array(all_predictions)
        all_probabilities = np.array(all_probabilities)
        all_user_ids = np.array(all_user_ids)
        
        # Calculate metrics
        metrics = calculate_all_metrics(
            all_targets, 
            all_predictions, 
            all_probabilities, 
            user_ids=all_user_ids,
            k_values=self.config['evaluation']['k_values']
        )
        metrics['loss'] = test_loss / len(test_loader)
        
        # Log results
        logger.info("Test Results:")
        for k, v in metrics.items():
            logger.info(f"  {k}: {v:.4f}")
        
        # Save test results
        results_file = os.path.join(self.config['logging']['log_dir'], 'test_results.csv')
        pd.DataFrame([metrics]).to_csv(results_file, index=False)
        logger.info(f"Saved test results to {results_file}")
        
        # Generate and save evaluation plots
        plots_dir = os.path.join(self.config['logging']['log_dir'], 'evaluation_plots')
        os.makedirs(plots_dir, exist_ok=True)
        self.generate_evaluation_plots(all_targets, all_predictions, all_probabilities, plots_dir)
        
        return metrics
    
    def generate_evaluation_plots(self, targets, predictions, probabilities, output_dir):
        """
        Generate and save evaluation plots
        
        Args:
            targets (np.ndarray): Ground truth labels
            predictions (np.ndarray): Predicted binary labels
            probabilities (np.ndarray): Predicted probabilities
            output_dir (str): Directory to save plots
        """
        import matplotlib.pyplot as plt
        import seaborn as sns
        from sklearn.metrics import roc_curve, precision_recall_curve, auc, confusion_matrix
        
        logger.info("Generating evaluation plots")
        
        # Plot confusion matrix
        cm = confusion_matrix(targets, predictions)
        
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
        
        # Plot prediction distribution over time (if enough samples)
        if len(targets) > 100:
            window_size = min(100, len(targets) // 10)
            rolling_accuracy = np.convolve(predictions == targets, 
                                          np.ones(window_size)/window_size, 
                                          mode='valid')
            
            plt.figure(figsize=(10, 6))
            plt.plot(rolling_accuracy, color='green')
            plt.axhline(y=np.mean(predictions == targets), color='red', linestyle='--', 
                       label=f'Overall Accuracy: {np.mean(predictions == targets):.3f}')
            plt.xlabel('Sample Index')
            plt.ylabel(f'Accuracy (Rolling Window Size: {window_size})')
            plt.title('Prediction Accuracy Over Samples')
            plt.legend()
            plt.grid(True)
            plt.savefig(os.path.join(output_dir, "accuracy_trend.png"))
            plt.close()
        
        # Plot calibration curve (reliability diagram)
        from sklearn.calibration import calibration_curve
        
        plt.figure(figsize=(10, 6))
        
        # Use around 10 bins
        n_bins = min(10, len(np.unique(probabilities)))
        if n_bins >= 5:  # Only if we have enough unique probability values
            fraction_of_positives, mean_predicted_value = calibration_curve(
                targets, probabilities, n_bins=n_bins)
            
            plt.plot(mean_predicted_value, fraction_of_positives, "s-", label="Model")
            plt.plot([0, 1], [0, 1], "k:", label="Perfectly calibrated")
            
            plt.xlabel("Mean predicted probability")
            plt.ylabel("Fraction of positives")
            plt.title("Calibration Curve (Reliability Diagram)")
            plt.legend()
            plt.grid(True)
            plt.savefig(os.path.join(output_dir, "calibration_curve.png"))
        plt.close()
        
        logger.info(f"Evaluation plots saved to {output_dir}") 