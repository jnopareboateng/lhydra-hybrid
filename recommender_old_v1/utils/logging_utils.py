import logging
import os
import sys
import time
import torch
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, roc_curve, precision_recall_curve
import pandas as pd

def setup_logging(log_dir="logs", log_level=logging.INFO):
    """
    Setup logging configuration
    
    Args:
        log_dir (str): Directory to save log files
        log_level (int): Logging level
    
    Returns:
        logging.Logger: Logger object
    """
    # Create log directory if it doesn't exist
    os.makedirs(log_dir, exist_ok=True)
    
    # Create a unique log filename based on timestamp
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    log_filename = os.path.join(log_dir, f"training_{timestamp}.log")
    
    # Configure logging
    logger = logging.getLogger()
    logger.setLevel(log_level)
    
    # Create file handler
    file_handler = logging.FileHandler(log_filename)
    file_handler.setLevel(log_level)
    
    # Create console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(log_level)
    
    # Create formatter and add it to the handlers
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)
    
    # Add handlers to logger
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    logger.info(f"Logging setup complete. Log file: {log_filename}")
    return logger

class TensorboardLogger:
    """
    Class for logging metrics to Tensorboard
    """
    def __init__(self, log_dir="logs/tensorboard"):
        """
        Initialize the Tensorboard logger
        
        Args:
            log_dir (str): Directory to save tensorboard logs
        """
        os.makedirs(log_dir, exist_ok=True)
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        self.log_dir = os.path.join(log_dir, timestamp)
        self.writer = SummaryWriter(log_dir=self.log_dir)
        logging.info(f"Tensorboard logging setup complete. Log directory: {self.log_dir}")
    
    def log_scalar(self, tag, value, step):
        """
        Log a scalar value to tensorboard
        
        Args:
            tag (str): Name of the scalar
            value (float): Value to log
            step (int): Step/iteration number
        """
        self.writer.add_scalar(tag, value, step)
    
    def log_scalars(self, main_tag, tag_value_dict, step):
        """
        Log multiple scalars under the same main tag
        
        Args:
            main_tag (str): Main tag name
            tag_value_dict (dict): Dictionary of tag names and values
            step (int): Step/iteration number
        """
        self.writer.add_scalars(main_tag, tag_value_dict, step)
    
    def log_histogram(self, tag, values, step):
        """
        Log a histogram to tensorboard
        
        Args:
            tag (str): Name of the histogram
            values (torch.Tensor or numpy.ndarray): Values to create histogram from
            step (int): Step/iteration number
        """
        if isinstance(values, torch.Tensor):
            values = values.detach().cpu().numpy()
        self.writer.add_histogram(tag, values, step)
    
    def log_image(self, tag, image, step):
        """
        Log an image to tensorboard
        
        Args:
            tag (str): Name of the image
            image (torch.Tensor or numpy.ndarray): Image to log
            step (int): Step/iteration number
        """
        self.writer.add_image(tag, image, step)
    
    def log_figure(self, tag, figure, step):
        """
        Log a matplotlib figure to tensorboard
        
        Args:
            tag (str): Name of the figure
            figure (matplotlib.figure.Figure): Figure to log
            step (int): Step/iteration number
        """
        self.writer.add_figure(tag, figure, step)
    
    def log_confusion_matrix(self, tag, y_true, y_pred, step, class_names=None):
        """
        Log a confusion matrix as a figure to tensorboard
        
        Args:
            tag (str): Name of the confusion matrix
            y_true (numpy.ndarray): True labels
            y_pred (numpy.ndarray): Predicted labels
            step (int): Step/iteration number
            class_names (list): List of class names
        """
        cm = confusion_matrix(y_true, (y_pred > 0.5).astype(int))
        
        if class_names is None:
            class_names = ['Negative', 'Positive']
        
        fig, ax = plt.subplots(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=class_names, yticklabels=class_names)
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.title('Confusion Matrix')
        
        self.log_figure(tag, fig, step)
        plt.close(fig)
    
    def log_roc_curve(self, tag, y_true, y_pred_proba, step):
        """
        Log a ROC curve as a figure to tensorboard
        
        Args:
            tag (str): Name of the ROC curve
            y_true (numpy.ndarray): True labels
            y_pred_proba (numpy.ndarray): Predicted probabilities
            step (int): Step/iteration number
        """
        fpr, tpr, _ = roc_curve(y_true, y_pred_proba)
        
        fig, ax = plt.subplots(figsize=(10, 8))
        ax.plot(fpr, tpr, label='ROC curve')
        ax.plot([0, 1], [0, 1], 'k--', label='Random')
        ax.set_xlabel('False Positive Rate')
        ax.set_ylabel('True Positive Rate')
        ax.set_title('Receiver Operating Characteristic (ROC) Curve')
        ax.legend(loc='lower right')
        
        self.log_figure(tag, fig, step)
        plt.close(fig)
    
    def log_precision_recall_curve(self, tag, y_true, y_pred_proba, step):
        """
        Log a precision-recall curve as a figure to tensorboard
        
        Args:
            tag (str): Name of the precision-recall curve
            y_true (numpy.ndarray): True labels
            y_pred_proba (numpy.ndarray): Predicted probabilities
            step (int): Step/iteration number
        """
        precision, recall, _ = precision_recall_curve(y_true, y_pred_proba)
        
        fig, ax = plt.subplots(figsize=(10, 8))
        ax.plot(recall, precision, label='Precision-Recall curve')
        ax.set_xlabel('Recall')
        ax.set_ylabel('Precision')
        ax.set_title('Precision-Recall Curve')
        ax.legend(loc='lower left')
        
        self.log_figure(tag, fig, step)
        plt.close(fig)
    
    def log_model_graph(self, model, input_to_model):
        """
        Log the model graph to tensorboard
        
        Args:
            model (torch.nn.Module): PyTorch model
            input_to_model (torch.Tensor or tuple): Example input to the model
        """
        self.writer.add_graph(model, input_to_model)
    
    def log_hyperparameters(self, hparam_dict, metric_dict):
        """
        Log hyperparameters and metrics
        
        Args:
            hparam_dict (dict): Dictionary of hyperparameters
            metric_dict (dict): Dictionary of metrics
        """
        self.writer.add_hparams(hparam_dict, metric_dict)
    
    def close(self):
        """Close the tensorboard writer"""
        self.writer.close()

def log_training_info(config, logger=None):
    """
    Log training configuration information
    
    Args:
        config (dict): Training configuration
        logger (logging.Logger): Logger object
    """
    if logger is None:
        logger = logging.getLogger()
    
    logger.info("=" * 50)
    logger.info("TRAINING CONFIGURATION")
    logger.info("=" * 50)
    
    # Log model architecture
    logger.info("Model Architecture:")
    for key, value in config['model'].items():
        if isinstance(value, dict):
            logger.info(f"  {key}:")
            for k, v in value.items():
                logger.info(f"    {k}: {v}")
        else:
            logger.info(f"  {key}: {value}")
    
    # Log training parameters
    logger.info("Training Parameters:")
    for key, value in config['training'].items():
        if isinstance(value, dict):
            logger.info(f"  {key}:")
            for k, v in value.items():
                logger.info(f"    {k}: {v}")
        else:
            logger.info(f"  {key}: {value}")
    
    # Log evaluation metrics
    logger.info("Evaluation Metrics:")
    for key, value in config['evaluation'].items():
        logger.info(f"  {key}: {value}")
    
    logger.info("=" * 50)

def log_epoch_metrics(epoch, train_metrics, val_metrics, logger=None, tb_logger=None):
    """
    Log metrics for the current epoch
    
    Args:
        epoch (int): Current epoch
        train_metrics (dict): Training metrics
        val_metrics (dict): Validation metrics
        logger (logging.Logger): Logger object
        tb_logger (TensorboardLogger): Tensorboard logger
    """
    if logger is None:
        logger = logging.getLogger()
    
    # Log to text logger
    logger.info("-" * 50)
    logger.info(f"Epoch {epoch} Results:")
    
    # Format training metrics
    train_msg = "Training: "
    for key, value in train_metrics.items():
        train_msg += f"{key}={value:.4f} "
    logger.info(train_msg)
    
    # Format validation metrics
    val_msg = "Validation: "
    for key, value in val_metrics.items():
        val_msg += f"{key}={value:.4f} "
    logger.info(val_msg)
    
    # Log to tensorboard if available
    if tb_logger is not None:
        # Log each metric separately
        for key, value in train_metrics.items():
            tb_logger.log_scalar(f"train/{key}", value, epoch)
        
        for key, value in val_metrics.items():
            tb_logger.log_scalar(f"val/{key}", value, epoch)
        
        # Log loss comparison
        if 'loss' in train_metrics and 'loss' in val_metrics:
            tb_logger.log_scalars('loss', {
                'train': train_metrics['loss'],
                'val': val_metrics['loss']
            }, epoch)
        
        # Log accuracy comparison
        if 'accuracy' in train_metrics and 'accuracy' in val_metrics:
            tb_logger.log_scalars('accuracy', {
                'train': train_metrics['accuracy'],
                'val': val_metrics['accuracy']
            }, epoch)

def log_inference_results(users, items, scores, top_k=10, output_file=None):
    """
    Log inference results (top-k recommendations for each user)
    
    Args:
        users (list): List of user IDs
        items (list): List of item IDs
        scores (numpy.ndarray): Scores for each user-item pair
        top_k (int): Number of top recommendations to log
        output_file (str): Path to save results to CSV
    
    Returns:
        pandas.DataFrame: DataFrame with recommendations
    """
    results = []
    unique_users = set(users)
    
    for user in unique_users:
        # Get indices where this user appears
        user_idx = [i for i, u in enumerate(users) if u == user]
        
        # Get items and scores for this user
        user_items = [items[i] for i in user_idx]
        user_scores = scores[user_idx]
        
        # Sort by score (descending)
        sorted_idx = np.argsort(-user_scores)
        
        # Take top-k
        top_items = [user_items[i] for i in sorted_idx[:top_k]]
        top_scores = [user_scores[i] for i in sorted_idx[:top_k]]
        
        # Add to results
        for rank, (item, score) in enumerate(zip(top_items, top_scores)):
            results.append({
                'user_id': user,
                'item_id': item,
                'score': score,
                'rank': rank + 1
            })
    
    # Create DataFrame
    df = pd.DataFrame(results)
    
    # Log to file if specified
    if output_file:
        df.to_csv(output_file, index=False)
        logging.info(f"Saved recommendations to {output_file}")
    
    return df 