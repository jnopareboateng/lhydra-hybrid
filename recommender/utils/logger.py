"""
Logging utilities for the music recommender system.

This module provides a consistent logging configuration and helper functions
for logging throughout the project.
"""

import os
import logging
import sys
from pathlib import Path
from typing import Optional, Union, Dict, Any
import yaml
import time
import functools
import inspect


def setup_logger(name: str, log_file: Optional[str] = None, 
                 level: int = logging.INFO, 
                 log_to_console: bool = True,
                 log_format: str = '%(asctime)s - %(name)s - %(levelname)s - %(message)s') -> logging.Logger:
    """
    Set up a logger with consistent formatting.
    
    Args:
        name: Name of the logger
        log_file: Path to the log file (if None, no file logging)
        level: Logging level
        log_to_console: Whether to log to console
        log_format: Log message format
        
    Returns:
        Configured logger instance
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)
    formatter = logging.Formatter(log_format)
    
    # Clear existing handlers
    logger.handlers = []
    
    # Add console handler if requested
    if log_to_console:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
    
    # Add file handler if log_file provided
    if log_file:
        # Create directory if needed
        log_dir = os.path.dirname(log_file)
        if log_dir:
            os.makedirs(log_dir, exist_ok=True)
            
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    return logger


def log_execution_time(logger: Optional[logging.Logger] = None,
                       level: int = logging.INFO) -> callable:
    """
    Decorator to log function execution time.
    
    Args:
        logger: Logger to use (if None, creates a new one)
        level: Logging level for the timing message
        
    Returns:
        Decorator function
    """
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            func_logger = logger or logging.getLogger(func.__module__)
            
            # Get function details
            func_name = func.__name__
            module_name = func.__module__
            
            # Log start message
            start_msg = f"Starting {func_name} in {module_name}"
            func_logger.log(level, start_msg)
            
            # Track time
            start_time = time.time()
            
            # Execute function
            result = func(*args, **kwargs)
            
            # Calculate execution time
            execution_time = time.time() - start_time
            
            # Log completion message
            end_msg = f"Completed {func_name} in {module_name} (took {execution_time:.2f} seconds)"
            func_logger.log(level, end_msg)
            
            return result
        return wrapper
    return decorator


class MetricsLogger:
    """
    Logger for tracking and recording model training and evaluation metrics.
    
    This class provides methods for logging metrics during training and
    evaluation, and writes them to a structured log file.
    """
    
    def __init__(self, log_dir: str, experiment_name: str):
        """
        Initialize the metrics logger.
        
        Args:
            log_dir: Directory to store metric logs
            experiment_name: Name of the experiment
        """
        self.log_dir = Path(log_dir)
        self.experiment_name = experiment_name
        
        # Create log directory
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        # Set up main logger
        self.logger = setup_logger(
            name=f"metrics.{experiment_name}",
            log_file=str(self.log_dir / f"{experiment_name}_metrics.log")
        )
        
        # Storage for metrics
        self.epoch_metrics = {}
        self.current_epoch = 0
        self.training_start_time = None
        
        # Metrics file path
        self.metrics_file = self.log_dir / f"{experiment_name}_metrics.yaml"
        
        # Initialize metrics file if it doesn't exist
        if not self.metrics_file.exists():
            self._save_metrics({})
    
    def start_training(self) -> None:
        """Mark the start of training for timing purposes."""
        self.training_start_time = time.time()
        self.logger.info(f"Started training experiment: {self.experiment_name}")
    
    def end_training(self) -> None:
        """Mark the end of training and log total time."""
        if self.training_start_time:
            training_time = time.time() - self.training_start_time
            self.logger.info(f"Completed training experiment: {self.experiment_name}")
            self.logger.info(f"Total training time: {training_time:.2f} seconds")
            
            # Save total training time to metrics
            self._update_metrics({"total_training_time": training_time})
    
    def log_epoch_metrics(self, epoch: int, metrics: Dict[str, Any],
                         phase: str = 'train') -> None:
        """
        Log metrics for a training/validation epoch.
        
        Args:
            epoch: Epoch number
            metrics: Dictionary of metric names and values
            phase: Training phase ('train' or 'val')
        """
        self.current_epoch = epoch
        
        # Format metrics for logging
        metrics_str = ", ".join([f"{k}: {v:.4f}" for k, v in metrics.items()])
        self.logger.info(f"Epoch {epoch} ({phase}): {metrics_str}")
        
        # Update stored metrics
        epoch_key = f"epoch_{epoch}"
        phase_metrics = {f"{phase}_{k}": v for k, v in metrics.items()}
        
        if epoch_key not in self.epoch_metrics:
            self.epoch_metrics[epoch_key] = {}
        
        self.epoch_metrics[epoch_key].update(phase_metrics)
        
        # Save to file
        self._update_metrics(self.epoch_metrics)
    
    def log_evaluation_metrics(self, metrics: Dict[str, Any]) -> None:
        """
        Log metrics from model evaluation.
        
        Args:
            metrics: Dictionary of metric names and values
        """
        # Format metrics for logging
        metrics_str = ", ".join([f"{k}: {v:.4f}" for k, v in metrics.items()])
        self.logger.info(f"Evaluation: {metrics_str}")
        
        # Update stored metrics
        eval_metrics = {f"eval_{k}": v for k, v in metrics.items()}
        
        # Save to file
        self._update_metrics({"evaluation": eval_metrics})
    
    def log_hyperparameters(self, hyperparams: Dict[str, Any]) -> None:
        """
        Log model hyperparameters.
        
        Args:
            hyperparams: Dictionary of hyperparameter names and values
        """
        # Log hyperparameters
        for name, value in hyperparams.items():
            self.logger.info(f"Hyperparameter {name}: {value}")
        
        # Save to file
        self._update_metrics({"hyperparameters": hyperparams})
    
    def _save_metrics(self, metrics: Dict[str, Any]) -> None:
        """
        Save metrics to YAML file.
        
        Args:
            metrics: Dictionary of metrics to save
        """
        with open(self.metrics_file, 'w') as f:
            yaml.dump(metrics, f, default_flow_style=False)
    
    def _update_metrics(self, new_metrics: Dict[str, Any]) -> None:
        """
        Update metrics file with new values.
        
        Args:
            new_metrics: Dictionary of new metrics to add/update
        """
        # Load existing metrics
        try:
            with open(self.metrics_file, 'r') as f:
                metrics = yaml.safe_load(f) or {}
        except FileNotFoundError:
            metrics = {}
        
        # Update with new metrics
        metrics.update(new_metrics)
        
        # Save back to file
        self._save_metrics(metrics)
    
    def get_best_metric(self, metric_name: str, phase: str = 'val',
                      higher_is_better: bool = True) -> tuple:
        """
        Get the best value for a specific metric across all epochs.
        
        Args:
            metric_name: Name of the metric to find best value for
            phase: Training phase ('train' or 'val')
            higher_is_better: Whether higher values are better
            
        Returns:
            Tuple of (best_value, best_epoch)
        """
        best_value = float('-inf') if higher_is_better else float('inf')
        best_epoch = -1
        
        full_metric_name = f"{phase}_{metric_name}"
        
        for epoch_key, metrics in self.epoch_metrics.items():
            if full_metric_name in metrics:
                value = metrics[full_metric_name]
                
                if (higher_is_better and value > best_value) or \
                   (not higher_is_better and value < best_value):
                    best_value = value
                    best_epoch = int(epoch_key.split('_')[1])
        
        return best_value, best_epoch 