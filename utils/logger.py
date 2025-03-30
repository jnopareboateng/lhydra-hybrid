import logging
import os
import sys
import time
from datetime import datetime
import traceback
import yaml
import inspect

class LhydraLogger:
    """
    A comprehensive logger for the Lhydra Hybrid Music Recommender System.
    This logger provides detailed tracking of program execution, function calls,
    performance metrics, and errors across all modules of the system.
    """
    
    # Singleton pattern to ensure only one logger instance
    _instance = None
    
    def __new__(cls, log_dir="logs", log_level=logging.INFO, config_file=None):
        if cls._instance is None:
            cls._instance = super(LhydraLogger, cls).__new__(cls)
            cls._instance._initialized = False
        return cls._instance
    
    def __init__(self, log_dir="logs", log_level=logging.INFO, config_file=None):
        if self._initialized:
            return
            
        self.log_dir = log_dir
        self.log_level = log_level
        self.start_time = time.time()
        
        # Create log directory if it doesn't exist
        os.makedirs(log_dir, exist_ok=True)
        
        # Set up the logger
        self.logger = logging.getLogger("Lhydra")
        self.logger.setLevel(log_level)
        self.logger.propagate = False
        
        # Clear any existing handlers
        if self.logger.handlers:
            self.logger.handlers.clear()
        
        # Create timestamp for log files
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # File handler for detailed logs
        file_handler = logging.FileHandler(
            os.path.join(log_dir, f"lhydra_{timestamp}.log")
        )
        
        # Console handler
        console_handler = logging.StreamHandler(sys.stdout)
        
        # Define formatter
        formatter = logging.Formatter(
            "[%(asctime)s] [%(levelname)s] [%(module)s:%(funcName)s:%(lineno)d] - %(message)s"
        )
        
        # Set formatters
        file_handler.setFormatter(formatter)
        console_handler.setFormatter(formatter)
        
        # Add handlers
        self.logger.addHandler(file_handler)
        self.logger.addHandler(console_handler)
        
        # Load config if provided
        self.config = None
        if config_file and os.path.exists(config_file):
            with open(config_file, 'r') as f:
                self.config = yaml.safe_load(f)
                self.logger.info(f"Loaded configuration from {config_file}")
        
        self.logger.info(f"Logger initialized. Log directory: {os.path.abspath(log_dir)}")
        self._initialized = True
    
    def log_function_call(self, func_name=None, args=None, kwargs=None):
        """Log a function call with its arguments"""
        if func_name is None:
            # Get calling function's name if not provided
            frame = inspect.currentframe().f_back
            func_name = frame.f_code.co_name
        
        args_str = str(args) if args else "None"
        kwargs_str = str(kwargs) if kwargs else "None"
        
        self.logger.debug(f"FUNCTION CALL: {func_name} - Args: {args_str}, Kwargs: {kwargs_str}")
        
    def log_data_stats(self, dataset_name, data_shape, feature_names=None, missing_values=None):
        """Log statistics about a dataset"""
        self.logger.info(f"DATASET: {dataset_name} - Shape: {data_shape}")
        
        if feature_names:
            self.logger.debug(f"Features: {feature_names}")
        
        if missing_values:
            self.logger.info(f"Missing values: {missing_values}")
    
    def log_model_info(self, model_name, model_params, architecture=None):
        """Log information about a model"""
        self.logger.info(f"MODEL: {model_name} - Parameters: {model_params}")
        
        if architecture:
            self.logger.debug(f"Architecture: {architecture}")
    
    def log_training_metrics(self, epoch, metrics, validation=False):
        """Log training or validation metrics"""
        prefix = "VALIDATION" if validation else "TRAINING"
        metrics_str = ", ".join([f"{k}: {v:.4f}" for k, v in metrics.items()])
        self.logger.info(f"{prefix} METRICS - Epoch {epoch}: {metrics_str}")
    
    def log_prediction(self, input_data, predictions, actual=None):
        """Log prediction details"""
        self.logger.debug(f"PREDICTION - Input: {input_data[:10]}... Predicted: {predictions[:10]}...")
        
        if actual is not None:
            self.logger.debug(f"Actual: {actual[:10]}...")
    
    def log_error(self, error_msg, exc_info=None):
        """Log an error with stack trace"""
        self.logger.error(f"ERROR: {error_msg}")
        
        if exc_info:
            tb = traceback.format_exception(type(exc_info), exc_info, exc_info.__traceback__)
            self.logger.error(f"Traceback: {''.join(tb)}")
    
    def log_execution_time(self, start_time=None, operation=None):
        """Log execution time of an operation"""
        if start_time is None:
            start_time = self.start_time
            
        elapsed = time.time() - start_time
        
        if operation:
            self.logger.info(f"EXECUTION TIME - {operation}: {elapsed:.2f} seconds")
        else:
            self.logger.info(f"TOTAL EXECUTION TIME: {elapsed:.2f} seconds")
    
    def log_hyperparameters(self, hyperparams):
        """Log hyperparameters used for training"""
        self.logger.info(f"HYPERPARAMETERS: {hyperparams}")
    
    def log_file_access(self, filename, operation):
        """Log file access operations"""
        self.logger.debug(f"FILE ACCESS - {operation}: {filename}")
    
    def log_system_info(self, info):
        """Log system information (GPU, memory, etc.)"""
        self.logger.info(f"SYSTEM INFO: {info}")
    
    def log_experiment_start(self, experiment_name, description=None):
        """Log the start of an experiment"""
        self.logger.info(f"EXPERIMENT START: {experiment_name}")
        
        if description:
            self.logger.info(f"Description: {description}")
    
    def log_experiment_end(self, experiment_name, results=None):
        """Log the end of an experiment"""
        self.logger.info(f"EXPERIMENT END: {experiment_name}")
        
        if results:
            results_str = ", ".join([f"{k}: {v:.4f}" if isinstance(v, float) else f"{k}: {v}" 
                                   for k, v in results.items()])
            self.logger.info(f"Results: {results_str}")
    
    def info(self, message):
        """Log an info message"""
        self.logger.info(message)
    
    def debug(self, message):
        """Log a debug message"""
        self.logger.debug(message)
    
    def warning(self, message):
        """Log a warning message"""
        self.logger.warning(message)
    
    def error(self, message, exc_info=None):
        """Log an error message"""
        if exc_info:
            self.logger.error(message, exc_info=exc_info)
        else:
            self.logger.error(message)
    
    def critical(self, message, exc_info=None):
        """Log a critical message"""
        if exc_info:
            self.logger.critical(message, exc_info=exc_info)
        else:
            self.logger.critical(message)

# Function decorator for logging function calls
def log_function(logger=None):
    """Decorator to log function calls."""
    def decorator(func):
        def wrapper(*args, **kwargs):
            nonlocal logger
            if logger is None:
                logger = LhydraLogger()
            
            # Log function call
            logger.log_function_call(func.__name__, args, kwargs)
            
            # Track execution time
            start_time = time.time()
            
            try:
                result = func(*args, **kwargs)
                # Log execution time
                elapsed = time.time() - start_time
                logger.debug(f"Function {func.__name__} executed in {elapsed:.4f} seconds")
                return result
            except Exception as e:
                logger.error(f"Error in {func.__name__}: {str(e)}", exc_info=e)
                raise
                
        return wrapper
    return decorator 