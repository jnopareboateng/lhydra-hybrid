#!/usr/bin/env python
"""
Script to preprocess data for the music recommender system.

This script loads raw data, applies preprocessing according to configuration,
and saves the preprocessed data for model training and evaluation.
"""

import os
import sys
import argparse
import logging
from pathlib import Path
from datetime import datetime

# Add project root to path to allow imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from data.preprocessor import FeaturePreprocessor

def setup_logging(log_level, log_dir='logs'):
    """Set up logging with both console and file output.
    
    Args:
        log_level: Logging level (DEBUG, INFO, etc.)
        log_dir: Directory to save log files
    """
    # Create log directory if it doesn't exist
    os.makedirs(log_dir, exist_ok=True)
    
    # Create timestamped log file name
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(log_dir, f"preprocessing_{timestamp}.log")
    
    # Reset root logger
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)
    
    # Configure logging - ensure we capture all messages
    numeric_level = getattr(logging, log_level)
    logging.root.setLevel(numeric_level)
    
    # Create formatter
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    
    # Create and add console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(numeric_level)
    console_handler.setFormatter(formatter)
    logging.root.addHandler(console_handler)
    
    # Create and add file handler
    file_handler = logging.FileHandler(log_file, 'w')  # 'w' mode to create a new file
    file_handler.setLevel(numeric_level)
    file_handler.setFormatter(formatter)
    logging.root.addHandler(file_handler)
    
    # Log the initial message to verify logging works
    logging.info(f"Logging initialized. Log file: {log_file}")
    
    return log_file

def main():
    """Main function to run preprocessing pipeline."""
    parser = argparse.ArgumentParser(description='Preprocess data for the music recommender system')
    parser.add_argument('--input', type=str, required=True, 
                        help='Path to input data CSV file')
    parser.add_argument('--output', type=str, required=True, 
                        help='Directory to save preprocessed data')
    parser.add_argument('--config', type=str, required=True, 
                        help='Path to feature configuration YAML file')
    parser.add_argument('--log-level', type=str, choices=['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'],
                        default='INFO', help='Set the logging level')
    parser.add_argument('--log-dir', type=str, default='logs',
                        help='Directory to save log files')
    
    args = parser.parse_args()
    
    # Set up logging
    log_file = setup_logging(args.log_level, args.log_dir)
    logger = logging.getLogger(__name__)
    
    try:
        logger.info("Starting preprocessing pipeline")
        logger.info(f"Input data: {args.input}")
        logger.info(f"Output directory: {args.output}")
        logger.info(f"Configuration: {args.config}")
        
        # Create output directory if it doesn't exist
        output_dir = Path(args.output)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize and run preprocessor
        preprocessor = FeaturePreprocessor(args.config)
        train_data, validation_data, test_data = preprocessor.preprocess(args.input, args.output)
        
        # Log results
        logger.info("Preprocessing complete")
        logger.info(f"Train set: {len(train_data)} rows")
        logger.info(f"Validation set: {len(validation_data)} rows")
        logger.info(f"Test set: {len(test_data)} rows")
        
        # Copy log file to output directory for reference
        output_log_file = os.path.join(args.output, "preprocessing.log")
        with open(log_file, 'r') as src, open(output_log_file, 'w') as dst:
            dst.write(src.read())
        logger.info(f"Log file copied to: {output_log_file}")
        
        # Return success
        return 0
        
    except Exception as e:
        logger.error(f"Error during preprocessing: {str(e)}", exc_info=True)
        return 1

if __name__ == "__main__":
    sys.exit(main()) 