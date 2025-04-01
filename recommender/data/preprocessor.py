"""
Data preprocessing module for the music recommender system.

This module handles loading data, preprocessing features, engineering new features,
and preparing the data for model training and inference.
"""

import os
import yaml
import logging
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Union, Optional
from sklearn.preprocessing import StandardScaler, MinMaxScaler, OneHotEncoder
from sklearn.model_selection import train_test_split
import joblib
from datetime import datetime

# Configure logging
# First, determine the module's directory and create an absolute path for logs
MODULE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(MODULE_DIR, '..'))
LOG_DIR = os.path.join(PROJECT_ROOT, 'logs')

# Create logs directory if it doesn't exist
os.makedirs(LOG_DIR, exist_ok=True)

# Now set up logging with the absolute path
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        # Adding a file handler for log persistence with absolute path
        logging.FileHandler(os.path.join(LOG_DIR, 'preprocessor.log'), mode='a')
    ]
)
logger = logging.getLogger(__name__)


class FeaturePreprocessor:
    """
    Preprocesses features for the music recommender system based on configuration.
    
    This class handles:
    - Loading and parsing configuration
    - Data loading and validation
    - Feature extraction and transformation
    - Feature engineering
    - Train/validation/test splitting
    - Saving processed data and preprocessing objects
    """
    
    def __init__(self, config_path: str):
        """
        Initialize the preprocessor with the given configuration.
        
        Args:
            config_path: Path to the feature configuration YAML file
        """
        self.config_path = config_path
        self.config = self._load_config(config_path)
        
        # Initialize preprocessing objects
        self.scalers = {}
        self.encoders = {}
        self.feature_stats = {}
        self.categorical_mappings = {}
        
        logger.info(f"Initialized FeaturePreprocessor with config from {config_path}")
        
    def _load_config(self, config_path: str) -> Dict:
        """
        Load and parse the YAML configuration file.
        
        Args:
            config_path: Path to the configuration file
            
        Returns:
            Dict containing configuration settings
        """
        try:
            with open(config_path, 'r') as file:
                config = yaml.safe_load(file)
            logger.info(f"Successfully loaded configuration from {config_path}")
            return config
        except Exception as e:
            logger.error(f"Failed to load configuration from {config_path}: {str(e)}")
            raise
    
    def load_data(self, data_path: str) -> pd.DataFrame:
        """
        Load data from the specified path.
        
        Args:
            data_path: Path to the data file (CSV)
            
        Returns:
            DataFrame containing the loaded data
        """
        try:
            data = pd.read_csv(data_path)
            logger.info(f"Successfully loaded data from {data_path}: {data.shape[0]} rows, {data.shape[1]} columns")
            return data
        except Exception as e:
            logger.error(f"Failed to load data from {data_path}: {str(e)}")
            raise

    def _handle_missing_values(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Handle missing values in the data according to configuration.
        
        Args:
            data: DataFrame with possibly missing values
            
        Returns:
            DataFrame with handled missing values
        """
        logger.info("Handling missing values")
        
        # Get missing value handling configuration
        numerical_method = self.config['missing_value_handling']['numerical']['method']
        categorical_method = self.config['missing_value_handling']['categorical']['method']
        
        # Process all configured features
        for feature_type in ['user_features', 'track_features']:
            if feature_type not in self.config:
                continue
                
            for category in self.config[feature_type]:
                for feature in self.config[feature_type][category]:
                    feature_name = feature['name']
                    
                    if feature_name not in data.columns:
                        logger.warning(f"Feature {feature_name} not found in data")
                        continue
                    
                    # Handle based on feature type
                    if feature['type'] == 'numerical':
                        if numerical_method == 'median':
                            median_value = data[feature_name].median()
                            self.feature_stats[f"{feature_name}_median"] = median_value
                            # Fix for FutureWarning - avoid inplace on slice
                            data[feature_name] = data[feature_name].fillna(median_value)
                        elif numerical_method == 'mean':
                            mean_value = data[feature_name].mean()
                            self.feature_stats[f"{feature_name}_mean"] = mean_value
                            # Fix for FutureWarning - avoid inplace on slice
                            data[feature_name] = data[feature_name].fillna(mean_value)
                        elif numerical_method == 'zero':
                            # Fix for FutureWarning - avoid inplace on slice
                            data[feature_name] = data[feature_name].fillna(0)
                    
                    elif feature['type'] == 'categorical':
                        if categorical_method == 'unknown_category':
                            # Fix for FutureWarning - avoid inplace on slice
                            data[feature_name] = data[feature_name].fillna('unknown')
                        elif categorical_method == 'most_frequent':
                            most_freq = data[feature_name].mode()[0]
                            self.feature_stats[f"{feature_name}_most_frequent"] = most_freq
                            # Fix for FutureWarning - avoid inplace on slice
                            data[feature_name] = data[feature_name].fillna(most_freq)
        
        logger.info("Completed missing value handling")
        return data

    def _treat_outliers(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Handle outliers in numerical features according to configuration.
        
        Args:
            data: DataFrame with possibly outlier values
            
        Returns:
            DataFrame with treated outliers
        """
        logger.info("Treating outliers in numerical features")
        
        method = self.config['outlier_treatment']['method']
        threshold = self.config['outlier_treatment']['cap_threshold']
        
        # Process all configured numerical features
        for feature_type in ['user_features', 'track_features']:
            if feature_type not in self.config:
                continue
                
            for category in self.config[feature_type]:
                for feature in self.config[feature_type][category]:
                    if feature['type'] != 'numerical':
                        continue
                        
                    feature_name = feature['name']
                    
                    if feature_name not in data.columns:
                        logger.warning(f"Feature {feature_name} not found in data")
                        continue
                    
                    # Skip if preprocessing doesn't specify outlier treatment
                    if 'preprocessing' not in feature or 'outlier_treatment' not in feature['preprocessing']:
                        continue
                    
                    if feature['preprocessing'].get('outlier_treatment') == 'cap_3std':
                        # Calculate bounds
                        mean = data[feature_name].mean()
                        std = data[feature_name].std()
                        lower_bound = mean - threshold * std
                        upper_bound = mean + threshold * std
                        
                        # Store bounds for inference
                        self.feature_stats[f"{feature_name}_lower_bound"] = lower_bound
                        self.feature_stats[f"{feature_name}_upper_bound"] = upper_bound
                        
                        # Apply capping
                        data[feature_name] = data[feature_name].clip(lower=lower_bound, upper=upper_bound)
                        
                        logger.info(f"Capped outliers for {feature_name}: [{lower_bound}, {upper_bound}]")
        
        logger.info("Completed outlier treatment")
        return data
    
    def _normalize_numerical_features(self, data: pd.DataFrame, fit: bool = True) -> pd.DataFrame:
        """
        Normalize numerical features according to configuration.
        
        Args:
            data: DataFrame with raw numerical features
            fit: Whether to fit the scalers (True) or apply existing ones (False)
            
        Returns:
            DataFrame with normalized features
        """
        logger.info(f"{'Fitting and applying' if fit else 'Applying'} normalization to numerical features")
        
        # Process all configured numerical features
        for feature_type in ['user_features', 'track_features']:
            if feature_type not in self.config:
                continue
                
            for category in self.config[feature_type]:
                for feature in self.config[feature_type][category]:
                    if feature['type'] != 'numerical':
                        continue
                        
                    feature_name = feature['name']
                    
                    if feature_name not in data.columns:
                        logger.warning(f"Feature {feature_name} not found in data")
                        continue
                    
                    # Skip if normalization not specified
                    if 'preprocessing' not in feature or not feature['preprocessing'].get('normalize', False):
                        continue
                    
                    # Apply log transform if specified
                    if feature['preprocessing'].get('log_transform', False):
                        # Handle potential non-positive values
                        min_val = data[feature_name].min()
                        if min_val <= 0:
                            offset = abs(min_val) + 1
                            data[feature_name] = data[feature_name] + offset
                            self.feature_stats[f"{feature_name}_log_offset"] = offset
                            
                        # Apply log transform
                        data[feature_name] = np.log1p(data[feature_name])
                        logger.info(f"Applied log transform to {feature_name}")
                    
                    # Normalize the feature
                    if fit:
                        scaler = MinMaxScaler()
                        data[feature_name] = scaler.fit_transform(data[[feature_name]])
                        self.scalers[feature_name] = scaler
                    else:
                        if feature_name in self.scalers:
                            data[feature_name] = self.scalers[feature_name].transform(data[[feature_name]])
                        else:
                            logger.warning(f"No scaler found for {feature_name} during transform")
                    
                    logger.info(f"Normalized feature: {feature_name}")
        
        logger.info("Completed numerical feature normalization")
        return data
    
    def _encode_categorical_features(self, data: pd.DataFrame, fit: bool = True) -> pd.DataFrame:
        """
        Encode categorical features according to configuration.
        
        Args:
            data: DataFrame with raw categorical features
            fit: Whether to fit the encoders (True) or apply existing ones (False)
            
        Returns:
            DataFrame with encoded features
        """
        logger.info(f"{'Fitting and applying' if fit else 'Applying'} encoding to categorical features")
        
        # Process all configured categorical features
        for feature_type in ['user_features', 'track_features']:
            if feature_type not in self.config:
                continue
                
            for category in self.config[feature_type]:
                for feature in self.config[feature_type][category]:
                    if feature['type'] != 'categorical':
                        continue
                        
                    feature_name = feature['name']
                    
                    if feature_name not in data.columns:
                        logger.warning(f"Feature {feature_name} not found in data")
                        continue
                    
                    # Skip if encoding not specified
                    if 'preprocessing' not in feature or 'encoding' not in feature['preprocessing']:
                        continue
                    
                    encoding_method = feature['preprocessing']['encoding']
                    
                    if encoding_method == 'one_hot':
                        # Handle high-cardinality features
                        if 'min_frequency' in feature['preprocessing']:
                            min_freq = feature['preprocessing']['min_frequency']
                            value_counts = data[feature_name].value_counts()
                            rare_categories = value_counts[value_counts < min_freq].index.tolist()
                            
                            # Replace rare categories with "other"
                            if rare_categories:
                                data[feature_name] = data[feature_name].replace(rare_categories, 'other')
                                logger.info(f"Replaced {len(rare_categories)} rare categories in {feature_name} with 'other'")
                        
                        if fit:
                            # Fit one-hot encoder
                            encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
                            encoder.fit(data[[feature_name]])
                            self.encoders[feature_name] = encoder
                            
                            # Store category mapping for reference
                            self.categorical_mappings[feature_name] = {
                                'categories': encoder.categories_[0].tolist()
                            }
                        
                        # Transform the data
                        if feature_name in self.encoders:
                            encoded = self.encoders[feature_name].transform(data[[feature_name]])
                            encoded_cols = [f"{feature_name}_{cat}" for cat in self.categorical_mappings[feature_name]['categories']]
                            
                            # Create a DataFrame with encoded columns to avoid fragmentation
                            encoded_df = pd.DataFrame(
                                encoded, 
                                columns=encoded_cols,
                                index=data.index
                            )
                            
                            # Use concat instead of adding columns one by one
                            data = pd.concat([data, encoded_df], axis=1)
                            
                            logger.info(f"One-hot encoded {feature_name} into {len(encoded_cols)} categories")
                        else:
                            logger.warning(f"No encoder found for {feature_name} during transform")
                    
                    elif encoding_method == 'entity_embedding':
                        # For entity embeddings, we'll create a mapping to integers
                        # The actual embeddings will be handled by the model
                        if fit:
                            # Handle high-cardinality features
                            if 'min_frequency' in feature['preprocessing']:
                                min_freq = feature['preprocessing']['min_frequency']
                                value_counts = data[feature_name].value_counts()
                                frequent_categories = value_counts[value_counts >= min_freq].index.tolist()
                                
                                if 'max_categories' in feature['preprocessing']:
                                    max_cats = feature['preprocessing']['max_categories']
                                    if len(frequent_categories) > max_cats:
                                        frequent_categories = value_counts.nlargest(max_cats).index.tolist()
                                
                                # Create mapping with frequent categories + unknown
                                mapping = {cat: idx+1 for idx, cat in enumerate(frequent_categories)}
                                mapping['unknown'] = 0  # Reserve 0 for unknown/rare
                                
                                self.categorical_mappings[feature_name] = {
                                    'mapping': mapping,
                                    'embedding_dim': feature['preprocessing'].get('embedding_dim', 16)
                                }
                                
                                logger.info(f"Created entity mapping for {feature_name} with {len(mapping)-1} categories")
                            else:
                                # Create mapping for all categories
                                categories = data[feature_name].unique().tolist()
                                mapping = {cat: idx+1 for idx, cat in enumerate(categories)}
                                mapping['unknown'] = 0
                                
                                self.categorical_mappings[feature_name] = {
                                    'mapping': mapping,
                                    'embedding_dim': feature['preprocessing'].get('embedding_dim', 16)
                                }
                                
                                logger.info(f"Created entity mapping for {feature_name} with {len(mapping)-1} categories")
                        
                        # Apply the mapping
                        if feature_name in self.categorical_mappings:
                            mapping = self.categorical_mappings[feature_name]['mapping']
                            data[f"{feature_name}_id"] = data[feature_name].map(
                                lambda x: mapping.get(x, 0)  # Default to 0 (unknown) if not in mapping
                            )
                            
                            logger.info(f"Applied entity mapping to {feature_name}")
                        else:
                            logger.warning(f"No mapping found for {feature_name} during transform")
        
        logger.info("Completed categorical feature encoding")
        return data
    
    def _engineer_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Apply feature engineering according to configuration.
        
        Args:
            data: DataFrame with preprocessed features
            
        Returns:
            DataFrame with additional engineered features
        """
        logger.info("Applying feature engineering")
        
        # Apply compatibility features if enabled
        if self.config['feature_engineering']['compatibility_features']['enabled']:
            for feature_def in self.config['feature_engineering']['compatibility_features']['features']:
                user_feature, track_feature = feature_def['feature_pair']
                method = feature_def['method']
                
                if user_feature not in data.columns or track_feature not in data.columns:
                    logger.warning(f"Cannot create compatibility feature: {user_feature} or {track_feature} not found")
                    continue
                
                if method == 'absolute_difference':
                    feature_name = f"{track_feature}_compatibility"
                    data[feature_name] = abs(data[user_feature] - data[track_feature])
                    logger.info(f"Created compatibility feature: {feature_name}")
        
        # Apply temporal features if enabled
        if self.config['feature_engineering']['temporal_features']['enabled']:
            for feature_def in self.config['feature_engineering']['temporal_features']['features']:
                name = feature_def['name']
                based_on = feature_def['based_on']
                method = feature_def['method']
                
                if based_on not in data.columns:
                    logger.warning(f"Cannot create temporal feature: {based_on} not found")
                    continue
                
                if method == 'current_year_minus_release_year':
                    current_year = datetime.now().year
                    data[name] = current_year - data[based_on]
                    logger.info(f"Created temporal feature: {name}")
                
                elif method == 'binary_threshold':
                    threshold = feature_def['threshold']
                    current_year = datetime.now().year
                    data[name] = (current_year - data[based_on] <= threshold).astype(int)
                    logger.info(f"Created binary temporal feature: {name}")
        
        # Apply interaction features if enabled
        if self.config['feature_engineering']['interaction_features']['enabled']:
            for feature_def in self.config['feature_engineering']['interaction_features']['features']:
                name = feature_def['name']
                feature_pair = feature_def['feature_pair']
                method = feature_def['method']
                
                if feature_pair[0] not in data.columns or feature_pair[1] not in data.columns:
                    logger.warning(f"Cannot create interaction feature: {feature_pair[0]} or {feature_pair[1]} not found")
                    continue
                
                if method == 'group_statistics':
                    # This is a placeholder for more complex interactions
                    # In a real implementation, this could calculate average preferences by age group for each genre
                    # For simplicity, we'll just create a placeholder column
                    data[name] = 1
                    logger.info(f"Created placeholder for interaction feature: {name}")
        
        logger.info("Completed feature engineering")
        return data
    
    def _prepare_target_variable(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Prepare the target variable for model training.
        
        For a recommender system, this typically involves creating a binary label
        indicating whether a user interacted with an item (or the interaction strength).
        
        Args:
            data: DataFrame with features
            
        Returns:
            DataFrame with prepared target variable
        """
        logger.info("Preparing target variable")
        
        # Create binary engagement target from playcount
        # Consider playcount > 1 as positive engagement
        data['target'] = (data['playcount'] > 1).astype(int)
        
        # Could also create a multi-class target for different engagement levels
        # data['engagement_level'] = pd.cut(
        #     data['playcount'], 
        #     bins=[0, 1, 5, 10, float('inf')], 
        #     labels=[0, 1, 2, 3]
        # ).astype(int)
        
        logger.info(f"Created target variable: {data['target'].mean():.2%} positive rate")
        return data
    
    def _split_data(self, data: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Split data into train, validation, and test sets.
        
        Args:
            data: Preprocessed DataFrame with features and target
            
        Returns:
            Tuple of (train_data, validation_data, test_data)
        """
        logger.info("Splitting data into train, validation, and test sets")
        
        train_ratio = self.config['data_split']['train_ratio']
        validation_ratio = self.config['data_split']['validation_ratio']
        test_ratio = self.config['data_split']['test_ratio']
        seed = self.config['data_split']['random_seed']
        method = self.config['data_split']['splitting_method']
        
        if method == 'user_stratified':
            # Split by users to avoid leakage
            users = data['user_id'].unique()
            
            # First split: train and temp (validation + test)
            train_users, temp_users = train_test_split(
                users, 
                test_size=(validation_ratio + test_ratio),
                random_state=seed
            )
            
            # Second split: validation and test from temp
            validation_ratio_adjusted = validation_ratio / (validation_ratio + test_ratio)
            validation_users, test_users = train_test_split(
                temp_users,
                test_size=(1 - validation_ratio_adjusted),
                random_state=seed
            )
            
            # Create the splits
            train_data = data[data['user_id'].isin(train_users)].copy()
            validation_data = data[data['user_id'].isin(validation_users)].copy()
            test_data = data[data['user_id'].isin(test_users)].copy()
            
            logger.info(f"Split data: Train={len(train_data)} rows, Validation={len(validation_data)} rows, Test={len(test_data)} rows")
            logger.info(f"Users: Train={len(train_users)}, Validation={len(validation_users)}, Test={len(test_users)}")
        
        else:  # Simple random split
            # First split: train and temp (validation + test)
            train_data, temp_data = train_test_split(
                data, 
                test_size=(validation_ratio + test_ratio),
                random_state=seed
            )
            
            # Second split: validation and test from temp
            validation_ratio_adjusted = validation_ratio / (validation_ratio + test_ratio)
            validation_data, test_data = train_test_split(
                temp_data,
                test_size=(1 - validation_ratio_adjusted),
                random_state=seed
            )
            
            logger.info(f"Split data: Train={len(train_data)} rows, Validation={len(validation_data)} rows, Test={len(test_data)} rows")
        
        return train_data, validation_data, test_data
    
    def preprocess(self, data_path: str, output_dir: str) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Preprocess the data according to configuration and save results.
        
        This is the main method that orchestrates the entire preprocessing pipeline.
        
        Args:
            data_path: Path to the input data file
            output_dir: Directory to save preprocessed data and objects
            
        Returns:
            Tuple of (train_data, validation_data, test_data)
        """
        logger.info(f"Starting preprocessing pipeline for {data_path}")
        
        # Load the data
        data = self.load_data(data_path)
        
        # Basic validation
        required_cols = ['track_id', 'user_id', 'playcount']
        for col in required_cols:
            if col not in data.columns:
                raise ValueError(f"Required column '{col}' not found in data")
        
        # Apply preprocessing steps
        data = self._handle_missing_values(data)
        data = self._treat_outliers(data)
        data = self._normalize_numerical_features(data, fit=True)
        data = self._encode_categorical_features(data, fit=True)
        data = self._engineer_features(data)
        data = self._prepare_target_variable(data)
        
        # Split the data
        train_data, validation_data, test_data = self._split_data(data)
        
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Save preprocessed data
        train_data.to_csv(os.path.join(output_dir, 'train_data.csv'), index=False)
        validation_data.to_csv(os.path.join(output_dir, 'validation_data.csv'), index=False)
        test_data.to_csv(os.path.join(output_dir, 'test_data.csv'), index=False)
        
        # Save preprocessor objects for inference
        preprocessor_path = os.path.join(output_dir, 'preprocessor.joblib')
        joblib.dump({
            'scalers': self.scalers,
            'encoders': self.encoders,
            'feature_stats': self.feature_stats,
            'categorical_mappings': self.categorical_mappings,
            'config': self.config
        }, preprocessor_path)
        
        logger.info(f"Saved preprocessor objects to {preprocessor_path}")
        logger.info("Preprocessing complete")
        
        return train_data, validation_data, test_data
    
    def transform(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Apply preprocessing transformations to new data.
        
        This method is used during inference to apply the same transformations
        that were learned during training.
        
        Args:
            data: Raw DataFrame to transform
            
        Returns:
            Transformed DataFrame
        """
        logger.info("Transforming new data with existing preprocessor")
        
        # Apply preprocessing steps without fitting
        data = self._handle_missing_values(data)
        data = self._treat_outliers(data)
        data = self._normalize_numerical_features(data, fit=False)
        data = self._encode_categorical_features(data, fit=False)
        data = self._engineer_features(data)
        
        logger.info("Transformation complete")
        return data
    
    @classmethod
    def load(cls, preprocessor_path: str) -> 'FeaturePreprocessor':
        """
        Load a saved preprocessor.
        
        Args:
            preprocessor_path: Path to the saved preprocessor
            
        Returns:
            Loaded FeaturePreprocessor instance
        """
        logger.info(f"Loading preprocessor from {preprocessor_path}")
        
        # Load saved objects
        saved_data = joblib.load(preprocessor_path)
        
        # Create a new instance with the saved config
        config_path = saved_data.get('config_path', None)
        instance = cls(config_path) if config_path else cls.__new__(cls)
        
        # Restore saved attributes
        instance.config = saved_data['config']
        instance.scalers = saved_data['scalers']
        instance.encoders = saved_data['encoders']
        instance.feature_stats = saved_data['feature_stats']
        instance.categorical_mappings = saved_data['categorical_mappings']
        
        logger.info("Loaded preprocessor successfully")
        return instance


# If run as a script
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Preprocess data for the music recommender system')
    parser.add_argument('--input', type=str, required=True, help='Path to input data CSV file')
    parser.add_argument('--output', type=str, required=True, help='Directory to save preprocessed data')
    parser.add_argument('--config', type=str, required=True, help='Path to feature configuration YAML file')
    
    args = parser.parse_args()
    
    # Initialize and run preprocessor
    preprocessor = FeaturePreprocessor(args.config)
    train_data, validation_data, test_data = preprocessor.preprocess(args.input, args.output)
    
    print(f"Preprocessing complete. Data saved to {args.output}")
    print(f"Train: {len(train_data)} rows, Validation: {len(validation_data)} rows, Test: {len(test_data)} rows") 