import logging
import pandas as pd
import numpy as np
import os
import re
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.model_selection import train_test_split
import sys
import yaml
from datetime import datetime
from typing import Optional

# Add root directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.logger import LhydraLogger, log_function

class MusicDataPreprocessor:
    """
    Preprocessor for the Lhydra Hybrid Music Recommender System.
    Handles data loading, cleaning, feature engineering, and splitting
    for the music recommendation task.
    """
    
    def __init__(self, config_path=None, logger=None):
        """
        Initialize the preprocessor with configuration settings.
        
        Args:
            config_path (str): Path to the YAML configuration file.
            logger (LhydraLogger): Logger instance for tracking operations.
        """
        self.logger = logger or LhydraLogger()
        self.logger.info("Initializing MusicDataPreprocessor")
        
        # Default configuration based on features.md recommendations
        self.config = {
            # User features
            'user_demographic_features': ['age', 'gender', 'country'],
            'user_listening_features': ['monthly_hours', 'genre_diversity', 'top_genre'],
            'user_audio_preferences': ['avg_danceability', 'avg_energy', 'avg_key', 
                                     'avg_loudness', 'avg_mode', 'avg_speechiness',
                                     'avg_acousticness', 'avg_instrumentalness',
                                     'avg_liveness', 'avg_valence', 'avg_tempo',
                                     'avg_time_signature'],
            
            # Track features
            'track_metadata_features': ['artist', 'main_genre', 'year', 'duration_ms'],
            'track_audio_features': ['danceability', 'energy', 'key', 'loudness', 'mode',
                                   'speechiness', 'acousticness', 'instrumentalness',
                                   'liveness', 'valence', 'tempo', 'time_signature'],
            
            # Target configuration
            'target_column': 'playcount',
            'target_threshold': 5,
            
            # Data splitting configuration
            'test_size': 0.2,
            'validation_size': 0.1,
            'random_state': 42,
            'embedding_dim': 32
        }
        
        # Load configuration from file if provided
        if config_path and os.path.exists(config_path):
            self.logger.log_file_access(config_path, "read")
            with open(config_path, 'r') as f:
                loaded_config = yaml.safe_load(f)
                
                # Handle nested structure from training_config.yaml
                if 'data' in loaded_config:
                    data_config = loaded_config['data']
                    
                    # Update configuration with data section
                    for key, value in data_config.items():
                        self.config[key] = value
                else:
                    # Flat configuration structure
                    self.config.update(loaded_config)
                    
                self.logger.info(f"Loaded configuration from {config_path}")
        
        # Initialize preprocessors
        self.scalers = {}
        self.encoders = {}
        self.label_encoders = {}
        
        # Store processed data
        self.data = None
        self.train_data = None
        self.val_data = None
        self.test_data = None
        
        # Create combined feature lists for easier access
        self._create_feature_lists()
        
        self.logger.info(f"MusicDataPreprocessor initialized with config: {self.config}")
    
    def _create_feature_lists(self):
        """Create combined feature lists for easier access"""
        # Check if categorical_features and numerical_features are already provided in the config
        if 'categorical_features' not in self.config or not self.config['categorical_features']:
            # Create categorical features list from user and track features
            self.config['categorical_features'] = []
            
            # Add categorical features from user demographics
            if 'gender' in self.config.get('user_demographic_features', []):
                self.config['categorical_features'].append('gender')
            if 'country' in self.config.get('user_demographic_features', []):
                self.config['categorical_features'].append('country')
                
            # Add categorical features from user listening
            if 'top_genre' in self.config.get('user_listening_features', []):
                self.config['categorical_features'].append('top_genre')
                
            # Add categorical features from track metadata
            if 'main_genre' in self.config.get('track_metadata_features', []):
                self.config['categorical_features'].append('main_genre')
            if 'artist' in self.config.get('track_metadata_features', []):
                self.config['categorical_features'].append('artist')
        
        if 'numerical_features' not in self.config or not self.config['numerical_features']:
            # Create numerical features list from user and track features
            self.config['numerical_features'] = []
            
            # Add numerical features from user demographics
            if 'age' in self.config.get('user_demographic_features', []):
                self.config['numerical_features'].append('age')
                
            # Add numerical features from user listening
            if 'monthly_hours' in self.config.get('user_listening_features', []):
                self.config['numerical_features'].append('monthly_hours')
            if 'genre_diversity' in self.config.get('user_listening_features', []):
                self.config['numerical_features'].append('genre_diversity')
                
            # Add numerical features from track metadata
            if 'year' in self.config.get('track_metadata_features', []):
                self.config['numerical_features'].append('year')
            if 'duration_ms' in self.config.get('track_metadata_features', []):
                self.config['numerical_features'].append('duration_ms')
        
        # Create all_audio_features list
        track_audio = self.config.get('track_audio_features', [])
        user_audio = self.config.get('user_audio_preferences', [])
        
        # Handle backward compatibility with old config format
        if not track_audio and 'audio_features' in self.config:
            track_audio = self.config.get('audio_features', [])
            # Update the config with the new key
            self.config['track_audio_features'] = track_audio
            
        if not user_audio and 'avg_audio_features' in self.config:
            user_audio = self.config.get('avg_audio_features', [])
            # Update the config with the new key
            self.config['user_audio_preferences'] = user_audio
            
        self.config['all_audio_features'] = list(set(track_audio + user_audio))
        
        self.logger.debug(f"Created feature lists: categorical={len(self.config['categorical_features'])}, " +
                         f"numerical={len(self.config['numerical_features'])}, " +
                         f"audio={len(self.config['all_audio_features'])}")
    
    @log_function()
    def load_data(self, file_path):
        """
        Load data from CSV file.
        
        Args:
            file_path (str): Path to the CSV file.
            
        Returns:
            pd.DataFrame: Loaded dataframe.
        """
        self.logger.log_file_access(file_path, "read")
        self.logger.info(f"Loading data from {file_path}")
        
        try:
            df = pd.read_csv(file_path)
            self.logger.log_data_stats("Raw Data", df.shape, 
                                      feature_names=list(df.columns),
                                      missing_values=df.isnull().sum().to_dict())
            self.data = df
            return df
        except Exception as e:
            self.logger.error(f"Error loading data from {file_path}: {str(e)}", exc_info=e)
            raise
    
    @log_function()
    def clean_data(self, df=None):
        """
        Clean data by handling missing values and outliers.
        
        Args:
            df (pd.DataFrame, optional): DataFrame to clean. If None, uses self.data.
            
        Returns:
            pd.DataFrame: Cleaned dataframe.
        """
        if df is None:
            df = self.data
            
        if df is None:
            self.logger.error("No data available for cleaning")
            raise ValueError("No data available for cleaning")
            
        self.logger.info("Cleaning data")
        
        # Make a copy to avoid modifying the original
        df_clean = df.copy()
        
        # Handle missing values
        numeric_cols = df_clean.select_dtypes(include=['int64', 'float64']).columns
        categorical_cols = df_clean.select_dtypes(include=['object']).columns
        
        # Fill numeric missing values with median
        for col in numeric_cols:
            if df_clean[col].isnull().any():
                median_val = df_clean[col].median()
                # Avoid inplace operation
                df_clean[col] = df_clean[col].fillna(median_val)
                self.logger.debug(f"Filled missing values in {col} with median: {median_val}")
        
        # Fill categorical missing values with mode
        for col in categorical_cols:
            if df_clean[col].isnull().any():
                mode_val = df_clean[col].mode()[0]
                # Avoid inplace operation
                df_clean[col] = df_clean[col].fillna(mode_val)
                self.logger.debug(f"Filled missing values in {col} with mode: {mode_val}")
        
        # Handle outliers for numerical columns (capping at 3 std devs)
        for col in numeric_cols:
            # Check if column is in our features of interest
            if col in self.config['numerical_features'] or col in self.config['all_audio_features']:
                mean, std = df_clean[col].mean(), df_clean[col].std()
                lower_bound, upper_bound = mean - 3*std, mean + 3*std
                
                # Handle integer vs float type compatibility
                if df_clean[col].dtype == 'int64':
                    lower_bound = int(lower_bound)
                    upper_bound = int(upper_bound)
                
                # Create a new series with capped values instead of using loc
                series = df_clean[col].copy()
                outliers_lower = series < lower_bound
                outliers_upper = series > upper_bound
                
                if outliers_lower.any() or outliers_upper.any():
                    series = series.mask(outliers_lower, lower_bound)
                    series = series.mask(outliers_upper, upper_bound)
                    df_clean[col] = series
                    self.logger.debug(f"Capped outliers in {col}: {outliers_lower.sum() + outliers_upper.sum()} values")
        
        # Log cleaning stats
        self.logger.log_data_stats("Cleaned Data", df_clean.shape, 
                                  missing_values=df_clean.isnull().sum().to_dict())
        
        self.data = df_clean
        return df_clean
    
    @log_function()
    def engineer_features(self, df=None):
        """
        Perform feature engineering to create new features based on features.md recommendations.
        
        Args:
            df (pd.DataFrame, optional): DataFrame for feature engineering. If None, uses self.data.
            
        Returns:
            pd.DataFrame: DataFrame with engineered features.
        """
        if df is None:
            df = self.data
            
        if df is None:
            self.logger.error("No data available for feature engineering")
            raise ValueError("No data available for feature engineering")
            
        self.logger.info("Engineering features")
        
        # Make a copy to avoid modifying the original
        df_engineered = df.copy()
        
        # 1. Create binary engagement target based on playcount threshold
        target_threshold = self.config['target_threshold']
        df_engineered['high_engagement'] = (df_engineered[self.config['target_column']] > target_threshold).astype(int)
        self.logger.debug(f"Created binary engagement target with threshold {target_threshold}")
        
        # 2. Process user demographic features
        if 'age' in df_engineered.columns:
            # Create age buckets as recommended
            bins = [0, 18, 25, 35, 50, float('inf')]
            labels = ['teen', 'young_adult', 'adult', 'middle_aged', 'senior']
            df_engineered['age_group'] = pd.cut(df_engineered['age'], bins=bins, labels=labels)
            self.logger.debug("Created age_group categorical feature")
        
        # 3. Process track metadata
        if 'year' in df_engineered.columns:
            current_year = datetime.now().year
            df_engineered['song_age'] = current_year - df_engineered['year']
            df_engineered['is_recent'] = (df_engineered['song_age'] <= 5).astype(int)
            self.logger.debug("Created song age and recency features")
        
        if 'duration_ms' in df_engineered.columns:
            # Convert to minutes for easier interpretation
            df_engineered['duration_min'] = df_engineered['duration_ms'] / 60000
            # Create duration categories
            bins = [0, 2, 4, 6, float('inf')]
            labels = ['short', 'medium', 'long', 'very_long']
            df_engineered['duration_category'] = pd.cut(df_engineered['duration_min'], 
                                                      bins=bins, labels=labels)
            self.logger.debug("Created duration features")
        
        # 4. Process audio features - focusing on the recommended ones from features.md
        # Handle correlated features (energy-loudness, valence-danceability)
        
        # Energy and loudness correlation (0.77)
        if 'energy' in df_engineered.columns and 'loudness' in df_engineered.columns:
            # Create interaction feature
            df_engineered['energy_loudness'] = df_engineered['energy'] * df_engineered['loudness']
            # Create ratio feature (helps to identify cases where the correlation doesn't hold)
            df_engineered['energy_loudness_ratio'] = np.where(
                df_engineered['loudness'] == 0,
                0,  # Avoid division by zero
                df_engineered['energy'] / (df_engineered['loudness'].abs() / df_engineered['loudness'].abs().max())
            )
            self.logger.debug("Created energy-loudness interaction features")
        
        # Valence and danceability correlation (0.52)
        if 'valence' in df_engineered.columns and 'danceability' in df_engineered.columns:
            # Create interaction feature
            df_engineered['valence_danceability'] = df_engineered['valence'] * df_engineered['danceability']
            # Create difference feature to capture discrepancies
            df_engineered['valence_danceability_diff'] = df_engineered['valence'] - df_engineered['danceability']
            self.logger.debug("Created valence-danceability interaction features")
        
        # 5. Energy-valence quadrants (mood categories from music psychology)
        if 'energy' in df_engineered.columns and 'valence' in df_engineered.columns:
            conditions = [
                (df_engineered['energy'] > 0.5) & (df_engineered['valence'] > 0.5),
                (df_engineered['energy'] > 0.5) & (df_engineered['valence'] <= 0.5),
                (df_engineered['energy'] <= 0.5) & (df_engineered['valence'] > 0.5),
                (df_engineered['energy'] <= 0.5) & (df_engineered['valence'] <= 0.5)
            ]
            choices = ['happy', 'angry', 'peaceful', 'sad']
            df_engineered['mood_category'] = np.select(conditions, choices, default='unknown')
            self.logger.debug("Created mood_category based on energy-valence quadrants")
        
        # 6. User preferences vs track features (feature difference)
        # This captures how much a track deviates from a user's average preferences
        for feature in [f for f in self.config['track_audio_features'] if f in df_engineered.columns]:
            avg_feature = f"avg_{feature}"
            if avg_feature in df_engineered.columns:
                df_engineered[f"{feature}_diff"] = df_engineered[feature] - df_engineered[avg_feature]
                # Also create absolute difference to measure deviation regardless of direction
                df_engineered[f"{feature}_abs_diff"] = np.abs(df_engineered[feature] - df_engineered[avg_feature])
                self.logger.debug(f"Created difference feature {feature}_diff and {feature}_abs_diff")
        
        # 7. Create age-genre interactions (based on features.md recommendations)
        if 'age_group' in df_engineered.columns and 'main_genre' in df_engineered.columns:
            df_engineered['age_group_genre'] = df_engineered['age_group'].astype(str) + "_" + df_engineered['main_genre']
            self.logger.debug("Created age_group_genre interaction feature")
        
        # 8. Create listening behavior interaction features
        if 'genre_diversity' in df_engineered.columns and 'monthly_hours' in df_engineered.columns:
            # This can capture if a user explores diverse genres but for short periods
            # or listens to fewer genres but more deeply
            df_engineered['listening_depth'] = df_engineered['monthly_hours'] / (df_engineered['genre_diversity'] + 1)
            self.logger.debug("Created listening_depth feature")
        
        # Log feature engineering stats
        new_features = [col for col in df_engineered.columns if col not in df.columns]
        self.logger.info(f"Created {len(new_features)} new features: {new_features}")
        self.logger.log_data_stats("Engineered Data", df_engineered.shape)
        
        self.data = df_engineered
        return df_engineered
    
    @log_function()
    def encode_features(self, df=None, fit=True):
        """
        Encode categorical features and scale numerical features.
        
        Args:
            df (pd.DataFrame, optional): DataFrame for encoding. If None, uses self.data.
            fit (bool): Whether to fit the encoders/scalers or just transform.
            
        Returns:
            pd.DataFrame: DataFrame with encoded features.
        """
        if df is None:
            df = self.data
            
        if df is None:
            self.logger.error("No data available for encoding")
            raise ValueError("No data available for encoding")
            
        self.logger.info(f"Encoding features (fit={fit})")
        
        # Make a copy to avoid modifying the original
        df_encoded = df.copy()
        
        # 1. Scale numerical features
        numerical_features = [col for col in self.config['numerical_features'] if col in df_encoded.columns]
        
        # Add audio features (both track and user preferences)
        for col in df_encoded.columns:
            if any(audio_feature in col for audio_feature in self.config['all_audio_features']):
                if col not in numerical_features:
                    numerical_features.append(col)
        
        # Add engineered numerical features
        engineered_numerical = [col for col in df_encoded.columns 
                              if any(suffix in col for suffix in ['_diff', '_ratio', 'song_age', 'duration_min', 'listening_depth'])]
        numerical_features += engineered_numerical
        
        # Remove duplicates
        numerical_features = list(set(numerical_features))
        
        if numerical_features:
            if fit:
                self.scalers['numerical'] = StandardScaler()
                scaled_values = self.scalers['numerical'].fit_transform(df_encoded[numerical_features])
            else:
                if 'numerical' not in self.scalers:
                    self.logger.error("Scaler not fitted for numerical features")
                    raise ValueError("Scaler not fitted for numerical features")
                scaled_values = self.scalers['numerical'].transform(df_encoded[numerical_features])
            
            # Create new dataframe with scaled values to avoid fragmentation
            scaled_df = pd.DataFrame(
                scaled_values, 
                columns=numerical_features,
                index=df_encoded.index
            )
            
            # Replace original columns with scaled values
            for col in numerical_features:
                df_encoded[col] = scaled_df[col]
            
            self.logger.debug(f"Scaled {len(numerical_features)} numerical features")
        
        # 2. Encode categorical features
        categorical_features = [col for col in self.config['categorical_features'] 
                              if col in df_encoded.columns]
        
        # Add engineered categorical features
        engineered_categorical = [col for col in df_encoded.columns 
                                if col in ['duration_category', 'mood_category', 'age_group', 
                                          'age_group_genre']]
        categorical_features += engineered_categorical
        
        # Remove duplicates
        categorical_features = list(set(categorical_features))
        
        # Prepare to collect one-hot encoded columns for batched assignment
        encoded_data = {}
        cols_to_drop = []
        
        # One-hot encode categorical features
        if categorical_features:
            for col in categorical_features:
                cols_to_drop.append(col)
                if fit:
                    self.encoders[col] = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
                    # Reshape to 2D array for OneHotEncoder
                    encoded = self.encoders[col].fit_transform(df_encoded[col].values.reshape(-1, 1))
                else:
                    if col not in self.encoders:
                        self.logger.error(f"Encoder not fitted for categorical feature: {col}")
                        raise ValueError(f"Encoder not fitted for categorical feature: {col}")
                    encoded = self.encoders[col].transform(df_encoded[col].values.reshape(-1, 1))
                
                # Create column names for one-hot encoded values
                encoded_cols = [f"{col}_{cat}" for cat in self.encoders[col].categories_[0]]
                for i, encoded_col in enumerate(encoded_cols):
                    encoded_data[encoded_col] = encoded[:, i]
            
            # Drop original categorical columns
            df_encoded = df_encoded.drop(columns=cols_to_drop)
            
            # Add all encoded columns at once to avoid fragmentation
            encoded_df = pd.DataFrame(encoded_data, index=df_encoded.index)
            df_encoded = pd.concat([df_encoded, encoded_df], axis=1)
            
            self.logger.debug(f"One-hot encoded {len(categorical_features)} categorical features")
        
        # 3. Encode track and artist identifiers (if present)
        id_cols = ['track_id', 'user_id', 'artist']
        for col in id_cols:
            if col in df_encoded.columns:
                if fit:
                    self.label_encoders[col] = LabelEncoder()
                    df_encoded[col] = self.label_encoders[col].fit_transform(df_encoded[col])
                else:
                    if col not in self.label_encoders:
                        self.logger.error(f"Encoder not fitted for ID column: {col}")
                        raise ValueError(f"Encoder not fitted for ID column: {col}")
                    # Handle unseen categories
                    df_encoded[col] = df_encoded[col].map(
                        lambda x: -1 if x not in self.label_encoders[col].classes_ 
                        else self.label_encoders[col].transform([x])[0]
                    )
                
                self.logger.debug(f"Label encoded {col} with {len(self.label_encoders[col].classes_)} unique values")
        
        # Log encoding stats
        self.logger.log_data_stats("Encoded Data", df_encoded.shape)
        
        return df_encoded
    
    @log_function()
    def split_data(self, df=None, stratify_col='high_engagement'):
        """
        Split data into training, validation, and test sets.
        
        Args:
            df (pd.DataFrame, optional): DataFrame to split. If None, uses self.data.
            stratify_col (str): Column to use for stratified splitting.
            
        Returns:
            tuple: (train_df, val_df, test_df) split DataFrames.
        """
        if df is None:
            df = self.data
            
        if df is None:
            self.logger.error("No data available for splitting")
            raise ValueError("No data available for splitting")
            
        self.logger.info("Splitting data into train, validation, and test sets")
        
        # Get split sizes from config
        test_size = self.config['test_size']
        val_size = self.config['validation_size']
        random_state = self.config['random_state']
        
        # Determine stratify column
        stratify = None
        if stratify_col in df.columns:
            stratify = df[stratify_col]
            self.logger.debug(f"Using {stratify_col} for stratified splitting")
        
        # First split: train+val vs test
        train_val_df, test_df = train_test_split(
            df, test_size=test_size, random_state=random_state, stratify=stratify
        )
        
        # Update stratify for second split
        if stratify is not None:
            stratify = train_val_df[stratify_col]
        
        # Second split: train vs val
        # Adjust validation size relative to train+val size
        val_size_adjusted = val_size / (1 - test_size)
        train_df, val_df = train_test_split(
            train_val_df, test_size=val_size_adjusted, 
            random_state=random_state, stratify=stratify
        )
        
        # Log split sizes
        self.logger.info(f"Data split: train={len(train_df)}, val={len(val_df)}, test={len(test_df)}")
        
        # Store split data
        self.train_data = train_df
        self.val_data = val_df
        self.test_data = test_df
        
        return train_df, val_df, test_df
    
    def _get_columns_to_keep(self):
        """
        Identify which columns should be kept for the model training.
        
        Returns:
            list: List of column names to keep
        """
        columns_to_keep = []
        
        # Add user features
        columns_to_keep.extend(self.config.get('user_demographic_features', []))
        columns_to_keep.extend(self.config.get('user_listening_features', []))
        columns_to_keep.extend(self.config.get('user_audio_preferences', []))
        
        # Add track features
        columns_to_keep.extend(self.config.get('track_metadata_features', []))
        columns_to_keep.extend(self.config.get('track_audio_features', []))
        
        # Add target column
        columns_to_keep.append(self.config['target_column'])
        
        # Add ID columns (needed for recommendations)
        columns_to_keep.extend(['user_id', 'track_id'])
        
        # Remove duplicates
        columns_to_keep = list(set(columns_to_keep))
        
        self.logger.debug(f"Identified {len(columns_to_keep)} columns to keep: {columns_to_keep}")
        
        return columns_to_keep
    
    def drop_unused_columns(self, df):
        """
        Drop columns that are not used in the model training.
        
        Args:
            df (pd.DataFrame): DataFrame to clean
            
        Returns:
            pd.DataFrame: DataFrame with only the required columns
        """
        if df is None:
            self.logger.error("No data available for dropping columns")
            raise ValueError("No data available for dropping columns")
        
        # Get columns to keep
        columns_to_keep = self._get_columns_to_keep()
        
        # Get columns that exist in the dataframe
        available_columns = [col for col in columns_to_keep if col in df.columns]
        
        # Create a new dataframe with only the required columns
        df_clean = df[available_columns].copy()
        
        dropped_columns = set(df.columns) - set(available_columns)
        self.logger.info(f"Dropped {len(dropped_columns)} unused columns: {dropped_columns}")
        
        return df_clean
    
    @log_function()
    def preprocess_pipeline(self, file_path, save_dir=None, drop_unused:Optional[bool]=True):
        """
        Run the complete preprocessing pipeline.
        
        Args:
            file_path (str): Path to the input data file.
            save_dir (str, optional): Directory to save processed data files.
            drop_unused (bool): Whether to drop unused columns during preprocessing.
            
        Returns:
            tuple: (train_df, val_df, test_df) processed and split DataFrames.
        """
        self.logger.log_experiment_start("Data Preprocessing", 
                                        f"Processing data from {file_path}")
        
        # 1. Load data
        df = self.load_data(file_path)
        
        # Optional: Drop unused columns early
        if drop_unused:
            df = self.drop_unused_columns(df)
        
        # 2. Clean data
        df_clean = self.clean_data(df)
        
        # 3. Engineer features
        df_engineered = self.engineer_features(df_clean)
        
        # 4. Encode features (fit encoders on this data)
        df_encoded = self.encode_features(df_engineered, fit=True)
        
        # 5. Split data
        train_df, val_df, test_df = self.split_data(df_encoded)
        
        # 6. Create feature manifest
        manifest = self.create_feature_manifest(train_df)
        
        # 7. Save processed data if save_dir is provided
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
            self.logger.info(f"Saving processed data to {save_dir}")
            
            # Save data splits
            train_path = os.path.join(save_dir, "train_data.csv")
            val_path = os.path.join(save_dir, "val_data.csv")
            test_path = os.path.join(save_dir, "test_data.csv")
            
            train_df.to_csv(train_path, index=False)
            val_df.to_csv(val_path, index=False)
            test_df.to_csv(test_path, index=False)
            
            # Save preprocessor for later use
            import joblib
            preprocessor_path = os.path.join(save_dir, "preprocessor.joblib")
            joblib.dump(self, preprocessor_path)
            
            # Save feature manifest
            manifest_path = self.save_feature_manifest(save_dir)
            
            self.logger.log_file_access(train_path, "write")
            self.logger.log_file_access(val_path, "write")
            self.logger.log_file_access(test_path, "write")
            self.logger.log_file_access(preprocessor_path, "write")
            self.logger.log_file_access(manifest_path, "write")
        
        # Check compatibility of validation and test sets
        val_report = self.check_feature_compatibility(val_df, manifest)
        test_report = self.check_feature_compatibility(test_df, manifest)
        
        # Log any compatibility warnings
        if not val_report["is_compatible"]:
            self.logger.warning(f"Validation set is not fully compatible with manifest: {val_report['issues']}")
        if not test_report["is_compatible"]:
            self.logger.warning(f"Test set is not fully compatible with manifest: {test_report['issues']}")
        
        self.logger.log_experiment_end("Data Preprocessing", {
            "train_size": len(train_df),
            "val_size": len(val_df),
            "test_size": len(test_df),
            "user_features_count": manifest["user_features"]["dimension"],
            "item_features_count": manifest["item_features"]["dimension"],
            "total_features": len(train_df.columns) - len(manifest["ids"]["columns"]) - 1  # Excluding ID and target columns
        })
        
        return train_df, val_df, test_df
    
    @log_function()
    def get_feature_names(self):
        """
        Get the names of features after preprocessing.
        
        Returns:
            list: Feature names.
        """
        if self.train_data is None:
            self.logger.warning("Training data not available. Features unknown.")
            return []
        
        # Exclude target columns
        features = [col for col in self.train_data.columns 
                   if col not in ['playcount', 'high_engagement']]
        
        return features
    
    @log_function()
    def get_embedding_dims(self):
        """
        Get dimensions for embedding layers for categorical features.
        
        Returns:
            dict: Dictionary with feature name as key and embedding dimension as value.
        """
        embedding_dims = {}
        
        # Set embedding dimensions for ID columns
        for col in ['track_id', 'user_id', 'artist']:
            if col in self.label_encoders:
                # Rule of thumb: embedding_dim = min(50, (n_categories + 1) // 2)
                n_categories = len(self.label_encoders[col].classes_)
                embedding_dims[col] = min(self.config['embedding_dim'], (n_categories + 1) // 2)
                self.logger.debug(f"Set embedding dimension for {col}: {embedding_dims[col]}")
        
        return embedding_dims
    
    @log_function()
    def create_feature_manifest(self, df=None):
        """
        Create a manifest of features to ensure consistency between training and evaluation.
        
        Args:
            df (pd.DataFrame, optional): DataFrame to analyze. If None, uses self.train_data.
            
        Returns:
            dict: Manifest containing feature information.
        """
        if df is None:
            df = self.train_data
            
        if df is None:
            self.logger.error("No data available for creating feature manifest")
            raise ValueError("No data available for creating feature manifest")
            
        self.logger.info("Creating feature manifest")
        
        # Initialize manifest
        manifest = {
            "user_features": {
                "columns": [],
                "dimension": 0,
                "types": {}
            },
            "item_features": {
                "columns": [],
                "dimension": 0,
                "types": {}
            },
            "ids": {
                "columns": [],
                "cardinalities": {}
            },
            "target": {
                "column": self.config.get('target_column', 'playcount'),
                "high_engagement_column": "high_engagement",
                "threshold": self.config.get('target_threshold', 5)
            },
            "metadata": {
                "creation_timestamp": datetime.now().isoformat(),
                "original_columns_count": len(df.columns),
                "version": "1.0"
            }
        }
        
        # Group columns by type
        user_features = []
        item_features = []
        id_columns = []
        
        # User feature groups
        demographic_cols = [col for col in df.columns if col.startswith(('gender_', 'age_', 'region_', 'country_'))]
        user_features.extend(demographic_cols)
        
        listening_cols = [col for col in df.columns if col.startswith(('monthly_hours', 'genre_diversity', 'top_genre_'))]
        user_features.extend(listening_cols)
        
        audio_profile_cols = [col for col in df.columns if col.startswith('avg_')]
        user_features.extend(audio_profile_cols)
        
        user_engineered_cols = [col for col in df.columns 
                            if any(x in col for x in ['listening_depth', 'age_group_'])]
        user_features.extend(user_engineered_cols)
        
        # Item feature groups
        audio_cols = [col for col in df.columns 
                    if col in ['danceability', 'energy', 'key', 'loudness', 'mode',
                            'speechiness', 'acousticness', 'instrumentalness',
                            'liveness', 'valence', 'tempo', 'time_signature']]
        item_features.extend(audio_cols)
        
        genre_cols = [col for col in df.columns if col.startswith('main_genre_')]
        item_features.extend(genre_cols)
        
        temporal_cols = [col for col in df.columns 
                        if col in ['year', 'song_age', 'is_recent']]
        item_features.extend(temporal_cols)
        
        item_engineered_cols = [col for col in df.columns 
                            if any(x in col for x in ['duration_', 'mood_category', 'energy_loudness', 
                                                    'valence_danceability', '_diff', '_abs_diff'])]
        item_features.extend(item_engineered_cols)
        
        # ID columns
        for col in ['user_id', 'track_id', 'artist']:
            if col in df.columns:
                id_columns.append(col)
                if col in self.label_encoders:
                    manifest["ids"]["cardinalities"][col] = len(self.label_encoders[col].classes_)
                else:
                    manifest["ids"]["cardinalities"][col] = df[col].nunique()
        
        # Remove duplicates while preserving order
        user_features = list(dict.fromkeys(user_features))
        item_features = list(dict.fromkeys(item_features))
        
        # Store in manifest
        manifest["user_features"]["columns"] = user_features
        manifest["user_features"]["dimension"] = len(user_features)
        manifest["item_features"]["columns"] = item_features
        manifest["item_features"]["dimension"] = len(item_features)
        manifest["ids"]["columns"] = id_columns
        
        # Store data types for each column
        for col in user_features:
            manifest["user_features"]["types"][col] = str(df[col].dtype)
        
        for col in item_features:
            manifest["item_features"]["types"][col] = str(df[col].dtype)
        
        # Add combined feature counts for model input dimensions
        manifest["model_dimensions"] = {
            "user_input_dim": len(user_features),
            "item_input_dim": len(item_features)
        }
        
        # Add scaler info if available
        if hasattr(self, 'scalers') and self.scalers:
            manifest["scaling"] = {
                "numerical_features_scaled": 'numerical' in self.scalers,
                "scaler_type": "StandardScaler"
            }
        
        # Add encoding info if available
        if hasattr(self, 'encoders') and self.encoders:
            manifest["encoding"] = {
                "categorical_features_encoded": list(self.encoders.keys()),
                "encoder_type": "OneHotEncoder"
            }
        
        self.logger.info(f"Created feature manifest with {len(user_features)} user features and {len(item_features)} item features")
        self.feature_manifest = manifest
        
        return manifest
    
    @log_function()
    def save_feature_manifest(self, output_dir):
        """
        Save the feature manifest to a file.
        
        Args:
            output_dir (str): Directory to save the manifest file.
            
        Returns:
            str: Path to the saved manifest file.
        """
        if not hasattr(self, 'feature_manifest'):
            self.logger.warning("Feature manifest not created yet. Creating now.")
            self.create_feature_manifest()
        
        os.makedirs(output_dir, exist_ok=True)
        manifest_path = os.path.join(output_dir, "feature_manifest.yaml")
        
        with open(manifest_path, 'w') as f:
            yaml.dump(self.feature_manifest, f, default_flow_style=False)
        
        self.logger.info(f"Saved feature manifest to {manifest_path}")
        
        return manifest_path
        
    @log_function()
    def check_feature_compatibility(self, df, manifest=None):
        """
        Check if a dataframe is compatible with the feature manifest.
        
        Args:
            df (pd.DataFrame): DataFrame to check.
            manifest (dict, optional): Feature manifest to check against. If None, uses self.feature_manifest.
            
        Returns:
            dict: Compatibility report with any issues found.
        """
        if manifest is None:
            if not hasattr(self, 'feature_manifest'):
                self.logger.warning("Feature manifest not created yet. Creating now.")
                self.create_feature_manifest()
            manifest = self.feature_manifest
        
        self.logger.info("Checking feature compatibility")
        
        report = {
            "is_compatible": True,
            "issues": [],
            "missing_user_features": [],
            "missing_item_features": [],
            "extra_user_features": [],
            "extra_item_features": [],
            "dimension_mismatch": {
                "user": False,
                "item": False
            }
        }
        
        # Check user features
        expected_user_features = set(manifest["user_features"]["columns"])
        actual_user_features = set(col for col in df.columns if col in expected_user_features)
        
        report["missing_user_features"] = list(expected_user_features - actual_user_features)
        
        # Check item features
        expected_item_features = set(manifest["item_features"]["columns"])
        actual_item_features = set(col for col in df.columns if col in expected_item_features)
        
        report["missing_item_features"] = list(expected_item_features - actual_item_features)
        
        # Check for extra features (not in the manifest)
        all_expected_features = expected_user_features | expected_item_features | set(manifest["ids"]["columns"])
        all_expected_features.add(manifest["target"]["high_engagement_column"])
        
        extra_features = set(df.columns) - all_expected_features
        
        # Categorize extra features
        for feature in extra_features:
            if any(feature.startswith(prefix) for prefix in ['gender_', 'age_', 'region_', 'country_', 'monthly_hours', 'genre_diversity', 'top_genre_', 'avg_', 'listening_depth']):
                report["extra_user_features"].append(feature)
            elif any(feature.startswith(prefix) for prefix in ['main_genre_', 'duration_', 'mood_category', 'energy_loudness', 'valence_danceability']):
                report["extra_item_features"].append(feature)
        
        # Check dimension compatibility
        if len(actual_user_features) != manifest["user_features"]["dimension"]:
            report["dimension_mismatch"]["user"] = True
            report["issues"].append(f"User feature dimension mismatch: expected {manifest['user_features']['dimension']}, got {len(actual_user_features)}")
            report["is_compatible"] = False
        
        if len(actual_item_features) != manifest["item_features"]["dimension"]:
            report["dimension_mismatch"]["item"] = True
            report["issues"].append(f"Item feature dimension mismatch: expected {manifest['item_features']['dimension']}, got {len(actual_item_features)}")
            report["is_compatible"] = False
        
        # Check if target column exists
        if manifest["target"]["high_engagement_column"] not in df.columns:
            report["issues"].append(f"Target column '{manifest['target']['high_engagement_column']}' not found")
            report["is_compatible"] = False
        
        # Check if ID columns exist
        for id_col in manifest["ids"]["columns"]:
            if id_col not in df.columns:
                report["issues"].append(f"ID column '{id_col}' not found")
                report["is_compatible"] = False
        
        # Log report
        if report["is_compatible"]:
            self.logger.info("Data is compatible with feature manifest")
        else:
            self.logger.warning(f"Data is not compatible with feature manifest. Issues: {report['issues']}")
        
        return report


if __name__ == "__main__":
    # Example usage
    logger = LhydraLogger(log_dir="logs", log_level=logging.INFO)
    logger.info("Starting data preprocessing script")
    
    # Load configuration 
    import argparse
    parser = argparse.ArgumentParser(description="Preprocess music data for Lhydra recommender")
    parser.add_argument("--config", type=str, default=None, help="Path to config file")
    parser.add_argument("--input", type=str, required=True, help="Path to input data file")
    parser.add_argument("--output", type=str, default="data/preprocessed", help="Output directory")
    parser.add_argument("--keep-all", action="store_false", dest="drop_unused",
                        help="Keep all columns instead of dropping unused ones", default=True)
    args = parser.parse_args()
    
    # Initialize preprocessor
    preprocessor = MusicDataPreprocessor(config_path=args.config, logger=logger)
    
    # Run preprocessing pipeline
    train_df, val_df, test_df = preprocessor.preprocess_pipeline(
        args.input, args.output, drop_unused=args.drop_unused
    )
    
    logger.info("Data preprocessing completed successfully") 