import numpy as np
import pandas as pd
import os
import re
import logging
import joblib
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.multioutput import MultiOutputRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
# Replace gensim with this simpler approach
import requests
from zipfile import ZipFile
from io import BytesIO
import nltk
from nltk.tokenize import word_tokenize

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class TagBasedAudioFeaturePredictor:
    """Predicts audio features from music tags using a machine learning approach."""
    
    # Audio features to predict
    AUDIO_FEATURES = [
        'danceability', 'energy', 'key', 'loudness', 'mode', 
        'speechiness', 'acousticness', 'instrumentalness', 
        'liveness', 'valence', 'tempo'
    ]
    
    def __init__(self, model_dir="models", use_pretrained=True, include_metadata=True):
        """Initialize the predictor with model directory."""
        self.model_dir = model_dir
        self.model = None
        self.vectorizer = None
        self.feature_columns = None
        self.use_pretrained = use_pretrained
        self.include_metadata = include_metadata
        self.embedding_model = {}  # Dictionary to store word vectors
        self.embedding_dim = 100  # Default dimension
        self.feature_names = []
        os.makedirs(model_dir, exist_ok=True)
        
        # Download nltk resources if needed
        try:
            nltk.data.find('tokenizers/punkt')
        except LookupError:
            logger.info("Downloading NLTK punkt tokenizer...")
            nltk.download('punkt', quiet=True)
        try:
            nltk.data.find('tokenizers/punkt_tab')
        except LookupError:
            logger.info("Downloading NLTK punkt_tab tokenizer...")
            nltk.download('punkt_tab', quiet=True)
    
    def _preprocess_tags(self, tags_series):
        """Process tag strings into a clean format."""
        processed_tags = []
        
        for tags in tags_series:
            if pd.isna(tags):
                processed_tags.append("")
                continue
            
            # Convert to lowercase and remove punctuation
            clean_tags = re.sub(r'[^\w\s]', ' ', str(tags).lower())
            # Replace multiple spaces with a single space
            clean_tags = re.sub(r'\s+', ' ', clean_tags).strip()
            processed_tags.append(clean_tags)
            
        return processed_tags
    
    def _prepare_data(self, df):
        """Prepare data for training or prediction."""
        if not isinstance(df, pd.DataFrame):
            raise TypeError("Input must be a pandas DataFrame")
        
        # Clean the tags
        if 'tags' in df.columns:
            df['processed_tags'] = self._preprocess_tags(df['tags'])
        elif 'genres' in df.columns:
            # Handle Spotify dataset format
            df['processed_tags'] = self._preprocess_tags(df['genres'])
            # Add main_genre to processed_tags for better signal
            if 'main_genre' in df.columns:
                df['processed_tags'] = df.apply(
                    lambda x: f"{x['processed_tags']} {x['main_genre'].lower()}" 
                    if not pd.isna(x['main_genre']) else x['processed_tags'], 
                    axis=1
                )
                
            # Add first_genre, second_genre, third_genre as additional signal
            genre_cols = ['first_genre', 'second_genre', 'third_genre']
            for col in genre_cols:
                if col in df.columns:
                    df['processed_tags'] = df.apply(
                        lambda x: f"{x['processed_tags']} {x[col].lower()}" 
                        if not pd.isna(x[col]) and x[col] != 'Unknown' else x['processed_tags'], 
                        axis=1
                    )
        else:
            raise ValueError("DataFrame must contain either 'tags' or 'genres' column")
        
        # Add artist and track name as additional features
        if self.include_metadata:
            if 'artist' in df.columns:
                df['processed_artist'] = self._preprocess_tags(df['artist'])
            
            # Extract track name from different possible column names
            if 'name' in df.columns:
                df['processed_name'] = self._preprocess_tags(df['name'])
            elif 'Artist and Title' in df.columns:
                # Extract track name from "Artist - Title" format
                df['processed_name'] = df['Artist and Title'].apply(
                    lambda x: self._preprocess_tags([x.split(' - ')[1] if ' - ' in str(x) else ''])[0]
                )
        
        # Add handcrafted features based on genre keywords
        df['has_dance_related'] = df['processed_tags'].apply(
            lambda x: 1 if any(w in x for w in ['dance', 'edm', 'club', 'party', 'disco']) else 0
        )
        df['has_acoustic_related'] = df['processed_tags'].apply(
            lambda x: 1 if any(w in x for w in ['acoustic', 'folk', 'indie', 'singer']) else 0
        )
        df['has_live_related'] = df['processed_tags'].apply(
            lambda x: 1 if any(w in x for w in ['live', 'concert', 'performance']) else 0
        )
        df['has_rock_related'] = df['processed_tags'].apply(
            lambda x: 1 if any(w in x for w in ['rock', 'metal', 'hard', 'punk', 'alternative']) else 0
        )
        df['has_rap_related'] = df['processed_tags'].apply(
            lambda x: 1 if any(w in x for w in ['rap', 'hip hop', 'trap', 'urban']) else 0
        )
        
        return df
    
    def _download_glove_embeddings(self):
        """Download GloVe word embeddings."""
        logger.info("Downloading GloVe embeddings...")
        glove_path = os.path.join(self.model_dir, "glove.6B.100d.txt")
        
        # If embeddings already exist, load them
        if os.path.exists(glove_path):
            logger.info("Loading existing GloVe embeddings...")
        else:
            # Download file
            try:
                url = "https://nlp.stanford.edu/data/glove.6B.zip"
                logger.info(f"Downloading GloVe embeddings from {url}")
                response = requests.get(url)
                z = ZipFile(BytesIO(response.content))
                z.extract("glove.6B.100d.txt", self.model_dir)
                logger.info("GloVe embeddings downloaded successfully")
            except Exception as e:
                logger.error(f"Error downloading GloVe embeddings: {e}")
                self.use_pretrained = False
                return
        
        # Load embeddings
        self.embedding_dim = 100
        try:
            with open(glove_path, 'r', encoding='utf-8') as f:
                for line in f:
                    values = line.split()
                    word = values[0]
                    vector = np.array(values[1:], dtype='float32')
                    self.embedding_model[word] = vector
            logger.info(f"Loaded {len(self.embedding_model)} word vectors")
        except Exception as e:
            logger.error(f"Error loading GloVe embeddings: {e}")
            self.use_pretrained = False
    
    def _get_text_embedding(self, text):
        """Convert text to embedding vector using pretrained model."""
        if not text:
            # Return zero vector for empty text
            return np.zeros(self.embedding_dim)
        
        words = word_tokenize(text)
        word_vectors = [self.embedding_model[word] for word in words if word in self.embedding_model]
        
        if not word_vectors:
            return np.zeros(self.embedding_dim)
        
        # Average word vectors to get document vector
        return np.mean(word_vectors, axis=0)
    
    def _vectorize_data(self, df):
        """Convert text data to numerical features."""
        # If using pretrained embeddings
        if self.use_pretrained and self.embedding_model:
            # Get embeddings from tags
            tag_features = np.array([self._get_text_embedding(text) for text in df['processed_tags']])
            feature_names = [f'tag_emb_{i}' for i in range(tag_features.shape[1])]
            
            # Add metadata embeddings if available
            if self.include_metadata:
                features_list = [tag_features]
                features_names = feature_names
                
                if 'processed_artist' in df.columns:
                    artist_features = np.array([self._get_text_embedding(text) for text in df['processed_artist']])
                    artist_feature_names = [f'artist_emb_{i}' for i in range(artist_features.shape[1])]
                    features_list.append(artist_features)
                    features_names.extend(artist_feature_names)
                    
                if 'processed_name' in df.columns:
                    name_features = np.array([self._get_text_embedding(text) for text in df['processed_name']])
                    name_feature_names = [f'name_emb_{i}' for i in range(name_features.shape[1])]
                    features_list.append(name_features)
                    features_names.extend(name_feature_names)
                
                # Get handcrafted features
                handcrafted_cols = ['has_dance_related', 'has_acoustic_related', 
                                   'has_live_related', 'has_rock_related', 'has_rap_related']
                handcrafted_features = df[handcrafted_cols].values
                features_list.append(handcrafted_features)
                features_names.extend(handcrafted_cols)
                
                # Combine all features
                features = np.hstack(features_list)
            else:
                features = tag_features
                features_names = feature_names
            
            self.feature_names = features_names
            return features
        
        # Fallback to TF-IDF if pretrained embeddings not available
        else:
            if not self.vectorizer:
                self.vectorizer = TfidfVectorizer(max_features=500, stop_words='english')
                tfidf_features = self.vectorizer.fit_transform(df['processed_tags']).toarray()
                
                # Add handcrafted features
                handcrafted_cols = ['has_dance_related', 'has_acoustic_related', 
                                   'has_live_related', 'has_rock_related', 'has_rap_related']
                handcrafted_features = df[handcrafted_cols].values
                
                # Combine TF-IDF with handcrafted features
                features = np.hstack([tfidf_features, handcrafted_features])
                
                # Set feature names
                self.feature_names = list(self.vectorizer.get_feature_names_out()) + handcrafted_cols
            else:
                tfidf_features = self.vectorizer.transform(df['processed_tags']).toarray()
                handcrafted_cols = ['has_dance_related', 'has_acoustic_related', 
                                   'has_live_related', 'has_rock_related', 'has_rap_related']
                handcrafted_features = df[handcrafted_cols].values
                features = np.hstack([tfidf_features, handcrafted_features])
            
            return features
    
    def fit(self, train_df):
        """Train the model on the provided DataFrame."""
        logger.info("Preparing training data...")
        train_df = self._prepare_data(train_df)
        
        # Drop rows with NaN in audio features
        train_df = train_df.dropna(subset=self.AUDIO_FEATURES)
        
        if len(train_df) == 0:
            raise ValueError("No valid training data after removing NaN values")
        
        logger.info(f"Training with {len(train_df)} samples")
        
        # Load pretrained embeddings if using them
        if self.use_pretrained:
            self._download_glove_embeddings()
        
        # Convert text to numerical features
        X = self._vectorize_data(train_df)
        y = train_df[self.AUDIO_FEATURES].values
        
        # Use different models for different feature types
        logger.info("Training specialized models...")
        
        # Categorize features
        musical_features = ['danceability', 'energy', 'acousticness', 
                           'instrumentalness', 'valence', 'loudness']
        rhythm_features = ['tempo']
        technical_features = ['speechiness', 'liveness']
        categorical_features = ['key', 'mode']
        
        # Get indices for different feature groups
        musical_indices = [self.AUDIO_FEATURES.index(f) for f in musical_features]
        rhythm_indices = [self.AUDIO_FEATURES.index(f) for f in rhythm_features]
        technical_indices = [self.AUDIO_FEATURES.index(f) for f in technical_features]
        categorical_indices = [self.AUDIO_FEATURES.index(f) for f in categorical_features]
        
        # Create base models with different parameters for each feature type
        models = {}
        
        # For musical features - use standard RandomForest
        musical_model = RandomForestRegressor(
            n_estimators=200, 
            max_depth=15,
            min_samples_split=5,
            n_jobs=-1,
            random_state=42
        )
        
        # For rhythm features - deeper trees
        rhythm_model = RandomForestRegressor(
            n_estimators=250, 
            max_depth=30,
            min_samples_split=3,
            n_jobs=-1,
            random_state=42
        )
        
        # For technical features - use gradient boosting
        technical_model = GradientBoostingRegressor(
            n_estimators=150,
            max_depth=10,
            learning_rate=0.05,
            random_state=42
        )
        
        # For categorical features - use shallow forest
        categorical_model = RandomForestRegressor(
            n_estimators=100,
            max_depth=8,
            min_samples_leaf=10,
            random_state=42
        )
        
        # Train MultiOutputRegressor for each feature group
        self.model = {}
        
        logger.info("Training models for musical features...")
        musical_regressor = MultiOutputRegressor(musical_model)
        musical_regressor.fit(X, y[:, musical_indices])
        self.model['musical'] = musical_regressor
        
        logger.info("Training models for rhythm features...")
        rhythm_regressor = MultiOutputRegressor(rhythm_model)
        rhythm_regressor.fit(X, y[:, rhythm_indices])
        self.model['rhythm'] = rhythm_regressor
        
        logger.info("Training models for technical features...")
        technical_regressor = MultiOutputRegressor(technical_model)
        technical_regressor.fit(X, y[:, technical_indices])
        self.model['technical'] = technical_regressor
        
        logger.info("Training models for categorical features...")
        categorical_regressor = MultiOutputRegressor(categorical_model)
        categorical_regressor.fit(X, y[:, categorical_indices])
        self.model['categorical'] = categorical_regressor
        
        # Store feature indices
        self.feature_indices = {
            'musical': musical_indices,
            'rhythm': rhythm_indices,
            'technical': technical_indices,
            'categorical': categorical_indices
        }
        
        # Save for future reference
        self.feature_columns = self.AUDIO_FEATURES
        
        logger.info("Model training completed")
        return self
    
    def evaluate(self, test_df):
        """Evaluate the model on test data."""
        if not self.model:
            raise ValueError("Model not trained yet. Call fit() first.")
        
        logger.info("Evaluating model on test data...")
        test_df = self._prepare_data(test_df)
        test_df = test_df.dropna(subset=self.AUDIO_FEATURES)
        
        if len(test_df) == 0:
            raise ValueError("No valid test data after removing NaN values")
        
        X_test = self._vectorize_data(test_df)
        y_true = test_df[self.AUDIO_FEATURES].values
        
        # Make predictions using each specialized model
        y_pred = np.zeros_like(y_true)
        
        for model_type, indices in self.feature_indices.items():
            y_pred[:, indices] = self.model[model_type].predict(X_test)
        
        # Post-process predictions
        # Round key and mode to integers
        for i in self.feature_indices['categorical']:
            y_pred[:, i] = np.round(y_pred[:, i])
        
        # Calculate metrics
        results = {}
        for i, feature in enumerate(self.AUDIO_FEATURES):
            mse = mean_squared_error(y_true[:, i], y_pred[:, i])
            r2 = r2_score(y_true[:, i], y_pred[:, i])
            results[feature] = {'mse': mse, 'r2': r2}
            logger.info(f"{feature}: MSE = {mse:.4f}, R² = {r2:.4f}")
        
        # Overall metrics
        overall_mse = mean_squared_error(y_true, y_pred)
        logger.info(f"Overall MSE: {overall_mse:.4f}")
        
        # Analyze feature importance
        self._analyze_feature_importance()
        
        return results
    
    def _analyze_feature_importance(self):
        """Analyze and visualize feature importance for each audio feature."""
        if not self.model or not self.feature_names:
            logger.warning("No model or feature names available for importance analysis")
            return
        
        logger.info("Analyzing feature importance...")
        
        # Get feature importance for each model type and target
        importance_by_target = {}
        
        for model_type, indices in self.feature_indices.items():
            model = self.model[model_type]
            
            for i, target_idx in enumerate(indices):
                target = self.AUDIO_FEATURES[target_idx]
                
                # Extract feature importance from the model for this target
                if hasattr(model.estimators_[i], 'feature_importances_'):
                    estimator = model.estimators_[i]
                    importance = estimator.feature_importances_
                    
                    # Sort by importance
                    sorted_indices = np.argsort(importance)[::-1]
                    
                    # Map to feature names
                    if len(importance) == len(self.feature_names):
                        # Map full feature names
                        feature_importance = {self.feature_names[j]: importance[j] for j in sorted_indices}
                    else:
                        # Map with indices if length mismatch
                        feature_importance = {f"Feature_{j}": importance[j] for j in sorted_indices}
                    
                    importance_by_target[target] = feature_importance
        
        # Save top 10 features for each target
        os.makedirs(self.model_dir, exist_ok=True)
        with open(os.path.join(self.model_dir, "feature_importance.txt"), "w") as f:
            for target, importance in importance_by_target.items():
                f.write(f"\nTop 10 features for {target}:\n")
                for i, (feature, imp) in enumerate(list(importance.items())[:10]):
                    f.write(f"{i+1}. {feature}: {imp:.4f}\n")
        
        # Plot feature importance for selected targets
        self._plot_feature_importance(importance_by_target)
    
    def _plot_feature_importance(self, importance_by_target, targets=None, top_n=15):
        """Create feature importance plots for selected targets."""
        if targets is None:
            # Default to plotting features from different categories
            targets = ['danceability', 'energy', 'key', 'tempo']
        
        # Create a multi-panel figure
        n_plots = len(targets)
        fig, axes = plt.subplots(n_plots, 1, figsize=(10, 4 * n_plots))
        if n_plots == 1:
            axes = [axes]
        
        for i, target in enumerate(targets):
            if target not in importance_by_target:
                continue
                
            # Get top N features by importance
            importance = importance_by_target[target]
            features = list(importance.keys())[:top_n]
            values = list(importance.values())[:top_n]
            
            # Create shorter labels for display
            labels = []
            for f in features:
                if f.startswith('tag_emb_') or f.startswith('artist_emb_') or f.startswith('name_emb_'):
                    labels.append(f.split('_')[0] + '_emb')
                else:
                    labels.append(f[:15] + '...' if len(f) > 15 else f)
            
            # Plot horizontal bar chart
            ax = axes[i]
            ax.barh(range(len(features)), values, align='center')
            ax.set_yticks(range(len(features)))
            ax.set_yticklabels(labels)
            ax.set_title(f'Feature Importance for {target}')
            ax.set_xlabel('Importance')
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.model_dir, "feature_importance.png"))
        logger.info(f"Feature importance plots saved to {os.path.join(self.model_dir, 'feature_importance.png')}")
    
    def predict(self, df):
        """Predict audio features for the given DataFrame."""
        if not self.model:
            raise ValueError("Model not trained yet. Call fit() first.")
        
        logger.info(f"Predicting audio features for {len(df)} tracks...")
        df = self._prepare_data(df)
        
        X = self._vectorize_data(df)
        
        # Initialize predictions array
        predictions = np.zeros((X.shape[0], len(self.AUDIO_FEATURES)))
        
        # Make predictions using each specialized model
        for model_type, indices in self.feature_indices.items():
            predictions[:, indices] = self.model[model_type].predict(X)
        
        # Convert predictions to DataFrame
        results_df = pd.DataFrame(
            predictions, 
            columns=self.feature_columns,
            index=df.index
        )
        
        # Post-process predictions
        # For example, 'key' and 'mode' are typically integers
        results_df['key'] = results_df['key'].round().astype(int)
        results_df['mode'] = results_df['mode'].round().astype(int)
        
        # Ensure values are in expected ranges
        results_df['danceability'] = results_df['danceability'].clip(0, 1)
        results_df['energy'] = results_df['energy'].clip(0, 1)
        results_df['speechiness'] = results_df['speechiness'].clip(0, 1)
        results_df['acousticness'] = results_df['acousticness'].clip(0, 1)
        results_df['instrumentalness'] = results_df['instrumentalness'].clip(0, 1)
        results_df['liveness'] = results_df['liveness'].clip(0, 1)
        results_df['valence'] = results_df['valence'].clip(0, 1)
        
        logger.info("Prediction completed")
        return results_df
    
    def save(self, filename=None):
        """Save the trained model and vectorizer to disk."""
        if not self.model:
            raise ValueError("No model to save. Train the model first.")
        
        if filename is None:
            filename = "tag_audio_predictor"
        
        model_path = os.path.join(self.model_dir, f"{filename}_model.joblib")
        
        # Save relevant components
        components = {
            'model': self.model,
            'feature_columns': self.feature_columns,
            'feature_names': self.feature_names,
            'feature_indices': self.feature_indices,
            'use_pretrained': self.use_pretrained,
            'include_metadata': self.include_metadata
        }
        
        # Save vectorizer if it exists
        if self.vectorizer is not None:
            components['vectorizer'] = self.vectorizer
        
        joblib.dump(components, model_path)
        logger.info(f"Model saved to {model_path}")
        
        return {'model_path': model_path}
    
    def load(self, model_path):
        """Load a saved model from disk."""
        components = joblib.load(model_path)
        
        self.model = components['model']
        self.feature_columns = components['feature_columns']
        self.feature_names = components['feature_names']
        self.feature_indices = components['feature_indices']
        self.use_pretrained = components['use_pretrained']
        self.include_metadata = components['include_metadata']
        
        if 'vectorizer' in components:
            self.vectorizer = components['vectorizer']
        
        logger.info(f"Model loaded from {model_path}")
        return self


def main():
    """Main function to train and evaluate the model."""
    # Load the data
    logger.info("Loading data...")
    music_info_path = os.path.join('data', 'raw', 'Music Info.csv')
    spotify_path = os.path.join('data', 'raw', 'spotify_full_list_20102023.csv')
    
    try:
        music_info = pd.read_csv(music_info_path)
        logger.info(f"Loaded Music Info with {len(music_info)} tracks")
    except FileNotFoundError:
        logger.error(f"Could not find Music Info file at {music_info_path}")
        return
    
    # Drop rows with missing values to speed up training
    music_info = music_info.dropna(subset=['tags'] + TagBasedAudioFeaturePredictor.AUDIO_FEATURES)
    logger.info(f"After removing rows with NaN: {len(music_info)} tracks")
    
    # Split the data
    train_df, test_df = train_test_split(
        music_info, test_size=0.2, random_state=42
    )
    
    # Create and train the model
    predictor = TagBasedAudioFeaturePredictor(use_pretrained=True, include_metadata=True)
    predictor.fit(train_df)
    
    # Evaluate the model
    evaluation = predictor.evaluate(test_df)
    
    # Save the model
    predictor.save()
    
    # Demonstrate prediction on Spotify data
    try:
        spotify_df = pd.read_csv(spotify_path)
        logger.info(f"Loaded Spotify data with {len(spotify_df)} tracks")
        
        # Predict audio features
        predicted_features = predictor.predict(spotify_df)
        
        # Add predictions to the original dataframe
        result_df = pd.concat([spotify_df, predicted_features], axis=1)
        
        # Save the results
        output_path = os.path.join('data', 'processed', 'spotify_with_predicted_features.csv')
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        result_df.to_csv(output_path, index=False)
        logger.info(f"Predictions saved to {output_path}")
        
    except FileNotFoundError:
        logger.warning(f"Could not find Spotify file at {spotify_path}, skipping prediction")


if __name__ == "__main__":
    main()