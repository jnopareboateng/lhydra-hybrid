"""
This script is responsible for predicting audio features from input data with various machine learning techniques.
It includes data preprocessing, model training, and evaluation steps.
"""

import numpy as np
import pandas as pd
import os
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import lightgbm as lgbm
# Import properly from the tag_audio_features_prediction module
from utils.tag_audio_features_prediction import TagBasedAudioFeaturePredictor
# Add tqdm for progress visualization
from tqdm import tqdm

# Check if GPU is available for LightGBM
def check_gpu_availability():
    """Check if GPU is available for LightGBM and return appropriate device parameters."""
    try:
        # Create a small dataset to test GPU usage
        X = np.random.rand(100, 10)
        y = np.random.rand(100)
        
        # Try to train a model with GPU
        params = {'device': 'gpu', 'gpu_platform_id': 0, 'gpu_device_id': 0}
        dtrain = lgbm.Dataset(X, label=y)
        gbm = lgbm.train(params, dtrain, num_boost_round=10)
        print("✓ GPU is available and will be used for LightGBM training")
        return params
    except Exception as e:
        print(f"✗ GPU training not available: {e}")
        print("✓ Using CPU for training instead")
        return {'device': 'cpu'}

# Load datasets
try:
    old_df = pd.read_csv("old_dataset_with_audio.csv")
    new_df = pd.read_csv("new_dataset_without_audio.csv")
except FileNotFoundError:
    print("Input datasets not found. Using alternative paths...")
    # Try alternative paths
    data_dir = os.path.join('data', 'raw')
    old_df = pd.read_csv(os.path.join(data_dir, "Music Info.csv"))
    new_df = pd.read_csv(os.path.join(data_dir, "spotify_full_list_20102023.csv"))

# Check GPU availability for LightGBM
lgbm_device_params = check_gpu_availability()

# 1. Better feature engineering
def extract_genre_features(df):
    # Create genre embeddings from tags
    from collections import Counter
    
    # Identify the tag column (could be 'tags' or 'genres')
    tag_column = 'tags' if 'tags' in df.columns else 'genres'
    
    # One-hot encode top N genres
    print(f"Counting tag frequencies from {tag_column} column...")
    top_genres = Counter([g.strip() for tags in tqdm(df[tag_column].dropna(), desc="Scanning tags") 
                          for g in str(tags).split(',')])
    top_genres = [g for g, _ in top_genres.most_common(50)]
    
    print(f"Creating genre matrix for {len(top_genres)} top genres...")
    genre_matrix = np.zeros((len(df), len(top_genres)))
    
    for i, tags in enumerate(tqdm(df[tag_column].fillna(''), desc="Building genre features")):
        for genre in str(tags).split(','):
            genre = genre.strip()
            if genre in top_genres:
                idx = top_genres.index(genre)
                genre_matrix[i, idx] = 1
    
    genre_df = pd.DataFrame(genre_matrix, columns=top_genres, index=df.index)
    return pd.concat([df, genre_df], axis=1)

# Extract artist-level features
def extract_artist_features(df):
    # Check if required columns exist
    required_columns = ['artist', 'danceability', 'energy', 'loudness', 'tempo']
    if not all(col in df.columns for col in required_columns):
        print("Missing required columns for artist feature extraction")
        return df
    
    print("Extracting artist-level audio feature statistics...")
    # Aggregate statistics by artist
    artist_features = df.groupby('artist').agg({
        'danceability': ['mean', 'std'],
        'energy': ['mean', 'std'],
        'loudness': ['mean', 'std'],
        'tempo': ['mean', 'std']
    }).reset_index()
    
    # Flatten multi-level columns
    artist_features.columns = ['artist'] + [
        f'{col[0]}_{col[1]}' for col in artist_features.columns[1:]
    ]
    
    # Merge back to original dataframe
    return df.merge(artist_features, on='artist', how='left')

# Initialize the tag-based predictor
print("Initializing TagBasedAudioFeaturePredictor...")
predictor = TagBasedAudioFeaturePredictor(use_pretrained=True, include_metadata=True, use_gpu=lgbm_device_params['device'] == 'gpu')

# Check if we have audio features in the old dataset to train
has_audio_features = all(feature in old_df.columns for feature in predictor.AUDIO_FEATURES)

if has_audio_features:
    print("Training using existing audio features in the dataset...")
    # Prepare data using the existing TagBasedAudioFeaturePredictor
    prepared_df = predictor._prepare_data(old_df)
    
    # Split the data
    print("Splitting dataset into training and test sets...")
    train_df, test_df = train_test_split(prepared_df, test_size=0.2, random_state=42)
    
    # Train the model
    predictor.fit(train_df)
    
    # Evaluate the model
    print("Evaluating model performance...")
    evaluation = predictor.evaluate(test_df)
    
    # Save the model
    print("Saving trained model...")
    predictor.save()
    
    # Apply models to predict audio features in new dataset
    if 'tags' not in new_df.columns and 'genres' in new_df.columns:
        # Map genre information to tags format if needed
        print("Converting genres to tags format...")
        new_df['tags'] = new_df['genres'] 
    
    print(f"Predicting audio features for {len(new_df)} tracks...")
    predicted_features = predictor.predict(new_df)
    
    # Add predictions to the original dataframe
    new_df_enriched = pd.concat([new_df, predicted_features], axis=1)
    
    # Save the enhanced dataset
    print("Saving predictions to file...")
    new_df_enriched.to_csv('new_dataset_with_predicted_audio.csv', index=False)
    
else:
    print("No audio features found in the dataset. Using alternative approach...")
    
    # Apply feature engineering to old dataset
    print("Enriching data with feature engineering...")
    old_df_enriched = extract_artist_features(old_df)
    old_df_enriched = extract_genre_features(old_df_enriched)
    
    # Define the audio features to predict
    categorical_features = ['key', 'mode', 'time_signature']
    continuous_features = ['danceability', 'energy', 'loudness', 'speechiness', 
                          'acousticness', 'instrumentalness', 'liveness', 
                          'valence', 'tempo']
    
    # Initialize model dictionaries
    categorical_models = {}
    continuous_models = {}
    
    # Feature selection - exclude target columns from predictors
    predictors = [col for col in old_df_enriched.columns 
                  if col not in continuous_features + categorical_features]
    
    # Train specialized models for each target feature
    print("Training categorical feature models...")
    for feature in tqdm(categorical_features, desc="Training categorical models"):
        if feature not in old_df_enriched.columns:
            print(f"Feature {feature} not found in dataset, skipping...")
            continue
            
        # For categorical features, use classification approaches
        X = old_df_enriched[predictors]
        y = old_df_enriched[feature]
        
        # Handle missing values
        X = X.fillna(-1)  # Special value for missing
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Use specialized models for categorical data with GPU acceleration if available
        base_params = {
            'n_estimators': 100,
            'random_state': 42,
            **lgbm_device_params  # Add GPU parameters if available
        }
        
        if feature == 'key':  # Multi-class
            model = lgbm.LGBMClassifier(**base_params)
        else:  # Binary for mode
            model = lgbm.LGBMClassifier(**base_params)
        
        model.fit(X_train, y_train)
        categorical_models[feature] = model
        
        # Evaluate
        y_pred = model.predict(X_test)
        accuracy = (y_pred == y_test).mean()
        print(f"{feature} prediction accuracy: {accuracy:.4f}")
    
    print("Training continuous feature models...")
    for feature in tqdm(continuous_features, desc="Training regression models"):
        if feature not in old_df_enriched.columns:
            print(f"Feature {feature} not found in dataset, skipping...")
            continue
            
        # For continuous features
        X = old_df_enriched[predictors]
        y = old_df_enriched[feature]
        
        # Handle missing values
        X = X.fillna(-1)  # Special value for missing
        
        # Skip if all values are missing
        if y.isnull().all():
            print(f"Skipping {feature} as all values are missing")
            continue
            
        # Fill NaN in target with mean (only for training)
        y = y.fillna(y.mean())
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Base parameters including GPU if available
        base_params = {
            'random_state': 42,
            **lgbm_device_params  # Add GPU parameters if available
        }
        
        # Try a more powerful model with hyperparameter tuning
        if feature in ['danceability', 'energy', 'valence']:
            model = lgbm.LGBMRegressor(
                **base_params, 
                n_estimators=200, 
                learning_rate=0.05, 
                max_depth=7
            )
        elif feature in ['loudness', 'tempo']:
            model = lgbm.LGBMRegressor(
                **base_params, 
                n_estimators=200, 
                learning_rate=0.05, 
                max_depth=9
            )
        else:
            model = lgbm.LGBMRegressor(
                **base_params, 
                n_estimators=150, 
                learning_rate=0.05
            )
        
        model.fit(X_train, y_train)
        continuous_models[feature] = model
        
        # Evaluate
        y_pred = model.predict(X_test)
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        
        print(f"{feature}: MSE = {mse:.4f}, R² = {r2:.4f}")
    
    # Apply models to predict audio features in new dataset
    print("Applying feature engineering to new dataset...")
    new_df_enriched = extract_genre_features(new_df)
    
    # Align features between old and new datasets
    print("Aligning features between datasets...")
    for col in set(predictors) - set(new_df_enriched.columns):
        new_df_enriched[col] = 0  # or appropriate default
    
    # Ensure we only use columns that exist in both datasets
    valid_predictors = [col for col in predictors if col in new_df_enriched.columns]
    
    # Fill missing values
    new_df_enriched = new_df_enriched.fillna(-1)
    
    # Predict each audio feature
    print("Predicting categorical features...")
    for feature, model in tqdm(categorical_models.items(), desc="Predicting categorical features"):
        try:
            new_df_enriched[feature] = model.predict(new_df_enriched[valid_predictors])
        except Exception as e:
            print(f"Error predicting {feature}: {e}")
    
    print("Predicting continuous features...")
    for feature, model in tqdm(continuous_models.items(), desc="Predicting continuous features"):
        try:
            new_df_enriched[feature] = model.predict(new_df_enriched[valid_predictors])
        except Exception as e:
            print(f"Error predicting {feature}: {e}")
    
    # Save the enhanced dataset
    print("Saving predictions to file...")
    new_df_enriched.to_csv('new_dataset_with_predicted_audio.csv', index=False)

# Check if we need to do domain adaptation for engagement metrics
track_id_field = 'track_id' if 'track_id' in old_df.columns else 'id'
engagement_field_old = 'playcount' if 'playcount' in old_df.columns else None
engagement_field_new = 'streams' if 'streams' in new_df.columns else None

if track_id_field in old_df.columns and track_id_field in new_df.columns and engagement_field_old and engagement_field_new:
    # Find overlapping tracks
    overlapping_tracks = set(old_df[track_id_field]).intersection(set(new_df[track_id_field]))
    print(f"Found {len(overlapping_tracks)} overlapping tracks for domain adaptation")
    
    if len(overlapping_tracks) > 0:
        print("Building engagement mapping model...")
        # Build engagement mapping model
        overlap_old = old_df[old_df[track_id_field].isin(overlapping_tracks)]
        overlap_new = new_df[new_df[track_id_field].isin(overlapping_tracks)]
        
        # Create mapping dataframe
        mapping_df = pd.DataFrame({
            'track_id': list(overlapping_tracks),
            'old_engagement': overlap_old.set_index(track_id_field).loc[list(overlapping_tracks), engagement_field_old],
            'new_engagement': overlap_new.set_index(track_id_field).loc[list(overlapping_tracks), engagement_field_new]
        })
        
        # Apply log transformation to handle the scale differences
        mapping_df['log_old'] = np.log1p(mapping_df['old_engagement'])
        mapping_df['log_new'] = np.log1p(mapping_df['new_engagement'])
        
        # Train a model to map between engagement metrics
        from sklearn.linear_model import LinearRegression
        engagement_model = LinearRegression()
        engagement_model.fit(
            mapping_df[['log_old']].values, 
            mapping_df['log_new'].values
        )
        
        # Function to convert between engagement metrics
        def convert_engagement(old_value):
            log_old = np.log1p(old_value)
            log_new = engagement_model.predict([[log_old]])[0]
            return np.expm1(log_new)
        
        # Add converted engagement to old dataset
        print("Converting engagement metrics and applying to dataset...")
        old_df['predicted_streams'] = old_df[engagement_field_old].apply(convert_engagement)
        
        # Evaluate conversion
        print("Engagement conversion evaluation:")
        test_tracks = list(overlapping_tracks)[:10]  # Sample for display
        compare_df = pd.DataFrame({
            'track_id': test_tracks,
            'actual_playcount': overlap_old.set_index(track_id_field).loc[test_tracks, engagement_field_old],
            'actual_streams': overlap_new.set_index(track_id_field).loc[test_tracks, engagement_field_new],
            'predicted_streams': [convert_engagement(overlap_old.set_index(track_id_field).loc[t, engagement_field_old]) 
                                for t in test_tracks]
        })
        print(compare_df)
        
        # Save the enhanced old dataset
        print("Saving dataset with engagement mapping...")
        old_df.to_csv('old_dataset_with_engagement_mapping.csv', index=False)

print("Processing complete. Audio features predicted and datasets enhanced.")