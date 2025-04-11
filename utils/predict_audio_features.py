import os
import pandas as pd
import numpy as np
import joblib
import logging
import argparse
from tag_audio_features_prediction import TagBasedAudioFeaturePredictor

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def load_model(model_path):
    """Load the trained model from disk."""
    try:
        logger.info(f"Loading model from {model_path}")
        # Create a new predictor
        predictor = TagBasedAudioFeaturePredictor(use_pretrained=True, include_metadata=True)
        
        # Load the model using the built-in method which properly sets up all components
        predictor.load(model_path)
        
        # Verify model loaded correctly
        if predictor.model and predictor.feature_names:
            logger.info(f"Model loaded successfully with {len(predictor.feature_names)} features")
            logger.info(f"Audio features to predict: {predictor.AUDIO_FEATURES}")
            
            # Check if we're using GloVe embeddings
            if predictor.use_pretrained:
                logger.info("Using GloVe embeddings for feature extraction")
                # If embedding_model is empty, load it
                if not predictor.embedding_model:
                    logger.info("Loading GloVe embeddings...")
                    predictor._download_glove_embeddings()
            else:
                logger.info("Using TF-IDF vectorization for feature extraction")
        else:
            logger.error("Model loading failed or model is incomplete")
            return None
            
        return predictor
    except Exception as e:
        logger.error(f"Error loading model: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return None

def load_data(input_file):
    """Load input data for prediction."""
    try:
        logger.info(f"Loading data from {input_file}")
        if input_file.endswith('.csv'):
            df = pd.read_csv(input_file)
        elif input_file.endswith(('.xls', '.xlsx')):
            df = pd.read_excel(input_file)
        else:
            raise ValueError("Unsupported file format. Use CSV or Excel.")
        
        logger.info(f"Loaded {len(df)} records from {input_file}")
        return df
    except Exception as e:
        logger.error(f"Error loading data: {e}")
        return None

def validate_input_data(df):
    """Ensure input data has required columns."""
    # Handle "Artist and Title" column that contains both artist and track name
    if 'Artist and Title' in df.columns:
        logger.info("Found 'Artist and Title' column, extracting artist and track name...")
        # Split the "Artist - Title" format
        artist_title = df['Artist and Title'].str.split(' - ', n=1, expand=True)
        if 'artist' not in df.columns:
            df['artist'] = artist_title[0]
        if 'name' not in df.columns:
            df['name'] = artist_title[1]
    
    # Check for tag or genre information
    if 'tags' in df.columns:
        logger.info("Found 'tags' column in the data")
    elif 'genres' in df.columns:
        logger.info("Found 'genres' column in the data")
    else:
        logger.warning("No tag or genre information found. Predictions may be less accurate.")
        # Add empty tags column
        df['tags'] = ""
    
    # Check for artist information
    if 'artist' not in df.columns and 'Artist' in df.columns:
        logger.info("Renaming 'Artist' to 'artist'")
        df['artist'] = df['Artist']
    elif 'artist' not in df.columns:
        logger.warning("No artist information found. Using empty values.")
        df['artist'] = ""
    
    # Check for track name information
    if 'name' not in df.columns and 'track_name' in df.columns:
        logger.info("Renaming 'track_name' to 'name'")
        df['name'] = df['track_name']
    elif 'name' not in df.columns and 'title' in df.columns:
        logger.info("Renaming 'title' to 'name'")
        df['name'] = df['title']
    elif 'name' not in df.columns and 'Title' in df.columns:
        logger.info("Renaming 'Title' to 'name'")
        df['name'] = df['Title']
    elif 'name' not in df.columns:
        logger.warning("No track name information found. Using empty values.")
        df['name'] = ""
    
    return df

def make_predictions(predictor, df, sample_size=None):
    """Make predictions for audio features with option to use a sample."""
    try:
        # Optionally use a sample for predictions to save time
        if sample_size and sample_size < len(df):
            logger.info(f"Using a sample of {sample_size} records for prediction")
            sample_df = df.sample(sample_size, random_state=42)
        else:
            sample_df = df
            
        logger.info(f"Making predictions for {len(sample_df)} tracks...")
        
        # Predict audio features
        try:
            predictions = predictor.predict(sample_df)
        except Exception as e:
            logger.error(f"Error during prediction: {e}")
            import traceback
            logger.error(traceback.format_exc())
            raise
        
        # Retain all original columns by merging with the predictions
        logger.info("Merging predictions with original data...")
        
        # Format decimal places appropriately for each audio feature
        for feature in predictions.columns:
            if feature in ['danceability', 'energy', 'speechiness', 'acousticness', 
                          'instrumentalness', 'liveness', 'valence']:
                # Normalized features (0-1) - 3 decimal places
                predictions[feature] = predictions[feature].round(3)
            elif feature in ['loudness']:
                # Loudness typically has 1 decimal place
                predictions[feature] = predictions[feature].round(1)
            elif feature in ['tempo']:
                # Tempo typically has 1 decimal place
                predictions[feature] = predictions[feature].round(1)
            elif feature in ['key', 'mode']:
                # Categorical features should be integers
                predictions[feature] = predictions[feature].round().astype(int)
        
        # Create result dataframe with all original columns
        result_df = sample_df.copy()
        
        # Add predicted features, overwriting any existing audio features
        for feature in predictor.AUDIO_FEATURES:
            if feature in predictions.columns:
                result_df[feature] = predictions[feature].values
        
        logger.info(f"Successfully generated predictions for {len(result_df)} tracks")
        # Return the complete dataframe with original data and predictions
        return result_df
    except Exception as e:
        logger.error(f"Error making predictions: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return None

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Predict audio features from music metadata.')
    parser.add_argument('--input', '-i', default= 'data/raw/spotify_full_list_20102023.csv')
    parser.add_argument('--output', '-o', default='predicted_audio_features.csv', 
                        help='Output file path (default: predicted_audio_features.csv)')
    parser.add_argument('--model', '-m', default='models/tag_audio_predictor_model.joblib',
                        help='Path to model file (default: models/tag_audio_predictor_model.joblib)')
    parser.add_argument('--sample', '-s', type=int, default=100,
                        help='Sample size for prediction (default: 100)')
    return parser.parse_args()

def main():
    """Main function to predict audio features."""
    # Parse command line arguments
    args = parse_args()
    
    # Load the model
    model_path = args.model
    predictor = load_model(model_path)
    if predictor is None:
        return
    
    # Load the data
    input_file = args.input
    df = load_data(input_file)
    if df is None:
        return
    
    # Validate and prepare the input data
    df = validate_input_data(df)
    
    # Make predictions (with optional sampling)
    predictions = make_predictions(predictor, df, args.sample)
    if predictions is None:
        return
    
    # Save the predictions
    output_file = args.output
    logger.info(f"Saving predictions to {output_file}")
    predictions.to_csv(output_file, index=False)
    
    # Display summary information
    logger.info("\nPrediction Summary:")
    logger.info(f"Number of records processed: {len(predictions)}")
    logger.info(f"Audio features predicted: {', '.join(predictor.AUDIO_FEATURES)}")
    
    # Show sample statistics
    logger.info("\nAudio feature statistics:")
    for feature in predictor.AUDIO_FEATURES:
        if feature in predictions.columns:
            mean_val = predictions[feature].mean()
            std_val = predictions[feature].std()
            min_val = predictions[feature].min()
            max_val = predictions[feature].max()
            
            # Format the statistics based on feature type
            if feature in ['danceability', 'energy', 'speechiness', 'acousticness', 
                          'instrumentalness', 'liveness', 'valence']:
                # Normalized features (0-1)
                logger.info(f"{feature}: Mean = {mean_val:.3f}, StdDev = {std_val:.3f}, Range = [{min_val:.3f}, {max_val:.3f}]")
            elif feature in ['loudness', 'tempo']:
                # Continuous features with 1 decimal place
                logger.info(f"{feature}: Mean = {mean_val:.1f}, StdDev = {std_val:.1f}, Range = [{min_val:.1f}, {max_val:.1f}]")
            elif feature in ['key', 'mode']:
                # Categorical features as integers
                logger.info(f"{feature}: Mean = {mean_val:.0f}, StdDev = {std_val:.1f}, Range = [{min_val:.0f}, {max_val:.0f}]")
            else:
                # Default format with 4 decimal places
                logger.info(f"{feature}: Mean = {mean_val:.4f}, StdDev = {std_val:.4f}, Range = [{min_val:.4f}, {max_val:.4f}]")
    
    logger.info(f"\nPredictions saved to {output_file}")
    
    # Print sample of predictions with appropriate formatting
    logger.info("\nSample of predictions (first 5 rows):")
    pd.set_option('display.float_format', lambda x: f'{x:.3f}' if 0 <= x <= 1 else 
                                                 (f'{x:.1f}' if isinstance(x, float) else f'{x}'))
    sample_features = predictions[predictor.AUDIO_FEATURES].head(5)
    logger.info("\n" + sample_features.to_string())
    
    # Also show a sample with the predicted features alongside original track info
    display_cols = []
    # Add ID and basic info columns
    for col in ['Unnamed: 0', 'Artist and Title', 'artist', 'name']:
        if col in predictions.columns:
            display_cols.append(col)
    # Add a couple of audio features
    for feature in ['danceability', 'energy', 'tempo', 'key']:
        if feature in predictor.AUDIO_FEATURES:
            display_cols.append(feature)
    
    # Show sample with original data and predicted features
    if display_cols:
        logger.info("\nSample with original track info and predicted features:")
        sample_with_info = predictions[display_cols].head(5)
        logger.info("\n" + sample_with_info.to_string())

if __name__ == "__main__":
    main() 