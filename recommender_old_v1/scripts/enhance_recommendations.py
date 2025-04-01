#!/usr/bin/env python
import os
import pandas as pd
import argparse
import logging
from pathlib import Path

def setup_logging(log_level=logging.INFO):
    """Set up logging configuration."""
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler('enhance_recommendations.log')
        ]
    )
    return logging.getLogger('enhance_recommendations')

def load_item_features(item_features_path):
    """
    Load item features from the given path.
    Tries to identify and extract useful metadata for tracks/items.
    """
    logger = logging.getLogger('enhance_recommendations')
    logger.info(f"Loading item features from {item_features_path}")
    
    item_df = pd.read_csv(item_features_path)
    logger.info(f"Loaded {len(item_df)} items with columns: {item_df.columns.tolist()}")
    
    # Check if the metadata file exists in the same directory
    metadata_path = Path(item_features_path).parent / "item_metadata.csv"
    if metadata_path.exists():
        logger.info(f"Found metadata file at {metadata_path}")
        metadata_df = pd.read_csv(metadata_path)
        logger.info(f"Loaded metadata with columns: {metadata_df.columns.tolist()}")
        
        # Merge with item features if there's a common key
        if 'track_id' in item_df.columns and 'track_id' in metadata_df.columns:
            item_df = pd.merge(item_df, metadata_df, on='track_id', how='left')
            logger.info(f"Merged metadata with item features. Final columns: {item_df.columns.tolist()}")
    
    return item_df

def load_recommendations(recommendations_path):
    """Load recommendation CSV file."""
    logger = logging.getLogger('enhance_recommendations')
    logger.info(f"Loading recommendations from {recommendations_path}")
    
    recs_df = pd.read_csv(recommendations_path)
    logger.info(f"Loaded {len(recs_df)} recommendations with columns: {recs_df.columns.tolist()}")
    
    return recs_df

def extract_metadata_from_spotify_sample(spotify_sample_path):
    """Extract metadata fields from Spotify sample data to use as a reference."""
    logger = logging.getLogger('enhance_recommendations')
    logger.info(f"Loading Spotify sample from {spotify_sample_path}")
    
    try:
        sample_df = pd.read_csv(spotify_sample_path)
        logger.info(f"Loaded sample with columns: {sample_df.columns.tolist()}")
        
        # Extract unique track information
        track_metadata = sample_df.drop_duplicates(subset=['track_id'])[
            ['track_id', 'name', 'artist', 'spotify_id', 'spotify_preview_url', 
             'tags', 'genre', 'main_genre', 'year']
        ]
        
        logger.info(f"Extracted metadata for {len(track_metadata)} unique tracks")
        return track_metadata
    except Exception as e:
        logger.error(f"Error reading Spotify sample: {e}")
        return None

def map_item_ids_to_metadata(item_df, spotify_metadata=None):
    """
    Create a mapping from item_id to metadata.
    If Spotify metadata is available, try to map based on common identifiers.
    """
    logger = logging.getLogger('enhance_recommendations')
    
    item_metadata = {}
    
    # Check if we have track_id column
    if 'track_id' in item_df.columns:
        track_id_col = 'track_id'
    else:
        # Assume the last column is the track_id
        track_id_col = item_df.columns[-1]
        logger.warning(f"No explicit track_id column found, using the last column: {track_id_col}")
    
    # Create a mapping table
    for _, row in item_df.iterrows():
        item_id = row[track_id_col]
        
        # Start with basic metadata
        metadata = {
            'track_id': item_id,
            'name': f"Track {item_id}",  # Default name
            'artist': "Unknown Artist",  # Default artist
        }
        
        # Add any additional metadata if available
        for col in item_df.columns:
            if col != track_id_col and not col.startswith('0') and not pd.isna(row[col]):
                # Skip numerical features but keep metadata columns
                if col in ['name', 'artist', 'genre', 'year', 'tags']:
                    metadata[col] = row[col]
        
        item_metadata[item_id] = metadata
    
    # Try to enhance with Spotify metadata if available
    if spotify_metadata is not None:
        enhanced_count = 0
        
        # Try to match based on track_id
        for i, item in item_metadata.items():
            spotify_match = spotify_metadata[spotify_metadata['track_id'] == str(i)]
            
            if len(spotify_match) > 0:
                # Update metadata with Spotify data
                for col in spotify_match.columns:
                    if col in ['name', 'artist', 'genre', 'year', 'tags', 'spotify_id', 'spotify_preview_url']:
                        item_metadata[i][col] = spotify_match.iloc[0][col]
                enhanced_count += 1
        
        logger.info(f"Enhanced {enhanced_count} tracks with Spotify metadata")
    
    return item_metadata

def enhance_recommendations(recommendations_df, item_metadata, output_path):
    """
    Enhance recommendations with item metadata and save to a new CSV.
    """
    logger = logging.getLogger('enhance_recommendations')
    logger.info("Enhancing recommendations with metadata")
    
    # Create a new DataFrame with additional columns
    enhanced_recs = recommendations_df.copy()
    
    # Add metadata columns
    enhanced_recs['name'] = enhanced_recs['item_id'].map(lambda x: item_metadata.get(x, {}).get('name', f"Track {x}"))
    enhanced_recs['artist'] = enhanced_recs['item_id'].map(lambda x: item_metadata.get(x, {}).get('artist', "Unknown Artist"))
    
    # Add other available metadata
    all_metadata_fields = set()
    for meta in item_metadata.values():
        all_metadata_fields.update(meta.keys())
    
    for field in all_metadata_fields:
        if field not in ['track_id', 'name', 'artist']:
            enhanced_recs[field] = enhanced_recs['item_id'].map(
                lambda x: item_metadata.get(x, {}).get(field, None)
            )
    
    # Save the enhanced recommendations
    enhanced_recs.to_csv(output_path, index=False)
    logger.info(f"Saved enhanced recommendations to {output_path}")
    
    return enhanced_recs

def main():
    parser = argparse.ArgumentParser(description='Enhance recommendation CSVs with track metadata')
    parser.add_argument('--recommendations', required=True, help='Path to recommendations CSV file')
    parser.add_argument('--item-features', required=True, help='Path to item features CSV file')
    parser.add_argument('--spotify-sample', help='Path to Spotify sample data for additional metadata')
    parser.add_argument('--output', help='Output path for enhanced recommendations')
    parser.add_argument('--debug', action='store_true', help='Enable debug logging')
    
    args = parser.parse_args()
    
    # Setup logging
    logger = setup_logging(logging.DEBUG if args.debug else logging.INFO)
    logger.info("Starting recommendation enhancement process")
    
    # Set default output path if not specified
    if args.output is None:
        input_path = Path(args.recommendations)
        args.output = str(input_path.parent / f"{input_path.stem}_enhanced.csv")
    
    # Load data
    item_df = load_item_features(args.item_features)
    recs_df = load_recommendations(args.recommendations)
    
    # Load Spotify sample metadata if provided
    spotify_metadata = None
    if args.spotify_sample:
        spotify_metadata = extract_metadata_from_spotify_sample(args.spotify_sample)
    
    # Create metadata mapping
    item_metadata = map_item_ids_to_metadata(item_df, spotify_metadata)
    
    # Enhance recommendations
    enhanced_recs = enhance_recommendations(recs_df, item_metadata, args.output)
    
    # Print sample of enhanced recommendations
    print("\nSample of enhanced recommendations:")
    print(enhanced_recs.head())
    
    logger.info("Recommendation enhancement completed")

if __name__ == "__main__":
    main() 