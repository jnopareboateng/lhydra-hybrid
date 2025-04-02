#!/usr/bin/env python
"""
Script to extract user and track data from Spotify sample dataset.

This script reads the Spotify sample dataset, extracts unique users and tracks,
and saves them to separate CSV files for use with the recommendation system.
"""

import os
import sys
import argparse
import logging
import pandas as pd
import numpy as np
from pathlib import Path

# Add project root to path to allow imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from utils.logger import setup_logger

def extract_user_data(df):
    """
    Extract unique user data from the dataset.
    
    Args:
        df (pd.DataFrame): Input dataset
        
    Returns:
        pd.DataFrame: DataFrame with unique user data
    """
    # Identify user-related columns
    user_columns = [col for col in df.columns if col.startswith('user_')]
    
    # If no user columns found, extract based on common patterns
    if not user_columns:
        potential_user_columns = ['user_id', 'user', 'subscriber_id', 'subscriber', 'listener_id', 'listener']
        user_columns = [col for col in potential_user_columns if col in df.columns]
    
    # Add 'user_id' if not present but there's a column that might be a user identifier
    if 'user_id' not in user_columns:
        for col in ['user', 'subscriber_id', 'subscriber', 'listener_id', 'listener']:
            if col in df.columns:
                df['user_id'] = df[col]
                user_columns.append('user_id')
                break
    
    # If still no user_id, create one based on unique combinations of user columns
    if 'user_id' not in user_columns and user_columns:
        df['user_id'] = df.groupby(user_columns).ngroup().astype(str)
        user_columns.append('user_id')
    
    # If no user columns found, create user_id based on unique user patterns in listening history
    if not user_columns:
        logger.warning("No explicit user columns found. Creating synthetic user IDs.")
        # Try to find columns that might identify listeners
        if 'session_id' in df.columns:
            df['user_id'] = df['session_id']
        else:
            # Create sequential user IDs
            df['user_id'] = range(1, len(df) + 1)
        user_columns = ['user_id']
    
    # Get unique user data
    user_df = df[user_columns].drop_duplicates().reset_index(drop=True)
    
    # Ensure user_id is the first column
    if 'user_id' in user_df.columns and user_df.columns[0] != 'user_id':
        cols = ['user_id'] + [col for col in user_df.columns if col != 'user_id']
        user_df = user_df[cols]
    
    # Add demographic data if available
    demographic_cols = ['age', 'gender', 'country', 'city', 'subscription_type', 'premium']
    for col in demographic_cols:
        if col in df.columns:
            # Get the most common value for each user
            temp = df.groupby('user_id')[col].agg(lambda x: x.mode().iloc[0] if not x.mode().empty else np.nan)
            user_df = user_df.merge(temp.reset_index(), on='user_id', how='left')
    
    return user_df

def extract_track_data(df):
    """
    Extract unique track data from the dataset.
    
    Args:
        df (pd.DataFrame): Input dataset
        
    Returns:
        pd.DataFrame: DataFrame with unique track data
    """
    # Identify track-related columns
    track_columns = [col for col in df.columns if col.startswith('track_') or col.startswith('song_')]
    
    # If no track columns found, extract based on common patterns
    if not track_columns:
        potential_track_columns = ['track_id', 'song_id', 'track', 'song', 'item_id', 'item']
        track_columns = [col for col in potential_track_columns if col in df.columns]
    
    # Add 'track_id' if not present but there's a column that might be a track identifier
    if 'track_id' not in track_columns:
        for col in ['song_id', 'track', 'song', 'item_id', 'item']:
            if col in df.columns:
                df['track_id'] = df[col]
                track_columns.append('track_id')
                break
    
    # If still no track_id, create one based on track name and artist if available
    if 'track_id' not in track_columns:
        track_name_cols = [col for col in ['track_name', 'song_name', 'title'] if col in df.columns]
        artist_cols = [col for col in ['artist', 'artist_name', 'performer'] if col in df.columns]
        
        if track_name_cols and artist_cols:
            df['track_id'] = df[track_name_cols[0]] + ' - ' + df[artist_cols[0]]
            track_columns.append('track_id')
        elif track_name_cols:
            df['track_id'] = df[track_name_cols[0]]
            track_columns.append('track_id')
        else:
            # Create sequential track IDs
            logger.warning("No explicit track columns found. Creating synthetic track IDs.")
            unique_tracks = df.drop_duplicates(subset=df.columns)
            df['track_id'] = unique_tracks.index.astype(str)
            track_columns = ['track_id']
    
    # Add common track metadata columns if they exist
    metadata_cols = ['name', 'title', 'track_name', 'song_name', 
                     'artist', 'artist_name', 'performer',
                     'album', 'album_name', 
                     'genre', 'category',
                     'release_date', 'release_year', 'year',
                     'duration', 'length', 'tempo', 'popularity']
    
    for col in metadata_cols:
        if col in df.columns and col not in track_columns:
            track_columns.append(col)
    
    # Get unique track data
    track_df = df[track_columns].drop_duplicates().reset_index(drop=True)
    
    # Ensure track_id is the first column
    if 'track_id' in track_df.columns and track_df.columns[0] != 'track_id':
        cols = ['track_id'] + [col for col in track_df.columns if col != 'track_id']
        track_df = track_df[cols]
    
    # Standardize column names if needed
    rename_dict = {}
    for col in track_df.columns:
        if col in ['name', 'title', 'song_name'] and 'track_name' not in track_df.columns:
            rename_dict[col] = 'track_name'
        elif col in ['artist_name', 'performer'] and 'artist' not in track_df.columns:
            rename_dict[col] = 'artist'
        elif col in ['album_name'] and 'album' not in track_df.columns:
            rename_dict[col] = 'album'
        elif col in ['category'] and 'genre' not in track_df.columns:
            rename_dict[col] = 'genre'
        elif col in ['release_year', 'year'] and 'year' not in track_df.columns:
            rename_dict[col] = 'year'
        elif col in ['length'] and 'duration' not in track_df.columns:
            rename_dict[col] = 'duration'
    
    if rename_dict:
        track_df = track_df.rename(columns=rename_dict)
    
    # Add calculated features if possible
    if 'year' not in track_df.columns and 'release_date' in track_df.columns:
        try:
            track_df['year'] = pd.to_datetime(track_df['release_date']).dt.year
        except:
            pass
    
    # Add aggregated listening data if available
    if 'listen_count' in df.columns:
        listen_counts = df.groupby('track_id')['listen_count'].sum().reset_index()
        track_df = track_df.merge(listen_counts, on='track_id', how='left')
    
    return track_df

def extract_interactions(df, user_df, track_df):
    """
    Extract user-track interactions from the dataset.
    
    Args:
        df (pd.DataFrame): Input dataset
        user_df (pd.DataFrame): User DataFrame
        track_df (pd.DataFrame): Track DataFrame
        
    Returns:
        pd.DataFrame: DataFrame with user-track interactions
    """
    # Identify necessary columns
    interactions_cols = ['user_id', 'track_id']
    
    # Add rating/interaction columns if they exist
    rating_cols = [col for col in ['rating', 'like', 'liked', 'score', 'play_count', 'listen_count', 'listen_time', 'playback_seconds'] 
                  if col in df.columns]
    
    interactions_cols.extend(rating_cols)
    
    # Create interactions dataframe
    interactions_df = df[interactions_cols].copy()
    
    # If no explicit rating/like column, try to create one
    if not any(col in interactions_df.columns for col in ['rating', 'like', 'liked']):
        if 'listen_count' in interactions_df.columns:
            # Convert listen count to binary like (1 if above median, else 0)
            median_listens = interactions_df['listen_count'].median()
            interactions_df['liked'] = (interactions_df['listen_count'] > median_listens).astype(int)
        elif 'listen_time' in interactions_df.columns or 'playback_seconds' in interactions_df.columns:
            # Use listen time as proxy for interest
            time_col = 'listen_time' if 'listen_time' in interactions_df.columns else 'playback_seconds'
            median_time = interactions_df[time_col].median()
            interactions_df['liked'] = (interactions_df[time_col] > median_time).astype(int)
        else:
            # No explicit rating, assume all interactions are positive
            interactions_df['liked'] = 1
    
    # Ensure liked column is properly named
    if 'liked' not in interactions_df.columns and 'like' in interactions_df.columns:
        interactions_df['liked'] = interactions_df['like']
    elif 'liked' not in interactions_df.columns and 'rating' in interactions_df.columns:
        # Convert ratings to binary likes (assume ratings > 3 on 5-point scale are positive)
        max_rating = interactions_df['rating'].max()
        if max_rating <= 1:  # Already binary
            interactions_df['liked'] = interactions_df['rating']
        elif max_rating <= 5:  # Assume 5-point scale
            interactions_df['liked'] = (interactions_df['rating'] > 3).astype(int)
        else:  # Assume 10-point scale
            interactions_df['liked'] = (interactions_df['rating'] > 6).astype(int)
    
    return interactions_df

def main():
    """Main function to extract data from Spotify sample dataset."""
    parser = argparse.ArgumentParser(description="Extract user and track data from Spotify sample dataset")
    parser.add_argument("--input", type=str, default="data/spotify_sample_data.csv",
                       help="Path to Spotify sample dataset CSV file")
    parser.add_argument("--output-dir", type=str, default="data",
                       help="Directory to save extracted data files")
    parser.add_argument("--log-level", type=str, choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
                       default="INFO", help="Set the logging level")
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Set up logging
    log_dir = "logs"
    os.makedirs(log_dir, exist_ok=True)
    
    global logger
    logger = setup_logger(
        name="extract_data",
        log_file=os.path.join(log_dir, "extract_data.log"),
        level=getattr(logging, args.log_level)
    )
    
    try:
        logger.info("Starting data extraction")
        
        # Load the dataset
        logger.info(f"Loading dataset from {args.input}")
        df = pd.read_csv(args.input)
        logger.info(f"Loaded dataset with {len(df)} rows and {len(df.columns)} columns")
        
        # Display sample and column information
        logger.info(f"Columns in dataset: {', '.join(df.columns)}")
        
        # Extract user data
        logger.info("Extracting user data")
        user_df = extract_user_data(df)
        logger.info(f"Extracted {len(user_df)} unique users")
        
        # Extract track data
        logger.info("Extracting track data")
        track_df = extract_track_data(df)
        logger.info(f"Extracted {len(track_df)} unique tracks")
        
        # Extract interactions
        logger.info("Extracting user-track interactions")
        interactions_df = extract_interactions(df, user_df, track_df)
        logger.info(f"Extracted {len(interactions_df)} user-track interactions")
        
        # Save the extracted data
        user_file = os.path.join(args.output_dir, "users.csv")
        track_file = os.path.join(args.output_dir, "tracks.csv")
        interactions_file = os.path.join(args.output_dir, "interactions.csv")
        
        user_df.to_csv(user_file, index=False)
        track_df.to_csv(track_file, index=False)
        interactions_df.to_csv(interactions_file, index=False)
        
        logger.info(f"User data saved to {user_file}")
        logger.info(f"Track data saved to {track_file}")
        logger.info(f"Interaction data saved to {interactions_file}")
        
        # Print summary
        print(f"\nData extraction completed successfully:")
        print(f"- Extracted {len(user_df)} unique users saved to {user_file}")
        print(f"- Extracted {len(track_df)} unique tracks saved to {track_file}")
        print(f"- Extracted {len(interactions_df)} interactions saved to {interactions_file}")
        print("\nYou can now use these files with the recommendation scripts:")
        print("python recommender/scripts/generate_recommendations.py --model [model_path] --config [config_path] " +
              f"--user-data {user_file} --track-data {track_file}")
        
        return 0
    
    except Exception as e:
        logger.error(f"Error extracting data: {str(e)}", exc_info=True)
        return 1

if __name__ == "__main__":
    sys.exit(main()) 