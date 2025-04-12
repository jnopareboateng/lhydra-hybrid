import pandas as pd
import numpy as np
import logging
import os

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- Configuration ---
INPUT_TRACK_FILE = 'data/processed/spotify_streams_data.csv'
OUTPUT_INTERACTION_FILE = 'data/processed/synthetic_user_history.csv'
OUTPUT_TRACK_FILE = 'data/processed/spotify_tracks_with_ids.csv'
NUM_SYNTHETIC_USERS = 10000
MIN_SONGS_PER_USER = 50
MAX_SONGS_PER_USER = 100
# The number of songs each user listens to is uniformly sampled between these two values
PLAYCOUNT_LAMBDA = 1.5  # Parameter for exponential distribution (higher lambda = more lower playcounts)
# ---------------------

def generate_synthetic_data():
    """Generates synthetic user listening history based on track popularity."""
    logger.info(f"Loading track data from {INPUT_TRACK_FILE}...")
    try:
        tracks_df = pd.read_csv(INPUT_TRACK_FILE)
        # Drop rows with missing essential info if any (like song_name, artist_name)
        tracks_df.dropna(subset=['song_name', 'artist_name', 'playcount'], inplace=True)
        tracks_df.reset_index(drop=True, inplace=True)
    except FileNotFoundError:
        logger.error(f"Error: Input track file not found at {INPUT_TRACK_FILE}")
        return
    except Exception as e:
        logger.error(f"Error loading track data: {e}")
        return

    # Add a unique track_id (using index)
    tracks_df['track_id'] = tracks_df.index
    logger.info(f"Loaded {len(tracks_df)} tracks. Added track_id.")

    # Normalize overall playcounts to use as sampling weights (popularity bias)
    # Add a small epsilon to avoid zero probability for unpopular songs
    total_playcount = tracks_df['playcount'].sum()
    tracks_df['popularity_weight'] = (tracks_df['playcount'] + 1) / (total_playcount + len(tracks_df))

    logger.info(f"Generating {NUM_SYNTHETIC_USERS} synthetic users...")
    user_ids = range(NUM_SYNTHETIC_USERS)
    interactions = []

    for user_id in user_ids:
        if (user_id + 1) % 1000 == 0:
            logger.info(f"Generating interactions for user {user_id + 1}/{NUM_SYNTHETIC_USERS}...")

        # Determine number of unique songs for this user
        num_songs = np.random.randint(MIN_SONGS_PER_USER, MAX_SONGS_PER_USER + 1)

        # Sample songs based on popularity
        try:
            sampled_indices = np.random.choice(
                tracks_df.index,
                size=num_songs,
                replace=False, # Each user listens to unique songs in this batch
                p=tracks_df['popularity_weight']
            )
        except ValueError as e:
            logger.warning(f"Could not sample {num_songs} songs for user {user_id} due to weights issue: {e}. Skipping user or adjusting.")
            # Fallback: Sample uniformly if weights cause issues (e.g., sum not 1)
            if tracks_df['popularity_weight'].sum() > 1.001 or tracks_df['popularity_weight'].sum() < 0.999:
                 logger.warning("Popularity weights do not sum to 1. Sampling uniformly.")
                 tracks_df['popularity_weight'] = np.ones(len(tracks_df)) / len(tracks_df)
                 sampled_indices = np.random.choice(tracks_df.index, size=num_songs, replace=False, p=tracks_df['popularity_weight'])
            else:
                 continue # Skip user if other weight issue

        sampled_track_ids = tracks_df.loc[sampled_indices, 'track_id'].tolist()

        # Generate playcounts for each sampled song
        # Using an exponential distribution shifted by 1 (min playcount is 1)
        playcounts = np.random.exponential(scale=1.0/PLAYCOUNT_LAMBDA, size=num_songs)
        playcounts = np.maximum(1, np.round(playcounts + 1)).astype(int) # Ensure min playcount is 1

        # Add to interactions list
        for i in range(num_songs):
            interactions.append({
                'user_id': user_id,
                'track_id': sampled_track_ids[i],
                'playcount': playcounts[i]
            })

    interactions_df = pd.DataFrame(interactions)
    logger.info(f"Generated {len(interactions_df)} total interactions.")

    # Ensure output directories exist
    os.makedirs(os.path.dirname(OUTPUT_INTERACTION_FILE), exist_ok=True)
    os.makedirs(os.path.dirname(OUTPUT_TRACK_FILE), exist_ok=True)

    # Save the synthetic interactions
    logger.info(f"Saving synthetic interactions to {OUTPUT_INTERACTION_FILE}...")
    interactions_df.to_csv(OUTPUT_INTERACTION_FILE, index=False)

    # Save the tracks data with the added track_id (dropping the temporary weight column)
    logger.info(f"Saving updated track data with track_id to {OUTPUT_TRACK_FILE}...")
    tracks_df.drop(columns=['popularity_weight'], inplace=True)
    tracks_df.to_csv(OUTPUT_TRACK_FILE, index=False)

    logger.info("Synthetic data generation complete.")

if __name__ == "__main__":
    generate_synthetic_data()
