import numpy as np
import pandas as pd
import random
from collections import Counter
import logging
import os

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
logger.info("Starting demographic generation process...")

# Load the datasets
users_history = pd.read_csv("data/raw/User Listening History.csv")
music_info = pd.read_csv("data/raw/Music Info.csv")
    
logger.info("User history shape: %s", users_history.shape)
logger.info("Music info shape: %s", music_info.shape)

# Filter music to songs from 2010 and above
original_count = len(music_info)
music_info = music_info[music_info['year'] >= 2010]
filtered_count = len(music_info)
filter_percentage = (filtered_count / original_count) * 100

logger.info(f"Filtered to songs from 2010 and above: {filtered_count} records ({filter_percentage:.2f}% of original)")
logger.info(f"Filtered music info shape: {music_info.shape}")

# Check if the filtered dataset meets our size requirement (over 50K and at least 40% of original)
if filtered_count < 50000 or filter_percentage < 40:
    logger.warning(f"Filtered dataset is too small: {filtered_count} records ({filter_percentage:.2f}% of original)")
    logger.info("Using top 100K songs instead of year filtering")
    # If filtered dataset is too small, use the top 100K songs instead
    music_info = pd.read_csv("data/raw/Music Info.csv")
    music_info = music_info.sort_values('year', ascending=False).head(50000)
    logger.info(f"Using top 50K songs by year: {music_info.shape}")

# Get unique users and sample if too many
unique_users = users_history["user_id"].unique()
n_users = len(unique_users)
logger.info("Total unique users: %d", n_users)

# Take a reasonable sample - adjust based on your computational resources
sample_size = min(n_users, 10000)  # Limit to 10K users
sampled_users = np.random.choice(unique_users, sample_size, replace=False)
logger.info("Working with sample of %d users", sample_size)

# Filter user history to just sampled users
sampled_history = users_history[users_history["user_id"].isin(sampled_users)]

# Further filter user history to match with filtered music_info
sampled_history = sampled_history[sampled_history['track_id'].isin(music_info['track_id'])]

logger.info("Filtered user history shape: %s", sampled_history.shape)
logger.info("Sampled user history head:\n%s", sampled_history.head())

# Check for missing values in the music info dataset
missing_values = music_info.isnull().sum()
logger.info("Missing values in music info dataset:\n%s", missing_values[missing_values > 0])

# Extract genre information - create a simplified genre map
def extract_main_genre(tag_string):
    if pd.isna(tag_string):
        return "unknown"

    tags = str(tag_string).lower().split(",")

    # Define genre hierarchy with primary genres
    genre_map = {
        "rock": ["rock", "metal", "punk", "alternative", "grunge", "hard rock", "classic rock"],
        "electronic": ["electronic", "edm", "house", "techno", "dubstep", "trance"],
        "hip_hop": ["hip-hop", "rap", "trap", "drill"],
        "pop": ["pop", "dance pop", "k-pop", "electropop", "synth pop"],
        "classical": ["classical", "baroque", "orchestra", "piano"],
        "jazz": ["jazz", "bebop", "fusion", "blues", "funk"],
        "folk": ["folk", "acoustic", "singer-songwriter", "bluegrass", "americana"],
        "latin": ["latin", "reggae", "salsa", "bachata", "afrobeat"],
        "rb_soul": ["r&b", "soul", "motown"],
        "country": ["country", "bluegrass", "americana"],
        "religious": ["christian", "gospel", "worship", "spiritual"],
    }

    # Check for matches
    for main_genre, sub_genres in genre_map.items():
        if any(sub in " ".join(tags) for sub in sub_genres):
            return main_genre

    return "other"

# Add main genre to music info
logger.info("Extracting main genres...")
music_info["main_genre"] = music_info["tags"].apply(extract_main_genre)

# Merge user history with full music info to get all required columns
logger.info("Merging user history with music info...")
full_user_tracks = sampled_history.merge(
    music_info,
    on="track_id",
    how="inner"
)

# Create user preference profile based on genres and all audio features
logger.info("Building user preferences...")

# List of all audio features to include
audio_features = [
    "danceability", "energy", "key", "loudness", "mode", 
    "speechiness", "acousticness", "instrumentalness", 
    "liveness", "valence", "tempo", "time_signature"
]

# Verify which audio features are actually in the dataset
available_features = [feat for feat in audio_features if feat in music_info.columns]
if len(available_features) < len(audio_features):
    missing_features = set(audio_features) - set(available_features)
    logger.warning(f"Missing audio features in dataset: {missing_features}")
    logger.info(f"Using available audio features: {available_features}")

# Get genre preferences for each user (with simplified processing)
user_genre_counts = full_user_tracks.groupby(["user_id", "main_genre"])["playcount"].sum()
user_genre_prefs = user_genre_counts.unstack(fill_value=0)

# Normalize to get percentages
row_sums = user_genre_prefs.sum(axis=1)
user_genre_prefs = user_genre_prefs.div(row_sums, axis=0) * 100
user_genre_prefs.fillna(0, inplace=True)

# Add top genre column
user_genre_prefs["top_genre"] = user_genre_prefs.idxmax(axis=1)
user_genre_prefs["top_genre"] = user_genre_prefs["top_genre"].fillna("unknown")

# Get average audio features for each user (optimized implementation)
user_audio_features = pd.DataFrame(index=user_genre_prefs.index)
user_audio_features["total_playcount"] = full_user_tracks.groupby("user_id")["playcount"].sum()

# Calculate weighted averages for audio features
for feature in available_features:
    if feature in full_user_tracks.columns:
        # More efficient weighted average calculation
        weighted_feature = full_user_tracks["playcount"] * full_user_tracks[feature]
        user_audio_features[f"avg_{feature}"] = weighted_feature.groupby(full_user_tracks["user_id"]).sum() / user_audio_features["total_playcount"]

# Reset index and merge user features
user_genre_prefs = user_genre_prefs.reset_index()
user_audio_features = user_audio_features.reset_index()
user_features = user_genre_prefs.merge(user_audio_features, on="user_id")

# Define simplified demographic correlations
genre_age_map = {
    "rock": [0.15, 0.30, 0.25, 0.15, 0.10, 0.05],  # More middle-aged
    "electronic": [0.40, 0.35, 0.15, 0.05, 0.03, 0.02],  # Young adult
    "hip_hop": [0.45, 0.30, 0.15, 0.05, 0.03, 0.02],  # Youngest demographic
    "pop": [0.35, 0.30, 0.15, 0.10, 0.05, 0.05],  # Younger skew
    "classical": [0.10, 0.15, 0.15, 0.20, 0.20, 0.20],  # Oldest demographic
    "jazz": [0.05, 0.15, 0.20, 0.25, 0.20, 0.15],  # Older demographic
    "folk": [0.15, 0.20, 0.20, 0.20, 0.15, 0.10],  # Balanced but older
    "latin": [0.30, 0.30, 0.20, 0.10, 0.05, 0.05],  # Younger adult
    "rb_soul": [0.20, 0.25, 0.25, 0.15, 0.10, 0.05],  # Mixed age groups
    "country": [0.15, 0.20, 0.25, 0.20, 0.15, 0.05],  # Middle-aged skew
    "religious": [0.15, 0.20, 0.20, 0.20, 0.15, 0.10],  # Balanced distribution
    "other": [0.30, 0.30, 0.15, 0.10, 0.10, 0.05],  # Generic distribution
    "unknown": [0.30, 0.30, 0.15, 0.10, 0.10, 0.05],  # Generic distribution
}

genre_gender_map = {
    "rock": {"Male": 0.60, "Female": 0.40},
    "electronic": {"Male": 0.60, "Female": 0.40},
    "hip_hop": {"Male": 0.55, "Female": 0.45},
    "pop": {"Male": 0.35, "Female": 0.65},
    "classical": {"Male": 0.48, "Female": 0.52},
    "jazz": {"Male": 0.55, "Female": 0.45},
    "folk": {"Male": 0.45, "Female": 0.55},
    "latin": {"Male": 0.40, "Female": 0.60},
    "rb_soul": {"Male": 0.45, "Female": 0.55},
    "country": {"Male": 0.48, "Female": 0.52},
    "religious": {"Male": 0.45, "Female": 0.55},
    "other": {"Male": 0.50, "Female": 0.50},
    "unknown": {"Male": 0.50, "Female": 0.50},
}

genre_region_map = {
    "rock": {
        "Europe": 0.35,
        "North America": 0.35,
        "Latin America": 0.15,
        "Rest of World": 0.15,
    },
    "electronic": {
        "Europe": 0.40,
        "North America": 0.25,
        "Latin America": 0.15,
        "Rest of World": 0.20,
    },
    "hip_hop": {
        "Europe": 0.20,
        "North America": 0.45,
        "Latin America": 0.20,
        "Rest of World": 0.15,
    },
    "pop": {
        "Europe": 0.30,
        "North America": 0.30,
        "Latin America": 0.20,
        "Rest of World": 0.20,
    },
    "classical": {
        "Europe": 0.40,
        "North America": 0.30,
        "Latin America": 0.10,
        "Rest of World": 0.20,
    },
    "jazz": {
        "Europe": 0.30,
        "North America": 0.40,
        "Latin America": 0.15,
        "Rest of World": 0.15,
    },
    "folk": {
        "Europe": 0.35,
        "North America": 0.40,
        "Latin America": 0.10,
        "Rest of World": 0.15,
    },
    "latin": {
        "Europe": 0.15,
        "North America": 0.20,
        "Latin America": 0.55,
        "Rest of World": 0.10,
    },
    "rb_soul": {
        "Europe": 0.20,
        "North America": 0.45,
        "Latin America": 0.20,
        "Rest of World": 0.15,
    },
    "country": {
        "Europe": 0.15,
        "North America": 0.60,
        "Latin America": 0.15,
        "Rest of World": 0.10,
    },
    "religious": {
        "Europe": 0.25,
        "North America": 0.35,
        "Latin America": 0.25,
        "Rest of World": 0.15,
    },
    "other": {
        "Europe": 0.27,
        "North America": 0.28,
        "Latin America": 0.22,
        "Rest of World": 0.23,
    },
    "unknown": {
        "Europe": 0.27,
        "North America": 0.28,
        "Latin America": 0.22,
        "Rest of World": 0.23,
    },
}

# Define age groups
age_groups = [
    (18, 24),  # 31.51%
    (25, 34),  # 31.41%
    (35, 44),  # ~15%
    (45, 54),  # ~10%
    (55, 64),  # ~8%
    (65, 100),  # ~4%
]

# Use audio features to refine segmentation with all available features
def adjust_demographics(row):
    top_genre = row["top_genre"]
    
    # Get base probabilities from genre
    age_probs = genre_age_map.get(top_genre, genre_age_map["other"]).copy()
    gender_probs = genre_gender_map.get(top_genre, genre_gender_map["other"]).copy()
    region_probs = genre_region_map.get(top_genre, genre_region_map["other"]).copy()

    # Adjust based on audio features if available
    if "avg_energy" in row and row["avg_energy"] > 0.7:
        for i in range(2):  # Boost first two age groups
            age_probs[i] *= 1.2

    if "avg_valence" in row and row["avg_valence"] > 0.6:
        gender_probs["Female"] *= 1.1

    if "avg_danceability" in row and row["avg_danceability"] > 0.7:
        region_probs["Latin America"] *= 1.2
        region_probs["Europe"] *= 1.1
        
    if "avg_acousticness" in row and row["avg_acousticness"] > 0.7:
        for i in range(3, 6):  # Boost older age groups
            age_probs[i] *= 1.15
            
    if "avg_speechiness" in row and row["avg_speechiness"] > 0.3:
        for i in range(2):  # Boost youngest age groups
            age_probs[i] *= 1.2
        region_probs["North America"] *= 1.1

    # Normalize probabilities efficiently
    age_probs = np.array(age_probs)
    age_probs = age_probs / age_probs.sum()

    gender_values = np.array(list(gender_probs.values()))
    gender_values = gender_values / gender_values.sum()
    gender_probs = {k: gender_values[i] for i, k in enumerate(gender_probs.keys())}

    region_values = np.array(list(region_probs.values()))
    region_values = region_values / region_values.sum()
    region_probs = {k: region_values[i] for i, k in enumerate(region_probs.keys())}

    return age_probs, gender_probs, region_probs

# Create country lists by region
countries_by_region = {
    "Europe": ["UK", "Germany", "France", "Spain", "Italy", "Sweden", "Netherlands", "Poland"],
    "North America": ["USA", "Canada"],
    "Latin America": ["Brazil", "Mexico", "Argentina", "Colombia", "Chile"],
    "Rest of World": ["Japan", "Australia", "India", "South Korea", "South Africa", "UAE"],
}

# Generate demographics
logger.info("Generating demographics...")
demographics = []

for _, row in user_features.iterrows():
    user_id = row["user_id"]

    # Get adjusted demographic probabilities
    age_probs, gender_probs, region_probs = adjust_demographics(row)

    # Assign age
    age_group_idx = np.random.choice(len(age_groups), p=age_probs)
    age_range = age_groups[age_group_idx]
    age = np.random.randint(age_range[0], age_range[1] + 1)

    # Assign gender
    gender = np.random.choice(list(gender_probs.keys()), p=list(gender_probs.values()))

    # Assign region
    region = np.random.choice(list(region_probs.keys()), p=list(region_probs.values()))

    # Assign country
    country = np.random.choice(countries_by_region[region])

    # Assign listening hours based on age and gender
    base_hours = 0
    if age < 25:
        base_hours = 25 + np.random.normal(5, 3)
    elif age < 35:
        base_hours = 20 + np.random.normal(4, 3)
    elif age < 45:
        base_hours = 15 + np.random.normal(5, 2)
    else:
        base_hours = 10 + np.random.normal(5, 2)

    # Gen Z women tend to listen more
    if age < 25 and gender == "Female":
        base_hours *= 1.2

    monthly_hours = max(5, min(60, base_hours))  # Cap between 5 and 60 hours

    # Calculate genre diversity (number of genres with >5% listening)
    genre_cols = [col for col in user_genre_prefs.columns if col != "top_genre" and col != "user_id"]
    diversity_score = sum(1 for col in genre_cols if row[col] > 5)

    # Create demographic record
    demo_data = {
        "user_id": user_id,
        "age": age,
        "gender": gender,
        "region": region,
        "country": country,
        "monthly_hours": monthly_hours,
        "top_genre": row["top_genre"],
        "genre_diversity": diversity_score,
    }
    
    # Add all audio features
    for feature in available_features:
        feature_col = f"avg_{feature}"
        if feature_col in row:
            demo_data[feature_col] = row[feature_col]
    
    demographics.append(demo_data)

# Create final demographics dataframe
user_demographics = pd.DataFrame(demographics)

# Create the final dataset with all required columns by merging with full_user_tracks
logger.info("Creating final merged dataset with all required columns...")

# Aggregate playcount by track_id and user_id
playcount_agg = full_user_tracks.groupby(['track_id', 'user_id'])['playcount'].sum().reset_index()

# Get unique tracks data
track_columns = [
    'track_id', 'name', 'artist', 'spotify_preview_url', 'spotify_id', 
    'tags', 'genre', 'main_genre', 'year', 'duration_ms'
] + available_features

unique_tracks = full_user_tracks[track_columns].drop_duplicates('track_id')

# Create the base of the final dataset
final_dataset = playcount_agg.merge(unique_tracks, on='track_id', how='left')

# Now merge with user_demographics
final_dataset = final_dataset.merge(
    user_demographics, 
    on='user_id', 
    how='inner',
    suffixes=('', '_user_avg')
)

# Save the final dataset with all required columns
logger.info(f"Final dataset shape: {final_dataset.shape}")
logger.info(f"Final dataset columns: {final_dataset.columns.tolist()}")
logger.info("Saving final dataset...")
final_dataset.to_csv("spotify_complete_dataset.csv", index=False)

# Also save the user demographics separately for reference
logger.info("Saving demographic file...")
user_demographics.to_csv("spotify_user_demographics.csv", index=False)

# Display summary statistics
logger.info("\nDemographic Distribution Overview:")
logger.info(f"Age: Mean={user_demographics['age'].mean():.1f}, Median={user_demographics['age'].median()}")

logger.info("\nAge Groups:")
age_groups_count = user_demographics['age'].value_counts(bins=[18, 24, 34, 44, 54, 64, 100]).sort_index()
logger.info(f"{age_groups_count}")

logger.info("\nGender Distribution:")
gender_dist = user_demographics["gender"].value_counts(normalize=True) * 100
logger.info(f"{gender_dist}")

logger.info("\nRegion Distribution:")
region_dist = user_demographics["region"].value_counts(normalize=True) * 100
logger.info(f"{region_dist}")

logger.info("\nTop Genres:")
top_genres = user_demographics["top_genre"].value_counts().head(10)
logger.info(f"{top_genres}")

# Display year distribution of songs used
logger.info("\nYear Distribution of Filtered Songs:")
year_desc = music_info["year"].describe()
logger.info(f"{year_desc}")

# Display audio features summary
logger.info("\nAudio Features Summary:")
for feature in available_features:
    if f"avg_{feature}" in user_demographics.columns:
        logger.info(f"{feature}: Mean={user_demographics[f'avg_{feature}'].mean():.4f}")

logger.info("\nProcess complete!")

# Calculate additional quality metrics for final dataset
logger.info("\nPerforming quality checks on final dataset...")

# Check for missing values in the final dataset
missing_values_final = final_dataset.isnull().sum()
logger.info(f"Missing values in final dataset:\n{missing_values_final[missing_values_final > 0]}")

# Check for numerical outliers in key columns
for feature in available_features:
    if feature in final_dataset.columns:
        q1 = final_dataset[feature].quantile(0.25)
        q3 = final_dataset[feature].quantile(0.75)
        iqr = q3 - q1
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr
        outliers = final_dataset[(final_dataset[feature] < lower_bound) | (final_dataset[feature] > upper_bound)]
        if len(outliers) > 0:
            outlier_pct = (len(outliers) / len(final_dataset)) * 100
            logger.info(f"Outliers in {feature}: {len(outliers)} records ({outlier_pct:.2f}%)")

# Check demographic distribution matches our expected probabilities
logger.info("\nVerifying demographic distributions match expected probabilities...")

# Age distribution check
age_group_data = []
for i, (min_age, max_age) in enumerate(age_groups):
    group_name = f"{min_age}-{max_age}"
    actual_count = len(user_demographics[(user_demographics['age'] >= min_age) & (user_demographics['age'] <= max_age)])
    actual_pct = (actual_count / len(user_demographics)) * 100
    
    # Calculate expected percentages (average across all genres)
    expected_pcts = [genre_age_map[genre][i] for genre in genre_age_map.keys()]
    expected_pct = np.mean(expected_pcts) * 100
    
    age_group_data.append({
        'group': group_name,
        'actual_pct': actual_pct,
        'expected_pct': expected_pct,
        'difference': actual_pct - expected_pct
    })

age_distribution_check = pd.DataFrame(age_group_data)
logger.info(f"Age distribution check:\n{age_distribution_check}")

# Gender distribution check
gender_actual = user_demographics['gender'].value_counts(normalize=True) * 100
gender_expected = {gender: np.mean([probs[gender] for probs in genre_gender_map.values()]) * 100 
                  for gender in ['Male', 'Female']}
logger.info(f"Gender distribution - Actual vs Expected:")
for gender in gender_expected:
    logger.info(f"{gender}: Actual {gender_actual.get(gender, 0):.2f}% vs Expected {gender_expected[gender]:.2f}%")

# Additional playcount analysis
logger.info("\nPlaycount Statistics:")
logger.info(f"Mean playcount per user-track: {final_dataset['playcount'].mean():.2f}")
logger.info(f"Median playcount per user-track: {final_dataset['playcount'].median():.2f}")
logger.info(f"Min playcount: {final_dataset['playcount'].min()}")
logger.info(f"Max playcount: {final_dataset['playcount'].max()}")

# Check for users with unusually high playcounts
user_total_plays = final_dataset.groupby('user_id')['playcount'].sum()
high_playcount_users = user_total_plays[user_total_plays > user_total_plays.quantile(0.95)]
if len(high_playcount_users) > 0:
    logger.info(f"Users with unusually high playcounts (top 5%):")
    logger.info(f"{high_playcount_users.sort_values(ascending=False).head()}")

logger.info("Quality checks complete!")
