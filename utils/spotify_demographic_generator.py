import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from collections import Counter
import matplotlib.pyplot as plt
import seaborn as sns

# Load datasets
users_history = pd.read_csv("/kaggle/input/million-song-dataset-spotify-lastfm/User Listening History.csv")
music_info = pd.read_csv("/kaggle/input/million-song-dataset-spotify-lastfm/Music Info.csv")

print(f"Unique users: {users_history['user_id'].nunique()}")
print(f"Unique tracks: {music_info['track_id'].nunique()}")

# Step 1: First cluster the songs based on audio features
# Select audio features for clustering
audio_features = ['danceability', 'energy', 'key', 'loudness', 'mode', 
                 'speechiness', 'acousticness', 'instrumentalness', 
                 'liveness', 'valence', 'tempo']

# Extract genre information
def extract_main_genres(tag_string):
    if pd.isna(tag_string):
        return []
    
    tags = str(tag_string).lower().split(',')
    genres = []
    main_genres = ['rock', 'pop', 'hip hop', 'rap', 'electronic', 'dance', 
                  'jazz', 'classical', 'r&b', 'country', 'indie', 'folk']
    
    for genre in main_genres:
        if any(genre in tag for tag in tags):
            genres.append(genre)
    
    return genres if genres else ['other']

# Add genre features to music_info
music_info['main_genres'] = music_info['tags'].apply(extract_main_genres)

# One-hot encode main genres
for genre in ['rock', 'pop', 'hip hop', 'rap', 'electronic', 'dance', 
             'jazz', 'classical', 'r&b', 'country', 'indie', 'folk', 'other']:
    music_info[f'genre_{genre}'] = music_info['main_genres'].apply(lambda x: 1 if genre in x else 0)

# Combine audio features and genre information for clustering
genre_features = [col for col in music_info.columns if col.startswith('genre_')]
music_features_df = music_info[['track_id'] + audio_features + genre_features].copy()

# Normalize features for song clustering
scaler = StandardScaler()
features_to_scale = audio_features + genre_features
music_features_df_scaled = music_features_df.copy()
music_features_df_scaled[features_to_scale] = scaler.fit_transform(music_features_df[features_to_scale])

# Cluster songs (we'll create 20 song clusters)
n_song_clusters = 20
song_kmeans = KMeans(n_clusters=n_song_clusters, random_state=42, n_init=10)
music_features_df_scaled['song_cluster'] = song_kmeans.fit_predict(
    music_features_df_scaled[features_to_scale]
)
song_clusters = music_features_df_scaled[['track_id', 'song_cluster']]

# Analyze the song clusters to understand what each represents
cluster_profile = []
for cluster_id in range(n_song_clusters):
    cluster_tracks = music_features_df[music_features_df_scaled['song_cluster'] == cluster_id]
    
    # Get average audio features
    avg_features = cluster_tracks[audio_features].mean().to_dict()
    
    # Get most common genres
    genre_counts = cluster_tracks[genre_features].sum()
    top_genres = genre_counts.nlargest(3).index.tolist()
    top_genres = [g.replace('genre_', '') for g in top_genres]
    
    cluster_profile.append({
        'cluster_id': cluster_id,
        'size': len(cluster_tracks),
        'top_genres': top_genres,
        **avg_features
    })

song_cluster_profile = pd.DataFrame(cluster_profile)
print("Song Cluster Profiles:")
print(song_cluster_profile[['cluster_id', 'size', 'top_genres', 'danceability', 'energy', 'valence']].head())

# Step 2: Create user preference distributions across song clusters
# Merge user history with song clusters
user_song_clusters = users_history.merge(song_clusters, on='track_id')

# Create distribution of song clusters for each user (weighted by playcount)
user_cluster_distributions = user_song_clusters.groupby(['user_id', 'song_cluster'])['playcount'].sum().reset_index()
total_plays = user_cluster_distributions.groupby('user_id')['playcount'].sum().reset_index()
total_plays.rename(columns={'playcount': 'total_playcount'}, inplace=True)
user_cluster_distributions = user_cluster_distributions.merge(total_plays, on='user_id')
user_cluster_distributions['percentage'] = (user_cluster_distributions['playcount'] / 
                                          user_cluster_distributions['total_playcount']) * 100

# Create a pivot table to get user distribution vectors
user_vectors = user_cluster_distributions.pivot_table(
    index='user_id', 
    columns='song_cluster', 
    values='percentage', 
    fill_value=0
).reset_index()

# Step 3: Calculate diversity metrics for each user
def calculate_diversity(row):
    # Convert percentages to probabilities
    probs = row[1:] / 100.0  # Exclude user_id column
    probs = probs[probs > 0]  # Only consider non-zero probabilities
    
    # Shannon entropy as diversity measure
    if len(probs) > 0:
        entropy = -np.sum(probs * np.log2(probs))
        return entropy
    return 0

user_vectors['listening_diversity'] = user_vectors.apply(calculate_diversity, axis=1)

# Step 4: Cluster users based on their listening distributions
user_feature_cols = [col for col in user_vectors.columns if isinstance(col, int)]
user_scaler = StandardScaler()
user_scaled_features = user_scaler.fit_transform(user_vectors[user_feature_cols])

# We'll create 8 user clusters
n_user_clusters = 8
user_kmeans = KMeans(n_clusters=n_user_clusters, random_state=42, n_init=10)
user_vectors['user_cluster'] = user_kmeans.fit_predict(user_scaled_features)

# Analyze user clusters to understand what each represents
user_cluster_profiles = []
for cluster_id in range(n_user_clusters):
    cluster_users = user_vectors[user_vectors['user_cluster'] == cluster_id]
    
    # Get average distribution across song clusters
    avg_distribution = cluster_users[user_feature_cols].mean().to_dict()
    
    # Find top song clusters for this user cluster
    top_3_song_clusters = sorted(avg_distribution.items(), key=lambda x: x[1], reverse=True)[:3]
    
    # Map those to the song cluster profiles
    top_genres = []
    for song_cluster, percentage in top_3_song_clusters:
        top_genres.extend(song_cluster_profile.loc[song_cluster_profile['cluster_id'] == song_cluster, 'top_genres'].iloc[0])
    
    # Average diversity
    avg_diversity = cluster_users['listening_diversity'].mean()
    
    user_cluster_profiles.append({
        'user_cluster_id': cluster_id,
        'size': len(cluster_users),
        'top_song_clusters': [sc[0] for sc in top_3_song_clusters],
        'top_genres': list(set(top_genres)),
        'avg_diversity': avg_diversity
    })

user_cluster_profile_df = pd.DataFrame(user_cluster_profiles)
print("\nUser Cluster Profiles:")
print(user_cluster_profile_df.head())

# Step 5: Assign demographics based on user clusters and known Spotify demographics
# Define demographic distributions
age_groups = [
    (18, 24),  # 31.51%
    (25, 34),  # 31.41%
    (35, 44),  # ~15%
    (45, 54),  # ~10%
    (55, 64),  # ~8%
    (65, 100)  # ~4%
]

# Base probabilities from Spotify data
base_age_probabilities = [0.3151, 0.3141, 0.15, 0.10, 0.08, 0.04]
base_gender_probabilities = {'Female': 0.56, 'Male': 0.44}
base_region_probabilities = {
    'Europe': 0.27, 
    'North America': 0.28, 
    'Latin America': 0.22, 
    'Rest of World': 0.23
}

# Adjust demographics based on user cluster characteristics
# This is where we encode the correlations between music taste and demographics
def adjust_demographics_by_genres(cluster_profile):
    """Adjust demographic probabilities based on genre preferences"""
    age_adj = base_age_probabilities.copy()
    gender_adj = base_gender_probabilities.copy()
    region_adj = base_region_probabilities.copy()
    
    # These adjustments are based on research about music preferences by demographic
    # Implement correlations between genres and demographics
    genres = cluster_profile['top_genres']
    
    # Age correlations
    if any(g in genres for g in ['hip hop', 'rap', 'dance', 'electronic']):
        # Increase younger age groups probability
        age_adj[0] *= 1.3  # 18-24
        age_adj[1] *= 1.2  # 25-34
    
    if any(g in genres for g in ['rock', 'indie', 'alternative']):
        # Slight increase in middle age groups
        age_adj[1] *= 1.15  # 25-34
        age_adj[2] *= 1.2   # 35-44
    
    if any(g in genres for g in ['jazz', 'classical', 'folk']):
        # Increase older age groups probability
        age_adj[3] *= 1.3   # 45-54
        age_adj[4] *= 1.3   # 55-64
        age_adj[5] *= 1.5   # 65+
    
    # Gender correlations
    if any(g in genres for g in ['hip hop', 'rap', 'rock']):
        gender_adj['Male'] *= 1.15
    
    if any(g in genres for g in ['pop', 'dance']):
        gender_adj['Female'] *= 1.1
    
    # Region correlations
    if any(g in genres for g in ['hip hop', 'rap', 'r&b']):
        region_adj['North America'] *= 1.2
    
    if any(g in genres for g in ['electronic', 'dance']):
        region_adj['Europe'] *= 1.2
    
    if any(g in genres for g in ['latin']):
        region_adj['Latin America'] *= 1.5
    
    # Also consider diversity factor (high diversity = different pattern)
    diversity = cluster_profile['avg_diversity']
    if diversity > 2.5:  # High diversity
        # More balanced across demographics
        age_adj = [0.25, 0.25, 0.15, 0.15, 0.1, 0.1]
    
    # Normalize to ensure probabilities sum to 1
    age_adj = np.array(age_adj)
    age_adj = age_adj / age_adj.sum()
    
    gender_values = np.array(list(gender_adj.values()))
    gender_values = gender_values / gender_values.sum()
    gender_adj = {k: gender_values[i] for i, k in enumerate(gender_adj.keys())}
    
    region_values = np.array(list(region_adj.values()))
    region_values = region_values / region_values.sum()
    region_adj = {k: region_values[i] for i, k in enumerate(region_adj.keys())}
    
    return age_adj, gender_adj, region_adj

# Create final user demographics
countries_by_region = {
    'Europe': ['UK', 'Germany', 'France', 'Spain', 'Italy', 'Sweden', 'Netherlands', 'Poland'],
    'North America': ['USA', 'Canada'],
    'Latin America': ['Brazil', 'Mexico', 'Argentina', 'Colombia', 'Chile'],
    'Rest of World': ['Japan', 'Australia', 'India', 'South Korea', 'South Africa', 'UAE']
}

user_demographics = []

for cluster_id, cluster_profile in user_cluster_profile_df.iterrows():
    # Get users in this cluster
    cluster_users = user_vectors[user_vectors['user_cluster'] == cluster_profile['user_cluster_id']]['user_id'].values
    n_cluster_users = len(cluster_users)
    
    # Adjust demographic probabilities based on cluster profile
    age_probs, gender_probs, region_probs = adjust_demographics_by_genres(cluster_profile)
    
    # Assign demographics to each user in the cluster
    for user_id in cluster_users:
        # Age
        age_group_idx = np.random.choice(len(age_groups), p=age_probs)
        age_range = age_groups[age_group_idx]
        age = np.random.randint(age_range[0], age_range[1] + 1)
        
        # Gender
        gender = np.random.choice(list(gender_probs.keys()), p=list(gender_probs.values()))
        
        # Region
        region = np.random.choice(list(region_probs.keys()), p=list(region_probs.values()))
        
        # Country within region
        country = np.random.choice(countries_by_region[region])
        
        # Listening hours (correlated with age and gender)
        base_hours = 0
        if age < 25:
            base_hours = 25 + np.random.normal(5, 3)
        elif age < 35:
            base_hours = 20 + np.random.normal(4, 3)
        elif age < 45:
            base_hours = 15 + np.random.normal(5, 2)
        else:
            base_hours = 10 + np.random.normal(5, 2)
        
        # Gen Z women tend to listen more (30 hrs vs 24 hrs for men)
        if age < 25 and gender == 'Female':
            base_hours *= 1.2
        
        monthly_hours = max(5, min(60, base_hours))  # Cap between 5 and 60 hours
        
        # Get diversity from user vectors
        user_diversity = user_vectors.loc[user_vectors['user_id'] == user_id, 'listening_diversity'].values[0]
        
        user_demographics.append({
            'user_id': user_id,
            'age': age,
            'gender': gender,
            'region': region,
            'country': country,
            'monthly_hours': monthly_hours,
            'user_cluster': cluster_profile['user_cluster_id'],
            'listening_diversity': user_diversity,
            'top_genres': cluster_profile['top_genres']
        })

# Create final demographics dataframe
user_demographics_df = pd.DataFrame(user_demographics)

# Generate some visualizations for demographic distribution
plt.figure(figsize=(15, 10))

# Age distribution
plt.subplot(2, 2, 1)
sns.histplot(user_demographics_df['age'], bins=20)
plt.title('Age Distribution')

# Gender distribution
plt.subplot(2, 2, 2)
sns.countplot(x='gender', data=user_demographics_df)
plt.title('Gender Distribution')

# Region distribution
plt.subplot(2, 2, 3)
sns.countplot(y='region', data=user_demographics_df)
plt.title('Region Distribution')

# Listening hours by age group
plt.subplot(2, 2, 4)
user_demographics_df['age_group'] = pd.cut(user_demographics_df['age'], [18, 24, 34, 44, 54, 64, 100], 
                                         labels=['18-24', '25-34', '35-44', '45-54', '55-64', '65+'])
sns.boxplot(x='age_group', y='monthly_hours', data=user_demographics_df)
plt.title('Listening Hours by Age Group')

plt.tight_layout()
plt.savefig('demographic_distributions.png')

# Save the final synthetic demographics
user_demographics_df.to_csv('rich_synthetic_user_demographics.csv', index=False)

print(f"\nCreated rich synthetic demographics for {len(user_demographics_df)} users")
print(user_demographics_df.head())

# Additional analysis - show listening patterns by demographic
print("\nAverage listening diversity by age group:")
print(user_demographics_df.groupby('age_group')['listening_diversity'].mean())

print("\nTop user clusters by gender:")
print(user_demographics_df.groupby(['gender', 'user_cluster']).size().reset_index().rename(columns={0:'count'}).sort_values(['gender', 'count'], ascending=[True, False]))