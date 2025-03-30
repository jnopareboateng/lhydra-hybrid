# Best Columns for Training

## User Features

1. **Demographics**:

- `age`: Keep as a continuous variable but consider bucketing into meaningful groups (teens, young adults, etc.)
- `gender`: Use as categorical feature
- `country`: Better than region for cultural context, but encode efficiently

2. **Listening Behavior**:

- `monthly_hours`: Strong indicator of overall engagement
- `genre_diversity`: Shows openness to different music styles
- `top_genre`: User's preferred genre

3. **Audio Preferences**:

- The average audio feature columns (`avg_danceability`, `avg_energy`, etc.): These capture listening preferences
  - `avg_danceability`
  - `avg_energy`
  - `avg_key`
  - `avg_loudness`
  - `avg_mode`
  - `avg_speechiness`
  - `avg_acousticness`
  - `avg_instrumentalness`
  - `avg_liveness`
  - `avg_valence`
  - `avg_tempo`
  - `avg_time_signature`

## Track Features

1. **Metadata**:

- `artist`: Critical for modeling artist preferences
- `main_genre`: Key categorization
- `year`: Captures temporal preferences
- `duration_ms`: Song length affects engagement

2. **Audio Features**:

- `danceability`, `energy`, `valence`: Emotional characteristics
- `acousticness`, `instrumentalness`: Style indicators
- `tempo`, `loudness`: Physical characteristics

## Target Variable

- `playcount` transformed into your binary engagement metric (>5 = high engagement)
