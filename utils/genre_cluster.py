import pandas as pd
import numpy as np
from collections import defaultdict


class GenreClustering:
    MAIN_GENRES = {
        "rock": ["rock", "metal", "punk", "alternative", "grunge", "indie"],
        "electronic": ["electronic", "dance", "edm", "house", "techno", "dubstep"],
        "hip_hop": ["hip-hop", "rap", "trap", "drill"],
        "pop": ["pop", "teen pop", "dance pop", "k-pop"],
        "classical": ["classical", "baroque", "orchestra"],
        "jazz": ["jazz", "bebop", "fusion"],
        "folk": ["folk", "acoustic", "singer-songwriter"],
        "rb_soul": ["rnb", "r&b", "soul", "motown"],
        "country": ["country", "bluegrass", "americana"],
        "world": ["latin", "reggae", "afrobeat", "world"],
        "religious": ["christian", "gospel", "spiritual", "worship"],
    }

    def __init__(self, genres_series):
        self.genres = genres_series
        self.genre_map = self._create_genre_map()

    def _create_genre_map(self):
        """Create mapping of subgenres to main genres"""
        genre_map = {}
        for main_genre, subgenres in self.MAIN_GENRES.items():
            for subgenre in subgenres:
                genre_map[subgenre] = main_genre
        return genre_map

    def classify_genre(self, genre):
        """Map a genre to its main category"""
        genre = str(genre).lower()
        for key_term, main_genre in self.genre_map.items():
            if key_term in genre:
                return main_genre
        return "other"

    def cluster_genres(self):
        """Group genres into main categories"""
        clustered = defaultdict(list)
        counts = defaultdict(int)

        for genre, count in self.genres.items():
            main_genre = self.classify_genre(genre)
            clustered[main_genre].append((genre, count))
            counts[main_genre] += count

        return clustered, counts


# Usage
genres_series = pd.Series(
    {
        genre: count
        for genre, count in zip(
            df["genre"].value_counts().index, df["genre"].value_counts().values
        )
    }
)
clusterer = GenreClustering(genres_series)
clusters, counts = clusterer.cluster_genres()

# Print results
for main_genre, subgenres in clusters.items():
    print(f"\n## {main_genre.upper()} (Total: {counts[main_genre]})")
    for subgenre, count in sorted(subgenres, key=lambda x: x[1], reverse=True):
        print(f"- {subgenre}: {count}")
