#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import sys
import argparse
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Generate sample data for Hybrid Music Recommender')
    
    parser.add_argument('--output-dir', type=str, default='data',
                        help='Directory to save generated data')
    parser.add_argument('--num-users', type=int, default=1000,
                        help='Number of users to generate')
    parser.add_argument('--num-items', type=int, default=5000,
                        help='Number of items to generate')
    parser.add_argument('--num-interactions', type=int, default=50000,
                        help='Number of interactions to generate')
    parser.add_argument('--user-features', type=int, default=10,
                        help='Number of user features to generate')
    parser.add_argument('--item-features', type=int, default=15,
                        help='Number of item features to generate')
    parser.add_argument('--playcount-max', type=int, default=100,
                        help='Maximum playcount value')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    
    return parser.parse_args()

def generate_user_features(num_users, num_features, seed):
    """Generate synthetic user features"""
    np.random.seed(seed)
    
    # Create DataFrame
    columns = [f'user_feature_{i}' for i in range(num_features)]
    user_features = pd.DataFrame(np.random.randn(num_users, num_features), columns=columns)
    
    # Add user_id column
    user_features['user_id'] = [f'U{i:06d}' for i in range(num_users)]
    
    # Add some demographic features
    user_features['age'] = np.random.randint(18, 70, size=num_users)
    user_features['gender'] = np.random.choice(['M', 'F', 'O'], size=num_users, p=[0.48, 0.48, 0.04])
    user_features['subscription_type'] = np.random.choice(['free', 'premium', 'family'], 
                                                        size=num_users, p=[0.6, 0.3, 0.1])
    user_features['active_days'] = np.random.randint(1, 1000, size=num_users)
    
    # Reorder columns to put user_id first
    cols = ['user_id'] + [col for col in user_features.columns if col != 'user_id']
    user_features = user_features[cols]
    
    return user_features

def generate_item_features(num_items, num_features, seed):
    """Generate synthetic item features"""
    np.random.seed(seed + 1)
    
    # Create DataFrame
    columns = [f'item_feature_{i}' for i in range(num_features)]
    item_features = pd.DataFrame(np.random.randn(num_items, num_features), columns=columns)
    
    # Add item_id column
    item_features['item_id'] = [f'I{i:06d}' for i in range(num_items)]
    
    # Add music-specific features
    item_features['duration'] = np.random.randint(60, 600, size=num_items)  # in seconds
    item_features['release_year'] = np.random.randint(1970, 2023, size=num_items)
    
    # Add genre probabilities
    genres = ['pop', 'rock', 'hip_hop', 'electronic', 'jazz', 'classical', 'country']
    for genre in genres:
        item_features[f'genre_{genre}'] = np.random.random(size=num_items)
    
    # Normalize genre probabilities to sum to 1
    genre_cols = [f'genre_{genre}' for genre in genres]
    genre_sums = item_features[genre_cols].sum(axis=1)
    for genre in genre_cols:
        item_features[genre] = item_features[genre] / genre_sums
    
    # Add popularity score
    item_features['popularity'] = np.random.pareto(1, size=num_items) * 10
    item_features['popularity'] = np.minimum(item_features['popularity'], 100)
    
    # Reorder columns to put item_id first
    cols = ['item_id'] + [col for col in item_features.columns if col != 'item_id']
    item_features = item_features[cols]
    
    return item_features

def generate_interactions(num_interactions, user_ids, item_ids, playcount_max, seed):
    """Generate synthetic user-item interactions"""
    np.random.seed(seed + 2)
    
    # Sample user and item IDs with replacement (skewed toward popular items)
    popularity = np.random.pareto(1, size=len(item_ids))
    item_probs = popularity / popularity.sum()
    
    user_indices = np.random.randint(0, len(user_ids), size=num_interactions)
    item_indices = np.random.choice(len(item_ids), size=num_interactions, p=item_probs)
    
    # Create DataFrame
    interactions = pd.DataFrame({
        'user_id': [user_ids[i] for i in user_indices],
        'item_id': [item_ids[i] for i in item_indices],
        'playcount': np.random.randint(1, playcount_max + 1, size=num_interactions)
    })
    
    # Add timestamp
    start_date = datetime(2022, 1, 1)
    timestamps = []
    for _ in range(num_interactions):
        days_to_add = np.random.randint(0, 365)
        hours_to_add = np.random.randint(0, 24)
        minutes_to_add = np.random.randint(0, 60)
        timestamp = start_date + timedelta(days=days_to_add, hours=hours_to_add, minutes=minutes_to_add)
        timestamps.append(timestamp)
    
    interactions['timestamp'] = timestamps
    
    # Remove duplicates and keep the latest interaction
    interactions = interactions.sort_values('timestamp').drop_duplicates(
        subset=['user_id', 'item_id'], keep='last')
    
    return interactions

def main():
    """Main function for generating sample data"""
    # Parse arguments
    args = parse_args()
    
    # Set random seed
    np.random.seed(args.seed)
    
    print(f"Generating sample data with {args.num_users} users, {args.num_items} items, "
          f"and approximately {args.num_interactions} interactions...")
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Generate user features
    print("Generating user features...")
    user_features = generate_user_features(args.num_users, args.user_features, args.seed)
    
    # Generate item features
    print("Generating item features...")
    item_features = generate_item_features(args.num_items, args.item_features, args.seed)
    
    # Get user and item IDs
    user_ids = user_features['user_id'].values
    item_ids = item_features['item_id'].values
    
    # Generate interactions
    print("Generating user-item interactions...")
    interactions = generate_interactions(
        args.num_interactions,
        user_ids,
        item_ids,
        args.playcount_max,
        args.seed
    )
    
    # Split interactions into train, validation, and test sets
    print("Splitting interactions into train, validation, and test sets...")
    
    # Shuffle interactions
    interactions = interactions.sample(frac=1, random_state=args.seed).reset_index(drop=True)
    
    # Split by time to simulate realistic testing
    interactions = interactions.sort_values('timestamp')
    
    n = len(interactions)
    train_idx = int(n * 0.7)
    val_idx = int(n * 0.85)
    
    train_df = interactions.iloc[:train_idx].copy()
    val_df = interactions.iloc[train_idx:val_idx].copy()
    test_df = interactions.iloc[val_idx:].copy()
    
    # Save generated data
    print("Saving generated data...")
    
    user_features.to_csv(os.path.join(args.output_dir, 'user_features.csv'), index=False)
    item_features.to_csv(os.path.join(args.output_dir, 'item_features.csv'), index=False)
    train_df.to_csv(os.path.join(args.output_dir, 'train.csv'), index=False)
    val_df.to_csv(os.path.join(args.output_dir, 'validation.csv'), index=False)
    test_df.to_csv(os.path.join(args.output_dir, 'test.csv'), index=False)
    
    # Print summary
    print("\nData generation completed!")
    print(f"User features: {len(user_features)} users with {args.user_features} features")
    print(f"Item features: {len(item_features)} items with {args.item_features} features")
    print(f"Interactions: {len(interactions)} total interactions")
    print(f"  - Train: {len(train_df)} interactions")
    print(f"  - Validation: {len(val_df)} interactions")
    print(f"  - Test: {len(test_df)} interactions")
    print(f"\nData saved to: {args.output_dir}/")

if __name__ == '__main__':
    main() 