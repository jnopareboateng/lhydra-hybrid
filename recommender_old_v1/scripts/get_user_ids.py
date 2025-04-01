#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Script to print available user IDs from the user features file and analyze its structure
"""

import pandas as pd
import random
import argparse
import os

def main():
    parser = argparse.ArgumentParser(description='Print available user IDs from a user features file')
    parser.add_argument('--file', type=str, default='notebooks/shipping_10k_data/features/user_features.csv',
                        help='Path to the user features CSV file')
    parser.add_argument('--count', type=int, default=10,
                        help='Number of user IDs to print')
    parser.add_argument('--check-id', type=str, default=None,
                        help='Check if a specific user ID exists')
    args = parser.parse_args()
    
    if not os.path.exists(args.file):
        print(f"Error: File {args.file} does not exist")
        return
    
    print(f"Loading user features from {args.file}")
    user_df = pd.read_csv(args.file)
    
    print("\nFile Information:")
    print(f"  - Number of rows: {len(user_df)}")
    print(f"  - Number of columns: {len(user_df.columns)}")
    print(f"  - DataFrame index type: {type(user_df.index)}")
    print(f"  - DataFrame index name: {user_df.index.name}")
    
    print("\nFirst 5 column names:")
    for i, col in enumerate(list(user_df.columns)[:5]):
        print(f"  - Column {i+1}: {col} (type: {user_df[col].dtype})")
    
    # Check for user ID column
    user_id_cols = [col for col in user_df.columns if 'user' in col.lower() or 'id' in col.lower()]
    if user_id_cols:
        print("\nPotential user ID columns:")
        for col in user_id_cols:
            print(f"  - {col} (type: {user_df[col].dtype})")
    
    # Try to determine the user ID column
    user_id_col = None
    if 'user_id' in user_df.columns:
        user_id_col = 'user_id'
    elif user_df.index.name == 'user_id':
        user_id_col = user_df.index.name
        user_df = user_df.reset_index()
    elif user_id_cols:
        user_id_col = user_id_cols[0]
        print(f"\nUsing '{user_id_col}' as the user ID column")
    else:
        print("\nWarning: No 'user_id' column found. Available columns:")
        for col in user_df.columns:
            print(f"  - {col}")
        
        # Check if index might contain user IDs
        if user_df.index.name is None and not user_df.index.equals(pd.RangeIndex(len(user_df))):
            print("\nIndex appears to contain values that might be user IDs:")
            print(f"First 5 index values: {list(user_df.index[:5])}")
            user_df = user_df.reset_index()
            user_id_col = 'index'
            print("Using index values as user IDs")
    
    # Continue only if we've identified a user ID column
    if user_id_col is None:
        print("\nError: Unable to identify user ID column. Add a --column parameter to specify?")
        return
    
    user_ids = user_df[user_id_col].values
    print(f"\nFound {len(user_ids)} user IDs")
    
    # Check for NaN or invalid values
    invalid_count = sum(1 for uid in user_ids if pd.isna(uid))
    if invalid_count > 0:
        print(f"Warning: Found {invalid_count} NaN/invalid user IDs")
    
    # Print the first N user IDs
    print(f"\nFirst {min(args.count, len(user_ids))} user IDs:")
    for i, uid in enumerate(user_ids[:args.count]):
        print(f"{i+1}. {uid}")
    
    # Check if specific ID exists
    if args.check_id is not None:
        check_id = args.check_id
        # Try different formats for comparison
        if user_df[user_id_col].dtype == 'int64':
            try:
                check_id = int(check_id)
            except ValueError:
                print(f"Warning: Cannot convert '{check_id}' to integer for comparison")
        
        exists = check_id in user_ids
        exists_str = any(str(uid) == str(check_id) for uid in user_ids)
        
        print(f"\nUser ID '{check_id}':")
        print(f"  - Exact match: {'found' if exists else 'NOT found'}")
        print(f"  - String match: {'found' if exists_str else 'NOT found'}")
        
        if exists_str and not exists:
            matching_ids = [uid for uid in user_ids if str(uid) == str(check_id)]
            print(f"  - Found as string match: {matching_ids}")
            print(f"  - Data type mismatch: {check_id} ({type(check_id)}) vs {matching_ids[0]} ({type(matching_ids[0])})")
    
    # Print 5 random user IDs for testing
    print("\n5 random user IDs for testing:")
    random_ids = random.sample(list(user_ids), min(5, len(user_ids)))
    for i, uid in enumerate(random_ids):
        print(f"{i+1}. {uid}")
    
    # Save some sample IDs to a file for convenience
    with open('sample_user_ids.txt', 'w') as f:
        f.write("# Sample user IDs for testing\n")
        for uid in random_ids:
            f.write(f"{uid}\n")
    print(f"\nSaved 5 sample user IDs to 'sample_user_ids.txt'")

if __name__ == "__main__":
    main() 