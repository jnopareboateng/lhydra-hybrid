#!/usr/bin/env python3
"""
Test script to demonstrate the feature alignment capabilities
of the Lhydra Hybrid Music Recommender System.
"""

import os
import sys
import argparse
import torch
import pandas as pd
import yaml

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import project modules
from models.two_tower_model import TwoTowerHybridModel
from data.preprocessor import MusicDataPreprocessor
from utils.logger import LhydraLogger

def test_feature_alignment(model_path, data_path, manifest_path=None, output_path=None):
    """
    Test the feature alignment functionality by comparing aligned vs unaligned feature sets.
    
    Args:
        model_path (str): Path to the trained model
        data_path (str): Path to test data
        manifest_path (str, optional): Path to feature manifest (if not included with model)
        output_path (str, optional): Directory to save results
    """
    # Create logger
    logger = LhydraLogger(log_dir="logs/feature_alignment_test")
    logger.info("Starting feature alignment test")
    
    # Create output directory if needed
    if output_path:
        os.makedirs(output_path, exist_ok=True)
    
    # Load test data
    logger.info(f"Loading test data from {data_path}")
    test_df = pd.read_csv(data_path)
    logger.info(f"Loaded {len(test_df)} samples with {len(test_df.columns)} columns")
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    
    # Load model with manifest
    logger.info(f"Loading model from {model_path}")
    model = TwoTowerHybridModel.load(model_path, device=device, manifest_path=manifest_path)
    model.eval()
    
    # Check if feature manifest is available
    if hasattr(model, 'feature_manifest') and model.feature_manifest is not None:
        logger.info("Feature manifest is loaded and available for feature alignment")
        logger.info(f"Expected user features: {len(model.feature_manifest.get('user_features', []))}")
        logger.info(f"Expected item features: {len(model.feature_manifest.get('item_features', []))}")
    else:
        logger.error("No feature manifest available. Cannot demonstrate feature alignment.")
        return
    
    # Create a preprocessor to prepare features
    preprocessor = MusicDataPreprocessor(logger=logger)
    
    # Prepare a small batch of data for testing
    batch_size = min(10, len(test_df))
    sample_df = test_df.head(batch_size).copy()
    
    # Extract user and item features
    user_features = {}
    item_features = {}
    
    # Create a simple feature extractor 
    def extract_features(df, prefix_list, feature_dict):
        for prefix in prefix_list:
            cols = [col for col in df.columns if col.startswith(prefix)]
            for col in cols:
                feature_dict[col] = torch.tensor(df[col].values, dtype=torch.float32)
    
    # Extract user features
    user_prefixes = ['user_', 'gender_', 'age_', 'region_', 'country_', 'monthly_hours', 
                     'genre_diversity', 'top_genre_', 'avg_']
    extract_features(sample_df, user_prefixes, user_features)
    
    # Extract item features
    item_prefixes = ['track_', 'danceability', 'energy', 'key', 'loudness', 'mode',
                    'speechiness', 'acousticness', 'instrumentalness', 'liveness', 
                    'valence', 'tempo', 'main_genre_', 'year', 'song_age']
    extract_features(sample_df, item_prefixes, item_features)
    
    # Make sure IDs are included
    if 'user_id' in sample_df.columns:
        user_features['user_id'] = torch.tensor(sample_df['user_id'].values)
    if 'track_id' in sample_df.columns:
        item_features['track_id'] = torch.tensor(sample_df['track_id'].values)
    
    # Print feature statistics before alignment
    logger.info("=== Before Feature Alignment ===")
    logger.info(f"User features: {len(user_features)}")
    logger.info(f"Item features: {len(item_features)}")
    
    # Use the align_features method
    logger.info("Aligning features with manifest...")
    aligned_user_features, aligned_item_features = model.align_features(user_features, item_features)
    
    # Print feature statistics after alignment
    logger.info("=== After Feature Alignment ===")
    logger.info(f"User features: {len(aligned_user_features)}")
    logger.info(f"Item features: {len(aligned_item_features)}")
    
    # Compare feature sets
    logger.info("=== Feature Comparison ===")
    
    def compare_feature_sets(original, aligned, feature_type):
        # Find keys in original but not in aligned
        removed = set(original.keys()) - set(aligned.keys())
        # Find keys in aligned but not in original (these are generated with zeros)
        added = set(aligned.keys()) - set(original.keys())
        # Find keys in both
        kept = set(original.keys()) & set(aligned.keys())
        
        logger.info(f"{feature_type} features comparison:")
        logger.info(f"  - Original count: {len(original)}")
        logger.info(f"  - Aligned count: {len(aligned)}")
        logger.info(f"  - Removed features: {len(removed)}")
        logger.info(f"  - Added features (zeroed): {len(added)}")
        logger.info(f"  - Kept features: {len(kept)}")
        
        if len(removed) > 0:
            logger.info(f"  - Sample removed features: {list(removed)[:5]}")
        if len(added) > 0:
            logger.info(f"  - Sample added features: {list(added)[:5]}")
            
        return {
            "original_count": len(original),
            "aligned_count": len(aligned),
            "removed": list(removed),
            "added": list(added),
            "kept": list(kept)
        }
    
    user_comparison = compare_feature_sets(user_features, aligned_user_features, "User")
    item_comparison = compare_feature_sets(item_features, aligned_item_features, "Item")
    
    # Save comparison report if output path provided
    if output_path:
        comparison_report = {
            "user_features": user_comparison,
            "item_features": item_comparison,
            "manifest_summary": {
                "user_features_count": len(model.feature_manifest.get('user_features', [])),
                "item_features_count": len(model.feature_manifest.get('item_features', [])),
            }
        }
        
        report_path = os.path.join(output_path, "feature_alignment_report.yaml")
        with open(report_path, 'w') as f:
            yaml.dump(comparison_report, f)
        logger.info(f"Saved feature alignment report to {report_path}")
    
    # Attempt prediction with both aligned and unaligned features
    logger.info("=== Testing Prediction ===")
    
    # Function to try prediction with error handling
    def try_prediction(user_feat, item_feat, name):
        try:
            with torch.inference_mode():
                outputs = model(user_feat, item_feat)
            logger.info(f"{name} prediction successful! Output shape: {outputs.shape}")
            return True
        except Exception as e:
            logger.error(f"{name} prediction failed: {str(e)}")
            return False
    
    # Move features to device
    if isinstance(user_features, dict):
        user_features_device = {k: v.to(device) for k, v in user_features.items()}
        aligned_user_features_device = {k: v.to(device) for k, v in aligned_user_features.items()}
    else:
        user_features_device = user_features.to(device)
        aligned_user_features_device = aligned_user_features.to(device)
        
    if isinstance(item_features, dict):
        item_features_device = {k: v.to(device) for k, v in item_features.items()}
        aligned_item_features_device = {k: v.to(device) for k, v in aligned_item_features.items()}
    else:
        item_features_device = item_features.to(device)
        aligned_item_features_device = aligned_item_features.to(device)
    
    # Try prediction with unaligned features
    unaligned_success = try_prediction(user_features_device, item_features_device, "Unaligned")
    
    # Try prediction with aligned features
    aligned_success = try_prediction(aligned_user_features_device, aligned_item_features_device, "Aligned")
    
    # Summary
    logger.info("=== Test Summary ===")
    logger.info(f"Unaligned prediction: {'Success' if unaligned_success else 'Failed'}")
    logger.info(f"Aligned prediction: {'Success' if aligned_success else 'Failed'}")
    
    if aligned_success and not unaligned_success:
        logger.info("CONCLUSION: Feature alignment successfully fixed compatibility issues!")
    elif aligned_success and unaligned_success:
        logger.info("CONCLUSION: Both feature sets work, alignment is not critical for this data.")
    else:
        logger.info("CONCLUSION: Issues persist even with feature alignment. Further investigation needed.")

def main():
    parser = argparse.ArgumentParser(description="Test the feature alignment capabilities")
    parser.add_argument("--model", type=str, required=True, help="Path to trained model")
    parser.add_argument("--data", type=str, required=True, help="Path to test data")
    parser.add_argument("--manifest", type=str, help="Path to feature manifest (optional)")
    parser.add_argument("--output", type=str, help="Directory to save results (optional)")
    
    args = parser.parse_args()
    
    test_feature_alignment(
        model_path=args.model,
        data_path=args.data,
        manifest_path=args.manifest,
        output_path=args.output
    )

if __name__ == "__main__":
    main() 