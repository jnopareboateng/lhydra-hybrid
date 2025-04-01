"""
Metrics for evaluating the recommender system.

This module provides functions for calculating various metrics to evaluate
the performance of the recommender model, including classification metrics
and ranking metrics.
"""

import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score, confusion_matrix
)
import logging

logger = logging.getLogger(__name__)


def classification_metrics(y_true, y_pred, y_score=None, threshold=0.5):
    """
    Calculate classification metrics.
    
    Args:
        y_true (numpy.ndarray): Ground truth labels (0 or 1)
        y_pred (numpy.ndarray): Predicted scores or probabilities
        y_score (numpy.ndarray, optional): Raw scores for AUC calculation
        threshold (float): Threshold for converting probabilities to binary predictions
        
    Returns:
        dict: Dictionary containing classification metrics
    """
    if y_score is None:
        y_score = y_pred.copy()
    
    # Convert probabilities to binary predictions
    if y_pred.max() <= 1.0 and y_pred.min() >= 0.0 and not np.all(np.isin(y_pred, [0, 1])):
        y_pred_binary = (y_pred >= threshold).astype(int)
    else:
        y_pred_binary = y_pred
    
    metrics = {}
    
    try:
        # Basic metrics
        metrics['accuracy'] = accuracy_score(y_true, y_pred_binary)
        metrics['precision'] = precision_score(y_true, y_pred_binary, zero_division=0)
        metrics['recall'] = recall_score(y_true, y_pred_binary, zero_division=0)
        metrics['f1'] = f1_score(y_true, y_pred_binary, zero_division=0)
        
        # AUC and AP if we have enough positive and negative samples
        if len(np.unique(y_true)) > 1:
            metrics['auc'] = roc_auc_score(y_true, y_score)
            metrics['average_precision'] = average_precision_score(y_true, y_score)
        else:
            logger.warning("Only one class present in y_true. ROC AUC score is not defined in that case.")
    
    except Exception as e:
        logger.error(f"Error calculating classification metrics: {str(e)}")
    
    return metrics


def dcg_at_k(relevance, k):
    """
    Calculate Discounted Cumulative Gain at k.
    
    Args:
        relevance (numpy.ndarray): Array of relevance scores
        k (int): Number of top items to consider
        
    Returns:
        float: DCG@k
    """
    relevance = np.asarray(relevance)[:k]
    if len(relevance) == 0:
        return 0.0
    
    return np.sum(relevance / np.log2(np.arange(2, len(relevance) + 2)))


def ndcg_at_k(y_true, y_score, k):
    """
    Calculate Normalized Discounted Cumulative Gain at k.
    
    Args:
        y_true (numpy.ndarray): Ground truth relevance scores
        y_score (numpy.ndarray): Predicted scores
        k (int): Number of top items to consider
        
    Returns:
        float: NDCG@k
    """
    # Sort predictions and get the indices in descending order
    sorted_indices = np.argsort(y_score)[::-1]
    
    # Get the relevance of the sorted predictions
    relevance = np.asarray(y_true)[sorted_indices]
    
    # Calculate DCG
    dcg = dcg_at_k(relevance, k)
    
    # Calculate ideal DCG (relevance sorted by true relevance)
    ideal_relevance = np.sort(y_true)[::-1]
    idcg = dcg_at_k(ideal_relevance, k)
    
    if idcg == 0:
        return 0.0
    
    return dcg / idcg


def average_precision_at_k(y_true, y_score, k):
    """
    Calculate Average Precision at k.
    
    Args:
        y_true (numpy.ndarray): Ground truth relevance scores
        y_score (numpy.ndarray): Predicted scores
        k (int): Number of top items to consider
        
    Returns:
        float: AP@k
    """
    # Sort predictions and get the indices in descending order
    sorted_indices = np.argsort(y_score)[::-1]
    
    # Get the relevance of the sorted predictions
    relevance = np.asarray(y_true)[sorted_indices]
    
    # Only consider the top k items
    relevance = relevance[:k]
    
    if len(relevance) == 0 or np.sum(relevance) == 0:
        return 0.0
    
    # Calculate precision at each position where there is a relevant item
    precision_at_i = np.cumsum(relevance) / np.arange(1, len(relevance) + 1)
    
    # Only consider precision at positions where the item is relevant
    return np.sum(precision_at_i * relevance) / np.sum(relevance)


def mean_average_precision(y_true, y_score, k):
    """
    Calculate Mean Average Precision at k.
    
    Args:
        y_true (numpy.ndarray): Ground truth relevance scores
        y_score (numpy.ndarray): Predicted scores
        k (int): Number of top items to consider
        
    Returns:
        float: MAP@k
    """
    if len(y_true) == 0:
        return 0.0
    
    return average_precision_at_k(y_true, y_score, k)


def hit_rate_at_k(y_true, y_score, k):
    """
    Calculate Hit Rate at k.
    
    Args:
        y_true (numpy.ndarray): Ground truth relevance scores
        y_score (numpy.ndarray): Predicted scores
        k (int): Number of top items to consider
        
    Returns:
        float: Hit Rate@k
    """
    # Sort predictions and get the indices in descending order
    sorted_indices = np.argsort(y_score)[::-1]
    
    # Get the relevance of the sorted predictions
    relevance = np.asarray(y_true)[sorted_indices]
    
    # Only consider the top k items
    relevance = relevance[:k]
    
    # Check if at least one relevant item is in the top k
    return 1.0 if np.sum(relevance) > 0 else 0.0


def ranking_metrics(y_true, y_score, k_values=None):
    """
    Calculate ranking metrics at different k values.
    
    Args:
        y_true (numpy.ndarray): Ground truth relevance scores
        y_score (numpy.ndarray): Predicted scores
        k_values (list, optional): List of k values for metrics
        
    Returns:
        dict: Dictionary containing ranking metrics
    """
    if k_values is None:
        k_values = [5, 10]
    
    metrics = {}
    
    try:
        for k in k_values:
            metrics[f'ndcg@{k}'] = ndcg_at_k(y_true, y_score, k)
            metrics[f'map@{k}'] = mean_average_precision(y_true, y_score, k)
            metrics[f'hit_rate@{k}'] = hit_rate_at_k(y_true, y_score, k)
    
    except Exception as e:
        logger.error(f"Error calculating ranking metrics: {str(e)}")
    
    return metrics


def calculate_metrics(y_true, y_pred, classification_metrics_list=None, ranking_metrics_list=None):
    """
    Calculate all specified metrics.
    
    Args:
        y_true (numpy.ndarray): Ground truth labels
        y_pred (numpy.ndarray): Predicted scores or probabilities
        classification_metrics_list (list, optional): List of classification metrics to calculate
        ranking_metrics_list (list, optional): List of ranking metrics to calculate
        
    Returns:
        dict: Dictionary containing all calculated metrics
    """
    # Initialize metrics dictionary
    metrics = {}
    
    # Calculate classification metrics if specified
    if classification_metrics_list:
        class_metrics = classification_metrics(y_true, y_pred)
        for metric_name in classification_metrics_list:
            if metric_name in class_metrics:
                metrics[metric_name] = class_metrics[metric_name]
    
    # Calculate ranking metrics if specified
    if ranking_metrics_list:
        # Extract k values from metric names
        k_values = set()
        for metric_name in ranking_metrics_list:
            if '@' in metric_name:
                try:
                    k = int(metric_name.split('@')[1])
                    k_values.add(k)
                except (ValueError, IndexError):
                    continue
        
        # Calculate ranking metrics for all k values
        if k_values:
            rank_metrics = ranking_metrics(y_true, y_pred, list(k_values))
            for metric_name in ranking_metrics_list:
                if metric_name in rank_metrics:
                    metrics[metric_name] = rank_metrics[metric_name]
    
    return metrics 