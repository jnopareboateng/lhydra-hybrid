import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
import logging

logger = logging.getLogger(__name__)

def calculate_classification_metrics(y_true, y_pred, y_pred_proba=None):
    """
    Calculate classification metrics
    
    Args:
        y_true (numpy.ndarray): True labels
        y_pred (numpy.ndarray): Predicted labels (after threshold)
        y_pred_proba (numpy.ndarray, optional): Predicted probabilities
        
    Returns:
        dict: Dictionary containing the metrics
    """
    metrics = {}
    
    # Convert to binary predictions if needed
    if np.any((y_pred > 0) & (y_pred < 1)):
        y_pred_binary = (y_pred > 0.5).astype(int)
    else:
        y_pred_binary = y_pred.astype(int)
    
    # Basic metrics
    metrics['accuracy'] = accuracy_score(y_true, y_pred_binary)
    metrics['precision'] = precision_score(y_true, y_pred_binary, zero_division=0)
    metrics['recall'] = recall_score(y_true, y_pred_binary, zero_division=0)
    metrics['f1'] = f1_score(y_true, y_pred_binary, zero_division=0)
    
    # AUC if probabilities are provided
    if y_pred_proba is not None:
        try:
            metrics['auc'] = roc_auc_score(y_true, y_pred_proba)
        except Exception as e:
            logger.warning(f"Could not calculate AUC: {e}")
            metrics['auc'] = np.nan
    
    # Confusion matrix values
    cm = confusion_matrix(y_true, y_pred_binary)
    if cm.shape == (2, 2):
        tn, fp, fn, tp = cm.ravel()
        metrics['tn'] = tn
        metrics['fp'] = fp
        metrics['fn'] = fn
        metrics['tp'] = tp
    
    return metrics

def dcg_at_k(relevance, k):
    """
    Calculate Discounted Cumulative Gain at k
    
    Args:
        relevance (numpy.ndarray): Array of relevance scores
        k (int): Number of items to consider
        
    Returns:
        float: DCG@k
    """
    relevance = np.asarray(relevance)[:k]
    if len(relevance) == 0:
        return 0.0
    
    return np.sum(relevance / np.log2(np.arange(2, len(relevance) + 2)))

def ndcg_at_k(relevance, k):
    """
    Calculate Normalized Discounted Cumulative Gain at k
    
    Args:
        relevance (numpy.ndarray): Array of relevance scores
        k (int): Number of items to consider
        
    Returns:
        float: NDCG@k
    """
    relevance = np.asarray(relevance)
    
    # Create ideal relevance scores (sorted in descending order)
    ideal_relevance = np.sort(relevance)[::-1]
    
    # Calculate DCG for actual and ideal relevance
    dcg = dcg_at_k(relevance, k)
    idcg = dcg_at_k(ideal_relevance, k)
    
    if idcg == 0:
        return 0.0
    
    return dcg / idcg

def average_precision_at_k(relevance, k):
    """
    Calculate Average Precision at k
    
    Args:
        relevance (numpy.ndarray): Array of relevance scores (binary)
        k (int): Number of items to consider
        
    Returns:
        float: AP@k
    """
    relevance = np.asarray(relevance)[:k]
    if len(relevance) == 0:
        return 0.0
    
    # Calculate precision at each position where a relevant item was found
    precision_at_i = np.cumsum(relevance) / np.arange(1, len(relevance) + 1)
    
    # Only consider positions where relevant items were found
    return np.sum(precision_at_i * relevance) / np.sum(relevance) if np.sum(relevance) > 0 else 0.0

def calculate_ranking_metrics(y_true, y_pred, k_values=None):
    """
    Calculate ranking metrics
    
    Args:
        y_true (numpy.ndarray): True relevance scores
        y_pred (numpy.ndarray): Predicted scores
        k_values (list, optional): List of k values for which to calculate metrics
        
    Returns:
        dict: Dictionary containing the metrics
    """
    if k_values is None:
        k_values = [5, 10, 20]
    
    metrics = {}
    
    # For each user, sort items by predicted score and calculate metrics
    for k in k_values:
        ndcg_values = []
        ap_values = []
        
        # Get item indices sorted by predicted score (descending)
        rank_indices = np.argsort(-y_pred)
        
        # Get relevance of ranked items
        ranked_relevance = y_true[rank_indices]
        
        # Calculate NDCG@k
        ndcg = ndcg_at_k(ranked_relevance, k)
        metrics[f'ndcg@{k}'] = ndcg
        
        # Calculate MAP@k (only for binary relevance)
        if np.all(np.isin(y_true, [0, 1])):
            ap = average_precision_at_k(ranked_relevance, k)
            metrics[f'map@{k}'] = ap
    
    return metrics

def calculate_batch_ranking_metrics(y_true_batch, y_pred_batch, user_ids, k_values=None):
    """
    Calculate ranking metrics for a batch of users
    
    Args:
        y_true_batch (numpy.ndarray): True relevance scores
        y_pred_batch (numpy.ndarray): Predicted scores
        user_ids (numpy.ndarray): User IDs for each prediction
        k_values (list, optional): List of k values for which to calculate metrics
        
    Returns:
        dict: Dictionary containing the metrics
    """
    if k_values is None:
        k_values = [5, 10, 20]
    
    metrics = {f'ndcg@{k}': [] for k in k_values}
    
    # Check if true values are binary (0 or 1)
    is_binary = np.all(np.isin(y_true_batch, [0, 1]))
    if is_binary:
        for k in k_values:
            metrics[f'map@{k}'] = []
    
    # Group by users
    unique_users = np.unique(user_ids)
    
    for user in unique_users:
        # Get indices for this user
        user_indices = np.where(user_ids == user)[0]
        
        # Get true and predicted scores for this user
        user_y_true = y_true_batch[user_indices]
        user_y_pred = y_pred_batch[user_indices]
        
        # Calculate metrics for this user
        user_metrics = calculate_ranking_metrics(user_y_true, user_y_pred, k_values)
        
        # Add to metrics lists
        for metric, value in user_metrics.items():
            metrics[metric].append(value)
    
    # Calculate mean of metrics
    for metric in metrics:
        if len(metrics[metric]) > 0:
            metrics[metric] = np.mean(metrics[metric])
        else:
            metrics[metric] = np.nan
    
    return metrics

def calculate_coverage(all_item_ids, recommended_item_ids):
    """
    Calculate catalog coverage
    
    Args:
        all_item_ids (list): List of all item IDs in the catalog
        recommended_item_ids (list): List of item IDs that were recommended
        
    Returns:
        float: Coverage metric (percentage of catalog covered)
    """
    all_items_set = set(all_item_ids)
    recommended_items_set = set(recommended_item_ids)
    
    if len(all_items_set) == 0:
        return 0.0
    
    return len(recommended_items_set) / len(all_items_set)

def calculate_diversity(recommended_item_features):
    """
    Calculate diversity of recommendations
    
    Args:
        recommended_item_features (numpy.ndarray): Features of recommended items
        
    Returns:
        float: Diversity metric (average pairwise distance)
    """
    n_items = recommended_item_features.shape[0]
    
    if n_items <= 1:
        return 0.0
    
    # Calculate pairwise distances
    sum_distance = 0.0
    count = 0
    
    for i in range(n_items):
        for j in range(i+1, n_items):
            # Euclidean distance
            distance = np.sqrt(np.sum((recommended_item_features[i] - recommended_item_features[j])**2))
            sum_distance += distance
            count += 1
    
    return sum_distance / count if count > 0 else 0.0

def calculate_novelty(item_popularity, recommended_item_ids):
    """
    Calculate novelty of recommendations
    
    Args:
        item_popularity (dict): Dictionary mapping item IDs to popularity scores
        recommended_item_ids (list): List of recommended item IDs
        
    Returns:
        float: Novelty metric (inverse of average popularity)
    """
    if not recommended_item_ids:
        return 0.0
    
    # Get popularity scores for recommended items
    popularities = [item_popularity.get(item_id, 0) for item_id in recommended_item_ids]
    
    # Calculate inverse of average popularity
    avg_popularity = np.mean(popularities) if popularities else 0
    
    if avg_popularity == 0:
        return 0.0
    
    return 1.0 / avg_popularity

def calculate_serendipity(user_profile, recommended_items, item_similarity_matrix):
    """
    Calculate serendipity of recommendations
    
    Args:
        user_profile (list): List of item IDs in the user's profile
        recommended_items (list): List of recommended item IDs
        item_similarity_matrix (dict): Dictionary mapping item pairs to similarity scores
        
    Returns:
        float: Serendipity metric
    """
    if not user_profile or not recommended_items:
        return 0.0
    
    serendipity_scores = []
    
    for rec_item in recommended_items:
        # Calculate max similarity to items in user profile
        max_sim = 0.0
        for profile_item in user_profile:
            item_pair = tuple(sorted([rec_item, profile_item]))
            sim = item_similarity_matrix.get(item_pair, 0.0)
            max_sim = max(max_sim, sim)
        
        # Serendipity is inverse of similarity
        serendipity_scores.append(1.0 - max_sim)
    
    return np.mean(serendipity_scores) if serendipity_scores else 0.0

def calculate_all_metrics(y_true, y_pred, y_pred_proba=None, user_ids=None, k_values=None):
    """
    Calculate all metrics
    
    Args:
        y_true (numpy.ndarray): True labels/relevance
        y_pred (numpy.ndarray): Predicted labels
        y_pred_proba (numpy.ndarray, optional): Predicted probabilities
        user_ids (numpy.ndarray, optional): User IDs for ranking metrics
        k_values (list, optional): List of k values for ranking metrics
        
    Returns:
        dict: Dictionary containing all metrics
    """
    metrics = {}
    
    # Classification metrics
    classification_metrics = calculate_classification_metrics(y_true, y_pred, y_pred_proba)
    metrics.update(classification_metrics)
    
    # Ranking metrics if user_ids provided
    if user_ids is not None:
        # Use probabilities if available, otherwise use predictions
        pred_scores = y_pred_proba if y_pred_proba is not None else y_pred
        ranking_metrics = calculate_batch_ranking_metrics(y_true, pred_scores, user_ids, k_values)
        metrics.update(ranking_metrics)
    
    return metrics 