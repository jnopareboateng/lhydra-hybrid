import torch
import numpy as np
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score, confusion_matrix
)
from typing import Dict, List, Tuple, Union, Optional
import pandas as pd
import os
import sys

# Add root directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.logger import LhydraLogger, log_function

class RecommenderMetrics:
    """
    Metrics for evaluating music recommendation systems.
    Includes standard classification metrics and ranking metrics.
    """
    
    def __init__(self, logger=None):
        """
        Initialize the metrics calculator.
        
        Args:
            logger (LhydraLogger): Logger instance.
        """
        self.logger = logger or LhydraLogger()
        self.logger.info("Initializing RecommenderMetrics")
    
    @staticmethod
    @log_function()
    def calculate_classification_metrics(y_true: np.ndarray, y_pred: np.ndarray, 
                                      y_prob: Optional[np.ndarray] = None) -> Dict[str, float]:
        """
        Calculate standard classification metrics.
        
        Args:
            y_true (np.ndarray): Ground truth labels.
            y_pred (np.ndarray): Predicted binary labels.
            y_prob (np.ndarray, optional): Predicted probabilities.
            
        Returns:
            Dict[str, float]: Dictionary with classification metrics.
        """
        metrics = {}
        
        # Basic classification metrics
        metrics['accuracy'] = accuracy_score(y_true, y_pred)
        metrics['precision'] = precision_score(y_true, y_pred, zero_division=0)
        metrics['recall'] = recall_score(y_true, y_pred, zero_division=0)
        metrics['f1'] = f1_score(y_true, y_pred, zero_division=0)
        
        # Confusion matrix values
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        metrics['true_negatives'] = int(tn)
        metrics['false_positives'] = int(fp)
        metrics['false_negatives'] = int(fn)
        metrics['true_positives'] = int(tp)
        
        # Probability-based metrics if probabilities are provided
        if y_prob is not None:
            metrics['roc_auc'] = roc_auc_score(y_true, y_prob)
            metrics['average_precision'] = average_precision_score(y_true, y_prob)
        
        return metrics
    
    @staticmethod
    @log_function()
    def calculate_ndcg(y_true: np.ndarray, y_score: np.ndarray, k: int = 10) -> float:
        """
        Calculate Normalized Discounted Cumulative Gain at k.
        
        Args:
            y_true (np.ndarray): Ground truth relevance scores.
            y_score (np.ndarray): Predicted scores.
            k (int): Number of top items to consider.
            
        Returns:
            float: NDCG@k score.
        """
        # Get indices of top k predictions
        top_k_indices = np.argsort(y_score)[::-1][:k]
        
        # Get true relevance scores of top k predictions
        top_k_relevance = y_true[top_k_indices]
        
        # Calculate DCG
        dcg = np.sum(top_k_relevance / np.log2(np.arange(2, len(top_k_relevance) + 2)))
        
        # Get indices of top k ground truth items
        ideal_indices = np.argsort(y_true)[::-1][:k]
        
        # Get relevance scores of top k ground truth items
        ideal_relevance = y_true[ideal_indices]
        
        # Calculate IDCG
        idcg = np.sum(ideal_relevance / np.log2(np.arange(2, len(ideal_relevance) + 2)))
        
        # Handle case where IDCG is 0
        if idcg == 0:
            return 0.0
        
        # Calculate NDCG
        ndcg = dcg / idcg
        
        return ndcg
    
    @staticmethod
    @log_function()
    def calculate_map(y_true: List[np.ndarray], y_score: List[np.ndarray], k: int = 10) -> float:
        """
        Calculate Mean Average Precision at k.
        
        Args:
            y_true (List[np.ndarray]): List of ground truth relevance scores for each user.
            y_score (List[np.ndarray]): List of predicted scores for each user.
            k (int): Number of top items to consider.
            
        Returns:
            float: MAP@k score.
        """
        average_precisions = []
        
        for true_relevance, scores in zip(y_true, y_score):
            # Get indices of top k predictions
            top_k_indices = np.argsort(scores)[::-1][:k]
            
            # Get true relevance scores of top k predictions
            top_k_relevance = true_relevance[top_k_indices]
            
            # Calculate precision at each position
            precisions = []
            for i in range(len(top_k_relevance)):
                # Only consider relevant items
                if top_k_relevance[i] > 0:
                    # Precision at position i+1
                    precision_at_i = np.sum(top_k_relevance[:i+1] > 0) / (i + 1)
                    precisions.append(precision_at_i)
            
            # Calculate average precision
            if len(precisions) > 0:
                ap = np.mean(precisions)
            else:
                ap = 0.0
            
            average_precisions.append(ap)
        
        # Calculate MAP
        map_score = np.mean(average_precisions)
        
        return map_score
    
    @staticmethod
    @log_function()
    def calculate_hit_rate(y_true: List[np.ndarray], y_score: List[np.ndarray], k: int = 10) -> float:
        """
        Calculate Hit Rate at k.
        
        Args:
            y_true (List[np.ndarray]): List of ground truth relevance scores for each user.
            y_score (List[np.ndarray]): List of predicted scores for each user.
            k (int): Number of top items to consider.
            
        Returns:
            float: Hit Rate@k score.
        """
        hit_count = 0
        
        for true_relevance, scores in zip(y_true, y_score):
            # Get indices of top k predictions
            top_k_indices = np.argsort(scores)[::-1][:k]
            
            # Get true relevance scores of top k predictions
            top_k_relevance = true_relevance[top_k_indices]
            
            # Check if at least one relevant item is in top k
            if np.sum(top_k_relevance > 0) > 0:
                hit_count += 1
        
        # Calculate hit rate
        hit_rate = hit_count / len(y_true)
        
        return hit_rate
    
    @log_function()
    def evaluate_model(self, y_true: np.ndarray, y_pred: np.ndarray, 
                     y_prob: np.ndarray) -> Dict[str, float]:
        """
        Evaluate model performance using multiple metrics.
        
        Args:
            y_true (np.ndarray): Ground truth labels.
            y_pred (np.ndarray): Predicted binary labels.
            y_prob (np.ndarray): Predicted probabilities.
            
        Returns:
            Dict[str, float]: Dictionary with evaluation metrics.
        """
        self.logger.info("Evaluating model performance")
        
        # Calculate classification metrics
        metrics = self.calculate_classification_metrics(y_true, y_pred, y_prob)
        
        # Log metrics
        for metric, value in metrics.items():
            self.logger.info(f"{metric}: {value:.4f}")
            
        return metrics
    
    @log_function()
    def evaluate_by_cohort(self, y_true: np.ndarray, y_pred: np.ndarray, 
                         y_prob: np.ndarray, cohorts: pd.DataFrame) -> pd.DataFrame:
        """
        Evaluate model performance by cohort.
        
        Args:
            y_true (np.ndarray): Ground truth labels.
            y_pred (np.ndarray): Predicted binary labels.
            y_prob (np.ndarray): Predicted probabilities.
            cohorts (pd.DataFrame): DataFrame with cohort information.
            
        Returns:
            pd.DataFrame: DataFrame with evaluation metrics by cohort.
        """
        self.logger.info("Evaluating model performance by cohort")
        
        results = []
        
        # Cohort columns to analyze
        cohort_columns = ['age_group', 'gender', 'region', 'main_genre']
        
        for col in cohort_columns:
            if col in cohorts.columns:
                self.logger.info(f"Evaluating by {col}")
                
                # Get unique cohort values
                cohort_values = cohorts[col].unique()
                
                for value in cohort_values:
                    # Get indices for this cohort
                    cohort_mask = cohorts[col] == value
                    
                    # Skip cohorts with too few samples
                    if np.sum(cohort_mask) < 10:
                        continue
                    
                    # Calculate metrics for this cohort
                    cohort_metrics = self.calculate_classification_metrics(
                        y_true[cohort_mask], y_pred[cohort_mask], y_prob[cohort_mask]
                    )
                    
                    # Add cohort information
                    cohort_metrics['cohort_type'] = col
                    cohort_metrics['cohort_value'] = value
                    cohort_metrics['size'] = np.sum(cohort_mask)
                    
                    results.append(cohort_metrics)
        
        # Convert to DataFrame
        results_df = pd.DataFrame(results)
        
        return results_df
    
    @log_function()
    def evaluate_ranking(self, y_true: List[np.ndarray], y_score: List[np.ndarray], 
                       k_values: List[int] = [5, 10, 20]) -> Dict[str, Dict[int, float]]:
        """
        Evaluate ranking performance using multiple metrics.
        
        Args:
            y_true (List[np.ndarray]): List of ground truth relevance scores for each user.
            y_score (List[np.ndarray]): List of predicted scores for each user.
            k_values (List[int]): List of k values to evaluate.
            
        Returns:
            Dict[str, Dict[int, float]]: Dictionary with ranking metrics for different k values.
        """
        self.logger.info("Evaluating ranking performance")
        
        metrics = {
            'ndcg': {},
            'map': {},
            'hit_rate': {}
        }
        
        for k in k_values:
            # Calculate NDCG@k
            ndcg_values = []
            for user_true, user_score in zip(y_true, y_score):
                ndcg = self.calculate_ndcg(user_true, user_score, k=k)
                ndcg_values.append(ndcg)
            
            metrics['ndcg'][k] = np.mean(ndcg_values)
            
            # Calculate MAP@k
            metrics['map'][k] = self.calculate_map(y_true, y_score, k=k)
            
            # Calculate Hit Rate@k
            metrics['hit_rate'][k] = self.calculate_hit_rate(y_true, y_score, k=k)
            
            # Log metrics
            self.logger.info(f"NDCG@{k}: {metrics['ndcg'][k]:.4f}")
            self.logger.info(f"MAP@{k}: {metrics['map'][k]:.4f}")
            self.logger.info(f"HR@{k}: {metrics['hit_rate'][k]:.4f}")
        
        return metrics 