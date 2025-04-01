"""
Visualization utilities for model performance analysis.

This module provides functions for creating various visualizations
to analyze and evaluate the performance of the recommender model.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    confusion_matrix, 
    roc_curve, 
    precision_recall_curve,
    average_precision_score, 
    roc_auc_score,
    auc
)
import logging
from typing import Dict, List, Tuple, Optional, Union
import torch
from pathlib import Path
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import matplotlib.gridspec as gridspec
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.figure import Figure
from matplotlib.axes import Axes
import json

logger = logging.getLogger(__name__)

# Define a consistent color palette for all plots
COLORS = {
    'primary': '#1f77b4',  # Blue
    'secondary': '#ff7f0e',  # Orange
    'tertiary': '#2ca02c',  # Green
    'quaternary': '#d62728',  # Red
    'background': '#f8f9fa',
    'grid': '#e6e6e6',
    'text': '#333333'
}

# Set up the default plot style
def set_plot_style():
    """Set the default style for all plots."""
    plt.style.use('seaborn-v0_8-whitegrid')
    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans', 'Liberation Sans', 'Bitstream Vera Sans', 'sans-serif']
    plt.rcParams['axes.facecolor'] = COLORS['background']
    plt.rcParams['axes.edgecolor'] = COLORS['text']
    plt.rcParams['axes.labelcolor'] = COLORS['text']
    plt.rcParams['figure.facecolor'] = 'white'
    plt.rcParams['text.color'] = COLORS['text']
    plt.rcParams['grid.color'] = COLORS['grid']
    plt.rcParams['grid.linestyle'] = '--'
    plt.rcParams['grid.linewidth'] = 0.6
    plt.rcParams['axes.grid'] = True
    plt.rcParams['axes.axisbelow'] = True
    plt.rcParams['axes.labelsize'] = 12
    plt.rcParams['axes.titlesize'] = 14
    plt.rcParams['xtick.labelsize'] = 10
    plt.rcParams['ytick.labelsize'] = 10
    plt.rcParams['legend.fontsize'] = 10
    plt.rcParams['figure.titlesize'] = 16


def plot_learning_curves(
    train_metrics: Dict[str, List[float]],
    val_metrics: Dict[str, List[float]],
    save_path: str,
    figsize: Tuple[int, int] = (12, 8)
) -> plt.Figure:
    """
    Plot learning curves for training and validation metrics.
    
    Args:
        train_metrics: Dictionary of training metrics by epoch
        val_metrics: Dictionary of validation metrics by epoch
        save_path: Path to save the plot
        figsize: Figure size (width, height) in inches
        
    Returns:
        The matplotlib Figure object
    """
    set_plot_style()
    num_metrics = len(train_metrics)
    
    # Determine the grid layout based on number of metrics
    n_cols = min(3, num_metrics)
    n_rows = (num_metrics + n_cols - 1) // n_cols
    
    fig = plt.figure(figsize=figsize)
    
    for i, (metric_name, train_values) in enumerate(train_metrics.items()):
        val_values = val_metrics.get(metric_name, [])
        
        ax = fig.add_subplot(n_rows, n_cols, i+1)
        epochs = range(1, len(train_values) + 1)
        
        # Plot training values
        ax.plot(epochs, train_values, 'o-', color=COLORS['primary'], label=f'Training {metric_name}')
        
        # Plot validation values if available
        if val_values:
            ax.plot(epochs, val_values, 'o-', color=COLORS['secondary'], label=f'Validation {metric_name}')
        
        ax.set_title(f'{metric_name.capitalize()} over epochs')
        ax.set_xlabel('Epochs')
        ax.set_ylabel(metric_name.capitalize())
        ax.legend()
        
        # Add grid
        ax.grid(True, linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    
    # Save the figure
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    logger.info(f"Learning curves plot saved to {save_path}")
    
    return fig


def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    save_path: str,
    threshold: float = 0.5,
    class_names: List[str] = None,
    figsize: Tuple[int, int] = (8, 7)
) -> plt.Figure:
    """
    Plot confusion matrix for binary classification.
    
    Args:
        y_true: Ground truth binary labels
        y_pred: Predicted probabilities or scores
        save_path: Path to save the plot
        threshold: Threshold for binarizing predictions
        class_names: Names of the classes (default: ['Negative', 'Positive'])
        figsize: Figure size (width, height) in inches
        
    Returns:
        The matplotlib Figure object
    """
    set_plot_style()
    
    if class_names is None:
        class_names = ['Negative', 'Positive']
    
    # Convert predictions to binary if they are probabilities
    y_pred_binary = (y_pred >= threshold).astype(int)
    
    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred_binary)
    
    # Calculate additional metrics
    tn, fp, fn, tp = cm.ravel()
    accuracy = (tp + tn) / (tp + tn + fp + fn)
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    
    # Create a custom colormap from blue to orange
    custom_cmap = LinearSegmentedColormap.from_list(
        'blue_orange', [COLORS['primary'], 'white', COLORS['secondary']], N=100
    )
    
    fig, ax = plt.subplots(figsize=figsize)
    
    # Plot the confusion matrix
    sns.heatmap(
        cm, 
        annot=True, 
        fmt='d', 
        cmap=custom_cmap,
        square=True,
        linewidths=0.5,
        cbar=True,
        cbar_kws={'shrink': 0.8, 'label': 'Count'},
        annot_kws={'size': 14},
        xticklabels=class_names,
        yticklabels=class_names
    )
    
    # Set labels
    ax.set_ylabel('True Label', fontsize=12, labelpad=10)
    ax.set_xlabel('Predicted Label', fontsize=12, labelpad=10)
    
    # Add a title with metrics
    title = f'Confusion Matrix\n'
    title += f'Accuracy: {accuracy:.3f} | Precision: {precision:.3f} | Recall: {recall:.3f} | F1: {f1:.3f}'
    plt.title(title, fontsize=14, pad=20)
    
    plt.tight_layout()
    
    # Save the figure
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    logger.info(f"Confusion matrix plot saved to {save_path}")
    
    return fig


def plot_roc_curve(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    save_path: str,
    figsize: Tuple[int, int] = (8, 8)
) -> plt.Figure:
    """
    Plot ROC curve and calculate AUC.
    
    Args:
        y_true: Ground truth binary labels
        y_pred: Predicted probabilities or scores
        save_path: Path to save the plot
        figsize: Figure size (width, height) in inches
        
    Returns:
        The matplotlib Figure object
    """
    set_plot_style()
    
    # Calculate ROC curve and ROC area
    fpr, tpr, thresholds = roc_curve(y_true, y_pred)
    roc_auc = auc(fpr, tpr)
    
    # Find best threshold based on Youden's J statistic
    optimal_idx = np.argmax(tpr - fpr)
    optimal_threshold = thresholds[optimal_idx]
    
    fig, ax = plt.subplots(figsize=figsize)
    
    # Plot ROC curve
    ax.plot(
        fpr, tpr, 
        color=COLORS['primary'],
        lw=2, 
        label=f'ROC curve (AUC = {roc_auc:.3f})'
    )
    
    # Plot diagonal (random) line
    ax.plot([0, 1], [0, 1], color='gray', lw=1, linestyle='--', label='Random')
    
    # Mark optimal threshold point
    ax.plot(
        fpr[optimal_idx], tpr[optimal_idx], 
        'o', 
        markersize=10,
        fillstyle='none',
        markeredgewidth=2,
        markeredgecolor=COLORS['quaternary'],
        label=f'Optimal threshold = {optimal_threshold:.3f}'
    )
    
    # Set plot properties
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('False Positive Rate (1 - Specificity)', fontsize=12)
    ax.set_ylabel('True Positive Rate (Sensitivity)', fontsize=12)
    ax.set_title('Receiver Operating Characteristic (ROC) Curve', fontsize=14, pad=20)
    ax.legend(loc='lower right', fontsize=10)
    
    # Add grid
    ax.grid(True, linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    
    # Save the figure
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    logger.info(f"ROC curve plot saved to {save_path}")
    
    return fig


def plot_precision_recall_curve(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    save_path: str,
    figsize: Tuple[int, int] = (8, 8)
) -> plt.Figure:
    """
    Plot Precision-Recall curve and calculate Average Precision (AP).
    
    Args:
        y_true: Ground truth binary labels
        y_pred: Predicted probabilities or scores
        save_path: Path to save the plot
        figsize: Figure size (width, height) in inches
        
    Returns:
        The matplotlib Figure object
    """
    set_plot_style()
    
    # Calculate precision-recall curve
    precision, recall, thresholds = precision_recall_curve(y_true, y_pred)
    
    # Calculate average precision
    ap = average_precision_score(y_true, y_pred)
    
    # Calculate F1 score at each threshold and find the best threshold
    f1_scores = 2 * (precision[:-1] * recall[:-1]) / (precision[:-1] + recall[:-1] + 1e-10)
    best_idx = np.argmax(f1_scores)
    best_threshold = thresholds[best_idx]
    best_f1 = f1_scores[best_idx]
    
    fig, ax = plt.subplots(figsize=figsize)
    
    # Plot precision-recall curve
    ax.plot(
        recall, precision, 
        color=COLORS['primary'],
        lw=2, 
        label=f'PR curve (AP = {ap:.3f})'
    )
    
    # Mark best F1 point
    ax.plot(
        recall[best_idx], precision[best_idx], 
        'o', 
        markersize=10,
        fillstyle='none',
        markeredgewidth=2,
        markeredgecolor=COLORS['quaternary'],
        label=f'Best F1 = {best_f1:.3f} at threshold = {best_threshold:.3f}'
    )
    
    # Calculate no skill line (for imbalanced classes)
    no_skill = np.sum(y_true) / len(y_true)
    ax.plot([0, 1], [no_skill, no_skill], linestyle='--', color='gray', label='No Skill')
    
    # Set plot properties
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('Recall', fontsize=12)
    ax.set_ylabel('Precision', fontsize=12)
    ax.set_title('Precision-Recall Curve', fontsize=14, pad=20)
    ax.legend(loc='best', fontsize=10)
    
    # Add grid
    ax.grid(True, linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    
    # Save the figure
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    logger.info(f"Precision-recall curve plot saved to {save_path}")
    
    return fig


def plot_prediction_distribution(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    save_path: str,
    figsize: Tuple[int, int] = (10, 8)
) -> plt.Figure:
    """
    Plot the distribution of predictions for positive and negative samples.
    
    Args:
        y_true: Ground truth binary labels
        y_pred: Predicted probabilities or scores
        save_path: Path to save the plot
        figsize: Figure size (width, height) in inches
        
    Returns:
        The matplotlib Figure object
    """
    set_plot_style()
    
    # Split predictions by true class
    pos_preds = y_pred[y_true == 1]
    neg_preds = y_pred[y_true == 0]
    
    fig, ax = plt.subplots(figsize=figsize)
    
    # Plot histograms
    bins = np.linspace(0, 1, 40)
    
    ax.hist(
        pos_preds, 
        bins=bins, 
        alpha=0.6, 
        color=COLORS['secondary'], 
        label='Positive samples', 
        density=True
    )
    
    ax.hist(
        neg_preds, 
        bins=bins, 
        alpha=0.6, 
        color=COLORS['primary'], 
        label='Negative samples', 
        density=True
    )
    
    # Add a vertical line at threshold 0.5
    ax.axvline(x=0.5, linestyle='--', color=COLORS['quaternary'], linewidth=1.5, 
               label='Default threshold (0.5)')
    
    # Calculate optimal threshold using Youden's J statistic
    fpr, tpr, thresholds = roc_curve(y_true, y_pred)
    optimal_idx = np.argmax(tpr - fpr)
    optimal_threshold = thresholds[optimal_idx]
    
    # Add a vertical line at optimal threshold
    ax.axvline(x=optimal_threshold, linestyle='-', color=COLORS['quaternary'], linewidth=1.5, 
               label=f'Optimal threshold ({optimal_threshold:.3f})')
    
    # Set plot properties
    ax.set_xlabel('Prediction Score', fontsize=12)
    ax.set_ylabel('Density', fontsize=12)
    ax.set_title('Distribution of Predictions by True Class', fontsize=14, pad=20)
    ax.legend(fontsize=10)
    
    # Calculate overlap percentage
    bins_count = 100
    hist_pos, _ = np.histogram(pos_preds, bins=bins_count, range=(0, 1), density=True)
    hist_neg, _ = np.histogram(neg_preds, bins=bins_count, range=(0, 1), density=True)
    overlap = np.sum(np.minimum(hist_pos, hist_neg)) / bins_count
    
    # Add overlap text
    ax.text(
        0.02, 0.95, 
        f'Histogram overlap: {overlap:.1%}', 
        transform=ax.transAxes,
        fontsize=11,
        verticalalignment='top',
        bbox=dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0.8)
    )
    
    plt.tight_layout()
    
    # Save the figure
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    logger.info(f"Prediction distribution plot saved to {save_path}")
    
    return fig


def plot_metrics_radar(
    metrics: Dict[str, float],
    save_path: str,
    figsize: Tuple[int, int] = (8, 8)
) -> plt.Figure:
    """
    Create a radar chart for visualizing multiple metrics.
    
    Args:
        metrics: Dictionary of metrics and their values
        save_path: Path to save the plot
        figsize: Figure size (width, height) in inches
        
    Returns:
        The matplotlib Figure object
    """
    set_plot_style()
    
    # Get metrics and values
    labels = list(metrics.keys())
    values = list(metrics.values())
    
    # Number of metrics
    num_metrics = len(labels)
    
    # Calculate angles for each metric
    angles = np.linspace(0, 2 * np.pi, num_metrics, endpoint=False).tolist()
    
    # Make the plot circular by repeating the first value
    values.append(values[0])
    angles.append(angles[0])
    labels.append(labels[0])
    
    # Create plot
    fig, ax = plt.subplots(figsize=figsize, subplot_kw=dict(polar=True))
    
    # Plot metrics
    ax.plot(angles, values, 'o-', linewidth=2, color=COLORS['primary'])
    ax.fill(angles, values, alpha=0.25, color=COLORS['primary'])
    
    # Set category labels
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels[:-1], fontsize=12)
    
    # Draw y-axis circles
    ax.set_ylim(0, 1)
    ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_yticklabels(['0.2', '0.4', '0.6', '0.8', '1.0'], fontsize=10)
    
    # Add metric values as annotations
    for i, (angle, value) in enumerate(zip(angles[:-1], values[:-1])):
        if value < 0.2:  # Place outside the plot if value is too small
            ha = 'left' if np.cos(angle) < 0 else 'right'
            va = 'bottom' if np.sin(angle) < 0 else 'top'
            xytext = (1.1 * np.cos(angle), 1.1 * np.sin(angle))
            xy = (1.05 * np.cos(angle), 1.05 * np.sin(angle))
            ax.annotate(
                f'{value:.3f}', 
                xy=xy,
                xytext=xytext,
                ha=ha, 
                va=va, 
                fontsize=10,
                bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8)
            )
        else:
            ha = 'left' if np.cos(angle) < 0 else 'right'
            va = 'bottom' if np.sin(angle) < 0 else 'top'
            ax.annotate(
                f'{value:.3f}', 
                xy=(value * np.cos(angle), value * np.sin(angle)),
                xytext=(0.05 * np.cos(angle), 0.05 * np.sin(angle)),
                textcoords='offset points',
                ha=ha, 
                va=va, 
                fontsize=10,
                bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8)
            )
    
    # Set plot title
    plt.title('Model Performance Metrics', fontsize=14, pad=20)
    
    # Adjust grid lines
    ax.grid(True, linestyle='--', alpha=0.7)
    
    # Save the figure
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    logger.info(f"Metrics radar plot saved to {save_path}")
    
    return fig 


def plot_embedding_visualization(
    embeddings: np.ndarray,
    labels: np.ndarray,
    save_path: str,
    method: str = 'tsne',
    perplexity: int = 30,
    n_components: int = 2,
    figsize: Tuple[int, int] = (10, 8)
) -> plt.Figure:
    """
    Visualize embeddings using dimensionality reduction (t-SNE or PCA).
    
    Args:
        embeddings: Embedding vectors (n_samples, n_features)
        labels: Binary labels for coloring points
        save_path: Path to save the plot
        method: Dimensionality reduction method ('tsne' or 'pca')
        perplexity: Perplexity parameter for t-SNE
        n_components: Number of components for dimensionality reduction
        figsize: Figure size (width, height) in inches
        
    Returns:
        The matplotlib Figure object
    """
    set_plot_style()
    
    # Apply dimensionality reduction
    if method.lower() == 'tsne':
        reducer = TSNE(n_components=n_components, perplexity=perplexity, random_state=42)
        reduced_data = reducer.fit_transform(embeddings)
        method_name = f't-SNE (perplexity={perplexity})'
    elif method.lower() == 'pca':
        reducer = PCA(n_components=n_components)
        reduced_data = reducer.fit_transform(embeddings)
        explained_var = reducer.explained_variance_ratio_
        total_var = np.sum(explained_var)
        method_name = f'PCA (explained variance: {total_var:.2%})'
    else:
        raise ValueError(f"Unknown method: {method}. Choose 'tsne' or 'pca'.")
    
    # Create the plot
    fig = plt.figure(figsize=figsize)
    
    if n_components == 2:
        # 2D plot
        ax = fig.add_subplot(111)
        
        scatter = ax.scatter(
            reduced_data[:, 0],
            reduced_data[:, 1],
            c=labels,
            cmap='coolwarm',
            alpha=0.7,
            s=30,
            edgecolors='k',
            linewidths=0.5
        )
        
        # Add legend
        legend = ax.legend(*scatter.legend_elements(), title="Class")
        ax.add_artist(legend)
        
        # Add grid
        ax.grid(True, linestyle='--', alpha=0.7)
        
        # Set labels
        ax.set_xlabel(f'Component 1', fontsize=12)
        ax.set_ylabel(f'Component 2', fontsize=12)
        
    elif n_components == 3:
        # 3D plot
        ax = fig.add_subplot(111, projection='3d')
        
        scatter = ax.scatter(
            reduced_data[:, 0],
            reduced_data[:, 1],
            reduced_data[:, 2],
            c=labels,
            cmap='coolwarm',
            alpha=0.7,
            s=30,
            edgecolors='k',
            linewidths=0.5
        )
        
        # Add legend
        legend = ax.legend(*scatter.legend_elements(), title="Class")
        ax.add_artist(legend)
        
        # Set labels
        ax.set_xlabel(f'Component 1', fontsize=12)
        ax.set_ylabel(f'Component 2', fontsize=12)
        ax.set_zlabel(f'Component 3', fontsize=12)
    
    # Set title
    plt.title(f'Embeddings visualization using {method_name}', fontsize=14, pad=20)
    
    plt.tight_layout()
    
    # Save the figure
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    logger.info(f"Embedding visualization saved to {save_path}")
    
    return fig


def plot_feature_importance(
    feature_names: List[str],
    importance_scores: np.ndarray,
    save_path: str,
    feature_type: str = 'combined',
    figsize: Tuple[int, int] = (12, 10)
) -> plt.Figure:
    """
    Visualize feature importance scores.
    
    Args:
        feature_names: Names of the features
        importance_scores: Importance scores for each feature
        save_path: Path to save the plot
        feature_type: Type of features ('user', 'track', or 'combined')
        figsize: Figure size (width, height) in inches
        
    Returns:
        The matplotlib Figure object
    """
    set_plot_style()
    
    # Sort features by importance
    sorted_indices = np.argsort(importance_scores)[::-1]
    sorted_features = [feature_names[i] for i in sorted_indices]
    sorted_scores = importance_scores[sorted_indices]
    
    # Create figure and axis
    fig, ax = plt.subplots(figsize=figsize)
    
    # Set colors based on feature type
    if feature_type.lower() == 'user':
        colors = COLORS['primary']
        title = 'User Feature Importance'
    elif feature_type.lower() == 'track':
        colors = COLORS['secondary']
        title = 'Track Feature Importance'
    else:
        # Use different colors for user and track features
        colors = []
        for feat in sorted_features:
            if feat.startswith('user_') or feat.startswith('u_'):
                colors.append(COLORS['primary'])
            elif feat.startswith('track_') or feat.startswith('t_'):
                colors.append(COLORS['secondary'])
            else:
                colors.append(COLORS['tertiary'])
        title = 'Feature Importance'
    
    # Create horizontal bar chart
    bars = ax.barh(
        sorted_features,
        sorted_scores,
        color=colors,
        alpha=0.8,
        edgecolor='k',
        linewidth=0.5
    )
    
    # Add values at the end of each bar
    for i, v in enumerate(sorted_scores):
        ax.text(
            v + 0.01,
            i,
            f'{v:.3f}',
            va='center',
            fontsize=9
        )
    
    # Add custom legend for combined features
    if feature_type.lower() == 'combined':
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor=COLORS['primary'], edgecolor='k', label='User features'),
            Patch(facecolor=COLORS['secondary'], edgecolor='k', label='Track features'),
            Patch(facecolor=COLORS['tertiary'], edgecolor='k', label='Other features')
        ]
        ax.legend(handles=legend_elements, loc='lower right')
    
    # Set plot properties
    ax.set_title(title, fontsize=14, pad=20)
    ax.set_xlabel('Importance Score', fontsize=12)
    ax.set_ylabel('Feature', fontsize=12)
    
    # Add grid
    ax.grid(True, linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    
    # Save the figure
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    logger.info(f"Feature importance plot saved to {save_path}")
    
    return fig


def plot_metrics_over_time(
    metrics_history: Dict[str, List[Dict[str, float]]],
    save_path: str,
    figsize: Tuple[int, int] = (12, 10)
) -> plt.Figure:
    """
    Plot multiple metrics over time for different models or runs.
    
    Args:
        metrics_history: Dictionary mapping run/model name to a list of metrics dictionaries per epoch
        save_path: Path to save the plot
        figsize: Figure size (width, height) in inches
        
    Returns:
        The matplotlib Figure object
    """
    set_plot_style()
    
    # Extract all unique metric names
    all_metrics = set()
    for run_metrics in metrics_history.values():
        for epoch_metrics in run_metrics:
            all_metrics.update(epoch_metrics.keys())
    
    # Remove common metrics to exclude
    metrics_to_exclude = {'loss', 'epoch', 'timestamp'}
    all_metrics = [m for m in all_metrics if m not in metrics_to_exclude]
    
    # Determine layout
    n_metrics = len(all_metrics)
    n_cols = min(3, n_metrics)
    n_rows = (n_metrics + n_cols - 1) // n_cols
    
    # Create figure
    fig = plt.figure(figsize=figsize)
    
    # Color map for different runs
    cmap = plt.cm.get_cmap('tab10')
    
    # Plot each metric
    for i, metric in enumerate(all_metrics):
        ax = fig.add_subplot(n_rows, n_cols, i+1)
        
        # Plot each run
        for j, (run_name, run_metrics) in enumerate(metrics_history.items()):
            # Extract metric values
            epochs = range(1, len(run_metrics) + 1)
            values = [m.get(metric, float('nan')) for m in run_metrics]
            
            # Plot the metric
            ax.plot(
                epochs, 
                values, 
                'o-', 
                color=cmap(j % 10), 
                alpha=0.8, 
                label=run_name
            )
        
        # Set plot properties
        ax.set_title(f'{metric.capitalize()}', fontsize=12)
        ax.set_xlabel('Epochs', fontsize=10)
        ax.set_ylabel(metric.capitalize(), fontsize=10)
        
        # Add grid
        ax.grid(True, linestyle='--', alpha=0.7)
        
        # Only add legend to the first plot
        if i == 0:
            ax.legend(fontsize=8)
    
    plt.tight_layout()
    
    # Add overall title
    plt.suptitle('Metrics Evolution Over Time', fontsize=16, y=1.02)
    
    # Save the figure
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    logger.info(f"Metrics over time plot saved to {save_path}")
    
    return fig


def generate_performance_report(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    train_metrics: Dict[str, List[float]],
    val_metrics: Dict[str, List[float]],
    model_name: str,
    output_dir: str,
    embeddings: Optional[np.ndarray] = None,
    feature_names: Optional[List[str]] = None,
    feature_importance: Optional[np.ndarray] = None,
    metrics_history: Optional[Dict[str, List[Dict[str, float]]]] = None
) -> Dict[str, str]:
    """
    Generate a comprehensive performance report with multiple visualizations.
    
    Args:
        y_true: Ground truth binary labels
        y_pred: Predicted probabilities or scores
        train_metrics: Dictionary of training metrics by epoch
        val_metrics: Dictionary of validation metrics by epoch
        model_name: Name of the model (used for filenames)
        output_dir: Directory to save the plots
        embeddings: Optional user/item embeddings for visualization
        feature_names: Optional list of feature names
        feature_importance: Optional feature importance scores
        metrics_history: Optional dictionary of metrics history for different runs
        
    Returns:
        Dictionary mapping plot type to file path
    """
    # Ensure output directory exists
    report_dir = os.path.join(output_dir, f"{model_name}_report")
    os.makedirs(report_dir, exist_ok=True)
    
    # Dictionary to store paths to all generated plots
    plot_paths = {}
    
    # 1. Learning curves
    if train_metrics and val_metrics:
        learning_curves_path = os.path.join(report_dir, "learning_curves.png")
        plot_learning_curves(train_metrics, val_metrics, learning_curves_path)
        plot_paths['learning_curves'] = learning_curves_path
    
    # 2. Confusion matrix
    conf_matrix_path = os.path.join(report_dir, "confusion_matrix.png")
    plot_confusion_matrix(y_true, y_pred, conf_matrix_path)
    plot_paths['confusion_matrix'] = conf_matrix_path
    
    # 3. ROC curve
    roc_curve_path = os.path.join(report_dir, "roc_curve.png")
    plot_roc_curve(y_true, y_pred, roc_curve_path)
    plot_paths['roc_curve'] = roc_curve_path
    
    # 4. Precision-Recall curve
    pr_curve_path = os.path.join(report_dir, "precision_recall_curve.png")
    plot_precision_recall_curve(y_true, y_pred, pr_curve_path)
    plot_paths['precision_recall_curve'] = pr_curve_path
    
    # 5. Prediction distribution
    pred_dist_path = os.path.join(report_dir, "prediction_distribution.png")
    plot_prediction_distribution(y_true, y_pred, pred_dist_path)
    plot_paths['prediction_distribution'] = pred_dist_path
    
    # 6. Metrics radar chart
    # Calculate final metrics
    from sklearn.metrics import (
        accuracy_score, precision_score, recall_score, f1_score,
        roc_auc_score, average_precision_score
    )
    
    metrics = {
        'accuracy': accuracy_score(y_true, y_pred >= 0.5),
        'precision': precision_score(y_true, y_pred >= 0.5),
        'recall': recall_score(y_true, y_pred >= 0.5),
        'f1': f1_score(y_true, y_pred >= 0.5),
        'roc_auc': roc_auc_score(y_true, y_pred),
        'average_precision': average_precision_score(y_true, y_pred)
    }
    
    radar_path = os.path.join(report_dir, "metrics_radar.png")
    plot_metrics_radar(metrics, radar_path)
    plot_paths['metrics_radar'] = radar_path
    
    # 7. Embedding visualization (if provided)
    if embeddings is not None and len(embeddings) == len(y_true):
        # 2D t-SNE
        tsne_2d_path = os.path.join(report_dir, "embedding_tsne_2d.png")
        plot_embedding_visualization(embeddings, y_true, tsne_2d_path, method='tsne', n_components=2)
        plot_paths['tsne_2d'] = tsne_2d_path
        
        # 3D t-SNE
        tsne_3d_path = os.path.join(report_dir, "embedding_tsne_3d.png")
        plot_embedding_visualization(embeddings, y_true, tsne_3d_path, method='tsne', n_components=3)
        plot_paths['tsne_3d'] = tsne_3d_path
        
        # 2D PCA
        pca_path = os.path.join(report_dir, "embedding_pca.png")
        plot_embedding_visualization(embeddings, y_true, pca_path, method='pca', n_components=2)
        plot_paths['pca'] = pca_path
    
    # 8. Feature importance (if provided)
    if feature_names is not None and feature_importance is not None:
        feature_imp_path = os.path.join(report_dir, "feature_importance.png")
        plot_feature_importance(feature_names, feature_importance, feature_imp_path)
        plot_paths['feature_importance'] = feature_imp_path
    
    # 9. Metrics over time for different runs (if provided)
    if metrics_history is not None:
        metrics_time_path = os.path.join(report_dir, "metrics_over_time.png")
        plot_metrics_over_time(metrics_history, metrics_time_path)
        plot_paths['metrics_over_time'] = metrics_time_path
    
    # 10. Save metrics to JSON file
    metrics_path = os.path.join(report_dir, "metrics.json")
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=2)
    plot_paths['metrics_json'] = metrics_path
    
    # Log the report generation
    logger.info(f"Performance report generated in {report_dir}")
    logger.info(f"Generated {len(plot_paths)} visualizations")
    
    return plot_paths 