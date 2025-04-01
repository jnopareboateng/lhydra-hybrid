"""
Visualization package for the recommender system.

This package contains utilities for visualizing model performance,
data distributions, and other relevant analytics.
"""

from .performance_plots import (
    plot_learning_curves,
    plot_confusion_matrix,
    plot_roc_curve,
    plot_precision_recall_curve,
    plot_prediction_distribution,
    plot_metrics_radar,
    plot_embedding_visualization,
    plot_feature_importance,
    plot_metrics_over_time,
    generate_performance_report
)

__all__ = [
    'plot_learning_curves',
    'plot_confusion_matrix',
    'plot_roc_curve',
    'plot_precision_recall_curve',
    'plot_prediction_distribution',
    'plot_metrics_radar',
    'plot_embedding_visualization',
    'plot_feature_importance',
    'plot_metrics_over_time',
    'generate_performance_report'
] 