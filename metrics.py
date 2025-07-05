"""
This module provides functions to compute common classification metrics.
Functions:
    compute_metrics(y_true, y_pred):
        Computes accuracy, precision, recall, and F1 score for the given true and predicted labels.
        Uses macro averaging and handles zero division by returning 0.
    compute_metrics_embeddings(y_true, y_pred):
        Computes the same metrics as `compute_metrics`. Intended for use with embedding-based classification tasks.
"""
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

def compute_metrics(y_true, y_pred):
    acc = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average='macro', zero_division=0)
    recall = recall_score(y_true, y_pred, average='macro', zero_division=0)
    f1 = f1_score(y_true, y_pred, average='macro', zero_division=0)
    return {
        "accuracy": acc,
        "precision": precision,
        "recall": recall,
        "f1": f1
    }

def compute_metrics_embeddings(y_true, y_pred):
    # Same as above (metrics for ID/classification)
    return compute_metrics(y_true, y_pred)
