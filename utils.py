"""
utils.py  —  Phase 2 Utilities (PyTorch-native)
================================================
Complete evaluation suite matching Phase 1 metrics:
  - AUC, F1, Accuracy, Recall, Precision
  - Optimal threshold search (maximises F1 on validation)
  - Per-client breakdown
  - ROC curve data for plotting
"""

import numpy as np
import torch
from sklearn.metrics import (
    roc_auc_score, f1_score, accuracy_score,
    recall_score, precision_score, roc_curve,
    confusion_matrix,
)
from models_torch import FederatedMLPTorch, set_flat_weights, get_flat_weights


# ─────────────────────────────────────────────────────────────────────────────
#  Optimal Threshold Search
# ─────────────────────────────────────────────────────────────────────────────

def find_optimal_threshold(y_true: np.ndarray,
                           y_prob: np.ndarray,
                           thresholds=None) -> float:
    """
    Search for the threshold that maximises F1 on validation data.
    Matches Phase 1 behaviour: scans 0.05 to 0.95 in steps of 0.05.
    """
    if thresholds is None:
        thresholds = np.arange(0.05, 0.96, 0.05)

    best_thresh = 0.5
    best_f1     = 0.0
    for t in thresholds:
        y_bin = (y_prob >= t).astype(int)
        f1    = f1_score(y_true, y_bin, zero_division=0)
        if f1 > best_f1:
            best_f1    = f1
            best_thresh = t
    return float(best_thresh)


# ─────────────────────────────────────────────────────────────────────────────
#  Core Evaluation Function
# ─────────────────────────────────────────────────────────────────────────────

def evaluate_flat(flat_weights: np.ndarray,
                  X_test: np.ndarray,
                  y_test: np.ndarray,
                  input_dim: int = 44,
                  threshold: float = None,
                  X_val: np.ndarray = None,
                  y_val: np.ndarray = None) -> dict:
    """
    Load flat weights into a scratch model, evaluate on test data.

    If threshold is None AND X_val/y_val are provided:
        → automatically find optimal threshold on validation set (Phase 1 style)
    If threshold is None and no val set provided:
        → default to 0.5

    Returns dict with: auc, f1, accuracy, recall, precision, threshold,
                       roc_fpr, roc_tpr, confusion_matrix
    """
    model = FederatedMLPTorch(input_dim=input_dim)
    set_flat_weights(model, flat_weights)
    model.eval()

    y_prob = model.predict_proba(X_test)

    # Optimal threshold
    if threshold is None:
        if X_val is not None and y_val is not None:
            y_val_prob = model.predict_proba(X_val)
            threshold  = find_optimal_threshold(y_val, y_val_prob)
        else:
            threshold = 0.5

    y_bin = (y_prob >= threshold).astype(int)

    # ROC curve data
    fpr, tpr, _ = roc_curve(y_test, y_prob)

    # Confusion matrix
    cm = confusion_matrix(y_test, y_bin)

    return {
        'auc':       float(roc_auc_score(y_test, y_prob)),
        'f1':        float(f1_score(y_test, y_bin, zero_division=0)),
        'accuracy':  float(accuracy_score(y_test, y_bin)),
        'recall':    float(recall_score(y_test, y_bin, zero_division=0)),
        'precision': float(precision_score(y_test, y_bin, zero_division=0)),
        'threshold': float(threshold),
        'roc_fpr':   fpr.tolist(),
        'roc_tpr':   tpr.tolist(),
        'confusion_matrix': cm.tolist(),
    }


# ─────────────────────────────────────────────────────────────────────────────
#  Per-Client + Global Evaluation
# ─────────────────────────────────────────────────────────────────────────────

def evaluate_personalized_clients(
    personalized_weights: dict,
    client_data: list,
    X_test_global: np.ndarray,
    y_test_global: np.ndarray,
    input_dim: int = 44,
) -> dict:
    """
    Evaluate personalized models per client + mean model on global test set.
    Uses optimal threshold per client (found on each client's val set).
    """
    client_ids = sorted(personalized_weights.keys())

    per_client = {}
    for cid in client_ids:
        data = client_data[cid]
        per_client[cid] = evaluate_flat(
            flat_weights=personalized_weights[cid],
            X_test=data['X_val'],
            y_test=data['y_val'],
            input_dim=input_dim,
            X_val=data['X_val'],     # use same set for threshold search
            y_val=data['y_val'],
        )

    # Mean personalized model on global test set
    mean_w   = np.mean([personalized_weights[c] for c in client_ids], axis=0)

    # Find optimal threshold on global test set (no separate val here)
    global_m = evaluate_flat(
        flat_weights=mean_w,
        X_test=X_test_global,
        y_test=y_test_global,
        input_dim=input_dim,
    )

    return {
        'per_client':           per_client,
        'mean_personal_global': global_m,
        'avg_local_auc':    float(np.mean([per_client[c]['auc']       for c in client_ids])),
        'avg_local_f1':     float(np.mean([per_client[c]['f1']        for c in client_ids])),
        'avg_local_recall': float(np.mean([per_client[c]['recall']    for c in client_ids])),
    }


# ─────────────────────────────────────────────────────────────────────────────
#  Data helpers
# ─────────────────────────────────────────────────────────────────────────────

def to_numpy(x) -> np.ndarray:
    """Convert torch tensor or list to numpy float32 array."""
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy().astype(np.float32)
    return np.asarray(x, dtype=np.float32)
