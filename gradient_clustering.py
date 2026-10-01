"""
gradient_clustering.py  —  Stage 1: FedCM Gradient-Path Clustering (PyTorch)
=============================================================================
Uses EXACT per-step gradients from PyTorch autograd — no approximations.

Previous NumPy version used Adam first-moment estimates as a proxy.
This version uses model.parameters().grad directly after each backward pass,
giving true gradient vectors for cosine similarity computation.

Paper: FedCM — IJCAI 2025
"""

import numpy as np
from models_torch import FederatedMLPTorch, train_local


# ─────────────────────────────────────────────────────────────────────────────
#  Stage 1a: Local Training + Exact Gradient Collection
# ─────────────────────────────────────────────────────────────────────────────

def train_with_gradient_collection(
    model: FederatedMLPTorch,
    X: np.ndarray,
    y: np.ndarray,
    n_epochs: int = 5,
    batch_size: int = 32,
    lr: float = 0.001,
    sample_weights: np.ndarray = None,
    global_flat_weights: np.ndarray = None,
    mu: float = 0.01,
    use_fedprox: bool = False,
) -> tuple:
    """
    Train locally and collect EXACT gradient vectors per mini-batch step.

    Returns
    -------
    final_flat_weights : np.ndarray — flat weights after training
    gradient_path      : np.ndarray, shape (n_steps, n_params)
                         True gradient at each mini-batch step (via autograd)
    """
    return train_local(
        model=model,
        X_train=X,
        y_train=y,
        n_epochs=n_epochs,
        batch_size=batch_size,
        lr=lr,
        sample_weights=sample_weights,
        global_flat_weights=(global_flat_weights if use_fedprox else None),
        mu=mu,
        collect_gradients=True,   # exact gradient capture enabled
    )


# ─────────────────────────────────────────────────────────────────────────────
#  Stage 1b: Cosine Similarity Between Gradient Paths
# ─────────────────────────────────────────────────────────────────────────────

def path_cosine_similarity(path_i: np.ndarray,
                           path_j: np.ndarray) -> float:
    """
    Cosine similarity between two gradient-path matrices (flattened).

    Parameters
    ----------
    path_i, path_j : np.ndarray, shape (n_steps, n_params)

    Returns
    -------
    float in [-1, 1]
        +1 = identical learning trajectories
         0 = orthogonal
        -1 = opposite directions
    """
    # Handle mismatched step counts (different batch counts per client)
    min_steps = min(len(path_i), len(path_j))
    flat_i = path_i[:min_steps].flatten()
    flat_j = path_j[:min_steps].flatten()

    ni, nj = np.linalg.norm(flat_i), np.linalg.norm(flat_j)
    if ni < 1e-10 or nj < 1e-10:
        return 0.0

    return float(np.dot(flat_i, flat_j) / (ni * nj))


def build_similarity_matrix(gradient_paths: dict) -> tuple:
    """
    Build symmetric K×K cosine similarity matrix from client gradient paths.

    Parameters
    ----------
    gradient_paths : dict {client_id: np.ndarray of shape (n_steps, n_params)}

    Returns
    -------
    sim_matrix : np.ndarray, shape (K, K)  — symmetric, diagonal = 1.0
    client_ids : list — ordered client IDs matching rows/columns
    """
    client_ids = sorted(gradient_paths.keys())
    K = len(client_ids)
    sim_matrix = np.eye(K)

    for i in range(K):
        for j in range(i + 1, K):
            s = path_cosine_similarity(
                gradient_paths[client_ids[i]],
                gradient_paths[client_ids[j]]
            )
            sim_matrix[i, j] = s
            sim_matrix[j, i] = s

    return sim_matrix, client_ids


# ─────────────────────────────────────────────────────────────────────────────
#  Stage 1c: FedCM Drift Detection
# ─────────────────────────────────────────────────────────────────────────────

def compute_update_directions(prev_weights: dict,
                              curr_weights: dict) -> dict:
    """
    Compute Δw_k = w_new - w_old (update direction) for each client.
    """
    return {k: curr_weights[k] - prev_weights[k] for k in curr_weights}


def detect_drifted_clients(update_directions: dict,
                           pi_matrix: np.ndarray,
                           client_ids: list,
                           tau: float = 0.50) -> list:
    """
    Identify clients whose update direction has diverged from their
    soft-weighted cluster direction.

    Parameters
    ----------
    update_directions : dict {client_id: Δw flat np.ndarray}
    pi_matrix : np.ndarray, shape (n_clients, n_clusters)
    client_ids : list
    tau : float — drift threshold

    Returns
    -------
    list of client IDs flagged as drifted
    """
    drifted = []
    n_clients, n_clusters = pi_matrix.shape

    # Compute aggregate update direction for each cluster expert
    cluster_updates = []
    for c in range(n_clusters):
        c_dir = np.zeros_like(update_directions[client_ids[0]])
        total_pi = 0.0
        for i, cid in enumerate(client_ids):
            w_pi = pi_matrix[i, c]
            c_dir += w_pi * update_directions[cid]
            total_pi += w_pi
        if total_pi > 1e-6:
            c_dir /= total_pi
        cluster_updates.append(c_dir)

    # Compare each client's delta to its expected mixture direction
    for i, cid in enumerate(client_ids):
        delta_k = update_directions[cid]

        expected_dir = np.zeros_like(delta_k)
        for c in range(n_clusters):
            expected_dir += pi_matrix[i, c] * cluster_updates[c]

        nk = np.linalg.norm(delta_k)
        ne = np.linalg.norm(expected_dir)
        if nk < 1e-10 or ne < 1e-10:
            continue

        cos_sim = np.dot(delta_k, expected_dir) / (nk * ne)
        if cos_sim < tau:
            drifted.append(cid)

    return drifted
