"""
soft_membership.py  —  Stage 2: FedSPD-Inspired Soft Cluster Membership
=========================================================================
Converts the gradient-path cosine similarity matrix (Stage 1) into
probabilistic soft membership vectors π for each client, then performs
soft-weighted aggregation to produce K expert cluster models.

Paper: FedSPD — A Soft-clustering Approach for Personalized Decentralized
       Federated Learning (UAI 2025)
"""

import numpy as np


# ─────────────────────────────────────────────────────────────────────────────
#  Soft Membership Computation (FedSPD)
# ─────────────────────────────────────────────────────────────────────────────

def _softmax(x: np.ndarray, temperature: float = 1.0) -> np.ndarray:
    """Numerically stable softmax."""
    x = x / temperature
    e = np.exp(x - np.max(x))
    return e / e.sum()


def compute_soft_membership(sim_matrix: np.ndarray,
                            temperature: float = 1.0,
                            min_clusters: int = 2,
                            max_clusters: int = None) -> tuple:
    """
    Convert K x K cosine similarity matrix into soft membership matrix.

    Each row k gives hospital k's probability of belonging to each
    'virtual cluster' (represented by each other hospital as a proxy).

    To get a compact representation with fewer clusters than clients,
    we first run a lightweight spectral grouping then compute soft
    membership over those cluster centroids.

    Parameters
    ----------
    sim_matrix : np.ndarray, shape (K, K)
    temperature : float — controls sharpness of membership (lower = sharper)
    min_clusters : int — minimum number of clusters
    max_clusters : int — maximum number of clusters (defaults to K-1)

    Returns
    -------
    pi_matrix : np.ndarray, shape (K, n_clusters)
        Soft membership matrix. pi_matrix[k, j] = probability client k
        belongs to cluster j. Rows sum to 1.
    n_clusters : int
        Optimal number of clusters found.
    cluster_labels : np.ndarray, shape (K,)
        Hard cluster assignment for each client (argmax of soft membership).
    """
    K = sim_matrix.shape[0]
    if max_clusters is None:
        max_clusters = max(K - 1, min_clusters)

    # ── Find optimal n_clusters via silhouette on similarity matrix ──
    best_k, best_score = min_clusters, -1.0
    dist_matrix = 1.0 - np.clip(sim_matrix, -1.0, 1.0)  # similarity → distance

    for k in range(min_clusters, min(max_clusters + 1, K)):
        labels = _spectral_cluster(sim_matrix, k)
        score = _silhouette_score(dist_matrix, labels)
        if score > best_score:
            best_score = score
            best_k = k

    cluster_labels = _spectral_cluster(sim_matrix, best_k)

    # ── Compute cluster centroid similarities ──
    # Centroid similarity of client i to cluster j =
    #   mean similarity of client i to all members of cluster j
    pi_matrix = np.zeros((K, best_k))
    for i in range(K):
        centroid_sims = np.zeros(best_k)
        for j in range(best_k):
            members = np.where(cluster_labels == j)[0]
            if len(members) == 0:
                centroid_sims[j] = 0.0
            else:
                centroid_sims[j] = np.mean(sim_matrix[i, members])
        pi_matrix[i] = _softmax(centroid_sims, temperature=temperature)

    return pi_matrix, best_k, cluster_labels


def _spectral_cluster(sim_matrix: np.ndarray, k: int) -> np.ndarray:
    """
    Lightweight spectral clustering on similarity matrix.
    Uses top-k eigenvectors of normalized Laplacian + k-means.
    """
    from numpy.linalg import eigh

    # Degree matrix and normalized Laplacian
    D = np.diag(sim_matrix.sum(axis=1))
    D_inv_sqrt = np.diag(1.0 / np.sqrt(np.maximum(np.diag(D), 1e-10)))
    L_sym = np.eye(len(sim_matrix)) - D_inv_sqrt @ sim_matrix @ D_inv_sqrt

    # Top-k eigenvectors (smallest eigenvalues)
    eigenvalues, eigenvectors = eigh(L_sym)
    U = eigenvectors[:, :k]

    # Normalize rows
    norms = np.linalg.norm(U, axis=1, keepdims=True)
    U = U / np.maximum(norms, 1e-10)

    # K-means on eigenvector rows
    return _kmeans(U, k)


def _kmeans(X: np.ndarray, k: int, n_init: int = 10,
            max_iter: int = 100) -> np.ndarray:
    """Simple K-means returning label array."""
    best_labels, best_inertia = None, np.inf

    for _ in range(n_init):
        # Random initialization
        centers = X[np.random.choice(len(X), k, replace=False)]
        labels = np.zeros(len(X), dtype=int)

        for _ in range(max_iter):
            # Assign
            dists = np.linalg.norm(
                X[:, None, :] - centers[None, :, :], axis=2)
            new_labels = dists.argmin(axis=1)
            if np.all(new_labels == labels):
                break
            labels = new_labels
            # Update centers
            for j in range(k):
                members = X[labels == j]
                if len(members) > 0:
                    centers[j] = members.mean(axis=0)

        inertia = sum(
            np.linalg.norm(X[i] - centers[labels[i]]) ** 2
            for i in range(len(X))
        )
        if inertia < best_inertia:
            best_inertia = inertia
            best_labels = labels.copy()

    return best_labels


def _silhouette_score(dist_matrix: np.ndarray,
                      labels: np.ndarray) -> float:
    """Mean silhouette score over all samples."""
    n = len(labels)
    scores = []
    unique = np.unique(labels)
    if len(unique) < 2:
        return -1.0
    for i in range(n):
        own = labels[i]
        own_members = np.where(labels == own)[0]
        own_members = own_members[own_members != i]
        if len(own_members) == 0:
            a = 0.0
        else:
            a = dist_matrix[i, own_members].mean()
        b_vals = []
        for c in unique:
            if c == own:
                continue
            other = np.where(labels == c)[0]
            b_vals.append(dist_matrix[i, other].mean())
        b = min(b_vals) if b_vals else 0.0
        denom = max(a, b)
        scores.append((b - a) / denom if denom > 0 else 0.0)
    return float(np.mean(scores))


# ─────────────────────────────────────────────────────────────────────────────
#  Soft Weighted Aggregation → Expert Models
# ─────────────────────────────────────────────────────────────────────────────

def soft_aggregate(client_weights: dict,
                   client_n_samples: dict,
                   pi_matrix: np.ndarray,
                   client_ids: list,
                   n_clusters: int) -> list:
    """
    Produce n_clusters expert model weight vectors via soft-weighted FedAvg.

    Expert j's weights:
        w_expert_j = Σ_k (π_{k,j} * n_k * w_k) / Σ_k (π_{k,j} * n_k)

    Parameters
    ----------
    client_weights : dict {client_id: weight_vector}
    client_n_samples : dict {client_id: int}
    pi_matrix : np.ndarray, shape (K, n_clusters)
    client_ids : list — ordered list matching pi_matrix rows
    n_clusters : int

    Returns
    -------
    expert_weights : list of np.ndarray, length n_clusters
        One weight vector per expert cluster model.
    """
    expert_weights = []
    for j in range(n_clusters):
        weighted_sum = None
        total_weight = 0.0
        for i, cid in enumerate(client_ids):
            w = pi_matrix[i, j] * client_n_samples[cid]
            total_weight += w
            if weighted_sum is None:
                weighted_sum = w * client_weights[cid].copy()
            else:
                weighted_sum += w * client_weights[cid]

        if total_weight < 1e-10:
            # Fallback: uniform average if all memberships near zero
            all_w = np.stack([client_weights[cid] for cid in client_ids])
            expert_weights.append(all_w.mean(axis=0))
        else:
            expert_weights.append(weighted_sum / total_weight)

    return expert_weights
