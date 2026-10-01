"""
personalization.py  —  Stage 3: MCFL Multi-Expert Personalization (PyTorch)
=============================================================================
Each hospital optimizes mixture weights α over K expert models such that:

    w_personalized = Σ_j  α_j * w_expert_j

α is found by minimizing the hospital's local BCE loss using
finite-difference gradient descent in the softmax-parameterized space.

PyTorch upgrade: loss evaluation uses exact PyTorch forward pass
(faster and more numerically stable than the NumPy version).

Paper: MCFL — Big Data Mining and Analytics, 2024
"""

import numpy as np
import torch
import torch.nn.functional as F
from models_torch import FederatedMLPTorch, set_flat_weights, get_flat_weights


# ─────────────────────────────────────────────────────────────────────────────
#  Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _softmax(z: np.ndarray) -> np.ndarray:
    e = np.exp(z - np.max(z))
    return e / e.sum()


def _blend(alpha: np.ndarray, expert_weights: list) -> np.ndarray:
    """Compute α-blended flat weight vector."""
    out = np.zeros_like(expert_weights[0])
    for j, w in enumerate(expert_weights):
        out += alpha[j] * w
    return out


def _bce_loss_torch(model: FederatedMLPTorch,
                    flat_w: np.ndarray,
                    X: np.ndarray,
                    y: np.ndarray) -> float:
    """Load flat weights into model, run forward pass, return BCE loss."""
    set_flat_weights(model, flat_w)
    model.eval()
    with torch.no_grad():
        X_t = torch.tensor(X, dtype=torch.float32)
        y_t = torch.tensor(y, dtype=torch.float32)
        y_hat = model(X_t)
        loss = F.binary_cross_entropy(y_hat, y_t)
    return float(loss.item())


# ─────────────────────────────────────────────────────────────────────────────
#  Mixture Weight Optimization
# ─────────────────────────────────────────────────────────────────────────────

def optimize_mixture_weights(
    model: FederatedMLPTorch,
    X_local: np.ndarray,
    y_local: np.ndarray,
    expert_weights: list,
    n_iter: int = 60,
    lr: float = 0.10,
    epsilon: float = 1e-3,
    verbose: bool = False,
) -> np.ndarray:
    """
    Find optimal mixture weights α over expert models via finite-difference
    gradient descent in softmax-parameterized space.

    Optimization problem:
        min_{z}  BCE( model(Σ_j softmax(z)_j * w_expert_j),  y_local )

    Parameters
    ----------
    model         : scratch FederatedMLPTorch used for evaluation only
    X_local, y_local : hospital's local validation set
    expert_weights : list of flat np.ndarray — one per cluster expert
    n_iter        : gradient descent iterations
    lr            : learning rate for z update
    epsilon       : finite-difference step size
    verbose       : print progress

    Returns
    -------
    alpha : np.ndarray, shape (n_experts,) — optimal mixture weights (sum=1)
    """
    n_experts = len(expert_weights)

    # Save model state — restore after optimization
    saved_flat = get_flat_weights(model)

    # Initialize z uniformly (equal mixture)
    z = np.zeros(n_experts)

    for it in range(n_iter):
        alpha = _softmax(z)
        w_center = _blend(alpha, expert_weights)
        loss_center = _bce_loss_torch(model, w_center, X_local, y_local)

        # Finite-difference gradient of loss w.r.t. z
        grad_z = np.zeros(n_experts)
        for i in range(n_experts):
            z_plus = z.copy();  z_plus[i] += epsilon
            z_minus = z.copy(); z_minus[i] -= epsilon

            w_plus  = _blend(_softmax(z_plus),  expert_weights)
            w_minus = _blend(_softmax(z_minus), expert_weights)

            loss_plus  = _bce_loss_torch(model, w_plus,  X_local, y_local)
            loss_minus = _bce_loss_torch(model, w_minus, X_local, y_local)

            grad_z[i] = (loss_plus - loss_minus) / (2.0 * epsilon)

        z = z - lr * grad_z

        if verbose and (it % 10 == 0 or it == n_iter - 1):
            print(f"    [MCFL iter {it:03d}]  "
                  f"loss={loss_center:.4f}  "
                  f"alpha={_softmax(z).round(3)}")

    # Restore model
    set_flat_weights(model, saved_flat)

    return _softmax(z)


# ─────────────────────────────────────────────────────────────────────────────
#  Personalize All Clients
# ─────────────────────────────────────────────────────────────────────────────

def personalize_all_clients(
    clients_data: dict,
    expert_weights: list,
    input_dim: int = 44,
    lr_model: float = 0.001,
    n_iter: int = 60,
    lr_alpha: float = 0.10,
    verbose: bool = False,
) -> tuple:
    """
    For every client, optimize mixture weights α and return personalized models.

    Parameters
    ----------
    clients_data : dict {client_id: {'X_val': ..., 'y_val': ...}}
    expert_weights : list of flat np.ndarray (one per cluster expert)
    input_dim : int
    lr_model, n_iter, lr_alpha : hyperparameters
    verbose : bool

    Returns
    -------
    personalized_weights : dict {client_id: flat np.ndarray}
    alpha_matrix         : dict {client_id: np.ndarray of mixture weights}
    """
    # One scratch model shared across all clients (weights replaced per client)
    scratch = FederatedMLPTorch(input_dim=input_dim)

    personalized_weights = {}
    alpha_matrix = {}

    for cid, data in clients_data.items():
        if verbose:
            print(f"  Personalizing client {cid} ...")

        alpha = optimize_mixture_weights(
            model=scratch,
            X_local=data['X_val'],
            y_local=data['y_val'],
            expert_weights=expert_weights,
            n_iter=n_iter,
            lr=lr_alpha,
            verbose=verbose,
        )

        w_personal = _blend(alpha, expert_weights)
        personalized_weights[cid] = w_personal
        alpha_matrix[cid] = alpha

        if verbose:
            print(f"    alpha = {alpha.round(3)}")

    return personalized_weights, alpha_matrix
