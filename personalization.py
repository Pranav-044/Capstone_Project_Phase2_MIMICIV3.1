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
    lambda_ditto: float = 0.0,
    ditto_lr: float = 0.0005,
    ditto_epochs: int = 5,
) -> tuple:
    """
    For every client, optimize mixture weights α and return personalized models.

    If lambda_ditto > 0, applies Ditto proximal fine-tuning after mixture
    blending to further adapt each hospital's model to its local data while
    preventing excessive drift from the expert blend (Paper 5 fix).

    Parameters
    ----------
    clients_data  : dict {client_id: {'X_val': ..., 'y_val': ...}}
    expert_weights: list of flat np.ndarray OR dict {client_id: list of experts}
    input_dim     : int
    lr_model, n_iter, lr_alpha : MCFL hyperparameters
    verbose       : bool
    lambda_ditto  : float — Ditto proximal penalty weight (0 = disabled)
    ditto_lr      : float — SGD learning rate for Ditto fine-tuning
    ditto_epochs  : int   — number of Ditto fine-tuning epochs

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

        # ── Handle PPFL client-specific experts vs global experts
        if isinstance(expert_weights, dict):
            client_experts = expert_weights[cid]
        else:
            client_experts = expert_weights

        # ── Stage 3a: MCFL — optimize α over expert models ─────────────────
        alpha = optimize_mixture_weights(
            model=scratch,
            X_local=data['X_val'],
            y_local=data['y_val'],
            expert_weights=client_experts,
            n_iter=n_iter,
            lr=lr_alpha,
            verbose=verbose,
        )

        w_personal = _blend(alpha, client_experts)

        # ── Stage 3b: Ditto proximal fine-tuning (Paper 5 fix) ──────────────
        if lambda_ditto > 0.0:
            w_personal = ditto_finetune(
                model=scratch,
                X_local=data['X_val'],
                y_local=data['y_val'],
                global_flat_weights=w_personal,   # anchor = blended expert
                lambda_ditto=lambda_ditto,
                lr=ditto_lr,
                n_epochs=ditto_epochs,
                verbose=verbose,
            )

        personalized_weights[cid] = w_personal
        alpha_matrix[cid] = alpha

        if verbose:
            print(f"    alpha = {alpha.round(3)}")

    return personalized_weights, alpha_matrix


# ─────────────────────────────────────────────────────────────────────────────
#  Ditto Proximal Fine-Tuning  (Paper 5 – Fairness Fix)
#
#  Original Ditto (Li et al., NeurIPS 2021) runs local SGD with a
#  proximal penalty anchored to the GLOBAL FedAvg model.
#
#  Our extension anchors to the MCFL blended expert model (w_alpha),
#  which is a soft-cluster-weighted expert instead of a hard global model.
#  This is the key contribution: Ditto-on-top-of-soft-clustering reduces
#  inter-hospital AUROC disparity better than Ditto-on-top-of-FedAvg.
#
#  Loss:
#    L_total = BCE(y_hat, y) + (lambda_ditto / 2) * ||w_local - w_global||^2
# ─────────────────────────────────────────────────────────────────────────────

def ditto_finetune(
    model: FederatedMLPTorch,
    X_local: np.ndarray,
    y_local: np.ndarray,
    global_flat_weights: np.ndarray,
    lambda_ditto: float = 0.01,
    lr: float = 0.0005,
    n_epochs: int = 5,
    batch_size: int = 32,
    verbose: bool = False,
) -> np.ndarray:
    """
    Fine-tune a personalized model with Ditto proximal regularization.

    Starts from the MCFL blended expert weights (w_alpha) and runs local
    SGD with a proximal penalty that prevents the model from drifting too
    far from the globally-agreed expert blend.

    This resolves Paper 5's open problem: hard-clustering + vanilla Ditto
    worsens fairness for less-biased hospitals because their local model
    drifts into a hard cluster that may not represent them well.
    Anchoring Ditto to the SOFT expert blend instead of the raw global
    model provides a fairer, more adaptive regularization target.

    Loss per mini-batch:
        BCE(y_hat, y) + (lambda_ditto / 2) * ||w_local - w_global||^2

    Parameters
    ----------
    model               : FederatedMLPTorch — reused scratch model
    X_local, y_local    : hospital's local training / validation data
    global_flat_weights : np.ndarray — the blended expert weight vector
                          used as the proximal anchor (w_alpha)
    lambda_ditto        : float — strength of the proximal penalty
                          (0 → pure local training, ∞ → no movement)
    lr                  : float — SGD learning rate
    n_epochs            : int   — number of local fine-tuning epochs
    batch_size          : int   — mini-batch size
    verbose             : bool  — print loss per epoch if True

    Returns
    -------
    np.ndarray — flat personalized weights after Ditto fine-tuning
    """
    import torch
    import torch.nn.functional as F
    from torch.utils.data import DataLoader, TensorDataset
    from models_torch import set_flat_weights, get_flat_weights

    # Start from the blended expert model
    set_flat_weights(model, global_flat_weights)
    model.train()

    # Freeze the proximal anchor (constant reference)
    w_global_t = torch.tensor(global_flat_weights, dtype=torch.float32)

    # Use SGD (momentum) for stability on small local datasets
    optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9)

    X_t = torch.tensor(X_local, dtype=torch.float32)
    y_t = torch.tensor(y_local, dtype=torch.float32)
    dataset = TensorDataset(X_t, y_t)
    loader  = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    for epoch in range(n_epochs):
        epoch_loss = 0.0
        n_batches  = 0

        for X_b, y_b in loader:
            optimizer.zero_grad()

            y_hat = model(X_b)
            bce   = F.binary_cross_entropy(y_hat, y_b)

            # Proximal term: (lambda_ditto / 2) * ||w - w_global||^2
            flat_w = torch.cat([p.flatten() for p in model.parameters()])
            prox   = (lambda_ditto / 2.0) * torch.sum((flat_w - w_global_t) ** 2)

            loss = bce + prox
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            n_batches  += 1

        if verbose:
            avg = epoch_loss / max(n_batches, 1)
            print(f"    [Ditto epoch {epoch+1}/{n_epochs}] loss={avg:.4f}")

    model.eval()
    return get_flat_weights(model)
