"""
models_torch.py  —  PyTorch FederatedMLP for FedICU Phase 2
=============================================================
Identical architecture to Phase 1 NumPy MLP:
    Input(44) → Linear(128) → ReLU → Dropout(0.3)
             → Linear(64)  → ReLU → Dropout(0.2)
             → Linear(32)  → ReLU
             → Linear(1)   → Sigmoid

Key upgrade over Phase 1:
    - Exact gradients via PyTorch autograd (.grad attribute)
    - No more Adam first-moment approximation for FedCM
    - FedProx proximal term added directly to loss (cleaner)
    - He initialisation via nn.init.kaiming_normal_
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset


# ─────────────────────────────────────────────────────────────────────────────
#  Model Definition
# ─────────────────────────────────────────────────────────────────────────────

class FederatedMLPTorch(nn.Module):
    """
    4-layer MLP for federated ICU mortality prediction.
    Architecture: 44 → 128 → 64 → 32 → 1
    """

    def __init__(self, input_dim: int = 44,
                 dropout1: float = 0.30,
                 dropout2: float = 0.20):
        super().__init__()
        self.input_dim = input_dim

        self.net = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Dropout(dropout1),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(dropout2),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid(),
        )
        self._init_weights()

    def _init_weights(self):
        """He (Kaiming) initialisation for all Linear layers."""
        for m in self.net.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight,
                                        mode='fan_in',
                                        nonlinearity='relu')
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Run inference on numpy array, return numpy probabilities."""
        self.eval()
        with torch.no_grad():
            x = torch.tensor(X, dtype=torch.float32)
            return self.forward(x).cpu().numpy()

    def predict_binary(self, X: np.ndarray,
                       threshold: float = 0.5) -> np.ndarray:
        return (self.predict_proba(X) >= threshold).astype(int)


# ─────────────────────────────────────────────────────────────────────────────
#  Weight Utilities (flat numpy ↔ PyTorch parameters)
# ─────────────────────────────────────────────────────────────────────────────

def get_flat_weights(model: FederatedMLPTorch) -> np.ndarray:
    """Flatten all model parameters → 1-D numpy array."""
    return torch.cat([
        p.detach().flatten() for p in model.parameters()
    ]).cpu().numpy()


def set_flat_weights(model: FederatedMLPTorch,
                     flat_weights: np.ndarray):
    """Load a flat numpy weight vector into the model (in-place)."""
    flat_t = torch.tensor(flat_weights, dtype=torch.float32)
    offset = 0
    with torch.no_grad():
        for p in model.parameters():
            n = p.numel()
            p.copy_(flat_t[offset: offset + n].reshape(p.shape))
            offset += n


def clone_model(model: FederatedMLPTorch) -> FederatedMLPTorch:
    """Deep-copy a model (weights only)."""
    new_model = FederatedMLPTorch(model.input_dim)
    set_flat_weights(new_model, get_flat_weights(model))
    return new_model


# ─────────────────────────────────────────────────────────────────────────────
#  FedAvg Aggregation (flat-vector weighted average)
# ─────────────────────────────────────────────────────────────────────────────

def fedavg_aggregate(client_flat_weights: dict,
                     client_n_samples: dict) -> np.ndarray:
    """
    Standard FedAvg: weighted average of client weight vectors.

    Parameters
    ----------
    client_flat_weights : dict {client_id: flat np.ndarray}
    client_n_samples    : dict {client_id: int}

    Returns
    -------
    aggregated : np.ndarray  flat weight vector
    """
    total = sum(client_n_samples.values())
    agg = None
    for cid, w in client_flat_weights.items():
        weight = client_n_samples[cid] / total
        agg = weight * w if agg is None else agg + weight * w
    return agg


# ─────────────────────────────────────────────────────────────────────────────
#  Local Training  (returns flat weights + EXACT gradient path)
# ─────────────────────────────────────────────────────────────────────────────

def train_local(
    model: FederatedMLPTorch,
    X_train: np.ndarray,
    y_train: np.ndarray,
    n_epochs: int = 5,
    batch_size: int = 32,
    lr: float = 0.001,
    sample_weights: np.ndarray = None,
    global_flat_weights: np.ndarray = None,   # for FedProx
    mu: float = 0.01,
    collect_gradients: bool = True,
) -> tuple:
    """
    Train locally for n_epochs and optionally collect gradient path.

    Parameters
    ----------
    model           : FederatedMLPTorch (mutated in-place)
    X_train, y_train: local training data (numpy)
    n_epochs        : local epochs
    batch_size      : mini-batch size
    lr              : Adam learning rate
    sample_weights  : per-sample weights (from SMOTE / class balancing)
    global_flat_weights : if not None, add FedProx proximal term
    mu              : FedProx coefficient
    collect_gradients : whether to record gradient path (Stage 1)

    Returns
    -------
    final_flat_weights : np.ndarray — flat weight vector after training
    gradient_path      : np.ndarray, shape (n_steps, n_params)
                         Exact per-step gradient vectors (NOT approximations)
    """
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr,
                                 betas=(0.9, 0.999), eps=1e-8)

    X_t = torch.tensor(X_train, dtype=torch.float32)
    y_t = torch.tensor(y_train, dtype=torch.float32)
    sw_t = (torch.tensor(sample_weights, dtype=torch.float32)
            if sample_weights is not None else None)

    # FedProx: freeze global weights as reference tensor
    if global_flat_weights is not None:
        global_t = torch.tensor(global_flat_weights, dtype=torch.float32)
    else:
        global_t = None

    dataset = TensorDataset(X_t, y_t) if sw_t is None else \
              TensorDataset(X_t, y_t, sw_t)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    gradient_path = []

    for _ in range(n_epochs):
        for batch in loader:
            optimizer.zero_grad()

            if sw_t is None:
                X_b, y_b = batch
                sw_b = None
            else:
                X_b, y_b, sw_b = batch

            y_hat = model(X_b)

            # Weighted BCE loss
            if sw_b is not None:
                loss = F.binary_cross_entropy(
                    y_hat, y_b, weight=sw_b, reduction='mean')
            else:
                loss = F.binary_cross_entropy(y_hat, y_b)

            # FedProx proximal term: (mu/2) * ||w - w_global||^2
            if global_t is not None:
                flat_local = torch.cat([p.flatten()
                                        for p in model.parameters()])
                prox = (mu / 2.0) * torch.sum((flat_local - global_t) ** 2)
                loss = loss + prox

            loss.backward()

            # ── Exact gradient capture (Stage 1 / FedCM) ──────────────────
            if collect_gradients:
                grads = []
                for p in model.parameters():
                    if p.grad is not None:
                        grads.append(p.grad.detach().flatten())
                    else:
                        grads.append(torch.zeros(p.numel()))
                gradient_path.append(
                    torch.cat(grads).cpu().numpy()
                )
            # ──────────────────────────────────────────────────────────────

            optimizer.step()

    model.eval()
    return get_flat_weights(model), np.array(gradient_path)
