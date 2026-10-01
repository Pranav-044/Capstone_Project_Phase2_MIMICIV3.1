"""
smoke_test.py — Quick sanity check for all Phase 2 modules
Run: python smoke_test.py
"""
import numpy as np
import torch
torch.manual_seed(42)
np.random.seed(42)

print("=" * 55)
print("  Phase 2 Smoke Test (PyTorch)")
print("=" * 55)

# ── 1. Model ──────────────────────────────────────────────
from models_torch import (FederatedMLPTorch, get_flat_weights,
                          set_flat_weights, train_local)
print("\n[1] FederatedMLPTorch ...")
model = FederatedMLPTorch(input_dim=44)
flat  = get_flat_weights(model)
print(f"    Param count: {len(flat):,}  (expected 16,129)")
assert len(flat) == 16129, f"Wrong param count: {len(flat)}"

# Predict
X_dummy = np.random.randn(10, 44).astype(np.float32)
probs = model.predict_proba(X_dummy)
assert probs.shape == (10,)
assert probs.min() >= 0 and probs.max() <= 1
print(f"    predict_proba OK  |  range: [{probs.min():.3f}, {probs.max():.3f}]")

# ── 2. Local training + gradient collection ───────────────
print("\n[2] train_local + gradient collection ...")
y_dummy = (np.random.rand(100) > 0.85).astype(np.float32)
X_tr = np.random.randn(100, 44).astype(np.float32)
flat_after, grad_path = train_local(
    model=model, X_train=X_tr, y_train=y_dummy,
    n_epochs=2, batch_size=32, lr=0.001, collect_gradients=True)
print(f"    flat_after shape: {flat_after.shape}")
print(f"    grad_path shape:  {grad_path.shape}  "
      f"(steps × params = {grad_path.shape[0]} × {grad_path.shape[1]})")
assert grad_path.shape[1] == 16129
print("    PASS")

# ── 3. Gradient clustering ────────────────────────────────
print("\n[3] Gradient-path similarity matrix ...")
from gradient_clustering import (
    train_with_gradient_collection, build_similarity_matrix)

models  = {i: FederatedMLPTorch(44) for i in range(3)}
flat0   = get_flat_weights(models[0])
for m in models.values():
    set_flat_weights(m, flat0)

grad_paths = {}
for cid, m in models.items():
    _, gp = train_with_gradient_collection(
        model=m, X=X_tr, y=y_dummy,
        n_epochs=1, batch_size=32, lr=0.001)
    grad_paths[cid] = gp

sim_matrix, cids = build_similarity_matrix(grad_paths)
print(f"    Similarity matrix:\n{np.round(sim_matrix, 3)}")
assert sim_matrix.shape == (3, 3)
assert np.allclose(sim_matrix, sim_matrix.T, atol=1e-6), "Not symmetric!"
print("    PASS")

# ── 4. Soft membership ────────────────────────────────────
print("\n[4] Soft membership (FedSPD) ...")
from soft_membership import compute_soft_membership, soft_aggregate
pi_matrix, n_clusters, labels = compute_soft_membership(sim_matrix)
print(f"    n_clusters={n_clusters}  hard_labels={labels}")
print(f"    pi matrix:\n{np.round(pi_matrix, 3)}")
assert pi_matrix.shape == (3, n_clusters)
assert np.allclose(pi_matrix.sum(axis=1), 1.0, atol=1e-5), "Rows don't sum to 1!"

client_weights = {cid: get_flat_weights(models[cid]) for cid in models}
client_n = {cid: 100 for cid in models}
experts = soft_aggregate(client_weights, client_n, pi_matrix, list(models.keys()), n_clusters)
print(f"    {len(experts)} expert weight vectors, each shape {experts[0].shape}")
print("    PASS")

# ── 5. Personalization ────────────────────────────────────
print("\n[5] MCFL personalization ...")
from personalization import optimize_mixture_weights, personalize_all_clients
scratch = FederatedMLPTorch(44)
y_val   = (np.random.rand(20) > 0.85).astype(np.float32)
X_val   = np.random.randn(20, 44).astype(np.float32)
alpha   = optimize_mixture_weights(scratch, X_val, y_val, experts,
                                   n_iter=5, verbose=False)
print(f"    alpha={alpha.round(3)}  sum={alpha.sum():.4f}")
assert abs(alpha.sum() - 1.0) < 1e-5
assert len(alpha) == n_clusters
print("    PASS")

# ── 6. Evaluation ─────────────────────────────────────────
print("\n[6] Evaluation utils ...")
from utils import evaluate_flat
metrics = evaluate_flat(get_flat_weights(scratch), X_val, y_val)
print(f"    AUC={metrics['auc']:.3f}  F1={metrics['f1']:.3f}  "
      f"Acc={metrics['accuracy']:.3f}")
print("    PASS")

print("\n" + "=" * 55)
print("  ALL TESTS PASSED OK")
print("=" * 55)
