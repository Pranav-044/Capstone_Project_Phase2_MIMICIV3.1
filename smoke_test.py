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

# ══════════════════════════════════════════════════════
#  EXTENDED TESTS — New modules (Tasks 1-4)
# ══════════════════════════════════════════════════════

print("\n" + "=" * 55)
print("  Extended Tests — Paper 5, 6, 7 Modules")
print("=" * 55)

# ── T1. Body-Head Weight Split (Paper 6 — PPFL) ───────
print("\n[T1] Body-Head weight split ...")
from models_torch import (
    get_body_weights, set_body_weights,
    get_head_weights, set_head_weights,
    body_weight_count, head_weight_count,
)

m = FederatedMLPTorch(input_dim=44)
full_flat = get_flat_weights(m)

body = get_body_weights(m)
head = get_head_weights(m)

body_n = body_weight_count(m)
head_n = head_weight_count(m)

print(f"    Total params : {len(full_flat):,}")
print(f"    Body  params : {body_n:,}  ({body_n/len(full_flat)*100:.1f}%)")
print(f"    Head  params : {head_n:,}  ({head_n/len(full_flat)*100:.1f}%)")

assert body_n + head_n == len(full_flat), \
    f"Body+Head mismatch: {body_n}+{head_n} != {len(full_flat)}"
assert len(body) == body_n
assert len(head) == head_n

# Round-trip: zero body, reload, check body is zero but head unchanged
m2 = FederatedMLPTorch(input_dim=44)
set_flat_weights(m2, full_flat)
set_body_weights(m2, np.zeros_like(body))

body_after = get_body_weights(m2)
head_after = get_head_weights(m2)

assert np.allclose(body_after, 0.0, atol=1e-6), "Body not zeroed!"
assert np.allclose(head_after, head,  atol=1e-6), "Head changed unexpectedly!"
print("    Body+Head round-trip  PASS")
print("    PASS")

# ── T2. Ditto Proximal Fine-Tuning (Paper 5) ──────────
print("\n[T2] Ditto proximal fine-tuning ...")
from personalization import ditto_finetune

X_ditto = np.random.randn(50, 44).astype(np.float32)
y_ditto = (np.random.rand(50) > 0.85).astype(np.float32)

m_ditto = FederatedMLPTorch(44)
w_anchor = get_flat_weights(m_ditto)   # starting anchor

w_finetuned = ditto_finetune(
    model=m_ditto,
    X_local=X_ditto,
    y_local=y_ditto,
    global_flat_weights=w_anchor,
    lambda_ditto=0.01,
    lr=0.0005,
    n_epochs=3,
    verbose=False,
)

# Weights should have changed from anchor
drift = np.linalg.norm(w_finetuned - w_anchor)
print(f"    Weight drift from anchor: {drift:.6f}")
assert drift > 0, "Ditto fine-tuning produced zero weight change!"
assert len(w_finetuned) == len(w_anchor), "Shape mismatch after Ditto!"
print("    PASS")

# ── T3. Fairness Metrics (Paper 5) ────────────────────
print("\n[T3] Fairness metrics ...")
from evaluate_fairness import (
    compute_fairness_metrics,
    comm_reduction_ratio,
    compare_all_methods,
)

dummy_auroc = {0: 0.80, 1: 0.65, 2: 0.72, 3: 0.90, 4: 0.55}
fm = compute_fairness_metrics(dummy_auroc)

print(f"    Mean AUROC   : {fm['mean_auroc']:.4f}  (expected ~0.724)")
print(f"    Std  AUROC   : {fm['std_auroc']:.4f}")
print(f"    Worst AUROC  : {fm['worst_client_auroc']:.4f}  (expected 0.55)")
print(f"    Best  AUROC  : {fm['best_client_auroc']:.4f}  (expected 0.90)")
print(f"    Range AUROC  : {fm['range_auroc']:.4f}  (expected 0.35)")

assert abs(fm['worst_client_auroc'] - 0.55) < 1e-6, "Wrong worst AUROC"
assert abs(fm['best_client_auroc']  - 0.90) < 1e-6, "Wrong best AUROC"
assert abs(fm['range_auroc']        - 0.35) < 1e-6, "Wrong range AUROC"
assert fm['n_clients'] == 5

# Communication overhead check
m_comm = FederatedMLPTorch(44)
ratio  = comm_reduction_ratio(m_comm)
print(f"    Full MB/round: {ratio['full_model_mb']:.4f}")
print(f"    Body MB/round: {ratio['body_only_mb']:.4f}")
print(f"    Head MB/round: {ratio['head_only_mb']:.4f}")
print(f"    Body ratio   : {ratio['reduction_ratio']*100:.1f}% of full model")
print(f"    Savings (Head): {ratio['savings_pct']:.1f}% "
      f"(head is {100-ratio['savings_pct']:.1f}% of full)")
# Body is 86.9% of params; PPFL transmits BODY only → saves the 13.1% head.
# The key PPFL benefit: hospitals keep head weights LOCALLY (no transmission).
# Verify body and head together = full model
assert ratio['body_params'] + ratio['head_params'] == ratio['total_params'], \
    "body + head != total params!"
# The head savings should be > 5% (even for shallow MLPs it's the final layers)
assert ratio['savings_pct'] > 5.0, \
    f"Expected >5% head param savings but got {ratio['savings_pct']:.1f}%"
print("    PASS")

# ── T4. Feature Heterogeneity Simulator (Paper 7) ─────
print("\n[T4] Feature heterogeneity simulation ...")

# Import using importlib to avoid the Phase-1 'models' dependency
# that lives at the top of federated_training.py.
# We instead define the needed functions inline for the test.
import sys, importlib, types

# Temporarily stub the missing 'models' module so the import succeeds
if 'models' not in sys.modules:
    stub = types.ModuleType('models')
    # Provide minimal stubs so the import doesn't crash
    class _FakeMLP: pass
    stub.FederatedMLP    = _FakeMLP
    stub.fedavg_aggregate  = lambda *a, **k: None
    stub.fedprox_aggregate = lambda *a, **k: None
    stub.dp_aggregate      = lambda *a, **k: None
    sys.modules['models'] = stub

from federated_training import (
    simulate_feature_heterogeneity,
    get_available_feature_dims,
    MIMIC_FEATURE_GROUPS,
)
import copy

# Build dummy client_data (3 hospitals)
n_hosp = 3
dummy_clients = []
for _ in range(n_hosp):
    dummy_clients.append({
        'X_train': np.random.randn(80, 44).astype(np.float32),
        'X_val':   np.random.randn(20, 44).astype(np.float32),
        'y_train': np.zeros(80, dtype=np.float32),
        'y_val':   np.zeros(20, dtype=np.float32),
        'n_samples': 100,
        'mortality_rate': 0.1,
        'sample_weights': np.ones(80, dtype=np.float32),
        'indices': np.arange(100),
    })

# Fix assignment: hospital 0 missing vitals (group 0), etc.
fixed_missing = {0: 0, 1: 1, 2: 2}
masked_clients, missing_groups, mask_info = simulate_feature_heterogeneity(
    dummy_clients,
    missing_group_per_hospital=fixed_missing,
    verbose=True,
)

# Verify zeros in masked columns
for cid, info in mask_info.items():
    cols = info['masked_cols']
    if cols:
        train_zeros = masked_clients[cid]['X_train'][:, cols]
        val_zeros   = masked_clients[cid]['X_val']  [:, cols]
        assert np.allclose(train_zeros, 0.0), \
            f"Hospital {cid} X_train not zeroed for cols {cols[:3]}"
        assert np.allclose(val_zeros,   0.0), \
            f"Hospital {cid} X_val not zeroed for cols {cols[:3]}"
        print(f"    Hospital {cid}: cols {cols[:3]}... are zero  OK")

# Verify originals NOT modified
for cid in range(n_hosp):
    cols = mask_info[cid]['masked_cols']
    if cols:
        orig_vals = dummy_clients[cid]['X_train'][:, cols]
        assert not np.allclose(orig_vals, 0.0), \
            "Original client_data was mutated! Should be deep-copied."

# Available dim check
avail_dims = get_available_feature_dims(mask_info, total_features=44)
for cid, dim in avail_dims.items():
    expected = 44 - mask_info[cid]['n_masked']
    assert dim == expected, f"Hospital {cid}: dim {dim} != expected {expected}"
print(f"    Available dims per hospital: {avail_dims}")
print("    PASS")

print("\n" + "=" * 55)
print("  ALL EXTENDED TESTS PASSED OK")
print("=" * 55)
