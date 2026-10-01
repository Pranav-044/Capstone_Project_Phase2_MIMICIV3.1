"""
phase2_training.py  —  Combined Phase 2 Training Pipeline (PyTorch)
====================================================================
Three-stage FL loop:

  Stage 1 (FedCM)  → exact gradient-path cosine similarity matrix
  Stage 2 (FedSPD) → soft membership matrix π via softmax
  Stage 3 (MCFL)   → per-hospital personalized model via mixture weights α
"""

import numpy as np
import time
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__),
                             '..', 'fl_physionet_project'))
from federated_training import partition_dirichlet, prepare_client_data

from models_torch import FederatedMLPTorch, set_flat_weights, get_flat_weights
from gradient_clustering import (
    train_with_gradient_collection,
    build_similarity_matrix,
    compute_update_directions,
    detect_drifted_clients,
)
from soft_membership import compute_soft_membership, soft_aggregate
from personalization import personalize_all_clients
from utils import evaluate_flat, evaluate_personalized_clients


# ─────────────────────────────────────────────────────────────────────────────
#  Main Training Loop
# ─────────────────────────────────────────────────────────────────────────────

def run_phase2(
    client_data: list,
    X_test: np.ndarray,
    y_test: np.ndarray,
    n_rounds: int = 20,
    n_epochs: int = 5,
    batch_size: int = 32,
    input_dim: int = 44,
    lr: float = 0.001,
    use_fedprox: bool = False,
    mu: float = 0.01,
    soft_temperature: float = 1.0,
    drift_tau: float = 0.50,
    mcfl_n_iter: int = 60,
    mcfl_lr_alpha: float = 0.10,
    verbose: bool = True,
    seed: int = 42,
) -> dict:
    """
    Run Phase 2 combined FL pipeline.

    Parameters
    ----------
    client_data  : list of dicts from prepare_client_data()
    X_test, y_test : global test set (never seen during training)
    n_rounds, n_epochs, batch_size, input_dim, lr : training hyperparameters
    use_fedprox, mu : FedProx settings
    soft_temperature : softmax temperature for FedSPD membership
    drift_tau : FedCM drift detection threshold
    mcfl_n_iter, mcfl_lr_alpha : MCFL α optimization settings
    verbose, seed

    Returns
    -------
    dict with round_metrics, personalized_metrics, alpha_history,
         sim_history, pi_history, drift_history,
         final_expert_weights, final_personal_weights
    """
    torch_seed = seed
    import torch
    torch.manual_seed(torch_seed)
    np.random.seed(seed)

    n_clients  = len(client_data)
    client_ids = list(range(n_clients))

    # ── Initialize one PyTorch model per client ───────────────────────────────
    models = {cid: FederatedMLPTorch(input_dim=input_dim)
              for cid in client_ids}

    # All clients start from the same random init (reproducibility)
    init_flat = get_flat_weights(models[0])
    for cid in client_ids:
        set_flat_weights(models[cid], init_flat)

    # ── History ───────────────────────────────────────────────────────────────
    round_metrics       = []
    alpha_history       = []
    sim_history         = []
    pi_history          = []
    drift_history       = []

    prev_weights        = {cid: init_flat.copy() for cid in client_ids}
    personalized_weights = None
    expert_weights      = None
    pi_matrix           = None

    # ── Round loop ────────────────────────────────────────────────────────────
    for rnd in range(1, n_rounds + 1):
        t0 = time.time()
        if verbose:
            print(f"\n{'='*60}")
            print(f"  ROUND {rnd}/{n_rounds}")
            print(f"{'='*60}")

        # ── STAGE 1: Local training + exact gradient collection ───────────────
        curr_weights   = {}
        gradient_paths = {}

        for cid in client_ids:
            data = client_data[cid]

            # Warm-start from personalized model (Round > 1)
            if personalized_weights is not None:
                set_flat_weights(models[cid], personalized_weights[cid])

            flat_w, grad_path = train_with_gradient_collection(
                model=models[cid],
                X=data['X_train'],
                y=data['y_train'],
                n_epochs=n_epochs,
                batch_size=batch_size,
                lr=lr,
                sample_weights=data.get('sample_weights'),
                global_flat_weights=(prev_weights[cid] if use_fedprox else None),
                mu=mu,
                use_fedprox=use_fedprox,
            )
            curr_weights[cid]   = flat_w
            gradient_paths[cid] = grad_path

            if verbose:
                print(f"  Client {cid}: trained  "
                      f"(n={data['n_samples']}, "
                      f"grad steps={len(grad_path)})")

        # ── FedCM Drift Detection (Round > 1) ─────────────────────────────────
        drifted = []
        if rnd > 1 and pi_matrix is not None:
            update_dirs = compute_update_directions(prev_weights, curr_weights)
            drifted = detect_drifted_clients(
                update_dirs, pi_matrix, client_ids, tau=drift_tau)
            if verbose and drifted:
                print(f"  [FedCM] Drifted clients detected: {drifted}")
        drift_history.append(drifted)

        # ── STAGE 2a: Gradient-path cosine similarity matrix ─────────────────
        sim_matrix, _ = build_similarity_matrix(gradient_paths)
        sim_history.append(sim_matrix.copy())

        if verbose:
            print(f"\n  [FedSPD] Cosine similarity matrix:\n"
                  f"{np.round(sim_matrix, 3)}")

        # ── STAGE 2b: Soft membership π ───────────────────────────────────────
        pi_matrix, n_clusters, hard_labels = compute_soft_membership(
            sim_matrix, temperature=soft_temperature)
        pi_history.append(pi_matrix.copy())

        if verbose:
            print(f"  [FedSPD] Clusters={n_clusters}  "
                  f"hard_labels={hard_labels}")
            print(f"  [FedSPD] Soft pi:\n{np.round(pi_matrix, 3)}")

        # ── STAGE 2c: Soft-weighted expert aggregation ────────────────────────
        client_n_samples = {cid: client_data[cid]['n_samples']
                            for cid in client_ids}
        expert_weights = soft_aggregate(
            curr_weights, client_n_samples,
            pi_matrix, client_ids, n_clusters)

        if verbose:
            print(f"  [FedSPD] Produced {n_clusters} expert models")

        # ── STAGE 3: MCFL personalization ─────────────────────────────────────
        clients_val = {
            cid: {'X_val': client_data[cid]['X_val'],
                  'y_val': client_data[cid]['y_val']}
            for cid in client_ids
        }

        if verbose:
            print(f"\n  [MCFL] Optimizing mixture weights alpha ...")

        personalized_weights, alpha_dict = personalize_all_clients(
            clients_data=clients_val,
            expert_weights=expert_weights,
            input_dim=input_dim,
            lr_model=lr,
            n_iter=mcfl_n_iter,
            lr_alpha=mcfl_lr_alpha,
            verbose=False,
        )
        alpha_history.append({cid: alpha_dict[cid].round(4)
                               for cid in client_ids})

        if verbose:
            for cid in client_ids:
                print(f"    Client {cid}: alpha = {alpha_dict[cid].round(3)}")

        # ── Evaluate ──────────────────────────────────────────────────────────
        eval_res = evaluate_personalized_clients(
            personalized_weights, client_data,
            X_test, y_test, input_dim)

        gm = eval_res['mean_personal_global']
        round_metrics.append({
            'round':         rnd,
            'global_auc':    gm['auc'],
            'global_f1':     gm['f1'],
            'global_acc':    gm['accuracy'],
            'global_recall': gm['recall'],
            'global_prec':   gm['precision'],
            'threshold':     gm['threshold'],
            'avg_local_auc': eval_res['avg_local_auc'],
            'avg_local_f1':  eval_res.get('avg_local_f1', 0.0),
            'avg_local_rec': eval_res.get('avg_local_recall', 0.0),
            'n_clusters':    n_clusters,
            'drifted':       drifted,
            'time_sec':      time.time() - t0,
        })

        if verbose:
            rm = round_metrics[-1]
            print(f"\n  OK Round {rnd} | "
                  f"Global AUC={rm['global_auc']:.4f} | "
                  f"F1={rm['global_f1']:.4f} | "
                  f"Recall={rm['global_recall']:.4f} | "
                  f"Thresh={rm['threshold']:.2f} | "
                  f"Clusters={rm['n_clusters']} | "
                  f"Time={rm['time_sec']:.1f}s")

        # Update prev weights for next round
        prev_weights = {cid: curr_weights[cid].copy() for cid in client_ids}

    # ── Final evaluation ──────────────────────────────────────────────────────
    final_eval = evaluate_personalized_clients(
        personalized_weights, client_data, X_test, y_test, input_dim)

    print("\n" + "=" * 60)
    print("  PHASE 2 COMPLETE")
    print("=" * 60)
    fm = final_eval['mean_personal_global']
    print(f"  Final Global AUC       : {fm['auc']:.4f}")
    print(f"  Final Global F1        : {fm['f1']:.4f}")
    print(f"  Final Global Recall    : {fm['recall']:.4f}")
    print(f"  Final Global Precision : {fm['precision']:.4f}")
    print(f"  Final Global Accuracy  : {fm['accuracy']:.4f}")
    print(f"  Optimal Threshold      : {fm['threshold']:.2f}")
    print(f"  Final Avg Local AUC    : {final_eval['avg_local_auc']:.4f}")
    print("=" * 60)

    return {
        'round_metrics':          round_metrics,
        'personalized_metrics':   final_eval,
        'alpha_history':          alpha_history,
        'sim_history':            sim_history,
        'pi_history':             pi_history,
        'drift_history':          drift_history,
        'final_expert_weights':   expert_weights,
        'final_personal_weights': personalized_weights,
    }
