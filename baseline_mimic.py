"""
baseline_mimic.py  —  Phase 1 Baseline (FedAvg) on MIMIC-IV Dataset
=====================================================================
Runs plain FedAvg on the new MIMIC-IV dataset to provide a baseline
comparison against the Phase 2 method (FedCM + FedSPD + MCFL).
"""

import numpy as np
import torch
import os
import time

from main_phase2 import load_data  # Loads the local MIMIC-IV CSV
from federated_training import partition_dirichlet, prepare_client_data
from models_torch import FederatedMLPTorch, set_flat_weights, get_flat_weights, fedavg_aggregate, train_local
from utils import evaluate_flat

def run_baseline(rounds=20, epochs=5, seed=42):
    torch.manual_seed(seed)
    np.random.seed(seed)

    print("\n[1/4] Loading local MIMIC-IV data (no BigQuery needed)...")
    csv_path = os.path.join(os.path.dirname(__file__), 'mimic_iv_phase2_features.csv')
    X_train, X_test, y_train, y_test = load_data(csv_path, target_col='in_hospital_death', seed=seed)
    input_dim = X_train.shape[1]

    print("\n[2/4] Partitioning data for 5 clients...")
    client_indices = partition_dirichlet(y_train, 5, alpha=0.5, seed=seed)
    client_data = prepare_client_data(X_train, y_train, client_indices, test_size=0.20, seed=seed, use_smote=False)

    print("\n[3/4] Running Plain FedAvg (Old Approach) on MIMIC-IV...")
    
    # Initialize global model
    global_model = FederatedMLPTorch(input_dim=input_dim)
    global_flat = get_flat_weights(global_model)
    
    # Initialize client models
    client_models = {i: FederatedMLPTorch(input_dim=input_dim) for i in range(5)}
    
    for rnd in range(1, rounds + 1):
        t0 = time.time()
        client_weights = {}
        client_samples = {}
        
        # Local Training
        for cid in range(5):
            set_flat_weights(client_models[cid], global_flat)
            data = client_data[cid]
            flat_w, _ = train_local(
                model=client_models[cid],
                X_train=data['X_train'],
                y_train=data['y_train'],
                n_epochs=epochs,
                batch_size=32,
                lr=0.001,
                sample_weights=data.get('sample_weights'),
                collect_gradients=False # Standard FedAvg doesn't need gradients
            )
            client_weights[cid] = flat_w
            client_samples[cid] = data['n_samples']
            
        # Global Aggregation
        global_flat = fedavg_aggregate(client_weights, client_samples)
        
        # Evaluation
        res = evaluate_flat(global_flat, X_test, y_test, input_dim=input_dim)
        print(f"  Round {rnd:2d} | Global Test AUC: {res['auc']:.4f} | F1: {res['f1']:.4f} | Time: {time.time()-t0:.1f}s")
        
    print("\n" + "="*50)
    print("  FINAL BASELINE RESULTS (MIMIC-IV)")
    print("="*50)
    print(f"  Final Global AUC       : {res['auc']:.4f}")
    print(f"  Final Global F1        : {res['f1']:.4f}")
    print(f"  Final Global Recall    : {res['recall']:.4f}")
    print(f"  Final Global Precision : {res['precision']:.4f}")
    print(f"  Final Global Accuracy  : {res['accuracy']:.4f}")
    print(f"  Threshold              : {res['threshold']:.2f}")
    print("="*50)

if __name__ == '__main__':
    run_baseline()
