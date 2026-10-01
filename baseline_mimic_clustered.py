"""
baseline_mimic_clustered.py
===========================
Runs Phase 1 Agglomerative Clustering approach on the NEW MIMIC-IV dataset
to provide the exact 'middle row' comparison for the final review.
"""

import numpy as np
import torch
import os
import sys
import time
from sklearn.metrics import roc_auc_score, f1_score

# Add Phase 1 to path to import Agglomerative clustering
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'fl_physionet_project')))
from clustering import AgglomerativeClientClusterer

# Import Phase 2 tools
from main_phase2 import load_data
from federated_training import partition_dirichlet, prepare_client_data
from models_torch import FederatedMLPTorch, set_flat_weights, get_flat_weights, fedavg_aggregate, train_local
from utils import evaluate_flat

def run_clustered_baseline(rounds=20, epochs=5, seed=42):
    torch.manual_seed(seed)
    np.random.seed(seed)

    print("\n[1/4] Loading local MIMIC-IV data (NEW DATASET)...")
    csv_path = os.path.join(os.path.dirname(__file__), 'mimic_iv_phase2_features.csv')
    X_train, X_test, y_train, y_test = load_data(csv_path, target_col='in_hospital_death', seed=seed)
    input_dim = X_train.shape[1]
    
    # We need feature names for clustering
    import pandas as pd
    df = pd.read_csv(csv_path, nrows=1)
    feature_names = [c for c in df.columns if c != 'in_hospital_death' and c not in ['stay_id', 'subject_id']]

    print("\n[2/4] Partitioning data for 5 clients...")
    client_indices = partition_dirichlet(y_train, 5, alpha=0.5, seed=seed)
    client_data = prepare_client_data(X_train, y_train, client_indices, test_size=0.20, seed=seed, use_smote=False)

    print("\n[3/4] Running Phase 1 Agglomerative Clustering...")
    clusterer = AgglomerativeClientClusterer(n_clusters=None, linkage='ward')
    agglom_labels = clusterer.cluster(client_data, feature_names)
    
    unique_clusters = set(agglom_labels)
    cluster_models = {cid: FederatedMLPTorch(input_dim=input_dim) for cid in unique_clusters}
    
    print("\n[4/4] Training Cluster-Specific Models (Phase 1 Approach)...")
    for rnd in range(1, rounds + 1):
        for cid in unique_clusters:
            # Find clients in this cluster
            cluster_clients = [i for i, l in enumerate(agglom_labels) if l == cid]
            
            client_weights = {}
            client_samples = {}
            global_flat = get_flat_weights(cluster_models[cid])
            
            for client_id in cluster_clients:
                data = client_data[client_id]
                client_model = FederatedMLPTorch(input_dim=input_dim)
                set_flat_weights(client_model, global_flat)
                
                flat_w, _ = train_local(
                    model=client_model,
                    X_train=data['X_train'],
                    y_train=data['y_train'],
                    n_epochs=epochs,
                    batch_size=32,
                    lr=0.001,
                    sample_weights=data.get('sample_weights'),
                    collect_gradients=False
                )
                client_weights[client_id] = flat_w
                client_samples[client_id] = data['n_samples']
                
            # Aggregate for this cluster
            new_global = fedavg_aggregate(client_weights, client_samples)
            set_flat_weights(cluster_models[cid], new_global)
            
        print(f"  Round {rnd:2d} finished for {len(unique_clusters)} clusters.")

    # Evaluate Global Ensemble
    # Weight each cluster model by total training samples in that cluster
    print("\nEvaluating on Global Test Set...")
    cluster_sizes = {}
    for i, cid in enumerate(agglom_labels):
        cluster_sizes[cid] = cluster_sizes.get(cid, 0) + client_data[i]['n_samples']
    total_samples = sum(cluster_sizes.values())
    
    y_pred_ensemble = np.zeros(len(X_test))
    for cid, model in cluster_models.items():
        weight = cluster_sizes[cid] / total_samples
        y_prob = model.predict_proba(X_test)
        y_pred_ensemble += weight * y_prob

    auc = roc_auc_score(y_test, y_pred_ensemble)
    y_bin = (y_pred_ensemble >= 0.5).astype(int)
    f1 = f1_score(y_test, y_bin, zero_division=0)
    
    print("="*50)
    print("  FINAL CLUSTERED BASELINE RESULTS (MIMIC-IV)")
    print("="*50)
    print(f"  Final Global AUC       : {auc:.4f}")
    print(f"  Final Global F1        : {f1:.4f}")
    print("="*50)

if __name__ == '__main__':
    run_clustered_baseline()
