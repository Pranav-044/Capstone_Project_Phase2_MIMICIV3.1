"""
Federated Training Pipeline
============================
Implements:
  - Dirichlet Non-IID data partitioning
  - SMOTE oversampling per client  (optional, --use_smote)
  - FedAvg training  (tracks per-round val metrics + client weight drift)
  - FedProx training (proximal penalty, better for Non-IID)
  - Clustered FL training  (static, one-shot clustering)
  - Adaptive Clustered FL training  (re-clusters every N rounds)
"""

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
try:
    from models import FederatedMLP, fedavg_aggregate, fedprox_aggregate, dp_aggregate
except ModuleNotFoundError:
    pass


# ──────────────────────────────────────────────────────────────────────────────
#  Data partitioning
# ──────────────────────────────────────────────────────────────────────────────

def partition_dirichlet(y, n_clients, alpha=0.5, seed=42):
    """Partition data among clients using Dirichlet distribution (Non-IID)."""
    np.random.seed(seed)
    n_classes = len(np.unique(y))
    client_indices = [[] for _ in range(n_clients)]

    for class_idx in range(n_classes):
        class_indices = np.where(y == class_idx)[0]
        np.random.shuffle(class_indices)

        proportions = np.random.dirichlet([alpha] * n_clients)
        proportions = np.maximum(proportions, 0.01)
        proportions /= proportions.sum()

        splits = (proportions * len(class_indices)).astype(int)
        diff = len(class_indices) - splits.sum()
        for i in range(abs(diff)):
            splits[i % n_clients] += 1 if diff > 0 else -1

        current = 0
        for client_id in range(n_clients):
            end = current + splits[client_id]
            client_indices[client_id].extend(class_indices[current:end].tolist())
            current = end

    for i in range(n_clients):
        np.random.shuffle(client_indices[i])
        client_indices[i] = np.array(client_indices[i])

    return client_indices


def _apply_smote(X_train, y_train, seed, client_idx):
    """Apply SMOTE to balance a client's training data. Returns augmented X, y."""
    try:
        from imblearn.over_sampling import SMOTE
    except ImportError:
        if client_idx == 0:
            print("    WARNING: imbalanced-learn not installed. "
                  "Run: pip install imbalanced-learn")
        return X_train, y_train

    n_pos = int(y_train.sum())
    n_neg = int(len(y_train) - n_pos)

    if n_pos < 2 or n_neg < 2:
        return X_train, y_train   # not enough samples for SMOTE

    k = min(5, n_pos - 1)
    if k < 1:
        return X_train, y_train

    try:
        sm = SMOTE(random_state=seed, k_neighbors=k)
        X_res, y_res = sm.fit_resample(X_train, y_train)
        print(f"    Client {client_idx}: SMOTE {n_pos} -> "
              f"{int(y_res.sum())} positives | total {len(y_res)}")
        return X_res, y_res
    except Exception as e:
        print(f"    Client {client_idx}: SMOTE skipped ({e})")
        return X_train, y_train


def prepare_client_data(X, y, client_indices, test_size=0.2, seed=42,
                        use_smote=False):
    """Split each client's data into train/validation sets.

    Parameters
    ----------
    use_smote : bool
        If True, applies SMOTE on each client's training split to address
        class imbalance. Requires imbalanced-learn.

    Returns
    -------
    client_data : list of dict
        Keys: X_train, y_train, sample_weights, X_val, y_val,
              indices, mortality_rate, n_samples
    """
    if use_smote:
        print("  Applying SMOTE oversampling per client ...")

    client_data = []
    for idx, indices in enumerate(client_indices):
        X_client = X[indices]
        y_client = y[indices]

        try:
            X_train, X_val, y_train, y_val = train_test_split(
                X_client, y_client, test_size=test_size,
                stratify=y_client, random_state=seed
            )
        except ValueError:
            X_train, X_val, y_train, y_val = train_test_split(
                X_client, y_client, test_size=test_size, random_state=seed
            )

        # Apply SMOTE on training split only (never on validation)
        if use_smote:
            X_train, y_train = _apply_smote(X_train, y_train, seed, idx)

        # Sample weights (used when SMOTE is off; with SMOTE data is balanced)
        if use_smote:
            sample_weights = np.ones_like(y_train, dtype=float)
        else:
            n_pos = np.sum(y_train)
            n_neg = len(y_train) - n_pos
            if n_pos > 0 and n_neg > 0:
                w_pos = len(y_train) / (2.0 * n_pos)
                w_neg = len(y_train) / (2.0 * n_neg)
                sample_weights = np.where(y_train == 1, w_pos, w_neg)
            else:
                sample_weights = np.ones_like(y_train, dtype=float)

        client_data.append({
            'X_train': X_train,
            'y_train': y_train,
            'sample_weights': sample_weights,
            'X_val': X_val,
            'y_val': y_val,
            'indices': indices,
            'mortality_rate': y_client.mean(),
            'n_samples': len(indices)
        })

    return client_data


# ──────────────────────────────────────────────────────────────────────────────
#  Internal: compute validation metrics on combined client val sets
# ──────────────────────────────────────────────────────────────────────────────

def _round_val_metrics(model, client_data, threshold=0.5):
    """Compute accuracy, F1, AUC on combined validation data for one round."""
    all_y_true, all_y_pred = [], []
    for data in client_data:
        all_y_true.extend(data['y_val'].tolist())
        all_y_pred.extend(model.predict(data['X_val']).tolist())

    y_true = np.array(all_y_true)
    y_pred = np.array(all_y_pred)
    y_bin  = (y_pred >= threshold).astype(int)

    acc = accuracy_score(y_true, y_bin)
    f1  = f1_score(y_true, y_bin, zero_division=0)
    try:
        auc_ = roc_auc_score(y_true, y_pred)
    except ValueError:
        auc_ = 0.5

    return acc, f1, auc_


def _weight_l2_drift(local_weights, global_weights):
    """L2 norm between local and global flattened weight vectors."""
    local_flat  = np.concatenate([w.flatten() for w in local_weights])
    global_flat = np.concatenate([w.flatten() for w in global_weights])
    return float(np.linalg.norm(local_flat - global_flat))


# ──────────────────────────────────────────────────────────────────────────────
#  FedAvg
# ──────────────────────────────────────────────────────────────────────────────

def train_fedavg(client_data, input_dim, n_rounds=20, local_epochs=5, lr=0.001):
    """Train a global model using standard Federated Averaging.

    Returns
    -------
    global_model       : FederatedMLP
    round_losses       : list[float]     — per-round weighted avg training loss
    client_histories   : list[dict]
    round_val_metrics  : dict            — {'accuracy': [...], 'f1': [...], 'auc': [...]}
    drift_matrix       : np.ndarray      — shape (n_rounds, n_clients), L2 weight drift
    """
    print(f"\n{'='*60}")
    print(f"  FEDERATED AVERAGING (FedAvg)")
    print(f"  Clients: {len(client_data)} | Rounds: {n_rounds} | "
          f"Local Epochs: {local_epochs}")
    print(f"{'='*60}")

    global_model = FederatedMLP(input_dim, lr=lr)
    round_losses  = []
    client_histories = [{'train_losses': [], 'val_losses': []}
                        for _ in range(len(client_data))]
    round_val_metrics = {'accuracy': [], 'f1': [], 'auc': []}
    drift_rows = []

    for round_idx in range(n_rounds):
        all_weights, all_sizes, round_loss = [], [], 0
        global_w = global_model.get_weights()
        round_drifts = []

        for client_idx, data in enumerate(client_data):
            local_model = FederatedMLP(input_dim, lr=lr)
            local_model.set_weights(global_w)

            for _ in range(local_epochs):
                loss = local_model.train_step(
                    data['X_train'], data['y_train'],
                    data.get('sample_weights')
                )

            # Drift BEFORE aggregation
            round_drifts.append(
                _weight_l2_drift(local_model.get_weights(), global_w)
            )

            val_pred = local_model.predict(data['X_val'])
            val_loss = local_model.compute_loss(
                val_pred.reshape(-1, 1), data['y_val']
            )
            client_histories[client_idx]['train_losses'].append(loss)
            client_histories[client_idx]['val_losses'].append(val_loss)

            all_weights.append(local_model.get_weights())
            all_sizes.append(data['n_samples'])
            round_loss += loss * data['n_samples']

        global_model.set_weights(fedavg_aggregate(all_weights, all_sizes))

        avg_loss = round_loss / sum(all_sizes)
        round_losses.append(avg_loss)
        drift_rows.append(round_drifts)

        # Per-round global val metrics
        acc, f1, auc_ = _round_val_metrics(global_model, client_data)
        round_val_metrics['accuracy'].append(acc)
        round_val_metrics['f1'].append(f1)
        round_val_metrics['auc'].append(auc_)

        if (round_idx + 1) % 5 == 0:
            print(f"  Round {round_idx+1:3d}/{n_rounds} | "
                  f"Loss: {avg_loss:.4f} | Val Acc: {acc:.4f} | F1: {f1:.4f}")

    return (global_model, round_losses, client_histories,
            round_val_metrics, np.array(drift_rows))


# ──────────────────────────────────────────────────────────────────────────────
#  FedProx
# ──────────────────────────────────────────────────────────────────────────────

def train_fedprox(client_data, input_dim, n_rounds=20, local_epochs=5,
                  lr=0.001, mu=0.01, use_dp=False,
                  noise_multiplier=0.01, clip_norm=1.0):
    """Train a global model using FedProx (proximal penalty).

    Returns
    -------
    global_model      : FederatedMLP
    round_losses      : list[float]
    client_histories  : list[dict]
    round_val_metrics : dict   {'accuracy': [...], 'f1': [...], 'auc': [...]}
    drift_matrix      : np.ndarray  (n_rounds, n_clients)
    """
    dp_tag = " + Gaussian DP" if use_dp else ""
    print(f"\n{'='*60}")
    print(f"  FedProx (mu={mu}){dp_tag}")
    print(f"  Clients: {len(client_data)} | Rounds: {n_rounds} | "
          f"Local Epochs: {local_epochs}")
    print(f"{'='*60}")

    global_model = FederatedMLP(input_dim, lr=lr)
    round_losses  = []
    client_histories = [{'train_losses': [], 'val_losses': []}
                        for _ in range(len(client_data))]
    round_val_metrics = {'accuracy': [], 'f1': [], 'auc': []}
    drift_rows = []

    for round_idx in range(n_rounds):
        all_weights, all_sizes, round_loss = [], [], 0
        global_w = global_model.get_weights()
        round_drifts = []

        for client_idx, data in enumerate(client_data):
            local_model = FederatedMLP(input_dim, lr=lr)
            local_model.set_weights(global_w)

            for _ in range(local_epochs):
                loss = local_model.train_step_prox(
                    data['X_train'], data['y_train'],
                    global_weights=global_w, mu=mu,
                    sample_weights=data.get('sample_weights')
                )

            round_drifts.append(
                _weight_l2_drift(local_model.get_weights(), global_w)
            )

            val_pred = local_model.predict(data['X_val'])
            val_loss = local_model.compute_loss(
                val_pred.reshape(-1, 1), data['y_val']
            )
            client_histories[client_idx]['train_losses'].append(loss)
            client_histories[client_idx]['val_losses'].append(val_loss)

            all_weights.append(local_model.get_weights())
            all_sizes.append(data['n_samples'])
            round_loss += loss * data['n_samples']

        if use_dp:
            agg_w = dp_aggregate(all_weights, all_sizes,
                                 noise_multiplier=noise_multiplier,
                                 clip_norm=clip_norm, seed=round_idx)
        else:
            agg_w = fedprox_aggregate(all_weights, all_sizes)

        global_model.set_weights(agg_w)

        avg_loss = round_loss / sum(all_sizes)
        round_losses.append(avg_loss)
        drift_rows.append(round_drifts)

        acc, f1, auc_ = _round_val_metrics(global_model, client_data)
        round_val_metrics['accuracy'].append(acc)
        round_val_metrics['f1'].append(f1)
        round_val_metrics['auc'].append(auc_)

        if (round_idx + 1) % 5 == 0:
            print(f"  Round {round_idx+1:3d}/{n_rounds} | "
                  f"Loss: {avg_loss:.4f} | Val Acc: {acc:.4f} | F1: {f1:.4f}")

    return (global_model, round_losses, client_histories,
            round_val_metrics, np.array(drift_rows))


# ──────────────────────────────────────────────────────────────────────────────
#  Clustered FL (static)
# ──────────────────────────────────────────────────────────────────────────────

def train_clustered_fl(client_data, cluster_labels, input_dim,
                       n_rounds=20, local_epochs=5, lr=0.001,
                       label='Clustered FL'):
    """Train cluster-specific models using static (one-shot) clustering.

    Returns
    -------
    cluster_models    : dict {cluster_id: FederatedMLP}
    cluster_losses    : dict {cluster_id: list[float]}
    round_val_metrics : dict {'f1': [...], 'accuracy': [...]}  — global avg per round
    """
    cluster_labels = list(cluster_labels)
    unique_clusters = set(cluster_labels)
    valid_clusters  = sorted([c for c in unique_clusters if c >= 0])

    # Outlier clients (label -1 from DBSCAN) are intentionally excluded 
    # from valid_clusters. This matches the "Divergent Clients (Filtered)" 
    # block in the architecture diagram, preventing negative transfer.

    if not valid_clusters:
        valid_clusters = [0]
        cluster_labels = [0] * len(client_data)

    print(f"\n{'='*60}")
    print(f"  {label.upper()}")
    print(f"  Clusters: {len(valid_clusters)} | Rounds: {n_rounds}")
    print(f"{'='*60}")

    cluster_models = {}
    cluster_losses  = {}
    # Per-round val metrics: list-of-dicts {cluster_id: (f1, acc, n_samples)}
    per_round_cluster_metrics = []

    for cluster_id in valid_clusters:
        cluster_client_ids = [i for i, c in enumerate(cluster_labels)
                              if c == cluster_id]
        cluster_clients = [client_data[i] for i in cluster_client_ids]

        if not cluster_clients:
            continue

        print(f"\n  Cluster {cluster_id}: {len(cluster_clients)} clients")

        model  = FederatedMLP(input_dim, lr=lr)
        losses = []

        for round_idx in range(n_rounds):
            all_weights, all_sizes, round_loss = [], [], 0
            for data in cluster_clients:
                local_model = FederatedMLP(input_dim, lr=lr)
                local_model.set_weights(model.get_weights())

                for _ in range(local_epochs):
                    loss = local_model.train_step(
                        data['X_train'], data['y_train'],
                        data.get('sample_weights')
                    )

                all_weights.append(local_model.get_weights())
                all_sizes.append(data['n_samples'])
                round_loss += loss * data['n_samples']

            model.set_weights(fedavg_aggregate(all_weights, all_sizes))
            avg_loss = round_loss / sum(all_sizes)
            losses.append(avg_loss)

            # Compute weighted val metrics across this cluster's clients
            total_n, weighted_f1, weighted_acc = 0, 0.0, 0.0
            for data in cluster_clients:
                preds = model.predict(data['X_val'])
                binary = (preds >= 0.5).astype(int)
                n = len(data['y_val'])
                try:
                    wf1 = f1_score(data['y_val'], binary, zero_division=0)
                except Exception:
                    wf1 = 0.0
                wacc = accuracy_score(data['y_val'], binary)
                weighted_f1  += wf1  * n
                weighted_acc += wacc * n
                total_n      += n

            if len(per_round_cluster_metrics) <= round_idx:
                per_round_cluster_metrics.append({})
            per_round_cluster_metrics[round_idx][cluster_id] = {
                'f1':  weighted_f1  / total_n if total_n > 0 else 0,
                'acc': weighted_acc / total_n if total_n > 0 else 0,
                'n':   total_n,
            }

            if (round_idx + 1) % 5 == 0:
                print(f"    Round {round_idx+1:3d}/{n_rounds} | "
                      f"Loss: {avg_loss:.4f} | "
                      f"Val Acc: {per_round_cluster_metrics[round_idx][cluster_id]['acc']:.4f} | "
                      f"F1: {per_round_cluster_metrics[round_idx][cluster_id]['f1']:.4f}")

        cluster_models[cluster_id] = model
        cluster_losses[cluster_id] = losses

    # Aggregate per-round metrics across all clusters (weighted by cluster size)
    round_val_metrics = {'f1': [], 'accuracy': []}
    for round_metrics in per_round_cluster_metrics:
        total_n = sum(cm['n'] for cm in round_metrics.values())
        if total_n == 0:
            round_val_metrics['f1'].append(0.0)
            round_val_metrics['accuracy'].append(0.0)
        else:
            round_val_metrics['f1'].append(
                sum(cm['f1'] * cm['n'] for cm in round_metrics.values()) / total_n)
            round_val_metrics['accuracy'].append(
                sum(cm['acc'] * cm['n'] for cm in round_metrics.values()) / total_n)

    return cluster_models, cluster_losses, round_val_metrics


# ──────────────────────────────────────────────────────────────────────────────
#  Adaptive Clustered FL (re-clusters every N rounds)
# ──────────────────────────────────────────────────────────────────────────────

def train_adaptive_clustered_fl(client_data, clusterer, input_dim,
                                n_rounds=20, local_epochs=5, lr=0.001,
                                recluster_every=5, feature_names=None,
                                label='Adaptive Clustered FL'):
    """Clustered FL with periodic re-clustering as models evolve."""
    print(f"\n{'='*60}")
    print(f"  {label.upper()} (re-cluster every {recluster_every} rounds)")
    print(f"  Clients: {len(client_data)} | Rounds: {n_rounds}")
    print(f"{'='*60}")

    print(f"\n  [Round 0] Initial clustering ...")
    cluster_labels = clusterer.cluster(client_data, feature_names)
    recluster_history = [{'round': 0, 'labels': list(cluster_labels)}]

    unique_clusters = sorted(set(cluster_labels))
    cluster_models = {cid: FederatedMLP(input_dim, lr=lr)
                      for cid in unique_clusters}
    cluster_losses  = {cid: [] for cid in unique_clusters}

    for round_idx in range(n_rounds):

        if round_idx > 0 and round_idx % recluster_every == 0:
            print(f"\n  [Round {round_idx}] Re-clustering ...")
            new_labels = clusterer.cluster(client_data, feature_names)
            recluster_history.append({'round': round_idx,
                                      'labels': list(new_labels)})

            new_models = {}
            for cid in sorted(set(new_labels)):
                old_cid_counts = {}
                for i, nc in enumerate(new_labels):
                    if nc == cid:
                        old_c = cluster_labels[i]
                        old_cid_counts[old_c] = old_cid_counts.get(old_c, 0) + 1

                if old_cid_counts:
                    source_cid = max(old_cid_counts, key=old_cid_counts.get)
                    if source_cid in cluster_models:
                        new_m = FederatedMLP(input_dim, lr=lr)
                        new_m.set_weights(cluster_models[source_cid].get_weights())
                        new_models[cid] = new_m
                    else:
                        new_models[cid] = FederatedMLP(input_dim, lr=lr)
                else:
                    new_models[cid] = FederatedMLP(input_dim, lr=lr)

                if cid not in cluster_losses:
                    cluster_losses[cid] = []

            cluster_labels = new_labels
            cluster_models = new_models

        for cid in sorted(set(cluster_labels)):
            members = [i for i, c in enumerate(cluster_labels) if c == cid]
            cluster_clients = [client_data[i] for i in members]

            if not cluster_clients or cid not in cluster_models:
                continue

            model = cluster_models[cid]
            all_weights, all_sizes, round_loss = [], [], 0

            for data in cluster_clients:
                local_model = FederatedMLP(input_dim, lr=lr)
                local_model.set_weights(model.get_weights())

                for _ in range(local_epochs):
                    loss = local_model.train_step(
                        data['X_train'], data['y_train'],
                        data.get('sample_weights')
                    )

                all_weights.append(local_model.get_weights())
                all_sizes.append(data['n_samples'])
                round_loss += loss * data['n_samples']

            model.set_weights(fedavg_aggregate(all_weights, all_sizes))
            avg_loss = round_loss / sum(all_sizes)
            cluster_losses[cid].append(avg_loss)

        if (round_idx + 1) % 5 == 0:
            loss_str = ' | '.join(
                f"C{cid}:{cluster_losses[cid][-1]:.4f}"
                for cid in sorted(cluster_models)
                if cluster_losses.get(cid)
            )
            print(f"  Round {round_idx+1:3d}/{n_rounds} | {loss_str}")

    return cluster_models, cluster_losses, cluster_labels, recluster_history


# ──────────────────────────────────────────────────────────────────────────────
#  Feature Heterogeneity Simulator  (Paper 7 – LCFed Fix)
#
#  Real-world EHR data: hospitals often record DIFFERENT clinical variables.
#  e.g., Hospital A (community) may lack advanced lab panels;
#        Hospital B (ICU-specialist) may have full vital sign monitoring.
#
#  This simulates that scenario by masking one feature group per hospital,
#  then tests whether our FedCM gradient-soft-clustering can handle
#  feature-space heterogeneity ON TOP OF distributional heterogeneity.
#  (Paper 7 LCFed only handles feature heterogeneity, not both.)
#
#  Feature Groups (44 MIMIC-IV features split into 4 clinical domains):
#    Group 0 — Vitals          (indices  0–10)   11 features
#    Group 1 — Lab values      (indices 11–25)   15 features
#    Group 2 — Demographics    (indices 26–35)   10 features
#    Group 3 — Severity scores (indices 36–43)    8 features
# ──────────────────────────────────────────────────────────────────────────────

# Default feature group boundaries for MIMIC-IV 44-feature dataset
MIMIC_FEATURE_GROUPS = {
    0: list(range(0,  11)),   # Vitals          (11 features)
    1: list(range(11, 26)),   # Lab values      (15 features)
    2: list(range(26, 36)),   # Demographics    (10 features)
    3: list(range(36, 44)),   # Severity scores  (8 features)
}

MIMIC_FEATURE_GROUP_NAMES = {
    0: 'Vitals (0–10)',
    1: 'Lab Values (11–25)',
    2: 'Demographics (26–35)',
    3: 'Severity Scores (36–43)',
}


def simulate_feature_heterogeneity(
    client_data: list,
    feature_groups: dict = None,
    missing_group_per_hospital: dict = None,
    seed: int = 42,
    verbose: bool = True,
) -> tuple:
    """
    Simulate per-hospital feature heterogeneity by masking one feature
    group per hospital (zero-imputed).

    Models Paper 7 (LCFed) scenario: hospitals have different clinical
    feature sets due to equipment, recording practices, or data governance.
    Addresses LCFed's open problem: combining feature-space heterogeneity
    with distributional-space soft clustering has never been done.

    Strategy:
    ---------
    Each hospital is randomly assigned ONE feature group to "mask"
    (columns set to 0.0 — mean-imputed zero for standardised data).
    The masking is applied to BOTH X_train and X_val to simulate
    a hospital that never records those features.

    Parameters
    ----------
    client_data              : list of dicts from prepare_client_data()
                               (each dict has 'X_train', 'X_val', ...)
    feature_groups           : dict {group_id: list_of_col_indices}
                               Defaults to MIMIC_FEATURE_GROUPS (4 groups)
    missing_group_per_hospital : dict {client_idx: group_id}
                               Optional. If None, randomly assigned.
                               Set to {} for NO masking (identity transform).
    seed                     : int — random seed for group assignment
    verbose                  : bool — print masking summary

    Returns
    -------
    masked_client_data : list — copy of client_data with masked X_train/X_val
    missing_groups     : dict {client_idx: group_id} — which group was masked
    mask_info          : dict — summary of masking configuration

    Notes
    -----
    - Original client_data is NOT modified in-place; copies are returned.
    - X values for masked columns are set to 0.0 (standard z-score mean).
    - Hospital adapter (FeatureProjectionAdapter in models_torch.py) can
      then learn to project the reduced feature space back to 44 dims.
    """
    import copy

    if feature_groups is None:
        feature_groups = MIMIC_FEATURE_GROUPS

    n_clients = len(client_data)
    n_groups  = len(feature_groups)

    # Assign missing group per hospital
    if missing_group_per_hospital is None:
        rng = np.random.RandomState(seed)
        # Rotate through groups so at least one hospital per group
        base = [i % n_groups for i in range(n_clients)]
        rng.shuffle(base)
        missing_group_per_hospital = {i: base[i] for i in range(n_clients)}

    # Deep-copy client_data to avoid mutating originals
    masked_client_data = copy.deepcopy(client_data)

    mask_info = {}
    for cid, data in enumerate(masked_client_data):
        group_id = missing_group_per_hospital.get(cid, None)

        if group_id is None:
            # No masking for this hospital
            mask_info[cid] = {
                'masked_group': None,
                'masked_cols':  [],
                'group_name':   'None (full features)',
                'n_masked':     0,
            }
            continue

        cols_to_mask = feature_groups[group_id]
        group_name   = MIMIC_FEATURE_GROUP_NAMES.get(group_id, f'Group {group_id}')

        # Apply zero-masking (mean-imputed for standardised data)
        data['X_train'][:, cols_to_mask] = 0.0
        data['X_val']  [:, cols_to_mask] = 0.0

        mask_info[cid] = {
            'masked_group': group_id,
            'masked_cols':  cols_to_mask,
            'group_name':   group_name,
            'n_masked':     len(cols_to_mask),
        }

        if verbose:
            print(f"  Hospital {cid}: masked {group_name} "
                  f"({len(cols_to_mask)} features, cols {cols_to_mask[:3]}...)")

    if verbose:
        print(f"\n  Feature heterogeneity summary:")
        print(f"    Total hospitals    : {n_clients}")
        print(f"    Total feature groups: {n_groups}")
        for gid, gname in MIMIC_FEATURE_GROUP_NAMES.items():
            affected = [c for c, info in mask_info.items()
                        if info['masked_group'] == gid]
            print(f"    Group {gid} ({gname}): masked for hospitals {affected}")

    return masked_client_data, missing_group_per_hospital, mask_info


def get_available_feature_dims(
    mask_info: dict,
    total_features: int = 44,
) -> dict:
    """
    Return the number of available (non-masked) features per hospital.

    Used by FeatureProjectionAdapter to know each hospital's input_dim.

    Parameters
    ----------
    mask_info      : dict — output of simulate_feature_heterogeneity()
    total_features : int — full feature count (default: 44 for MIMIC-IV)

    Returns
    -------
    dict {client_idx: int} — available feature count per hospital
    """
    return {
        cid: total_features - info['n_masked']
        for cid, info in mask_info.items()
    }
