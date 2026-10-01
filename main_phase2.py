"""
main_phase2.py  —  Phase 2 CLI Entry Point (PyTorch)
=====================================================
Runs the combined FedCM + FedSPD + MCFL pipeline.
Uses MIMIC-IV v3.1 as the primary dataset.

Usage
-----
# MIMIC-IV v3.1 (default)
python main_phase2.py --seed 42 --use_smote --rounds 20 --verbose

# Note: Before running for the first time, extract the features:
# python mimic_bigquery_extractor.py --project YOUR_GCP_PROJECT_ID
"""

import argparse
import json
import os
import sys
import numpy as np
import torch

# ── Path setup ────────────────────────────────────────────────────────────────
PHASE1_DIR = os.path.join(os.path.dirname(__file__), '..', 'fl_physionet_project')
sys.path.insert(0, PHASE1_DIR)

from federated_training import partition_dirichlet, prepare_client_data
from phase2_training import run_phase2


# ─────────────────────────────────────────────────────────────────────────────
#  Data Loading (supports PhysioNet and MIMIC-IV)
# ─────────────────────────────────────────────────────────────────────────────

def load_data(data_path: str, target_col: str, seed: int = 42):
    """
    Load preprocessed CSV data (MIMIC-IV v3.1).
    Returns X_train, X_test, y_train, y_test (numpy arrays).
    """
    import pandas as pd
    from sklearn.model_selection import train_test_split

    if not os.path.exists(data_path):
        raise FileNotFoundError(
            f"Data file not found: {data_path}\n"
            "  For MIMIC-IV: run `python mimic_bigquery_extractor.py --project YOUR_GCP_ID` first."
        )

    df = pd.read_csv(data_path)

    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' not in data. "
                         f"Available: {list(df.columns[:10])} ...")

    # Drop non-feature columns
    drop_cols = ['stay_id', 'subject_id', 'hadm_id', 'intime',
                 'outtime', 'rn', 'discharge_location']
    drop_cols = [c for c in drop_cols if c in df.columns]
    feature_cols = [c for c in df.columns
                    if c != target_col and c not in drop_cols]

    X = df[feature_cols].values.astype(np.float32)
    y = df[target_col].values.astype(np.float32)

    print(f"  Loaded: {X.shape[0]} patients, {X.shape[1]} features")
    print(f"  Mortality rate: {y.mean():.3f}")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, stratify=y, random_state=seed)

    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train).astype(np.float32)
    X_test  = scaler.transform(X_test).astype(np.float32)

    return X_train, X_test, y_train, y_test


# ─────────────────────────────────────────────────────────────────────────────
#  Results Saving
# ─────────────────────────────────────────────────────────────────────────────

def save_results(results: dict, out_dir: str):
    os.makedirs(out_dir, exist_ok=True)

    # Round metrics
    with open(os.path.join(out_dir, 'round_metrics.json'), 'w') as f:
        # Make serialisable (convert numpy types)
        metrics = []
        for rm in results['round_metrics']:
            m = {k: (v.tolist() if hasattr(v, 'tolist') else v)
                 for k, v in rm.items()}
            metrics.append(m)
        json.dump(metrics, f, indent=2)

    # Alpha history
    alpha_ser = [
        {str(k): v.tolist() for k, v in r.items()}
        for r in results['alpha_history']
    ]
    with open(os.path.join(out_dir, 'alpha_history.json'), 'w') as f:
        json.dump(alpha_ser, f, indent=2)

    # Summary
    final = results['personalized_metrics']['mean_personal_global']
    summary = {
        'final_global_auc':      final['auc'],
        'final_global_f1':       final['f1'],
        'final_global_accuracy': final['accuracy'],
        'final_avg_local_auc':   results['personalized_metrics']['avg_local_auc'],
    }
    with open(os.path.join(out_dir, 'summary.json'), 'w') as f:
        json.dump(summary, f, indent=2)

    print(f"\n  Results saved to: {out_dir}/")


# ─────────────────────────────────────────────────────────────────────────────
#  CLI
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description='FedICU Phase 2 — PyTorch Gradient-Soft-CFL + MCFL')

    # Data
    mimic_default = os.path.join(
        os.path.dirname(__file__), 'mimic_iv_phase2_features.csv')
    p.add_argument('--data_path', type=str, default=mimic_default,
                   help='Path to preprocessed MIMIC-IV CSV')
    p.add_argument('--target', type=str, default='in_hospital_death',
                   help='Target column name')

    # FL settings
    p.add_argument('--seed',       type=int,   default=42)
    p.add_argument('--rounds',     type=int,   default=20)
    p.add_argument('--epochs',     type=int,   default=5)
    p.add_argument('--clients',    type=int,   default=5)
    p.add_argument('--batch_size', type=int,   default=32)
    p.add_argument('--lr',         type=float, default=0.001)
    p.add_argument('--alpha_dir',  type=float, default=0.5,
                   help='Dirichlet alpha for Non-IID partitioning')
    p.add_argument('--use_smote',  action='store_true')
    p.add_argument('--use_fedprox',action='store_true')
    p.add_argument('--mu',         type=float, default=0.01)

    # Phase 2 hyperparameters
    p.add_argument('--temperature',type=float, default=1.0,
                   help='Softmax temperature for FedSPD soft membership')
    p.add_argument('--tau',        type=float, default=0.50,
                   help='FedCM drift detection threshold')
    p.add_argument('--mcfl_iter',  type=int,   default=60,
                   help='MCFL α optimization iterations')
    p.add_argument('--mcfl_lr',    type=float, default=0.10,
                   help='MCFL α learning rate')

    p.add_argument('--verbose',    action='store_true')
    p.add_argument('--out_dir',    type=str,
                   default=os.path.join(os.path.dirname(__file__), 'results'))
    return p.parse_args()


def main():
    args = parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    print("\n" + "=" * 60)
    print("  FedICU Phase 2  |  PyTorch  |  Gradient-Soft-CFL + MCFL")
    print("=" * 60)
    for k, v in vars(args).items():
        print(f"  {k:<20}: {v}")
    print("=" * 60 + "\n")

    # Load
    print("[1/4] Loading data ...")
    X_train, X_test, y_train, y_test = load_data(
        args.data_path, args.target, seed=args.seed)
    input_dim = X_train.shape[1]
    print(f"  Train: {len(X_train)}  Test: {len(X_test)}")

    # Partition
    print(f"\n[2/4] Partitioning into {args.clients} Non-IID clients "
          f"(alpha={args.alpha_dir}) ...")
    client_indices = partition_dirichlet(
        y_train, args.clients, alpha=args.alpha_dir, seed=args.seed)
    client_data = prepare_client_data(
        X_train, y_train, client_indices,
        test_size=0.20, seed=args.seed, use_smote=args.use_smote)
    for i, cd in enumerate(client_data):
        print(f"  Client {i}: n={cd['n_samples']} | "
              f"mortality={cd['mortality_rate']:.3f}")

    # Train
    print(f"\n[3/4] Running Phase 2 ({args.rounds} rounds) ...")
    results = run_phase2(
        client_data=client_data,
        X_test=X_test,
        y_test=y_test,
        n_rounds=args.rounds,
        n_epochs=args.epochs,
        batch_size=args.batch_size,
        input_dim=input_dim,
        lr=args.lr,
        use_fedprox=args.use_fedprox,
        mu=args.mu,
        soft_temperature=args.temperature,
        drift_tau=args.tau,
        mcfl_n_iter=args.mcfl_iter,
        mcfl_lr_alpha=args.mcfl_lr,
        verbose=args.verbose,
        seed=args.seed,
    )

    # Save
    print(f"\n[4/4] Saving results ...")
    save_results(results, args.out_dir)

    # Print summary
    print("\n" + "=" * 60)
    print("  FINAL RESULTS")
    print("=" * 60)
    final = results['personalized_metrics']['mean_personal_global']
    print(f"  {'Method':<40} {'AUC':>7} {'F1':>7} {'Acc':>7}")
    print(f"  {'-'*61}")
    print(f"  {'Gradient-Soft-CFL + MCFL (Phase 2 Ours)':<40} "
          f"{final['auc']:>7.4f} "
          f"{final['f1']:>7.4f} "
          f"{final['accuracy']:>7.4f}")
    print(f"\n  (Run Phase 1 main.py for FedAvg/DBSCAN/Agglom baselines)")
    print("=" * 60)


if __name__ == '__main__':
    main()
