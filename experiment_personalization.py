"""Reproducible Phase 2 ablation and hyperparameter experiments.

Compares plain FedAvg against the Phase 2 MCFL personalization pipeline and
its Ditto + PPFL extension.  Every configuration uses the same train/test
split and client partition for a given seed, and is saved in its own folder.
"""

import argparse
import json
from pathlib import Path
from time import perf_counter

import numpy as np
import torch

from evaluate_fairness import compute_fairness_metrics, comm_reduction_ratio
from federated_training import partition_dirichlet, prepare_client_data
from main_phase2 import load_data
from models_torch import (
    FederatedMLPTorch,
    fedavg_aggregate,
    get_flat_weights,
    set_flat_weights,
    train_local,
)
from phase2_training import run_phase2
from utils import evaluate_flat


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run reproducible personalization ablations and tuning."
    )
    parser.add_argument("--rounds", type=int, default=20)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out_dir", type=Path, default=Path("results/experiments"))
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Use the smaller 10-round, 2-epoch screening configuration.",
    )
    return parser.parse_args()


def make_client_data(seed):
    data_path = Path(__file__).with_name("mimic_iv_phase2_features.csv")
    X_train, X_test, y_train, y_test = load_data(
        str(data_path), "in_hospital_death", seed=seed
    )
    client_indices = partition_dirichlet(
        y_train, n_clients=5, alpha=0.5, seed=seed
    )
    client_data = prepare_client_data(
        X_train,
        y_train,
        client_indices,
        test_size=0.20,
        seed=seed,
        use_smote=False,
    )
    return client_data, X_test, y_test, X_train.shape[1]


def run_fedavg(client_data, X_test, y_test, input_dim, rounds, epochs, seed):
    """Train a plain FedAvg model and evaluate it on every client validation set."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    global_model = FederatedMLPTorch(input_dim=input_dim)
    global_weights = get_flat_weights(global_model)
    client_models = {
        client_id: FederatedMLPTorch(input_dim=input_dim)
        for client_id in range(len(client_data))
    }

    for _ in range(rounds):
        local_weights = {}
        client_sizes = {}
        for client_id, data in enumerate(client_data):
            set_flat_weights(client_models[client_id], global_weights)
            local_weights[client_id], _ = train_local(
                client_models[client_id],
                data["X_train"],
                data["y_train"],
                n_epochs=epochs,
                batch_size=32,
                lr=0.001,
                sample_weights=data.get("sample_weights"),
                collect_gradients=False,
            )
            client_sizes[client_id] = data["n_samples"]
        global_weights = fedavg_aggregate(local_weights, client_sizes)

    global_metrics = evaluate_flat(global_weights, X_test, y_test, input_dim=input_dim)
    per_client_auc = {
        client_id: evaluate_flat(
            global_weights,
            data["X_val"],
            data["y_val"],
            input_dim=input_dim,
        )["auc"]
        for client_id, data in enumerate(client_data)
    }
    return global_metrics, per_client_auc


def record_result(name, configuration, global_metrics, per_client_auc, input_dim, elapsed):
    model = FederatedMLPTorch(input_dim=input_dim)
    communication = comm_reduction_ratio(model)
    is_ppfl = configuration.get("mode") == "full"
    fairness = compute_fairness_metrics(per_client_auc)
    return {
        "name": name.replace(chr(206) + chr(187), "lambda"),
        "configuration": configuration,
        "global_auc": global_metrics["auc"],
        "global_f1": global_metrics["f1"],
        "global_accuracy": global_metrics["accuracy"],
        "global_recall": global_metrics["recall"],
        "global_precision": global_metrics["precision"],
        "mean_client_auc": fairness["mean_auroc"],
        "std_client_auc": fairness["std_auroc"],
        "worst_client_auc": fairness["worst_client_auroc"],
        "range_client_auc": fairness["range_auroc"],
        "communication_mb_per_round": (
            communication["body_only_mb"] if is_ppfl else communication["full_model_mb"]
        ),
        "communication_savings_pct": communication["savings_pct"] if is_ppfl else 0.0,
        "per_client_auc": per_client_auc,
        "elapsed_seconds": elapsed,
    }


def run_phase2_configuration(client_data, X_test, y_test, input_dim, config, rounds, epochs, seed):
    result = run_phase2(
        client_data=client_data,
        X_test=X_test,
        y_test=y_test,
        n_rounds=rounds,
        n_epochs=epochs,
        batch_size=32,
        input_dim=input_dim,
        lr=0.001,
        mcfl_n_iter=60,
        mcfl_lr_alpha=config["mcfl_lr"],
        verbose=False,
        seed=seed,
        mode=config["mode"],
        lambda_ditto=config.get("lambda_ditto", 0.0),
        ppfl_split=config.get("ppfl_split", False),
    )
    per_client_auc = {
        client_id: metrics["auc"]
        for client_id, metrics in result["personalized_metrics"]["per_client"].items()
    }
    return result["personalized_metrics"]["mean_personal_global"], per_client_auc


def write_table(records, output_path):
    headers = [
        "Experiment", "Global AUC", "Mean client AUC", "Worst client AUC",
        "Client AUC SD", "Comm. MB/round", "Savings", "Time (s)",
    ]
    lines = ["| " + " | ".join(headers) + " |", "|" + "|".join(["---"] * len(headers)) + "|"]
    for record in records:
        lines.append(
            "| {name} | {global_auc:.4f} | {mean_client_auc:.4f} | "
            "{worst_client_auc:.4f} | {std_client_auc:.4f} | "
            "{communication_mb_per_round:.4f} | {communication_savings_pct:.1f}% | "
            "{elapsed_seconds:.1f} |".format(**record)
        )
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    args = parse_args()
    import sys
    sys.stdout.reconfigure(errors="replace")
    rounds, epochs = (10, 2) if args.quick else (args.rounds, args.epochs)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    client_data, X_test, y_test, input_dim = make_client_data(args.seed)

    configurations = [
        ("FedAvg baseline", {"mode": "fedavg"}),
        ("MCFL personalization", {"mode": "standard", "mcfl_lr": 0.10}),
        ("Full personalization (default)", {"mode": "full", "lambda_ditto": 0.01, "mcfl_lr": 0.10, "ppfl_split": True}),
        ("Full: Ditto λ=0.001", {"mode": "full", "lambda_ditto": 0.001, "mcfl_lr": 0.10, "ppfl_split": True}),
        ("Full: Ditto λ=0.1", {"mode": "full", "lambda_ditto": 0.1, "mcfl_lr": 0.10, "ppfl_split": True}),
        ("Full: MCFL lr=0.03", {"mode": "full", "lambda_ditto": 0.01, "mcfl_lr": 0.03, "ppfl_split": True}),
        ("Full: MCFL lr=0.30", {"mode": "full", "lambda_ditto": 0.01, "mcfl_lr": 0.30, "ppfl_split": True}),
    ]

    records = []
    for name, config in configurations:
        print(
            f"\nRunning {name} ({rounds} rounds, {epochs} local epochs) ...",
            flush=True,
        )
        started = perf_counter()
        if config["mode"] == "fedavg":
            global_metrics, per_client_auc = run_fedavg(
                client_data, X_test, y_test, input_dim, rounds, epochs, args.seed
            )
        else:
            global_metrics, per_client_auc = run_phase2_configuration(
                client_data, X_test, y_test, input_dim, config, rounds, epochs, args.seed
            )
        elapsed = perf_counter() - started
        records.append(record_result(
            name, config, global_metrics, per_client_auc, input_dim, elapsed
        ))
        (args.out_dir / "experiment_results.json").write_text(
            json.dumps(records, indent=2), encoding="utf-8"
        )
        write_table(records, args.out_dir / "experiment_results.md")

    (args.out_dir / "experiment_results.json").write_text(
        json.dumps(records, indent=2), encoding="utf-8"
    )
    write_table(records, args.out_dir / "experiment_results.md")
    print(f"\nSaved results to {args.out_dir}")
    print((args.out_dir / "experiment_results.md").read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
