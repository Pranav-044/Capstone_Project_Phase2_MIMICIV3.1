"""
plot_results.py  —  Visualize Phase 2 FL Results
==================================================
Run after main_phase2.py completes:
    python plot_results.py

Generates:
  1. AUC per round (Global + Avg Local)
  2. Number of clusters per round
  3. Client alpha mixture weights (final round)
  4. Cosine similarity heatmap (final round)
"""

import json
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')   # no display needed — saves to file
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

RESULTS_DIR = os.path.join(os.path.dirname(__file__), 'results')
PLOTS_DIR   = os.path.join(RESULTS_DIR, 'plots')
os.makedirs(PLOTS_DIR, exist_ok=True)


def load_results():
    with open(os.path.join(RESULTS_DIR, 'round_metrics.json'))  as f:
        round_metrics = json.load(f)
    with open(os.path.join(RESULTS_DIR, 'alpha_history.json'))  as f:
        alpha_history = json.load(f)
    with open(os.path.join(RESULTS_DIR, 'summary.json'))        as f:
        summary = json.load(f)
    return round_metrics, alpha_history, summary


def plot_auc_per_round(round_metrics):
    rounds      = [r['round']         for r in round_metrics]
    global_auc  = [r['global_auc']    for r in round_metrics]
    local_auc   = [r['avg_local_auc'] for r in round_metrics]

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(rounds, global_auc, 'b-o', linewidth=2, markersize=5,
            label='Global Test AUC')
    ax.plot(rounds, local_auc,  'g-s', linewidth=2, markersize=5,
            label='Avg Local Personalized AUC')
    ax.set_xlabel('Federated Round', fontsize=12)
    ax.set_ylabel('AUC (ROC)', fontsize=12)
    ax.set_title('FedICU Phase 2  —  AUC per Federated Round\n'
                 'FedCM + FedSPD + MCFL on MIMIC-IV v3.1', fontsize=13)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0.5, 1.0)
    ax.set_xticks(rounds)

    # Annotate final values
    ax.annotate(f"{global_auc[-1]:.4f}",
                xy=(rounds[-1], global_auc[-1]),
                xytext=(-30, 10), textcoords='offset points',
                fontsize=10, color='blue',
                arrowprops=dict(arrowstyle='->', color='blue'))
    ax.annotate(f"{local_auc[-1]:.4f}",
                xy=(rounds[-1], local_auc[-1]),
                xytext=(-30, -20), textcoords='offset points',
                fontsize=10, color='green',
                arrowprops=dict(arrowstyle='->', color='green'))

    plt.tight_layout()
    path = os.path.join(PLOTS_DIR, '1_auc_per_round.png')
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  Saved: {path}")


def plot_clusters_per_round(round_metrics):
    rounds   = [r['round']      for r in round_metrics]
    clusters = [r['n_clusters'] for r in round_metrics]

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.bar(rounds, clusters, color='steelblue', alpha=0.8, edgecolor='navy')
    ax.set_xlabel('Federated Round', fontsize=12)
    ax.set_ylabel('Number of Expert Clusters', fontsize=12)
    ax.set_title('FedSPD Dynamic Cluster Detection per Round\n'
                 'MIMIC-IV v3.1 — 5 Hospital Clients', fontsize=13)
    ax.set_xticks(rounds)
    ax.set_yticks(range(1, max(clusters) + 2))
    ax.grid(True, axis='y', alpha=0.3)
    plt.tight_layout()
    path = os.path.join(PLOTS_DIR, '2_clusters_per_round.png')
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  Saved: {path}")


def plot_alpha_weights(alpha_history):
    """Plot final round mixture weights per client."""
    final_alpha = alpha_history[-1]
    client_ids  = sorted(final_alpha.keys(), key=lambda x: int(x))
    n_clients   = len(client_ids)
    n_experts   = len(final_alpha[client_ids[0]])

    data = np.array([final_alpha[c] for c in client_ids])

    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(n_clients)
    width = 0.8 / n_experts
    colors = plt.cm.Set2(np.linspace(0, 1, n_experts))

    for j in range(n_experts):
        ax.bar(x + j * width, data[:, j], width,
               label=f'Expert {j+1}', color=colors[j], alpha=0.85,
               edgecolor='grey')

    ax.set_xlabel('Hospital Client', fontsize=12)
    ax.set_ylabel('Mixture Weight (alpha)', fontsize=12)
    ax.set_title('MCFL Personalized Mixture Weights per Hospital Client\n'
                 '(Final Round — MIMIC-IV v3.1)', fontsize=13)
    ax.set_xticks(x + width * (n_experts - 1) / 2)
    ax.set_xticklabels([f'Client {c}' for c in client_ids])
    ax.legend(title='Expert Model', fontsize=10)
    ax.set_ylim(0, 1)
    ax.grid(True, axis='y', alpha=0.3)
    plt.tight_layout()
    path = os.path.join(PLOTS_DIR, '3_alpha_weights_final_round.png')
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  Saved: {path}")


def plot_summary_table(summary, round_metrics):
    fig, ax = plt.subplots(figsize=(9, 3))
    ax.axis('off')

    rows = [
        ['Metric',                          'Value'],
        ['Dataset',                         'MIMIC-IV v3.1 (18,725 ICU patients)'],
        ['FL Method',                       'FedCM + FedSPD + MCFL (Papers 1-3)'],
        ['Hospital Clients',                '5 (Non-IID Dirichlet alpha=0.5)'],
        ['Federated Rounds',                str(len(round_metrics))],
        ['Final Global Test AUC',           f"{summary['final_global_auc']:.4f}"],
        ['Final Avg Local Personalized AUC',f"{summary['final_avg_local_auc']:.4f}"],
        ['Final Global F1 Score',           f"{summary['final_global_f1']:.4f}"],
        ['Final Global Accuracy',           f"{summary['final_global_accuracy']:.4f}"],
    ]

    table = ax.table(
        cellText=rows[1:],
        colLabels=rows[0],
        cellLoc='left',
        loc='center',
        colWidths=[0.45, 0.55],
    )
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 1.6)

    # Style header
    for j in range(2):
        table[0, j].set_facecolor('#2c5f8a')
        table[0, j].set_text_props(color='white', fontweight='bold')

    # Highlight AUC row
    for j in range(2):
        table[5, j].set_facecolor('#d4edda')

    ax.set_title('FedICU Phase 2 — Final Results Summary',
                 fontsize=13, fontweight='bold', pad=20)
    plt.tight_layout()
    path = os.path.join(PLOTS_DIR, '4_results_summary.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {path}")


def main():
    print("\n  Loading results ...")
    round_metrics, alpha_history, summary = load_results()
    n_rounds = len(round_metrics)
    print(f"  Found {n_rounds} rounds of results")

    print("\n  Generating plots ...")
    plot_auc_per_round(round_metrics)
    plot_clusters_per_round(round_metrics)
    plot_alpha_weights(alpha_history)
    plot_summary_table(summary, round_metrics)

    print(f"\n  All plots saved to: {PLOTS_DIR}")
    print("\n  Final Results:")
    print(f"    Global Test AUC   : {summary['final_global_auc']:.4f}")
    print(f"    Avg Local AUC     : {summary['final_avg_local_auc']:.4f}")
    print(f"    Global F1 Score   : {summary['final_global_f1']:.4f}")
    print(f"    Global Accuracy   : {summary['final_global_accuracy']:.4f}")


if __name__ == '__main__':
    main()
