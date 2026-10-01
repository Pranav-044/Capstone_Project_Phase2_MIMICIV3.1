# Phase 2 — Combined Implementation Plan
## Unified: FedCM + FedSPD + MCFL → Single Three-Stage Pipeline

---

## APPROACH: METHOD C (COMBINED)

```
Stage 1 (FedCM)  → Gradient-path cosine similarity between hospitals
Stage 2 (FedSPD) → Similarity → soft membership probability vectors π
Stage 3 (MCFL)   → Cluster models → experts; hospital learns mixture weights α
```

---

## PIPELINE (per communication round)

```
Local Training
  ↓ saves gradient at every mini-batch step
Gradient Path Matrix per hospital: shape (n_steps, n_params)
  ↓ cosine similarity between all hospital pairs
K×K Similarity Matrix
  ↓ softmax row-wise
Soft Membership Matrix π  [K hospitals × K clusters]
  ↓ weighted aggregation
K Expert Models (one per cluster)
  ↓ each hospital optimizes α over experts using local loss
Personalized Model per hospital
  ↓ used as warm start for next round
```

---

## FILES TO CREATE (inside fl_physionet_project_phase2/)

| File | Purpose |
|---|---|
| `gradient_clustering.py` | FedCM: collect gradient paths + cosine similarity matrix |
| `soft_membership.py` | FedSPD: similarity → soft π vectors + soft aggregation |
| `personalization.py` | MCFL: optimize mixture weights α + build personalized model |
| `phase2_training.py` | Combined pipeline: runs all 3 stages per round |
| `evaluate_phase2.py` | Evaluate personalized models on MIMIC-IV global test set |
| `main_phase2.py` | CLI entry point |

---

## PHASE 1 FILES (READ-ONLY, imported from parent folder)

Do NOT edit these — import them:
- `../fl_physionet_project/models.py` — FederatedMLP class
- `../fl_physionet_project/preprocessing.py` — feature extraction
- `../fl_physionet_project/federated_training.py` — Dirichlet partitioning, SMOTE
- `../fl_physionet_project/data_loader.py` — MIMIC-IV data

---

## COMPARISON TABLE (final output)

| Method | Clustering | Membership | Personalized? |
|---|---|---|---|
| FedAvg | None | N/A | No |
| DBSCAN-CFL | Weight Euclidean | Hard | No |
| Agglom-CFL | Weight Euclidean | Hard | No |
| **Gradient-Soft-CFL (Ours)** | Gradient Cosine | **Soft** | **Yes (MCFL)** |
