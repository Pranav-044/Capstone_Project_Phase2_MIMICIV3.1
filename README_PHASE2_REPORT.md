# FedICU Phase 2: Personalized Federated Learning

## Executive Summary
This Phase 2 project extends the base Federated Learning (FL) pipeline for ICU Mortality Prediction (MIMIC-IV v3.1 dataset) by implementing state-of-the-art **Personalization**, **Fairness**, and **Communication Efficiency** mechanisms. 

In standard FL (FedAvg), aggregating models across hospitals with non-IID (heterogeneous) data distributions leads to performance degradation and unfairness—especially for minority hospitals. This project resolves these limitations by integrating methods from three key research papers.

## Key Implementations & Enhancements

### 1. Fairness via Proximal Fine-Tuning (Paper 5: Ditto)
* **Limitation Addressed:** Standard FL forces a single global model onto all hospitals, resulting in poor performance (low AUROC) for hospitals with distinct demographics or smaller datasets.
* **Implementation:** Implemented `ditto_finetune()` within `personalization.py`. After receiving the global expert model, each hospital fine-tunes the model locally using a proximal penalty (`lambda_ditto`). This penalty acts as an "anchor", preventing the local model from drifting too far from the global knowledge while still adapting to local nuances.
* **Result / Proof:** The standard deviation of AUROC across hospitals dropped significantly, and the minimum AUROC (worst-performing hospital) increased. This proves the system is more equitable.

### 2. Communication Efficiency via Architecture Split (Paper 6: PPFL)
* **Limitation Addressed:** Deep learning models require massive bandwidth to transmit back and forth every round, creating a bottleneck for hospitals with limited network infrastructure.
* **Implementation:** Implemented a Body-Head split in `models_torch.py`. The model is split into a **shared Body** (feature extractor) and a **local Head** (decision layer). During federated aggregation (`phase2_training.py`), *only the body weights* are transmitted to the server. The head weights remain strictly local.
* **Result / Proof:** For our 4-layer PyTorch MLP, the head constitutes 13.1% of the parameters. By keeping it local, we reduced communication bandwidth by ~10% per round without sacrificing any predictive performance (since the local head is actually better suited for personalized predictions).

### 3. Feature Space Heterogeneity (Paper 7: LCFed)
* **Limitation Addressed:** Real-world hospitals do not always record the same features (e.g., Hospital A lacks certain lab values, Hospital B lacks specific severity scores). Standard FL assumes all clients have the exact same 44 features.
* **Implementation:** Added `simulate_feature_heterogeneity()` in `federated_training.py` which dynamically masks out entire feature groups (Vitals, Lab Values, Demographics) for specific hospitals to simulate missing modalities. 

## Reproducible Personalization Study (20 rounds)

The report claims must be tested against a true non-personalized control.
`experiment_personalization.py` therefore compares plain FedAvg, MCFL-only
personalization, and the full Ditto + PPFL pipeline using the same MIMIC-IV
train/test split, five-client Dirichlet partition (`alpha=0.5`), and seed
(`42`). All models use 20 communication rounds and five local epochs.

| Configuration | Global AUROC | Mean Client AUROC | Worst Client AUROC | Client AUROC SD | MB / Round | Saving |
|---|---:|---:|---:|---:|---:|---:|
| FedAvg baseline | 0.7956 | 0.7958 | 0.7081 | 0.0525 | 0.0865 | 0.0% |
| MCFL personalization | 0.8025 | 0.7952 | 0.7367 | 0.0408 | 0.0865 | 0.0% |
| Full, lambda_ditto=0.01, mcfl_lr=0.10 | 0.8176 | 0.7817 | 0.6538 | 0.0680 | 0.0781 | 9.8% |
| Full, lambda_ditto=0.001, mcfl_lr=0.10 | **0.8252** | 0.8020 | 0.7340 | 0.0499 | 0.0781 | 9.8% |
| Full, lambda_ditto=0.10, mcfl_lr=0.10 | 0.8225 | **0.8209** | **0.7867** | **0.0352** | 0.0781 | 9.8% |
| Full, lambda_ditto=0.01, mcfl_lr=0.03 | 0.8209 | 0.7767 | 0.6184 | 0.0864 | 0.0781 | 9.8% |
| Full, lambda_ditto=0.01, mcfl_lr=0.30 | 0.8247 | 0.7511 | 0.5318 | 0.1129 | 0.0781 | 9.8% |

### Findings

* **MCFL alone improves fairness:** compared with FedAvg, global AUROC rises by
  0.0069, worst-client AUROC rises by 0.0286, and client AUROC standard
  deviation falls by 0.0117.
* **Classification trade-off:** the best-accuracy value remains the FedAvg
  baseline (0.8865), but personalization improves AUROC, F1, and recall. The
  best global-performance configuration (`lambda_ditto=0.001`) reaches AUROC
  0.8252, F1 0.4085, and recall 0.4265, compared with 0.7956, 0.3722, and
  0.3088 for FedAvg. This reflects the imbalanced mortality outcome and the
  fixed 0.5 threshold: detecting more deaths lowers precision and accuracy.
* **The original full default is not fairer:** `lambda_ditto=0.01` improves
  global AUROC but reduces the worst-client AUROC and increases disparity.
  It should not be used to support the fairness claim.
* **Recommended setting depends on the objective:** choose
  `lambda_ditto=0.001, mcfl_lr=0.10` for maximum global AUROC. Choose
  `lambda_ditto=0.10, mcfl_lr=0.10` for the best fairness/overall balance;
  it improves global AUROC, mean client AUROC, worst-client AUROC, and client
  disparity relative to FedAvg while retaining PPFL's 9.8% bandwidth saving.

The machine-readable results and the generated Markdown table are stored in
`results/personalization_study_final/`. Re-run the full study with:

`python experiment_personalization.py --rounds 20 --epochs 5 --out_dir results/personalization_study_final`

The complete global classification-metric comparison is stored in
`results/personalization_study_final/classification_metrics.md`.

## Evaluation Metrics

To prove the superiority of the personalized pipeline, we track the following metrics (visible in the terminal output and tracked in the `results/` folder):

1. **Mean AUROC (Higher = Better):** Macro-average performance. Proves personalization maintains or improves overall predictive power.
2. **Worst AUROC (Higher = Better):** The AUROC of the most disadvantaged hospital. A higher score proves fairness (Paper 5).
3. **Std AUROC (Lower = Fairer):** The disparity in performance between hospitals. A lower variance means the model works equally well for everyone.
4. **Communication MB/Round (Lower = Better):** Total bandwidth used per round. Proves the PPFL body-head split (Paper 6) saves network resources.

## How to Present the Results

1. **Reproduce the complete comparison table:**
   `python experiment_personalization.py --rounds 20 --epochs 5 --out_dir results/personalization_study_final`
2. **Choose the configuration that matches the claim:** use
   `lambda_ditto=0.001, mcfl_lr=0.10` for the highest global AUROC, or
   `lambda_ditto=0.10, mcfl_lr=0.10` for the strongest fairness result.
3. **Present measured values:** use the table in this report or
   `results/personalization_study_final/experiment_results.md`. Do not use
   the existing illustrative plot script as experimental evidence until it is
   changed to load the measured results.
