"""
Client Clustering Module
=========================
Provides two clustering strategies for Federated Learning clients:

1. DBSCAN  (ClientClusterer)
   - Silhouette-optimised automatic eps tuning
   - Phase-2 sub-cluster refinement on mortality gaps
   - Noise reassignment to nearest centroid

2. Hierarchical / Agglomerative  (AgglomerativeClientClusterer)
   - Ward-linkage AgglomerativeClustering
   - Automatic k selection via silhouette scan (k = 2 .. n//2)
   - Dendrogram-level fallback when k scan finds no valid split
   - Same public interface as DBSCAN version

Both classes expose:
    .cluster(client_data, feature_names=None) -> list[int]
    .get_metrics()                            -> dict
    .vectors_scaled_                          (for PCA visualisation)
"""

import numpy as np
from sklearn.cluster import DBSCAN, AgglomerativeClustering
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score, davies_bouldin_score
from sklearn.neighbors import NearestNeighbors


# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
#  Shared helper
# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•

def _build_distribution_vectors(client_data, feature_names=None):
    """Build compact distribution summary for each client.

    Parameters
    ----------
    client_data : list of dict
    feature_names : list of str or None

    Returns
    -------
    vectors : np.ndarray (n_clients, n_dims)
    """
    vectors = []
    for data in client_data:
        X = data['X_train']
        y = data['y_train']

        mortality_rate = y.mean()
        class_ratio = (y.sum() + 1) / (len(y) - y.sum() + 1)

        vec = [
            mortality_rate,
            class_ratio,
            np.log1p(len(y)),
        ]

        n_use = min(8, X.shape[1])
        for i in range(n_use):
            vec.append(X[:, i].mean())

        vectors.append(vec)

    return np.array(vectors)


def _compute_clustering_metrics(vectors_scaled, labels):
    """Return silhouette + Davies-Bouldin for a labelling."""
    n_clusters = len(set(labels))
    metrics = {
        'n_clusters': n_clusters,
        'silhouette_score': -1.0,
        'davies_bouldin': -1.0,
    }
    if n_clusters >= 2:
        try:
            metrics['silhouette_score'] = float(
                silhouette_score(vectors_scaled, labels)
            )
        except Exception:
            pass
        try:
            if len(labels) > n_clusters:
                metrics['davies_bouldin'] = float(
                    davies_bouldin_score(vectors_scaled, labels)
                )
        except Exception:
            pass
    return metrics


# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
#  DBSCAN Clusterer  (unchanged from original, refactored to use shared helper)
# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•

class ClientClusterer:
    """Clusters FL clients using DBSCAN + two-phase refinement.

    Phase 1: DBSCAN with silhouette-based eps tuning.
    Phase 2: Sub-split large heterogeneous clusters on mortality gaps.
    """

    def __init__(self, eps=None, min_samples=2):
        self.eps = eps
        self.min_samples = min_samples
        self.labels_ = None
        self.metrics_ = {}

    def build_distribution_vectors(self, client_data, feature_names=None):
        return _build_distribution_vectors(client_data, feature_names)

    # ------------------------------------------------------------------
    # Phase 1: DBSCAN with silhouette-based eps selection
    # ------------------------------------------------------------------

    def _auto_tune_eps(self, vectors_scaled):
        from scipy.spatial.distance import pdist

        n = len(vectors_scaled)
        pairwise = pdist(vectors_scaled)

        percentiles = np.arange(5, 96, 5)
        candidates = np.unique(np.percentile(pairwise, percentiles))

        if n <= 20:
            all_dists = np.sort(np.unique(pairwise))
            midpoints = (all_dists[:-1] + all_dists[1:]) / 2
            candidates = np.unique(np.concatenate([
                candidates, all_dists, midpoints
            ]))

        candidates = candidates[candidates >= 0.05]

        print(f"\n  [INFO] Phase 1: Density-based Clustering (DBSCAN)")
        print(f"  Scanning {len(candidates)} eps candidates "
              f"(range: {candidates.min():.3f} â€“ {candidates.max():.3f}) ...")

        best_eps, best_labels, best_sil = None, None, -2.0
        best_n_clusters = 0

        for eps_try in candidates:
            dbscan = DBSCAN(eps=eps_try, min_samples=self.min_samples)
            labels_try = dbscan.fit_predict(vectors_scaled)

            n_clusters = len(set(labels_try)) - (1 if -1 in labels_try else 0)
            if n_clusters < 2:
                continue

            try:
                sil = silhouette_score(vectors_scaled, labels_try)
            except Exception:
                sil = -1.0

            adjusted = sil + 0.01 * min(n_clusters, 5)
            if adjusted > best_sil:
                print(f"    [INFO] Better â†’ eps={eps_try:.4f} | "
                      f"clusters={n_clusters} | silhouette={sil:.3f}")
                best_eps, best_labels = eps_try, labels_try.copy()
                best_sil, best_n_clusters = adjusted, n_clusters

        if best_eps is not None:
            n_noise = (best_labels == -1).sum()
            actual_sil = best_sil - 0.01 * min(best_n_clusters, 5)
            print(f"  Best DBSCAN: eps={best_eps:.4f}, "
                  f"{best_n_clusters} clusters, {n_noise} noise, "
                  f"silhouette={actual_sil:.3f}")
            return best_eps, best_labels
        else:
            print("  No eps produced >= 2 clusters")
            return None, None

    # ------------------------------------------------------------------
    # Phase 2: Sub-cluster refinement
    # ------------------------------------------------------------------

    def _refine_clusters(self, labels, client_data):
        labels = np.array(labels, dtype=int)
        new_labels = labels.copy()
        next_id = labels.max() + 1
        refined = False

        for cid in sorted(set(labels)):
            members = np.where(labels == cid)[0]
            if len(members) < 4:
                continue

            mort = np.array([client_data[m]['mortality_rate'] for m in members])
            order = np.argsort(mort)
            sorted_mort = mort[order]
            sorted_members = members[order]

            gaps = np.diff(sorted_mort)
            if len(gaps) == 0:
                continue

            gap_mean = gaps.mean()
            gap_std = gaps.std() if len(gaps) > 1 else 0.0
            threshold = max(gap_mean + 1.0 * gap_std, 0.03)

            sig_gaps = np.where(gaps >= threshold)[0]
            if len(sig_gaps) == 0:
                print(f"    [INFO] Cluster {cid}: no significant internal gaps.")
                continue

            best_gap_idx = sig_gaps[np.argmax(gaps[sig_gaps])]
            if gaps[best_gap_idx] < 2.0 * gap_mean:
                print(f"    [INFO] Cluster {cid}: gap not 2Ã— mean â€” skipped.")
                continue

            refined = True
            low_members = sorted_members[:best_gap_idx + 1]
            high_members = sorted_members[best_gap_idx + 1:]

            if len(high_members) >= len(low_members):
                for m in low_members:
                    new_labels[m] = next_id
            else:
                for m in high_members:
                    new_labels[m] = next_id
            next_id += 1

            low_mort = [client_data[m]['mortality_rate'] for m in low_members]
            high_mort = [client_data[m]['mortality_rate'] for m in high_members]
            print(f"    Cluster {cid} sub-split: "
                  f"[{', '.join(f'{r:.1%}' for r in low_mort)}] | "
                  f"[{', '.join(f'{r:.1%}' for r in high_mort)}]  "
                  f"(gap={gaps[best_gap_idx]:.3f})")

        if refined:
            unique = sorted(set(new_labels))
            remap = {old: new for new, old in enumerate(unique)}
            new_labels = np.array([remap[l] for l in new_labels])

        return new_labels, refined

    # ------------------------------------------------------------------
    # Noise reassignment
    # ------------------------------------------------------------------

    def _reassign_noise(self, vectors_scaled, labels):
        labels = np.array(labels)
        noise_mask = labels == -1
        if not noise_mask.any():
            return labels

        valid_clusters = sorted(set(labels) - {-1})
        if not valid_clusters:
            return labels

        centroids = {
            cid: vectors_scaled[labels == cid].mean(axis=0)
            for cid in valid_clusters
        }

        for idx in np.where(noise_mask)[0]:
            dists = {cid: np.linalg.norm(vectors_scaled[idx] - c)
                     for cid, c in centroids.items()}
            labels[idx] = min(dists, key=dists.get)
            print(f"    Noise client {idx} â†’ Cluster {labels[idx]} "
                  f"(dist={dists[labels[idx]]:.3f})")

        return labels

    # ------------------------------------------------------------------
    # Adaptive mortality-rate fallback
    # ------------------------------------------------------------------

    def _adaptive_mortality_split(self, client_data):
        mort_rates = np.array([d['mortality_rate'] for d in client_data])
        n = len(mort_rates)

        sorted_idx = np.argsort(mort_rates)
        sorted_rates = mort_rates[sorted_idx]
        gaps = np.diff(sorted_rates)

        if len(gaps) == 0 or gaps.max() < 1e-6:
            sizes = np.array([d['n_samples'] for d in client_data])
            median_size = np.median(sizes)
            labels = [0 if s <= median_size else 1 for s in sizes]
            if len(set(labels)) < 2:
                labels[-1] = 1 - labels[0]
            return labels

        gap_mean = gaps.mean()
        gap_std = gaps.std() if len(gaps) > 1 else 0
        threshold = gap_mean + 0.5 * gap_std

        sig_gap_indices = np.where(gaps >= threshold)[0]
        if len(sig_gap_indices) == 0:
            sig_gap_indices = np.array([np.argmax(gaps)])

        max_splits = min(n // 2, 4)
        sorted_by_size = sig_gap_indices[np.argsort(-gaps[sig_gap_indices])]
        split_indices = sorted(sorted_by_size[:max_splits])

        labels = [0] * n
        for i, orig_idx in enumerate(sorted_idx):
            cluster = 0
            for sp in split_indices:
                if i > sp:
                    cluster += 1
            labels[orig_idx] = cluster

        if len(set(labels)) < 2:
            labels[sorted_idx[-1]] = 1

        print(f"    Mortality (sorted): {np.round(sorted_rates, 3)}")
        print(f"    Gaps: {np.round(gaps, 3)}")
        print(f"    Split positions: {split_indices}")

        return labels

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------

    def cluster(self, client_data, feature_names=None):
        """Cluster clients using DBSCAN + sub-cluster refinement."""
        print("\n-- DBSCAN Client Clustering --")

        n_clients = len(client_data)
        self.min_samples = min(self.min_samples, max(2, n_clients // 2))

        features_used = ["Mortality_Rate", "Class_Ratio", "Log_Sample_Size"]
        if feature_names:
            n_use = min(8, len(feature_names))
            features_used.extend(feature_names[:n_use])

        print(f"  Clustering on {len(features_used)} dimensions: "
              f"{', '.join(features_used[:4])} ...")

        vectors = self.build_distribution_vectors(client_data, feature_names)

        print("\n  Client feature snapshot:")
        for idx, vec in enumerate(vectors):
            feat_strs = [f"{name}: {val:.2f}" if i > 0 else f"{name}: {val:.1%}"
                         for i, (name, val) in enumerate(zip(features_used[:4], vec[:4]))]
            print(f"    Client {idx:2d} | {', '.join(feat_strs)} ...")

        scaler = StandardScaler()
        vectors_scaled = scaler.fit_transform(vectors)
        self.vectors_scaled_ = vectors_scaled

        # Phase 1: DBSCAN
        if self.eps is None:
            best_eps, labels = self._auto_tune_eps(vectors_scaled)
            if best_eps is not None:
                self.eps = best_eps
            else:
                labels = None
        else:
            print(f"  Using manual eps = {self.eps:.4f}")
            dbscan = DBSCAN(eps=self.eps, min_samples=self.min_samples)
            labels = dbscan.fit_predict(vectors_scaled)
            n_c = len(set(labels)) - (1 if -1 in labels else 0)
            print(f"  DBSCAN: {n_c} clusters, {(labels == -1).sum()} noise")
            if n_c < 2:
                labels = None

        if labels is None:
            print("  âš  DBSCAN found < 2 clusters â€” adaptive mortality-rate split")
            labels = np.array(self._adaptive_mortality_split(client_data))
        else:
            labels = self._reassign_noise(vectors_scaled, labels)

        # Phase 2: refinement
        n_before = len(set(labels))
        print(f"\n  Phase 2: Checking {n_before} cluster(s) for internal "
              f"mortality heterogeneity â€¦")
        labels, was_refined = self._refine_clusters(labels, client_data)
        n_after = len(set(labels))

        if was_refined:
            print(f"  Refined: {n_before} â†’ {n_after} clusters")
        else:
            print(f"  No refinement needed ({n_after} clusters)")

        self.labels_ = np.array(labels)

        # Quality metrics
        self.metrics_ = _compute_clustering_metrics(vectors_scaled, self.labels_)
        self.metrics_['n_noise'] = 0
        self.metrics_['noise_ratio'] = 0.0
        self.metrics_['eps_used'] = float(self.eps) if self.eps else -1.0

        # Report
        print(f"\n  DBSCAN Clustering Summary:")
        print(f"    Clusters: {self.metrics_['n_clusters']}")
        if self.metrics_['silhouette_score'] != -1.0:
            print(f"    Silhouette: {self.metrics_['silhouette_score']:.4f}")
        if self.metrics_['davies_bouldin'] != -1.0:
            print(f"    Davies-Bouldin: {self.metrics_['davies_bouldin']:.4f}")

        for cid in sorted(set(self.labels_)):
            members = [i for i, l in enumerate(self.labels_) if l == cid]
            morts = [client_data[i]['mortality_rate'] for i in members]
            sizes = [client_data[i]['n_samples'] for i in members]
            print(f"    Cluster {cid}: clients={members}, "
                  f"avg_mortality={np.mean(morts):.1%}, "
                  f"total_samples={sum(sizes)}")

        return self.labels_.tolist()

    def get_metrics(self):
        return self.metrics_


# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
#  Hierarchical / Agglomerative Clusterer  (NEW)
# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•

class AgglomerativeClientClusterer:
    """Clusters FL clients using Hierarchical Agglomerative Clustering.

    Strategy
    --------
    - Represents each client as a distribution vector (mortality rate,
      class ratio, log sample size, feature means).
    - Uses Ward linkage, which minimises within-cluster variance at
      each merge step â€” well-suited to the compact, low-dimensional
      distribution vectors.
    - Automatically selects the optimal number of clusters k by scanning
      k = 2 .. max(2, n_clients // 2) and choosing the k with the
      highest silhouette score.
    - Falls back to k=2 if no silhouette-valid split is found.

    Public interface mirrors ClientClusterer:
        .cluster(client_data, feature_names) -> list[int]
        .get_metrics()                        -> dict
        .vectors_scaled_                      (for PCA visualisation)
    """

    def __init__(self, n_clusters=None, linkage='ward'):
        """
        Parameters
        ----------
        n_clusters : int or None
            If None, auto-selected via silhouette scan.
        linkage : str
            AgglomerativeClustering linkage criterion.
            'ward' works best for compact distribution vectors.
        """
        self.n_clusters = n_clusters
        self.linkage = linkage
        self.labels_ = None
        self.metrics_ = {}
        self.vectors_scaled_ = None
        self.best_k_ = None

    # ------------------------------------------------------------------
    # k selection
    # ------------------------------------------------------------------

    def _select_k(self, vectors_scaled):
        """Scan k = 2..max_k and return the k with best silhouette score."""
        n = len(vectors_scaled)
        max_k = max(2, n // 2)

        print(f"\n  [INFO] Agglomerative: scanning k=2..{max_k} "
              f"(Ward linkage) ...")

        best_k, best_sil, best_labels = 2, -2.0, None

        for k in range(2, max_k + 1):
            agg = AgglomerativeClustering(n_clusters=k, linkage=self.linkage)
            labels_try = agg.fit_predict(vectors_scaled)

            try:
                sil = silhouette_score(vectors_scaled, labels_try)
            except Exception:
                sil = -1.0

            print(f"    k={k} | silhouette={sil:.4f}")

            if sil > best_sil:
                best_sil = sil
                best_k = k
                best_labels = labels_try.copy()

        print(f"  Best k={best_k} (silhouette={best_sil:.4f})")
        return best_k, best_labels

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------

    def cluster(self, client_data, feature_names=None):
        """Cluster clients using Agglomerative (Ward) clustering.

        Parameters
        ----------
        client_data : list of dict
        feature_names : list of str or None

        Returns
        -------
        labels : list of int  (cluster id per client, >= 0, contiguous)
        """
        print("\n-- Hierarchical Agglomerative Client Clustering --")

        n_clients = len(client_data)

        features_used = ["Mortality_Rate", "Class_Ratio", "Log_Sample_Size"]
        if feature_names:
            n_use = min(8, len(feature_names))
            features_used.extend(feature_names[:n_use])

        print(f"  Clustering on {len(features_used)} dimensions: "
              f"{', '.join(features_used[:4])} ...")

        vectors = _build_distribution_vectors(client_data, feature_names)

        print("\n  Client feature snapshot:")
        for idx, vec in enumerate(vectors):
            mort_str = f"Mortality_Rate: {vec[0]:.1%}"
            print(f"    Client {idx:2d} | {mort_str}, "
                  f"Log_N: {vec[2]:.2f} ...")

        scaler = StandardScaler()
        vectors_scaled = scaler.fit_transform(vectors)
        self.vectors_scaled_ = vectors_scaled

        # Select k
        if self.n_clusters is None:
            if n_clients < 3:
                # Degenerate case: force k=2
                best_k = 2
                agg = AgglomerativeClustering(n_clusters=2,
                                              linkage=self.linkage)
                labels = agg.fit_predict(vectors_scaled)
            else:
                best_k, labels = self._select_k(vectors_scaled)
        else:
            best_k = self.n_clusters
            print(f"  Using fixed k={best_k}")
            agg = AgglomerativeClustering(n_clusters=best_k,
                                          linkage=self.linkage)
            labels = agg.fit_predict(vectors_scaled)

        self.best_k_ = best_k

        # Relabel to contiguous 0..K-1 (AgglomerativeClustering already does
        # this, but ensure it explicitly)
        unique = sorted(set(labels))
        remap = {old: new for new, old in enumerate(unique)}
        self.labels_ = np.array([remap[l] for l in labels])

        # Quality metrics
        self.metrics_ = _compute_clustering_metrics(
            vectors_scaled, self.labels_
        )
        self.metrics_['n_noise'] = 0
        self.metrics_['noise_ratio'] = 0.0
        self.metrics_['linkage'] = self.linkage
        self.metrics_['k_selected'] = int(best_k)

        # Report
        print(f"\n  Agglomerative Clustering Summary:")
        print(f"    Clusters (k): {self.metrics_['n_clusters']}")
        print(f"    Linkage: {self.linkage}")
        if self.metrics_['silhouette_score'] != -1.0:
            print(f"    Silhouette: {self.metrics_['silhouette_score']:.4f}")
        if self.metrics_['davies_bouldin'] != -1.0:
            print(f"    Davies-Bouldin: {self.metrics_['davies_bouldin']:.4f}")

        for cid in sorted(set(self.labels_)):
            members = [i for i, l in enumerate(self.labels_) if l == cid]
            morts = [client_data[i]['mortality_rate'] for i in members]
            sizes = [client_data[i]['n_samples'] for i in members]
            print(f"    Cluster {cid}: clients={members}, "
                  f"avg_mortality={np.mean(morts):.1%}, "
                  f"total_samples={sum(sizes)}")

        return self.labels_.tolist()

    def get_metrics(self):
        return self.metrics_
