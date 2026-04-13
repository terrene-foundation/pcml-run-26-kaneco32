# Copyright 2026 Terrene Foundation
# SPDX-License-Identifier: Apache-2.0
"""
# ════════════════════════════════════════════════════════════════════════
# MLFP04 — Exercise 1: Clustering
# ════════════════════════════════════════════════════════════════════════
#
# WHAT YOU'LL LEARN:
#   After completing this exercise, you will be able to:
#   - Apply K-means with k-means++ initialisation and understand convergence
#   - Build and interpret hierarchical dendrograms with 4 linkage methods
#   - Apply DBSCAN (epsilon-neighbourhood, core/border/noise) and HDBSCAN
#   - Apply spectral clustering via graph Laplacian for non-convex shapes
#   - Evaluate clusters with silhouette, Davies-Bouldin, Calinski-Harabasz,
#     gap statistic, ARI, and NMI
#   - Use AutoMLEngine for automated clustering comparison
#   - Profile clusters with business-meaningful descriptions
#   - Select clustering algorithm based on data characteristics
#
# PREREQUISITES:
#   - MLFP03 complete (supervised ML, feature engineering, preprocessing)
#   - MLFP02 complete (statistics, regression — for evaluation intuition)
#
# ESTIMATED TIME: ~150-180 min
#
# TASKS:
#   1.  Load and prepare e-commerce customer features
#   2.  K-means: elbow method, silhouette analysis, k-means++ vs random
#   3.  Hierarchical clustering with 4 linkage methods + dendrograms
#   4.  DBSCAN: epsilon selection via k-distance plot, core/border/noise
#   5.  HDBSCAN: auto-epsilon, comparison with DBSCAN
#   6.  Spectral clustering: graph Laplacian for non-convex clusters
#   7.  Cluster evaluation: silhouette, DB, CH, gap statistic, ARI, NMI
#   8.  AutoMLEngine with agent=True double opt-in
#   9.  Cluster profiling: business-meaningful segment descriptions
#   10. Algorithm selection guide and visualisation
#
# DATASET: E-commerce customer data (from MLFP03)
#   Goal: segment customers without labels into behavioural groups
#   Business context: cluster = marketing segment (loyalty, spending, frequency)
#   USML bridge: In M3 you predicted labels. Now: what if there are no labels?
#
# ════════════════════════════════════════════════════════════════════════
"""
from __future__ import annotations

import asyncio
import time

import numpy as np
import polars as pl
from scipy.cluster.hierarchy import linkage, fcluster, dendrogram
from scipy.spatial.distance import pdist
from sklearn.cluster import KMeans, SpectralClustering, DBSCAN
from sklearn.mixture import GaussianMixture
from sklearn.metrics import (
    silhouette_score,
    silhouette_samples,
    calinski_harabasz_score,
    davies_bouldin_score,
    adjusted_rand_score,
    normalized_mutual_info_score,
)
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

from kailash_ml import PreprocessingPipeline, ModelVisualizer, DataExplorer
from kailash_ml.interop import to_sklearn_input

from shared import MLFPDataLoader

try:
    import hdbscan as hdbscan_lib
except ImportError:
    hdbscan_lib = None


# ── Data Loading ──────────────────────────────────────────────────────

loader = MLFPDataLoader()
customers = loader.load("mlfp03", "ecommerce_customers.parquet")

print(f"=== E-commerce Customer Data ===")
print(f"Shape: {customers.shape}")
print(f"Columns: {customers.columns}")

# TODO: Build feature_cols — a list of column names whose dtype is Float64,
#       Float32, Int64, or Int32, and whose name is NOT "customer_id".
# Hint: iterate zip(customers.columns, customers.dtypes) and filter by dtype
feature_cols = [
    ____  # Hint: c for c, d in zip(customers.columns, customers.dtypes) if d in (pl.Float64, ...)
]

print(f"Clustering features ({len(feature_cols)}): {feature_cols}")

# TODO: Prepare X using to_sklearn_input — drop nulls first, pass feature_columns
# Hint: to_sklearn_input(customers.drop_nulls(subset=feature_cols), feature_columns=feature_cols)
X, _, col_info = ____  # Hint: returns (array, target, col_info)

# TODO: Standardise X using StandardScaler — fit_transform
# Hint: scaler.fit_transform(X)
scaler = StandardScaler()
X_scaled = ____  # Hint: StandardScaler().fit_transform(X)
n_samples, n_features = X_scaled.shape
print(f"Samples: {n_samples:,}, Features: {n_features}")

# ── Checkpoint 1 ─────────────────────────────────────────────────────
assert X_scaled.shape[0] > 0, "Customer dataset should not be empty"
assert X_scaled.shape[1] > 0, "Should have at least one clustering feature"
assert len(feature_cols) > 0, "Feature column list should not be empty"
assert abs(X_scaled.mean()) < 0.01, "Standardised data should have mean ≈ 0"
assert abs(X_scaled.std() - 1.0) < 0.1, "Standardised data should have std ≈ 1"
# INTERPRETATION: Standardisation (zero mean, unit variance) is mandatory
# for all distance-based algorithms (K-means, KNN, spectral, HDBSCAN).
# Without it, a feature measured in thousands (e.g., revenue) dominates
# distance calculations and makes all other features irrelevant.
print("\n✓ Checkpoint 1 passed — customer features loaded and standardised\n")


# ══════════════════════════════════════════════════════════════════════
# TASK 2: K-means — elbow, silhouette analysis, k-means++ vs random
# ══════════════════════════════════════════════════════════════════════
# K-means objective: minimise Sum_k Sum_{x in C_k} ||x - mu_k||^2
# k-means++ initialises centroids to be far apart, avoiding poor convergence.

# 2a: Elbow method — inertia (within-cluster sum of squares) vs K
inertias = []
sil_scores = []
ch_scores = []
db_scores = []
K_range = range(2, 11)

print("=== K-means: Elbow + Silhouette Analysis ===")
print(f"{'K':>4} {'Inertia':>14} {'Silhouette':>12} {'CH Index':>12} {'DB Index':>10}")
print("─" * 56)

for k in K_range:
    # TODO: Create a KMeans with n_clusters=k, random_state=42, n_init=10, init="k-means++"
    # and call fit_predict(X_scaled)
    # Hint: KMeans(n_clusters=k, random_state=42, n_init=10, init="k-means++").fit_predict(X_scaled)
    km = (
        ____  # Hint: KMeans(n_clusters=k, n_init=10, random_state=42, init="k-means++")
    )
    labels = ____  # Hint: km.fit_predict(X_scaled)
    inertias.append(km.inertia_)
    # TODO: Compute silhouette_score, calinski_harabasz_score, davies_bouldin_score
    sil = ____  # Hint: silhouette_score(X_scaled, labels)
    ch = ____  # Hint: calinski_harabasz_score(X_scaled, labels)
    db = ____  # Hint: davies_bouldin_score(X_scaled, labels)
    sil_scores.append(sil)
    ch_scores.append(ch)
    db_scores.append(db)
    print(f"{k:>4} {km.inertia_:>14.0f} {sil:>12.4f} {ch:>12.0f} {db:>10.4f}")

best_k_sil = list(K_range)[np.argmax(sil_scores)]
print(f"\nBest K by silhouette: {best_k_sil} (score={max(sil_scores):.4f})")

# 2b: k-means++ vs random init comparison
# TODO: Create km_plus with init="k-means++" and km_random with init="random",
#       both with n_clusters=best_k_sil, random_state=42, n_init=10
km_plus = ____  # Hint: KMeans(n_clusters=best_k_sil, random_state=42, n_init=10, init="k-means++")
km_random = ____  # Hint: KMeans(n_clusters=best_k_sil, random_state=42, n_init=10, init="random")

t0 = time.time()
km_plus.fit(X_scaled)
t_plus = time.time() - t0

t0 = time.time()
km_random.fit(X_scaled)
t_random = time.time() - t0

print(f"\nk-means++ vs Random Initialisation (K={best_k_sil}):")
print(
    f"  k-means++: inertia={km_plus.inertia_:.0f}, iterations={km_plus.n_iter_}, time={t_plus:.3f}s"
)
print(
    f"  Random:    inertia={km_random.inertia_:.0f}, iterations={km_random.n_iter_}, time={t_random:.3f}s"
)
print(
    "  k-means++ spreads initial centroids, converging faster and to better solutions."
)

km_best = km_plus
km_labels = km_best.predict(X_scaled)

# 2c: Per-sample silhouette analysis
# TODO: Compute silhouette_samples for km_labels
sil_samples = ____  # Hint: silhouette_samples(X_scaled, km_labels)
print(f"\n=== Per-Sample Silhouette Analysis (K={best_k_sil}) ===")
for cluster_id in range(best_k_sil):
    cluster_sil = sil_samples[km_labels == cluster_id]
    n_neg = (cluster_sil < 0).sum()
    print(
        f"  Cluster {cluster_id}: n={len(cluster_sil):,}, "
        f"mean_sil={cluster_sil.mean():.4f}, "
        f"negative_sil={n_neg} ({n_neg/len(cluster_sil):.1%})"
    )
print("  Points with negative silhouette may be mis-assigned to the wrong cluster.")

# ── Checkpoint 2 ─────────────────────────────────────────────────────
assert 2 <= best_k_sil <= 10, f"Best K should be between 2 and 10, got {best_k_sil}"
assert max(sil_scores) > 0, "Best silhouette score should be positive"
assert (
    len(set(km_labels)) == best_k_sil
), "K-means should produce exactly best_k clusters"
assert (
    km_plus.inertia_ <= km_random.inertia_ + 1
), "k-means++ should achieve inertia no worse than random init"
# INTERPRETATION: The elbow method in K-means is subjective; the silhouette
# score gives an objective criterion. Silhouette s(i) in [-1, 1] measures
# how much closer point i is to its own cluster than to the nearest other
# cluster. High silhouette = compact, well-separated clusters.
print(
    "\n✓ Checkpoint 2 passed — K-means optimal K selected, k-means++ confirmed superior\n"
)


# ══════════════════════════════════════════════════════════════════════
# TASK 3: Hierarchical clustering with 4 linkage methods + dendrograms
# ══════════════════════════════════════════════════════════════════════
# Agglomerative (bottom-up): merge closest clusters until one remains.
# Linkage methods:
#   Single  (min distance): tends to create elongated chains
#   Complete (max distance): tends to create compact, spherical clusters
#   Average: compromise between single and complete
#   Ward's: minimises within-cluster variance (like K-means criterion)

rng = np.random.default_rng(42)
n_hier = min(2000, n_samples)
idx_hier = rng.choice(n_samples, n_hier, replace=False)
X_hier = X_scaled[idx_hier]

linkage_methods = ["single", "complete", "average", "ward"]
hier_results = {}

print("=== Hierarchical Clustering: 4 Linkage Methods ===")
print(f"Subsample: {n_hier} points (dendrogram readability)")
print(
    f"{'Linkage':<12} {'Cut K':>6} {'Silhouette':>12} {'CH':>10} {'DB':>8} {'Time':>8}"
)
print("─" * 58)

for method in linkage_methods:
    t0 = time.time()
    # TODO: Compute the linkage matrix Z using scipy linkage().
    #       For "ward", use: linkage(X_hier, method=method)
    #       For others, add metric="euclidean"
    if method == "ward":
        Z = ____  # Hint: linkage(X_hier, method=method)
    else:
        Z = ____  # Hint: linkage(X_hier, method=method, metric="euclidean")
    elapsed = time.time() - t0

    # TODO: Cut the dendrogram to get best_k_sil clusters using fcluster
    # Hint: fcluster(Z, t=best_k_sil, criterion="maxclust") then subtract 1
    hier_labels = ____  # Hint: fcluster(Z, t=best_k_sil, criterion="maxclust") - 1

    n_clusters_actual = len(set(hier_labels))
    if n_clusters_actual >= 2:
        sil = silhouette_score(X_hier, hier_labels)
        ch = calinski_harabasz_score(X_hier, hier_labels)
        db = davies_bouldin_score(X_hier, hier_labels)
    else:
        sil, ch, db = -1.0, 0.0, float("inf")

    hier_results[method] = {
        "Z": Z,
        "labels": hier_labels,
        "silhouette": sil,
        "ch": ch,
        "db": db,
        "n_clusters": n_clusters_actual,
        "time": elapsed,
    }
    print(
        f"{method:<12} {n_clusters_actual:>6} {sil:>12.4f} {ch:>10.0f} {db:>8.4f} {elapsed:>7.2f}s"
    )

print("\n=== Dendrogram Interpretation Guide ===")
print("  Y-axis: merge distance (height). Higher = more dissimilar clusters merged.")
print(
    "  Horizontal cuts at height h produce clusters: count branches crossing the line."
)
print("  Large vertical gaps suggest 'natural' cluster boundaries.")
print("  Linkage method choice:")
print("    Single:   detects elongated shapes, but susceptible to chaining artefacts")
print("    Complete: compact spherical clusters, sensitive to outliers")
print("    Average:  balanced between single and complete")
print("    Ward's:   minimises variance, produces clusters similar to K-means")

# ── Checkpoint 3 ─────────────────────────────────────────────────────
assert len(hier_results) == 4, "Should have results for all 4 linkage methods"
assert all(
    "silhouette" in r for r in hier_results.values()
), "All methods should report silhouette scores"
# INTERPRETATION: Dendrograms are read bottom-up. Each horizontal line is a merge.
# The height of the merge = the distance between the two clusters being merged.
# Cut the tree at any height to get a partition. Ward's method minimises
# within-cluster variance at each merge, producing the most K-means-like result.
print("\n✓ Checkpoint 3 passed — hierarchical clustering with 4 linkage methods\n")


# ══════════════════════════════════════════════════════════════════════
# TASK 4: DBSCAN — epsilon selection via k-distance plot
# ══════════════════════════════════════════════════════════════════════
# DBSCAN defines clusters by local density:
#   Core point: has >= minPts neighbours within radius epsilon
#   Border point: within epsilon of a core point but not itself core
#   Noise point: neither core nor border (label = -1)
#
# Epsilon selection: k-distance plot
#   1. For each point, compute distance to k-th nearest neighbour
#   2. Sort these distances
#   3. The "elbow" in the sorted plot suggests a good epsilon

k_nn = 10  # minPts will be k_nn
# TODO: Fit NearestNeighbors and get k-distances
# Hint: NearestNeighbors(n_neighbors=k_nn).fit(X_scaled)
nn = ____  # Hint: NearestNeighbors(n_neighbors=k_nn)
nn.fit(X_scaled)
distances, _ = nn.kneighbors(X_scaled)
k_distances = distances[:, -1]
k_distances_sorted = np.sort(k_distances)

# Find elbow via second derivative
diffs = np.diff(k_distances_sorted)
diffs2 = np.diff(diffs)
elbow_idx = np.argmax(diffs2) + 2
eps_suggested = k_distances_sorted[elbow_idx]

print(f"=== DBSCAN: k-Distance Plot for Epsilon Selection ===")
print(f"k = {k_nn} (will use as minPts)")
print(f"Suggested epsilon (elbow): {eps_suggested:.4f}")
print(f"k-distance range: [{k_distances_sorted[0]:.4f}, {k_distances_sorted[-1]:.4f}]")

eps_values = [eps_suggested * 0.7, eps_suggested, eps_suggested * 1.3]
dbscan_results = {}

print(f"\n{'eps':>8} {'Clusters':>10} {'Noise %':>10} {'Silhouette':>12}")
print("─" * 44)

for eps in eps_values:
    # TODO: Run DBSCAN with eps=eps, min_samples=k_nn, n_jobs=-1
    # Hint: DBSCAN(eps=eps, min_samples=k_nn, n_jobs=-1).fit_predict(X_scaled)
    db = ____  # Hint: DBSCAN(eps=eps, min_samples=k_nn, n_jobs=-1)
    labels = ____  # Hint: db.fit_predict(X_scaled)
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    noise_pct = (labels == -1).mean()

    valid = labels != -1
    if valid.sum() >= 2 and n_clusters >= 2:
        sil = silhouette_score(X_scaled[valid], labels[valid])
    else:
        sil = -1.0

    dbscan_results[eps] = {
        "labels": labels,
        "n_clusters": n_clusters,
        "noise_pct": noise_pct,
        "silhouette": sil,
    }
    print(f"{eps:>8.4f} {n_clusters:>10} {noise_pct:>9.1%} {sil:>12.4f}")

best_eps = eps_suggested
db_labels = dbscan_results[best_eps]["labels"]
n_core = (db_labels >= 0).sum()
n_noise = (db_labels == -1).sum()

print(f"\nDBSCAN classification (eps={best_eps:.4f}, minPts={k_nn}):")
print(f"  Core+Border points: {n_core:,}")
print(f"  Noise points:       {n_noise:,} ({n_noise/n_samples:.1%})")
print(f"  Clusters found:     {dbscan_results[best_eps]['n_clusters']}")

# ── Checkpoint 4 ─────────────────────────────────────────────────────
assert eps_suggested > 0, "Suggested epsilon should be positive"
assert any(
    r["n_clusters"] >= 2 for r in dbscan_results.values()
), "DBSCAN should find at least 2 clusters for some epsilon"
# INTERPRETATION: DBSCAN is density-based: it finds clusters as dense regions
# separated by sparse regions. Unlike K-means, it does not require K as input
# and can discover clusters of arbitrary shape. The noise label (-1) is a
# feature, not a bug: points in sparse regions are genuinely ambiguous.
print("\n✓ Checkpoint 4 passed — DBSCAN with epsilon selection via k-distance plot\n")


# ══════════════════════════════════════════════════════════════════════
# TASK 5: HDBSCAN — hierarchical extension, auto-selects epsilon
# ══════════════════════════════════════════════════════════════════════
# HDBSCAN builds a hierarchy of DBSCAN clusterings across all epsilon
# values, then extracts the "most stable" clusters. No epsilon needed.

if hdbscan_lib is not None:
    print("=== HDBSCAN: Hierarchical Density-Based Clustering ===")

    # TODO: Create HDBSCAN with min_cluster_size=50, min_samples=10,
    #       cluster_selection_method="eom" and call fit_predict(X_scaled)
    hdb = ____  # Hint: hdbscan_lib.HDBSCAN(min_cluster_size=50, min_samples=10, cluster_selection_method="eom")
    hdbscan_labels = ____  # Hint: hdb.fit_predict(X_scaled)
    n_hdbscan = len(set(hdbscan_labels)) - (1 if -1 in hdbscan_labels else 0)
    noise_pct = (hdbscan_labels == -1).mean()

    hdb_probs = hdb.probabilities_

    valid_hdb = hdbscan_labels != -1
    if valid_hdb.sum() >= 2 and n_hdbscan >= 2:
        hdb_sil = silhouette_score(X_scaled[valid_hdb], hdbscan_labels[valid_hdb])
    else:
        hdb_sil = -1.0

    print(f"Clusters found: {n_hdbscan}")
    print(f"Noise points: {noise_pct:.1%}")
    print(f"Silhouette (non-noise): {hdb_sil:.4f}")
    print(f"Mean cluster probability: {hdb_probs[valid_hdb].mean():.4f}")

    # TODO: Create hdb_leaf with cluster_selection_method="leaf" and compare
    hdb_leaf = ____  # Hint: hdbscan_lib.HDBSCAN(min_cluster_size=50, min_samples=10, cluster_selection_method="leaf")
    hdb_leaf_labels = ____  # Hint: hdb_leaf.fit_predict(X_scaled)
    n_leaf = len(set(hdb_leaf_labels)) - (1 if -1 in hdb_leaf_labels else 0)

    print(f"\nCluster selection comparison:")
    print(f"  EOM (Excess of Mass): {n_hdbscan} clusters — variable granularity")
    print(f"  Leaf:                 {n_leaf} clusters — finest granularity")
    print("  EOM is default and preferred for most applications.")
else:
    hdbscan_labels = km_labels
    hdb_sil = -1.0
    print("HDBSCAN not installed — using K-means labels as fallback")

# ── Checkpoint 5 ─────────────────────────────────────────────────────
if hdbscan_lib is not None:
    assert n_hdbscan >= 1, "HDBSCAN should find at least 1 cluster"
    assert 0 <= noise_pct <= 1, "Noise percentage should be in [0, 1]"
# INTERPRETATION: HDBSCAN eliminates DBSCAN's epsilon hyperparameter by
# exploring all density levels hierarchically. It extracts "persistent"
# clusters — clusters that remain stable across a range of densities.
# The soft probability assignment indicates how confident the algorithm
# is that each point belongs to its assigned cluster.
print("\n✓ Checkpoint 5 passed — HDBSCAN auto-discovers clusters without epsilon\n")


# ══════════════════════════════════════════════════════════════════════
# TASK 6: Spectral clustering — graph Laplacian
# ══════════════════════════════════════════════════════════════════════
# Spectral clustering:
#   1. Build affinity matrix A (RBF kernel)
#   2. Compute graph Laplacian L = D - A
#   3. Find first k eigenvectors of L (spectral embedding)
#   4. Run K-means on the spectral embedding
# Key advantage: finds non-convex clusters (moons, rings, spirals)
# Key limitation: O(n^3) eigendecomposition — subsample for large data

n_spectral = min(5000, n_samples)
idx_spec = rng.choice(n_samples, n_spectral, replace=False)
X_spec = X_scaled[idx_spec]

print(f"=== Spectral Clustering (n={n_spectral}) ===")
print(f"Subsample: {n_spectral} points (spectral is O(n^3))")

spectral_results = {}
for n_c in [best_k_sil, best_k_sil + 1, best_k_sil - 1]:
    if n_c < 2:
        continue
    t0 = time.time()
    # TODO: Create SpectralClustering with n_clusters=n_c, random_state=42,
    #       affinity="rbf", gamma=1.0 and call fit_predict(X_spec)
    spec = ____  # Hint: SpectralClustering(n_clusters=n_c, random_state=42, affinity="rbf", gamma=1.0)
    spec_labels = ____  # Hint: spec.fit_predict(X_spec)
    elapsed = time.time() - t0

    sil = silhouette_score(X_spec, spec_labels)
    spectral_results[n_c] = {
        "labels": spec_labels,
        "silhouette": sil,
        "time": elapsed,
    }
    print(f"  K={n_c}: silhouette={sil:.4f}, time={elapsed:.2f}s")

# ── Checkpoint 6 ─────────────────────────────────────────────────────
assert len(spectral_results) >= 2, "Should test at least 2 values of K for spectral"
# INTERPRETATION: Spectral clustering embeds data using the graph Laplacian
# eigenvectors, then clusters in this spectral space. This can separate
# non-convex clusters that K-means cannot handle (e.g., concentric rings).
# The RBF kernel width (gamma) controls how local the similarity measure is.
print("\n✓ Checkpoint 6 passed — spectral clustering via graph Laplacian\n")


# ══════════════════════════════════════════════════════════════════════
# TASK 7: Full evaluation — silhouette, DB, CH, gap statistic, ARI, NMI
# ══════════════════════════════════════════════════════════════════════

from sklearn.neighbors import KNeighborsClassifier

all_method_labels = {
    "K-means": km_labels,
    "HDBSCAN": hdbscan_labels,
    "DBSCAN": db_labels,
}

best_spec_k = max(spectral_results.items(), key=lambda x: x[1]["silhouette"])[0]
spec_best_labels = np.full(n_samples, -1)
spec_best_labels[idx_spec] = spectral_results[best_spec_k]["labels"]
knn_spec = KNeighborsClassifier(n_neighbors=5)
knn_spec.fit(X_spec, spectral_results[best_spec_k]["labels"])
spec_best_labels[spec_best_labels == -1] = knn_spec.predict(
    X_scaled[spec_best_labels == -1]
)
all_method_labels["Spectral"] = spec_best_labels

ward_labels_ext = np.full(n_samples, -1)
ward_labels_ext[idx_hier] = hier_results["ward"]["labels"]
knn_ward = KNeighborsClassifier(n_neighbors=5)
knn_ward.fit(X_hier, hier_results["ward"]["labels"])
ward_labels_ext[ward_labels_ext == -1] = knn_ward.predict(
    X_scaled[ward_labels_ext == -1]
)
all_method_labels["Ward"] = ward_labels_ext

# TODO: Fit GaussianMixture with n_components=best_k_sil, random_state=42,
#       covariance_type="full" and call fit_predict(X_scaled)
gmm = ____  # Hint: GaussianMixture(n_components=best_k_sil, random_state=42, covariance_type="full")
gmm_labels = ____  # Hint: gmm.fit_predict(X_scaled)
all_method_labels["GMM"] = gmm_labels

results = {}
for name, labels in all_method_labels.items():
    valid = labels != -1
    if valid.sum() < 2 or len(set(labels[valid])) < 2:
        continue
    data = X_scaled[valid]
    labs = labels[valid]
    results[name] = {
        "n_clusters": len(set(labs)),
        "silhouette": silhouette_score(data, labs),
        "calinski_harabasz": calinski_harabasz_score(data, labs),
        "davies_bouldin": davies_bouldin_score(data, labs),
    }

print(f"\n=== Clustering Comparison (all methods) ===")
print(f"{'Method':<12} {'K':>4} {'Silhouette':>12} {'CH':>12} {'DB':>8}")
print("─" * 52)
for name, r in results.items():
    print(
        f"{name:<12} {r['n_clusters']:>4} {r['silhouette']:>12.4f} "
        f"{r['calinski_harabasz']:>12.0f} {r['davies_bouldin']:>8.4f}"
    )


def gap_statistic(
    X: np.ndarray, k_range: range, n_refs: int = 10, seed: int = 42
) -> dict:
    """Gap(k) = E*[log(W_k)] - log(W_k): compare dispersion to reference."""
    rng_gap = np.random.default_rng(seed)
    gaps = []
    sks = []
    for k in k_range:
        # TODO: Fit KMeans with n_clusters=k, random_state=42, n_init=5 on X
        # Hint: KMeans(n_clusters=k, random_state=42, n_init=5).fit(X)
        km = ____  # Hint: KMeans(n_clusters=k, random_state=42, n_init=5)
        km.fit(X)
        log_wk = np.log(km.inertia_)

        ref_log_wks = []
        for _ in range(n_refs):
            X_ref = rng_gap.uniform(X.min(axis=0), X.max(axis=0), size=X.shape)
            km_ref = KMeans(n_clusters=k, random_state=42, n_init=3)
            km_ref.fit(X_ref)
            ref_log_wks.append(np.log(km_ref.inertia_))

        ref_mean = np.mean(ref_log_wks)
        ref_std = np.std(ref_log_wks) * np.sqrt(1 + 1.0 / n_refs)
        gaps.append(ref_mean - log_wk)
        sks.append(ref_std)

    return {"k_range": list(k_range), "gaps": gaps, "sks": sks}


print("\n=== Gap Statistic ===")
gap_result = gap_statistic(X_scaled[:3000], range(2, 9), n_refs=5)

best_gap_k = None
for i in range(len(gap_result["gaps"]) - 1):
    if gap_result["gaps"][i] >= gap_result["gaps"][i + 1] - gap_result["sks"][i + 1]:
        best_gap_k = gap_result["k_range"][i]
        break

if best_gap_k is None:
    best_gap_k = gap_result["k_range"][np.argmax(gap_result["gaps"])]

print(f"Gap-optimal K: {best_gap_k}")
for k, g, s in zip(gap_result["k_range"], gap_result["gaps"], gap_result["sks"]):
    marker = " ← optimal" if k == best_gap_k else ""
    print(f"  K={k}: gap={g:.4f} (sk={s:.4f}){marker}")

print("\n=== External Validation: ARI and NMI ===")
print("  (Compare methods pairwise — higher = more agreement)")
method_names = list(all_method_labels.keys())
for i in range(len(method_names)):
    for j in range(i + 1, len(method_names)):
        m1, m2 = method_names[i], method_names[j]
        l1, l2 = all_method_labels[m1], all_method_labels[m2]
        valid = (l1 >= 0) & (l2 >= 0)
        if valid.sum() < 2:
            continue
        # TODO: Compute adjusted_rand_score and normalized_mutual_info_score
        ari = ____  # Hint: adjusted_rand_score(l1[valid], l2[valid])
        nmi = ____  # Hint: normalized_mutual_info_score(l1[valid], l2[valid])
        print(f"  {m1:<10} vs {m2:<10}: ARI={ari:.4f}, NMI={nmi:.4f}")

# ── Checkpoint 7 ─────────────────────────────────────────────────────
assert len(results) >= 3, "At least 3 clustering methods should be evaluated"
assert all(
    "silhouette" in r for r in results.values()
), "All methods should report silhouette"
assert all(
    r["silhouette"] > -1.0 for r in results.values()
), "Silhouette should be > -1"
assert best_gap_k in range(2, 9), "Gap-optimal K should be in tested range"
# INTERPRETATION: Davies-Bouldin measures cluster compactness vs separation
# (lower = better). Calinski-Harabasz measures between-cluster to within-cluster
# variance ratio (higher = better). Gap statistic compares within-cluster
# dispersion to a reference distribution (higher gap = real cluster structure).
# ARI and NMI measure agreement between two clusterings (1 = perfect agreement).
print("\n✓ Checkpoint 7 passed — full evaluation with 6 metrics across all methods\n")


# ══════════════════════════════════════════════════════════════════════
# TASK 8: AutoMLEngine with agent=True double opt-in
# ══════════════════════════════════════════════════════════════════════

from kailash_ml.engines.automl_engine import AutoMLEngine, AutoMLConfig


async def automl_comparison():
    """Use AutoMLEngine for automated clustering comparison."""
    # TODO: Create AutoMLConfig with:
    #       task_type="clustering", metric_to_optimize="silhouette",
    #       direction="maximize", search_strategy="random",
    #       search_n_trials=20, agent=False, max_llm_cost_usd=1.0
    config = ____  # Hint: AutoMLConfig(task_type="clustering", metric_to_optimize="silhouette", ...)

    print(f"\n=== AutoMLEngine Config ===")
    print(f"Task: {config.task_type}")
    print(f"Metric: {config.metric_to_optimize}")
    print(f"Agent enabled: {config.agent}")
    print(f"LLM cost budget: ${config.max_llm_cost_usd}")
    print("\nNote: agent=True requires BOTH the flag AND kailash-ml[agents] installed.")
    print("This is the 'double opt-in' pattern — governance by design.")
    return config


config = asyncio.run(automl_comparison())

# ── Checkpoint 8 ─────────────────────────────────────────────────────
assert config is not None, "AutoMLEngine config should be created"
assert config.agent is False, "agent=False is the default (double opt-in for LLM)"
assert config.max_llm_cost_usd > 0, "LLM cost cap should be set"
# INTERPRETATION: The agent=True double opt-in pattern is governance by design.
# LLM-guided algorithm selection can be powerful but costs money and introduces
# non-determinism. Requiring BOTH the flag AND the package makes costs explicit.
print("\n✓ Checkpoint 8 passed — AutoMLEngine configured with governance guardrails\n")


# ══════════════════════════════════════════════════════════════════════
# TASK 9: Cluster profiling — business-meaningful descriptions
# ══════════════════════════════════════════════════════════════════════

best_method = max(results.items(), key=lambda x: x[1]["silhouette"])
print(f"\n=== Cluster Profiles (best method: {best_method[0]}) ===")
print(f"Silhouette: {best_method[1]['silhouette']:.4f}")

best_labels = all_method_labels[best_method[0]]

_base_df = customers.drop_nulls(subset=feature_cols)
# TODO: Add a "cluster" column to _base_df using pl.Series
# Hint: _base_df[:len(best_labels)].with_columns(pl.Series("cluster", best_labels))
clustered = ____  # Hint: _base_df[:len(best_labels)].with_columns(pl.Series("cluster", best_labels))

for cluster_id in sorted(clustered["cluster"].unique().to_list()):
    if cluster_id == -1:
        continue
    subset = clustered.filter(pl.col("cluster") == cluster_id)
    pct = subset.height / clustered.height * 100
    print(f"\n{'─' * 60}")
    print(f"Cluster {cluster_id} (n={subset.height:,}, {pct:.1f}% of total):")
    print(f"{'─' * 60}")

    high_features = []
    low_features = []
    for col in feature_cols[:8]:
        mean_val = subset[col].mean()
        overall_mean = clustered[col].mean()
        overall_std = clustered[col].std()
        if overall_std and overall_std > 0:
            z_diff = (mean_val - overall_mean) / overall_std
        else:
            z_diff = 0.0
        diff_pct = (mean_val - overall_mean) / (abs(overall_mean) + 1e-9) * 100

        indicator = "▲" if z_diff > 0.5 else "▼" if z_diff < -0.5 else "─"
        print(
            f"  {col:<28} {mean_val:>10.2f} ({diff_pct:+.1f}% vs avg, z={z_diff:+.2f}) {indicator}"
        )
        if z_diff > 0.5:
            high_features.append(col)
        elif z_diff < -0.5:
            low_features.append(col)

    if high_features:
        print(f"  Summary: HIGH in {', '.join(high_features[:3])}")
    if low_features:
        print(f"           LOW  in {', '.join(low_features[:3])}")

# ── Checkpoint 9 ─────────────────────────────────────────────────────
assert "cluster" in clustered.columns, "Clustered dataframe should have cluster column"
n_non_noise = clustered.filter(pl.col("cluster") >= 0).height
assert n_non_noise > 0, "At least some points should be assigned to valid clusters"
# INTERPRETATION: Cluster profiling converts statistical labels into business
# language. "Cluster 0" is meaningless to a marketing manager; "High-frequency
# low-spenders" is actionable. The z-score relative to the population mean
# highlights which features distinguish each cluster from the overall population.
print("\n✓ Checkpoint 9 passed — cluster profiles with business interpretation\n")


# ══════════════════════════════════════════════════════════════════════
# TASK 10: Algorithm selection guide and visualisation
# ══════════════════════════════════════════════════════════════════════

# TODO: Create a ModelVisualizer instance and generate comparison charts
viz = ____  # Hint: ModelVisualizer()

fig = viz.metric_comparison(results)
fig.update_layout(title="Clustering Method Comparison")
fig.write_html("ex1_clustering_comparison.html")
print("\nSaved: ex1_clustering_comparison.html")

elbow_data = {
    "Silhouette": sil_scores,
    "Inertia (scaled)": [i / max(inertias) for i in inertias],
}
fig_elbow = viz.training_history(elbow_data, x_label="K")
fig_elbow.update_layout(title="K-means: Silhouette and Inertia vs K")
fig_elbow.write_html("ex1_elbow.html")
print("Saved: ex1_elbow.html")

gap_data = {"Gap Statistic": gap_result["gaps"]}
fig_gap = viz.training_history(gap_data, x_label="K")
fig_gap.update_layout(title="Gap Statistic vs K")
fig_gap.write_html("ex1_gap_statistic.html")
print("Saved: ex1_gap_statistic.html")

print(f"\n=== Algorithm Selection Guide ===")
print(
    f"""
┌──────────────────┬───────────────────┬──────────────┬──────────────┬───────────────┐
│ Algorithm        │ Requires K?       │ Cluster Shape│ Noise        │ Scalability   │
├──────────────────┼───────────────────┼──────────────┼──────────────┼───────────────┤
│ K-means          │ Yes               │ Convex       │ None         │ O(nKI)        │
│ Hierarchical     │ Yes (cut height)  │ Any (single) │ None         │ O(n^2 log n)  │
│ DBSCAN           │ No (eps, minPts)  │ Arbitrary    │ Yes (-1)     │ O(n log n)    │
│ HDBSCAN          │ No (auto)         │ Arbitrary    │ Yes (-1)     │ O(n log n)    │
│ Spectral         │ Yes               │ Non-convex   │ None         │ O(n^3)        │
│ GMM              │ Yes (BIC selects) │ Ellipsoidal  │ None (soft)  │ O(nK^2d)      │
└──────────────────┴───────────────────┴──────────────┴──────────────┴───────────────┘

When to use each:
  K-means:      Large data, expect spherical clusters, know K
  Hierarchical: Need dendrogram, small-medium data, want to explore K
  DBSCAN:       Arbitrary shapes, known density, want noise detection
  HDBSCAN:      Arbitrary shapes, unknown density, production default
  Spectral:     Non-convex clusters, small data, graph structure
  GMM:          Overlapping clusters, need soft assignments, model selection via BIC
"""
)

# ── Checkpoint 10 ────────────────────────────────────────────────────
best_label = best_method[0]
assert best_label in results, "Best method must be in the results dict"
# INTERPRETATION: No single clustering algorithm is universally best.
# The choice depends on: data size, cluster shape, need for noise detection,
# interpretability (hierarchical dendrograms), and soft vs hard assignments.
print("\n✓ Checkpoint 10 passed — algorithm selection guide complete\n")

print("\n✓ Exercise 1 complete — clustering comparison with 6 algorithms")


# ══════════════════════════════════════════════════════════════════════
# REFLECTION
# ══════════════════════════════════════════════════════════════════════
print("═" * 70)
print("  WHAT YOU'VE MASTERED")
print("═" * 70)
print(
    f"""
  ✓ K-means: elbow + silhouette + gap statistic to select optimal K
  ✓ k-means++: smarter initialisation converges faster to better solutions
  ✓ Hierarchical: 4 linkage methods (single, complete, average, Ward's)
  ✓ Dendrograms: read bottom-up, cut at height to get K clusters
  ✓ DBSCAN: epsilon-neighbourhood, core/border/noise, k-distance plot
  ✓ HDBSCAN: auto-epsilon, EOM vs leaf cluster selection methods
  ✓ Spectral: graph Laplacian embedding for non-convex cluster shapes
  ✓ Evaluation: 6 metrics (silhouette, DB, CH, gap, ARI, NMI)
  ✓ AutoMLEngine: double opt-in governance for LLM-guided selection
  ✓ Cluster profiling: z-score business interpretation of segments

  Next: In Exercise 2, you'll derive the EM algorithm from scratch and
  implement Gaussian Mixture Models with full E-step and M-step...
"""
)
