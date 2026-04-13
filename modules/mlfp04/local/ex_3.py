# Copyright 2026 Terrene Foundation
# SPDX-License-Identifier: Apache-2.0
"""
# ════════════════════════════════════════════════════════════════════════
# MLFP04 — Exercise 3: Dimensionality Reduction
# ════════════════════════════════════════════════════════════════════════
#
# WHAT YOU'LL LEARN:
#   After completing this exercise, you will be able to:
#   - Derive PCA from SVD and verify the connection numerically
#   - Read scree plots and choose n_components by variance threshold
#   - Interpret PCA loadings to understand what each PC represents
#   - Compute and interpret reconstruction error as a function of k
#   - Apply Kernel PCA for nonlinear dimensionality reduction
#   - Apply t-SNE with perplexity tuning for local structure
#   - Apply UMAP for global+local structure with out-of-sample transform
#   - Select between PCA, Kernel PCA, t-SNE, and UMAP based on use case
#   - Understand manifold learning methods (Isomap, LLE, MDS)
#
# PREREQUISITES:
#   - MLFP04 Exercise 1 (clustering — PCA is often used as preprocessing)
#   - MLFP02 Lesson 2.5 (linear regression and linear algebra concepts)
#
# ESTIMATED TIME: ~150-180 min
#
# TASKS:
#   1.  PCA via SVD — explained variance, verify against sklearn
#   2.  Scree plot and cumulative variance — choosing n_components
#   3.  PCA loadings — which features drive each principal component
#   4.  Reconstruction error as a function of retained components
#   5.  Kernel PCA — nonlinear extension with RBF and polynomial kernels
#   6.  t-SNE — local structure, perplexity hyperparameter
#   7.  UMAP — global structure, hyperparameter tuning, out-of-sample
#   8.  Method comparison — silhouette in embedding space
#   9.  Manifold learning reference (Isomap, LLE, MDS)
#   10. Intrinsic dimensionality estimation
#
# DATASET: E-commerce customer data (from MLFP03)
#   Features: 10+ numeric customer behaviour metrics
#   Goal: compress to 2-5 dimensions while retaining 90%+ variance
#
# ════════════════════════════════════════════════════════════════════════
"""
from __future__ import annotations

import time

import numpy as np
import polars as pl
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA, KernelPCA
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

from kailash_ml import ModelVisualizer
from kailash_ml.interop import to_sklearn_input

from shared import MLFPDataLoader

try:
    import umap as umap_lib

    UMAP_AVAILABLE = True
except ImportError:
    UMAP_AVAILABLE = False
    print("umap-learn not installed — UMAP tasks will use PCA fallback")


# ── Data Loading ──────────────────────────────────────────────────────

loader = MLFPDataLoader()
customers = loader.load("mlfp03", "ecommerce_customers.parquet")

feature_cols = [
    c
    for c, d in zip(customers.columns, customers.dtypes)
    if d in (pl.Float64, pl.Float32, pl.Int64, pl.Int32) and c not in ("customer_id",)
]

X_raw, _, col_info = to_sklearn_input(
    customers.drop_nulls(subset=feature_cols),
    feature_columns=feature_cols,
)

scaler = StandardScaler()
X = scaler.fit_transform(X_raw)

n_samples, n_features = X.shape
print(f"=== E-commerce Customer Data ===")
print(f"Samples: {n_samples:,}, Features: {n_features}")
print(f"Feature names: {feature_cols}")


# ══════════════════════════════════════════════════════════════════════
# TASK 1: PCA via SVD — the connection explained
# ══════════════════════════════════════════════════════════════════════
#   X = U S V'    (Singular Value Decomposition)
#   Columns of V = principal component directions (loadings)
#   Scores = U S  or  X V
#   Explained variance for PC_k = sigma_k^2 / (n - 1)

print(f"\n=== PCA via SVD ===")

# TODO: Compute SVD of X (full_matrices=False for economy SVD)
U, S, Vt = ____  # Hint: np.linalg.svd(X, full_matrices=False)

# TODO: Compute explained variance from singular values
# Explained variance for each component = S_k^2 / (n_samples - 1)
explained_variance = ____  # Hint: S**2 / (n_samples - 1)
total_variance = explained_variance.sum()
explained_variance_ratio = explained_variance / total_variance
cumulative_evr = np.cumsum(explained_variance_ratio)

print(f"Total variance (should be near n_features={n_features}): {total_variance:.2f}")
print(f"\nTop 10 Principal Components:")
print(
    f"{'PC':>4} {'Singular Value':>16} {'Expl. Var':>12} {'Expl. Var %':>12} {'Cumulative %':>14}"
)
print("─" * 62)
for i in range(min(10, n_features)):
    print(
        f"{i + 1:>4} {S[i]:>16.4f} {explained_variance[i]:>12.4f} "
        f"{explained_variance_ratio[i]:>11.2%} {cumulative_evr[i]:>13.2%}"
    )

# TODO: Verify SVD result against sklearn PCA
# Create a full PCA (n_components=n_features), fit on X, check max absolute difference
pca_full = ____  # Hint: PCA(n_components=n_features)
pca_full.fit(X)
max_diff = np.abs(pca_full.explained_variance_ratio_ - explained_variance_ratio).max()
print(f"\nSVD vs sklearn PCA max difference: {max_diff:.2e} (should be near 0)")

# ── Checkpoint 1 ─────────────────────────────────────────────────────
assert max_diff < 1e-6, f"SVD and sklearn PCA should agree, got diff {max_diff:.2e}"
assert (
    abs(total_variance - n_features) < 0.5
), f"Total variance of standardised data should be near n_features ({n_features}), got {total_variance:.2f}"
assert (
    abs(cumulative_evr[-1] - 1.0) < 1e-6
), "Cumulative explained variance should sum to 1.0"
# INTERPRETATION: PCA via SVD is exact — sklearn's implementation is numerically
# identical. The right singular vectors V^T are the principal directions,
# and sigma_k^2/(n-1) are the eigenvalues of the covariance matrix.
print("\n✓ Checkpoint 1 passed — PCA via SVD verified against sklearn\n")


# ══════════════════════════════════════════════════════════════════════
# TASK 2: Scree plot and choosing n_components
# ══════════════════════════════════════════════════════════════════════

# TODO: Find component counts for 95%, 90%, 80% variance thresholds
# np.searchsorted returns the index where cumulative_evr first reaches threshold
n_95 = ____  # Hint: int(np.searchsorted(cumulative_evr, 0.95) + 1)
n_90 = ____  # Hint: int(np.searchsorted(cumulative_evr, 0.90) + 1)
n_80 = ____  # Hint: int(np.searchsorted(cumulative_evr, 0.80) + 1)

print(f"=== Variance Thresholds ===")
print(f"Components for 80% variance: {n_80}")
print(f"Components for 90% variance: {n_90}")
print(f"Components for 95% variance: {n_95}")
print(
    f"  (Original: {n_features} features -> {n_95} for 95% = {n_features / n_95:.1f}x compression)"
)

# Kaiser criterion: retain components with eigenvalue > 1
n_kaiser = int((explained_variance > 1.0).sum())
print(f"\nKaiser criterion (eigenvalue > 1): {n_kaiser} components")
print(f"  Eigenvalue > 1 means the component captures more than one original feature")

# Broken-stick model: compare against random partition of total variance
broken_stick = np.array(
    [
        sum(1.0 / j for j in range(i, n_features + 1)) / n_features
        for i in range(1, n_features + 1)
    ]
)
n_broken_stick = int((explained_variance_ratio > broken_stick).sum())
print(f"Broken-stick criterion: {n_broken_stick} components")
print(f"  Retains components that explain more than a random fragmentation")

# ── Checkpoint 2 ─────────────────────────────────────────────────────
assert n_95 <= n_features, "95% threshold should require fewer components than features"
assert n_90 <= n_95, "90% requires fewer components than 95%"
assert n_80 <= n_90, "80% requires fewer components than 90%"
assert 1 <= n_kaiser <= n_features, "Kaiser should select at least 1 component"
# INTERPRETATION: The scree plot elbow shows diminishing returns. For most ML
# tasks, retaining 90-95% variance is better than 100%: the remaining 5-10%
# is often noise, and removing it improves generalisation (denoising effect).
print("\n✓ Checkpoint 2 passed — variance thresholds and 3 selection criteria\n")

viz = ModelVisualizer()

# Scree plot
fig_scree = viz.training_history(
    {
        "Explained Variance %": (explained_variance_ratio[:20] * 100).tolist(),
        "Cumulative %": (cumulative_evr[:20] * 100).tolist(),
    },
    x_label="Principal Component",
)
fig_scree.update_layout(title="Scree Plot: Explained Variance by Component")
fig_scree.write_html("ex3_scree_plot.html")
print("Saved: ex3_scree_plot.html")


# ══════════════════════════════════════════════════════════════════════
# TASK 3: PCA loadings — feature contributions to each PC
# ══════════════════════════════════════════════════════════════════════
# Loadings: rows of V^T (= principal component directions in feature space)
# |Loading| close to 1: feature j strongly drives PC_k
# |Loading| close to 0: feature j barely contributes to PC_k

n_pcs_inspect = min(5, n_features)

# TODO: Extract the loadings matrix from Vt
# Loadings shape: (n_features, n_pcs_inspect)
# Vt has shape (n_features, n_features); rows are PC directions
loadings = ____  # Hint: Vt[:n_pcs_inspect, :].T  — transpose to get (n_features, n_pcs)

print(f"\n=== PCA Loadings (top {n_pcs_inspect} PCs) ===")
print(f"{'Feature':<30}", end="")
for i in range(n_pcs_inspect):
    print(f"{'PC' + str(i + 1):>10}", end="")
print()
print("─" * (30 + 10 * n_pcs_inspect))

for j, feat in enumerate(feature_cols):
    print(f"{feat:<30}", end="")
    for i in range(n_pcs_inspect):
        val = loadings[j, i]
        marker = " *" if abs(val) > 0.4 else "  "
        print(f"{val:>9.3f}{marker[1]}", end="")
    print()

print("\n* = strong loading (|loading| > 0.4)")

# Identify dominant features per PC
print("\nDominant features per principal component:")
for i in range(n_pcs_inspect):
    abs_loadings = np.abs(loadings[:, i])
    top_idx = np.argsort(abs_loadings)[::-1][:3]
    top_feats = [(feature_cols[j], loadings[j, i]) for j in top_idx]
    signs = [f"{feat} ({val:+.3f})" for feat, val in top_feats]
    print(f"  PC{i + 1}: {', '.join(signs)}")

# ── Checkpoint 3 ─────────────────────────────────────────────────────
assert loadings.shape == (n_features, n_pcs_inspect), "Loadings shape mismatch"
for i in range(n_pcs_inspect):
    col_norm = np.linalg.norm(loadings[:, i])
    assert abs(col_norm - 1.0) < 1e-6, f"PC{i + 1} loading should be unit length"
# INTERPRETATION: PCA loadings reveal the 'recipe' for each principal component.
# A loading of 0.7 for 'total_spend' on PC1 means PC1 is mostly driven by
# spending behaviour. This is how you assign business meaning to PCs.
print(
    "\n✓ Checkpoint 3 passed — PCA loadings are unit-norm, dominant features identified\n"
)


# ══════════════════════════════════════════════════════════════════════
# TASK 4: Reconstruction error as a function of retained components
# ══════════════════════════════════════════════════════════════════════
# Reconstruction: X_hat = X_proj @ Vt[:k, :]
# Reconstruction MSE = ||X - X_hat||^2 / (n * p) = unexplained variance

n_components_range = list(range(1, min(n_features + 1, 31)))
reconstruction_errors = []

for k in n_components_range:
    # TODO: Create PCA with k components, project X to low dim, then reconstruct
    pca_k = ____  # Hint: PCA(n_components=k)
    X_proj = pca_k.fit_transform(X)
    X_recon = ____  # Hint: pca_k.inverse_transform(X_proj)
    mse = np.mean((X - X_recon) ** 2)
    reconstruction_errors.append(mse)

print(f"\n=== Reconstruction Error ===")
print(f"{'Components':>12} {'MSE':>12} {'% Variance Retained':>22}")
print("─" * 50)
for k, mse in zip(n_components_range[::3], reconstruction_errors[::3]):
    pct_retained = 1.0 - mse / np.mean(X**2)
    print(f"{k:>12} {mse:>12.4f} {pct_retained:>21.2%}")

# Verify: reconstruction error = sum of discarded eigenvalues / (n*p)
for k in [1, 5, n_90]:
    theoretical_mse = explained_variance[k:].sum() / n_features
    actual_mse = reconstruction_errors[k - 1]
    print(
        f"\n  k={k}: theoretical MSE={theoretical_mse:.4f}, actual={actual_mse:.4f}, "
        f"diff={abs(theoretical_mse - actual_mse):.6f}"
    )

fig_recon = viz.training_history(
    {"Reconstruction MSE": reconstruction_errors},
    x_label="Number of PCA Components",
)
fig_recon.update_layout(title="PCA: Reconstruction Error vs Components Retained")
fig_recon.write_html("ex3_reconstruction_error.html")
print("\nSaved: ex3_reconstruction_error.html")

# ── Checkpoint 4 ─────────────────────────────────────────────────────
assert (
    reconstruction_errors[0] > reconstruction_errors[-1]
), "Reconstruction error should decrease with more components"
assert (
    reconstruction_errors[-1] < 0.1
), "With all components, reconstruction error should be near zero"
# INTERPRETATION: Reconstruction error = sum of unexplained variance. With k
# components: error = Sum_{i > k} sigma_i^2 / (n*p). The curve is the
# complement of the scree plot. Sweet spot: additional components add little.
print("\n✓ Checkpoint 4 passed — reconstruction error decreases with more components\n")


# ══════════════════════════════════════════════════════════════════════
# TASK 5: Kernel PCA — nonlinear dimensionality reduction
# ══════════════════════════════════════════════════════════════════════
# Standard PCA finds linear principal components. Kernel PCA applies the
# kernel trick to find nonlinear manifolds:
#   1. Map data to high-dimensional feature space via kernel K(x_i, x_j)
#   2. Run PCA in that space (without explicitly computing the mapping)
# Kernels: RBF (Gaussian), polynomial, sigmoid

n_for_embedding = n_90
print(f"\n=== Kernel PCA (n_components={n_for_embedding}) ===")

# Subsample for kernel PCA (kernel matrix is n x n)
rng = np.random.default_rng(42)
n_kpca = min(3000, n_samples)
idx_kpca = rng.choice(n_samples, n_kpca, replace=False)
X_kpca_input = X[idx_kpca]

kernel_results = {}
for kernel_name, kernel_params in [
    ("linear", {}),
    ("rbf", {"gamma": 0.1}),
    ("rbf", {"gamma": 1.0}),
    ("poly", {"degree": 3, "gamma": 0.1}),
]:
    label = f"{kernel_name}" + (
        f" (gamma={kernel_params.get('gamma', '')})" if "gamma" in kernel_params else ""
    )
    if "degree" in kernel_params:
        label = f"{kernel_name} (deg={kernel_params['degree']})"

    t0 = time.time()
    # TODO: Create KernelPCA with the given kernel and params, fit_transform X_kpca_input
    kpca = ____  # Hint: KernelPCA(n_components=min(n_for_embedding, 10), kernel=kernel_name, random_state=42, **kernel_params)
    X_kpca = ____  # Hint: kpca.fit_transform(X_kpca_input)
    elapsed = time.time() - t0

    # Cluster in reduced space and measure quality
    km = KMeans(n_clusters=4, random_state=42, n_init=5)
    labels_kpca = km.fit_predict(X_kpca)
    sil = silhouette_score(X_kpca, labels_kpca)

    kernel_results[label] = {
        "silhouette": sil,
        "time": elapsed,
        "n_components": X_kpca.shape[1],
    }
    print(f"  {label:<30}: sil={sil:.4f}, time={elapsed:.2f}s, dims={X_kpca.shape[1]}")

print("\nKernel PCA guide:")
print("  linear: equivalent to standard PCA")
print("  rbf:    captures radial nonlinearity (gamma controls width)")
print("  poly:   captures polynomial interactions (degree controls complexity)")
print("  Small gamma = wide kernel = smoother manifold")
print("  Large gamma = narrow kernel = more local/complex manifold")

# ── Checkpoint 5 ─────────────────────────────────────────────────────
assert len(kernel_results) >= 3, "Should test at least 3 kernel configurations"
# INTERPRETATION: Kernel PCA extends PCA to find nonlinear manifolds. The RBF
# kernel creates a smooth nonlinear embedding; polynomial captures interactions.
# However, Kernel PCA has no inverse_transform (cannot reconstruct) and the
# kernel matrix is O(n^2) in memory, limiting scalability.
print("\n✓ Checkpoint 5 passed — Kernel PCA with 3+ kernel configurations\n")

# Pre-reduce with PCA for t-SNE/UMAP
pca_pre = PCA(n_components=n_for_embedding, random_state=42)
X_pca = pca_pre.fit_transform(X)
print(f"Pre-reduced for t-SNE/UMAP: {X.shape} -> {X_pca.shape}")


# ══════════════════════════════════════════════════════════════════════
# TASK 6: t-SNE — local structure, perplexity
# ══════════════════════════════════════════════════════════════════════
# t-SNE minimises KL divergence between high-dim and low-dim similarities.
# Properties: preserves LOCAL structure, NOT global distances.
# No out-of-sample extension. O(n log n) with Barnes-Hut.
#
# Perplexity: effective number of nearest neighbours (5-50 typical)

n_tsne = min(3000, n_samples)
idx_tsne = rng.choice(n_samples, n_tsne, replace=False)
X_tsne_input = X_pca[idx_tsne]

print(f"\n=== t-SNE (n={n_tsne}) ===")
print(f"{'Perplexity':>12} {'KL Divergence':>16} {'Silhouette':>12} {'Time':>8}")
print("─" * 52)

tsne_results = {}
for perplexity in [5, 15, 30, 50]:
    t0 = time.time()
    # TODO: Create TSNE with 2 components, given perplexity, max_iter=1000, init="pca"
    tsne = ____  # Hint: TSNE(n_components=2, perplexity=perplexity, max_iter=1000, random_state=42, init="pca", learning_rate="auto")
    embedding = ____  # Hint: tsne.fit_transform(X_tsne_input)
    elapsed = time.time() - t0

    km = KMeans(n_clusters=4, random_state=42, n_init=5)
    labels_2d = km.fit_predict(embedding)
    sil = silhouette_score(embedding, labels_2d) if len(set(labels_2d)) > 1 else -1.0

    tsne_results[perplexity] = {
        "embedding": embedding,
        "kl_divergence": tsne.kl_divergence_,
        "silhouette": sil,
    }
    print(
        f"{perplexity:>12} {tsne.kl_divergence_:>16.4f} {sil:>12.4f} {elapsed:>7.1f}s"
    )

print("\nt-SNE perplexity guidance:")
print("  perplexity=5 : micro-clusters, tight local groups")
print("  perplexity=15: fine-grained local structure")
print("  perplexity=30: balanced (default recommendation)")
print("  perplexity=50: smoother, fewer isolated clusters")

print("\nt-SNE pitfalls (critical to understand):")
print("  1. Cluster SIZES in t-SNE do NOT reflect real cluster sizes")
print("  2. Distances BETWEEN clusters are NOT meaningful")
print("  3. Different runs give different layouts (stochastic)")
print("  4. No out-of-sample extension — new points require full refit")
print("  5. Do NOT use t-SNE output as features for downstream models")

# ── Checkpoint 6 ─────────────────────────────────────────────────────
assert len(tsne_results) == 4, "Should test t-SNE at 4 perplexity values"
for perp, res in tsne_results.items():
    assert res["embedding"].shape[1] == 2, "t-SNE should produce 2D embeddings"
    assert res["kl_divergence"] > 0, "KL divergence should be positive"
# INTERPRETATION: KL divergence measures how well the 2D embedding preserves
# the high-dimensional neighbourhood structure. Lower is better, but
# always visually inspect — low KL with high perplexity may mean collapse.
print("\n✓ Checkpoint 6 passed — t-SNE computed at 4 perplexity levels\n")


# ══════════════════════════════════════════════════════════════════════
# TASK 7: UMAP — global structure, hyperparameter tuning, out-of-sample
# ══════════════════════════════════════════════════════════════════════
# UMAP uses fuzzy topological representation:
#   1. Build weighted k-NN graph (high-dimensional)
#   2. Optimise low-dimensional layout to match graph structure
#
# Advantages over t-SNE:
#   - Preserves BOTH local AND global structure
#   - Supports out-of-sample transform (critical for production)
#   - Faster: O(n) amortised
#   - Can embed into any dimensionality (not just 2D)

print(f"\n=== UMAP Hyperparameter Comparison ===")

if UMAP_AVAILABLE:
    umap_configs = [
        {"n_neighbors": 5, "min_dist": 0.1, "label": "local (n=5, d=0.1)"},
        {"n_neighbors": 15, "min_dist": 0.1, "label": "default (n=15, d=0.1)"},
        {"n_neighbors": 30, "min_dist": 0.1, "label": "broad (n=30, d=0.1)"},
        {"n_neighbors": 15, "min_dist": 0.0, "label": "tight (n=15, d=0.0)"},
        {"n_neighbors": 15, "min_dist": 0.5, "label": "spread (n=15, d=0.5)"},
        {"n_neighbors": 50, "min_dist": 0.5, "label": "global (n=50, d=0.5)"},
    ]

    umap_results = {}
    for cfg in umap_configs:
        t0 = time.time()
        # TODO: Create UMAP reducer with given n_neighbors, min_dist, random_state=42
        reducer = ____  # Hint: umap_lib.UMAP(n_components=2, n_neighbors=cfg["n_neighbors"], min_dist=cfg["min_dist"], random_state=42, metric="euclidean")
        # TODO: Fit on subsample (idx_tsne), then transform full dataset for out-of-sample
        reducer.fit(____)  # Hint: fit on X_pca[idx_tsne]
        embedding_full = ____  # Hint: reducer.transform(X_pca)
        elapsed = time.time() - t0

        km_labels = KMeans(n_clusters=4, random_state=42, n_init=5).fit_predict(
            embedding_full
        )
        sil = (
            silhouette_score(embedding_full, km_labels)
            if len(set(km_labels)) > 1
            else -1.0
        )

        umap_results[cfg["label"]] = {
            "embedding": embedding_full,
            "silhouette": sil,
            "time": elapsed,
        }
        print(f"  {cfg['label']:<30}: sil={sil:.4f}, time={elapsed:.1f}s")

    print("\nUMAP hyperparameters:")
    print("  n_neighbors: size of local neighbourhood (like perplexity)")
    print("    Small -> fine local detail, Large -> more global structure")
    print("  min_dist: minimum distance between points in embedding")
    print("    Small (0.0) -> tight clusters, Large (1.0) -> spread out")
    print("\nOut-of-sample: reducer.transform(new_X) — production-ready!")

else:
    pca_2d = PCA(n_components=2, random_state=42)
    embedding_2d = pca_2d.fit_transform(X_pca)
    km_labels = KMeans(n_clusters=4, random_state=42, n_init=5).fit_predict(
        embedding_2d
    )
    sil = silhouette_score(embedding_2d, km_labels) if len(set(km_labels)) > 1 else -1.0
    umap_results = {
        "PCA 2D": {"embedding": embedding_2d, "silhouette": sil, "time": 0.0}
    }
    print(f"  PCA 2D fallback: sil={sil:.4f}")
    print("\nInstall umap-learn for UMAP: pip install umap-learn")

# ── Checkpoint 7 ─────────────────────────────────────────────────────
assert len(umap_results) >= 1, "Should have at least one UMAP/PCA result"
# INTERPRETATION: UMAP preserves both local and global structure, unlike t-SNE
# which only preserves local structure. The out-of-sample transform makes
# UMAP suitable for production — new customers can be embedded without refitting.
print("\n✓ Checkpoint 7 passed — UMAP with hyperparameter comparison\n")


# ══════════════════════════════════════════════════════════════════════
# TASK 8: Method comparison — silhouette in embedding space
# ══════════════════════════════════════════════════════════════════════

method_silhouettes = {}

# PCA embeddings at different dimensions
for n_comp in [2, n_80, n_90, n_95]:
    pca_test = PCA(n_components=n_comp, random_state=42)
    X_test = pca_test.fit_transform(X)
    km_l = KMeans(n_clusters=4, random_state=42, n_init=5).fit_predict(X_test)
    method_silhouettes[f"PCA {n_comp}d"] = {
        "Silhouette": silhouette_score(X_test, km_l)
    }

# t-SNE embeddings
for perp, res in tsne_results.items():
    km_l = KMeans(n_clusters=4, random_state=42, n_init=5).fit_predict(res["embedding"])
    method_silhouettes[f"t-SNE p={perp}"] = {
        "Silhouette": silhouette_score(res["embedding"], km_l)
    }

# UMAP embeddings
for label, res in umap_results.items():
    method_silhouettes[f"UMAP {label}"] = {"Silhouette": res["silhouette"]}

# Kernel PCA
for label, res in kernel_results.items():
    method_silhouettes[f"KernelPCA {label}"] = {"Silhouette": res["silhouette"]}

print(f"\n=== Method Comparison (4-cluster silhouette) ===")
for name, metrics in sorted(
    method_silhouettes.items(), key=lambda x: -x[1]["Silhouette"]
):
    print(f"  {name:<35}: {metrics['Silhouette']:.4f}")

fig_methods = viz.metric_comparison(
    {k: v for k, v in list(method_silhouettes.items())[:12]}
)
fig_methods.update_layout(title="Dimensionality Reduction: Cluster Quality Comparison")
fig_methods.write_html("ex3_method_comparison.html")
print("\nSaved: ex3_method_comparison.html")

# ── Checkpoint 8 ─────────────────────────────────────────────────────
assert len(method_silhouettes) >= 6, "Should compare at least 6 method configurations"
# INTERPRETATION: Silhouette in the embedding space measures how well the
# dimensionality reduction preserves cluster separability. However, high
# silhouette in 2D t-SNE does NOT mean the clusters are real — t-SNE can
# create visual artefacts. Always validate with original-space metrics.
print("\n✓ Checkpoint 8 passed — comprehensive method comparison\n")


# ══════════════════════════════════════════════════════════════════════
# TASK 9: Manifold learning reference (Isomap, LLE, MDS)
# ══════════════════════════════════════════════════════════════════════

print(f"\n=== Manifold Learning Reference ===")
print(
    f"""
Beyond PCA, t-SNE, and UMAP, other manifold learning methods exist:

┌──────────────────┬───────────────────────────────────────────────────────────┐
│ Method           │ Key Idea                                                  │
├──────────────────┼───────────────────────────────────────────────────────────┤
│ Isomap           │ Geodesic distances on k-NN graph (preserves distances    │
│                  │ along the manifold, not straight-line Euclidean)          │
│                  │ Best for: data on a smooth manifold (e.g., Swiss roll)    │
├──────────────────┼───────────────────────────────────────────────────────────┤
│ LLE              │ Locally Linear Embedding: reconstruct each point as      │
│                  │ weighted sum of neighbours, preserve weights in low-dim   │
│                  │ Best for: locally linear manifolds                        │
├──────────────────┼───────────────────────────────────────────────────────────┤
│ MDS              │ Multidimensional Scaling: preserve pairwise distances     │
│                  │ directly (stress minimisation)                            │
│                  │ Best for: distance matrices (e.g., survey dissimilarity)  │
├──────────────────┼───────────────────────────────────────────────────────────┤
│ Spectral Embed.  │ Graph Laplacian eigenvectors (same as spectral clustering│
│                  │ embedding step, without the final K-means)               │
│                  │ Best for: graph-structured data                          │
└──────────────────┴───────────────────────────────────────────────────────────┘

In practice, t-SNE (visualisation) and UMAP (visualisation + production) have
largely superseded these methods for most applications. Use Isomap/MDS when
you specifically need geodesic distance preservation or have a distance matrix.
"""
)

# Brief Isomap demo on subsample
from sklearn.manifold import Isomap

n_iso = min(2000, n_samples)
# TODO: Create Isomap with 2 components and 10 neighbours, fit_transform on subsample
iso = ____  # Hint: Isomap(n_components=2, n_neighbors=10)
X_iso = ____  # Hint: iso.fit_transform(X_pca[idx_tsne[:n_iso]])
km_iso = KMeans(n_clusters=4, random_state=42, n_init=5).fit_predict(X_iso)
sil_iso = silhouette_score(X_iso, km_iso)
print(f"Isomap demo (n={n_iso}): silhouette={sil_iso:.4f}")

# ── Checkpoint 9 ─────────────────────────────────────────────────────
assert X_iso.shape == (n_iso, 2), "Isomap should produce 2D embedding"
# INTERPRETATION: Isomap preserves geodesic distances — distances measured
# along the data manifold rather than straight through the ambient space.
# For data that lies on a curved surface (like a Swiss roll), Isomap
# "unrolls" the surface while t-SNE only preserves local neighbours.
print("\n✓ Checkpoint 9 passed — manifold learning reference with Isomap demo\n")


# ══════════════════════════════════════════════════════════════════════
# TASK 10: Intrinsic dimensionality estimation
# ══════════════════════════════════════════════════════════════════════
# How many dimensions does the data actually need? Three approaches:
#   1. PCA: cumulative variance >= threshold
#   2. Kaiser: eigenvalue > 1
#   3. MLE-based: maximum likelihood estimation of intrinsic dimension

print(f"\n=== Intrinsic Dimensionality Estimation ===")

# Approach 1: PCA thresholds (already computed)
print(f"PCA-based estimates:")
print(f"  80% variance: {n_80} dimensions")
print(f"  90% variance: {n_90} dimensions")
print(f"  95% variance: {n_95} dimensions")
print(f"  Kaiser (eigenvalue > 1): {n_kaiser} dimensions")
print(f"  Broken-stick: {n_broken_stick} dimensions")


# Approach 2: Correlation dimension (nearest-neighbour based)
# Use the growth rate of the number of neighbours within radius r
def estimate_intrinsic_dim_nn(
    X: np.ndarray, k_values: list[int], n_subsample: int = 1000
) -> float:
    """Estimate intrinsic dimension via nearest-neighbour distance scaling."""
    rng_dim = np.random.default_rng(42)
    idx = rng_dim.choice(len(X), min(n_subsample, len(X)), replace=False)
    X_sub = X[idx]

    from sklearn.neighbors import NearestNeighbors

    nn = NearestNeighbors(n_neighbors=max(k_values))
    nn.fit(X_sub)
    distances, _ = nn.kneighbors(X_sub)

    # MLE estimator: d_hat = 1 / mean(log(d_k / d_1))
    log_ratios = []
    for k in k_values:
        d_k = distances[:, k - 1]
        d_1 = distances[:, 0]
        valid = (d_k > 0) & (d_1 > 0)
        if valid.sum() > 10:
            ratio = np.log(d_k[valid] / d_1[valid])
            log_ratios.append(np.mean(ratio))

    if log_ratios:
        mean_log_ratio = np.mean(log_ratios)
        if mean_log_ratio > 0:
            return 1.0 / mean_log_ratio
    return float("nan")


# TODO: Call estimate_intrinsic_dim_nn on X with k_values=[5, 10, 20, 30]
intrinsic_dim_nn = ____  # Hint: estimate_intrinsic_dim_nn(X, k_values=[5, 10, 20, 30])
print(f"\nNearest-neighbour MLE estimate: {intrinsic_dim_nn:.1f} dimensions")

# Summary
print(f"\nDimensionality summary:")
print(f"  Original features: {n_features}")
print(f"  PCA 90% variance: {n_90}")
print(f"  Kaiser criterion: {n_kaiser}")
print(f"  NN MLE estimate:  {intrinsic_dim_nn:.1f}")
print(f"  Practical recommendation: use {n_90} dimensions for downstream ML")

# ── Checkpoint 10 ────────────────────────────────────────────────────
assert n_90 <= n_features, "Intrinsic dimension should be <= ambient dimension"
# INTERPRETATION: Intrinsic dimensionality tells you how many independent
# axes of variation the data actually has. If 20 features have intrinsic
# dimension 5, then 15 features are redundant (linear combinations of the
# other 5). This is why PCA works — most real data is lower-dimensional.
print("\n✓ Checkpoint 10 passed — intrinsic dimensionality estimated\n")


# ══════════════════════════════════════════════════════════════════════
# Summary table
# ══════════════════════════════════════════════════════════════════════
print(
    f"""
┌──────────────────┬─────────────┬──────────────┬───────────────┬──────────────┐
│ Method           │ Linear?     │ Global Struct│ Out-of-Sample │ Speed        │
├──────────────────┼─────────────┼──────────────┼───────────────┼──────────────┤
│ PCA              │ Yes         │ Yes          │ Yes           │ O(np min(n,p))│
│ Kernel PCA       │ No          │ Partial      │ Approx        │ O(n^2 p)     │
│ t-SNE            │ No          │ Local only   │ No            │ O(n log n)   │
│ UMAP             │ No          │ Both         │ Yes           │ O(n)         │
│ Isomap           │ No          │ Geodesic     │ Yes           │ O(n^2 log n) │
└──────────────────┴─────────────┴──────────────┴───────────────┴──────────────┘
"""
)

print("\n✓ Exercise 3 complete — PCA, Kernel PCA, t-SNE, UMAP, manifold learning")


# ══════════════════════════════════════════════════════════════════════
# REFLECTION
# ══════════════════════════════════════════════════════════════════════
print("═" * 70)
print("  WHAT YOU'VE MASTERED")
print("═" * 70)
print(
    f"""
  ✓ PCA via SVD: X = U Sigma V^T; PC directions = rows of V^T
  ✓ Scree plot: choose k where cumulative variance >= 90%
  ✓ 3 selection criteria: Kaiser, broken-stick, variance threshold
  ✓ Loadings: unit vectors showing which features drive each PC
  ✓ Reconstruction error: compression quality without labels
  ✓ Kernel PCA: RBF/poly kernels for nonlinear manifolds
  ✓ t-SNE: local structure, perplexity tuning, pitfalls
  ✓ UMAP: local + global, out-of-sample transform, production-ready
  ✓ Manifold learning: Isomap, LLE, MDS reference
  ✓ Intrinsic dimensionality: NN MLE estimate

  DECISION GUIDE:
    PCA        -> production (fast, invertible, interpretable)
    Kernel PCA -> nonlinear features when PCA is insufficient
    t-SNE      -> exploratory visualisation only (NEVER as features)
    UMAP       -> visualisation + feature extraction (supports transform)

  NEXT: Exercise 4 uses anomaly detection to find fraudulent transactions.
  You'll blend multiple detectors (IsolationForest + LOF + Z-score) using
  EnsembleEngine.blend() to beat any single method.
"""
)
print("═" * 70)
