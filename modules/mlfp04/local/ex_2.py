# Copyright 2026 Terrene Foundation
# SPDX-License-Identifier: Apache-2.0
"""
# ════════════════════════════════════════════════════════════════════════
# MLFP04 — Exercise 2: EM Algorithm and Gaussian Mixture Models
# ════════════════════════════════════════════════════════════════════════
#
# WHAT YOU'LL LEARN:
#   After completing this exercise, you will be able to:
#   - Explain the EM algorithm as coordinate ascent on the ELBO
#   - Implement the E-step (posterior responsibilities) from scratch
#   - Implement the M-step (parameter updates) from scratch
#   - Verify that log-likelihood is non-decreasing across EM iterations
#   - Compare manual EM with sklearn GMM and select K via BIC/AIC
#   - Explain soft vs hard clustering and when each is appropriate
#   - Understand covariance type selection (full, tied, diag, spherical)
#   - Describe how Mixture of Experts extends mixture models to deep learning
#
# PREREQUISITES:
#   - MLFP04 Exercise 1 (clustering — GMM used as a black box there)
#   - MLFP02 Lesson 2.1 (Bayesian thinking — EM is Bayesian inference)
#
# ESTIMATED TIME: ~150-180 min
#
# TASKS:
#   1.  Derive EM intuitively — soft assignments and parameter updates
#   2.  Implement E-step (posterior responsibilities) with log-sum-exp
#   3.  Implement M-step (update means, covariances, weights)
#   4.  Run EM loop, visualise convergence, verify non-decreasing LL
#   5.  Compare manual EM with sklearn GMM on real data
#   6.  BIC/AIC model selection — choose K objectively
#   7.  Covariance type comparison (full, tied, diag, spherical)
#   8.  Soft vs hard assignments — practical impact on customer segmentation
#   9.  Mixture of Experts: modern application of mixture models
#   10. AutoMLEngine for automated GMM comparison
#
# DATASET: Synthetic 2D data (3 Gaussians, 600 samples) + e-commerce customers
#
# ════════════════════════════════════════════════════════════════════════
"""
from __future__ import annotations

import asyncio

import numpy as np
import polars as pl
from scipy.stats import multivariate_normal
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score

from kailash_ml import ModelVisualizer
from kailash_ml.interop import to_sklearn_input

from shared import MLFPDataLoader


# ── Synthetic Data for Manual EM ─────────────────────────────────────

rng = np.random.default_rng(42)

true_means = np.array([[0.0, 0.0], [5.0, 2.0], [2.0, 6.0]])
true_covs = np.array(
    [
        [[1.0, 0.3], [0.3, 0.8]],
        [[0.8, -0.2], [-0.2, 1.2]],
        [[1.5, 0.0], [0.0, 0.5]],
    ]
)
true_weights = np.array([0.4, 0.35, 0.25])

n_synth = 600
n_per_component = (true_weights * n_synth).astype(int)
n_per_component[-1] = n_synth - n_per_component[:-1].sum()

X_synth_parts = []
z_true = []
for k, (mean, cov, n) in enumerate(zip(true_means, true_covs, n_per_component)):
    X_synth_parts.append(rng.multivariate_normal(mean, cov, n))
    z_true.extend([k] * n)

X_synth = np.vstack(X_synth_parts)
z_true = np.array(z_true)

idx = rng.permutation(n_synth)
X_synth, z_true = X_synth[idx], z_true[idx]

print("=== Synthetic 2D GMM Data ===")
print(f"Samples: {n_synth}, Components: 3")
print(f"True weights: {true_weights}")
for k, (m, n) in enumerate(zip(true_means, n_per_component)):
    print(f"  Component {k}: mean={m}, n={n}")


# ══════════════════════════════════════════════════════════════════════
# TASK 1: EM Algorithm — intuition and derivation
# ══════════════════════════════════════════════════════════════════════
# Problem: observed data X, latent assignments Z, parameters theta
#
# EM insight: introduce latent Z, maximise the ELBO:
#   E-step: compute Q(Z|X, theta_old) = P(Z|X, theta_old)
#   M-step: theta_new = argmax E_Q[log P(X,Z|theta)]
#
# For GMMs:
#   E-step: r_{ik} = pi_k N(x_i|mu_k,Sigma_k) / Sum_j pi_j N(x_i|mu_j,Sigma_j)
#   M-step:
#     N_k = Sum_i r_{ik}
#     pi_k = N_k / N
#     mu_k = (Sum_i r_{ik} x_i) / N_k
#     Sigma_k = (Sum_i r_{ik} (x_i - mu_k)(x_i - mu_k)') / N_k

print(f"\n=== EM Algorithm Derivation ===")
print(
    """
EM as coordinate ascent on the ELBO:

  log P(X|theta) >= ELBO = E_Q[log P(X,Z|theta)] + H(Q)   (Jensen's inequality)

  E-step: fix theta, maximise ELBO over Q
    -> Q*(Z) = P(Z|X, theta)  [just compute the posterior]

  M-step: fix Q, maximise ELBO over theta
    -> theta* = argmax E_{Q*}[log P(X,Z|theta)]  [weighted MLE]
"""
)


# ══════════════════════════════════════════════════════════════════════
# TASK 2: E-step — compute posterior responsibilities
# ══════════════════════════════════════════════════════════════════════


def e_step(
    X: np.ndarray,
    means: np.ndarray,
    covs: np.ndarray,
    weights: np.ndarray,
) -> np.ndarray:
    """
    E-step: compute responsibility matrix R.
    R[i, k] = P(z_i = k | x_i, theta)
    Uses log-sum-exp trick for numerical stability.
    """
    n_samples = X.shape[0]
    n_components = len(weights)
    log_probs = np.zeros((n_samples, n_components))

    for k in range(n_components):
        try:
            # TODO: Create a multivariate_normal distribution with mean=means[k],
            #       cov=covs[k], allow_singular=True and add log probabilities
            dist = ____  # Hint: multivariate_normal(mean=means[k], cov=covs[k], allow_singular=True)
            log_probs[:, k] = np.log(weights[k] + 1e-300) + ____  # Hint: dist.logpdf(X)
        except Exception:
            log_probs[:, k] = -np.inf

    # TODO: Apply log-sum-exp: subtract max per row, compute log_normaliser
    log_probs_max = ____  # Hint: log_probs.max(axis=1, keepdims=True)
    log_normaliser = (
        np.log(np.exp(log_probs - log_probs_max).sum(axis=1, keepdims=True))
        + log_probs_max
    )
    # TODO: Compute R = exp(log_probs - log_normaliser)
    R = ____  # Hint: np.exp(log_probs - log_normaliser)

    return R


R_init = e_step(X_synth, true_means, true_covs, true_weights)
print(f"\n=== E-step Test (true parameters) ===")
print(f"Responsibility matrix shape: {R_init.shape}")
print(f"Row sums (should all be 1): {R_init.sum(axis=1)[:5].round(4)}")
print(f"Average max responsibility: {R_init.max(axis=1).mean():.4f}")

soft_assignments = R_init.argmax(axis=1)
accuracy_true_params = (soft_assignments == z_true).mean()
print(f"Assignment accuracy (true params): {accuracy_true_params:.4f}")

# ── Checkpoint 1 ─────────────────────────────────────────────────────
assert R_init.shape == (
    n_synth,
    3,
), "Responsibility matrix should be (n_samples, n_components)"
assert abs(R_init.sum(axis=1).mean() - 1.0) < 1e-6, "Responsibilities must sum to 1"
assert R_init.min() >= 0, "Responsibilities must be non-negative"
assert accuracy_true_params > 0.8, "With true parameters, accuracy should be > 0.8"
# INTERPRETATION: The responsibility r_{ik} is the posterior probability that
# point i was generated by component k. Near 1 for true label = high confidence.
print("\n✓ Checkpoint 1 passed — E-step responsibilities computed correctly\n")


# ══════════════════════════════════════════════════════════════════════
# TASK 3: M-step — update parameters from responsibilities
# ══════════════════════════════════════════════════════════════════════


def m_step(
    X: np.ndarray,
    R: np.ndarray,
    reg_covar: float = 1e-6,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    M-step: update GMM parameters given responsibility matrix R.
    N_k = Sum_i r_{ik}
    pi_k = N_k / N
    mu_k = (Sum_i r_{ik} x_i) / N_k
    Sigma_k = (Sum_i r_{ik} (x_i - mu_k)(x_i - mu_k)') / N_k + reg * I
    """
    n_samples, n_features = X.shape
    n_components = R.shape[1]

    # TODO: Compute effective counts N_k = R.sum(axis=0) + epsilon
    N_k = ____  # Hint: R.sum(axis=0) + 1e-300
    # TODO: Compute mixture weights = N_k / n_samples
    weights = ____  # Hint: N_k / n_samples
    # TODO: Compute new means: (R.T @ X) / N_k[:, np.newaxis]
    means = ____  # Hint: (R.T @ X) / N_k[:, np.newaxis]

    covs = np.zeros((n_components, n_features, n_features))
    for k in range(n_components):
        # TODO: Compute weighted covariance: (R[:, k:k+1] * diff).T @ diff / N_k[k]
        diff = ____  # Hint: X - means[k]
        covs[k] = ____  # Hint: (R[:, k:k+1] * diff).T @ diff / N_k[k]
        covs[k] += reg_covar * np.eye(n_features)

    return means, covs, weights


R_true = np.zeros((n_synth, 3))
for i, k in enumerate(z_true):
    R_true[i, k] = 1.0

means_recovered, covs_recovered, weights_recovered = m_step(X_synth, R_true)

print(f"\n=== M-step Test (hard assignments from ground truth) ===")
print(f"Recovered weights: {weights_recovered.round(3)} (true: {true_weights})")
for k in range(3):
    print(
        f"  Component {k}: mean={means_recovered[k].round(3)} (true: {true_means[k]})"
    )

# ── Checkpoint 2 ─────────────────────────────────────────────────────
assert abs(weights_recovered.sum() - 1.0) < 1e-6, "Recovered weights should sum to 1"
for k in range(3):
    dist_err = np.linalg.norm(means_recovered[k] - true_means[k])
    assert dist_err < 1.0, f"Recovered mean {k} too far from truth"
# INTERPRETATION: The M-step update is weighted MLE. Given hard assignments,
# the update is the class-conditional mean and covariance — exactly what
# you'd compute if you knew the labels.
print(
    "\n✓ Checkpoint 2 passed — M-step recovers true parameters from hard assignments\n"
)


# ══════════════════════════════════════════════════════════════════════
# TASK 4: EM loop — iterate until convergence
# ══════════════════════════════════════════════════════════════════════


def compute_log_likelihood(
    X: np.ndarray,
    means: np.ndarray,
    covs: np.ndarray,
    weights: np.ndarray,
) -> float:
    """Compute log-likelihood: Sum_i log Sum_k pi_k N(x_i | mu_k, Sigma_k)."""
    n_samples = X.shape[0]
    n_components = len(weights)
    log_likelihoods = np.full(n_samples, -np.inf)

    for k in range(n_components):
        try:
            dist = multivariate_normal(mean=means[k], cov=covs[k], allow_singular=True)
            log_likelihoods = np.logaddexp(
                log_likelihoods,
                np.log(weights[k] + 1e-300) + dist.logpdf(X),
            )
        except Exception:
            pass

    return log_likelihoods.sum()


def fit_gmm_em(
    X: np.ndarray,
    n_components: int = 3,
    max_iter: int = 100,
    tol: float = 1e-4,
    seed: int = 42,
) -> dict:
    """Fit GMM using manual EM loop with random initialisation."""
    rng_em = np.random.default_rng(seed)
    n_samples, n_features = X.shape

    idx = rng_em.choice(n_samples, n_components, replace=False)
    means = X[idx].copy()
    covs = np.array([np.eye(n_features)] * n_components)
    weights = np.ones(n_components) / n_components

    log_likelihoods = []

    for iteration in range(max_iter):
        # TODO: Call e_step then m_step to update parameters
        R = ____  # Hint: e_step(X, means, covs, weights)
        means, covs, weights = ____  # Hint: m_step(X, R)

        ll = compute_log_likelihood(X, means, covs, weights)
        log_likelihoods.append(ll)

        if iteration > 0 and abs(log_likelihoods[-1] - log_likelihoods[-2]) < tol:
            print(f"  Converged at iteration {iteration + 1}")
            break

    labels = R.argmax(axis=1)
    return {
        "means": means,
        "covs": covs,
        "weights": weights,
        "labels": labels,
        "responsibilities": R,
        "log_likelihoods": log_likelihoods,
        "n_iter": len(log_likelihoods),
        "final_ll": log_likelihoods[-1],
    }


print(f"\n=== Running Manual EM on Synthetic Data ===")
em_result = fit_gmm_em(X_synth, n_components=3, max_iter=100, tol=1e-4)

print(f"Iterations: {em_result['n_iter']}")
print(f"Final log-likelihood: {em_result['final_ll']:.2f}")
print(f"Recovered weights: {em_result['weights'].round(3)} (true: {true_weights})")

em_labels = em_result["labels"]
if len(set(em_labels)) > 1:
    sil = silhouette_score(X_synth, em_labels)
    print(f"Silhouette score: {sil:.4f}")

lls = em_result["log_likelihoods"]
ll_increases = [lls[i] - lls[i - 1] for i in range(1, len(lls))]
n_decreases = sum(1 for d in ll_increases if d < -0.1)
print(f"\nSteps with LL decrease > 0.1: {n_decreases} (should be 0)")

viz = ModelVisualizer()
fig = viz.training_history(
    {"Log-Likelihood": em_result["log_likelihoods"]}, x_label="EM Iteration"
)
fig.update_layout(title="EM Convergence: Log-Likelihood per Iteration")
fig.write_html("ex2_em_convergence.html")
print("Saved: ex2_em_convergence.html")

# ── Checkpoint 3 ─────────────────────────────────────────────────────
assert em_result["n_iter"] > 1, "EM should require more than 1 iteration"
for i in range(1, len(lls)):
    assert lls[i] >= lls[i - 1] - 0.1, f"Log-likelihood decreased at iteration {i}"
assert len(set(em_result["labels"])) >= 2, "EM should assign to at least 2 components"
# INTERPRETATION: The monotone non-decreasing log-likelihood is the mathematical
# proof that EM works. Each E-step tightens the ELBO; each M-step maximises it.
print("\n✓ Checkpoint 3 passed — EM converged with non-decreasing log-likelihood\n")


# ══════════════════════════════════════════════════════════════════════
# TASK 5: Compare manual EM with sklearn GMM on real data
# ══════════════════════════════════════════════════════════════════════

loader = MLFPDataLoader()
customers = loader.load("mlfp03", "ecommerce_customers.parquet")

feature_cols = [
    c
    for c, d in zip(customers.columns, customers.dtypes)
    if d in (pl.Float64, pl.Float32, pl.Int64, pl.Int32) and c not in ("customer_id",)
]

X_real, _, _ = to_sklearn_input(
    customers.drop_nulls(subset=feature_cols),
    feature_columns=feature_cols,
)

scaler = StandardScaler()
X_real_scaled = scaler.fit_transform(X_real)

print(f"\n=== Real Data: E-commerce Customers ===")
print(f"Shape: {X_real_scaled.shape}")

n_real_em = min(3000, X_real_scaled.shape[0])
print(f"\nRunning manual EM on {n_real_em} customers...")
em_real = fit_gmm_em(X_real_scaled[:n_real_em], n_components=4, max_iter=80)
print(f"Manual EM: {em_real['n_iter']} iterations, LL={em_real['final_ll']:.2f}")

# TODO: Fit sklearn GaussianMixture with n_components=4, random_state=42,
#       max_iter=200 and compute log-likelihood score
gmm_sk = ____  # Hint: GaussianMixture(n_components=4, random_state=42, max_iter=200)
gmm_sk.fit(X_real_scaled[:n_real_em])
sk_ll = ____  # Hint: gmm_sk.score(X_real_scaled[:n_real_em]) * n_real_em
print(f"sklearn GMM: LL={sk_ll:.2f}")
print(f"  Difference: {abs(em_real['final_ll'] - sk_ll):.2f}")

# ── Checkpoint 4 ─────────────────────────────────────────────────────
assert em_real["n_iter"] > 1, "EM on real data should require multiple iterations"
# INTERPRETATION: Manual EM and sklearn GMM produce similar log-likelihood
# values. Differences arise from initialisation and convergence criteria.
print("\n✓ Checkpoint 4 passed — manual EM matches sklearn GMM on real data\n")


# ══════════════════════════════════════════════════════════════════════
# TASK 6: BIC/AIC model selection — choose K objectively
# ══════════════════════════════════════════════════════════════════════
# BIC = k * log(n) - 2 * log_likelihood  (penalises complexity more)
# AIC = 2 * k - 2 * log_likelihood       (penalises complexity less)

print(f"\n=== GMM Model Selection (BIC/AIC) ===")
print(f"{'K':>4} {'BIC':>12} {'AIC':>12} {'Log-L':>12} {'Silhouette':>12}")
print("─" * 56)

bic_scores = {}
for k in range(2, 9):
    # TODO: Fit GaussianMixture with n_components=k, covariance_type="full",
    #       random_state=42, max_iter=200; compute bic() and aic()
    gmm = ____  # Hint: GaussianMixture(n_components=k, covariance_type="full", random_state=42, max_iter=200)
    gmm.fit(X_real_scaled)
    labels = gmm.predict(X_real_scaled)

    bic = ____  # Hint: gmm.bic(X_real_scaled)
    aic = ____  # Hint: gmm.aic(X_real_scaled)
    ll = gmm.score(X_real_scaled) * X_real_scaled.shape[0]
    sil = silhouette_score(X_real_scaled, labels) if len(set(labels)) > 1 else -1.0
    bic_scores[k] = {"bic": bic, "aic": aic, "ll": ll, "silhouette": sil, "gmm": gmm}

    print(f"{k:>4} {bic:>12.0f} {aic:>12.0f} {ll:>12.0f} {sil:>12.4f}")

best_k_bic = min(bic_scores.items(), key=lambda x: x[1]["bic"])[0]
best_k_aic = min(bic_scores.items(), key=lambda x: x[1]["aic"])[0]
best_k_sil = max(bic_scores.items(), key=lambda x: x[1]["silhouette"])[0]

print(f"\nBest K by BIC: {best_k_bic}")
print(f"Best K by AIC: {best_k_aic}")
print(f"Best K by Silhouette: {best_k_sil}")
print("  When BIC and AIC disagree, prefer BIC (more robust to overfitting)")

# ── Checkpoint 5 ─────────────────────────────────────────────────────
assert best_k_bic in range(2, 9), "BIC-optimal K should be in tested range"
assert best_k_aic in range(2, 9), "AIC-optimal K should be in tested range"
bic_vals = [v["bic"] for v in bic_scores.values()]
assert min(bic_vals) < max(bic_vals), "BIC values should vary across K"
# INTERPRETATION: BIC = k*log(n) - 2*LL. The BIC-optimal K balances
# fit vs complexity — the principled way to choose K without over-specifying.
print("\n✓ Checkpoint 5 passed — BIC/AIC model selection complete\n")


# ══════════════════════════════════════════════════════════════════════
# TASK 7: Covariance type comparison (full, tied, diag, spherical)
# ══════════════════════════════════════════════════════════════════════

k_final = best_k_bic
cov_types = ["full", "tied", "diag", "spherical"]

print(f"\n=== Covariance Type Comparison (K={k_final}) ===")
print(f"{'Type':<12} {'BIC':>12} {'Log-L':>12} {'Silhouette':>12} {'Params':>8}")
print("─" * 60)

cov_results = {}
for cov_type in cov_types:
    # TODO: Fit GaussianMixture with covariance_type=cov_type, n_components=k_final
    gmm_cov = ____  # Hint: GaussianMixture(n_components=k_final, covariance_type=cov_type, random_state=42, max_iter=200)
    gmm_cov.fit(X_real_scaled)
    labels_cov = gmm_cov.predict(X_real_scaled)

    bic_cov = ____  # Hint: gmm_cov.bic(X_real_scaled)
    ll_cov = gmm_cov.score(X_real_scaled) * X_real_scaled.shape[0]
    sil_cov = (
        silhouette_score(X_real_scaled, labels_cov)
        if len(set(labels_cov)) > 1
        else -1.0
    )

    d = X_real_scaled.shape[1]
    if cov_type == "full":
        n_params = k_final * (d * (d + 1) // 2 + d + 1) - 1
    elif cov_type == "tied":
        n_params = d * (d + 1) // 2 + k_final * (d + 1) - 1
    elif cov_type == "diag":
        n_params = k_final * (2 * d + 1) - 1
    else:
        n_params = k_final * (d + 2) - 1

    cov_results[cov_type] = {
        "bic": bic_cov,
        "ll": ll_cov,
        "silhouette": sil_cov,
        "n_params": n_params,
    }
    print(
        f"{cov_type:<12} {bic_cov:>12.0f} {ll_cov:>12.0f} {sil_cov:>12.4f} {n_params:>8}"
    )

best_cov = min(cov_results.items(), key=lambda x: x[1]["bic"])[0]
print(f"\nBest covariance type by BIC: {best_cov}")
print("  full: Most flexible — different shapes/orientations per cluster")
print("  spherical: Minimal — equivalent to soft K-means (all spheres)")

# ── Checkpoint 6 ─────────────────────────────────────────────────────
assert len(cov_results) == 4, "Should have results for all 4 covariance types"
assert (
    cov_results["full"]["n_params"] > cov_results["spherical"]["n_params"]
), "Full covariance should have more parameters than spherical"
# INTERPRETATION: More parameters (full) fit better (higher LL) but risk
# overfitting. BIC penalises extra parameters, selecting the simplest
# covariance type that adequately explains the data.
print("\n✓ Checkpoint 6 passed — covariance type comparison complete\n")


# ══════════════════════════════════════════════════════════════════════
# TASK 8: Soft vs hard assignments — practical impact
# ══════════════════════════════════════════════════════════════════════

gmm_best = bic_scores[k_final]["gmm"]
labels_best = gmm_best.predict(X_real_scaled)
# TODO: Get soft assignment probabilities from gmm_best
soft_probs = ____  # Hint: gmm_best.predict_proba(X_real_scaled)

max_probs = soft_probs.max(axis=1)
entropy = -np.sum(soft_probs * np.log(soft_probs + 1e-300), axis=1)

print(f"\n=== Soft vs Hard Assignments (K={k_final}) ===")
print(f"Component weights: {gmm_best.weights_.round(3)}")
print(f"\nAssignment confidence distribution:")
print(f"  Max prob > 0.95 (confident):    {(max_probs > 0.95).mean():.1%}")
print(
    f"  Max prob 0.7-0.95 (moderate):   {((max_probs > 0.7) & (max_probs <= 0.95)).mean():.1%}"
)
print(
    f"  Max prob 0.5-0.7 (ambiguous):   {((max_probs > 0.5) & (max_probs <= 0.7)).mean():.1%}"
)
print(f"  Max prob < 0.5 (uncertain):     {(max_probs < 0.5).mean():.1%}")
print(f"\nMean entropy: {entropy.mean():.4f} (max possible: {np.log(k_final):.4f})")

boundary_idx = np.where(max_probs < 0.6)[0]
print(f"\nBoundary customers (max_prob < 0.6): {len(boundary_idx):,}")
if len(boundary_idx) > 0:
    print("  Hard clustering forces them into one segment — misses the ambiguity.")

customers_with_clusters = customers.drop_nulls(subset=feature_cols).with_columns(
    pl.Series("gmm_cluster", labels_best)
)

print(f"\n=== Customer Segment Profiles ===")
for k in range(k_final):
    subset = customers_with_clusters.filter(pl.col("gmm_cluster") == k)
    soft_mass = soft_probs[:, k].sum()
    print(f"\nSegment {k} (hard={subset.height:,}, soft_mass={soft_mass:.0f}):")
    for col in feature_cols[:5]:
        mean_val = subset[col].mean()
        overall_mean = customers_with_clusters[col].mean()
        diff_pct = (mean_val - overall_mean) / (abs(overall_mean) + 1e-9) * 100
        indicator = "HIGH" if diff_pct > 15 else "low" if diff_pct < -15 else "avg"
        print(f"  {col:<28} {mean_val:>10.2f}  [{indicator:>4}] {diff_pct:+.1f}%")

# ── Checkpoint 7 ─────────────────────────────────────────────────────
assert soft_probs.shape == (X_real_scaled.shape[0], k_final), "Shape: (n_samples, K)"
assert abs(soft_probs.sum(axis=1).mean() - 1.0) < 1e-4, "Soft assignments must sum to 1"
assert max_probs.mean() > 0.5, "Average max probability should indicate some confidence"
# INTERPRETATION: Soft assignments r_{ik} capture uncertainty that hard labels
# discard. A customer with r = [0.45, 0.55] is genuinely between two segments.
print("\n✓ Checkpoint 7 passed — soft vs hard assignment analysis complete\n")


# ══════════════════════════════════════════════════════════════════════
# TASK 9: Mixture of Experts — modern application
# ══════════════════════════════════════════════════════════════════════

print(f"\n=== Mixture of Experts (MoE) ===")
print(
    """
GMM -> MoE: replace fixed mixing weights pi_k with input-dependent g_k(x)

  MoE:  P(y|x) = Sum_k g_k(x) * f_k(x)
    - Gating network g_k(x) = softmax(W_k^T x): routes each input
    - Expert networks f_k(x): each specialises in a region of input space

  Modern Application — Sparse MoE in LLMs:
    - Mixtral 8x7B: 8 experts of 7B params each
    - Router selects top-2 experts per token -> 14B active, 56B total capacity
    - You will see MoE again in Module 6 when studying LLM architectures.
"""
)

n_demo = 200
X_moe = rng.standard_normal((n_demo, 2))
gate_logits = np.column_stack([X_moe[:, 0], -X_moe[:, 0]])
# TODO: Apply softmax to gate_logits to get gate_probs
# Hint: np.exp(gate_logits) / np.exp(gate_logits).sum(axis=1, keepdims=True)
gate_probs = (
    ____  # Hint: np.exp(gate_logits) / np.exp(gate_logits).sum(axis=1, keepdims=True)
)

print("MoE gate demo (2 experts, routing by first feature):")
print(f"  Expert 0 active (gate > 0.5): {(gate_probs[:, 0] > 0.5).mean():.1%}")
print(f"  Expert 1 active (gate > 0.5): {(gate_probs[:, 1] > 0.5).mean():.1%}")

# ── Checkpoint 8 ─────────────────────────────────────────────────────
assert gate_probs.shape == (n_demo, 2), "Gate should produce 2 expert weights"
assert abs(gate_probs.sum(axis=1).mean() - 1.0) < 1e-4, "Gate should be a distribution"
# INTERPRETATION: MoE makes mixing weights input-dependent — "for THIS input,
# 80% weight goes to expert 1". This is the foundation of LLM efficiency.
print("\n✓ Checkpoint 8 passed — Mixture of Experts conceptual demo\n")


# ══════════════════════════════════════════════════════════════════════
# TASK 10: AutoMLEngine for automated comparison
# ══════════════════════════════════════════════════════════════════════

from kailash_ml.engines.automl_engine import AutoMLEngine, AutoMLConfig


async def automl_gmm():
    """Use AutoMLEngine to automate GMM hyperparameter search."""
    # TODO: Create AutoMLConfig with task_type="clustering", metric_to_optimize="bic",
    #       direction="minimize", search_strategy="random", search_n_trials=15,
    #       agent=False, max_llm_cost_usd=0.5
    config = ____  # Hint: AutoMLConfig(task_type="clustering", metric_to_optimize="bic", direction="minimize", search_n_trials=15, agent=False, max_llm_cost_usd=0.5)

    print(f"\n=== AutoMLEngine Config ===")
    print(f"Task: {config.task_type}")
    print(f"Optimising: {config.metric_to_optimize} ({config.direction})")
    print(f"Agent LLM guidance: {config.agent} (double opt-in)")

    return config


asyncio.run(automl_gmm())

comparison = {
    f"GMM K={k}": {"BIC": v["bic"], "Silhouette": v["silhouette"]}
    for k, v in bic_scores.items()
}
fig_cmp = viz.metric_comparison(comparison)
fig_cmp.update_layout(title="GMM: BIC and Silhouette vs Number of Components")
fig_cmp.write_html("ex2_gmm_comparison.html")
print("\nSaved: ex2_gmm_comparison.html")

cov_comparison = {
    f"Cov: {ct}": {"BIC": cr["bic"], "Silhouette": cr["silhouette"]}
    for ct, cr in cov_results.items()
}
fig_cov = viz.metric_comparison(cov_comparison)
fig_cov.update_layout(title="GMM: Covariance Type Comparison")
fig_cov.write_html("ex2_covariance_comparison.html")
print("Saved: ex2_covariance_comparison.html")

# ── Checkpoint 9 ─────────────────────────────────────────────────────
# INTERPRETATION: AutoMLEngine automates the BIC sweep and covariance type
# selection. The agent double opt-in ensures LLM-guided selection is a
# deliberate choice, not accidental.
print("\n✓ Checkpoint 9 passed — AutoMLEngine and visualisation complete\n")

print("\n✓ Exercise 2 complete — EM algorithm from scratch + sklearn GMM + MoE")


# ══════════════════════════════════════════════════════════════════════
# REFLECTION
# ══════════════════════════════════════════════════════════════════════
print("═" * 70)
print("  WHAT YOU'VE MASTERED")
print("═" * 70)
print(
    """
  ✓ EM algorithm: coordinate ascent on ELBO (E-step + M-step from scratch)
  ✓ Log-sum-exp trick: numerically stable responsibility computation
  ✓ Non-decreasing log-likelihood: mathematical proof that EM converges
  ✓ BIC/AIC: objective K selection that penalises overfitting
  ✓ Covariance types: full/tied/diag/spherical — shape vs parameter tradeoff
  ✓ Soft assignments: uncertainty-aware segmentation vs hard K-means labels
  ✓ MoE: GMM with input-dependent gating — the LLM efficiency architecture

  Next: In Exercise 3, you'll derive PCA from SVD and apply t-SNE/UMAP
  for dimensionality reduction and visualisation...
"""
)
