# Copyright 2026 Terrene Foundation
# SPDX-License-Identifier: Apache-2.0
"""
# ════════════════════════════════════════════════════════════════════════
# MLFP03 — Exercise 2: Bias-Variance, Regularisation, and
#                        Cross-Validation
# ════════════════════════════════════════════════════════════════════════
#
# WHAT YOU'LL LEARN:
#   After completing this exercise, you will be able to:
#   - Demonstrate underfitting vs overfitting through polynomial experiments
#   - Decompose expected test error into Bias², Variance, and irreducible
#     noise via bootstrap
#   - Apply L1 (Lasso) and L2 (Ridge) regularisation and explain the
#     geometry (diamond vs circle constraint)
#   - Connect L2 regularisation to Bayesian Gaussian priors on coefficients
#   - Combine L1 and L2 with ElasticNet and select the mixing ratio
#   - Implement nested cross-validation for unbiased model selection
#   - Apply time-series cross-validation (walk-forward) for temporal data
#   - Use GroupKFold when observations are grouped (e.g., same patient)
#   - Visualise the regularisation path (coefficient trajectories)
#
# PREREQUISITES:
#   - MLFP03 Exercise 1 (feature engineering, dataset familiarity)
#   - MLFP02 Module (linear regression, Bayesian thinking from M2.1)
#
# ESTIMATED TIME: ~150-180 minutes
#
# TASKS:
#   1.  Demonstrate underfitting vs overfitting with polynomial models
#   2.  Visualise bias-variance decomposition across model complexity
#   3.  L2 regularisation (Ridge): geometry and coefficient shrinkage
#   4.  L1 regularisation (Lasso): sparsity and feature selection
#   5.  ElasticNet: combining L1 and L2
#   6.  Bayesian interpretation: L2 = Gaussian prior on weights
#   7.  Regularisation path visualisation (coefficient trajectories)
#   8.  Nested cross-validation for unbiased model selection
#   9.  Time-series cross-validation (walk-forward)
#   10. GroupKFold for grouped observations
#   11. Learning curve analysis
#   12. Compare all CV strategies and summarise
#
# DATASET: Singapore credit scoring data (from MLFP02)
#   Target: credit_utilisation (continuous — suitable for regression demo)
#   Rows: ~5,000 credit applicants | Features: financial + behavioural
#
# ════════════════════════════════════════════════════════════════════════
"""
from __future__ import annotations

import numpy as np
import polars as pl
from sklearn.linear_model import (
    Ridge,
    Lasso,
    ElasticNet,
    LinearRegression,
    RidgeCV,
    LassoCV,
)
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import (
    cross_val_score,
    KFold,
    StratifiedKFold,
    GroupKFold,
    TimeSeriesSplit,
    learning_curve,
)
from sklearn.metrics import mean_squared_error, r2_score

from kailash_ml import PreprocessingPipeline, ModelVisualizer
from kailash_ml.interop import to_sklearn_input

from shared import MLFPDataLoader


# ── Data Loading ──────────────────────────────────────────────────────

loader = MLFPDataLoader()
credit = loader.load("mlfp02", "sg_credit_scoring.parquet")

print(f"=== Singapore Credit Data ===")
print(f"Shape: {credit.shape}")
print(f"Default rate: {credit['default'].mean():.2%}")

target_col = (
    "credit_utilization" if "credit_utilization" in credit.columns else "income_sgd"
)

pipeline = PreprocessingPipeline()
result = pipeline.setup(
    data=credit,
    target=target_col,
    train_size=0.8,
    seed=42,
    normalize=True,
    categorical_encoding="ordinal",
    imputation_strategy="median",
)

print(f"\nTask type: {result.task_type}")
print(f"Train: {result.train_data.shape}, Test: {result.test_data.shape}")

X_train, y_train, col_info = to_sklearn_input(
    result.train_data,
    feature_columns=[c for c in result.train_data.columns if c != target_col],
    target_column=target_col,
)
X_test, y_test, _ = to_sklearn_input(
    result.test_data,
    feature_columns=[c for c in result.test_data.columns if c != target_col],
    target_column=target_col,
)
feature_names = col_info["feature_columns"]
print(f"Features: {len(feature_names)}")


# ══════════════════════════════════════════════════════════════════════
# TASK 1: Underfitting vs Overfitting — polynomial degree experiment
# ══════════════════════════════════════════════════════════════════════
# True function: y = sin(2πx) + noise

rng = np.random.default_rng(42)
n_pts = 100
x_1d = rng.uniform(0, 1, n_pts)
y_1d = np.sin(2 * np.pi * x_1d) + rng.normal(0, 0.2, n_pts)

x_1d_train, x_1d_test = x_1d[:80], x_1d[80:]
y_1d_train, y_1d_test = y_1d[:80], y_1d[80:]

x_train_2d = x_1d_train.reshape(-1, 1)
x_test_2d = x_1d_test.reshape(-1, 1)

print(f"\n=== Polynomial Degree Experiment ===")
print(f"{'Degree':>8} {'Train MSE':>12} {'Test MSE':>12} {'Gap':>10} {'Diagnosis':>15}")
print("─" * 62)

degree_results = {}
for degree in [1, 2, 4, 6, 9, 12, 15, 20]:
    # TODO: Build a sklearn Pipeline with PolynomialFeatures, StandardScaler,
    # and LinearRegression, then fit and compute train/test MSE.
    # Hint: Pipeline([("poly", PolynomialFeatures(degree)),
    #                 ("scaler", StandardScaler()),
    #                 ("model", LinearRegression())])
    poly_pipe = ____
    poly_pipe.fit(x_train_2d, y_1d_train)

    # TODO: Compute train_mse and test_mse using mean_squared_error
    # Hint: mean_squared_error(y_1d_train, poly_pipe.predict(x_train_2d))
    train_mse = ____
    test_mse = ____
    gap = test_mse - train_mse

    if degree <= 2:
        diagnosis = "Underfitting"
    elif degree <= 6:
        diagnosis = "Good fit"
    else:
        diagnosis = "Overfitting"

    degree_results[degree] = {"train_mse": train_mse, "test_mse": test_mse, "gap": gap}
    print(
        f"{degree:>8} {train_mse:>12.4f} {test_mse:>12.4f} "
        f"{gap:>10.4f} {diagnosis:>15}"
    )

# ── Checkpoint 1 ─────────────────────────────────────────────────────
assert (
    degree_results[1]["test_mse"] > degree_results[4]["test_mse"]
), "Degree=1 should have higher test error than degree=4"
assert (
    degree_results[20]["train_mse"] < degree_results[2]["train_mse"]
), "Degree=20 should have lower train error (memorisation)"
# INTERPRETATION: The gap between train_mse and test_mse is the overfit
# penalty. At degree=20, the model has memorised training data — high
# variance: it would look completely different on new training data.
print("\n✓ Checkpoint 1 passed — underfitting/overfitting pattern confirmed\n")


# ══════════════════════════════════════════════════════════════════════
# TASK 2: Bias-variance decomposition
# ══════════════════════════════════════════════════════════════════════
# Decompose expected test error = Bias² + Variance + Noise


def bias_variance_decomposition(
    degree: int,
    n_bootstrap: int = 50,
) -> dict[str, float]:
    """Estimate bias² and variance for a polynomial of given degree."""
    all_preds = []
    for _ in range(n_bootstrap):
        idx = rng.choice(len(y_1d_train), len(y_1d_train), replace=True)
        x_b, y_b = x_train_2d[idx], y_1d_train[idx]

        # TODO: Build the same Pipeline and fit on bootstrap sample
        # Hint: Pipeline([("poly", PolynomialFeatures(degree)),
        #                 ("scaler", StandardScaler()),
        #                 ("lr", LinearRegression())])
        model = ____
        model.fit(x_b, y_b)
        all_preds.append(model.predict(x_test_2d))

    preds = np.array(all_preds)  # (n_bootstrap, n_test)
    mean_pred = preds.mean(axis=0)

    y_true_noiseless = np.sin(2 * np.pi * x_1d_test)

    # TODO: Compute bias_sq = mean((mean_pred - y_true_noiseless)^2)
    # and variance = mean of variance across bootstrap predictions
    # Hint: bias_sq = np.mean((mean_pred - y_true_noiseless) ** 2)
    #       variance = np.mean(preds.var(axis=0))
    bias_sq = ____
    variance = ____
    noise = 0.04  # Known noise variance (sigma=0.2)²
    expected_error = bias_sq + variance + noise

    return {
        "bias_sq": bias_sq,
        "variance": variance,
        "noise": noise,
        "expected_error": expected_error,
    }


print(f"\n=== Bias-Variance Decomposition ===")
print(
    f"{'Degree':>8} {'Bias²':>10} {'Variance':>10} {'Noise':>8} "
    f"{'Expected':>12} {'Dominant':>12}"
)
print("─" * 66)

bv_results = {}
for degree in [1, 2, 3, 6, 10, 15]:
    bv = bias_variance_decomposition(degree, n_bootstrap=30)
    bv_results[degree] = bv
    dominant = "Bias" if bv["bias_sq"] > bv["variance"] else "Variance"
    print(
        f"{degree:>8} {bv['bias_sq']:>10.4f} {bv['variance']:>10.4f} "
        f"{bv['noise']:>8.4f} {bv['expected_error']:>12.4f} {dominant:>12}"
    )

# ── Checkpoint 2 ─────────────────────────────────────────────────────
assert (
    bv_results[1]["bias_sq"] > bv_results[10]["bias_sq"]
), "Degree=1 should have higher bias than degree=10"
assert (
    bv_results[1]["variance"] < bv_results[15]["variance"]
), "Degree=1 should have lower variance than degree=15"
# INTERPRETATION: As degree increases, Bias² drops but Variance rises.
# The sweet spot (degree 3-6) balances both. Cross-validation finds this
# automatically without knowing the true function.
print("\n✓ Checkpoint 2 passed — bias-variance decomposition computed\n")


# ══════════════════════════════════════════════════════════════════════
# TASK 3: L2 Regularisation (Ridge) — geometry and shrinkage
# ══════════════════════════════════════════════════════════════════════
# Ridge objective: min ||y - Xβ||² + α||β||²
# Geometry: L2 penalty defines a sphere — solution shrinks ALL coefficients
# toward zero but never exactly zero.
# Bayesian interpretation: L2 penalty ≡ Gaussian prior N(0, 1/α) on β.

alphas = [0.001, 0.01, 0.1, 1.0, 10.0, 100.0, 1000.0]

ridge_results = {}
for alpha in alphas:
    # TODO: Fit Ridge(alpha=alpha) and compute train/test MSE,
    # coefficient norm, and number of near-zero coefficients.
    # Hint: ridge = Ridge(alpha=alpha); ridge.fit(X_train, y_train)
    #       train_mse = mean_squared_error(y_train, ridge.predict(X_train))
    #       coef_norm = np.linalg.norm(ridge.coef_)
    #       n_zero = (np.abs(ridge.coef_) < 1e-6).sum()
    ridge = ____
    ridge.fit(X_train, y_train)
    train_mse = ____
    test_mse = ____
    coef_norm = ____
    n_zero = ____

    ridge_results[alpha] = {
        "train_mse": train_mse,
        "test_mse": test_mse,
        "coef_norm": coef_norm,
        "n_zero": n_zero,
        "coef": ridge.coef_.copy(),
    }

print(f"\n=== Ridge Regularisation (L2) ===")
print(
    f"{'Alpha':>10} {'Train MSE':>12} {'Test MSE':>12} "
    f"{'||β||₂':>10} {'Zero coefs':>12}"
)
print("─" * 60)
for alpha, r in ridge_results.items():
    print(
        f"{alpha:>10.3f} {r['train_mse']:>12.4f} {r['test_mse']:>12.4f} "
        f"{r['coef_norm']:>10.4f} {r['n_zero']:>12}"
    )

best_alpha_ridge = min(ridge_results.items(), key=lambda x: x[1]["test_mse"])
print(
    f"\nBest Ridge α = {best_alpha_ridge[0]} "
    f"(test MSE={best_alpha_ridge[1]['test_mse']:.4f})"
)

print("\nL2 Key Properties:")
print("  1. Shrinks all coefficients toward zero (never exactly zero)")
print("  2. Equivalent to MAP estimate with Gaussian prior: β ~ N(0, σ²/α)")
print("  3. Ridge closed-form: β = (X'X + αI)⁻¹X'y")
print("  4. Always invertible — regularisation fixes multicollinearity")

# ── Checkpoint 3 ─────────────────────────────────────────────────────
assert (
    ridge_results[1000.0]["coef_norm"] < ridge_results[0.001]["coef_norm"]
), "Higher alpha should produce smaller coefficient norm"
assert (
    ridge_results[1.0]["n_zero"] <= 2
), "Ridge should produce very few zero coefficients"
# INTERPRETATION: Watch how the coefficient norm drops as alpha increases.
# At alpha=0.001, Ridge barely differs from OLS. At alpha=1000, all
# coefficients are forced nearly to zero.
print("\n✓ Checkpoint 3 passed — Ridge regularisation behaviour confirmed\n")


# ══════════════════════════════════════════════════════════════════════
# TASK 4: L1 Regularisation (Lasso) — sparsity and feature selection
# ══════════════════════════════════════════════════════════════════════
# Lasso objective: min ||y - Xβ||² + α||β||₁
# Geometry: L1 penalty defines a diamond — MSE ellipsoid likely touches
# at a corner where some coordinates are exactly zero → SPARSITY.

lasso_results = {}
for alpha in alphas:
    # TODO: Fit Lasso(alpha=alpha, max_iter=10_000) and compute metrics
    # Hint: lasso = Lasso(alpha=alpha, max_iter=10_000); lasso.fit(X_train, y_train)
    #       coef_norm = np.linalg.norm(lasso.coef_, ord=1)   # L1 norm
    #       n_zero = (np.abs(lasso.coef_) < 1e-6).sum()
    lasso = ____
    lasso.fit(X_train, y_train)
    train_mse = mean_squared_error(y_train, lasso.predict(X_train))
    test_mse = mean_squared_error(y_test, lasso.predict(X_test))
    coef_norm = ____
    n_zero = ____

    lasso_results[alpha] = {
        "train_mse": train_mse,
        "test_mse": test_mse,
        "coef_norm": coef_norm,
        "n_zero": n_zero,
        "coef": lasso.coef_.copy(),
    }

print(f"\n=== Lasso Regularisation (L1) ===")
print(
    f"{'Alpha':>10} {'Train MSE':>12} {'Test MSE':>12} "
    f"{'||β||₁':>10} {'Zero coefs':>12}"
)
print("─" * 60)
for alpha, r in lasso_results.items():
    pct_sparse = r["n_zero"] / len(feature_names) * 100
    print(
        f"{alpha:>10.3f} {r['train_mse']:>12.4f} {r['test_mse']:>12.4f} "
        f"{r['coef_norm']:>10.4f} {r['n_zero']:>10} ({pct_sparse:.0f}%)"
    )

best_alpha_lasso = min(lasso_results.items(), key=lambda x: x[1]["test_mse"])
print(
    f"\nBest Lasso α = {best_alpha_lasso[0]} "
    f"(zero coefs={best_alpha_lasso[1]['n_zero']})"
)

# ── Checkpoint 4 ─────────────────────────────────────────────────────
assert (
    lasso_results[100.0]["n_zero"] > lasso_results[0.001]["n_zero"]
), "Higher alpha Lasso should zero out more coefficients"
# INTERPRETATION: Lasso's superpower vs Ridge. At high alpha, Lasso may
# eliminate 70% of features — automatic feature selection baked into
# the loss function.
print("\n✓ Checkpoint 4 passed — Lasso sparsity demonstrated\n")


# ══════════════════════════════════════════════════════════════════════
# TASK 5: ElasticNet — combining L1 + L2
# ══════════════════════════════════════════════════════════════════════
# l1_ratio controls the mix: 0 = pure Ridge, 1 = pure Lasso

en_results = {}
for l1_ratio in [0.1, 0.3, 0.5, 0.7, 0.9]:
    # TODO: Fit ElasticNet(alpha=0.1, l1_ratio=l1_ratio, max_iter=10_000)
    # Hint: en = ElasticNet(alpha=0.1, l1_ratio=l1_ratio, max_iter=10_000)
    en = ____
    en.fit(X_train, y_train)
    train_mse = mean_squared_error(y_train, en.predict(X_train))
    test_mse = mean_squared_error(y_test, en.predict(X_test))
    n_zero = (np.abs(en.coef_) < 1e-6).sum()

    en_results[l1_ratio] = {
        "train_mse": train_mse,
        "test_mse": test_mse,
        "n_zero": n_zero,
    }

print(f"\n=== ElasticNet (α=0.1) ===")
print(f"{'L1 ratio':>10} {'Train MSE':>12} {'Test MSE':>12} {'Zero coefs':>12}")
print("─" * 50)
for l1_ratio, r in en_results.items():
    print(
        f"{l1_ratio:>10.1f} {r['train_mse']:>12.4f} "
        f"{r['test_mse']:>12.4f} {r['n_zero']:>12}"
    )

# ── Checkpoint 5 ─────────────────────────────────────────────────────
assert (
    en_results[0.9]["n_zero"] >= en_results[0.1]["n_zero"]
), "l1_ratio=0.9 should zero out at least as many as l1_ratio=0.1"
# INTERPRETATION: ElasticNet solves Lasso's instability with correlated
# features. When two features are correlated, Lasso arbitrarily picks
# one. ElasticNet keeps both with smaller coefficients.
print("\n✓ Checkpoint 5 passed — ElasticNet l1_ratio effect confirmed\n")


# ══════════════════════════════════════════════════════════════════════
# TASK 6: Bayesian Interpretation — L2 as Gaussian Prior
# ══════════════════════════════════════════════════════════════════════
# MAP estimate: β_MAP = argmin ||y - Xβ||² + (σ²/τ²)||β||²
# This is exactly Ridge with α = σ²/τ²!

print(f"\n=== Bayesian Interpretation of L2 Regularisation ===")
print(
    """
MAP estimate under Gaussian prior:
  Prior:      β ~ N(0, τ²I)   (belief that β is close to zero)
  Likelihood: y|β ~ N(Xβ, σ²I)
  MAP objective: β_MAP = argmin ||y - Xβ||² + (σ²/τ²)||β||²
                                            ↑ This is α in Ridge!
"""
)

# TODO: Compute the implied alpha from noise variance and unit prior
# Hint: sigma_sq = mean_squared_error(y_train, Ridge(alpha=0).fit(X_train, y_train).predict(X_train))
#       tau_sq = 1.0
#       alpha_implied = sigma_sq / tau_sq
n, p = X_train.shape
sigma_sq = ____
tau_sq = 1.0
alpha_implied = ____

ridge_bayes = Ridge(alpha=alpha_implied)
ridge_bayes.fit(X_train, y_train)
test_mse_bayes = mean_squared_error(y_test, ridge_bayes.predict(X_test))
print(f"Implied α = σ²/τ² = {alpha_implied:.4f}")
print(f"Ridge with Bayesian α: Test MSE = {test_mse_bayes:.4f}")

# ── Checkpoint 6 ─────────────────────────────────────────────────────
assert alpha_implied > 0, "Implied alpha should be positive"
assert test_mse_bayes > 0, "Bayesian Ridge should produce valid predictions"
ols = LinearRegression().fit(X_train, y_train)
assert np.linalg.norm(ols.coef_) >= np.linalg.norm(
    ridge_bayes.coef_
), "OLS should have at least as large a norm as Ridge"
# INTERPRETATION: Choosing α is the same as specifying prior beliefs.
# Large α (small τ) = strong belief that β ≈ 0. Zero α = flat prior (OLS).
print("\n✓ Checkpoint 6 passed — Bayesian interpretation verified\n")


# ══════════════════════════════════════════════════════════════════════
# TASK 7: Regularisation path visualisation
# ══════════════════════════════════════════════════════════════════════

viz = ModelVisualizer()

print(f"\n=== Regularisation Path ===")
print(
    f"{'Alpha':>10} {'||β||₂ (Ridge)':>16} {'||β||₁ (Lasso)':>16} {'Non-zero (L)':>14}"
)
print("─" * 60)
for alpha in alphas:
    ridge_norm = ridge_results[alpha]["coef_norm"]
    lasso_norm = lasso_results[alpha]["coef_norm"]
    lasso_nz = len(feature_names) - lasso_results[alpha]["n_zero"]
    print(f"{alpha:>10.3f} {ridge_norm:>16.4f} {lasso_norm:>16.4f} {lasso_nz:>14}")

# TODO: Build the coefficient matrix for Ridge and save an HTML chart
# Hint: coef_matrix_ridge = np.array([ridge_results[a]["coef"] for a in alphas])
#       fig = viz.training_history({"||β||₂": [ridge_results[a]["coef_norm"] for a in alphas]},
#                                   x_label="Regularisation Strength (α)")
#       fig.write_html("ex2_ridge_path.html")
coef_matrix_ridge = ____
fig = ____
fig.update_layout(title="Ridge: Coefficient Norm vs Regularisation Strength")
fig.write_html("ex2_ridge_path.html")

fig_sparse = viz.training_history(
    {
        "Non-zero coefs": [
            len(feature_names) - lasso_results[a]["n_zero"] for a in alphas
        ]
    },
    x_label="Regularisation Strength (α)",
)
fig_sparse.update_layout(title="Lasso: Sparsity vs Regularisation Strength")
fig_sparse.write_html("ex2_lasso_sparsity.html")

print("\nSaved: ex2_ridge_path.html, ex2_lasso_sparsity.html")

# ── Checkpoint 7 ─────────────────────────────────────────────────────
assert coef_matrix_ridge.shape == (
    len(alphas),
    len(feature_names),
), "Coefficient matrix should have one row per alpha"
# INTERPRETATION: Ridge: all coefficients shrink smoothly toward zero.
# Lasso: coefficients snap to zero one by one — the "kink" reflects the
# L1 diamond geometry.
print("\n✓ Checkpoint 7 passed — regularisation path visualised\n")


# ══════════════════════════════════════════════════════════════════════
# TASK 8: Nested cross-validation for unbiased model selection
# ══════════════════════════════════════════════════════════════════════
# PROBLEM: Using the same CV to select hyperparameters AND estimate
# performance gives an optimistic estimate (information leakage).
# SOLUTION: Nested CV — outer loop estimates performance, inner
# loop selects hyperparameters.

print(f"\n=== Nested Cross-Validation ===")

# Standard (non-nested) CV for comparison — BIASED
# TODO: Fit RidgeCV(alphas=alphas, cv=5) and get test score
# Hint: ridge_cv = RidgeCV(alphas=alphas, cv=5); ridge_cv.fit(X_train, y_train)
#       standard_cv_score = ridge_cv.score(X_test, y_test)
ridge_cv = ____
ridge_cv.fit(X_train, y_train)
standard_cv_score = ____
print(f"Standard CV (biased): R²={standard_cv_score:.4f}, best α={ridge_cv.alpha_:.4f}")

# TODO: Implement nested CV with outer 5-fold and inner 3-fold
# For each outer fold: run inner CV to find best alpha, evaluate on outer test fold
# Hint: outer_cv = KFold(n_splits=5, shuffle=True, random_state=42)
#       inner_cv = KFold(n_splits=3, shuffle=True, random_state=42)
outer_cv = ____
inner_cv = ____

nested_scores = []
selected_alphas = []

for fold_idx, (train_idx, test_idx) in enumerate(outer_cv.split(X_train)):
    X_outer_train, X_outer_test = X_train[train_idx], X_train[test_idx]
    y_outer_train, y_outer_test = y_train[train_idx], y_train[test_idx]

    best_inner_score = -np.inf
    best_inner_alpha = alphas[0]
    for alpha in alphas:
        inner_scores = cross_val_score(
            Ridge(alpha=alpha), X_outer_train, y_outer_train, cv=inner_cv, scoring="r2"
        )
        if inner_scores.mean() > best_inner_score:
            best_inner_score = inner_scores.mean()
            best_inner_alpha = alpha

    ridge_selected = Ridge(alpha=best_inner_alpha)
    ridge_selected.fit(X_outer_train, y_outer_train)
    outer_score = r2_score(y_outer_test, ridge_selected.predict(X_outer_test))

    nested_scores.append(outer_score)
    selected_alphas.append(best_inner_alpha)
    print(
        f"  Fold {fold_idx + 1}: selected α={best_inner_alpha:.4f}, outer R²={outer_score:.4f}"
    )

nested_mean = np.mean(nested_scores)
nested_std = np.std(nested_scores)
print(f"\nNested CV: R² = {nested_mean:.4f} ± {nested_std:.4f}")
print(f"Standard CV: R² = {standard_cv_score:.4f}")
print(f"Optimism bias: {standard_cv_score - nested_mean:.4f}")

# ── Checkpoint 8 ─────────────────────────────────────────────────────
assert len(nested_scores) == 5, "Should have 5 outer fold scores"
# INTERPRETATION: The gap between standard CV and nested CV is the
# optimism bias from using the same data for selection and evaluation.
# Nested CV is always more honest — use it for reporting final performance.
print("\n✓ Checkpoint 8 passed — nested CV demonstrates optimism bias\n")


# ══════════════════════════════════════════════════════════════════════
# TASK 9: Time-series cross-validation (walk-forward)
# ══════════════════════════════════════════════════════════════════════
# For time-series data, standard k-fold leaks future information.
# TimeSeriesSplit ensures training only uses past data.

print(f"\n=== Time-Series Cross-Validation ===")

# TODO: Create TimeSeriesSplit(n_splits=5) and compare scores to k-fold
# Hint: tscv = TimeSeriesSplit(n_splits=5)
#       ts_scores = cross_val_score(Ridge(alpha=1.0), X_train, y_train, cv=tscv, scoring="r2")
tscv = ____

print("Walk-forward splits:")
for fold, (train_idx, test_idx) in enumerate(tscv.split(X_train)):
    print(
        f"  Fold {fold + 1}: train=[0:{train_idx[-1]+1}] ({len(train_idx)} samples), "
        f"test=[{test_idx[0]}:{test_idx[-1]+1}] ({len(test_idx)} samples)"
    )

kfold_scores = cross_val_score(Ridge(alpha=1.0), X_train, y_train, cv=5, scoring="r2")
ts_scores = ____

print(f"\nStandard 5-fold: R² = {kfold_scores.mean():.4f} ± {kfold_scores.std():.4f}")
print(f"Time-series 5-fold: R² = {ts_scores.mean():.4f} ± {ts_scores.std():.4f}")

# ── Checkpoint 9 ─────────────────────────────────────────────────────
assert len(ts_scores) == 5, "Should have 5 time-series CV scores"
# INTERPRETATION: If time-series CV gives substantially different scores
# than standard k-fold, your data has temporal dependence. Never shuffle
# temporal data for k-fold — it leaks the future!
print("\n✓ Checkpoint 9 passed — time-series CV demonstrated\n")


# ══════════════════════════════════════════════════════════════════════
# TASK 10: GroupKFold for grouped observations
# ══════════════════════════════════════════════════════════════════════
# When observations are grouped (e.g., multiple admissions per patient),
# GroupKFold ensures all observations from one group stay together.

print(f"\n=== GroupKFold ===")

n_samples = X_train.shape[0]
n_groups = n_samples // 5
groups = np.repeat(np.arange(n_groups), 5)[:n_samples]
rng.shuffle(groups)

# TODO: Create GroupKFold(n_splits=5) and compute scores with groups
# Hint: group_cv = GroupKFold(n_splits=5)
#       group_scores = cross_val_score(Ridge(alpha=1.0), X_train, y_train,
#                                      cv=group_cv, groups=groups, scoring="r2")
group_cv = ____
group_scores = ____

print(f"Standard k-fold: R² = {kfold_scores.mean():.4f} ± {kfold_scores.std():.4f}")
print(f"GroupKFold:       R² = {group_scores.mean():.4f} ± {group_scores.std():.4f}")

# Verify group integrity
for fold, (train_idx, test_idx) in enumerate(group_cv.split(X_train, groups=groups)):
    train_groups = set(groups[train_idx])
    test_groups = set(groups[test_idx])
    overlap = train_groups & test_groups
    assert len(overlap) == 0, f"Fold {fold}: groups overlap! {overlap}"

print("  ✓ All folds verified: no group appears in both train and test")

# ── Checkpoint 10 ────────────────────────────────────────────────────
assert len(group_scores) == 5, "Should have 5 GroupKFold scores"
# INTERPRETATION: GroupKFold typically gives LOWER scores because the
# model cannot exploit group-level identity. This is the honest estimate
# — in production you predict on NEW patients/customers.
print("\n✓ Checkpoint 10 passed — GroupKFold group integrity verified\n")


# ══════════════════════════════════════════════════════════════════════
# TASK 11: Learning curve analysis
# ══════════════════════════════════════════════════════════════════════
# Learning curves diagnose: MORE DATA needed vs BETTER MODEL needed.

print(f"\n=== Learning Curve Analysis ===")

models_for_lc = {
    "OLS (no regularisation)": LinearRegression(),
    "Ridge (α=1)": Ridge(alpha=1.0),
    "Lasso (α=0.1)": Lasso(alpha=0.1, max_iter=10_000),
}

train_sizes_frac = [0.1, 0.2, 0.4, 0.6, 0.8, 1.0]

for name, model_lc in models_for_lc.items():
    # TODO: Compute learning curve for each model
    # Hint: train_sizes, train_scores, test_scores = learning_curve(
    #           model_lc, X_train, y_train, train_sizes=train_sizes_frac,
    #           cv=5, scoring="r2", n_jobs=-1)
    train_sizes, train_scores, test_scores = ____

    print(f"\n--- {name} ---")
    print(f"{'Train size':>12} {'Train R²':>10} {'Test R²':>10} {'Gap':>10}")
    print("─" * 44)
    for size, tr_score, te_score in zip(
        train_sizes, train_scores.mean(axis=1), test_scores.mean(axis=1)
    ):
        gap = tr_score - te_score
        print(f"{size:>12} {tr_score:>10.4f} {te_score:>10.4f} {gap:>10.4f}")

# ── Checkpoint 11 ────────────────────────────────────────────────────
assert len(train_sizes) == len(train_sizes_frac), "Should have all training sizes"
# INTERPRETATION: If test curve is still rising at full training size,
# MORE DATA will help. If it plateaued, need a BETTER MODEL.
# Regularisation shrinks the gap between train and test curves.
print("\n✓ Checkpoint 11 passed — learning curve analysis complete\n")


# ══════════════════════════════════════════════════════════════════════
# TASK 12: Summary — compare all CV strategies
# ══════════════════════════════════════════════════════════════════════

print(f"\n{'=' * 70}")
print("CROSS-VALIDATION STRATEGY COMPARISON")
print(f"{'=' * 70}")

cv_summary = {
    "Standard k-fold": {
        "R²_mean": kfold_scores.mean(),
        "R²_std": kfold_scores.std(),
        "when": "i.i.d. data, no groups, no temporal",
    },
    "Nested CV": {
        "R²_mean": nested_mean,
        "R²_std": nested_std,
        "when": "Hyperparameter selection + performance estimate",
    },
    "Time-series": {
        "R²_mean": ts_scores.mean(),
        "R²_std": ts_scores.std(),
        "when": "Temporal data, prediction into the future",
    },
    "GroupKFold": {
        "R²_mean": group_scores.mean(),
        "R²_std": group_scores.std(),
        "when": "Grouped observations (patients, customers)",
    },
}

print(f"\n{'Strategy':<20} {'R²':>12} {'±':>4} {'When to use'}")
print("─" * 72)
for name, info in cv_summary.items():
    print(
        f"{name:<20} {info['R²_mean']:>12.4f} ± {info['R²_std']:.4f}  "
        f"{info['when']}"
    )

# ── Checkpoint 12 ────────────────────────────────────────────────────
assert len(cv_summary) == 4, "Should compare 4 CV strategies"
# INTERPRETATION: Match the CV strategy to the real-world prediction task.
# Standard k-fold on temporal data gives an optimistic estimate; GroupKFold
# on i.i.d. data is unnecessarily conservative.
print("\n✓ Checkpoint 12 passed — CV strategy comparison complete\n")


# ══════════════════════════════════════════════════════════════════════
# REFLECTION
# ══════════════════════════════════════════════════════════════════════
print("═" * 70)
print("  WHAT YOU'VE MASTERED")
print("═" * 70)
print(
    """
  ✓ Bias-Variance: complexity ↑ → Bias² ↓, Variance ↑
  ✓ Ridge (L2): uniform shrinkage, never exactly zero, Gaussian prior
  ✓ Lasso (L1): sparse selection, some β_i exactly = 0, Laplace prior
  ✓ ElasticNet: handles correlated features, blends both penalties
  ✓ Bayesian view: choosing α = specifying prior beliefs about β
  ✓ Regularisation path: coefficient trajectories as α varies
  ✓ Nested CV: unbiased model selection without information leakage
  ✓ Time-series CV: walk-forward validation for temporal data
  ✓ GroupKFold: grouped observations stay together
  ✓ Learning curves: diagnose data hunger vs model weakness

  CV STRATEGY SELECTION:
    i.i.d. data → k-fold
    temporal    → TimeSeriesSplit
    grouped     → GroupKFold
    reporting   → nested CV

  NEXT: Exercise 3 trains the complete supervised model zoo — SVM,
  KNN, Naive Bayes, Decision Trees, and Random Forests — on e-commerce
  churn data. You'll compute Gini impurity from scratch.
"""
)

print(
    "\n✓ Exercise 2 complete — bias-variance, L1/L2 geometry, Bayesian interpretation"
)
