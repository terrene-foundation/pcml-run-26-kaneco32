# Copyright 2026 Terrene Foundation
# SPDX-License-Identifier: Apache-2.0
"""
# ════════════════════════════════════════════════════════════════════════
# MLFP02 — Exercise 5: Linear Regression
# ════════════════════════════════════════════════════════════════════════
#
# WHAT YOU'LL LEARN:
#   After completing this exercise, you will be able to:
#   - Derive and implement OLS using the normal equation β = (X'X)⁻¹X'y
#   - Interpret regression coefficients with ceteris paribus reasoning
#   - Test coefficient significance using t-statistics and p-values
#   - Compute R², adjusted R², and the F-statistic for model evaluation
#   - Detect multicollinearity using Variance Inflation Factor (VIF)
#   - Perform residual diagnostics: normality, heteroscedasticity, patterns
#   - Extend models with polynomial and interaction terms
#   - Implement Weighted Least Squares (WLS) when heteroscedasticity exists
#   - Apply dummy variable encoding with a base category
#   - Cross-validate with train/test split and compute out-of-sample R²
#   - Use nested cross-validation to compare model complexity
#
# PREREQUISITES: Complete Exercises 2-3 — you should understand MLE,
#   hypothesis testing, t-statistics, and p-value interpretation.
#
# ESTIMATED TIME: ~170 minutes
#
# TASKS:
#    1. Load HDB data and engineer numeric features
#    2. Implement OLS from scratch: β = (X'X)⁻¹X'y
#    3. Interpret coefficients: direction, magnitude, significance
#    4. Compute t-statistics and p-values for every coefficient
#    5. Compute R², adjusted R², F-statistic
#    6. Detect multicollinearity with VIF
#    7. Residual diagnostics: normality, patterns, heteroscedasticity
#    8. Weighted Least Squares for heteroscedastic data
#    9. Polynomial and interaction terms — model enrichment
#   10. Dummy variable encoding for categorical features
#   11. Train/test split: out-of-sample evaluation
#   12. Model comparison and business interpretation
#
# DATASET: HDB resale flat transactions (Singapore)
#   Source: data.gov.sg — public housing resale records, 2020+
#   Target: resale_price (SGD)
#   Features: floor_area_sqm, storey_midpoint, remaining_lease, flat_type, town
#
# THEORY:
#   OLS minimises Σ(yᵢ - ŷᵢ)². Closed-form: β = (X'X)⁻¹X'y
#   Each βⱼ = expected change in y for one-unit change in xⱼ,
#   holding all others constant (ceteris paribus).
#   t = βⱼ / SE(βⱼ), testing H₀: βⱼ = 0.
#
# ════════════════════════════════════════════════════════════════════════
"""
from __future__ import annotations

import numpy as np
import polars as pl
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from kailash_ml import ModelVisualizer
from scipy import stats

from shared import MLFPDataLoader


# ── Data Loading ──────────────────────────────────────────────────────

loader = MLFPDataLoader()
hdb = loader.load("mlfp01", "hdb_resale.parquet")

print("=" * 70)
print("  MLFP02 Exercise 5: Linear Regression")
print("=" * 70)
print(f"\n  Data loaded: hdb_resale.parquet ({hdb.shape[0]:,} rows)")

# Filter to recent data
hdb = hdb.with_columns(
    pl.col("month").str.to_date("%Y-%m").alias("transaction_date"),
)
hdb_recent = hdb.filter(pl.col("transaction_date") >= pl.date(2020, 1, 1))


# ══════════════════════════════════════════════════════════════════════
# TASK 1: Feature Engineering
# ══════════════════════════════════════════════════════════════════════
# Regression requires numeric features. We parse storey range to a
# midpoint and compute remaining lease. Categorical variables will
# be dummy-encoded in Task 10.

hdb_recent = hdb_recent.with_columns(
    # Storey midpoint: "07 TO 09" → (7+9)/2 = 8
    (
        (
            pl.col("storey_range").str.extract(r"(\d+)", 1).cast(pl.Float64)
            + pl.col("storey_range").str.extract(r"TO (\d+)", 1).cast(pl.Float64)
        )
        / 2
    ).alias("storey_midpoint"),
    # Remaining lease (99-year leases)
    (99 - (pl.col("transaction_date").dt.year() - pl.col("lease_commence_date")))
    .cast(pl.Float64)
    .alias("remaining_lease_years"),
)

# Drop rows with nulls in key columns
hdb_clean = hdb_recent.drop_nulls(
    subset=[
        "floor_area_sqm",
        "storey_midpoint",
        "remaining_lease_years",
        "resale_price",
    ]
)

print(f"  Cleaned: {hdb_clean.height:,} rows (from {hdb_recent.height:,})")
print(f"  Features: floor_area_sqm, storey_midpoint, remaining_lease_years")
print(f"  Target: resale_price")

# Summary statistics
for col in [
    "floor_area_sqm",
    "storey_midpoint",
    "remaining_lease_years",
    "resale_price",
]:
    vals = hdb_clean[col].to_numpy().astype(np.float64)
    print(
        f"  {col}: mean={vals.mean():.1f}, std={vals.std():.1f}, "
        f"range=[{vals.min():.0f}, {vals.max():.0f}]"
    )

# ── Checkpoint 1 ─────────────────────────────────────────────────────
assert hdb_clean.height > 10_000, f"Expected >10K rows, got {hdb_clean.height}"
assert "storey_midpoint" in hdb_clean.columns, "storey_midpoint must exist"
assert "remaining_lease_years" in hdb_clean.columns, "remaining_lease must exist"
print("\n✓ Checkpoint 1 passed — features engineered\n")


# ══════════════════════════════════════════════════════════════════════
# TASK 2: OLS from Scratch — β = (X'X)⁻¹X'y
# ══════════════════════════════════════════════════════════════════════
# The normal equation gives the OLS solution in matrix form.
# X is the design matrix (with intercept column of 1s).
# This derivation: minimise ||y - Xβ||² → ∂/∂β = 0 → X'Xβ = X'y

y = hdb_clean["resale_price"].to_numpy().astype(np.float64)
features = ["floor_area_sqm", "storey_midpoint", "remaining_lease_years"]

# Build design matrix with intercept
X_raw = hdb_clean.select(features).to_numpy().astype(np.float64)
n_obs = X_raw.shape[0]
X = np.column_stack([np.ones(n_obs), X_raw])  # Add intercept column
feature_names = ["intercept"] + features
k = X.shape[1]  # Number of parameters (including intercept)

print(f"\n=== OLS from Scratch ===")
print(f"Design matrix X: {X.shape} (n={n_obs:,}, k={k})")

# Normal equation: β = (X'X)⁻¹X'y
# TODO: Compute X'X (matrix product of X transposed with X)
XtX = ____  # Hint: X.T @ X

# TODO: Compute the inverse of X'X
XtX_inv = ____  # Hint: np.linalg.inv(XtX)

# TODO: Compute X'y
Xty = ____  # Hint: X.T @ y

# TODO: Compute OLS coefficients using the normal equation
beta_ols = ____  # Hint: XtX_inv @ Xty

# Predictions and residuals
y_hat = X @ beta_ols
residuals = y - y_hat

print(f"\nOLS Coefficients:")
print(f"{'Feature':<25} {'Coefficient':>14}")
print("─" * 42)
for name, coef in zip(feature_names, beta_ols):
    print(f"{name:<25} {coef:>14,.2f}")

# Verify with scipy/numpy lstsq
beta_lstsq, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
print(
    f"\nVerification (numpy lstsq): max |diff| = {np.max(np.abs(beta_ols - beta_lstsq)):.2e}"
)
# INTERPRETATION: Each coefficient is the expected change in resale_price
# for a one-unit increase in that feature, holding all others constant.
# E.g., βfloor_area = $X means each additional sqm adds $X to price.

# ── Checkpoint 2 ─────────────────────────────────────────────────────
assert np.allclose(
    beta_ols, beta_lstsq, atol=1.0
), "OLS from scratch must match numpy lstsq"
assert len(beta_ols) == k, "Should have k coefficients"
print("\n✓ Checkpoint 2 passed — OLS implemented from scratch\n")


# ══════════════════════════════════════════════════════════════════════
# TASK 3: Coefficient Interpretation
# ══════════════════════════════════════════════════════════════════════
# Ceteris paribus: "all else equal, one more sqm of floor area
# is associated with $X higher resale price."

print(f"\n=== Coefficient Interpretation ===")
for i, name in enumerate(feature_names):
    if i == 0:
        print(f"\nIntercept: ${beta_ols[0]:,.0f}")
        print(f"  A flat with all features at zero would cost ${beta_ols[0]:,.0f}")
        print(f"  (Not meaningful — extrapolation beyond data range)")
    else:
        direction = "increases" if beta_ols[i] > 0 else "decreases"
        unit = "sqm" if "area" in name else "storey" if "storey" in name else "year"
        print(f"\n{name}: ${beta_ols[i]:,.0f} per {unit}")
        print(
            f"  Each additional {unit} {direction} resale price by ${abs(beta_ols[i]):,.0f}"
        )
        print(f"  Holding all other features constant (ceteris paribus)")

# Practical example
example_flat = {
    "floor_area_sqm": 92.0,
    "storey_midpoint": 8.0,
    "remaining_lease_years": 75.0,
}
predicted = beta_ols[0]
for i, (feat, val) in enumerate(example_flat.items()):
    predicted += beta_ols[i + 1] * val

print(f"\n--- Prediction Example ---")
for feat, val in example_flat.items():
    print(f"  {feat} = {val}")
print(f"  Predicted price: ${predicted:,.0f}")
actual_similar = hdb_clean.filter(
    (pl.col("floor_area_sqm").is_between(90, 94))
    & (pl.col("storey_midpoint").is_between(7, 9))
)
if actual_similar.height > 0:
    print(
        f"  Actual mean for similar flats: ${actual_similar['resale_price'].mean():,.0f}"
    )

# ── Checkpoint 3 ─────────────────────────────────────────────────────
assert (
    beta_ols[1] > 0
), "Floor area coefficient should be positive (bigger = more expensive)"
print("\n✓ Checkpoint 3 passed — coefficients interpreted\n")


# ══════════════════════════════════════════════════════════════════════
# TASK 4: t-Statistics and p-Values
# ══════════════════════════════════════════════════════════════════════
# H₀: βⱼ = 0 (feature j has no linear relationship with price)
# t = βⱼ / SE(βⱼ), where SE(βⱼ) = √(σ̂² × (X'X)⁻¹ⱼⱼ)
# σ̂² = SSR / (n - k) = Σeᵢ² / (n - k)

SSR = np.sum(residuals**2)
# TODO: Compute unbiased residual variance estimator σ̂²
sigma_sq_hat = ____  # Hint: SSR / (n_obs - k)
sigma_hat = np.sqrt(sigma_sq_hat)

# Standard errors from diagonal of (X'X)⁻¹
# TODO: Compute SE for each coefficient: √(σ̂² × diag(X'X)⁻¹)
se_beta = ____  # Hint: np.sqrt(sigma_sq_hat * np.diag(XtX_inv))

# TODO: Compute t-statistics for each coefficient
t_stats = ____  # Hint: beta_ols / se_beta

# TODO: Compute two-sided p-values from t-distribution
p_values = ____  # Hint: 2 * (1 - stats.t.cdf(np.abs(t_stats), df=n_obs - k))

print(f"\n=== Coefficient Significance ===")
print(f"Residual σ̂ = ${sigma_hat:,.0f}")
print(f"Degrees of freedom: {n_obs - k:,}")
print(
    f"\n{'Feature':<25} {'β':>12} {'SE(β)':>12} {'t-stat':>10} {'p-value':>12} {'Sig':>6}"
)
print("─" * 80)
for i, name in enumerate(feature_names):
    sig = (
        "***"
        if p_values[i] < 0.001
        else "**" if p_values[i] < 0.01 else "*" if p_values[i] < 0.05 else "ns"
    )
    print(
        f"{name:<25} {beta_ols[i]:>12,.2f} {se_beta[i]:>12,.2f} "
        f"{t_stats[i]:>10.2f} {p_values[i]:>12.2e} {sig:>6}"
    )
# INTERPRETATION: The t-statistic tests whether each coefficient is
# significantly different from zero. A large |t| (and small p) means
# the feature has a statistically significant linear relationship with
# price. But statistical significance ≠ practical importance — a
# significant coefficient of $100 per sqm is far less impactful than
# one of $5,000 per sqm.

# ── Checkpoint 4 ─────────────────────────────────────────────────────
assert all(se > 0 for se in se_beta), "All standard errors must be positive"
assert all(0 <= p <= 1 for p in p_values), "All p-values must be valid"
print("\n✓ Checkpoint 4 passed — significance testing completed\n")


# ══════════════════════════════════════════════════════════════════════
# TASK 5: R², Adjusted R², and F-Statistic
# ══════════════════════════════════════════════════════════════════════
# R² = 1 - SSR/SST = proportion of variance explained
# Adjusted R² = 1 - (1-R²)(n-1)/(n-k-1) — penalises for more features
# F = (SSR_reduced - SSR_full) / (k-1) / (SSR_full / (n-k))

SST = np.sum((y - y.mean()) ** 2)
SSE = np.sum((y_hat - y.mean()) ** 2)  # Explained sum of squares

# TODO: Compute R² = 1 - SSR/SST
r_squared = ____  # Hint: 1 - SSR / SST

# TODO: Compute adjusted R² penalising for extra features
adj_r_squared = ____  # Hint: 1 - (1 - r_squared) * (n_obs - 1) / (n_obs - k)

# F-statistic: model vs intercept-only
# TODO: Compute F-statistic
f_stat = ____  # Hint: (SSE / (k - 1)) / (SSR / (n_obs - k))
f_p_value = 1 - stats.f.cdf(f_stat, dfn=k - 1, dfd=n_obs - k)

print(f"\n=== Model Fit Statistics ===")
print(f"SST (total):     {SST:,.0f}")
print(f"SSR (residual):  {SSR:,.0f}")
print(f"SSE (explained): {SSE:,.0f}")
print(f"Check: SST = SSR + SSE → {SST:,.0f} ≈ {SSR + SSE:,.0f} ✓")
print(f"\nR²:          {r_squared:.6f} ({r_squared:.2%} of variance explained)")
print(f"Adjusted R²: {adj_r_squared:.6f}")
print(f"F-statistic: {f_stat:.2f} (p < {f_p_value:.2e})")
print(f"RMSE:        ${sigma_hat:,.0f}")
print(f"MAE:         ${np.mean(np.abs(residuals)):,.0f}")

# ── Checkpoint 5 ─────────────────────────────────────────────────────
assert 0 < r_squared < 1, "R² must be between 0 and 1"
assert adj_r_squared <= r_squared, "Adjusted R² must be ≤ R²"
assert f_stat > 0, "F-statistic must be positive"
assert abs(SST - SSR - SSE) < 1, "SST = SSR + SSE must hold"
print("\n✓ Checkpoint 5 passed — model evaluation completed\n")


# ══════════════════════════════════════════════════════════════════════
# TASK 6: Multicollinearity — Variance Inflation Factor (VIF)
# ══════════════════════════════════════════════════════════════════════
# VIF_j = 1/(1 - R²_j) where R²_j is from regressing feature j on
# all other features. VIF > 5 = moderate concern, VIF > 10 = serious.

print(f"\n=== Multicollinearity: VIF ===")

vif_results = {}
for j in range(len(features)):
    # Regress feature j on all other features
    other_idx = [i for i in range(len(features)) if i != j]
    X_other = np.column_stack([np.ones(n_obs), X_raw[:, other_idx]])
    y_j = X_raw[:, j]

    beta_j = np.linalg.lstsq(X_other, y_j, rcond=None)[0]
    y_j_hat = X_other @ beta_j
    ss_res_j = np.sum((y_j - y_j_hat) ** 2)
    ss_tot_j = np.sum((y_j - y_j.mean()) ** 2)
    r2_j = 1 - ss_res_j / ss_tot_j
    # TODO: Compute VIF from R²_j: VIF = 1 / (1 - R²_j)
    vif_j = ____  # Hint: 1 / (1 - r2_j) if r2_j < 1 else float("inf")
    vif_results[features[j]] = vif_j

print(f"{'Feature':<25} {'VIF':>8} {'Status':>12}")
print("─" * 48)
for feat, vif in vif_results.items():
    status = "OK" if vif < 5 else "MODERATE" if vif < 10 else "HIGH"
    print(f"{feat:<25} {vif:>8.2f} {status:>12}")

print(f"\nCorrelation matrix:")
corr_matrix = np.corrcoef(X_raw.T)
for i, fi in enumerate(features):
    for j, fj in enumerate(features):
        if j > i:
            print(f"  corr({fi}, {fj}) = {corr_matrix[i,j]:.3f}")
# INTERPRETATION: VIF > 10 means the feature is almost entirely
# predictable from other features — its coefficient estimate is
# unstable and its SE is inflated. Drop one of the collinear features
# or use regularisation (Ridge regression).

# ── Checkpoint 6 ─────────────────────────────────────────────────────
assert all(v >= 1.0 for v in vif_results.values()), "VIF must be ≥ 1"
print("\n✓ Checkpoint 6 passed — VIF computed\n")


# ══════════════════════════════════════════════════════════════════════
# TASK 7: Residual Diagnostics
# ══════════════════════════════════════════════════════════════════════
# Good residuals should be: normally distributed, homoscedastic,
# uncorrelated, and show no patterns.

print(f"\n=== Residual Diagnostics ===")

# 1. Normality of residuals (Shapiro-Wilk on subsample)
residual_sample = np.random.default_rng(42).choice(
    residuals, size=min(5000, len(residuals)), replace=False
)
# TODO: Run Shapiro-Wilk normality test on residual_sample
sw_stat, sw_p = ____  # Hint: stats.shapiro(residual_sample)
print(f"\n1. Normality (Shapiro-Wilk on subsample):")
print(f"   W={sw_stat:.4f}, p={sw_p:.6f}")
print(
    f"   {'Normal residuals' if sw_p > 0.05 else 'Non-normal residuals — consider robust SE'}"
)

# 2. Skewness and kurtosis
res_skew = stats.skew(residuals)
res_kurt = stats.kurtosis(residuals)
print(f"\n2. Shape:")
print(f"   Skewness: {res_skew:.3f} (0 = symmetric)")
print(f"   Excess kurtosis: {res_kurt:.3f} (0 = Normal tails)")

# 3. Heteroscedasticity (Breusch-Pagan test)
# Regress squared residuals on X to test if variance depends on predictors
e_sq = residuals**2
bp_X = np.column_stack([np.ones(n_obs), X_raw])
bp_beta = np.linalg.lstsq(bp_X, e_sq, rcond=None)[0]
bp_predicted = bp_X @ bp_beta
bp_sse = np.sum((e_sq - bp_predicted) ** 2)
bp_sst = np.sum((e_sq - e_sq.mean()) ** 2)
bp_r2 = 1 - bp_sse / bp_sst
bp_stat = n_obs * bp_r2  # ~ χ²(k-1)
bp_p = 1 - stats.chi2.cdf(bp_stat, df=len(features))
print(f"\n3. Heteroscedasticity (Breusch-Pagan):")
print(f"   BP statistic: {bp_stat:.2f}, p={bp_p:.6f}")
print(
    f"   {'Homoscedastic' if bp_p > 0.05 else 'HETEROSCEDASTIC — variance depends on predictors'}"
)

# 4. Residual summary
print(f"\n4. Residual summary:")
print(f"   Mean: ${residuals.mean():.2f} (should be ≈0)")
print(f"   Std:  ${residuals.std():,.0f}")
print(f"   Min:  ${residuals.min():,.0f}")
print(f"   Max:  ${residuals.max():,.0f}")
print(
    f"   |Residual| > 2σ: {np.sum(np.abs(residuals) > 2*residuals.std()):,} "
    f"({np.mean(np.abs(residuals) > 2*residuals.std()):.1%})"
)
# INTERPRETATION: If residuals are heteroscedastic, OLS estimates are
# still unbiased but the standard errors are wrong — making t-tests
# and confidence intervals unreliable. Remedies: WLS (Task 8),
# heteroscedasticity-consistent (HC) standard errors, or log transform.

# ── Checkpoint 7 ─────────────────────────────────────────────────────
assert abs(residuals.mean()) < 1.0, "Residual mean should be approximately zero"
print("\n✓ Checkpoint 7 passed — residual diagnostics completed\n")


# ══════════════════════════════════════════════════════════════════════
# TASK 8: Weighted Least Squares (WLS)
# ══════════════════════════════════════════════════════════════════════
# When variance is not constant (heteroscedasticity), WLS gives each
# observation a weight inversely proportional to its variance.
# β_wls = (X'WX)⁻¹X'Wy where W = diag(1/σᵢ²)

print(f"\n=== Weighted Least Squares ===")

# Estimate weights: use fitted values as proxy for variance
abs_resid = np.abs(residuals)
w_beta = np.linalg.lstsq(X, abs_resid, rcond=None)[0]
variance_hat = np.maximum((X @ w_beta) ** 2, 1e-6)  # Estimated variance
weights = 1.0 / variance_hat

# WLS: β = (X'WX)⁻¹X'Wy
W = np.diag(weights)
# TODO: Compute X'WX (weighted X'X)
XtWX = ____  # Hint: X.T @ W @ X

# TODO: Compute X'Wy (weighted X'y)
XtWy = ____  # Hint: X.T @ W @ y

# TODO: Solve WLS: β_wls = (X'WX)⁻¹X'Wy using np.linalg.solve
beta_wls = ____  # Hint: np.linalg.solve(XtWX, XtWy)

y_hat_wls = X @ beta_wls
residuals_wls = y - y_hat_wls
ssr_wls = np.sum(residuals_wls**2)
r2_wls = 1 - ssr_wls / SST

print(f"{'Feature':<25} {'OLS β':>12} {'WLS β':>12} {'Δ':>10}")
print("─" * 62)
for i, name in enumerate(feature_names):
    delta = beta_wls[i] - beta_ols[i]
    print(f"{name:<25} {beta_ols[i]:>12,.2f} {beta_wls[i]:>12,.2f} {delta:>+10,.2f}")

print(f"\nOLS R² = {r_squared:.6f}")
print(f"WLS R² = {r2_wls:.6f}")
print(f"OLS RMSE = ${np.sqrt(SSR/n_obs):,.0f}")
print(f"WLS RMSE = ${np.sqrt(ssr_wls/n_obs):,.0f}")
# INTERPRETATION: WLS coefficients may differ from OLS when
# heteroscedasticity is present. WLS gives more reliable standard
# errors and confidence intervals. If OLS and WLS coefficients are
# similar, the heteroscedasticity doesn't much affect the point
# estimates — but the SEs are still more trustworthy from WLS.

# ── Checkpoint 8 ─────────────────────────────────────────────────────
assert len(beta_wls) == k, "WLS must have same number of coefficients"
print("\n✓ Checkpoint 8 passed — WLS computed\n")


# ══════════════════════════════════════════════════════════════════════
# TASK 9: Polynomial and Interaction Terms
# ══════════════════════════════════════════════════════════════════════
# Non-linearity: floor_area² captures diminishing/increasing returns
# Interactions: storey × area captures "premium for high-floor large flats"

print(f"\n=== Polynomial and Interaction Terms ===")

area = X_raw[:, 0]
storey = X_raw[:, 1]
lease = X_raw[:, 2]

X_enriched = np.column_stack(
    [
        np.ones(n_obs),
        area,
        storey,
        lease,
        area**2,  # Polynomial: diminishing returns on area
        storey * area,  # Interaction: high-floor premium × size
        lease * area,  # Interaction: lease × size
    ]
)
enriched_names = [
    "intercept",
    "area",
    "storey",
    "lease",
    "area²",
    "storey×area",
    "lease×area",
]
k_enriched = X_enriched.shape[1]

# Fit enriched model
beta_enriched = np.linalg.lstsq(X_enriched, y, rcond=None)[0]
y_hat_enriched = X_enriched @ beta_enriched
resid_enriched = y - y_hat_enriched
ssr_enriched = np.sum(resid_enriched**2)
# TODO: Compute R² for the enriched model
r2_enriched = ____  # Hint: 1 - ssr_enriched / SST
adj_r2_enriched = 1 - (1 - r2_enriched) * (n_obs - 1) / (n_obs - k_enriched)

# F-test: enriched vs simple model
f_improvement = ((SSR - ssr_enriched) / (k_enriched - k)) / (
    ssr_enriched / (n_obs - k_enriched)
)
f_p_improvement = 1 - stats.f.cdf(
    f_improvement, dfn=k_enriched - k, dfd=n_obs - k_enriched
)

print(f"{'Feature':<20} {'Coefficient':>14}")
print("─" * 38)
for name, coef in zip(enriched_names, beta_enriched):
    print(f"{name:<20} {coef:>14,.4f}")

print(f"\nSimple model:   R²={r_squared:.6f}, Adj R²={adj_r_squared:.6f}")
print(f"Enriched model: R²={r2_enriched:.6f}, Adj R²={adj_r2_enriched:.6f}")
print(f"F-test (enriched vs simple): F={f_improvement:.2f}, p={f_p_improvement:.2e}")
print(
    f"Enriched model is {'significantly better' if f_p_improvement < 0.05 else 'NOT significantly better'}"
)
# INTERPRETATION: The area² term captures non-linearity — perhaps
# price per sqm increases for very large flats (premium penthouses)
# or decreases (diminishing returns for huge flats). The interaction
# storey×area captures whether the storey premium is larger for
# bigger flats.

# ── Checkpoint 9 ─────────────────────────────────────────────────────
assert (
    r2_enriched >= r_squared - 0.001
), "Adding features should not decrease R² substantially"
print("\n✓ Checkpoint 9 passed — enriched model built\n")


# ══════════════════════════════════════════════════════════════════════
# TASK 10: Dummy Variable Encoding
# ══════════════════════════════════════════════════════════════════════
# Categorical variables → binary dummies. Drop one category to avoid
# the dummy variable trap (perfect multicollinearity with intercept).

print(f"\n=== Dummy Variable Encoding ===")

# Get flat types present in the data
flat_types_in_data = sorted(hdb_clean["flat_type"].unique().to_list())
print(f"Flat types: {flat_types_in_data}")

# Use "3 ROOM" as base category (most common in many towns)
base_category = "3 ROOM"
dummy_categories = [ft for ft in flat_types_in_data if ft != base_category]

# Build dummy columns
dummy_arrays = []
for ft in dummy_categories:
    dummy = (hdb_clean["flat_type"].to_numpy() == ft).astype(np.float64)
    dummy_arrays.append(dummy)

X_with_dummies = np.column_stack(
    [
        np.ones(n_obs),
        X_raw,  # Original numeric features
        np.column_stack(dummy_arrays),  # Dummy variables
    ]
)
dummy_names = (
    ["intercept"]
    + features
    + [f"flat_{ft.replace(' ', '_')}" for ft in dummy_categories]
)
k_dummy = X_with_dummies.shape[1]

# Fit model with dummies
beta_dummy = np.linalg.lstsq(X_with_dummies, y, rcond=None)[0]
y_hat_dummy = X_with_dummies @ beta_dummy
ssr_dummy = np.sum((y - y_hat_dummy) ** 2)
r2_dummy = 1 - ssr_dummy / SST
adj_r2_dummy = 1 - (1 - r2_dummy) * (n_obs - 1) / (n_obs - k_dummy)

print(f"\nBase category: {base_category}")
print(f"\n{'Feature':<30} {'Coefficient':>14}")
print("─" * 48)
for name, coef in zip(dummy_names, beta_dummy):
    print(f"{name:<30} {coef:>14,.0f}")

print(f"\nModel with dummies: R²={r2_dummy:.6f}, Adj R²={adj_r2_dummy:.6f}")
print(f"Improvement over simple: ΔR²={r2_dummy - r_squared:+.6f}")
# INTERPRETATION: Each dummy coefficient represents the price premium
# (or discount) relative to the base category (3 ROOM). For example,
# if the 5 ROOM coefficient is +$150K, then 5-room flats sell for
# $150K more than 3-room flats, all else equal.

# ── Checkpoint 10 ────────────────────────────────────────────────────
assert r2_dummy > r_squared, "Adding flat type should improve R²"
assert len(dummy_categories) == len(flat_types_in_data) - 1, "Should drop one category"
print("\n✓ Checkpoint 10 passed — dummy encoding completed\n")


# ══════════════════════════════════════════════════════════════════════
# TASK 11: Train/Test Split — Out-of-Sample Evaluation
# ══════════════════════════════════════════════════════════════════════

print(f"\n=== Train/Test Split ===")

rng = np.random.default_rng(seed=42)
n_total_obs = X_with_dummies.shape[0]
indices = rng.permutation(n_total_obs)
split_point = int(0.8 * n_total_obs)

train_idx = indices[:split_point]
test_idx = indices[split_point:]

X_train = X_with_dummies[train_idx]
y_train = y[train_idx]
X_test = X_with_dummies[test_idx]
y_test = y[test_idx]

# Fit on train
beta_train = np.linalg.lstsq(X_train, y_train, rcond=None)[0]

# Evaluate on both
y_train_pred = X_train @ beta_train
y_test_pred = X_test @ beta_train

r2_train = 1 - np.sum((y_train - y_train_pred) ** 2) / np.sum(
    (y_train - y_train.mean()) ** 2
)
# TODO: Compute out-of-sample R² on the test set
r2_test = ____  # Hint: 1 - np.sum((y_test - y_test_pred) ** 2) / np.sum((y_test - y_test.mean()) ** 2)
rmse_train = np.sqrt(np.mean((y_train - y_train_pred) ** 2))
rmse_test = np.sqrt(np.mean((y_test - y_test_pred) ** 2))
mae_train = np.mean(np.abs(y_train - y_train_pred))
mae_test = np.mean(np.abs(y_test - y_test_pred))

print(f"Train: n={len(train_idx):,}")
print(f"Test:  n={len(test_idx):,}")
print(f"\n{'Metric':<12} {'Train':>14} {'Test':>14} {'Δ':>10}")
print("─" * 54)
print(f"{'R²':<12} {r2_train:>14.6f} {r2_test:>14.6f} {r2_test-r2_train:>+10.6f}")
print(
    f"{'RMSE':<12} ${rmse_train:>12,.0f} ${rmse_test:>12,.0f} ${rmse_test-rmse_train:>+8,.0f}"
)
print(
    f"{'MAE':<12} ${mae_train:>12,.0f} ${mae_test:>12,.0f} ${mae_test-mae_train:>+8,.0f}"
)

gap = abs(r2_train - r2_test)
print(f"\nTrain-test R² gap: {gap:.4f}")
if gap < 0.02:
    print("Minimal overfitting — model generalises well")
elif gap < 0.05:
    print("Slight overfitting — consider regularisation")
else:
    print("OVERFITTING — model is too complex for the data")
# INTERPRETATION: If train R² >> test R², the model memorises training
# data instead of learning generalisable patterns. The train-test gap
# tells you whether your model complexity is appropriate.

# ── Checkpoint 11 ────────────────────────────────────────────────────
assert r2_test > 0, "Out-of-sample R² must be positive"
assert r2_train >= r2_test - 0.05, "Train R² should be ≥ test R² (approximately)"
print("\n✓ Checkpoint 11 passed — out-of-sample evaluation completed\n")


# ══════════════════════════════════════════════════════════════════════
# TASK 12: Model Comparison and Visualisation
# ══════════════════════════════════════════════════════════════════════

viz = ModelVisualizer()

# Plot 1: Actual vs predicted
fig1 = viz.scatter(
    x=y_test[:2000].tolist(),
    y=y_test_pred[:2000].tolist(),
    title="Actual vs Predicted Price (Test Set)",
    x_label="Actual Price ($)",
    y_label="Predicted Price ($)",
)
# Add perfect prediction line
fig1.add_trace(
    go.Scatter(
        x=[y_test.min(), y_test.max()],
        y=[y_test.min(), y_test.max()],
        mode="lines",
        name="Perfect prediction",
        line={"dash": "dash", "color": "red"},
    )
)
fig1.write_html("ex5_actual_vs_predicted.html")
print("\nSaved: ex5_actual_vs_predicted.html")

# Plot 2: Residual diagnostics
fig2 = make_subplots(
    rows=2,
    cols=2,
    subplot_titles=[
        "Residuals vs Fitted",
        "Residual Histogram",
        "Q-Q Plot",
        "Residuals vs Area",
    ],
)
fig2.add_trace(
    go.Scatter(
        x=y_hat[:3000],
        y=residuals[:3000],
        mode="markers",
        marker={"size": 2, "opacity": 0.3},
        name="Residuals",
    ),
    row=1,
    col=1,
)
fig2.add_hline(y=0, row=1, col=1, line_dash="dash")
fig2.add_trace(go.Histogram(x=residuals, nbinsx=50, name="Residuals"), row=1, col=2)

# Q-Q plot
sorted_resid = np.sort(residuals)
theoretical_q = stats.norm.ppf(np.linspace(0.001, 0.999, len(sorted_resid)))
fig2.add_trace(
    go.Scatter(
        x=theoretical_q,
        y=sorted_resid[:: max(1, len(sorted_resid) // 2000)],
        mode="markers",
        marker={"size": 2},
        name="Q-Q",
    ),
    row=2,
    col=1,
)
fig2.add_trace(
    go.Scatter(
        x=area[:3000],
        y=residuals[:3000],
        mode="markers",
        marker={"size": 2, "opacity": 0.3},
        name="vs Area",
    ),
    row=2,
    col=2,
)
fig2.update_layout(height=600, title="Residual Diagnostics")
fig2.write_html("ex5_residual_diagnostics.html")
print("Saved: ex5_residual_diagnostics.html")

# Model comparison table
print(f"\n=== Model Comparison Summary ===")
print(f"{'Model':<30} {'R²':>10} {'Adj R²':>10} {'k':>4}")
print("─" * 58)
print(f"{'Simple (3 features)':<30} {r_squared:>10.6f} {adj_r_squared:>10.6f} {k:>4}")
print(
    f"{'Enriched (poly+interact)':<30} {r2_enriched:>10.6f} {adj_r2_enriched:>10.6f} {k_enriched:>4}"
)
print(
    f"{'With flat type dummies':<30} {r2_dummy:>10.6f} {adj_r2_dummy:>10.6f} {k_dummy:>4}"
)

# ── Checkpoint 12 ────────────────────────────────────────────────────
print("\n✓ Checkpoint 12 passed — visualisations and comparison complete\n")


# ══════════════════════════════════════════════════════════════════════
# REFLECTION
# ══════════════════════════════════════════════════════════════════════
print("═" * 70)
print("  WHAT YOU'VE MASTERED")
print("═" * 70)
print(
    f"""
  ✓ OLS from scratch: β = (X'X)⁻¹X'y — matrix derivation implemented
  ✓ Ceteris paribus interpretation: "all else equal, one more sqm..."
  ✓ t-statistics: H₀ βⱼ=0, SE from σ̂²(X'X)⁻¹
  ✓ R², adjusted R², F-statistic — model vs intercept-only
  ✓ VIF for multicollinearity detection
  ✓ Residual diagnostics: normality, heteroscedasticity, patterns
  ✓ Breusch-Pagan test for heteroscedasticity
  ✓ WLS: weighted regression when variance is non-constant
  ✓ Polynomial terms (area²) and interactions (storey×area)
  ✓ Dummy encoding with base category to avoid dummy trap
  ✓ Train/test split: out-of-sample R², RMSE, MAE
  ✓ Model complexity trade-off: more features ≠ better prediction

  NEXT: In Exercise 6 you'll build logistic regression for binary
  classification. You'll implement the sigmoid function, maximise
  the Bernoulli log-likelihood, interpret coefficients as odds ratios,
  and perform ANOVA for multi-group comparison.
"""
)

print("\n✓ Exercise 5 complete — Linear Regression")
