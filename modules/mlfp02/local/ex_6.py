# Copyright 2026 Terrene Foundation
# SPDX-License-Identifier: Apache-2.0
"""
# ════════════════════════════════════════════════════════════════════════
# MLFP02 — Exercise 6: Logistic Regression and Classification Foundations
# ════════════════════════════════════════════════════════════════════════
#
# WHAT YOU'LL LEARN:
#   After completing this exercise, you will be able to:
#   - Implement the sigmoid function with numerical stability
#   - Build logistic regression from scratch via MLE (Bernoulli likelihood)
#   - Compare from-scratch implementation with sklearn LogisticRegression
#   - Interpret coefficients as odds ratios: exp(β) = multiplicative change
#   - Optimise classification threshold using a cost matrix
#   - Evaluate with accuracy, confusion matrix, precision, recall, F1
#   - Plot and interpret ROC curves and compute AUC
#   - Plot precision-recall curves for imbalanced classes
#   - Assess model calibration with calibration curves and Brier score
#   - Perform one-way ANOVA and post-hoc Tukey HSD for multi-group comparison
#   - Visualise all classification and ANOVA results
#
# PREREQUISITES: Complete Exercise 5 — you should understand OLS,
#   t-statistics, the connection between linear models and the sigmoid,
#   and scipy.optimize.minimize.
#
# ESTIMATED TIME: ~170 minutes
#
# TASKS:
#    1. Load HDB data, create binary classification target
#    2. Implement sigmoid function and verify properties
#    3. Logistic regression from scratch (MLE on Bernoulli log-likelihood)
#    4. Compare with sklearn LogisticRegression
#    5. Interpret coefficients as odds ratios
#    6. Threshold optimisation using a cost matrix
#    7. Confusion matrix, precision, recall, F1
#    8. ROC curve and AUC
#    9. Precision-recall curve
#   10. Model calibration: calibration curve and Brier score
#   11. ANOVA: compare resale prices across flat types + Tukey HSD
#   12. Visualise and synthesise findings
#
# DATASET: HDB resale flat transactions (Singapore)
#   Source: data.gov.sg — public housing resale records, 2020+
#   Binary target: high_price = 1 if resale_price > median, else 0
#
# THEORY:
#   Logistic regression: log(P/(1-P)) = β₀ + β₁x₁ + ...
#   Sigmoid: P(y=1|X) = 1 / (1 + exp(-(Xβ)))
#   MLE: maximise Σ[yᵢ log(pᵢ) + (1-yᵢ) log(1-pᵢ)]
#   Odds ratio: exp(βⱼ) = multiplicative change in odds per unit of xⱼ
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
from scipy.optimize import minimize
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score,
    roc_curve,
    auc,
    precision_recall_curve,
    brier_score_loss,
)

from shared import MLFPDataLoader


# ── Data Loading ──────────────────────────────────────────────────────

loader = MLFPDataLoader()
hdb = loader.load("mlfp01", "hdb_resale.parquet")

print("=" * 70)
print("  MLFP02 Exercise 6: Logistic Regression and Classification")
print("=" * 70)
print(f"\n  Data loaded: hdb_resale.parquet ({hdb.shape[0]:,} rows)")

hdb_recent = hdb.filter(pl.col("month").str.to_date("%Y-%m") >= pl.date(2020, 1, 1))
print(f"  Filtered to 2020+: {hdb_recent.height:,} transactions")


# ══════════════════════════════════════════════════════════════════════
# TASK 1: Create Binary Classification Target
# ══════════════════════════════════════════════════════════════════════
# Target: high_price = 1 if resale_price > median
# This gives a balanced 50/50 split (baseline accuracy = 50%).

median_price = hdb_recent["resale_price"].median()
print(f"\n  Median price: ${median_price:,.0f}")

hdb_model = hdb_recent.with_columns(
    (pl.col("resale_price") > median_price).cast(pl.Int32).alias("high_price"),
    (
        (
            pl.col("storey_range").str.extract(r"(\d+)", 1).cast(pl.Float64)
            + pl.col("storey_range").str.extract(r"TO (\d+)", 1).cast(pl.Float64)
        )
        / 2.0
    ).alias("storey_mid"),
    (
        99
        - (
            pl.col("month").str.to_date("%Y-%m").dt.year()
            - pl.col("lease_commence_date")
        )
    )
    .cast(pl.Float64)
    .alias("remaining_lease"),
).drop_nulls(subset=["floor_area_sqm", "storey_mid", "high_price", "remaining_lease"])

n_total = hdb_model.height
n_positive = hdb_model.filter(pl.col("high_price") == 1).height
print(f"  Total: {n_total:,}, Positive: {n_positive:,} ({n_positive/n_total:.1%})")

# Prepare arrays
feature_cols = ["floor_area_sqm", "storey_mid", "remaining_lease"]
X_raw = hdb_model.select(feature_cols).to_numpy().astype(np.float64)
y = hdb_model["high_price"].to_numpy().astype(np.float64)

# Standardise features for optimisation stability
X_mean = X_raw.mean(axis=0)
X_std = X_raw.std(axis=0)
X_scaled = (X_raw - X_mean) / X_std
n_obs, n_feat = X_scaled.shape

# Add intercept
X = np.column_stack([np.ones(n_obs), X_scaled])
feature_names = ["intercept"] + feature_cols

# ── Checkpoint 1 ─────────────────────────────────────────────────────
assert abs(n_positive / n_total - 0.5) < 0.02, "Median split should give ~50/50"
assert X.shape == (n_obs, n_feat + 1), "Design matrix shape incorrect"
print("\n✓ Checkpoint 1 passed — binary target created\n")


# ══════════════════════════════════════════════════════════════════════
# TASK 2: Sigmoid Function — Implementation and Properties
# ══════════════════════════════════════════════════════════════════════
# σ(z) = 1 / (1 + exp(-z))
# Numerically stable: for z >> 0, use 1/(1+exp(-z));
# for z << 0, use exp(z)/(1+exp(z)) to avoid overflow.


def sigmoid(z: np.ndarray) -> np.ndarray:
    """Numerically stable sigmoid function."""
    result = np.zeros_like(z, dtype=np.float64)
    pos = z >= 0
    neg = ~pos
    # TODO: Implement sigmoid for positive z: 1 / (1 + exp(-z))
    result[pos] = ____  # Hint: 1.0 / (1.0 + np.exp(-z[pos]))
    exp_z = np.exp(z[neg])
    # TODO: Implement sigmoid for negative z: exp(z) / (1 + exp(z))
    result[neg] = ____  # Hint: exp_z / (1.0 + exp_z)
    return result


# Verify properties
print(f"\n=== Sigmoid Properties ===")
test_values = [-10, -5, -1, 0, 1, 5, 10]
for z in test_values:
    s = sigmoid(np.array([z]))[0]
    print(f"  σ({z:>3}) = {s:.6f}")

# Key properties
assert abs(sigmoid(np.array([0.0]))[0] - 0.5) < 1e-10, "σ(0) must equal 0.5"
assert sigmoid(np.array([100.0]))[0] > 0.999, "σ(large) must be near 1"
assert sigmoid(np.array([-100.0]))[0] < 0.001, "σ(very negative) must be near 0"

# Symmetry: σ(-z) = 1 - σ(z)
z_test = np.array([2.5])
assert abs(sigmoid(-z_test)[0] - (1 - sigmoid(z_test)[0])) < 1e-10, "Symmetry must hold"

# Derivative: σ'(z) = σ(z)(1 - σ(z))
s_val = sigmoid(z_test)[0]
# TODO: Compute analytical derivative of sigmoid
deriv_analytical = ____  # Hint: s_val * (1 - s_val)
deriv_numerical = (sigmoid(z_test + 1e-7)[0] - sigmoid(z_test - 1e-7)[0]) / (2e-7)
print(f"\n  Derivative at z=2.5:")
print(f"  Analytical σ'(z) = σ(z)(1-σ(z)) = {deriv_analytical:.6f}")
print(f"  Numerical:  {deriv_numerical:.6f}")
# INTERPRETATION: The sigmoid maps any real number to (0, 1) — perfect
# for modelling probabilities. At z=0, the probability is exactly 50%.
# The derivative is maximal at z=0 (steepest change) and diminishes
# toward the extremes — the model is most "uncertain" near 50%.

# ── Checkpoint 2 ─────────────────────────────────────────────────────
assert abs(deriv_analytical - deriv_numerical) < 1e-5, "Derivative check must pass"
print("\n✓ Checkpoint 2 passed — sigmoid implementation verified\n")


# ══════════════════════════════════════════════════════════════════════
# TASK 3: Logistic Regression from Scratch (MLE)
# ══════════════════════════════════════════════════════════════════════
# Bernoulli log-likelihood:
#   ℓ(β) = Σ[yᵢ log(pᵢ) + (1-yᵢ) log(1-pᵢ)]
# where pᵢ = σ(xᵢ'β)


def neg_log_likelihood_logistic(
    beta: np.ndarray, X: np.ndarray, y: np.ndarray
) -> float:
    """Negative log-likelihood for logistic regression."""
    z = X @ beta
    p = sigmoid(z)
    p = np.clip(p, 1e-15, 1 - 1e-15)
    # TODO: Compute Bernoulli log-likelihood: Σ[y*log(p) + (1-y)*log(1-p)]
    ll = ____  # Hint: np.sum(y * np.log(p) + (1 - y) * np.log(1 - p))
    return -ll


def neg_ll_gradient(beta: np.ndarray, X: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Gradient of the negative log-likelihood."""
    z = X @ beta
    p = sigmoid(z)
    # TODO: Return gradient of negative log-likelihood: -X'(y - p)
    return ____  # Hint: -X.T @ (y - p)


# Initial guess: zeros
beta0 = np.zeros(n_feat + 1)

# TODO: Optimise neg_log_likelihood_logistic using scipy.optimize.minimize
# Use method="L-BFGS-B", pass jac=neg_ll_gradient and args=(X, y)
result = ____  # Hint: minimize(neg_log_likelihood_logistic, beta0, args=(X, y), method="L-BFGS-B", jac=neg_ll_gradient, options={"maxiter": 1000, "ftol": 1e-12})

beta_scratch = result.x
ll_scratch = -result.fun

print(f"\n=== Logistic Regression from Scratch ===")
print(f"Converged: {result.success}")
print(f"Log-likelihood: {ll_scratch:.2f}")
print(f"\n{'Feature':<20} {'Coefficient':>14}")
print("─" * 38)
for name, coef in zip(feature_names, beta_scratch):
    print(f"{name:<20} {coef:>14.6f}")

# Predictions
p_scratch = sigmoid(X @ beta_scratch)
y_pred_scratch = (p_scratch >= 0.5).astype(int)
acc_scratch = accuracy_score(y, y_pred_scratch)
print(f"\nAccuracy (from scratch): {acc_scratch:.4f} ({acc_scratch:.1%})")

# ── Checkpoint 3 ─────────────────────────────────────────────────────
assert result.success, "Logistic regression must converge"
assert (
    acc_scratch > 0.55
), f"Accuracy should be above baseline 50%, got {acc_scratch:.1%}"
print("\n✓ Checkpoint 3 passed — logistic regression from scratch\n")


# ══════════════════════════════════════════════════════════════════════
# TASK 4: Compare with sklearn
# ══════════════════════════════════════════════════════════════════════

sklearn_model = LogisticRegression(
    penalty=None,
    max_iter=1000,
    solver="lbfgs",
    tol=1e-8,
)
sklearn_model.fit(X_scaled, y)

beta_sklearn = np.concatenate([[sklearn_model.intercept_[0]], sklearn_model.coef_[0]])

print(f"\n=== Comparison: Scratch vs sklearn ===")
print(f"{'Feature':<20} {'Scratch':>14} {'sklearn':>14} {'|Δ|':>10}")
print("─" * 62)
for i, name in enumerate(feature_names):
    diff = abs(beta_scratch[i] - beta_sklearn[i])
    print(f"{name:<20} {beta_scratch[i]:>14.6f} {beta_sklearn[i]:>14.6f} {diff:>10.6f}")

acc_sklearn = sklearn_model.score(X_scaled, y)
print(f"\nAccuracy (scratch): {acc_scratch:.6f}")
print(f"Accuracy (sklearn): {acc_sklearn:.6f}")
# INTERPRETATION: The coefficients should agree closely. Small differences
# arise from convergence tolerance and solver algorithms.

# ── Checkpoint 4 ─────────────────────────────────────────────────────
assert np.allclose(
    beta_scratch, beta_sklearn, atol=0.1
), "Scratch and sklearn coefficients should agree"
print("\n✓ Checkpoint 4 passed — implementation matches sklearn\n")


# ══════════════════════════════════════════════════════════════════════
# TASK 5: Odds Ratio Interpretation
# ══════════════════════════════════════════════════════════════════════
# Odds = P(y=1) / P(y=0) = P / (1-P)
# Log-odds: log(odds) = Xβ
# Odds ratio for feature j: exp(βⱼ) = multiplicative change in odds
# per one-unit change in xⱼ (on the scaled feature)

print(f"\n=== Odds Ratio Interpretation ===")

# Convert to original scale: βⱼ_original = βⱼ_scaled / σⱼ
beta_original = np.zeros(n_feat + 1)
beta_original[0] = beta_scratch[0] - np.sum(beta_scratch[1:] * X_mean / X_std)
beta_original[1:] = beta_scratch[1:] / X_std

print(f"\n{'Feature':<20} {'β (original)':>14} {'Odds Ratio':>12} {'Interpretation'}")
print("─" * 80)
for i in range(1, n_feat + 1):
    # TODO: Compute the odds ratio for feature i: exp(β_original[i])
    or_val = ____  # Hint: np.exp(beta_original[i])
    name = feature_names[i]
    if or_val > 1:
        interp = f"1-unit increase → {(or_val-1)*100:.1f}% higher odds"
    else:
        interp = f"1-unit increase → {(1-or_val)*100:.1f}% lower odds"
    print(f"{name:<20} {beta_original[i]:>14.6f} {or_val:>12.4f} {interp}")

# Practical examples
print(f"\n--- Practical Examples ---")
for feat, units, factor in [
    ("floor_area_sqm", "10 sqm", 10),
    ("storey_mid", "5 storeys", 5),
    ("remaining_lease", "10 years", 10),
]:
    idx = feature_names.index(feat)
    or_change = np.exp(beta_original[idx] * factor)
    print(
        f"  +{units} of {feat}: odds multiply by {or_change:.3f} "
        f"({(or_change-1)*100:+.1f}% change)"
    )
# INTERPRETATION: An odds ratio of 1.5 for floor_area_sqm means each
# extra sqm multiplies the odds of being "high price" by 1.5. Odds
# ratios are multiplicative, not additive — they compound with each
# unit increase.

# ── Checkpoint 5 ─────────────────────────────────────────────────────
assert np.exp(beta_original[1]) > 1, "Larger area should increase odds of high price"
print("\n✓ Checkpoint 5 passed — odds ratios interpreted\n")


# ══════════════════════════════════════════════════════════════════════
# TASK 6: Threshold Optimisation with Cost Matrix
# ══════════════════════════════════════════════════════════════════════
# Default threshold is 0.5, but optimal threshold depends on costs.
# Cost matrix: what's the cost of a false positive vs false negative?

print(f"\n=== Threshold Optimisation ===")

# Scenario: property valuation
# FP (predict high, actually low): buyer overpays → cost = $30K
# FN (predict low, actually high): seller underprices → cost = $50K
cost_fp = 30_000
cost_fn = 50_000

thresholds = np.linspace(0.1, 0.9, 81)
total_costs = []
accuracies = []
f1_scores_list = []

for t in thresholds:
    y_pred_t = (p_scratch >= t).astype(int)
    cm = confusion_matrix(y, y_pred_t)
    tn, fp, fn, tp = cm.ravel()
    cost = fp * cost_fp + fn * cost_fn
    total_costs.append(cost)
    accuracies.append(accuracy_score(y, y_pred_t))
    f1_scores_list.append(f1_score(y, y_pred_t, zero_division=0))

# TODO: Find the index of the threshold with minimum total cost
optimal_idx = ____  # Hint: np.argmin(total_costs)
optimal_threshold = thresholds[optimal_idx]
optimal_cost = total_costs[optimal_idx]

# Also find threshold that maximises F1
f1_idx = np.argmax(f1_scores_list)
f1_threshold = thresholds[f1_idx]

print(f"Cost matrix: FP=${cost_fp:,}, FN=${cost_fn:,}")
print(f"\nOptimal threshold (min cost): {optimal_threshold:.3f}")
print(f"  Total cost: ${optimal_cost:,.0f}")
print(f"  Accuracy at this threshold: {accuracies[optimal_idx]:.4f}")
print(f"\nF1-optimal threshold: {f1_threshold:.3f}")
print(f"  F1 at this threshold: {f1_scores_list[f1_idx]:.4f}")
print(f"\nDefault threshold (0.5):")
print(f"  Cost: ${total_costs[40]:,.0f}")
print(f"  Accuracy: {accuracies[40]:.4f}")
# INTERPRETATION: When FN costs more than FP, the optimal threshold
# is below 0.5 — we'd rather predict "high price" more aggressively
# to avoid missing expensive flats. The threshold should reflect the
# business cost of each type of error.

# ── Checkpoint 6 ─────────────────────────────────────────────────────
assert 0 < optimal_threshold < 1, "Optimal threshold must be valid"
assert optimal_cost <= total_costs[40], "Optimal cost must be ≤ default cost"
print("\n✓ Checkpoint 6 passed — threshold optimisation completed\n")


# ══════════════════════════════════════════════════════════════════════
# TASK 7: Confusion Matrix, Precision, Recall, F1
# ══════════════════════════════════════════════════════════════════════

print(f"\n=== Classification Metrics (threshold={optimal_threshold:.3f}) ===")

y_pred_opt = (p_scratch >= optimal_threshold).astype(int)
cm = confusion_matrix(y, y_pred_opt)
tn, fp, fn, tp = cm.ravel()

prec = precision_score(y, y_pred_opt)
rec = recall_score(y, y_pred_opt)
f1 = f1_score(y, y_pred_opt)
acc = accuracy_score(y, y_pred_opt)

print(f"\nConfusion Matrix:")
print(f"              Predicted Low  Predicted High")
print(f"  Actual Low    {tn:>10,}    {fp:>10,}")
print(f"  Actual High   {fn:>10,}    {tp:>10,}")

print(f"\n{'Metric':<15} {'Value':>10}")
print("─" * 28)
print(f"{'Accuracy':<15} {acc:>10.4f}")
print(f"{'Precision':<15} {prec:>10.4f}")
print(f"{'Recall':<15} {rec:>10.4f}")
print(f"{'F1 Score':<15} {f1:>10.4f}")
print(f"{'True Positives':<15} {tp:>10,}")
print(f"{'False Positives':<15} {fp:>10,}")
print(f"{'True Negatives':<15} {tn:>10,}")
print(f"{'False Negatives':<15} {fn:>10,}")
# INTERPRETATION: Precision = "of those I predicted high, how many
# actually were?" Recall = "of those actually high, how many did I
# catch?" F1 balances both. In this property context, recall matters
# more if missing high-value flats is costly.

# ── Checkpoint 7 ─────────────────────────────────────────────────────
assert tp + fp + tn + fn == n_obs, "Confusion matrix must sum to n"
assert 0 < prec <= 1, "Precision must be valid"
assert 0 < rec <= 1, "Recall must be valid"
print("\n✓ Checkpoint 7 passed — classification metrics computed\n")


# ══════════════════════════════════════════════════════════════════════
# TASK 8: ROC Curve and AUC
# ══════════════════════════════════════════════════════════════════════

fpr, tpr, roc_thresholds = roc_curve(y, p_scratch)
# TODO: Compute AUC from fpr and tpr arrays
roc_auc = ____  # Hint: auc(fpr, tpr)

# Find the threshold on ROC closest to top-left corner
distances = np.sqrt((0 - fpr) ** 2 + (1 - tpr) ** 2)
roc_optimal_idx = np.argmin(distances)
roc_optimal_threshold = roc_thresholds[roc_optimal_idx]

print(f"\n=== ROC Curve ===")
print(f"AUC = {roc_auc:.4f}")
print(f"ROC-optimal threshold (closest to top-left): {roc_optimal_threshold:.4f}")
print(f"  at FPR={fpr[roc_optimal_idx]:.4f}, TPR={tpr[roc_optimal_idx]:.4f}")
print(f"\nAUC interpretation:")
if roc_auc > 0.9:
    print(f"  Excellent discrimination")
elif roc_auc > 0.8:
    print(f"  Good discrimination")
elif roc_auc > 0.7:
    print(f"  Fair discrimination")
else:
    print(f"  Poor discrimination")

# ── Checkpoint 8 ─────────────────────────────────────────────────────
assert 0.5 <= roc_auc <= 1.0, "AUC must be between 0.5 and 1.0"
print("\n✓ Checkpoint 8 passed — ROC and AUC computed\n")


# ══════════════════════════════════════════════════════════════════════
# TASK 9: Precision-Recall Curve
# ══════════════════════════════════════════════════════════════════════

prec_curve, rec_curve, pr_thresholds = precision_recall_curve(y, p_scratch)
# TODO: Compute area under the precision-recall curve
pr_auc = ____  # Hint: auc(rec_curve, prec_curve)

print(f"\n=== Precision-Recall Curve ===")
print(f"PR-AUC = {pr_auc:.4f}")
print(f"Baseline (random): {n_positive/n_total:.4f}")
print(f"PR-AUC / baseline: {pr_auc / (n_positive/n_total):.2f}x better than random")
# INTERPRETATION: The PR curve is more informative than ROC when
# classes are imbalanced. For balanced classes (like ours), both
# tell a similar story. PR-AUC > baseline means the model does better
# than random guessing.

# ── Checkpoint 9 ─────────────────────────────────────────────────────
assert pr_auc > n_positive / n_total, "PR-AUC should beat random baseline"
print("\n✓ Checkpoint 9 passed — precision-recall curve computed\n")


# ══════════════════════════════════════════════════════════════════════
# TASK 10: Model Calibration — Calibration Curve and Brier Score
# ══════════════════════════════════════════════════════════════════════
# A well-calibrated model: when it predicts P=0.7, ~70% of those
# observations are actually positive. Brier score = mean((p - y)²).

print(f"\n=== Model Calibration ===")

# TODO: Compute the Brier score (mean squared error between probabilities and labels)
brier = ____  # Hint: brier_score_loss(y, p_scratch)

# Build calibration curve manually
n_bins = 10
bin_edges = np.linspace(0, 1, n_bins + 1)
cal_predicted = []
cal_observed = []
cal_counts = []

for i in range(n_bins):
    mask = (p_scratch >= bin_edges[i]) & (p_scratch < bin_edges[i + 1])
    if mask.sum() > 0:
        cal_predicted.append(p_scratch[mask].mean())
        cal_observed.append(y[mask].mean())
        cal_counts.append(mask.sum())

print(f"Brier score: {brier:.6f} (lower = better, 0 = perfect)")
print(f"Max Brier (random): 0.25")
print(f"Brier skill: {1 - brier / 0.25:.4f} (1 = perfect, 0 = random)")
print(f"\n{'Bin':>4} {'Predicted':>12} {'Observed':>12} {'Count':>8} {'Gap':>8}")
print("─" * 48)
for i in range(len(cal_predicted)):
    gap = abs(cal_predicted[i] - cal_observed[i])
    print(
        f"{i+1:>4} {cal_predicted[i]:>12.4f} {cal_observed[i]:>12.4f} "
        f"{cal_counts[i]:>8,} {gap:>8.4f}"
    )
# INTERPRETATION: Good calibration = predicted probabilities match
# observed frequencies. If predicted=0.8 but observed=0.6, the model
# is overconfident. Calibration matters when you use probabilities
# for decision-making (not just rankings).

# ── Checkpoint 10 ────────────────────────────────────────────────────
assert 0 <= brier <= 1, "Brier score must be between 0 and 1"
assert brier < 0.25, "Model should beat random (Brier < 0.25)"
print("\n✓ Checkpoint 10 passed — calibration assessed\n")


# ══════════════════════════════════════════════════════════════════════
# TASK 11: ANOVA — Multi-Group Comparison + Tukey HSD
# ══════════════════════════════════════════════════════════════════════
# One-way ANOVA generalises the t-test to 3+ groups.
# H₀: μ₁ = μ₂ = μ₃ = ... = μₖ (all group means are equal)
# H₁: at least one mean differs

print(f"\n=== One-Way ANOVA: Resale Price by Flat Type ===")

flat_types = ["2 ROOM", "3 ROOM", "4 ROOM", "5 ROOM", "EXECUTIVE"]
anova_groups = []
anova_labels = []

for ft in flat_types:
    group = (
        hdb_recent.filter(pl.col("flat_type") == ft)["resale_price"]
        .to_numpy()
        .astype(np.float64)
    )
    if len(group) > 10:
        anova_groups.append(group)
        anova_labels.append(ft)
        print(
            f"  {ft:<12}: n={len(group):>6,}, mean=${group.mean():>10,.0f}, std=${group.std():>8,.0f}"
        )

# TODO: Run one-way ANOVA across all anova_groups
f_anova, p_anova = ____  # Hint: stats.f_oneway(*anova_groups)
print(f"\nANOVA F-statistic: {f_anova:.2f}")
print(f"p-value: {p_anova:.2e}")
print(
    f"{'SIGNIFICANT' if p_anova < 0.05 else 'NOT significant'}: "
    f"{'at least one flat type has a different mean price' if p_anova < 0.05 else 'no evidence of difference'}"
)

# Effect size: eta-squared
ss_between = sum(
    len(g) * (g.mean() - np.concatenate(anova_groups).mean()) ** 2 for g in anova_groups
)
ss_total = sum(
    np.sum((g - np.concatenate(anova_groups).mean()) ** 2) for g in anova_groups
)
eta_squared = ss_between / ss_total
print(
    f"Effect size (η²): {eta_squared:.4f} ({eta_squared:.1%} of variance explained by flat type)"
)

# Post-hoc: Tukey HSD (pairwise comparisons)
print(f"\n--- Tukey HSD Post-Hoc Comparisons ---")
all_data = np.concatenate(anova_groups)
n_all = len(all_data)
k_groups = len(anova_groups)
ms_within = sum(np.sum((g - g.mean()) ** 2) for g in anova_groups) / (n_all - k_groups)

print(
    f"{'Comparison':<25} {'Diff ($)':>12} {'SE':>10} {'q':>8} {'p-value':>10} {'Sig':>6}"
)
print("─" * 75)
for i in range(k_groups):
    for j in range(i + 1, k_groups):
        diff = anova_groups[j].mean() - anova_groups[i].mean()
        se = np.sqrt(
            ms_within * (1 / len(anova_groups[i]) + 1 / len(anova_groups[j])) / 2
        )
        q_stat = abs(diff) / se
        n_comparisons = k_groups * (k_groups - 1) / 2
        t_stat = diff / (
            np.sqrt(ms_within)
            * np.sqrt(1 / len(anova_groups[i]) + 1 / len(anova_groups[j]))
        )
        p_pair = 2 * (1 - stats.t.cdf(abs(t_stat), df=n_all - k_groups))
        p_bonf = min(p_pair * n_comparisons, 1.0)
        sig = (
            "***"
            if p_bonf < 0.001
            else "**" if p_bonf < 0.01 else "*" if p_bonf < 0.05 else "ns"
        )
        label = f"{anova_labels[i]} vs {anova_labels[j]}"
        print(
            f"{label:<25} ${diff:>10,.0f} {se:>10,.0f} {q_stat:>8.2f} {p_bonf:>10.4f} {sig:>6}"
        )
# INTERPRETATION: ANOVA tells you SOME group differs. Tukey HSD tells
# you WHICH pairs differ. In property, this shows the price premium
# between flat types — critical for valuation models. Executive flats
# command a premium over 3-room; the question is how much.

# ── Checkpoint 11 ────────────────────────────────────────────────────
assert f_anova > 0, "F-statistic must be positive"
assert 0 <= eta_squared <= 1, "Eta-squared must be between 0 and 1"
print("\n✓ Checkpoint 11 passed — ANOVA and Tukey HSD completed\n")


# ══════════════════════════════════════════════════════════════════════
# TASK 12: Visualise All Results
# ══════════════════════════════════════════════════════════════════════

viz = ModelVisualizer()

# Plot 1: ROC curve
fig1 = go.Figure()
fig1.add_trace(go.Scatter(x=fpr, y=tpr, name=f"ROC (AUC={roc_auc:.3f})"))
fig1.add_trace(
    go.Scatter(
        x=[0, 1], y=[0, 1], name="Random", line={"dash": "dash", "color": "grey"}
    )
)
fig1.add_trace(
    go.Scatter(
        x=[fpr[roc_optimal_idx]],
        y=[tpr[roc_optimal_idx]],
        mode="markers",
        name=f"Optimal (t={roc_optimal_threshold:.3f})",
        marker={"size": 12, "color": "red"},
    )
)
fig1.update_layout(title="ROC Curve", xaxis_title="FPR", yaxis_title="TPR")
fig1.write_html("ex6_roc_curve.html")
print("\nSaved: ex6_roc_curve.html")

# Plot 2: Calibration curve
fig2 = go.Figure()
fig2.add_trace(
    go.Scatter(x=cal_predicted, y=cal_observed, mode="markers+lines", name="Model")
)
fig2.add_trace(
    go.Scatter(
        x=[0, 1], y=[0, 1], name="Perfect", line={"dash": "dash", "color": "grey"}
    )
)
fig2.update_layout(
    title=f"Calibration Curve (Brier={brier:.4f})",
    xaxis_title="Predicted Probability",
    yaxis_title="Observed Frequency",
)
fig2.write_html("ex6_calibration.html")
print("Saved: ex6_calibration.html")

# Plot 3: Threshold vs cost curve
fig3 = go.Figure()
fig3.add_trace(
    go.Scatter(x=thresholds, y=[c / 1e6 for c in total_costs], name="Total Cost ($M)")
)
fig3.add_vline(
    x=optimal_threshold,
    line_dash="dash",
    annotation_text=f"Optimal t={optimal_threshold:.2f}",
)
fig3.add_vline(
    x=0.5, line_dash="dot", line_color="red", annotation_text="Default t=0.5"
)
fig3.update_layout(
    title="Cost vs Classification Threshold",
    xaxis_title="Threshold",
    yaxis_title="Total Cost ($M)",
)
fig3.write_html("ex6_threshold_cost.html")
print("Saved: ex6_threshold_cost.html")

# Plot 4: ANOVA box plots
fig4 = go.Figure()
for label, group in zip(anova_labels, anova_groups):
    fig4.add_trace(go.Box(y=group[:5000], name=label))
fig4.update_layout(
    title="Resale Price Distribution by Flat Type (ANOVA)",
    yaxis_title="Resale Price ($)",
)
fig4.write_html("ex6_anova_boxplot.html")
print("Saved: ex6_anova_boxplot.html")

# ── Checkpoint 12 ────────────────────────────────────────────────────
print("\n✓ Checkpoint 12 passed — visualisations saved\n")


# ══════════════════════════════════════════════════════════════════════
# REFLECTION
# ══════════════════════════════════════════════════════════════════════
print("═" * 70)
print("  WHAT YOU'VE MASTERED")
print("═" * 70)
print(
    f"""
  ✓ Sigmoid: σ(z) = 1/(1+exp(-z)), numerically stable implementation
  ✓ Logistic regression from scratch: Bernoulli MLE with gradient
  ✓ sklearn comparison: verified implementation correctness
  ✓ Odds ratios: exp(β) = multiplicative change in odds per unit
  ✓ Threshold optimisation: cost matrix drives the decision boundary
  ✓ Confusion matrix: TP, FP, TN, FN and derived metrics
  ✓ Precision, recall, F1: trade-offs in classification
  ✓ ROC curve + AUC: discrimination ability across all thresholds
  ✓ Precision-recall curve: better for imbalanced classes
  ✓ Calibration: predicted probabilities match observed frequencies
  ✓ Brier score: overall measure of probabilistic accuracy
  ✓ One-way ANOVA: F-test for 3+ group means, η² effect size
  ✓ Tukey HSD: pairwise comparisons with multiple testing correction

  NEXT: In Exercise 7 you'll implement CUPED variance reduction
  for A/B tests, Bayesian A/B testing with posterior probabilities,
  sequential testing with always-valid p-values, and Difference-in-
  Differences for when randomisation is impossible.
"""
)

print("\n✓ Exercise 6 complete — Logistic Regression and Classification")
