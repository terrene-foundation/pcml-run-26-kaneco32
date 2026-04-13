# Copyright 2026 Terrene Foundation
# SPDX-License-Identifier: Apache-2.0
"""
# ════════════════════════════════════════════════════════════════════════
# MLFP03 — Exercise 5: Model Evaluation, Class Imbalance, and
#                        Calibration
# ════════════════════════════════════════════════════════════════════════
#
# WHAT YOU'LL LEARN:
#   After completing this exercise, you will be able to:
#   - Diagnose why accuracy fails on imbalanced datasets
#   - Explain SMOTE's failure modes (Lipschitz violation, noise
#     amplification, high-dimensional collapse)
#   - Apply cost-sensitive learning using a business cost matrix
#   - Implement Focal Loss (γ parameter) to down-weight easy examples
#   - Optimise classification threshold from cost-matrix principles
#   - Compute business ROI from threshold selection
#   - Calibrate model probabilities with Platt scaling and isotonic
#     regression
#   - Compare calibration methods using reliability diagrams
#   - Build a complete metrics taxonomy (classification + regression)
#
# PREREQUISITES:
#   - MLFP03 Exercise 4 (gradient boosting, AUC-PR)
#   - MLFP02 Module (Bayesian thinking — connects to calibrated probs)
#
# ESTIMATED TIME: ~150-180 minutes
#
# TASKS:
#   1.  Establish baseline (no imbalance handling)
#   2.  Complete metrics taxonomy: precision, recall, F1, specificity
#   3.  SMOTE oversampling — why it often fails in practice
#   4.  Cost-sensitive learning (sample weights)
#   5.  Focal Loss (derive γ parameter effect)
#   6.  Threshold optimisation from cost matrix
#   7.  Business ROI analysis of threshold selection
#   8.  Platt scaling calibration
#   9.  Isotonic regression calibration
#   10. Calibration comparison with reliability diagrams
#   11. Final comparison: all imbalance strategies
#   12. Production recommendation
#
# DATASET: Singapore credit scoring (from MLFP02)
#   Target: default (12% positive rate — realistic banking imbalance)
#   Business cost matrix: FP = $100, FN = $10,000
#   The 100:1 cost ratio drives every design decision in this exercise.
#
# KEY FORMULAS:
#   Precision = TP / (TP + FP)
#   Recall = TP / (TP + FN)
#   F1 = 2 * Precision * Recall / (Precision + Recall)
#   Focal Loss: FL(p_t) = -α_t * (1-p_t)^γ * log(p_t)
#   Brier Score: BS = (1/N) * Σ(p_i - y_i)²
#
# ════════════════════════════════════════════════════════════════════════
"""
from __future__ import annotations

import numpy as np
import polars as pl
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    average_precision_score,
    roc_auc_score,
    f1_score,
    precision_score,
    recall_score,
    precision_recall_curve,
    classification_report,
    brier_score_loss,
    confusion_matrix,
    accuracy_score,
)
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
import lightgbm as lgb

from kailash_ml import PreprocessingPipeline, ModelVisualizer
from kailash_ml.interop import to_sklearn_input

from shared import MLFPDataLoader


# ── Data Loading ──────────────────────────────────────────────────────

loader = MLFPDataLoader()
credit = loader.load("mlfp02", "sg_credit_scoring.parquet")

pipeline = PreprocessingPipeline()
result = pipeline.setup(
    credit,
    target="default",
    seed=42,
    normalize=False,
    categorical_encoding="ordinal",
)

X_train, y_train, col_info = to_sklearn_input(
    result.train_data,
    feature_columns=[c for c in result.train_data.columns if c != "default"],
    target_column="default",
)
X_test, y_test, _ = to_sklearn_input(
    result.test_data,
    feature_columns=[c for c in result.test_data.columns if c != "default"],
    target_column="default",
)

pos_rate = y_train.mean()
print(
    f"Default rate: {pos_rate:.2%} "
    f"(imbalance ratio: {(1 - pos_rate) / pos_rate:.0f}:1)"
)

# ── Checkpoint 1 ─────────────────────────────────────────────────────
assert 0 < pos_rate < 0.5, "Default rate should be a minority class"
assert X_train.shape[0] > 0, "Training set should not be empty"
# INTERPRETATION: A 12% default rate means 88% of the data is class 0.
# A model that predicts "no default" for every applicant gets 88% accuracy!
# This is why accuracy is the wrong metric for credit scoring.
print("\n✓ Checkpoint 1 passed — imbalanced data confirmed\n")


# ══════════════════════════════════════════════════════════════════════
# TASK 1: Baseline — no imbalance handling
# ══════════════════════════════════════════════════════════════════════

baseline = lgb.LGBMClassifier(n_estimators=300, random_state=42, verbose=-1)
baseline.fit(X_train, y_train)
y_proba_base = baseline.predict_proba(X_test)[:, 1]
y_pred_base = baseline.predict(X_test)

print(f"\n=== Baseline (no correction) ===")
print(f"Accuracy: {accuracy_score(y_test, y_pred_base):.4f}")
print(f"AUC-ROC:  {roc_auc_score(y_test, y_proba_base):.4f}")
print(f"AUC-PR:   {average_precision_score(y_test, y_proba_base):.4f}")
print(f"Brier:    {brier_score_loss(y_test, y_proba_base):.4f}")
print(f"F1:       {f1_score(y_test, y_pred_base):.4f}")


# ══════════════════════════════════════════════════════════════════════
# TASK 2: Complete metrics taxonomy
# ══════════════════════════════════════════════════════════════════════
# Understanding every metric and WHEN to use each one is critical.

print(f"\n=== Complete Metrics Taxonomy ===")

cm = confusion_matrix(y_test, y_pred_base)
tn, fp, fn, tp = cm.ravel()

precision = tp / (tp + fp) if (tp + fp) > 0 else 0
recall = tp / (tp + fn) if (tp + fn) > 0 else 0
specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

print(f"\nConfusion Matrix:")
print(f"  {'':>15} Predicted 0  Predicted 1")
print(f"  {'Actual 0':>15} {tn:>10}  {fp:>10}")
print(f"  {'Actual 1':>15} {fn:>10}  {tp:>10}")

print(f"\n--- Classification Metrics ---")
print(
    f"  Accuracy:    {(tp + tn) / (tp + tn + fp + fn):.4f}  (misleading at 12% pos rate!)"
)
print(
    f"  Precision:   {precision:.4f}  (of predicted defaults, how many actually default?)"
)
print(f"  Recall:      {recall:.4f}  (of actual defaults, how many did we catch?)")
print(
    f"  Specificity: {specificity:.4f}  (of actual non-defaults, how many correctly cleared?)"
)
print(f"  F1:          {f1:.4f}  (harmonic mean of precision and recall)")

print(f"\n--- When to use which metric ---")
print(f"  Accuracy:    NEVER for imbalanced data (88% by predicting majority)")
print(f"  Precision:   When FP cost is high (spam filter: annoying to misclassify)")
print(
    f"  Recall:      When FN cost is high (cancer screening: missing a case is deadly)"
)
print(f"  F1:          When you need to balance precision and recall")
print(
    f"  AUC-ROC:     Ranking quality across all thresholds (insensitive to imbalance)"
)
print(f"  AUC-PR:      Ranking quality for rare events (credit default, fraud)")
print(f"  Brier:       Probability calibration quality (proper scoring rule)")
print(f"  Log Loss:    Information-theoretic probability quality")

# ── Checkpoint 2 ─────────────────────────────────────────────────────
assert tp + tn + fp + fn == len(
    y_test
), "Confusion matrix should account for all samples"
assert (
    abs(precision - precision_score(y_test, y_pred_base)) < 1e-6
), "Manual precision should match sklearn"
# INTERPRETATION: The gap between accuracy (high) and recall (lower)
# exposes the imbalance problem.  The model gets high accuracy by being
# conservative — predicting "no default" most of the time — but misses
# actual defaults.  In banking, missing a default costs 100x more than
# a false alarm.
print("\n✓ Checkpoint 2 passed — complete metrics taxonomy reviewed\n")


# ══════════════════════════════════════════════════════════════════════
# TASK 3: SMOTE — and why it often fails
# ══════════════════════════════════════════════════════════════════════
# SMOTE creates synthetic minority examples by interpolating between
# nearest neighbours.  Problems:
# 1. Lipschitz violation: interpolation assumes smooth decision boundary
# 2. Noisy minority: amplifies noise in the minority class
# 3. High-dimensional collapse: in high dimensions, nearest neighbours
#    are nearly equidistant, making interpolation meaningless

from imblearn.over_sampling import SMOTE

smote = SMOTE(random_state=42)
# TODO: Apply SMOTE: call smote.fit_resample(X_train, y_train) to get
#       X_smote, y_smote (oversampled training data)
X_smote, y_smote = smote.fit_resample(____, ____)  # Hint: X_train, y_train
print(f"\n=== SMOTE ===")
print(f"Before SMOTE: {len(y_train):,} (pos={y_train.sum():.0f})")
print(f"After SMOTE:  {len(y_smote):,} (pos={y_smote.sum():.0f})")

smote_model = lgb.LGBMClassifier(n_estimators=300, random_state=42, verbose=-1)
smote_model.fit(X_smote, y_smote)
y_proba_smote = smote_model.predict_proba(X_test)[:, 1]
y_pred_smote = smote_model.predict(X_test)

print(f"AUC-ROC: {roc_auc_score(y_test, y_proba_smote):.4f}")
print(f"AUC-PR:  {average_precision_score(y_test, y_proba_smote):.4f}")
print(f"Brier:   {brier_score_loss(y_test, y_proba_smote):.4f}")
print(f"F1:      {f1_score(y_test, y_pred_smote):.4f}")
print(f"Recall:  {recall_score(y_test, y_pred_smote):.4f}")

print("\nSMOTE Failure Taxonomy:")
print("  1. Lipschitz: interpolated samples may cross decision boundary")
print("  2. Noise: noisy minority examples get amplified")
print("  3. Dimensionality: with 45 features, NN distances converge")
print(f"  → 92% citation rate in papers, ~6% production deployment")

# ── Checkpoint 3 ─────────────────────────────────────────────────────
assert len(y_smote) > len(y_train), "SMOTE should increase dataset size"
smote_pr = y_smote.mean()
assert smote_pr > pos_rate, "SMOTE should increase minority class proportion"
# INTERPRETATION: SMOTE often hurts calibration because synthetic
# samples near the boundary create false confidence in borderline
# predictions.  The Brier score may worsen even if AUC-PR improves.
print("\n✓ Checkpoint 3 passed — SMOTE applied and failure taxonomy reviewed\n")


# ══════════════════════════════════════════════════════════════════════
# TASK 4: Cost-sensitive learning (sample weights)
# ══════════════════════════════════════════════════════════════════════
# Weight minority class higher in the loss function.

# Method A: scale_pos_weight (class-balanced)
# TODO: Compute scale_weight = (1 - pos_rate) / pos_rate
scale_weight = ____  # Hint: (1 - pos_rate) / pos_rate
cost_model_a = lgb.LGBMClassifier(
    n_estimators=300,
    scale_pos_weight=scale_weight,
    random_state=42,
    verbose=-1,
)
cost_model_a.fit(X_train, y_train)
y_proba_cost_a = cost_model_a.predict_proba(X_test)[:, 1]

# Method B: custom sample weights (from cost matrix)
cost_fn = 10_000  # Cost of missing a default
cost_fp = 100  # Cost of a false alarm
# TODO: Compute sample_weights: cost_fn for positives (y=1), cost_fp for negatives
sample_weights = np.where(____, ____, ____)  # Hint: y_train == 1, cost_fn, cost_fp

cost_model_b = lgb.LGBMClassifier(n_estimators=300, random_state=42, verbose=-1)
cost_model_b.fit(X_train, y_train, sample_weight=sample_weights)
y_proba_cost_b = cost_model_b.predict_proba(X_test)[:, 1]

print(f"\n=== Cost-Sensitive Learning ===")
print(f"Method A (scale_pos_weight={scale_weight:.1f}):")
print(f"  AUC-ROC: {roc_auc_score(y_test, y_proba_cost_a):.4f}")
print(f"  AUC-PR:  {average_precision_score(y_test, y_proba_cost_a):.4f}")
print(f"  Brier:   {brier_score_loss(y_test, y_proba_cost_a):.4f}")
print(f"Method B (cost matrix: FN=${cost_fn:,}, FP=${cost_fp:,}):")
print(f"  AUC-ROC: {roc_auc_score(y_test, y_proba_cost_b):.4f}")
print(f"  AUC-PR:  {average_precision_score(y_test, y_proba_cost_b):.4f}")
print(f"  Brier:   {brier_score_loss(y_test, y_proba_cost_b):.4f}")

# ── Checkpoint 4 ─────────────────────────────────────────────────────
cost_auc_a = roc_auc_score(y_test, y_proba_cost_a)
assert cost_auc_a > 0.5, "Cost-sensitive model should beat random baseline"
# INTERPRETATION: Method A (scale_pos_weight) is equivalent to Method B
# when cost_fn/cost_fp = (1 - pos_rate) / pos_rate.  Method B is more
# general: you can specify any business cost matrix.
print("\n✓ Checkpoint 4 passed — cost-sensitive learning applied\n")


# ══════════════════════════════════════════════════════════════════════
# TASK 5: Focal Loss
# ══════════════════════════════════════════════════════════════════════
# Focal Loss: FL(p) = -α(1-p)^γ log(p)
# γ > 0 down-weights easy examples (well-classified)
# γ = 0 reduces to standard cross-entropy
# γ = 2 is the original setting (Lin et al., 2017)


def focal_loss_lgb(y_true, y_pred, gamma=2.0, alpha=0.25):
    """Custom focal loss for LightGBM (gradient and hessian)."""
    p = 1.0 / (1.0 + np.exp(-y_pred))  # sigmoid
    grad = alpha * (
        -((1 - p) ** gamma)
        * (gamma * p * np.log(np.clip(p, 1e-8, 1)) + (1 - p))
        * y_true
        + p**gamma
        * (gamma * (1 - p) * np.log(np.clip(1 - p, 1e-8, 1)) + p)
        * (1 - y_true)
    )
    hess = np.abs(grad) * (1 - np.abs(grad))
    hess = np.clip(hess, 1e-8, None)
    return grad, hess


# Focal loss approximation via class weighting
print(f"\n=== Focal Loss Approximation ===")
print("Testing different class-weight multipliers (focal-like effect):")
print(f"{'α mult':>8} {'AUC-PR':>10} {'Brier':>10} {'F1':>10}")
print("─" * 42)

focal_results = {}
for alpha_mult in [0.5, 1.0, 2.0, 5.0, 10.0]:
    # TODO: Compute pos_weight = alpha_mult * (count of class 0) / (count of class 1)
    pos_weight = alpha_mult * ((y_train == 0).sum() / max((y_train == 1).sum(), 1))
    m = lgb.LGBMClassifier(
        n_estimators=300,
        random_state=42,
        verbose=-1,
        scale_pos_weight=pos_weight,
    )
    m.fit(X_train, y_train)
    p = m.predict_proba(X_test)
    y_p = p[:, 1] if p.ndim == 2 else p
    y_p = np.clip(y_p, 0, 1)
    pr = average_precision_score(y_test, y_p)
    br = brier_score_loss(y_test, y_p)
    f1_val = f1_score(y_test, (y_p >= 0.5).astype(int))
    focal_results[alpha_mult] = {"auc_pr": pr, "brier": br, "f1": f1_val}
    print(f"{alpha_mult:>8.1f} {pr:>10.4f} {br:>10.4f} {f1_val:>10.4f}")

# ── Checkpoint 5 ─────────────────────────────────────────────────────
assert len(focal_results) == 5, "Should test 5 alpha multipliers"
# INTERPRETATION: Focal loss down-weights easy examples (those the model
# already classifies correctly with high confidence).  This forces the
# model to focus on hard examples near the decision boundary.  In
# practice, the effect is similar to cost-sensitive learning but with
# a smoother weighting scheme based on prediction confidence.
print("\n✓ Checkpoint 5 passed — focal loss approximation tested\n")


# ══════════════════════════════════════════════════════════════════════
# TASK 6: Threshold optimisation from cost matrix
# ══════════════════════════════════════════════════════════════════════
# Optimal threshold: t* = cost_FP / (cost_FP + cost_FN)
# This minimises expected total cost.

best_proba = y_proba_cost_a  # Use cost-sensitive model

# TODO: Derive optimal threshold from cost matrix formula
optimal_threshold = ____  # Hint: cost_fp / (cost_fp + cost_fn)
print(f"\n=== Threshold Optimisation ===")
print(f"Cost matrix: FP=${cost_fp:,}, FN=${cost_fn:,}")
print(f"Theoretical optimal threshold: {optimal_threshold:.4f}")
print(f"Default threshold (0.5) misses many defaults!")

# Evaluate at different thresholds
thresholds = np.arange(0.01, 0.50, 0.005)
threshold_results = []

for t in thresholds:
    y_pred_t = (best_proba >= t).astype(int)
    cm_t = confusion_matrix(y_test, y_pred_t)
    tn_t, fp_t, fn_t, tp_t = cm_t.ravel()
    total_cost = fp_t * cost_fp + fn_t * cost_fn
    prec = tp_t / (tp_t + fp_t) if (tp_t + fp_t) > 0 else 0
    rec = tp_t / (tp_t + fn_t) if (tp_t + fn_t) > 0 else 0
    threshold_results.append(
        {
            "threshold": t,
            "cost": total_cost,
            "fp": fp_t,
            "fn": fn_t,
            "tp": tp_t,
            "tn": tn_t,
            "precision": prec,
            "recall": rec,
        }
    )

# Find empirically optimal threshold
best_result = min(threshold_results, key=lambda x: x["cost"])
best_t = best_result["threshold"]

print(f"\nEmpirical sweep:")
print(
    f"{'Threshold':>10} {'Cost ($)':>12} {'Precision':>10} {'Recall':>10} {'FP':>6} {'FN':>6}"
)
print("─" * 58)
for r in threshold_results[::10]:  # Every 10th for display
    print(
        f"{r['threshold']:>10.3f} {r['cost']:>12,.0f} "
        f"{r['precision']:>10.4f} {r['recall']:>10.4f} "
        f"{r['fp']:>6} {r['fn']:>6}"
    )

print(f"\nOptimal threshold: {best_t:.3f}")
print(f"Minimum total cost: ${best_result['cost']:,.0f}")

# Compare with default threshold
y_pred_default = (best_proba >= 0.5).astype(int)
fp_d = ((y_pred_default == 1) & (y_test == 0)).sum()
fn_d = ((y_pred_default == 0) & (y_test == 1)).sum()
cost_default = fp_d * cost_fp + fn_d * cost_fn
print(f"Cost at threshold=0.5: ${cost_default:,.0f}")
savings = cost_default - best_result["cost"]
print(
    f"Savings from optimisation: ${savings:,.0f} ({savings / max(cost_default, 1):.1%})"
)

# ── Checkpoint 6 ─────────────────────────────────────────────────────
assert best_result["cost"] <= cost_default, "Optimised threshold should not cost more"
assert 0 < best_t < 1, "Optimal threshold should be in (0, 1)"
# INTERPRETATION: The optimal threshold from cost-matrix analysis is far
# below 0.5.  This reflects the asymmetry: catching a default that costs
# $10,000 is worth many false alarms at $100 each.
print("\n✓ Checkpoint 6 passed — threshold optimised from cost matrix\n")


# ══════════════════════════════════════════════════════════════════════
# TASK 7: Business ROI analysis
# ══════════════════════════════════════════════════════════════════════
# Translate model performance into dollar value for the business.

print(f"\n=== Business ROI Analysis ===")

# Scenario: bank processes 100,000 loan applications per year
n_applications = 100_000
default_rate = pos_rate
n_defaults = int(n_applications * default_rate)
n_non_defaults = n_applications - n_defaults

# Model at optimal threshold
recall_opt = best_result["recall"]
precision_opt = best_result["precision"]

defaults_caught = int(n_defaults * recall_opt)
defaults_missed = n_defaults - defaults_caught
false_alarms = int(defaults_caught / max(precision_opt, 0.01) - defaults_caught)

annual_fn_cost = defaults_missed * cost_fn
annual_fp_cost = false_alarms * cost_fp
annual_total_cost = annual_fn_cost + annual_fp_cost

# Without model (approve everyone)
no_model_cost = n_defaults * cost_fn

# TODO: Compute annual_savings = no_model_cost - annual_total_cost
annual_savings = ____  # Hint: no_model_cost - annual_total_cost

print(f"Annual volume: {n_applications:,} applications")
print(f"Default rate: {default_rate:.2%} → {n_defaults:,} defaults")
print(f"\nModel at threshold={best_t:.3f}:")
print(f"  Defaults caught: {defaults_caught:,} ({recall_opt:.1%})")
print(f"  Defaults missed: {defaults_missed:,}")
print(f"  False alarms: {false_alarms:,}")
print(f"\nCost Analysis:")
print(f"  Cost of missed defaults: ${annual_fn_cost:,.0f}")
print(f"  Cost of false alarms:    ${annual_fp_cost:,.0f}")
print(f"  Total model cost:        ${annual_total_cost:,.0f}")
print(f"  No-model cost:           ${no_model_cost:,.0f}")
print(f"  Annual savings:          ${annual_savings:,.0f}")

# ── Checkpoint 7 ─────────────────────────────────────────────────────
assert annual_savings >= 0, "Model should save money vs no model"
# INTERPRETATION: The ROI analysis translates precision/recall into
# dollars.  This is the language the business speaks.  "Our model has
# F1=0.72" means nothing to a CFO.  "Our model saves $8M per year in
# default losses" gets budget approval.
print("\n✓ Checkpoint 7 passed — business ROI analysis complete\n")


# ══════════════════════════════════════════════════════════════════════
# TASK 8: Platt scaling calibration
# ══════════════════════════════════════════════════════════════════════
# Platt scaling fits a logistic regression to the model's output scores,
# mapping them to calibrated probabilities.

# TODO: Create CalibratedClassifierCV with cost_model_a, method="sigmoid", cv=5
platt_model = CalibratedClassifierCV(
    ____, method=____, cv=____
)  # Hint: cost_model_a, "sigmoid", 5
platt_model.fit(X_train, y_train)
y_proba_platt = platt_model.predict_proba(X_test)[:, 1]

print(f"\n=== Platt Scaling Calibration ===")
print(f"AUC-PR:  {average_precision_score(y_test, y_proba_platt):.4f}")
print(f"Brier:   {brier_score_loss(y_test, y_proba_platt):.4f}")

# Reliability diagram data
prob_true_platt, prob_pred_platt = calibration_curve(y_test, y_proba_platt, n_bins=10)
print(f"\nReliability Diagram (Platt):")
print(f"{'Predicted':>10} {'Actual':>10} {'|Gap|':>10}")
print("─" * 34)
for pred, true in zip(prob_pred_platt, prob_true_platt):
    gap = abs(pred - true)
    print(f"{pred:>10.3f} {true:>10.3f} {gap:>10.3f}")

# ── Checkpoint 8 ─────────────────────────────────────────────────────
brier_platt = brier_score_loss(y_test, y_proba_platt)
assert brier_platt > 0, "Platt Brier should be positive"
assert brier_platt < 0.5, "Platt Brier should be reasonable"
# INTERPRETATION: Platt scaling assumes the calibration mapping is
# logistic (sigmoid).  This works well when the model's raw scores are
# approximately linear in log-odds of the true probability.
print("\n✓ Checkpoint 8 passed — Platt scaling calibration complete\n")


# ══════════════════════════════════════════════════════════════════════
# TASK 9: Isotonic regression calibration
# ══════════════════════════════════════════════════════════════════════
# Isotonic regression is a non-parametric calibration method.  It fits
# a step function that is monotonically non-decreasing.  More flexible
# than Platt but can overfit with small calibration sets.

# TODO: Create CalibratedClassifierCV with cost_model_a, method="isotonic", cv=5
iso_model = CalibratedClassifierCV(
    ____, method=____, cv=____
)  # Hint: cost_model_a, "isotonic", 5
iso_model.fit(X_train, y_train)
y_proba_iso = iso_model.predict_proba(X_test)[:, 1]

print(f"\n=== Isotonic Regression Calibration ===")
print(f"AUC-PR:  {average_precision_score(y_test, y_proba_iso):.4f}")
print(f"Brier:   {brier_score_loss(y_test, y_proba_iso):.4f}")

prob_true_iso, prob_pred_iso = calibration_curve(y_test, y_proba_iso, n_bins=10)
print(f"\nReliability Diagram (Isotonic):")
print(f"{'Predicted':>10} {'Actual':>10} {'|Gap|':>10}")
print("─" * 34)
for pred, true in zip(prob_pred_iso, prob_true_iso):
    gap = abs(pred - true)
    print(f"{pred:>10.3f} {true:>10.3f} {gap:>10.3f}")

# ── Checkpoint 9 ─────────────────────────────────────────────────────
brier_iso = brier_score_loss(y_test, y_proba_iso)
assert brier_iso > 0, "Isotonic Brier should be positive"
# INTERPRETATION: Isotonic regression is more flexible than Platt — it
# can correct non-monotonic miscalibrations that Platt cannot.  However,
# it is prone to overfitting on small calibration sets.  Use Platt for
# small datasets, isotonic for large ones.
print("\n✓ Checkpoint 9 passed — isotonic calibration complete\n")


# ══════════════════════════════════════════════════════════════════════
# TASK 10: Calibration comparison
# ══════════════════════════════════════════════════════════════════════

print(f"\n=== Calibration Comparison ===")
print(f"{'Method':<20} {'Brier':>8} {'AUC-PR':>8} {'AUC-ROC':>8}")
print("─" * 48)

cal_methods = {
    "Uncalibrated": y_proba_cost_a,
    "Platt Scaling": y_proba_platt,
    "Isotonic": y_proba_iso,
}
for name, proba in cal_methods.items():
    brier = brier_score_loss(y_test, proba)
    auc_pr = average_precision_score(y_test, proba)
    auc_roc = roc_auc_score(y_test, proba)
    print(f"{name:<20} {brier:>8.4f} {auc_pr:>8.4f} {auc_roc:>8.4f}")

# Visualise calibration curves
viz = ModelVisualizer()
for name, proba in cal_methods.items():
    # TODO: Call viz.calibration_curve(y_test, proba) for each method
    fig = viz.calibration_curve(____, ____)  # Hint: y_test, proba
    fig.update_layout(title=f"Calibration: {name}")
    fig.write_html(f"ex5_calibration_{name.lower().replace(' ', '_')}.html")

print("\nCalibration plots saved.")

# ── Checkpoint 10 ────────────────────────────────────────────────────
assert brier_platt <= 0.5, "Calibrated Brier should be reasonable"
# INTERPRETATION: The Brier score is a proper scoring rule — it
# simultaneously rewards both discrimination and calibration.  A model
# that improves AUC-PR but worsens Brier has better ranking but worse
# probability estimates.  For loan pricing, you need both.
print("\n✓ Checkpoint 10 passed — calibration comparison complete\n")


# ══════════════════════════════════════════════════════════════════════
# TASK 11: Final comparison — all imbalance strategies
# ══════════════════════════════════════════════════════════════════════

print(f"\n{'=' * 70}")
print("FINAL COMPARISON: ALL IMBALANCE HANDLING STRATEGIES")
print(f"{'=' * 70}")

all_results = {
    "Baseline (none)": {
        "proba": y_proba_base,
        "AUC_PR": average_precision_score(y_test, y_proba_base),
        "Brier": brier_score_loss(y_test, y_proba_base),
        "F1": f1_score(y_test, y_pred_base),
    },
    "SMOTE": {
        "proba": y_proba_smote,
        "AUC_PR": average_precision_score(y_test, y_proba_smote),
        "Brier": brier_score_loss(y_test, y_proba_smote),
        "F1": f1_score(y_test, y_pred_smote),
    },
    "Cost-sensitive": {
        "proba": y_proba_cost_a,
        "AUC_PR": average_precision_score(y_test, y_proba_cost_a),
        "Brier": brier_score_loss(y_test, y_proba_cost_a),
        "F1": f1_score(y_test, (y_proba_cost_a >= 0.5).astype(int)),
    },
    "Cost+Platt": {
        "proba": y_proba_platt,
        "AUC_PR": average_precision_score(y_test, y_proba_platt),
        "Brier": brier_score_loss(y_test, y_proba_platt),
        "F1": f1_score(y_test, (y_proba_platt >= 0.5).astype(int)),
    },
    "Cost+Isotonic": {
        "proba": y_proba_iso,
        "AUC_PR": average_precision_score(y_test, y_proba_iso),
        "Brier": brier_score_loss(y_test, y_proba_iso),
        "F1": f1_score(y_test, (y_proba_iso >= 0.5).astype(int)),
    },
}

print(f"\n{'Strategy':<20} {'AUC-PR':>10} {'Brier':>10} {'F1':>10}")
print("─" * 52)
for name, r in all_results.items():
    print(f"{name:<20} {r['AUC_PR']:>10.4f} {r['Brier']:>10.4f} {r['F1']:>10.4f}")

# Best strategy
best_strategy = max(all_results.items(), key=lambda x: x[1]["AUC_PR"])
print(f"\nBest by AUC-PR: {best_strategy[0]}")

fig = viz.metric_comparison(
    {
        name: {"AUC_PR": r["AUC_PR"], "Brier": r["Brier"]}
        for name, r in all_results.items()
    }
)
fig.update_layout(title="Class Imbalance Methods Comparison")
fig.write_html("ex5_imbalance_comparison.html")
print("Saved: ex5_imbalance_comparison.html")

# ── Checkpoint 11 ────────────────────────────────────────────────────
assert len(all_results) == 5, "Should compare 5 strategies"
# INTERPRETATION: Cost-sensitive + calibration almost always beats SMOTE
# for tabular financial data.  SMOTE is popular in papers but rare in
# production — the interpolation assumptions rarely hold in practice.
print("\n✓ Checkpoint 11 passed — all strategies compared\n")


# ══════════════════════════════════════════════════════════════════════
# TASK 12: Production recommendation
# ══════════════════════════════════════════════════════════════════════

print(f"\n=== Production Recommendation ===")
print(
    """
For credit scoring with 12% default rate and 100:1 cost ratio:

  1. Use COST-SENSITIVE learning (not SMOTE)
     → scale_pos_weight or custom sample weights from cost matrix
     → directly encodes business costs in the loss function

  2. CALIBRATE with isotonic regression (large data) or Platt (small data)
     → ensures "20% default risk" means 20% of those applicants default
     → critical for loan pricing and reserve calculations

  3. OPTIMISE THRESHOLD from cost matrix
     → t* = cost_FP / (cost_FP + cost_FN) ≈ 0.01 for this scenario
     → never use t=0.5 for asymmetric costs

  4. Report AUC-PR (not AUC-ROC) to stakeholders
     → AUC-PR measures what matters: finding the rare defaults
     → AUC-ROC can be misleadingly high with 88% majority class

  DO NOT use SMOTE in production unless you have:
  - Very small datasets (< 500 samples)
  - Low-dimensional data (< 10 features)
  - Verified that it improves calibration (rare)
"""
)

# ── Checkpoint 12 ────────────────────────────────────────────────────
print("\n✓ Checkpoint 12 passed — production recommendation documented\n")


# ══════════════════════════════════════════════════════════════════════
# REFLECTION
# ══════════════════════════════════════════════════════════════════════
print("═" * 70)
print("  WHAT YOU'VE MASTERED")
print("═" * 70)
print(
    """
  ✓ Why accuracy fails on imbalanced data (88% = "predict majority")
  ✓ Complete metrics taxonomy: precision, recall, F1, specificity, AUC-PR
  ✓ SMOTE: creates synthetic samples, but fails in high dimensions
  ✓ Cost-sensitive: encode business costs directly in the loss function
  ✓ Focal Loss: γ parameter down-weights easy examples automatically
  ✓ Threshold optimisation: t* = cost_FP / (cost_FP + cost_FN)
  ✓ Business ROI: translate model metrics into dollar savings
  ✓ Platt scaling: logistic calibration mapping
  ✓ Isotonic regression: non-parametric calibration
  ✓ Calibration comparison: reliability diagrams + Brier score

  KEY INSIGHT: In production, the best imbalance strategy depends on
  whether you need rankings (AUC-PR) or calibrated probabilities (Brier).
  Cost-sensitive learning + threshold optimisation is almost always
  better than SMOTE for tabular financial data.

  NEXT: Exercise 6 adds SHAP interpretability — explaining WHY the model
  makes each prediction.  This is required for regulatory compliance in
  credit scoring (right to explanation under PDPA and similar regulations).
"""
)

print("\n✓ Exercise 5 complete — class imbalance handling + calibration")
