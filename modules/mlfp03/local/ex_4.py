# Copyright 2026 Terrene Foundation
# SPDX-License-Identifier: Apache-2.0
"""
# ════════════════════════════════════════════════════════════════════════
# MLFP03 — Exercise 4: Gradient Boosting Deep Dive
# ════════════════════════════════════════════════════════════════════════
#
# WHAT YOU'LL LEARN:
#   After completing this exercise, you will be able to:
#   - Explain how boosting reduces bias by correcting residuals
#     sequentially (vs bagging's variance reduction)
#   - Derive and verify the XGBoost split gain formula from the
#     2nd-order Taylor expansion of the loss function
#   - Compare XGBoost, LightGBM, and CatBoost on the same dataset
#   - Tune key hyperparameters (learning rate, depth, regularisation)
#   - Analyse hyperparameter sensitivity using heatmaps
#   - Implement early stopping to prevent overfitting
#   - Interpret learning curves for each boosting library
#   - Choose AUC-PR over AUC-ROC for imbalanced classification tasks
#
# PREREQUISITES:
#   - MLFP03 Exercise 3 (model zoo, Random Forest — boosting extends
#     bagging)
#   - MLFP03 Exercise 2 (regularisation — L2 on leaf weights in XGBoost)
#
# ESTIMATED TIME: ~150-180 minutes
#
# TASKS:
#   1.  Load and prepare credit scoring data
#   2.  Boosting theory: from-scratch gradient boosting on 1D
#   3.  XGBoost split gain formula derivation and verification
#   4.  Train XGBoost, LightGBM, CatBoost with default params
#   5.  Compare learning curves and convergence speed
#   6.  Hyperparameter sensitivity: learning rate
#   7.  Hyperparameter sensitivity: max_depth heatmap
#   8.  Early stopping analysis
#   9.  Feature importance comparison across all three libraries
#   10. Comprehensive evaluation (AUC-PR, calibration, log loss)
#   11. Final comparison and model selection
#   12. Visualise with ModelVisualizer
#
# DATASET: Singapore credit scoring (from MLFP02)
#   Target: default (binary — 12% positive rate — imbalanced)
#   Rows: ~5,000 credit applications | Features: financial + behavioural
#   Key challenge: 12% default rate makes accuracy misleading — use AUC-PR
#
# THEORY:
#   Boosting: sequential ensemble where each tree corrects the errors of
#   the previous ensemble.  Reduces BIAS (vs bagging which reduces VARIANCE).
#
#   XGBoost objective: L = Σ l(y_i, ŷ_i) + Σ Ω(f_k)
#   Split gain: Gain = ½ [G_L²/(H_L+λ) + G_R²/(H_R+λ) - (G_L+G_R)²/(H_L+H_R+λ)] - γ
#     where G = Σ g_i (sum of first derivatives), H = Σ h_i (sum of second derivatives)
#
#   LightGBM: GOSS (gradient-based one-side sampling) + histogram splits
#   CatBoost: ordered boosting (prevents target leakage in categorical encoding)
#
# ════════════════════════════════════════════════════════════════════════
"""
from __future__ import annotations

import time

import numpy as np
import polars as pl
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.metrics import (
    average_precision_score,
    log_loss,
    brier_score_loss,
    roc_auc_score,
    classification_report,
    f1_score,
)
from sklearn.tree import DecisionTreeRegressor
import xgboost as xgb
import lightgbm as lgb
import catboost as cb

from kailash_ml import PreprocessingPipeline, ModelVisualizer
from kailash_ml.interop import to_sklearn_input

from shared import MLFPDataLoader


# ── Data Loading & Preparation ────────────────────────────────────────

loader = MLFPDataLoader()
credit = loader.load("mlfp02", "sg_credit_scoring.parquet")

print(f"=== Singapore Credit Data ===")
print(f"Shape: {credit.shape}")
print(f"Default rate: {credit['default'].mean():.2%}")

pipeline = PreprocessingPipeline()
# TODO: Call pipeline.setup() with normalize=False (tree models don't need it),
#       target="default", train_size=0.8, seed=42, categorical_encoding="ordinal",
#       imputation_strategy="median"
result = pipeline.setup(
    data=credit,
    target=____,  # Hint: "default"
    train_size=0.8,
    seed=42,
    normalize=____,  # Hint: False — tree models don't need normalisation
    categorical_encoding=____,  # Hint: "ordinal"
    imputation_strategy=____,  # Hint: "median"
)

print(f"\nTask type: {result.task_type}")
print(f"Train: {result.train_data.shape}, Test: {result.test_data.shape}")

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

feature_names = col_info["feature_columns"]
print(f"Features: {len(feature_names)}")

# ── Checkpoint 1 ─────────────────────────────────────────────────────
assert X_train.shape[0] > 0, "Training set empty"
assert len(feature_names) > 0, "No features"
assert y_train.mean() < 0.5, "Default rate should be minority class"
print("\n✓ Checkpoint 1 passed — credit data prepared for gradient boosting\n")


# ══════════════════════════════════════════════════════════════════════
# TASK 2: Boosting theory — from-scratch gradient boosting on 1D
# ══════════════════════════════════════════════════════════════════════
# Gradient boosting builds an additive model:
#   F_0(x) = initial prediction (e.g., log-odds of positive class)
#   F_m(x) = F_{m-1}(x) + η * h_m(x)
# where h_m fits the NEGATIVE GRADIENT (pseudo-residuals) of the loss.
#
# For binary classification with log-loss:
#   pseudo-residual_i = y_i - sigmoid(F_{m-1}(x_i))
#
# Each new tree corrects the DIRECTION the previous ensemble was wrong.

print("=== From-Scratch: Gradient Boosting (1D Demo) ===")

# Generate 1D demo data
rng = np.random.default_rng(42)
n_demo = 200
x_demo = rng.uniform(0, 1, n_demo).reshape(-1, 1)
# True probability: smooth logistic function
true_proba = 1 / (1 + np.exp(-10 * (x_demo.ravel() - 0.5)))
y_demo = rng.binomial(1, true_proba)

# Manual gradient boosting
learning_rate = 0.3
n_rounds = 10

# F_0: log-odds of positive class
pos_rate_demo = y_demo.mean()
F = np.full(n_demo, np.log(pos_rate_demo / (1 - pos_rate_demo)))

print(f"\nInitial prediction F_0 = log-odds = {F[0]:.4f}")
print(f"Initial probability = sigmoid(F_0) = {1 / (1 + np.exp(-F[0])):.4f}")
print(f"\n{'Round':>6} {'MSE(residuals)':>16} {'Mean |residual|':>16} {'Acc':>8}")
print("─" * 50)

for m in range(1, n_rounds + 1):
    # Current probabilities
    p = 1 / (1 + np.exp(-F))

    # Pseudo-residuals (negative gradient of log-loss)
    residuals = y_demo - p

    # Fit a shallow tree to the residuals
    tree = DecisionTreeRegressor(max_depth=3, random_state=42)
    tree.fit(x_demo, residuals)
    h = tree.predict(x_demo)

    # Update ensemble
    F = F + learning_rate * h

    # Metrics
    p_new = 1 / (1 + np.exp(-F))
    preds = (p_new >= 0.5).astype(int)
    acc = (preds == y_demo).mean()
    mse_resid = np.mean(residuals**2)
    mean_abs_resid = np.mean(np.abs(residuals))

    print(f"{m:>6} {mse_resid:>16.6f} {mean_abs_resid:>16.6f} {acc:>8.4f}")

print("\nKey insight:")
print("  Each round, the residuals get smaller → the ensemble improves.")
print("  η (learning rate) controls step size: smaller η = more robust but")
print("  needs more rounds.  This is the fundamental boosting mechanism.")

# ── Checkpoint 2 ─────────────────────────────────────────────────────
final_acc_demo = (((1 / (1 + np.exp(-F))) >= 0.5).astype(int) == y_demo).mean()
assert final_acc_demo > 0.6, "From-scratch boosting should converge"
# INTERPRETATION: Gradient boosting reduces bias by iteratively correcting
# errors.  Bagging (Random Forest) reduces variance by averaging independent
# trees.  This is the fundamental difference: boosting attacks systematic
# error, bagging attacks random error.
print("\n✓ Checkpoint 2 passed — from-scratch gradient boosting converged\n")


# ══════════════════════════════════════════════════════════════════════
# TASK 3: XGBoost split gain formula derivation
# ══════════════════════════════════════════════════════════════════════
# XGBoost uses a 2nd-order Taylor expansion of the loss:
#   L ≈ Σ [g_i * f(x_i) + ½ h_i * f(x_i)²] + Ω(f)
#
# where g_i = ∂L/∂ŷ (first derivative) and h_i = ∂²L/∂ŷ² (second)
#
# Split gain formula:
#   Gain = ½ [G_L²/(H_L+λ) + G_R²/(H_R+λ) - (G_L+G_R)²/(H_L+H_R+λ)] - γ
#
# G_L = Σ_{i∈left} g_i,  H_L = Σ_{i∈left} h_i
# λ = L2 regularisation on leaf weights, γ = min split loss

print("=== XGBoost Split Gain Formula ===")
print(
    """
For log-loss on binary classification:
  g_i = p_i - y_i        (first derivative: predicted - actual)
  h_i = p_i * (1 - p_i)  (second derivative: pred variance)

Split gain:
  Gain = ½ [G_L²/(H_L+λ) + G_R²/(H_R+λ) - (G_L+G_R)²/(H_L+H_R+λ)] - γ

The gain decomposes into:
  - Left child contribution:  G_L² / (H_L + λ)
  - Right child contribution: G_R² / (H_R + λ)
  - Parent (no split):        (G_L+G_R)² / (H_L+H_R + λ)
  - Complexity penalty: γ (one new leaf added)

If Gain < 0, the split is not worth making → pruning.
"""
)


def xgb_split_gain(
    g_left: float,
    h_left: float,
    g_right: float,
    h_right: float,
    lambda_reg: float = 1.0,
    gamma: float = 0.0,
) -> float:
    """Compute XGBoost split gain from gradient statistics."""
    left_score = g_left**2 / (h_left + lambda_reg)
    right_score = g_right**2 / (h_right + lambda_reg)
    parent_score = (g_left + g_right) ** 2 / (h_left + h_right + lambda_reg)
    return 0.5 * (left_score + right_score - parent_score) - gamma


# Numerical example: credit data split
# Suppose a node has 100 defaulters (y=1) and 800 non-defaulters (y=0)
# Current prediction p = 0.12 (overall default rate)
p_pred = 0.12
n_default = 100
n_no_default = 800

# Gradients for each class
g_default = p_pred - 1  # = -0.88 per defaulter
g_no_default = p_pred - 0  # = 0.12 per non-defaulter
h_per_sample = p_pred * (1 - p_pred)  # = 0.1056

# After a good split: left = mostly defaulters, right = mostly safe
g_left = 80 * g_default + 50 * g_no_default  # 80 defaults, 50 non-defaults
h_left = 130 * h_per_sample
g_right = 20 * g_default + 750 * g_no_default  # 20 defaults, 750 safe
h_right = 770 * h_per_sample

gain = xgb_split_gain(g_left, h_left, g_right, h_right, lambda_reg=1.0, gamma=0.0)
print(f"Example split gain: {gain:.4f}")
print(f"  Left:  G_L={g_left:.2f}, H_L={h_left:.2f}")
print(f"  Right: G_R={g_right:.2f}, H_R={h_right:.2f}")

# Show effect of regularisation
print(f"\n--- Regularisation Effect on Split Gain ---")
print(f"{'λ':>8} {'γ':>8} {'Gain':>10} {'Worth splitting?':>18}")
print("─" * 48)
for lam in [0.0, 1.0, 10.0, 100.0]:
    for gam in [0.0, 1.0, 5.0]:
        g = xgb_split_gain(g_left, h_left, g_right, h_right, lam, gam)
        worth = "Yes" if g > 0 else "No (prune)"
        print(f"{lam:>8.1f} {gam:>8.1f} {g:>10.4f} {worth:>18}")

# ── Checkpoint 3 ─────────────────────────────────────────────────────
assert gain > 0, "Good split should have positive gain"
assert (
    xgb_split_gain(g_left, h_left, g_right, h_right, 100.0, 5.0) < gain
), "Heavy regularisation should reduce gain"
# INTERPRETATION: The split gain formula is what makes XGBoost special.
# The 2nd-order Taylor expansion (using Hessians h_i) gives a more
# accurate quadratic approximation of the loss than plain gradient
# descent.  λ on leaf weights prevents any single leaf from having too
# much influence; γ penalises adding leaves at all.
print("\n✓ Checkpoint 3 passed — XGBoost split gain formula verified\n")


# ══════════════════════════════════════════════════════════════════════
# TASK 4: Train all three boosting algorithms with defaults
# ══════════════════════════════════════════════════════════════════════

# TODO: Populate models dict with XGBClassifier, LGBMClassifier, CatBoostClassifier
#       Each with n_estimators/iterations=500, learning_rate=0.1, max_depth/depth=6
models = {
    "XGBoost": xgb.XGBClassifier(
        n_estimators=____,  # Hint: 500
        learning_rate=____,  # Hint: 0.1
        max_depth=____,  # Hint: 6
        eval_metric="logloss",
        random_state=42,
        verbosity=0,
    ),
    "LightGBM": lgb.LGBMClassifier(
        n_estimators=____,  # Hint: 500
        learning_rate=____,  # Hint: 0.1
        max_depth=____,  # Hint: 6
        num_leaves=31,
        random_state=42,
        verbose=-1,
    ),
    "CatBoost": cb.CatBoostClassifier(
        iterations=____,  # Hint: 500
        learning_rate=____,  # Hint: 0.1
        depth=____,  # Hint: 6
        random_seed=42,
        verbose=0,
    ),
}

results = {}
for name, model in models.items():
    print(f"\nTraining {name}...")
    t0 = time.perf_counter()

    if name == "CatBoost":
        model.fit(X_train, y_train, eval_set=(X_test, y_test))
    else:
        model.fit(X_train, y_train, eval_set=[(X_test, y_test)])

    train_time = time.perf_counter() - t0

    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    results[name] = {
        "model": model,
        "y_pred": y_pred,
        "y_proba": y_proba,
        "auc_roc": roc_auc_score(y_test, y_proba),
        "auc_pr": average_precision_score(y_test, y_proba),
        "log_loss": log_loss(y_test, y_proba),
        "brier": brier_score_loss(y_test, y_proba),
        "f1": f1_score(y_test, y_pred),
        "train_time": train_time,
    }

    print(f"  Time: {train_time:.2f}s")
    print(f"  AUC-ROC: {results[name]['auc_roc']:.4f}")
    print(f"  AUC-PR:  {results[name]['auc_pr']:.4f}")
    print(f"  Log Loss: {results[name]['log_loss']:.4f}")
    print(f"  F1: {results[name]['f1']:.4f}")

# ── Checkpoint 4 ─────────────────────────────────────────────────────
for name, r in results.items():
    assert r["auc_roc"] > 0.5, f"{name} AUC-ROC should exceed random baseline"
    assert r["auc_pr"] > 0, f"{name} AUC-PR should be positive"
# INTERPRETATION: With 12% default rate, AUC-ROC can be misleadingly high.
# A model that never predicts default gets AUC-ROC ~0.5 but looks fine.
# AUC-PR rewards finding the rare positives — the metric that matters.
print("\n✓ Checkpoint 4 passed — all three boosting models trained\n")


# ══════════════════════════════════════════════════════════════════════
# TASK 5: Learning curves and convergence speed
# ══════════════════════════════════════════════════════════════════════

print("\n=== Learning Curves ===")

viz = ModelVisualizer()

for name, r in results.items():
    # TODO: Call viz.learning_curve() with model, X_train, y_train,
    #       cv=5, train_sizes=[0.1, 0.25, 0.5, 0.75, 1.0]
    fig = viz.learning_curve(
        ____,  # Hint: r["model"]
        ____,  # Hint: X_train
        ____,  # Hint: y_train
        cv=____,  # Hint: 5
        train_sizes=____,  # Hint: [0.1, 0.25, 0.5, 0.75, 1.0]
    )
    fig.update_layout(title=f"Learning Curve: {name}")
    fig.write_html(f"ex4_learning_curve_{name.lower().replace(' ', '_')}.html")
    print(f"  Saved: ex4_learning_curve_{name.lower().replace(' ', '_')}.html")

# ── Checkpoint 5 ─────────────────────────────────────────────────────
print("\n✓ Checkpoint 5 passed — learning curves generated\n")


# ══════════════════════════════════════════════════════════════════════
# TASK 6: Hyperparameter sensitivity — learning rate
# ══════════════════════════════════════════════════════════════════════

learning_rates = [0.01, 0.03, 0.05, 0.1, 0.2, 0.5]
lr_results = {name: {} for name in ["XGBoost", "LightGBM", "CatBoost"]}

print("\n=== Learning Rate Sensitivity ===")
for lr in learning_rates:
    # TODO: For each library, create a model with this lr (n_estimators/iterations=500),
    #       fit it, and store auc_roc and auc_pr in lr_results[library][lr]

    # XGBoost
    m = xgb.XGBClassifier(
        n_estimators=500,
        learning_rate=____,  # Hint: lr variable
        max_depth=6,
        eval_metric="logloss",
        random_state=42,
        verbosity=0,
    )
    m.fit(X_train, y_train)
    y_p = m.predict_proba(X_test)[:, 1]
    lr_results["XGBoost"][lr] = {
        "auc_roc": roc_auc_score(y_test, y_p),
        "auc_pr": average_precision_score(y_test, y_p),
    }

    # LightGBM
    m = lgb.LGBMClassifier(
        n_estimators=500,
        learning_rate=____,  # Hint: lr variable
        max_depth=6,
        random_state=42,
        verbose=-1,
    )
    m.fit(X_train, y_train)
    y_p = m.predict_proba(X_test)[:, 1]
    lr_results["LightGBM"][lr] = {
        "auc_roc": roc_auc_score(y_test, y_p),
        "auc_pr": average_precision_score(y_test, y_p),
    }

    # CatBoost
    m = cb.CatBoostClassifier(
        iterations=500,
        learning_rate=____,  # Hint: lr variable
        depth=6,
        random_seed=42,
        verbose=0,
    )
    m.fit(X_train, y_train)
    y_p = m.predict_proba(X_test)[:, 1]
    lr_results["CatBoost"][lr] = {
        "auc_roc": roc_auc_score(y_test, y_p),
        "auc_pr": average_precision_score(y_test, y_p),
    }

print(f"\n{'LR':>6} {'XGB AUC-PR':>12} {'LGB AUC-PR':>12} {'CB AUC-PR':>12}")
print("─" * 44)
for lr in learning_rates:
    print(
        f"{lr:>6.2f} "
        f"{lr_results['XGBoost'][lr]['auc_pr']:>12.4f} "
        f"{lr_results['LightGBM'][lr]['auc_pr']:>12.4f} "
        f"{lr_results['CatBoost'][lr]['auc_pr']:>12.4f}"
    )

print("\nInsight: smaller learning rates need more trees but generalise better.")
print("lr=0.01 with 500 trees may underfit; lr=0.5 may overfit.")

# ── Checkpoint 6 ─────────────────────────────────────────────────────
assert len(lr_results["XGBoost"]) == len(learning_rates), "All LR values tested"
print("\n✓ Checkpoint 6 passed — learning rate sensitivity analysed\n")


# ══════════════════════════════════════════════════════════════════════
# TASK 7: Hyperparameter sensitivity — max_depth heatmap
# ══════════════════════════════════════════════════════════════════════
# Vary learning_rate × max_depth to build a sensitivity heatmap.

depths_sweep = [3, 5, 6, 8, 10]
lr_sweep = [0.01, 0.05, 0.1, 0.2]

print("\n=== Hyperparameter Sensitivity Heatmap (XGBoost) ===")
print(f"{'':>10}", end="")
for d in depths_sweep:
    print(f"{'d=' + str(d):>10}", end="")
print()
print("─" * (10 + 10 * len(depths_sweep)))

# TODO: Create a 2D numpy array of zeros with shape (len(lr_sweep), len(depths_sweep))
heatmap_data = np.zeros(____)  # Hint: (len(lr_sweep), len(depths_sweep))
for i, lr in enumerate(lr_sweep):
    print(f"{'lr=' + str(lr):>10}", end="")
    for j, d in enumerate(depths_sweep):
        m = xgb.XGBClassifier(
            n_estimators=300,
            learning_rate=lr,
            max_depth=d,
            eval_metric="logloss",
            random_state=42,
            verbosity=0,
        )
        m.fit(X_train, y_train)
        y_p = m.predict_proba(X_test)[:, 1]
        # TODO: Compute AUC-PR and store in heatmap_data[i, j]
        auc_pr = ____  # Hint: average_precision_score(y_test, y_p)
        heatmap_data[i, j] = auc_pr
        print(f"{auc_pr:>10.4f}", end="")
    print()

# Find best combination
best_idx = np.unravel_index(heatmap_data.argmax(), heatmap_data.shape)
best_lr_hp = lr_sweep[best_idx[0]]
best_depth_hp = depths_sweep[best_idx[1]]
print(
    f"\nBest: lr={best_lr_hp}, depth={best_depth_hp} "
    f"(AUC-PR={heatmap_data[best_idx]:.4f})"
)

print("\nInsight: the heatmap reveals interaction effects between learning")
print("rate and tree depth.  Deep trees (d=10) + high lr (0.2) overfit.")
print("Shallow trees (d=3) + low lr (0.01) underfit.  The sweet spot is")
print("moderate depth + moderate learning rate.")

# ── Checkpoint 7 ─────────────────────────────────────────────────────
assert heatmap_data.shape == (len(lr_sweep), len(depths_sweep)), "Heatmap shape"
assert heatmap_data.max() > 0, "Best AUC-PR should be positive"
# INTERPRETATION: The heatmap is the visual proof that hyperparameters
# interact — you cannot tune them independently.  This is why Bayesian
# optimisation (Exercise 7) outperforms grid search: it models these
# interactions and explores efficiently.
print("\n✓ Checkpoint 7 passed — hyperparameter heatmap computed\n")


# ══════════════════════════════════════════════════════════════════════
# TASK 8: Early stopping analysis
# ══════════════════════════════════════════════════════════════════════
# Early stopping monitors validation loss and halts training when it
# stops improving.  This prevents overfitting with large n_estimators.

print("\n=== Early Stopping Analysis ===")

# TODO: Create XGBClassifier with n_estimators=2000, learning_rate=0.05,
#       max_depth=6, early_stopping_rounds=50
xgb_es = xgb.XGBClassifier(
    n_estimators=2000,
    learning_rate=0.05,
    max_depth=6,
    eval_metric="logloss",
    random_state=42,
    verbosity=0,
    early_stopping_rounds=____,  # Hint: 50
)
xgb_es.fit(X_train, y_train, eval_set=[(X_test, y_test)])
best_iteration_xgb = xgb_es.best_iteration
y_p_es = xgb_es.predict_proba(X_test)[:, 1]
auc_pr_es = average_precision_score(y_test, y_p_es)

print(f"XGBoost: best iteration = {best_iteration_xgb} / 2000")
print(f"XGBoost: AUC-PR with early stopping = {auc_pr_es:.4f}")

# TODO: Create LGBMClassifier with n_estimators=2000, learning_rate=0.05, max_depth=6
#       fit with eval_set=[(X_test, y_test)] and callbacks=[lgb.early_stopping(50, verbose=False)]
lgb_es = lgb.LGBMClassifier(
    n_estimators=2000,
    learning_rate=0.05,
    max_depth=6,
    random_state=42,
    verbose=-1,
)
lgb_es.fit(
    X_train,
    y_train,
    eval_set=[(X_test, y_test)],
    callbacks=[____],  # Hint: lgb.early_stopping(50, verbose=False)
)
best_iteration_lgb = lgb_es.best_iteration_
y_p_es_lgb = lgb_es.predict_proba(X_test)[:, 1]
auc_pr_es_lgb = average_precision_score(y_test, y_p_es_lgb)

print(f"LightGBM: best iteration = {best_iteration_lgb} / 2000")
print(f"LightGBM: AUC-PR with early stopping = {auc_pr_es_lgb:.4f}")

# Compare with fixed 500 rounds
print(f"\nComparison:")
print(f"  {'Model':<12} {'Fixed 500':>12} {'Early Stop':>12} {'Rounds Used':>12}")
print("  " + "─" * 50)
print(
    f"  {'XGBoost':<12} {results['XGBoost']['auc_pr']:>12.4f} "
    f"{auc_pr_es:>12.4f} {best_iteration_xgb:>12}"
)
print(
    f"  {'LightGBM':<12} {results['LightGBM']['auc_pr']:>12.4f} "
    f"{auc_pr_es_lgb:>12.4f} {best_iteration_lgb:>12}"
)

print("\nInsight: early stopping uses MORE trees (2000 budget) but stops")
print("when validation loss plateaus.  Often better than guessing n_estimators.")

# ── Checkpoint 8 ─────────────────────────────────────────────────────
assert best_iteration_xgb < 2000, "Early stopping should stop before 2000"
assert auc_pr_es > 0, "Early stopped model should have positive AUC-PR"
# INTERPRETATION: Early stopping is the practical alternative to tuning
# n_estimators.  Set n_estimators high (2000+), use early_stopping_rounds=50,
# and let the validation curve decide.  This is standard practice in Kaggle
# competitions and production ML systems.
print("\n✓ Checkpoint 8 passed — early stopping analysis complete\n")


# ══════════════════════════════════════════════════════════════════════
# TASK 9: Feature importance comparison
# ══════════════════════════════════════════════════════════════════════
# Each library computes feature importance differently:
#   XGBoost: gain (total reduction in loss from splits using the feature)
#   LightGBM: split (number of times the feature is used in splits)
#   CatBoost: PredictionValuesChange (average prediction change)
#
# These can give DIFFERENT rankings for the same data.

print("\n=== Feature Importance Comparison ===")

# Extract importances
xgb_imp = dict(
    zip(
        feature_names,
        results["XGBoost"]["model"].feature_importances_,
    )
)
lgb_imp = dict(
    zip(
        feature_names,
        results["LightGBM"]["model"].feature_importances_,
    )
)
cb_imp_raw = results["CatBoost"]["model"].get_feature_importance()
cb_imp = dict(zip(feature_names, cb_imp_raw / cb_imp_raw.sum()))

# Sort by XGBoost importance
top_features = sorted(xgb_imp.items(), key=lambda x: x[1], reverse=True)[:15]

print(f"\n{'Feature':<30} {'XGB':>8} {'LGB':>8} {'CB':>8}")
print("─" * 58)
for name, xgb_val in top_features:
    lgb_val = lgb_imp.get(name, 0)
    cb_val = cb_imp.get(name, 0)
    print(f"{name:<30} {xgb_val:>8.4f} {lgb_val:>8.4f} {cb_val:>8.4f}")

# Agreement analysis: how many features in each top-10?
xgb_top10 = set(
    name for name, _ in sorted(xgb_imp.items(), key=lambda x: x[1], reverse=True)[:10]
)
lgb_top10 = set(
    name for name, _ in sorted(lgb_imp.items(), key=lambda x: x[1], reverse=True)[:10]
)
cb_top10 = set(
    name for name, _ in sorted(cb_imp.items(), key=lambda x: x[1], reverse=True)[:10]
)

# TODO: Compute consensus features appearing in ALL THREE top-10 sets
consensus_top10 = ____  # Hint: xgb_top10 & lgb_top10 & cb_top10
print(f"\nTop-10 agreement (all three): {len(consensus_top10)} features")
print(f"Consensus features: {sorted(consensus_top10)}")
print(f"XGB ∩ LGB: {len(xgb_top10 & lgb_top10)}")
print(f"XGB ∩ CB:  {len(xgb_top10 & cb_top10)}")
print(f"LGB ∩ CB:  {len(lgb_top10 & cb_top10)}")

# ── Checkpoint 9 ─────────────────────────────────────────────────────
assert len(xgb_imp) == len(feature_names), "All features should have XGB importance"
# INTERPRETATION: Feature importance rankings differ across libraries
# because they measure different things.  Gain-based importance (XGBoost)
# and split-count importance (LightGBM) can rank correlated features
# differently.  Use SHAP (Exercise 6) for theoretically grounded importance.
print("\n✓ Checkpoint 9 passed — feature importance compared across libraries\n")


# ══════════════════════════════════════════════════════════════════════
# TASK 10: Comprehensive evaluation
# ══════════════════════════════════════════════════════════════════════

print(f"\n=== Comprehensive Model Comparison ===")
print(
    f"{'Model':<12} {'AUC-ROC':>10} {'AUC-PR':>10} {'Log Loss':>10} "
    f"{'Brier':>10} {'F1':>10} {'Time (s)':>10}"
)
print("─" * 74)
for name, r in results.items():
    print(
        f"{name:<12} {r['auc_roc']:>10.4f} {r['auc_pr']:>10.4f} "
        f"{r['log_loss']:>10.4f} {r['brier']:>10.4f} "
        f"{r['f1']:>10.4f} {r['train_time']:>10.2f}"
    )

# Add early-stopping models
print(
    f"{'XGB+ES':<12} {roc_auc_score(y_test, y_p_es):>10.4f} "
    f"{auc_pr_es:>10.4f} {log_loss(y_test, y_p_es):>10.4f} "
    f"{brier_score_loss(y_test, y_p_es):>10.4f} "
    f"{f1_score(y_test, (y_p_es >= 0.5).astype(int)):>10.4f} {'—':>10}"
)

# Calibration analysis
print(f"\n--- Calibration Analysis ---")
for name, r in results.items():
    # TODO: Call viz.calibration_curve(y_test, r["y_proba"]) for each model
    fig = viz.calibration_curve(____, ____)  # Hint: y_test, r["y_proba"]
    fig.update_layout(title=f"Calibration: {name}")
    fig.write_html(f"ex4_calibration_{name.lower().replace(' ', '_')}.html")

print("Calibration plots saved.")
print("A model with high AUC but poor calibration is NOT production-ready.")
print("Calibration = 'when the model says 20% chance, it happens 20% of the time'")

# ── Checkpoint 10 ────────────────────────────────────────────────────
for name, r in results.items():
    proba = r["y_proba"]
    assert proba.min() >= 0.0, f"{name} probabilities must be >= 0"
    assert proba.max() <= 1.0, f"{name} probabilities must be <= 1"
# INTERPRETATION: Calibration matters in credit: if the model says 15%
# default probability, the bank needs that to be a real probability,
# not just a ranking score.  Poorly calibrated models lead to wrong loan
# pricing and reserve calculations.
print("\n✓ Checkpoint 10 passed — comprehensive evaluation complete\n")


# ══════════════════════════════════════════════════════════════════════
# TASK 11: Final comparison and model selection
# ══════════════════════════════════════════════════════════════════════

best_model = max(results.items(), key=lambda x: x[1]["auc_pr"])
fastest_model = min(results.items(), key=lambda x: x[1]["train_time"])

print(f"\n=== Final Model Selection ===")
print(f"Best by AUC-PR: {best_model[0]} (AUC-PR={best_model[1]['auc_pr']:.4f})")
print(f"Fastest: {fastest_model[0]} (time={fastest_model[1]['train_time']:.2f}s)")

print(
    """
Boosting Library Decision Guide:
  ┌─────────────────────────────────────────────────────────────┐
  │ Library   │ Choose when                                     │
  │───────────┼─────────────────────────────────────────────────│
  │ XGBoost   │ Default choice.  Well-documented, stable,       │
  │           │ wide ecosystem.  Best regularisation options.    │
  │───────────┼─────────────────────────────────────────────────│
  │ LightGBM  │ Speed matters.  Large datasets (millions of     │
  │           │ rows).  Histogram-based splits are ~5-10x       │
  │           │ faster than XGBoost on big data.                │
  │───────────┼─────────────────────────────────────────────────│
  │ CatBoost  │ Many categoricals.  Ordered boosting prevents   │
  │           │ target leakage.  Best out-of-the-box with       │
  │           │ minimal tuning.                                 │
  └─────────────────────────────────────────────────────────────┘

Production recommendation:
  1. Start with LightGBM for speed during experimentation
  2. Compare with XGBoost for final model selection
  3. Use CatBoost when you have high-cardinality categoricals
  4. Always use early stopping — never guess n_estimators
"""
)

# ── Checkpoint 11 ────────────────────────────────────────────────────
assert best_model[1]["auc_pr"] > 0, "Best model should have positive AUC-PR"
print("\n✓ Checkpoint 11 passed — model selection guide complete\n")


# ══════════════════════════════════════════════════════════════════════
# TASK 12: Final visualisation
# ══════════════════════════════════════════════════════════════════════

metric_comparison = {
    name: {
        "AUC_ROC": r["auc_roc"],
        "AUC_PR": r["auc_pr"],
        "Log_Loss": r["log_loss"],
        "Brier": r["brier"],
    }
    for name, r in results.items()
}

# TODO: Call viz.metric_comparison(metric_comparison) to generate the comparison figure
fig = viz.metric_comparison(____)  # Hint: metric_comparison dict
fig.update_layout(title="Gradient Boosting Comparison: Singapore Credit Scoring")
fig.write_html("ex4_model_comparison.html")
print("Saved: ex4_model_comparison.html")

# TODO: Call viz.feature_importance(best_model[1]["model"], feature_names, top_n=15)
fig_fi = viz.feature_importance(
    ____, ____, top_n=15
)  # Hint: best_model[1]["model"], feature_names
fig_fi.update_layout(title=f"Feature Importance: {best_model[0]}")
fig_fi.write_html("ex4_feature_importance.html")
print("Saved: ex4_feature_importance.html")

# ── Checkpoint 12 ────────────────────────────────────────────────────
print("\n✓ Checkpoint 12 passed — all visualisations saved\n")


# ══════════════════════════════════════════════════════════════════════
# REFLECTION
# ══════════════════════════════════════════════════════════════════════
print("═" * 70)
print("  WHAT YOU'VE MASTERED")
print("═" * 70)
print(
    f"""
  ✓ Boosting theory: sequential bias reduction (vs bagging's variance)
  ✓ From-scratch gradient boosting: residual fitting, learning rate
  ✓ XGBoost split gain formula: G_L²/(H_L+λ) + G_R²/(H_R+λ) derivation
  ✓ Three libraries compared: XGBoost, LightGBM, CatBoost
  ✓ Learning rate sensitivity: smaller η → more robust, more rounds
  ✓ Hyperparameter heatmap: depth × learning rate interaction effects
  ✓ Early stopping: set high budget, let validation curve decide
  ✓ Feature importance: different libraries give different rankings
  ✓ Calibration analysis: probability reliability for credit decisions
  ✓ AUC-PR vs AUC-ROC: AUC-PR is the right metric for imbalanced data

  BEST MODEL (AUC-PR): {best_model[0]}

  KEY INSIGHT: Gradient boosting is the dominant algorithm for tabular
  data.  The three libraries trade off: XGBoost (stable, well-documented),
  LightGBM (fastest), CatBoost (best with categoricals).  In production,
  start with LightGBM for speed, then compare with XGBoost for final
  selection.  Always use early stopping.

  NEXT: Exercise 5 tackles class imbalance and calibration head-on —
  SMOTE, cost-sensitive learning, focal loss, and probability calibration.
  The 12% default rate you saw here will be a problem for naive training.
"""
)

print("\n✓ Exercise 4 complete — gradient boosting deep dive on credit data")
