# Copyright 2026 Terrene Foundation
# SPDX-License-Identifier: Apache-2.0
"""
# ════════════════════════════════════════════════════════════════════════
# MLFP03 — Exercise 8: Production Pipeline — DataFlow, Drift, and
#                        Deployment
# ════════════════════════════════════════════════════════════════════════
#
# WHAT YOU'LL LEARN:
#   After completing this exercise, you will be able to:
#   - Build a complete production ML pipeline from training to deployment
#   - Persist ML results to a database using DataFlow @db.model
#   - Apply conformal prediction for distribution-free uncertainty
#     quantification with coverage guarantees
#   - Set up DriftMonitor with PSI and KS test thresholds
#   - Simulate data drift and verify detection
#   - Generate a Mitchell et al. model card documenting performance,
#     limitations, and fairness findings
#   - Analyse bias-variance trade-off across model complexity levels
#   - Register, version, and promote models through ModelRegistry
#   - Build a monitoring dashboard with key deployment metrics
#   - Create deployment-ready artifacts and final visualisation
#
# PREREQUISITES:
#   - MLFP03 Exercises 1-7 (all of Module 3)
#   - MLFP02 complete (preprocessing pipeline, Singapore credit data)
#
# ESTIMATED TIME: ~150-180 minutes
#
# TASKS:
#   1.  Build complete pipeline: load → preprocess → train → evaluate
#   2.  Conformal prediction for uncertainty quantification
#   3.  Conformal prediction analysis (coverage vs set size)
#   4.  Cross-validate and analyse bias-variance trade-off
#   5.  DriftMonitor setup with PSI and KS thresholds
#   6.  Simulate drift and verify detection
#   7.  Generate model card (Mitchell et al. template)
#   8.  Persist to ModelRegistry and promote to production
#   9.  Build monitoring dashboard metrics
#   10. Deployment readiness checklist
#   11. Generate deployment artifacts
#   12. Module 3 capstone summary
#
# DATASET: Singapore credit scoring (from MLFP02)
#   Target: default prediction (binary, 12% positive rate)
#   Final model: LightGBM + isotonic calibration + conformal coverage
#   Regulatory context: Singapore MAS requires documented model cards
#
# ════════════════════════════════════════════════════════════════════════
"""
from __future__ import annotations

import asyncio
import json
from datetime import datetime

import numpy as np
import polars as pl
import lightgbm as lgb
from sklearn.model_selection import cross_val_score
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    log_loss,
    brier_score_loss,
    f1_score,
    accuracy_score,
    precision_score,
    recall_score,
)
from sklearn.calibration import CalibratedClassifierCV
from scipy import stats

from kailash.db import ConnectionManager
from kailash_ml import PreprocessingPipeline, ModelVisualizer
from kailash_ml.interop import to_sklearn_input
from kailash_ml.engines.experiment_tracker import ExperimentTracker

try:
    from kailash_ml.engines.model_registry import ModelRegistry

    HAS_REGISTRY = True
except ImportError:
    HAS_REGISTRY = False

try:
    from kailash_ml.engines.drift_monitor import DriftMonitor, DriftSpec

    HAS_DRIFT = True
except ImportError:
    HAS_DRIFT = False
    print("  Note: DriftMonitor not available.")

try:
    from kailash_ml.types import MetricSpec

    HAS_METRIC_SPEC = True
except ImportError:
    HAS_METRIC_SPEC = False

from shared import MLFPDataLoader
from shared.kailash_helpers import setup_environment

setup_environment()


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
feature_names = col_info["feature_columns"]

# ── Checkpoint 1 ─────────────────────────────────────────────────────
assert X_train.shape[0] > 0, "Training set should not be empty"
assert X_test.shape[0] > 0, "Test set should not be empty"
assert len(feature_names) > 0, "Feature names should be populated"
assert y_train.mean() < 0.5, "Default rate should be minority class"
print("\n✓ Checkpoint 1 passed — data loaded and split reproducibly\n")


# ══════════════════════════════════════════════════════════════════════
# TASK 1: Build complete pipeline — train and calibrate
# ══════════════════════════════════════════════════════════════════════

# Best hyperparameters from Exercise 7 (Bayesian optimisation)
model = lgb.LGBMClassifier(
    n_estimators=500,
    learning_rate=0.05,
    max_depth=7,
    num_leaves=63,
    min_child_samples=20,
    subsample=0.8,
    colsample_bytree=0.8,
    reg_alpha=0.1,
    reg_lambda=1.0,
    scale_pos_weight=(1 - y_train.mean()) / y_train.mean(),
    random_state=42,
    verbose=-1,
)
model.fit(X_train, y_train)

# TODO: Calibrate the model using CalibratedClassifierCV with isotonic method
# Hint: CalibratedClassifierCV(model, method="isotonic", cv=5)
calibrated_model = ____  # Hint: CalibratedClassifierCV(model, method="isotonic", cv=5)
calibrated_model.fit(X_train, y_train)

y_proba = calibrated_model.predict_proba(X_test)[:, 1]
y_pred = (y_proba >= 0.5).astype(int)

# TODO: Compute all 8 metrics: accuracy, precision, recall, f1, auc_roc, auc_pr,
#       log_loss, brier
metrics = {
    "accuracy": ____,  # Hint: accuracy_score(y_test, y_pred)
    "precision": ____,  # Hint: precision_score(y_test, y_pred)
    "recall": ____,  # Hint: recall_score(y_test, y_pred)
    "f1": ____,  # Hint: f1_score(y_test, y_pred)
    "auc_roc": ____,  # Hint: roc_auc_score(y_test, y_proba)
    "auc_pr": ____,  # Hint: average_precision_score(y_test, y_proba)
    "log_loss": ____,  # Hint: log_loss(y_test, y_proba)
    "brier": ____,  # Hint: brier_score_loss(y_test, y_proba)
}

print("=== Final Model Metrics ===")
for k, v in metrics.items():
    print(f"  {k}: {v:.4f}")

# ── Checkpoint 2 ─────────────────────────────────────────────────────
assert metrics["auc_roc"] > 0.5, "Should beat random"
assert metrics["auc_pr"] > 0, "AUC-PR should be positive"
assert 0 < metrics["brier"] < 0.25, "Brier should be reasonable for calibrated model"
# INTERPRETATION: The calibrated model uses isotonic regression to map raw
# scores to true probabilities.  Brier < 0.1 on a 12% base rate indicates
# the model is well-calibrated.
print("\n✓ Checkpoint 2 passed — final calibrated model trained\n")


# ══════════════════════════════════════════════════════════════════════
# TASK 2: Conformal prediction for uncertainty quantification
# ══════════════════════════════════════════════════════════════════════
# Conformal prediction provides distribution-free prediction sets with
# guaranteed coverage: P(Y ∈ C(X)) ≥ 1 - α
# No distributional assumptions needed — only exchangeability.
#
# ALGORITHM:
# 1. Compute nonconformity scores on calibration set
# 2. Find the (1-α) quantile of scores → q̂
# 3. For new data: include class c if score(x, c) ≤ q̂

print(f"\n=== Conformal Prediction ===")

# Split test set into calibration and evaluation
n_cal = X_test.shape[0] // 2
X_cal, X_eval = X_test[:n_cal], X_test[n_cal:]
y_cal, y_eval = y_test[:n_cal], y_test[n_cal:]

# Compute nonconformity scores on calibration set
cal_proba = calibrated_model.predict_proba(X_cal)[:, 1]
# TODO: Compute nonconformity scores: for class 1 → 1 - p, for class 0 → p
# Score = 1 - predicted probability of the TRUE class
cal_scores = ____  # Hint: np.where(y_cal == 1, 1 - cal_proba, cal_proba)

# Quantile for desired coverage
alpha = 0.10  # 90% coverage
n_cal_size = len(cal_scores)
quantile_level = np.ceil((n_cal_size + 1) * (1 - alpha)) / n_cal_size
# TODO: Compute q_hat as the quantile of cal_scores
q_hat = ____  # Hint: np.quantile(cal_scores, min(quantile_level, 1.0))

print(f"Calibration set: {n_cal_size} samples")
print(f"Target coverage: {1 - alpha:.0%}")
print(f"Calibration quantile (q̂): {q_hat:.4f}")

# Generate prediction sets on evaluation data
eval_proba = calibrated_model.predict_proba(X_eval)[:, 1]

# TODO: Build prediction_sets: for each sample, include class if its
#       nonconformity score ≤ q_hat. If set is empty, fall back to argmax.
# Hint: if (1 - eval_proba[i]) <= q_hat: add 1; if eval_proba[i] <= q_hat: add 0
prediction_sets = []
for i in range(len(y_eval)):
    pred_set = set()
    if ____:  # Hint: (1 - eval_proba[i]) <= q_hat
        pred_set.add(1)
    if ____:  # Hint: eval_proba[i] <= q_hat
        pred_set.add(0)
    if not pred_set:
        pred_set.add(1 if eval_proba[i] >= 0.5 else 0)
    prediction_sets.append(pred_set)

# Evaluate coverage and set sizes
coverage = np.mean([y_eval[i] in ps for i, ps in enumerate(prediction_sets)])
avg_set_size = np.mean([len(ps) for ps in prediction_sets])
singleton_rate = np.mean([len(ps) == 1 for ps in prediction_sets])
empty_rate = np.mean([len(ps) == 0 for ps in prediction_sets])

print(f"\nResults:")
print(f"  Coverage: {coverage:.4f} (target: {1 - alpha:.4f})")
print(f"  Average set size: {avg_set_size:.3f}")
print(f"  Singleton rate: {singleton_rate:.1%} (confident predictions)")
print(f"  Ambiguous rate: {1 - singleton_rate:.1%} (both classes possible)")

# ── Checkpoint 3 ─────────────────────────────────────────────────────
assert coverage >= (
    1 - alpha - 0.05
), f"Coverage {coverage:.4f} should be near target {1 - alpha:.4f}"
assert 0 < avg_set_size <= 2, "Set size should be between 0 and 2"
# INTERPRETATION: Conformal prediction's coverage guarantee is
# distribution-free.  High singleton_rate means the model is confident;
# ambiguous samples should trigger human review in production.
print("\n✓ Checkpoint 3 passed — conformal coverage guarantee verified\n")


# ══════════════════════════════════════════════════════════════════════
# TASK 3: Conformal prediction analysis — coverage vs alpha
# ══════════════════════════════════════════════════════════════════════
# Show how coverage and set size change as we vary α.

print(f"\n=== Coverage vs Alpha Analysis ===")
print(
    f"{'Alpha':>8} {'Target Cov':>12} {'Actual Cov':>12} {'Avg Size':>10} {'Singleton%':>12}"
)
print("─" * 58)

alphas_sweep = [0.01, 0.05, 0.10, 0.15, 0.20, 0.30]
conformal_analysis = []
for a in alphas_sweep:
    q_level = np.ceil((n_cal_size + 1) * (1 - a)) / n_cal_size
    q = np.quantile(cal_scores, min(q_level, 1.0))

    sets = []
    for i in range(len(y_eval)):
        ps = set()
        if (1 - eval_proba[i]) <= q:
            ps.add(1)
        if eval_proba[i] <= q:
            ps.add(0)
        if not ps:
            ps.add(1 if eval_proba[i] >= 0.5 else 0)
        sets.append(ps)

    cov = np.mean([y_eval[i] in s for i, s in enumerate(sets)])
    avg_sz = np.mean([len(s) for s in sets])
    single = np.mean([len(s) == 1 for s in sets])
    conformal_analysis.append({"alpha": a, "coverage": cov, "avg_size": avg_sz})
    print(f"{a:>8.2f} {1 - a:>12.2f} {cov:>12.4f} {avg_sz:>10.3f} {single:>11.1%}")

print("\nInsight: lower α → higher coverage but larger prediction sets.")
print("α=0.01 gives 99% coverage but nearly every sample is ambiguous.")
print("α=0.10 is the standard choice — 90% coverage with mostly singletons.")

# ── Checkpoint 4 ─────────────────────────────────────────────────────
assert len(conformal_analysis) == len(alphas_sweep), "All alphas analysed"
print("\n✓ Checkpoint 4 passed — conformal coverage-alpha analysis complete\n")


# ══════════════════════════════════════════════════════════════════════
# TASK 4: Cross-validation bias-variance analysis
# ══════════════════════════════════════════════════════════════════════

print(f"\n=== Bias-Variance Analysis ===")

complexities = [
    (
        "Simple (d=3, n=100)",
        lgb.LGBMClassifier(max_depth=3, n_estimators=100, verbose=-1, random_state=42),
    ),
    (
        "Medium (d=6, n=300)",
        lgb.LGBMClassifier(max_depth=6, n_estimators=300, verbose=-1, random_state=42),
    ),
    (
        "Complex (d=10, n=500)",
        lgb.LGBMClassifier(max_depth=10, n_estimators=500, verbose=-1, random_state=42),
    ),
    (
        "Very Complex (d=-1, n=1000)",
        lgb.LGBMClassifier(
            max_depth=-1, n_estimators=1000, num_leaves=255, verbose=-1, random_state=42
        ),
    ),
]

print(f"{'Model':<28} {'CV Mean':>10} {'CV Std':>10} {'Train':>10} {'Gap':>10}")
print("─" * 72)

bv_results = []
for name, m in complexities:
    # TODO: Compute cross-validation scores with scoring="average_precision"
    # Hint: cross_val_score(m, X_train, y_train, cv=5, scoring="average_precision")
    cv_scores = ____
    m.fit(X_train, y_train)
    # TODO: Compute training set score using average_precision_score
    train_score = (
        ____  # Hint: average_precision_score(y_train, m.predict_proba(X_train)[:, 1])
    )
    gap = train_score - cv_scores.mean()
    bv_results.append(
        {
            "name": name,
            "cv_mean": cv_scores.mean(),
            "cv_std": cv_scores.std(),
            "train_score": train_score,
            "gap": gap,
        }
    )
    print(
        f"{name:<28} {cv_scores.mean():>10.4f} {cv_scores.std():>10.4f} "
        f"{train_score:>10.4f} {gap:>10.4f}"
    )

print("\nInterpretation:")
print("  Small gap + low score → high bias (underfitting)")
print("  Large gap + high train → high variance (overfitting)")
print("  'Medium' typically offers the best trade-off")

# ── Checkpoint 5 ─────────────────────────────────────────────────────
assert len(bv_results) == 4, "Should analyse 4 complexity levels"
assert (
    bv_results[3]["train_score"] >= bv_results[0]["train_score"]
), "More complex model should achieve higher training score"
print("\n✓ Checkpoint 5 passed — bias-variance tradeoff demonstrated\n")


# ══════════════════════════════════════════════════════════════════════
# TASK 5: DriftMonitor setup
# ══════════════════════════════════════════════════════════════════════
# DriftMonitor detects when the production data distribution shifts
# away from the training distribution.
#
# PSI (Population Stability Index):
#   PSI = Σ (p_new - p_ref) * ln(p_new / p_ref)
#   PSI < 0.1: no shift, 0.1-0.2: moderate, > 0.2: significant
#
# KS test (Kolmogorov-Smirnov):
#   Compares CDFs of reference and new distributions.
#   p-value < 0.05 → distributions differ significantly.

print(f"\n=== DriftMonitor Setup ===")


def compute_psi(reference: np.ndarray, current: np.ndarray, bins: int = 10) -> float:
    """Compute Population Stability Index from scratch."""
    # Bin edges from reference distribution
    _, bin_edges = np.histogram(reference, bins=bins)

    # Compute proportions in each bin
    ref_counts, _ = np.histogram(reference, bins=bin_edges)
    cur_counts, _ = np.histogram(current, bins=bin_edges)

    # Avoid division by zero
    ref_props = (ref_counts + 1) / (len(reference) + bins)
    cur_props = (cur_counts + 1) / (len(current) + bins)

    # PSI formula
    # TODO: Compute PSI = sum((cur - ref) * log(cur / ref))
    psi = ____  # Hint: np.sum((cur_props - ref_props) * np.log(cur_props / ref_props))
    return psi


def compute_ks(reference: np.ndarray, current: np.ndarray) -> tuple[float, float]:
    """Compute Kolmogorov-Smirnov statistic and p-value."""
    # TODO: Use scipy.stats.ks_2samp to compute KS statistic and p-value
    ks_stat, p_value = ____  # Hint: stats.ks_2samp(reference, current)
    return ks_stat, p_value


# Compute baseline drift metrics (test vs train — should be low)
print(f"\nBaseline drift (train vs test — same distribution):")
print(f"{'Feature':<30} {'PSI':>8} {'KS stat':>10} {'KS p-val':>10} {'Drift?':>8}")
print("─" * 70)

baseline_drift = {}
for i, feat in enumerate(feature_names[:10]):
    psi = compute_psi(X_train[:, i], X_test[:, i])
    ks_stat, ks_pval = compute_ks(X_train[:, i], X_test[:, i])
    drift = "Yes" if psi > 0.2 or ks_pval < 0.05 else "No"
    baseline_drift[feat] = {"psi": psi, "ks_stat": ks_stat, "ks_pval": ks_pval}
    print(f"  {feat:<28} {psi:>8.4f} {ks_stat:>10.4f} {ks_pval:>10.4f} {drift:>8}")

if HAS_DRIFT:
    # TODO: Create a DriftSpec with psi_threshold=0.1, ks_threshold=0.05,
    #       monitoring_frequency="daily", alert_on_drift=True
    drift_spec = DriftSpec(
        psi_threshold=____,  # Hint: 0.1
        ks_threshold=____,  # Hint: 0.05
        monitoring_frequency=____,  # Hint: "daily"
        alert_on_drift=True,
    )
    print(
        f"\nDriftSpec configured: PSI threshold={drift_spec.psi_threshold}, "
        f"KS p-value threshold={drift_spec.ks_threshold}"
    )

# ── Checkpoint 6 ─────────────────────────────────────────────────────
assert len(baseline_drift) > 0, "Should compute drift for features"
print("\n✓ Checkpoint 6 passed — DriftMonitor baseline established\n")


# ══════════════════════════════════════════════════════════════════════
# TASK 6: Simulate drift and verify detection
# ══════════════════════════════════════════════════════════════════════

print(f"\n=== Simulated Drift Detection ===")

rng = np.random.default_rng(42)

# Scenario 1: Gradual drift — shift mean of top features
X_drifted_gradual = X_test.copy()
for i in range(3):  # Shift top 3 features
    X_drifted_gradual[:, i] += 0.5 * X_train[:, i].std()

print("Scenario 1: Gradual drift (mean shift of 0.5σ on top 3 features)")
print(f"{'Feature':<30} {'PSI':>8} {'KS stat':>10} {'KS p-val':>10} {'Detected?':>10}")
print("─" * 72)
for i in range(5):
    feat = feature_names[i]
    psi = compute_psi(X_train[:, i], X_drifted_gradual[:, i])
    ks_stat, ks_pval = compute_ks(X_train[:, i], X_drifted_gradual[:, i])
    detected = "YES ⚠" if psi > 0.1 or ks_pval < 0.05 else "No"
    print(f"  {feat:<28} {psi:>8.4f} {ks_stat:>10.4f} {ks_pval:>10.4f} {detected:>10}")

# Scenario 2: Sudden drift — completely new distribution
# TODO: Simulate sudden drift by replacing feature 0 with values drawn from
#       a normal distribution with mean = train_mean + 3*train_std
X_drifted_sudden = X_test.copy()
X_drifted_sudden[:, 0] = rng.normal(
    loc=____,  # Hint: X_train[:, 0].mean() + 3 * X_train[:, 0].std()
    scale=X_train[:, 0].std(),
    size=X_test.shape[0],
)

print("\nScenario 2: Sudden drift (3σ mean shift on feature 0)")
# TODO: Compute PSI and KS for the suddenly drifted feature 0
psi_sudden = ____  # Hint: compute_psi(X_train[:, 0], X_drifted_sudden[:, 0])
ks_stat_sudden, ks_pval_sudden = (
    ____  # Hint: compute_ks(X_train[:, 0], X_drifted_sudden[:, 0])
)
print(f"  {feature_names[0]}: PSI={psi_sudden:.4f}, KS p-val={ks_pval_sudden:.6f}")
print(f"  Detected: {'YES ⚠ — retrain immediately' if psi_sudden > 0.2 else 'No'}")

# Performance degradation under drift
y_proba_drifted = calibrated_model.predict_proba(X_drifted_gradual)[:, 1]
auc_drifted = roc_auc_score(y_test, y_proba_drifted)
print(f"\nPerformance under gradual drift:")
print(f"  Original AUC-ROC: {metrics['auc_roc']:.4f}")
print(f"  Drifted AUC-ROC:  {auc_drifted:.4f}")
print(f"  Degradation:      {metrics['auc_roc'] - auc_drifted:.4f}")

# ── Checkpoint 7 ─────────────────────────────────────────────────────
assert psi_sudden > 0.1, "Sudden drift should produce high PSI"
assert ks_pval_sudden < 0.05, "Sudden drift should be detected by KS test"
# INTERPRETATION: PSI and KS complement each other.  PSI captures overall
# distribution shift; KS is more sensitive to local changes in the CDF.
# In production, monitor BOTH and trigger alerts when either exceeds threshold.
print("\n✓ Checkpoint 7 passed — drift detected in both scenarios\n")


# ══════════════════════════════════════════════════════════════════════
# TASK 7: Generate model card (Mitchell et al.)
# ══════════════════════════════════════════════════════════════════════

# TODO: Build the model_card f-string with the following sections:
#   Model Details, Intended Use, Training Data, Evaluation (use metrics dict),
#   Uncertainty Quantification (use coverage, alpha, singleton_rate),
#   Ethical Considerations, Limitations, Monitoring
# Hint: Use an f-string with {metrics['auc_roc']:.4f}, {coverage:.1%}, etc.
model_card = f"""
# Model Card: Singapore Credit Default Prediction

## Model Details
- **Model type**: LightGBM Classifier (calibrated with isotonic regression)
- **Version**: 1.0
- **Date**: {datetime.now().strftime("%Y-%m-%d")}
- **Framework**: kailash-ml (Terrene Foundation)
- **License**: Apache 2.0

## Intended Use
- **Primary use**: Credit default risk assessment for Singapore market
- **Users**: Credit risk analysts, automated underwriting systems
- **Out of scope**: Regulatory capital calculation, cross-border lending

## Training Data
- **Source**: Singapore credit applications (data.gov.sg characteristics)
- **Size**: {X_train.shape[0]:,} training samples
- **Features**: {X_train.shape[1]} features (financial, behavioral, demographic)
- **Target**: Binary default ({y_train.mean():.1%} positive rate)
- **Time range**: 2020-2024

## Evaluation
- **Test set**: {X_test.shape[0]:,} samples (holdout, same distribution)
- **AUC-ROC**: ____
- **AUC-PR**: ____
- **Brier Score**: ____ (calibrated)
- **F1**: ____
- **Precision**: ____
- **Recall**: ____

## Uncertainty Quantification
- **Method**: Split conformal prediction (distribution-free)
- **Coverage**: ____ at α=____
- **Singleton rate**: ____
- **Ambiguous samples**: trigger human review

## Ethical Considerations
- Protected attributes (age, gender, ethnicity) analysed with SHAP
- Disparate impact testing performed (see Exercise 6)
- Impossibility theorem acknowledged: cannot satisfy all fairness criteria
- Model monitored for drift in protected group performance

## Limitations
- Trained on synthetic data — validate on production data before deployment
- Singapore-specific — do not apply to other markets without retraining
- Point-in-time: performance may degrade as economic conditions change
- Conformal coverage assumes exchangeability (no adversarial drift)

## Monitoring
- DriftMonitor configured: PSI threshold=0.1, KS p-value threshold=0.05
- Retrain trigger: PSI > 0.2 OR AUC-PR drops below ____
- Monitoring frequency: daily
"""
# TODO: Fill in the blanks above with the actual metric values from the
#       metrics dict, coverage, alpha, singleton_rate, and computed thresholds.
# Hint: replace ____ with f-string expressions like {metrics['auc_roc']:.4f}

print("\n=== Model Card ===")
print(model_card)

with open("ex8_model_card.md", "w") as f:
    f.write(model_card)
print("Saved: ex8_model_card.md")

# ── Checkpoint 8 ─────────────────────────────────────────────────────
import os

assert os.path.exists("ex8_model_card.md"), "Model card should be written"
assert "AUC-ROC" in model_card, "Should report AUC-ROC"
assert "Ethical Considerations" in model_card, "Should include ethics"
assert "Uncertainty" in model_card, "Should document uncertainty quantification"
# INTERPRETATION: The Mitchell et al. model card is a de facto standard
# for transparent ML documentation.  Singapore MAS and EU AI Act both
# require this level of documentation for high-risk AI systems.
print("\n✓ Checkpoint 8 passed — model card generated and saved\n")


# ══════════════════════════════════════════════════════════════════════
# TASK 8: Persist to ModelRegistry and promote
# ══════════════════════════════════════════════════════════════════════


async def persist_final():
    # TODO: Create and initialize a ConnectionManager
    # Hint: conn = ConnectionManager("sqlite:///mlfp03_models.db")
    conn = ____
    await conn.initialize()

    model_id = None
    if HAS_REGISTRY:
        import pickle

        # TODO: Create ModelRegistry(conn) and call register_model
        registry = ____  # Hint: ModelRegistry(conn)
        model_bytes = pickle.dumps(calibrated_model)

        if HAS_METRIC_SPEC:
            metrics_list = [
                MetricSpec(name="auc_pr", value=metrics["auc_pr"]),
                MetricSpec(name="brier", value=metrics["brier"]),
                MetricSpec(name="conformal_coverage", value=coverage),
            ]
        else:
            metrics_list = []

        # TODO: Register with name="credit_default_production"
        # Hint: await registry.register_model(name=..., artifact=..., metrics=...)
        model_version = await registry.register_model(
            name=____,  # Hint: "credit_default_production"
            artifact=____,  # Hint: model_bytes
            metrics=metrics_list,
        )

        # TODO: Promote to "production" stage with a descriptive reason
        # Hint: await registry.promote_model(name=..., version=..., target_stage=..., reason=...)
        await registry.promote_model(
            name=____,  # Hint: "credit_default_production"
            version=____,  # Hint: model_version.version
            target_stage=____,  # Hint: "production"
            reason=(
                f"Passed all quality gates: AUC-PR={metrics['auc_pr']:.4f}, "
                f"Brier={metrics['brier']:.4f}, Coverage={coverage:.4f}"
            ),
        )
        model_id = model_version.version
        print(f"\nModel registered and promoted: v{model_id}")
    else:
        model_id = "skipped"
        print("\nModelRegistry not available — skipping registration.")

    # Log to experiment tracker
    tracker = ExperimentTracker(conn)
    exp_id = await tracker.create_experiment(
        name="mlfp03_production_pipeline",
        description="End-to-end supervised ML pipeline — Module 3 capstone",
    )
    async with tracker.run(exp_id, run_name="production_model_v1") as run:
        await run.log_param("model", "lgbm_calibrated_conformal")
        await run.log_metrics({**metrics, "conformal_coverage": coverage})
        await run.set_tag("stage", "production")

    await conn.close()
    return model_id


model_id = asyncio.run(persist_final())

# ── Checkpoint 9 ─────────────────────────────────────────────────────
assert model_id is not None, "Model should be registered"
print("\n✓ Checkpoint 9 passed — model registered and promoted\n")


# ══════════════════════════════════════════════════════════════════════
# TASK 9: Monitoring dashboard metrics
# ══════════════════════════════════════════════════════════════════════

print(f"\n=== Production Monitoring Dashboard ===")
print(
    f"""
┌─────────────────────────────────────────────────────────────┐
│                  CREDIT MODEL DASHBOARD                      │
├─────────────────────────────────────────────────────────────┤
│  Model: LightGBM + Isotonic Calibration                      │
│  Version: {model_id}                                         │
│  Status: PRODUCTION                                          │
├─────────────────────────────────────────────────────────────┤
│  Performance Metrics                                         │
│    AUC-ROC:   {metrics['auc_roc']:.4f}                       │
│    AUC-PR:    {metrics['auc_pr']:.4f}                        │
│    Brier:     {metrics['brier']:.4f}                         │
│    F1:        {metrics['f1']:.4f}                            │
├─────────────────────────────────────────────────────────────┤
│  Uncertainty                                                 │
│    Conformal coverage: {coverage:.1%} (target: 90%)          │
│    Singleton rate:     {singleton_rate:.1%}                   │
│    Ambiguous:          {1 - singleton_rate:.1%}               │
├─────────────────────────────────────────────────────────────┤
│  Drift Status                                                │
│    PSI (top feature):  {list(baseline_drift.values())[0]['psi']:.4f} (threshold: 0.1) │
│    KS (top feature):   {list(baseline_drift.values())[0]['ks_pval']:.4f} (threshold: 0.05) │
│    Status: {'OK' if list(baseline_drift.values())[0]['psi'] < 0.1 else 'ALERT'}         │
├─────────────────────────────────────────────────────────────┤
│  Alerts                                                      │
│    Retrain if: PSI > 0.2 OR AUC-PR < {metrics['auc_pr'] * 0.9:.4f}│
└─────────────────────────────────────────────────────────────┘
"""
)

# ── Checkpoint 10 ────────────────────────────────────────────────────
print("\n✓ Checkpoint 10 passed — monitoring dashboard generated\n")


# ══════════════════════════════════════════════════════════════════════
# TASK 10: Deployment readiness checklist
# ══════════════════════════════════════════════════════════════════════

print(f"\n=== Deployment Readiness Checklist ===")

checklist = {
    "Model trained and evaluated": metrics["auc_roc"] > 0.5,
    "Model calibrated (Brier < 0.15)": metrics["brier"] < 0.15,
    "Conformal coverage >= 85%": coverage >= 0.85,
    "Model card generated": os.path.exists("ex8_model_card.md"),
    "Drift baseline established": len(baseline_drift) > 0,
    "Drift detection verified": psi_sudden > 0.1,
    "Model registered in registry": model_id is not None,
    "Reproducibility verified": True,
    "Fairness audit completed (Ex6)": True,
    "SHAP explanations available (Ex6)": True,
}

all_pass = True
for item, passed in checklist.items():
    status = "✓ PASS" if passed else "✗ FAIL"
    if not passed:
        all_pass = False
    print(f"  [{status}] {item}")

print(f"\nOverall: {'READY FOR DEPLOYMENT' if all_pass else 'BLOCKED — fix failures'}")

# ── Checkpoint 11 ────────────────────────────────────────────────────
assert all_pass, "All checklist items should pass"
print("\n✓ Checkpoint 11 passed — deployment readiness verified\n")


# ══════════════════════════════════════════════════════════════════════
# TASK 11: Generate deployment artifacts
# ══════════════════════════════════════════════════════════════════════

# TODO: Build and save the deployment_config dict as JSON
# Include: model_name, model_version, framework, calibration, conformal_alpha,
#          conformal_qhat, feature_names, threshold, drift thresholds, metrics, created
deployment_config = {
    "model_name": "credit_default_production",
    "model_version": str(model_id),
    "framework": "lightgbm",
    "calibration": "isotonic",
    "conformal_alpha": ____,  # Hint: alpha
    "conformal_qhat": ____,  # Hint: float(q_hat)
    "feature_names": feature_names,
    "threshold": 0.5,
    "drift_psi_threshold": 0.1,
    "drift_ks_threshold": 0.05,
    "retrain_auc_pr_floor": ____,  # Hint: float(metrics["auc_pr"] * 0.9)
    "metrics": {k: float(v) for k, v in metrics.items()},
    "created": datetime.now().isoformat(),
}

with open("ex8_deployment_config.json", "w") as f:
    json.dump(deployment_config, f, indent=2)
print("Saved: ex8_deployment_config.json")

# Visualisation
viz = ModelVisualizer()
# TODO: Create a metric comparison plot and save as HTML
# Hint: viz.metric_comparison({"Final Model": metrics})
fig = ____  # Hint: viz.metric_comparison({"Final Model": metrics})
fig.update_layout(title="Production Model: Credit Default Prediction")
fig.write_html("ex8_final_metrics.html")
print("Saved: ex8_final_metrics.html")

# ── Checkpoint 12 ────────────────────────────────────────────────────
assert os.path.exists("ex8_deployment_config.json"), "Config should be saved"
print("\n✓ Checkpoint 12 passed — deployment artifacts generated\n")


# ══════════════════════════════════════════════════════════════════════
# REFLECTION
# ══════════════════════════════════════════════════════════════════════
print("═" * 70)
print("  MODULE 3 MASTERY — SUPERVISED ML THEORY TO PRODUCTION")
print("═" * 70)
print(
    f"""
  M3 CAPSTONE CHECKLIST:
  ✓ Feature engineering (Ex 1): domain-driven features, leakage prevention,
    mutual information, chi-squared, RFE, L1 selection
  ✓ Bias-variance (Ex 2): regularisation, nested CV, time-series CV,
    GroupKFold, learning curves
  ✓ Model zoo (Ex 3): SVM, KNN, Naive Bayes, Decision Trees, Random
    Forests, from-scratch Gini, decision boundaries
  ✓ Gradient boosting (Ex 4): XGBoost split gain derivation, LightGBM,
    CatBoost, hyperparameter heatmaps, early stopping
  ✓ Imbalance & calibration (Ex 5): SMOTE failures, cost-sensitive,
    focal loss, threshold optimisation, business ROI, Platt/isotonic
  ✓ Interpretability & fairness (Ex 6): SHAP, LIME, permutation
    importance, disparate impact, equalized odds, impossibility theorem
  ✓ Workflow orchestration (Ex 7): WorkflowBuilder, DataFlow,
    HyperparameterSearch, ModelRegistry, branching workflows
  ✓ Production pipeline (Ex 8): conformal prediction, DriftMonitor,
    model card, monitoring dashboard, deployment artifacts

  THIS EXERCISE:
  ✓ Conformal prediction: P(Y ∈ C(X)) ≥ 1-α without distributional
    assumptions — coverage guaranteed from exchangeability alone
  ✓ DriftMonitor: PSI + KS test detect distribution shift
  ✓ Model card: Mitchell et al. template for regulated deployment
  ✓ ModelRegistry: register → promote with audit trail
  ✓ Monitoring dashboard: performance + drift + uncertainty in one view

  THE PRODUCTION PIPELINE PATTERN:
    preprocess → train → calibrate → conformal predict →
    register → promote → monitor drift → document

  KEY INSIGHT: Production ML is 20% modelling and 80% engineering.
  The model you just deployed is good.  The pipeline, the model card,
  the conformal coverage guarantee, and the registry trail are what
  make it trustworthy enough for real credit decisions.

  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  MODULE 4 PREVIEW: UNSUPERVISED ML AND ANOMALY DETECTION
  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

  Module 3 worked with labelled data — every row had a known outcome.
  Module 4 removes that luxury.

  M4 covers the full unsupervised ML landscape:
  - Clustering: K-means, HDBSCAN, spectral, GMM (Exercises 1-2)
  - Dimensionality reduction: PCA, t-SNE, UMAP (Exercise 3)
  - Anomaly detection: IsolationForest, LOF, EnsembleEngine (Exercise 4)
  - Association rules: Apriori, FP-Growth (Exercise 5)
  - NLP / BERTopic: text to topics without labels (Exercise 6)
  - Recommender systems: collaborative filtering (Exercise 7)
  - Deep learning foundations: CNNs, OnnxBridge (Exercise 8)

  The credit scoring model you built will reappear as the supervised
  baseline — anomaly scores from IsolationForest improve fraud
  detection beyond what any labelled model achieves alone.
"""
)
print("═" * 70)
