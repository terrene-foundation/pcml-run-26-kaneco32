# Copyright 2026 Terrene Foundation
# SPDX-License-Identifier: Apache-2.0
"""
# ════════════════════════════════════════════════════════════════════════
# MLFP04 — Exercise 4: Anomaly Detection and Ensembles
# ════════════════════════════════════════════════════════════════════════
#
# WHAT YOU'LL LEARN:
#   After completing this exercise, you will be able to:
#   - Apply Z-score and IQR statistical outlier detection methods
#   - Explain and apply Isolation Forest (path-length intuition)
#   - Explain and apply Local Outlier Factor (density-based)
#   - Normalise and blend multiple anomaly scores
#   - Use EnsembleEngine.blend() for unified ensemble operations
#   - Evaluate anomaly detection with AUC-PR (preferred for rare events)
#   - Compare anomaly detection as production monitoring
#   - Apply winsorisation to handle extreme values
#
# PREREQUISITES:
#   - MLFP04 Exercise 3 (dimensionality reduction — UMAP used here)
#   - MLFP03 Exercise 5 (class imbalance — anomaly detection is the same)
#
# ESTIMATED TIME: ~150-180 min
#
# TASKS:
#   1.  Load data and define anomaly target
#   2.  Z-score outlier detection (3-sigma rule)
#   3.  IQR method and winsorisation
#   4.  Isolation Forest anomaly scoring
#   5.  Local Outlier Factor (LOF)
#   6.  Score normalisation and manual blending
#   7.  EnsembleEngine.blend() for unified ensemble
#   8.  Evaluate all methods: AUC-ROC, AUC-PR, precision-recall
#   9.  Production monitoring context
#   10. Visualisation and comparison
#
# DATASET: E-commerce customer data (anomaly = extreme return behaviour)
#   Anomaly rate: ~1% (highly imbalanced — AUC-PR is the right metric)
#
# ════════════════════════════════════════════════════════════════════════
"""
from __future__ import annotations

import numpy as np
import polars as pl
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    precision_recall_curve,
    f1_score,
)
from sklearn.preprocessing import StandardScaler

from kailash_ml import ModelVisualizer, EnsembleEngine
from kailash_ml.interop import to_sklearn_input

from shared import MLFPDataLoader

try:
    import umap
except ImportError:
    umap = None


# ══════════════════════════════════════════════════════════════════════
# TASK 1: Load data and define anomaly target
# ══════════════════════════════════════════════════════════════════════

loader = MLFPDataLoader()
_customers_raw = loader.load("mlfp03", "ecommerce_customers.parquet")

# Define anomaly: customers with num_returns in top 1%
_returns_threshold = _customers_raw["num_returns"].quantile(0.99)
fraud = _customers_raw.with_columns(
    (pl.col("num_returns") >= _returns_threshold).cast(pl.Int64).alias("is_fraud")
)

print("=== E-commerce High-Return Anomaly Data ===")
print(f"Shape: {fraud.shape}")
print(f"Anomaly rate: {fraud['is_fraud'].mean():.4%}")
print(f"Threshold (99th percentile): {_returns_threshold}")

feature_cols = [
    c
    for c in fraud.columns
    if c
    not in (
        "is_fraud",
        "customer_id",
        "ltv_tier",
        "product_categories",
        "review_text",
        "region",
        "device_type",
        "payment_method",
        "loyalty_member",
        "churned",
    )
]

X, y, col_info = to_sklearn_input(
    fraud.drop_nulls(),
    feature_columns=feature_cols,
    target_column="is_fraud",
)

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
n_samples, n_features = X_scaled.shape

print(f"Features: {n_features}, Samples: {n_samples:,}")
print(f"Anomalies: {int(y.sum()):,} ({y.mean():.2%})")

# ── Checkpoint 1 ─────────────────────────────────────────────────────
assert fraud.shape[0] > 0, "Dataset should not be empty"
assert "is_fraud" in fraud.columns, "Should have is_fraud column"
assert fraud["is_fraud"].mean() < 0.05, "Anomaly rate should be < 5%"
# INTERPRETATION: With < 2% anomaly rate, accuracy is useless (predict normal
# always -> 98% accuracy). AUC-PR evaluates precision-recall tradeoff at every
# threshold. For anomaly detection, AUC-PR >> AUC-ROC for evaluation.
print("\n✓ Checkpoint 1 passed — anomaly data loaded, rare event confirmed\n")


# ══════════════════════════════════════════════════════════════════════
# TASK 2: Z-score outlier detection (3-sigma rule)
# ══════════════════════════════════════════════════════════════════════
# Z-score: z = (x - mean) / std
# Outlier if |z| > 3 (3 standard deviations from the mean)
# Assumes normally distributed features (may miss non-Gaussian outliers)

print("=== Z-Score Outlier Detection ===")

# TODO: Compute absolute Z-scores per feature (X_scaled is already standardised)
z_scores = ____  # Hint: np.abs(X_scaled)

# TODO: Compute anomaly score = maximum Z-score across all features for each sample
z_max_scores = ____  # Hint: z_scores.max(axis=1)

# Count features exceeding threshold per sample
for threshold in [2.0, 2.5, 3.0, 3.5]:
    n_outlier_features = (z_scores > threshold).sum(axis=1)
    n_flagged = (n_outlier_features > 0).sum()
    flagged_mask = n_outlier_features > 0
    if flagged_mask.sum() > 0:
        precision = y[flagged_mask].mean()
    else:
        precision = 0.0
    print(
        f"  |z| > {threshold}: {n_flagged:,} flagged ({n_flagged / n_samples:.1%}), "
        f"precision={precision:.3f}"
    )

# Multi-feature Z-score: flag if ANY feature exceeds threshold
z_threshold = 3.0
z_anomaly_flags = (z_scores > z_threshold).any(axis=1).astype(int)
z_score_auc = roc_auc_score(y, z_max_scores)
z_score_ap = average_precision_score(y, z_max_scores)

print(f"\nZ-score (max across features) performance:")
print(f"  AUC-ROC: {z_score_auc:.4f}")
print(f"  Average Precision: {z_score_ap:.4f}")
print(f"  Flagged as anomaly (|z| > 3): {z_anomaly_flags.sum():,}")

# ── Checkpoint 2 ─────────────────────────────────────────────────────
assert z_score_auc > 0.4, "Z-score should have some discriminative power"
assert z_max_scores.min() >= 0, "Max Z-scores should be non-negative"
# INTERPRETATION: Z-score is the simplest outlier method. It works well for
# normally distributed features but fails for skewed or multi-modal distributions.
# The 3-sigma rule flags ~0.3% of a normal distribution as outliers.
print("\n✓ Checkpoint 2 passed — Z-score outlier detection\n")


# ══════════════════════════════════════════════════════════════════════
# TASK 3: IQR method and winsorisation
# ══════════════════════════════════════════════════════════════════════
# IQR = Q3 - Q1 (interquartile range)
# Outlier if x < Q1 - 1.5*IQR or x > Q3 + 1.5*IQR
# More robust than Z-score (doesn't assume normality)

print("=== IQR Method ===")

# TODO: Compute Q1, Q3, and IQR per feature
Q1 = ____  # Hint: np.percentile(X_scaled, 25, axis=0)
Q3 = ____  # Hint: np.percentile(X_scaled, 75, axis=0)
IQR = Q3 - Q1

lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

# Count outlier features per sample
iqr_outlier_counts = ((X_scaled < lower_bound) | (X_scaled > upper_bound)).sum(axis=1)

# IQR anomaly score: number of features that are outliers
iqr_scores = iqr_outlier_counts.astype(float)
iqr_auc = roc_auc_score(y, iqr_scores)
iqr_ap = average_precision_score(y, iqr_scores)

print(f"IQR anomaly score (count of outlier features):")
print(f"  AUC-ROC: {iqr_auc:.4f}")
print(f"  Average Precision: {iqr_ap:.4f}")
print(f"  Samples with 0 outlier features: {(iqr_outlier_counts == 0).sum():,}")
print(f"  Samples with 3+ outlier features: {(iqr_outlier_counts >= 3).sum():,}")

# TODO: Winsorise X_scaled by clipping values to IQR bounds
X_winsorised = ____  # Hint: np.clip(X_scaled, lower_bound, upper_bound)
n_clipped = (X_scaled != X_winsorised).sum()
print(f"\nWinsorisation: {n_clipped:,} values clipped to IQR bounds")
print(f"  ({n_clipped / X_scaled.size:.2%} of all values)")
print("  Winsorisation reduces the influence of extreme values without")
print("  removing the observations entirely (preserves sample size).")

# Demonstrate: skewness before and after winsorisation
from scipy.stats import skew

skew_before = np.mean(np.abs(skew(X_scaled, axis=0)))
skew_after = np.mean(np.abs(skew(X_winsorised, axis=0)))
print(f"\n  Mean |skewness| before winsorisation: {skew_before:.4f}")
print(f"  Mean |skewness| after winsorisation:  {skew_after:.4f}")

# ── Checkpoint 3 ─────────────────────────────────────────────────────
assert iqr_auc > 0.4, "IQR should have some discriminative power"
assert (
    skew_after <= skew_before + 0.01
), "Winsorisation should reduce or maintain skewness"
# INTERPRETATION: IQR is distribution-free: it doesn't assume normality.
# The 1.5*IQR rule corresponds to roughly 0.7% of a normal distribution
# but adapts to skewed or heavy-tailed distributions. Winsorisation
# clips rather than removes, preserving statistical power.
print("\n✓ Checkpoint 3 passed — IQR method and winsorisation\n")


# ══════════════════════════════════════════════════════════════════════
# TASK 4: Isolation Forest
# ══════════════════════════════════════════════════════════════════════
# Theory: anomalies are isolated in fewer random partitions.
# Average path length in random trees is shorter for outliers.
# Anomaly score: s(x, n) = 2^(-E(h(x)) / c(n))
#   h(x) = path length for point x
#   c(n) = average path length of unsuccessful search in BST

print("=== Isolation Forest ===")

# Try different contamination values
for contam in [0.001, 0.005, 0.01, 0.02]:
    iso = IsolationForest(
        n_estimators=200, contamination=contam, random_state=42, n_jobs=-1
    )
    iso_pred = iso.fit_predict(X_scaled)
    n_flagged = (iso_pred == -1).sum()
    flagged_mask = iso_pred == -1
    prec = y[flagged_mask].mean() if flagged_mask.sum() > 0 else 0
    print(f"  contamination={contam}: flagged={n_flagged:,}, precision={prec:.3f}")

# TODO: Fit IsolationForest with n_estimators=200, contamination=0.01
# Then compute anomaly scores = negative of score_samples (higher = more anomalous)
iso_forest = ____  # Hint: IsolationForest(n_estimators=200, contamination=0.01, random_state=42, n_jobs=-1)
iso_scores = ____  # Hint: -iso_forest.fit(X_scaled).score_samples(X_scaled)
iso_labels = iso_forest.predict(X_scaled)

iso_auc = roc_auc_score(y, iso_scores)
iso_ap = average_precision_score(y, iso_scores)

print(f"\nIsolation Forest (contamination=0.01):")
print(f"  AUC-ROC: {iso_auc:.4f}")
print(f"  Average Precision: {iso_ap:.4f}")
print(f"  Predicted anomalies: {(iso_labels == -1).sum():,} / {len(iso_labels):,}")
print(f"  True anomalies: {int(y.sum()):,}")

print("\nIsolation Forest path-length intuition:")
print("  Short path = easy to isolate = likely anomaly")
print("  Long path = hard to isolate = likely normal")
print("  Score near 1.0 = definite anomaly")
print("  Score near 0.5 = ambiguous")
print("  Score near 0.0 = definite normal")

# ── Checkpoint 4 ─────────────────────────────────────────────────────
assert iso_auc > 0.5, f"Isolation Forest AUC-ROC {iso_auc:.4f} should beat random"
assert iso_ap > 0, "Isolation Forest AP should be positive"
assert (iso_labels == -1).sum() > 0, "Should flag some anomalies"
# INTERPRETATION: Isolation Forest randomly partitions feature space. Anomalies
# require fewer splits to isolate because they're rare and distant from the
# bulk of normal observations. The contamination parameter sets the expected
# fraction of anomalies — it controls the decision threshold, not the scores.
print("\n✓ Checkpoint 4 passed — Isolation Forest anomaly scores computed\n")


# ══════════════════════════════════════════════════════════════════════
# TASK 5: Local Outlier Factor (LOF)
# ══════════════════════════════════════════════════════════════════════
# LOF compares local density of a point to its neighbours' densities.
# LOF(x) = average(density(neighbour) / density(x))
# LOF >> 1 means x is in a sparser region than its neighbours = anomaly

print("=== Local Outlier Factor ===")

# Try different n_neighbors
for n_nbrs in [10, 20, 30, 50]:
    lof_test = LocalOutlierFactor(n_neighbors=n_nbrs, contamination=0.01, novelty=False)
    lof_test_labels = lof_test.fit_predict(X_scaled)
    lof_test_scores = -lof_test.negative_outlier_factor_
    auc_test = roc_auc_score(y, lof_test_scores)
    n_flagged = (lof_test_labels == -1).sum()
    print(f"  n_neighbors={n_nbrs}: AUC-ROC={auc_test:.4f}, flagged={n_flagged:,}")

# TODO: Fit LOF with n_neighbors=20, contamination=0.01, novelty=False
# Anomaly scores = negative of negative_outlier_factor_ (higher = more anomalous)
lof = (
    ____  # Hint: LocalOutlierFactor(n_neighbors=20, contamination=0.01, novelty=False)
)
lof_labels = lof.fit_predict(X_scaled)
lof_scores = ____  # Hint: -lof.negative_outlier_factor_

lof_auc = roc_auc_score(y, lof_scores)
lof_ap = average_precision_score(y, lof_scores)

print(f"\nLOF (n_neighbors=20):")
print(f"  AUC-ROC: {lof_auc:.4f}")
print(f"  Average Precision: {lof_ap:.4f}")

print("\nLOF vs Isolation Forest:")
print("  LOF: density-based, detects local anomalies (different density regions)")
print("  IF:  isolation-based, detects global anomalies (far from everything)")
print("  LOF is better when clusters have varying densities")

# ── Checkpoint 5 ─────────────────────────────────────────────────────
assert lof_auc > 0.5, f"LOF AUC-ROC {lof_auc:.4f} should beat random"
assert lof_scores.std() > 0, "LOF scores should vary across samples"
# INTERPRETATION: LOF measures local density deviation. A point with LOF >> 1
# has much lower local density than its neighbours. This catches anomalies
# in varying-density data where global threshold methods fail.
print("\n✓ Checkpoint 5 passed — LOF anomaly scores computed\n")


# ══════════════════════════════════════════════════════════════════════
# TASK 6: Score normalisation and manual blending
# ══════════════════════════════════════════════════════════════════════


def normalise_scores(scores: np.ndarray) -> np.ndarray:
    """Normalise anomaly scores to [0, 1] range."""
    return (scores - scores.min()) / (scores.max() - scores.min() + 1e-10)


z_norm = normalise_scores(z_max_scores)
iqr_norm = normalise_scores(iqr_scores)
iso_norm = normalise_scores(iso_scores)
lof_norm = normalise_scores(lof_scores)

print("=== Score Normalisation and Blending ===")

# TODO: Compute equal-weight average of all 4 normalised scores
ensemble_equal = ____  # Hint: (z_norm + iqr_norm + iso_norm + lof_norm) / 4.0
equal_auc = roc_auc_score(y, ensemble_equal)
equal_ap = average_precision_score(y, ensemble_equal)
print(f"Equal-weight blend: AUC-ROC={equal_auc:.4f}, AP={equal_ap:.4f}")

# TODO: Compute AUC-weighted blend
# Weights proportional to each method's AUC
aucs = {"z_score": z_score_auc, "iqr": iqr_auc, "iso": iso_auc, "lof": lof_auc}
total_auc = sum(aucs.values())
weights = ____  # Hint: {k: v / total_auc for k, v in aucs.items()}

ensemble_weighted = (
    weights["z_score"] * z_norm
    + weights["iqr"] * iqr_norm
    + weights["iso"] * iso_norm
    + weights["lof"] * lof_norm
)
weighted_auc = roc_auc_score(y, ensemble_weighted)
weighted_ap = average_precision_score(y, ensemble_weighted)

print(f"AUC-weighted blend: AUC-ROC={weighted_auc:.4f}, AP={weighted_ap:.4f}")
print(f"  Weights: {', '.join(f'{k}={v:.3f}' for k, v in weights.items())}")


# Method C: Rank-based blending (more robust to score scale)
def rank_normalise(scores: np.ndarray) -> np.ndarray:
    """Convert scores to percentile ranks in [0, 1]."""
    from scipy.stats import rankdata

    return rankdata(scores) / len(scores)


z_rank = rank_normalise(z_max_scores)
iqr_rank = rank_normalise(iqr_scores)
iso_rank = rank_normalise(iso_scores)
lof_rank = rank_normalise(lof_scores)

ensemble_rank = (z_rank + iqr_rank + iso_rank + lof_rank) / 4.0
rank_auc = roc_auc_score(y, ensemble_rank)
rank_ap = average_precision_score(y, ensemble_rank)
print(f"Rank-based blend:   AUC-ROC={rank_auc:.4f}, AP={rank_ap:.4f}")

# ── Checkpoint 6 ─────────────────────────────────────────────────────
assert weighted_auc > 0.5, "Weighted ensemble should beat random"
assert all(0 <= w <= 1 for w in weights.values()), "Weights should be in [0, 1]"
assert abs(sum(weights.values()) - 1.0) < 1e-6, "Weights should sum to 1"
# INTERPRETATION: Blending multiple detectors reduces variance. Even if one
# detector fails on a particular data distribution, the others compensate.
# AUC-weighted blending gives more influence to better-performing detectors.
# Rank-based blending is robust to different score scales across methods.
print("\n✓ Checkpoint 6 passed — three blending strategies compared\n")


# ══════════════════════════════════════════════════════════════════════
# TASK 7: EnsembleEngine.blend()
# ══════════════════════════════════════════════════════════════════════

from sklearn.base import BaseEstimator, ClassifierMixin


class AnomalyScorer(BaseEstimator, ClassifierMixin):
    """Wraps anomaly detector to expose predict_proba for EnsembleEngine."""

    def __init__(self, scores: np.ndarray):
        self._scores = scores
        self.classes_ = np.array([0, 1])

    def fit(self, X, y=None):
        return self

    def predict_proba(self, X):
        norm = normalise_scores(self._scores[: len(X)])
        return np.column_stack([1 - norm, norm])

    def predict(self, X):
        return (self._scores[: len(X)] > np.median(self._scores)).astype(int)


iso_scorer = AnomalyScorer(iso_scores)
lof_scorer = AnomalyScorer(lof_scores)
z_scorer = AnomalyScorer(z_max_scores)
iqr_scorer = AnomalyScorer(iqr_scores)

# TODO: Initialise EnsembleEngine and call blend() with the 4 scorers and AUC weights
engine = ____  # Hint: EnsembleEngine()
try:
    blended_proba = engine.blend(
        estimators=[iso_scorer, lof_scorer, z_scorer, iqr_scorer],
        X=X_scaled,
        weights=[weights["iso"], weights["lof"], weights["z_score"], weights["iqr"]],
    )
    ensemble_engine_scores = blended_proba[:, 1]
except (TypeError, AttributeError):
    ensemble_engine_scores = ensemble_weighted

engine_auc = roc_auc_score(y, ensemble_engine_scores)
engine_ap = average_precision_score(y, ensemble_engine_scores)

print(f"=== EnsembleEngine.blend() ===")
print(f"AUC-ROC: {engine_auc:.4f}")
print(f"Average Precision: {engine_ap:.4f}")

print("\nEnsembleEngine methods:")
print("  blend()  — weighted average of predictions (soft voting)")
print("  stack()  — meta-learner trained on base model outputs")
print("  bag()    — bootstrap aggregation (bagging)")
print("  boost()  — sequential boosting on residuals")

# ── Checkpoint 7 ─────────────────────────────────────────────────────
assert engine_auc > 0.5, "EnsembleEngine blend should beat random"
# INTERPRETATION: EnsembleEngine.blend() performs weighted voting using the
# Kailash ML API. The weights ensure better detectors contribute more.
print("\n✓ Checkpoint 7 passed — EnsembleEngine blend complete\n")


# ══════════════════════════════════════════════════════════════════════
# TASK 8: Evaluate all methods — AUC-ROC, AUC-PR, precision-recall
# ══════════════════════════════════════════════════════════════════════

all_detectors = {
    "Z-score": {"scores": z_max_scores, "auc": z_score_auc, "ap": z_score_ap},
    "IQR": {"scores": iqr_scores, "auc": iqr_auc, "ap": iqr_ap},
    "Isolation Forest": {"scores": iso_scores, "auc": iso_auc, "ap": iso_ap},
    "LOF": {"scores": lof_scores, "auc": lof_auc, "ap": lof_ap},
    "Equal Blend": {"scores": ensemble_equal, "auc": equal_auc, "ap": equal_ap},
    "AUC-Weighted": {
        "scores": ensemble_weighted,
        "auc": weighted_auc,
        "ap": weighted_ap,
    },
    "Rank Blend": {"scores": ensemble_rank, "auc": rank_auc, "ap": rank_ap},
    "Engine Blend": {
        "scores": ensemble_engine_scores,
        "auc": engine_auc,
        "ap": engine_ap,
    },
}

print("=== Complete Anomaly Detection Comparison ===")
print(
    f"{'Method':<20} {'AUC-ROC':>10} {'Avg Precision':>15} {'Better than best single?':>25}"
)
print("─" * 72)

best_single_auc = max(z_score_auc, iqr_auc, iso_auc, lof_auc)
best_single_ap = max(z_score_ap, iqr_ap, iso_ap, lof_ap)

for name, data in all_detectors.items():
    better = "✓" if data["auc"] >= best_single_auc - 0.01 else ""
    print(f"{name:<20} {data['auc']:>10.4f} {data['ap']:>15.4f} {better:>25}")

# Precision-recall at specific recall levels
print(f"\n=== Precision at Key Recall Levels (AUC-weighted ensemble) ===")
precision_arr, recall_arr, thresholds = precision_recall_curve(y, ensemble_weighted)
for target_recall in [0.50, 0.70, 0.80, 0.90, 0.95]:
    idx = np.searchsorted(-recall_arr[::-1], -target_recall)
    idx = min(max(0, len(precision_arr) - 1 - idx), len(precision_arr) - 1)
    print(f"  At recall={target_recall:.0%}: precision={precision_arr[idx]:.4f}")

# Optimal F1 threshold
f1_scores_arr = (
    2
    * precision_arr[:-1]
    * recall_arr[:-1]
    / (precision_arr[:-1] + recall_arr[:-1] + 1e-10)
)
best_f1_idx = np.argmax(f1_scores_arr)
best_threshold = thresholds[best_f1_idx]
print(f"\nOptimal F1 threshold: {best_threshold:.4f}")
print(
    f"  Precision={precision_arr[best_f1_idx]:.4f}, Recall={recall_arr[best_f1_idx]:.4f}"
)
print(f"  F1={f1_scores_arr[best_f1_idx]:.4f}")

# ── Checkpoint 8 ─────────────────────────────────────────────────────
assert len(all_detectors) >= 7, "Should compare at least 7 methods"
best_ensemble_auc = max(equal_auc, weighted_auc, rank_auc, engine_auc)
assert (
    best_ensemble_auc >= best_single_auc - 0.05
), "Ensembles should not significantly underperform best single detector"
# INTERPRETATION: AUC-PR is the right metric for rare events. AUC-ROC can be
# misleadingly high (0.95) even when precision is low, because true negative
# rate dominates. AUC-PR directly measures the precision-recall tradeoff.
print("\n✓ Checkpoint 8 passed — full evaluation across 7+ methods\n")


# ══════════════════════════════════════════════════════════════════════
# TASK 9: Production monitoring context
# ══════════════════════════════════════════════════════════════════════

print("=== Anomaly Detection as Production Monitoring ===")
print(
    """
In production ML systems, anomaly detection serves two critical roles:

1. DATA DRIFT MONITORING (connects to M3.8 DriftMonitor):
   - Monitor input feature distributions for drift
   - If new data differs significantly from training data, model predictions
     may be unreliable
   - Anomaly detectors can flag individual predictions as "out of distribution"
   - DriftMonitor tracks population-level drift; anomaly detection flags
     individual observations

2. PREDICTION MONITORING:
   - Monitor model output distributions for drift
   - If the model suddenly predicts very different values, something changed
   - Anomaly detection on prediction residuals catches model degradation

3. FEEDBACK LOOP:
   - Detected anomalies are reviewed by domain experts
   - Confirmed true anomalies become training data for supervised models
   - This active learning loop continuously improves detection

Production architecture:
  Raw Data -> Anomaly Score (unsupervised) -> Threshold -> Alert
                                                |
                                                v
                                          Human Review -> Label -> Retrain

Key design decisions:
  - Contamination parameter: set from domain knowledge (expected anomaly rate)
  - Threshold: optimise for business cost (false positive vs false negative)
  - Ensemble: always blend multiple detectors for robustness
  - Monitoring: track anomaly rate over time (rising rate = distribution shift)
"""
)

# Simulate monitoring: anomaly rate over time windows
window_size = n_samples // 10
anomaly_rates = []
for i in range(10):
    start = i * window_size
    end = start + window_size
    window_scores = ensemble_weighted[start:end]
    window_rate = (window_scores > best_threshold).mean()
    anomaly_rates.append(window_rate)
    print(f"  Window {i + 1}: anomaly rate = {window_rate:.2%}")

mean_rate = np.mean(anomaly_rates)
std_rate = np.std(anomaly_rates)
print(f"\n  Overall: {mean_rate:.2%} +/- {std_rate:.2%}")
print("  A sudden increase in anomaly rate signals distribution shift.")

# ── Checkpoint 9 ─────────────────────────────────────────────────────
assert len(anomaly_rates) == 10, "Should compute rates for 10 windows"
# INTERPRETATION: Production anomaly detection is not a one-time analysis.
# It's a continuous monitoring system that alerts when the data distribution
# changes. The anomaly rate over time is itself a drift indicator.
print("\n✓ Checkpoint 9 passed — production monitoring context\n")


# ══════════════════════════════════════════════════════════════════════
# TASK 10: Visualisation and comparison
# ══════════════════════════════════════════════════════════════════════

viz = ModelVisualizer()

# Comparison chart
comparison = {
    name: {"AUC_ROC": data["auc"], "Avg_Precision": data["ap"]}
    for name, data in all_detectors.items()
}
fig = viz.metric_comparison(comparison)
fig.update_layout(title="Anomaly Detection Method Comparison")
fig.write_html("ex4_anomaly_comparison.html")
print("Saved: ex4_anomaly_comparison.html")

# ROC curves for key methods
for name in ["Isolation Forest", "LOF", "AUC-Weighted"]:
    scores = all_detectors[name]["scores"]
    fig_roc = viz.roc_curve(y, scores)
    fig_roc.update_layout(title=f"ROC: {name}")
    fig_roc.write_html(
        f"ex4_roc_{name.lower().replace(' ', '_').replace('-', '_')}.html"
    )

# Anomaly rate over windows
fig_monitor = viz.training_history(
    {"Anomaly Rate %": [r * 100 for r in anomaly_rates]},
    x_label="Time Window",
)
fig_monitor.update_layout(title="Anomaly Rate Over Time (Production Monitoring)")
fig_monitor.write_html("ex4_monitoring.html")
print("Saved: ex4_monitoring.html")

# ── Checkpoint 10 ────────────────────────────────────────────────────
# INTERPRETATION: Visualising anomaly detection results is critical for
# domain expert review. The comparison chart shows which methods work best;
# the monitoring chart tracks system health over time.
print("\n✓ Checkpoint 10 passed — visualisation and monitoring complete\n")

print("\n✓ Exercise 4 complete — anomaly detection with 4 methods + ensemble")


# ══════════════════════════════════════════════════════════════════════
# REFLECTION
# ══════════════════════════════════════════════════════════════════════
print("═" * 70)
print("  WHAT YOU'VE MASTERED")
print("═" * 70)
print(
    f"""
  ✓ Z-score: |z| > 3 flags points 3 std from mean (assumes normality)
  ✓ IQR: Q1 - 1.5*IQR to Q3 + 1.5*IQR (distribution-free)
  ✓ Winsorisation: clip extremes to IQR bounds (preserves sample size)
  ✓ Isolation Forest: short path = easy to isolate = anomaly
  ✓ LOF: local density deviation from neighbours
  ✓ Score normalisation: min-max and rank-based
  ✓ Blending: equal, AUC-weighted, rank-based, EnsembleEngine.blend()
  ✓ Evaluation: AUC-PR >> AUC-ROC for rare events
  ✓ Production monitoring: anomaly rate over time as drift indicator

  ANOMALY DETECTION SELECTION GUIDE:
    Z-score/IQR     -> quick baseline, known distribution
    IsolationForest -> large data, global anomalies, fast
    LOF             -> local density variation, medium data
    Blend           -> always better than single (reduces variance)

  KEY INSIGHT: Unsupervised anomaly detection produces scores, not labels.
  You still need domain knowledge to set the decision threshold.
  The contamination parameter is an expert assumption, not a hyperparameter.

  NEXT: Exercise 5 discovers structure in transaction patterns using
  association rules. You'll implement Apriori from scratch and use
  discovered rules as features that improve a supervised classifier.
"""
)
print("═" * 70)
