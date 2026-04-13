# Copyright 2026 Terrene Foundation
# SPDX-License-Identifier: Apache-2.0
"""
# ════════════════════════════════════════════════════════════════════════
# MLFP03 — Exercise 3: The Complete Supervised Model Zoo
# ════════════════════════════════════════════════════════════════════════
#
# WHAT YOU'LL LEARN:
#   After completing this exercise, you will be able to:
#   - Train and compare SVM, KNN, Naive Bayes, Decision Trees, and
#     Random Forests using a consistent evaluation framework
#   - Compute Gini impurity from scratch and verify against sklearn
#   - Simulate a decision tree split step-by-step (from-scratch splitting)
#   - Use OOB (out-of-bag) estimation as a free cross-validation proxy
#   - Visualise decision boundaries in 2D PCA space
#   - Select the appropriate algorithm based on data characteristics
#     (size, dimensionality, interpretability requirements)
#   - Build a model comparison table with timing and justify the best
#     choice
#
# PREREQUISITES:
#   - MLFP03 Exercise 2 (bias-variance, regularisation, CV strategies)
#   - MLFP02 Module (statistics, Bayesian priors — connects to Naive Bayes)
#
# ESTIMATED TIME: ~150-180 minutes
#
# TASKS:
#   1.  Load e-commerce data, set up binary classification (churn)
#   2.  Preprocess: encode categoricals, scale numerics
#   3.  Train SVM (linear + RBF), tune C parameter
#   4.  Train KNN, experiment with k values and distance metrics
#   5.  Train Naive Bayes (GaussianNB), inspect class-conditional params
#   6.  From-scratch Gini impurity computation
#   7.  From-scratch decision tree splitting (best split search)
#   8.  Train Decision Tree with sklearn, depth tuning, tree visualisation
#   9.  Train Random Forest, extract feature importance, OOB score
#   10. Decision boundary visualisation in 2D PCA space
#   11. Comprehensive model comparison table with timing
#   12. When to use each model — decision guide
#
# DATASET: E-commerce customer data (mlfp03/ecommerce_customers.parquet)
#   Target: churned (binary — 0=retained, 1=churned)
#   Rows: ~5,000 customers | Features: behavioural + demographic
#   Why this dataset: realistic churn rates, mixed feature types
#
# THEORY:
#   SVM margin: maximise 2/||w|| subject to y_i(w . x_i + b) >= 1
#   Kernel trick: K(x, x') = phi(x) . phi(x') without computing phi
#   Gini impurity: G = 1 - Sum(p_k^2) for k classes
#   Information gain: IG = H(parent) - Sum(w_j * H(child_j))
#   OOB estimation: ~36.8% of samples excluded per bootstrap sample
#     (probability of NOT being drawn in n trials = (1 - 1/n)^n -> 1/e)
#
# ════════════════════════════════════════════════════════════════════════
"""
from __future__ import annotations

import time

import numpy as np
import polars as pl
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier, export_text
from sklearn.ensemble import RandomForestClassifier
from sklearn.decomposition import PCA
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    classification_report,
    roc_auc_score,
)

from kailash_ml import PreprocessingPipeline, ModelVisualizer
from kailash_ml.interop import to_sklearn_input

from shared import MLFPDataLoader


# ── Data Loading ──────────────────────────────────────────────────────

loader = MLFPDataLoader()
ecommerce = loader.load("mlfp03", "ecommerce_customers.parquet")

# SVM with RBF kernel is O(n²) — subsample to keep training time reasonable
ecommerce = ecommerce.sample(n=5000, seed=42)

print("=== E-Commerce Customer Dataset ===")
print(f"Shape: {ecommerce.shape} (subsampled for SVM tractability)")
print(f"Columns: {ecommerce.columns}")
print(f"Churn rate: {ecommerce['churned'].mean():.2%}")

# Drop text columns and high-cardinality strings
ecommerce = ecommerce.drop("customer_id", "review_text", "product_categories")

print(f"\nAfter dropping text/ID columns: {ecommerce.shape}")
print(f"Remaining columns: {ecommerce.columns}")


# ════════════════════════════════════════════════════════════════════════
# TASK 1: Set up classification task
# ════════════════════════════════════════════════════════════════════════

target_col = "churned"

pipeline = PreprocessingPipeline()
# TODO: Call pipeline.setup() with the ecommerce data, target="churned",
#       train_size=0.8, seed=42, normalize=True, normalize_method="zscore",
#       categorical_encoding="ordinal", imputation_strategy="median"
result = pipeline.setup(
    data=____,  # Hint: the polars DataFrame
    target=____,  # Hint: the target column name string
    train_size=____,  # Hint: 0.8 for 80/20 split
    seed=42,
    normalize=____,  # Hint: True — SVM and KNN require scaled features
    normalize_method=____,  # Hint: "zscore"
    categorical_encoding=____,  # Hint: "ordinal"
    imputation_strategy=____,  # Hint: "median"
)

print(f"\nTask type: {result.task_type}")
print(f"Train: {result.train_data.shape}, Test: {result.test_data.shape}")

feature_cols = [c for c in result.train_data.columns if c != target_col]

X_train, y_train, col_info = to_sklearn_input(
    result.train_data, feature_columns=feature_cols, target_column=target_col
)
X_test, y_test, _ = to_sklearn_input(
    result.test_data, feature_columns=feature_cols, target_column=target_col
)
feature_names = col_info["feature_columns"]

print(f"Features ({len(feature_names)}): {feature_names}")
print(f"Train: {X_train.shape}, Test: {X_test.shape}")
print(f"Train churn rate: {y_train.mean():.2%}")

# Cross-validation strategy — same folds for all models
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)


# ════════════════════════════════════════════════════════════════════════
# TASK 2: Preprocessing verification
# ════════════════════════════════════════════════════════════════════════

print("\n=== Feature Scales (post-normalisation) ===")
print(f"{'Feature':<30} {'Mean':>8} {'Std':>8} {'Min':>8} {'Max':>8}")
print("-" * 62)
for i, name in enumerate(feature_names[:10]):
    col = X_train[:, i]
    print(
        f"{name:<30} {col.mean():>8.3f} {col.std():>8.3f} "
        f"{col.min():>8.3f} {col.max():>8.3f}"
    )

# ── Checkpoint 1 ─────────────────────────────────────────────────────
assert X_train.shape[0] > 0, "Training set is empty"
assert X_test.shape[0] > 0, "Test set is empty"
assert len(feature_names) > 0, "No feature columns found"
assert 0 < y_train.mean() < 1, "Target should be binary with mixed labels"
print("\n✓ Checkpoint 1 passed — data prepared for 5-model comparison\n")


# ════════════════════════════════════════════════════════════════════════
# TASK 3: SVM — Support Vector Machines
# ════════════════════════════════════════════════════════════════════════
# THEORY: SVM finds the hyperplane that maximises the margin between
# classes.
#   Hard-margin SVM: maximise 2/||w|| s.t. y_i(w . x_i + b) >= 1
#   Soft-margin SVM: allow misclassification via slack variables ξ_i,
#     controlled by C (penalty parameter).
#     - C → ∞: hard margin (no misclassification allowed)
#     - C → 0: wide margin, many misclassifications tolerated
#
#   Kernel trick: map data to higher-dimensional space where linear
#   separation is possible. K(x, x') = phi(x) · phi(x') computes
#   the dot product in feature space without explicit transformation.
#     - Linear: K(x, x') = x · x'
#     - RBF: K(x, x') = exp(-gamma ||x - x'||²)

print("\n=== SVM: Support Vector Machines ===")

# 3a: Linear SVM — tune C parameter
print("\n--- Linear SVM: C parameter sweep ---")
print(f"{'C':>10} {'CV Accuracy':>14} {'CV F1':>10}")
print("-" * 38)

c_values = [0.01, 0.1, 1.0, 10.0, 100.0]
linear_svm_results = {}
for c_val in c_values:
    # TODO: Create SVC with kernel="linear", C=c_val, random_state=42
    svm_lin = SVC(kernel=____, C=____, random_state=42)
    # TODO: Compute cross_val_score for accuracy, then for f1
    acc_scores = cross_val_score(
        ____, ____, ____, cv=cv, scoring=____
    )  # Hint: "accuracy"
    f1_scores = cross_val_score(____, ____, ____, cv=cv, scoring=____)  # Hint: "f1"
    linear_svm_results[c_val] = {
        "accuracy": acc_scores.mean(),
        "f1": f1_scores.mean(),
    }
    print(f"{c_val:>10.2f} {acc_scores.mean():>14.4f} {f1_scores.mean():>10.4f}")

best_c_linear = max(linear_svm_results, key=lambda c: linear_svm_results[c]["f1"])

# 3b: RBF SVM — tune C parameter
print("\n--- RBF SVM: C parameter sweep ---")
print(f"{'C':>10} {'CV Accuracy':>14} {'CV F1':>10}")
print("-" * 38)

rbf_svm_results = {}
for c_val in c_values:
    svm_rbf = SVC(kernel="rbf", C=c_val, random_state=42)
    acc_scores = cross_val_score(svm_rbf, X_train, y_train, cv=cv, scoring="accuracy")
    f1_scores = cross_val_score(svm_rbf, X_train, y_train, cv=cv, scoring="f1")
    rbf_svm_results[c_val] = {
        "accuracy": acc_scores.mean(),
        "f1": f1_scores.mean(),
    }
    print(f"{c_val:>10.2f} {acc_scores.mean():>14.4f} {f1_scores.mean():>10.4f}")

best_c_rbf = max(rbf_svm_results, key=lambda c: rbf_svm_results[c]["f1"])

# Train final SVM on full training set
t0 = time.perf_counter()
# TODO: Create SVC with kernel="rbf", best_c_rbf, random_state=42, probability=True
#       then .fit(X_train, y_train)
svm_final = SVC(
    kernel=____, C=____, random_state=42, probability=____
)  # Hint: probability=True
svm_final.fit(X_train, y_train)
svm_train_time = time.perf_counter() - t0

svm_pred = svm_final.predict(X_test)
svm_prob = svm_final.predict_proba(X_test)[:, 1]

print(f"\nSVM Final (RBF, C={best_c_rbf}): trained in {svm_train_time:.2f}s")
print(classification_report(y_test, svm_pred, target_names=["Retained", "Churned"]))

print(f"  Support vectors: {svm_final.n_support_} (per class)")
print(
    f"  Total: {svm_final.support_vectors_.shape[0]} "
    f"({svm_final.support_vectors_.shape[0] / len(y_train):.1%} of training data)"
)

# ── Checkpoint 2 ─────────────────────────────────────────────────────
svm_acc = accuracy_score(y_test, svm_pred)
assert svm_acc > 0.5, f"SVM accuracy {svm_acc:.4f} should be above random"
# INTERPRETATION: If 80% of training samples are support vectors, the model
# has memorised the training set.  A well-trained SVM typically uses 10-30%.
print("\n✓ Checkpoint 2 passed — SVM trained and evaluated\n")


# ════════════════════════════════════════════════════════════════════════
# TASK 4: KNN — K-Nearest Neighbors
# ════════════════════════════════════════════════════════════════════════
# THEORY: KNN is instance-based (lazy) — no training phase.
#   Prediction: find the k closest training points, majority vote.
#   Distance metrics: Euclidean, Manhattan, Cosine
#   Curse of dimensionality: as dims grow, all points become equidistant.

print("\n=== KNN: K-Nearest Neighbors ===")

# 4a: Vary k
print("\n--- KNN: k value sweep ---")
print(f"{'k':>6} {'CV Accuracy':>14} {'CV F1':>10}")
print("-" * 34)

k_values = [1, 3, 5, 7, 11, 15, 21, 31]
knn_k_results = {}
for k in k_values:
    # TODO: Create KNeighborsClassifier with n_neighbors=k, metric="euclidean"
    knn = KNeighborsClassifier(n_neighbors=____, metric=____)  # Hint: "euclidean"
    acc_scores = cross_val_score(knn, X_train, y_train, cv=cv, scoring="accuracy")
    f1_scores = cross_val_score(knn, X_train, y_train, cv=cv, scoring="f1")
    knn_k_results[k] = {"accuracy": acc_scores.mean(), "f1": f1_scores.mean()}
    print(f"{k:>6} {acc_scores.mean():>14.4f} {f1_scores.mean():>10.4f}")

best_k = max(knn_k_results, key=lambda k: knn_k_results[k]["f1"])

# 4b: Compare distance metrics
print(f"\n--- KNN: distance metric comparison (k={best_k}) ---")
print(f"{'Metric':<12} {'CV Accuracy':>14} {'CV F1':>10}")
print("-" * 40)

metrics_list = ["euclidean", "manhattan", "cosine"]
knn_metric_results = {}
for metric in metrics_list:
    knn = KNeighborsClassifier(n_neighbors=best_k, metric=metric)
    acc_scores = cross_val_score(knn, X_train, y_train, cv=cv, scoring="accuracy")
    f1_scores = cross_val_score(knn, X_train, y_train, cv=cv, scoring="f1")
    knn_metric_results[metric] = {"accuracy": acc_scores.mean(), "f1": f1_scores.mean()}
    print(f"{metric:<12} {acc_scores.mean():>14.4f} {f1_scores.mean():>10.4f}")

best_metric = max(knn_metric_results, key=lambda m: knn_metric_results[m]["f1"])

# Train final KNN
t0 = time.perf_counter()
knn_final = KNeighborsClassifier(n_neighbors=best_k, metric=best_metric)
knn_final.fit(X_train, y_train)
knn_train_time = time.perf_counter() - t0

knn_pred = knn_final.predict(X_test)
knn_prob = knn_final.predict_proba(X_test)[:, 1]

print(f"\nKNN Final (k={best_k}, {best_metric}): trained in {knn_train_time:.4f}s")
print(classification_report(y_test, knn_pred, target_names=["Retained", "Churned"]))

# ── Checkpoint 3 ─────────────────────────────────────────────────────
knn_acc = accuracy_score(y_test, knn_pred)
assert knn_acc > 0.5, f"KNN accuracy {knn_acc:.4f} should be above random"
assert best_k > 1, "Best k should be > 1 (k=1 always overfits)"
# INTERPRETATION: k=1 memorises noise; k=n votes globally.  The
# cross-validated k balances both.  Training is near-zero (lazy) but
# prediction requires computing distances to all points — cost deferred.
print("\n✓ Checkpoint 3 passed — KNN trained and optimal k found\n")


# ════════════════════════════════════════════════════════════════════════
# TASK 5: Naive Bayes (GaussianNB)
# ════════════════════════════════════════════════════════════════════════
# THEORY: P(y|x_1,...,x_n) ∝ P(y) * ∏ P(x_i|y)
# GaussianNB assumes each P(x_i|y) ~ N(μ_iy, σ_iy²).

print("\n=== Naive Bayes (GaussianNB) ===")

t0 = time.perf_counter()
nb = GaussianNB()
nb.fit(X_train, y_train)
nb_train_time = time.perf_counter() - t0

nb_pred = nb.predict(X_test)
nb_prob = nb.predict_proba(X_test)[:, 1]

print(f"GaussianNB: trained in {nb_train_time:.4f}s")
print(classification_report(y_test, nb_pred, target_names=["Retained", "Churned"]))

# TODO: Compute CV accuracy scores for nb on X_train, y_train using cv, scoring="accuracy"
nb_cv_acc = cross_val_score(
    ____, ____, ____, cv=cv, scoring=____
)  # Hint: nb, X_train, y_train, "accuracy"
# TODO: Compute CV f1 scores for nb on X_train, y_train using cv, scoring="f1"
nb_cv_f1 = cross_val_score(
    ____, ____, ____, cv=cv, scoring=____
)  # Hint: nb, X_train, y_train, "f1"
print(f"CV Accuracy: {nb_cv_acc.mean():.4f} (±{nb_cv_acc.std():.4f})")
print(f"CV F1:       {nb_cv_f1.mean():.4f} (±{nb_cv_f1.std():.4f})")

# Inspect class-conditional parameters
print(
    f"\nClass priors: P(retained)={nb.class_prior_[0]:.4f}, "
    f"P(churned)={nb.class_prior_[1]:.4f}"
)
print(f"\nClass-conditional means:")
print(f"{'Feature':<30} {'Retained':>10} {'Churned':>10} {'|Diff|':>10}")
print("-" * 64)
for i, name in enumerate(feature_names[:10]):
    mu_0 = nb.theta_[0, i]
    mu_1 = nb.theta_[1, i]
    print(f"{name:<30} {mu_0:>10.4f} {mu_1:>10.4f} {abs(mu_1 - mu_0):>10.4f}")

# ── Checkpoint 4 ─────────────────────────────────────────────────────
nb_acc = accuracy_score(y_test, nb_pred)
assert nb_acc > 0.5, f"NB accuracy {nb_acc:.4f} should be above random"
assert abs(nb.class_prior_[0] + nb.class_prior_[1] - 1.0) < 1e-6, "Priors must sum to 1"
# INTERPRETATION: NB class priors directly reflect training class balance.
# This is Bayes' theorem in action, connecting back to M2 Bayesian thinking.
print("\n✓ Checkpoint 4 passed — Naive Bayes trained and class priors inspected\n")


# ════════════════════════════════════════════════════════════════════════
# TASK 6: From-scratch Gini impurity computation
# ════════════════════════════════════════════════════════════════════════
# Gini impurity: G(node) = 1 - Σ p_k² for k classes
#   Pure node: G = 0 (all one class)
#   Maximum impurity (binary): G = 0.5 (50/50 split)
#
# Information gain: IG = H(parent) - Σ w_j * H(child_j)
#   H (entropy) = -Σ p_k log₂(p_k)

print("\n=== From-Scratch: Gini Impurity ===")


def gini_impurity(y: np.ndarray) -> float:
    """Compute Gini impurity from scratch.

    G = 1 - Σ p_k² where p_k is the proportion of class k.
    """
    # TODO: Get unique classes and counts using np.unique(y, return_counts=True)
    classes, counts = np.unique(____, return_counts=____)  # Hint: y, True
    # TODO: Compute proportions = counts / len(y)
    proportions = ____  # Hint: counts / len(y)
    # TODO: Return 1.0 - np.sum(proportions**2)
    return ____  # Hint: 1.0 - np.sum(proportions**2)


def entropy(y: np.ndarray) -> float:
    """Compute entropy from scratch.

    H = -Σ p_k log₂(p_k)
    """
    classes, counts = np.unique(y, return_counts=True)
    proportions = counts / len(y)
    # Avoid log(0) by filtering out zero proportions
    proportions = proportions[proportions > 0]
    return -np.sum(proportions * np.log2(proportions))


# Manual computation at root node
n_retained = (y_train == 0).sum()
n_churned = (y_train == 1).sum()
n_total = len(y_train)

p_retained = n_retained / n_total
p_churned = n_churned / n_total

gini_root = gini_impurity(y_train)
entropy_root = entropy(y_train)

print(f"\n--- Root Node ---")
print(f"Training data: {n_retained} retained, {n_churned} churned (n={n_total})")
print(f"p(retained) = {p_retained:.4f}")
print(f"p(churned)  = {p_churned:.4f}")
print(f"Gini = 1 - (p_retained² + p_churned²)")
print(f"     = 1 - ({p_retained:.4f}² + {p_churned:.4f}²)")
print(f"     = 1 - ({p_retained**2:.4f} + {p_churned**2:.4f})")
print(f"     = {gini_root:.4f}")
print(f"Entropy = {entropy_root:.4f}")

# Verify against manual calculation
manual_gini = 1 - (p_retained**2 + p_churned**2)
assert abs(gini_root - manual_gini) < 1e-9, "Gini mismatch with manual calc"

# Demonstrate Gini for different class distributions
print(f"\n--- Gini Impurity for Different Distributions ---")
print(f"{'Distribution':<25} {'Gini':>8} {'Entropy':>8}")
print("-" * 43)
test_cases = [
    ("Pure: [100, 0]", np.array([0] * 100)),
    ("50/50: [50, 50]", np.array([0] * 50 + [1] * 50)),
    ("90/10: [90, 10]", np.array([0] * 90 + [1] * 10)),
    ("70/30: [70, 30]", np.array([0] * 70 + [1] * 30)),
    ("3-class: [40,30,30]", np.array([0] * 40 + [1] * 30 + [2] * 30)),
]
for label, y_demo in test_cases:
    print(f"  {label:<23} {gini_impurity(y_demo):>8.4f} {entropy(y_demo):>8.4f}")

# ── Checkpoint 5 ─────────────────────────────────────────────────────
assert (
    abs(gini_impurity(np.array([0] * 100)) - 0.0) < 1e-9
), "Pure node Gini should be 0"
assert (
    abs(gini_impurity(np.array([0] * 50 + [1] * 50)) - 0.5) < 1e-9
), "50/50 binary Gini should be 0.5"
# INTERPRETATION: Gini measures the expected error if you randomly label
# a sample from the node's distribution.  Pure = 0 error.  50/50 = max
# uncertainty.  The tree algorithm maximises REDUCTION in Gini per split.
print("\n✓ Checkpoint 5 passed — Gini impurity computed from scratch\n")


# ════════════════════════════════════════════════════════════════════════
# TASK 7: From-scratch decision tree splitting
# ════════════════════════════════════════════════════════════════════════
# Simulate the FIRST split of a decision tree by exhaustively searching
# over all features and all thresholds for the maximum Gini gain.


def best_split_search(X: np.ndarray, y: np.ndarray, feature_names: list[str]):
    """Find the best split by exhaustive search (from scratch).

    For each feature, try each unique value as a threshold and compute
    the weighted Gini impurity of the resulting children.
    Return the feature, threshold, and gain of the best split.
    """
    n = len(y)
    parent_gini = gini_impurity(y)
    best_gain = 0.0
    best_feature_idx = 0
    best_threshold = 0.0

    for feat_idx in range(X.shape[1]):
        # Get unique thresholds (midpoints between sorted unique values)
        sorted_vals = np.unique(X[:, feat_idx])
        if len(sorted_vals) < 2:
            continue
        thresholds = (sorted_vals[:-1] + sorted_vals[1:]) / 2

        # Sample thresholds if too many (efficiency)
        if len(thresholds) > 50:
            rng = np.random.default_rng(42)
            thresholds = rng.choice(thresholds, 50, replace=False)

        for threshold in thresholds:
            # TODO: Create left_mask where X[:, feat_idx] <= threshold
            left_mask = ____  # Hint: X[:, feat_idx] <= threshold
            # TODO: Create right_mask as ~left_mask
            right_mask = ____  # Hint: ~left_mask

            n_left = left_mask.sum()
            n_right = right_mask.sum()
            if n_left == 0 or n_right == 0:
                continue

            gini_left = gini_impurity(y[left_mask])
            gini_right = gini_impurity(y[right_mask])
            # TODO: Compute weighted_gini = (n_left/n)*gini_left + (n_right/n)*gini_right
            weighted_gini = ____  # Hint: weighted sum of child impurities
            # TODO: Compute gain = parent_gini - weighted_gini
            gain = ____  # Hint: parent_gini - weighted_gini

            if gain > best_gain:
                best_gain = gain
                best_feature_idx = feat_idx
                best_threshold = threshold

    return best_feature_idx, best_threshold, best_gain


print("\n=== From-Scratch: Best Split Search ===")
print("Searching all features × all thresholds for maximum Gini gain...")
t0 = time.perf_counter()
best_feat_idx, best_thresh, best_gain = best_split_search(
    X_train, y_train, feature_names
)
search_time = time.perf_counter() - t0

best_feat_name = feature_names[best_feat_idx]
left_mask = X_train[:, best_feat_idx] <= best_thresh
right_mask = ~left_mask

print(f"\nBest split found in {search_time:.2f}s:")
print(f"  Feature: {best_feat_name}")
print(f"  Threshold: {best_thresh:.4f}")
print(f"  Gini gain: {best_gain:.4f}")
print(f"  Left child: n={left_mask.sum()}, churn rate={y_train[left_mask].mean():.4f}")
print(
    f"  Right child: n={right_mask.sum()}, churn rate={y_train[right_mask].mean():.4f}"
)

# Verify against sklearn's first split
dt_verify = DecisionTreeClassifier(max_depth=1, random_state=42)
dt_verify.fit(X_train, y_train)
sklearn_feat_idx = dt_verify.tree_.feature[0]
sklearn_thresh = dt_verify.tree_.threshold[0]
print(f"\nsklearn Decision Stump verification:")
print(f"  Feature: {feature_names[sklearn_feat_idx]} (index {sklearn_feat_idx})")
print(f"  Threshold: {sklearn_thresh:.4f}")
print(
    f"  Match: {'Yes' if sklearn_feat_idx == best_feat_idx else 'Close (sampling differences)'}"
)

# ── Checkpoint 6 ─────────────────────────────────────────────────────
assert best_gain > 0, "Best split should have positive Gini gain"
assert 0 < left_mask.sum() < len(y_train), "Split should divide data"
# INTERPRETATION: This exhaustive search is exactly what sklearn does
# internally.  For each candidate split, compute weighted child Gini
# and pick the split that reduces impurity the most.  This is a GREEDY
# algorithm — it may not find the globally optimal tree, but each split
# is locally optimal.
print("\n✓ Checkpoint 6 passed — from-scratch splitting verified\n")


# ════════════════════════════════════════════════════════════════════════
# TASK 8: Train Decision Tree with sklearn
# ════════════════════════════════════════════════════════════════════════

print("\n=== Decision Tree ===")

# Depth tuning
print(f"\n--- Decision Tree: max_depth sweep ---")
print(f"{'Depth':>8} {'CV Accuracy':>14} {'CV F1':>10}")
print("-" * 36)

depths = [2, 3, 5, 7, 10, 15, None]
dt_results = {}
for depth in depths:
    # TODO: Create DecisionTreeClassifier with max_depth=depth, random_state=42
    dt = DecisionTreeClassifier(max_depth=____, random_state=42)  # Hint: depth variable
    acc_scores = cross_val_score(dt, X_train, y_train, cv=cv, scoring="accuracy")
    f1_scores = cross_val_score(dt, X_train, y_train, cv=cv, scoring="f1")
    label = str(depth) if depth is not None else "None"
    dt_results[label] = {
        "depth": depth,
        "accuracy": acc_scores.mean(),
        "f1": f1_scores.mean(),
    }
    print(f"{label:>8} {acc_scores.mean():>14.4f} {f1_scores.mean():>10.4f}")

best_depth_key = max(dt_results, key=lambda d: dt_results[d]["f1"])
best_depth = dt_results[best_depth_key]["depth"]

# Train final Decision Tree
t0 = time.perf_counter()
dt_final = DecisionTreeClassifier(max_depth=best_depth, random_state=42)
dt_final.fit(X_train, y_train)
dt_train_time = time.perf_counter() - t0

dt_pred = dt_final.predict(X_test)
dt_prob = dt_final.predict_proba(X_test)[:, 1]

print(f"\nDecision Tree (depth={best_depth_key}): trained in {dt_train_time:.4f}s")
print(classification_report(y_test, dt_pred, target_names=["Retained", "Churned"]))

# Tree structure visualisation
print("--- Tree Structure (first 4 levels) ---")
tree_text = export_text(dt_final, feature_names=feature_names, max_depth=4)
print(tree_text[:1500])  # Limit output

# Feature importance
dt_importances = dict(zip(feature_names, dt_final.feature_importances_))
dt_importances_sorted = dict(
    sorted(dt_importances.items(), key=lambda x: x[1], reverse=True)
)

print("--- Decision Tree Feature Importance ---")
print(f"{'Feature':<30} {'Importance':>12}")
print("-" * 44)
for name, imp in list(dt_importances_sorted.items())[:10]:
    bar = "#" * int(imp * 50)
    print(f"{name:<30} {imp:>12.4f}  {bar}")

print(f"\n  Tree depth: {dt_final.get_depth()}, Leaves: {dt_final.get_n_leaves()}")

# ── Checkpoint 7 ─────────────────────────────────────────────────────
dt_acc = accuracy_score(y_test, dt_pred)
assert dt_acc > 0.5, f"Decision Tree accuracy {dt_acc:.4f} should beat random"
# INTERPRETATION: Gini impurity is what the tree optimises at each split.
# The tree is GREEDY: locally optimal splits, globally possibly suboptimal.
print("\n✓ Checkpoint 7 passed — Decision Tree trained with depth tuning\n")


# ════════════════════════════════════════════════════════════════════════
# TASK 9: Random Forest — bagging, feature importance, OOB
# ════════════════════════════════════════════════════════════════════════
# Random Forest = bagging + feature subsampling.
# P(sample i NOT in bootstrap) = (1 - 1/n)^n → 1/e ≈ 0.368 → OOB est.

print("\n=== Random Forest ===")

t0 = time.perf_counter()
# TODO: Create RandomForestClassifier with n_estimators=200, max_features="sqrt",
#       oob_score=True, random_state=42, n_jobs=-1
rf = RandomForestClassifier(
    n_estimators=____,  # Hint: 200
    max_features=____,  # Hint: "sqrt"
    oob_score=____,  # Hint: True — enables out-of-bag score
    random_state=42,
    n_jobs=-1,
)
rf.fit(X_train, y_train)
rf_train_time = time.perf_counter() - t0

rf_pred = rf.predict(X_test)
rf_prob = rf.predict_proba(X_test)[:, 1]

print(f"Random Forest (200 trees): trained in {rf_train_time:.2f}s")
print(f"OOB Score: {rf.oob_score_:.4f}")
print(classification_report(y_test, rf_pred, target_names=["Retained", "Churned"]))

# TODO: Compute CV accuracy and f1 scores for rf using cross_val_score
rf_cv_acc = cross_val_score(
    ____, ____, ____, cv=cv, scoring=____
)  # Hint: rf, X_train, y_train, "accuracy"
rf_cv_f1 = cross_val_score(
    ____, ____, ____, cv=cv, scoring=____
)  # Hint: rf, X_train, y_train, "f1"
print(f"CV Accuracy: {rf_cv_acc.mean():.4f} (±{rf_cv_acc.std():.4f})")
print(f"CV F1:       {rf_cv_f1.mean():.4f} (±{rf_cv_f1.std():.4f})")

# Feature importance
rf_importances = dict(zip(feature_names, rf.feature_importances_))
rf_importances_sorted = dict(
    sorted(rf_importances.items(), key=lambda x: x[1], reverse=True)
)

print(f"\n--- Random Forest Feature Importance ---")
print(f"{'Feature':<30} {'Importance':>12}")
print("-" * 44)
for name, imp in rf_importances_sorted.items():
    bar = "#" * int(imp * 50)
    print(f"{name:<30} {imp:>12.4f}  {bar}")

# OOB convergence
print(f"\n--- OOB Convergence vs Number of Trees ---")
oob_scores = []
n_trees_list = [10, 25, 50, 75, 100, 150, 200]
for n_trees in n_trees_list:
    rf_temp = RandomForestClassifier(
        n_estimators=n_trees,
        max_features="sqrt",
        oob_score=True,
        random_state=42,
        n_jobs=-1,
    )
    rf_temp.fit(X_train, y_train)
    oob_scores.append(rf_temp.oob_score_)

print(f"{'Trees':>8} {'OOB Score':>12}")
print("-" * 24)
for n, oob in zip(n_trees_list, oob_scores):
    print(f"{n:>8} {oob:>12.4f}")

print(
    f"\n  OOB ({rf.oob_score_:.4f}) ≈ CV accuracy ({rf_cv_acc.mean():.4f}) "
    f"— confirms consistency."
)

# ── Checkpoint 8 ─────────────────────────────────────────────────────
rf_acc = accuracy_score(y_test, rf_pred)
assert rf_acc > 0.5, f"RF accuracy {rf_acc:.4f} should beat random"
assert rf.oob_score_ > 0.5, "OOB should beat random"
assert abs(rf.oob_score_ - rf_cv_acc.mean()) < 0.10, "OOB and CV should be within 10pp"
# INTERPRETATION: OOB and CV being close is a sign of consistent
# generalisation.  The OOB mechanism is essentially cross-validation
# built into the training process — free quality control.
print("\n✓ Checkpoint 8 passed — Random Forest trained with OOB validation\n")


# ════════════════════════════════════════════════════════════════════════
# TASK 10: Decision boundary visualisation in 2D PCA space
# ════════════════════════════════════════════════════════════════════════
# Project features to 2D using PCA, then plot each model's decision
# boundary.  This gives visual intuition for how each algorithm
# partitions the feature space.

print("\n=== Decision Boundary Visualisation (2D PCA) ===")

# TODO: Create PCA with n_components=2, random_state=42
pca = PCA(n_components=____, random_state=42)  # Hint: 2
# TODO: fit_transform X_train → X_train_2d
X_train_2d = pca.fit_transform(____)  # Hint: X_train
# TODO: transform X_test → X_test_2d
X_test_2d = pca.transform(____)  # Hint: X_test

print(f"PCA variance explained: {pca.explained_variance_ratio_}")
print(f"Total variance captured: {pca.explained_variance_ratio_.sum():.2%}")

# Train models on 2D data for boundary visualisation
models_2d = {
    "SVM (RBF)": SVC(kernel="rbf", C=best_c_rbf, random_state=42),
    "KNN": KNeighborsClassifier(n_neighbors=best_k, metric=best_metric),
    "Naive Bayes": GaussianNB(),
    "Decision Tree": DecisionTreeClassifier(max_depth=best_depth, random_state=42),
    "Random Forest": RandomForestClassifier(
        n_estimators=100, random_state=42, n_jobs=-1
    ),
}

boundary_scores = {}
for name, model_2d in models_2d.items():
    model_2d.fit(X_train_2d, y_train)
    pred_2d = model_2d.predict(X_test_2d)
    acc_2d = accuracy_score(y_test, pred_2d)
    boundary_scores[name] = acc_2d
    print(f"  {name}: 2D accuracy = {acc_2d:.4f}")

print("\nNote: 2D accuracy is lower than full-dimensional because PCA")
print("discards information.  The visualisation shows boundary SHAPE, not")
print("true performance.  SVM: smooth; KNN: jagged; Tree: axis-aligned.")

# ── Checkpoint 9 ─────────────────────────────────────────────────────
assert len(boundary_scores) == 5, "Should have 5 models in boundary comparison"
# INTERPRETATION: SVM creates smooth boundaries (RBF kernel maps to
# infinite dimensions).  KNN creates jagged, instance-specific boundaries.
# Decision trees create axis-aligned rectangular partitions.  Random Forest
# smooths the tree boundaries by averaging many trees.  Naive Bayes creates
# linear (Gaussian) boundaries because of the independence assumption.
print("\n✓ Checkpoint 9 passed — decision boundaries computed\n")


# ════════════════════════════════════════════════════════════════════════
# TASK 11: Comprehensive model comparison
# ════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 76)
print("MODEL COMPARISON: All 5 Supervised Learning Families")
print("=" * 76)

all_models = {
    "SVM (RBF)": {
        "model": svm_final,
        "pred": svm_pred,
        "prob": svm_prob,
        "train_time": svm_train_time,
    },
    "KNN": {
        "model": knn_final,
        "pred": knn_pred,
        "prob": knn_prob,
        "train_time": knn_train_time,
    },
    "Naive Bayes": {
        "model": nb,
        "pred": nb_pred,
        "prob": nb_prob,
        "train_time": nb_train_time,
    },
    "Decision Tree": {
        "model": dt_final,
        "pred": dt_pred,
        "prob": dt_prob,
        "train_time": dt_train_time,
    },
    "Random Forest": {
        "model": rf,
        "pred": rf_pred,
        "prob": rf_prob,
        "train_time": rf_train_time,
    },
}

comparison_rows = []
for name, info in all_models.items():
    acc = accuracy_score(y_test, info["pred"])
    f1 = f1_score(y_test, info["pred"])
    auc = roc_auc_score(y_test, info["prob"])
    comparison_rows.append(
        {
            "Model": name,
            "Accuracy": acc,
            "F1": f1,
            "AUC-ROC": auc,
            "Train Time (s)": info["train_time"],
        }
    )

print(f"\n{'Model':<18} {'Accuracy':>10} {'F1':>10} {'AUC-ROC':>10} {'Time (s)':>10}")
print("-" * 62)
for row in comparison_rows:
    print(
        f"{row['Model']:<18} {row['Accuracy']:>10.4f} {row['F1']:>10.4f} "
        f"{row['AUC-ROC']:>10.4f} {row['Train Time (s)']:>10.4f}"
    )

ranked = sorted(comparison_rows, key=lambda r: r["F1"], reverse=True)
print(f"\nRanking by F1:")
for i, row in enumerate(ranked, 1):
    print(f"  {i}. {row['Model']} (F1={row['F1']:.4f})")

# ── Checkpoint 10 ────────────────────────────────────────────────────
all_accuracies = [svm_acc, knn_acc, nb_acc, dt_acc, rf_acc]
assert all(a > 0.5 for a in all_accuracies), "All models should beat random"
assert len(comparison_rows) == 5, "Should have 5 models"
print("\n✓ Checkpoint 10 passed — all 5 models compared\n")


# ════════════════════════════════════════════════════════════════════════
# TASK 11b: Visualise comparison
# ════════════════════════════════════════════════════════════════════════

viz = ModelVisualizer()

metric_dict = {
    row["Model"]: {
        "Accuracy": row["Accuracy"],
        "F1": row["F1"],
        "AUC-ROC": row["AUC-ROC"],
    }
    for row in comparison_rows
}
# TODO: Call viz.metric_comparison(metric_dict) to create comparison figure
fig_compare = viz.metric_comparison(____)  # Hint: metric_dict
fig_compare.update_layout(title="Model Zoo: Performance Comparison")
fig_compare.write_html("ex3_model_comparison.html")
print("Saved: ex3_model_comparison.html")

# TODO: Call viz.training_history with OOB scores dict and x_label="Number of Trees"
fig_oob = viz.training_history(
    ____, x_label=____
)  # Hint: {"OOB Score": oob_scores}, "Number of Trees"
fig_oob.update_layout(title="Random Forest: OOB Score vs Number of Trees")
fig_oob.write_html("ex3_oob_convergence.html")
print("Saved: ex3_oob_convergence.html")


# ════════════════════════════════════════════════════════════════════════
# TASK 12: When to Use Each Model
# ════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 76)
print("WHEN TO USE EACH MODEL — Decision Guide")
print("=" * 76)

print(
    """
+-------------------+---------------------+---------------------+---------------+
| Model             | Best When           | Avoid When          | Key Tradeoff  |
+-------------------+---------------------+---------------------+---------------+
| SVM               | Clear margin,       | Very large n        | High accuracy |
|                   | high-dimensional    | (O(n²) kernel)      | but slow      |
+-------------------+---------------------+---------------------+---------------+
| KNN               | Small n, non-linear | High dimensions     | No training,  |
|                   | boundaries          | (curse of dim)      | slow predict  |
+-------------------+---------------------+---------------------+---------------+
| Naive Bayes       | Text classification,| Complex feature     | Extremely     |
|                   | fast baseline       | interactions        | fast, strong  |
|                   |                     |                     | assumptions   |
+-------------------+---------------------+---------------------+---------------+
| Decision Tree     | Interpretability    | High variance,      | Interpretable |
|                   | required (audit)    | noisy data          | but unstable  |
+-------------------+---------------------+---------------------+---------------+
| Random Forest     | Robust default,     | Need interpretable  | Robust, but   |
|                   | mixed types         | model; memory       | black-box     |
+-------------------+---------------------+---------------------+---------------+
"""
)

print("--- Complexity vs Interpretability ---")
print("  Interpretable: NB > DTree > KNN > RF > SVM (RBF)")
print("  Accurate:      RF > SVM > KNN > DTree > NB (typically)")
print("  Fast training:  NB > DTree > KNN > RF > SVM")
print("  Fast predict:  NB > DTree > RF > SVM > KNN")

# ── Checkpoint 11 ────────────────────────────────────────────────────
best_by_f1 = max(comparison_rows, key=lambda r: r["F1"])
# INTERPRETATION: No model wins on ALL criteria simultaneously.
# Regulatory contexts (credit, healthcare) may require interpretable
# models even at the cost of some accuracy.
print("\n✓ Checkpoint 11 passed — model selection guide complete\n")


# ══════════════════════════════════════════════════════════════════════
# REFLECTION
# ══════════════════════════════════════════════════════════════════════
print("═" * 70)
print("  WHAT YOU'VE MASTERED")
print("═" * 70)
print(
    f"""
  ✓ SVM: margin maximisation, kernel trick, C parameter
  ✓ KNN: instance-based, distance metrics, k selection via CV
  ✓ Naive Bayes: Bayes theorem, independence assumption, speed
  ✓ Decision Tree: Gini by hand, depth control, interpretability
  ✓ Random Forest: bagging, OOB estimation, feature importance
  ✓ From-scratch Gini impurity and decision tree splitting
  ✓ Decision boundary visualisation in 2D PCA space
  ✓ Comprehensive comparison: accuracy, F1, AUC-ROC, training time

  BEST MODEL BY F1: {best_by_f1['Model']} (F1={best_by_f1['F1']:.4f})

  KEY INSIGHT: No model wins on all criteria simultaneously.
  Interpretability and accuracy often conflict.  Random Forest is
  the safest default for tabular data; SVM excels in high dimensions;
  Decision Trees are the ONLY fully interpretable model in this zoo.

  NEXT: Exercise 4 dives deep into gradient boosting — XGBoost, LightGBM,
  and CatBoost.  These algorithms typically outperform the model zoo on
  tabular data by building trees sequentially, each correcting the errors
  of the previous one.
"""
)

print("-" * 76)
print("Exercise 3 complete — the supervised model zoo with 5 algorithm families")
print("-" * 76)
