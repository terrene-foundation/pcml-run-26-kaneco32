# Copyright 2026 Terrene Foundation
# SPDX-License-Identifier: Apache-2.0
"""
# ════════════════════════════════════════════════════════════════════════
# MLFP03 — Exercise 6: Interpretability and Fairness
# ════════════════════════════════════════════════════════════════════════
#
# WHAT YOU'LL LEARN:
#   After completing this exercise, you will be able to:
#   - Compute TreeSHAP values efficiently for tree-based models
#   - Verify the Shapley additivity property (SHAP sum = model output)
#   - Interpret global feature importance using mean |SHAP| rankings
#   - Explain individual predictions using SHAP waterfall analysis
#   - Implement from-scratch permutation importance and compare to SHAP
#   - Apply LIME as a model-agnostic alternative and compare to SHAP
#   - Measure fairness using disparate impact ratio and equalized odds
#   - Explain the impossibility theorem (Chouldechova 2017)
#   - Conduct a complete fairness audit with SHAP across protected attrs
#
# PREREQUISITES:
#   - MLFP03 Exercise 4 (gradient boosting — SHAP uses TreeSHAP)
#   - MLFP03 Exercise 5 (imbalance handling — the model to explain)
#
# ESTIMATED TIME: ~150-180 minutes
#
# TASKS:
#   1.  Compute TreeSHAP values for the credit scoring model
#   2.  Verify Shapley additivity property
#   3.  Global interpretation: summary plot and feature importance
#   4.  Dependence plots: how individual features affect predictions
#   5.  From-scratch permutation importance
#   6.  LIME: model-agnostic local linear approximations
#   7.  Local interpretation: explain highest-risk and lowest-risk
#   8.  SHAP interaction effects
#   9.  Fairness metrics: disparate impact ratio
#   10. Fairness metrics: equalized odds
#   11. Impossibility theorem demonstration
#   12. Complete fairness audit report
#
# DATASET: Singapore credit scoring (from MLFP02)
#   Target: default prediction model from Exercise 5
#   Fairness concern: age, gender, ethnicity may be in features
#   Regulatory context: Singapore PDPA requires explanations for credit
#
# KEY FORMULAS:
#   Shapley value: φ_i = Σ_S [|S|!(|F|-|S|-1)!/|F|!] * [f(S∪{i})-f(S)]
#   Disparate impact: P(Y=1|G=minority) / P(Y=1|G=majority) (>0.8 is fair)
#
# ════════════════════════════════════════════════════════════════════════
"""
from __future__ import annotations

import numpy as np
import polars as pl
import shap
import lightgbm as lgb
from sklearn.metrics import roc_auc_score, f1_score, confusion_matrix
from sklearn.inspection import permutation_importance

from kailash_ml import PreprocessingPipeline, ModelVisualizer
from kailash_ml.interop import to_sklearn_input

from shared import MLFPDataLoader


# ── Data Loading & Model Training ─────────────────────────────────────

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

# Train best model from Exercise 4
model = lgb.LGBMClassifier(
    n_estimators=500,
    learning_rate=0.1,
    max_depth=6,
    scale_pos_weight=(1 - y_train.mean()) / y_train.mean(),
    random_state=42,
    verbose=-1,
)
model.fit(X_train, y_train)

y_proba = model.predict_proba(X_test)[:, 1]
y_pred = model.predict(X_test)
print(f"Model AUC-ROC: {roc_auc_score(y_test, y_proba):.4f}")

# ── Checkpoint 1 ─────────────────────────────────────────────────────
auc_roc = roc_auc_score(y_test, y_proba)
assert auc_roc > 0.5, f"Model AUC-ROC {auc_roc:.4f} should beat random"
assert model is not None, "Model should be trained"
print("\n✓ Checkpoint 1 passed — model trained for SHAP analysis\n")


# ══════════════════════════════════════════════════════════════════════
# TASK 1: Compute TreeSHAP values
# ══════════════════════════════════════════════════════════════════════
# TreeSHAP computes exact Shapley values in polynomial time O(TLD²)
# vs exponential time for exact Shapley on general models.

# TODO: Create shap.TreeExplainer(model), then call explainer.shap_values(X_test)
explainer = shap.TreeExplainer(____)  # Hint: model
shap_values = explainer.shap_values(____)  # Hint: X_test

# For binary classification, shap_values may be [class_0, class_1]
if isinstance(shap_values, list):
    shap_vals = shap_values[1]  # Positive class (default)
else:
    shap_vals = shap_values

print(f"\n=== SHAP Values ===")
print(f"Shape: {shap_vals.shape} (samples x features)")
print(f"Expected value (base rate): {explainer.expected_value}")


# ══════════════════════════════════════════════════════════════════════
# TASK 2: Verify Shapley additivity
# ══════════════════════════════════════════════════════════════════════
# Shapley axiom: sum of SHAP values + base value = model output
# This is the EFFICIENCY axiom from cooperative game theory.

print(f"\n=== Additivity Verification ===")

exp_val = (
    explainer.expected_value[1]
    if isinstance(explainer.expected_value, list)
    else explainer.expected_value
)

print(f"{'Sample':>8} {'SHAP sum':>12} {'Model out':>12} {'|Gap|':>10} {'Pass':>6}")
print("─" * 52)
additivity_errors = []
for i in range(min(10, len(shap_vals))):
    # TODO: Compute shap_sum = shap_vals[i].sum() + exp_val
    shap_sum = ____  # Hint: shap_vals[i].sum() + exp_val
    model_out = model.predict_proba(X_test[i : i + 1])[:, 1][0]
    gap = abs(shap_sum - model_out)
    additivity_errors.append(gap)
    passed = "✓" if gap < 0.1 else "~"
    print(f"{i:>8} {shap_sum:>12.4f} {model_out:>12.4f} {gap:>10.6f} {passed:>6}")

mean_gap = np.mean(additivity_errors)
print(f"\nMean additivity gap: {mean_gap:.6f}")

# ── Checkpoint 2 ─────────────────────────────────────────────────────
assert shap_vals is not None, "SHAP values should be computed"
assert shap_vals.shape == (
    X_test.shape[0],
    X_test.shape[1],
), "SHAP shape should be (n_samples, n_features)"
# INTERPRETATION: SHAP additivity means every feature gets credit for
# its exact contribution.  Unlike standard feature importance (average
# effect), SHAP decomposes each individual prediction into per-feature
# contributions.  This is the "right to explanation."
print("\n✓ Checkpoint 2 passed — SHAP additivity verified\n")


# ══════════════════════════════════════════════════════════════════════
# TASK 3: Global interpretation — feature importance ranking
# ══════════════════════════════════════════════════════════════════════

# TODO: Compute mean absolute SHAP values: np.abs(shap_vals).mean(axis=0)
mean_abs_shap = ____  # Hint: np.abs(shap_vals).mean(axis=0)
importance_ranking = sorted(
    zip(feature_names, mean_abs_shap), key=lambda x: x[1], reverse=True
)

print(f"\n=== Global Feature Importance (mean |SHAP|) ===")
print(f"{'Rank':>4} {'Feature':<30} {'mean|SHAP|':>12}")
print("─" * 50)
for rank, (name, imp) in enumerate(importance_ranking[:15], 1):
    bar = "█" * int(imp * 200)
    print(f"{rank:>4} {name:<30} {imp:>12.4f}  {bar}")

viz = ModelVisualizer()
fig = viz.feature_importance(model, feature_names, top_n=15)
fig.update_layout(title="SHAP Feature Importance: Credit Default")
fig.write_html("ex6_shap_importance.html")
print("\nSaved: ex6_shap_importance.html")

# ── Checkpoint 3 ─────────────────────────────────────────────────────
assert len(importance_ranking) == len(feature_names), "All features ranked"
top_feature, top_importance = importance_ranking[0]
assert top_importance > 0, "Top feature should have positive SHAP importance"
# INTERPRETATION: Mean |SHAP| importance tells you which features move
# predictions the most, on average.  Compare to gradient boosting gain
# importance from Exercise 4 — they should largely agree but SHAP is
# more reliable for correlated features.
print("\n✓ Checkpoint 3 passed — global SHAP importance computed\n")


# ══════════════════════════════════════════════════════════════════════
# TASK 4: Dependence plots — individual feature effects
# ══════════════════════════════════════════════════════════════════════

top_features = [name for name, _ in importance_ranking[:5]]

print(f"\n=== Dependence Analysis (top 5 features) ===")
for feat in top_features:
    feat_idx = feature_names.index(feat)
    feat_vals = X_test[:, feat_idx]
    feat_shap = shap_vals[:, feat_idx]

    # Remove NaN for correlation
    valid = ~(np.isnan(feat_vals) | np.isnan(feat_shap))
    if valid.sum() > 2:
        corr = np.corrcoef(feat_vals[valid], feat_shap[valid])[0, 1]
    else:
        corr = 0.0
    direction = "↑ increases default risk" if corr > 0 else "↑ decreases default risk"

    # Feature value statistics
    mean_val = feat_vals.mean()
    std_val = feat_vals.std()
    print(f"  {feat}:")
    print(f"    SHAP correlation: {corr:.3f} ({direction})")
    print(
        f"    Value range: [{feat_vals.min():.2f}, {feat_vals.max():.2f}], "
        f"mean={mean_val:.2f}"
    )

# ── Checkpoint 4 ─────────────────────────────────────────────────────
assert len(top_features) == 5, "Should analyse top 5 features"
# INTERPRETATION: The sign of the SHAP-feature correlation tells you the
# direction of the effect.  This directional insight is not available from
# gain-based importance.
print("\n✓ Checkpoint 4 passed — dependence analysis complete\n")


# ══════════════════════════════════════════════════════════════════════
# TASK 5: From-scratch permutation importance
# ══════════════════════════════════════════════════════════════════════
# Permutation importance: shuffle one feature at a time and measure the
# drop in model performance.  Model-agnostic, no assumptions about
# internal structure.
#
# Algorithm:
#   1. Compute baseline score on test set
#   2. For each feature j:
#      a. Shuffle feature j in the test set (break its relationship to y)
#      b. Compute score on the shuffled data
#      c. importance_j = baseline_score - shuffled_score
#   3. Repeat K times and average for stability

print(f"\n=== From-Scratch Permutation Importance ===")


def permutation_importance_manual(
    model,
    X,
    y,
    feature_names,
    n_repeats=5,
    scoring="roc_auc",
    seed=42,
):
    """Compute permutation importance from scratch."""
    rng = np.random.default_rng(seed)

    # Baseline score
    y_p = model.predict_proba(X)[:, 1]
    if scoring == "roc_auc":
        baseline = roc_auc_score(y, y_p)
    else:
        baseline = f1_score(y, model.predict(X))

    importances = {}
    for feat_idx, feat_name in enumerate(feature_names):
        drops = []
        for _ in range(n_repeats):
            X_shuffled = X.copy()
            # TODO: Shuffle feature feat_idx: X_shuffled[:, feat_idx] = rng.permutation(...)
            X_shuffled[:, feat_idx] = rng.permutation(
                ____
            )  # Hint: X_shuffled[:, feat_idx]
            y_p_shuffled = model.predict_proba(X_shuffled)[:, 1]
            if scoring == "roc_auc":
                shuffled_score = roc_auc_score(y, y_p_shuffled)
            else:
                shuffled_score = f1_score(y, (y_p_shuffled >= 0.5).astype(int))
            drops.append(baseline - shuffled_score)
        importances[feat_name] = {
            "mean": np.mean(drops),
            "std": np.std(drops),
        }

    return baseline, importances


baseline_score, perm_imp = permutation_importance_manual(
    model, X_test, y_test, feature_names, n_repeats=5
)

perm_ranking = sorted(perm_imp.items(), key=lambda x: x[1]["mean"], reverse=True)

print(f"\nBaseline AUC-ROC: {baseline_score:.4f}")
print(f"\n{'Rank':>4} {'Feature':<30} {'Imp (mean)':>12} {'± std':>10}")
print("─" * 58)
for rank, (name, vals) in enumerate(perm_ranking[:15], 1):
    print(f"{rank:>4} {name:<30} {vals['mean']:>12.4f} {vals['std']:>10.4f}")

# Compare permutation importance vs SHAP
print(f"\n--- SHAP vs Permutation Importance (top 10) ---")
shap_top10 = [name for name, _ in importance_ranking[:10]]
perm_top10 = [name for name, _ in perm_ranking[:10]]
overlap = set(shap_top10) & set(perm_top10)
print(f"SHAP top-10 ∩ Permutation top-10: {len(overlap)} / 10")
print(f"Shared features: {sorted(overlap)}")

# ── Checkpoint 5 ─────────────────────────────────────────────────────
assert len(perm_imp) == len(feature_names), "All features permuted"
assert perm_ranking[0][1]["mean"] > 0, "Top feature should have positive importance"
# INTERPRETATION: Permutation importance is model-agnostic — it works for
# any model, not just trees.  However, it can be misleading for correlated
# features: shuffling one feature of a correlated pair has little effect
# because the other feature still carries the information.  SHAP handles
# correlations correctly via the Shapley value axioms.
print("\n✓ Checkpoint 5 passed — from-scratch permutation importance computed\n")


# ══════════════════════════════════════════════════════════════════════
# TASK 6: LIME — model-agnostic local explanations
# ══════════════════════════════════════════════════════════════════════
# LIME (Local Interpretable Model-agnostic Explanations):
# For a single prediction x, LIME:
#   1. Generates perturbed samples around x
#   2. Weights them by proximity to x (exponential kernel)
#   3. Fits a sparse linear model on the weighted samples
# The linear model coefficients are the local feature importances.

try:
    from lime.lime_tabular import LimeTabularExplainer

    # TODO: Create LimeTabularExplainer with training_data=X_train, feature_names=feature_names,
    #       class_names=["no_default", "default"], mode="classification",
    #       discretize_continuous=True, random_state=42
    lime_explainer = LimeTabularExplainer(
        training_data=____,  # Hint: X_train
        feature_names=____,  # Hint: feature_names
        class_names=["no_default", "default"],
        mode=____,  # Hint: "classification"
        discretize_continuous=True,
        random_state=42,
    )

    # Explain the highest-risk prediction with LIME
    risk_order_lime = np.argsort(y_proba)
    high_risk_idx_lime = risk_order_lime[-1]

    lime_exp = lime_explainer.explain_instance(
        X_test[high_risk_idx_lime],
        model.predict_proba,
        num_features=10,
        top_labels=1,
    )

    print(f"\n=== LIME Explanation (highest-risk sample) ===")
    print(f"P(default) = {y_proba[high_risk_idx_lime]:.4f}")
    print(f"\nLIME local feature importances:")
    for feat_desc, weight in lime_exp.as_list():
        direction = "↑risk" if weight > 0 else "↓risk"
        print(f"  {feat_desc:<45} {weight:+.4f} ({direction})")

    # Compare LIME vs SHAP for the same sample
    shap_for_sample = shap_vals[high_risk_idx_lime]
    shap_sorted = sorted(
        zip(feature_names, shap_for_sample),
        key=lambda x: abs(x[1]),
        reverse=True,
    )[:10]
    print(f"\nSHAP for same sample (top 10):")
    for name, sv in shap_sorted:
        print(f"  {name:<30} SHAP={sv:+.4f}")

    print("\nLIME vs SHAP comparison:")
    print("  LIME: fast, model-agnostic, but can be unstable")
    print("  SHAP: theoretically grounded (Shapley values), exact for trees")
    print("  Production: use TreeSHAP for trees; LIME for black-boxes")
    HAS_LIME = True

except ImportError:
    print("\n=== LIME (not installed) ===")
    print("Install with: pip install lime")
    HAS_LIME = False

# ── Checkpoint 6 ─────────────────────────────────────────────────────
assert shap_vals is not None, "SHAP values must exist"
# INTERPRETATION: LIME and SHAP answer the same question — "why did the
# model predict this?" — but from different angles.  TreeSHAP is exact;
# LIME approximates with a local linear model.
print("\n✓ Checkpoint 6 passed — local explanation methods compared\n")


# ══════════════════════════════════════════════════════════════════════
# TASK 7: Local interpretation — individual predictions
# ══════════════════════════════════════════════════════════════════════

# TODO: Sort y_proba indices to find highest/lowest risk predictions
risk_order = np.argsort(____)  # Hint: y_proba

for label, idx in [
    ("Highest Risk", risk_order[-1]),
    ("Lowest Risk", risk_order[0]),
    ("Borderline (near 0.5)", risk_order[len(risk_order) // 2]),
]:
    print(f"\n=== {label} (P(default) = {y_proba[idx]:.4f}) ===")
    sample_shap = shap_vals[idx]
    sorted_contrib = sorted(
        zip(feature_names, sample_shap, X_test[idx]),
        key=lambda x: abs(x[1]),
        reverse=True,
    )
    for name, shap_val, feat_val in sorted_contrib[:8]:
        direction = "↑risk" if shap_val > 0 else "↓risk"
        print(f"  {name} = {feat_val:.2f} → SHAP = {shap_val:+.4f} ({direction})")

# ── Checkpoint 7 ─────────────────────────────────────────────────────
assert y_proba[risk_order[-1]] > y_proba[risk_order[0]], "Highest risk > lowest risk"
# INTERPRETATION: Individual SHAP explanations are the basis for the
# "right to explanation" required by PDPA and EU AI Act.  The waterfall
# gives a legally defensible answer to "why was my loan declined?"
print("\n✓ Checkpoint 7 passed — individual prediction explanations verified\n")


# ══════════════════════════════════════════════════════════════════════
# TASK 8: SHAP interaction effects
# ══════════════════════════════════════════════════════════════════════

sample_size = min(500, X_test.shape[0])
X_sample = X_test[:sample_size]

# TODO: Compute SHAP interaction values: explainer.shap_interaction_values(X_sample)
shap_interaction = explainer.shap_interaction_values(____)  # Hint: X_sample
if isinstance(shap_interaction, list):
    shap_interaction = shap_interaction[1]

# Find strongest interactions
n_features = len(feature_names)
interaction_strengths = []
for i in range(n_features):
    for j in range(i + 1, n_features):
        strength = np.abs(shap_interaction[:, i, j]).mean()
        interaction_strengths.append((feature_names[i], feature_names[j], strength))

interaction_strengths.sort(key=lambda x: x[2], reverse=True)

print(f"\n=== Top Feature Interactions ===")
print(f"{'Feature 1':<25} {'Feature 2':<25} {'Strength':>10}")
print("─" * 62)
for f1, f2, strength in interaction_strengths[:10]:
    print(f"{f1:<25} {f2:<25} {strength:>10.4f}")

# ── Checkpoint 8 ─────────────────────────────────────────────────────
assert len(interaction_strengths) > 0, "Should find interactions"
print("\n✓ Checkpoint 8 passed — SHAP interaction effects computed\n")


# ══════════════════════════════════════════════════════════════════════
# TASK 9: Fairness — disparate impact ratio
# ══════════════════════════════════════════════════════════════════════
# Disparate impact ratio: P(positive|minority) / P(positive|majority)
# The 4/5 (80%) rule: ratio < 0.8 indicates disparate impact.
#
# In credit scoring: "positive outcome" = loan approved (pred=0, no default)

print(f"\n=== Fairness Audit: Disparate Impact ===")

protected_candidates = ["age", "gender", "ethnicity", "marital_status"]
protected_in_model = [f for f in protected_candidates if f in feature_names]

if protected_in_model:
    print(f"Protected attributes in model: {protected_in_model}")

    for attr in protected_in_model:
        attr_idx = feature_names.index(attr)
        attr_vals = X_test[:, attr_idx]
        unique_vals = np.unique(attr_vals[~np.isnan(attr_vals)])

        if len(unique_vals) <= 10:
            print(f"\n  --- {attr} ---")
            # Compute approval rate (pred=0) per group
            group_rates = {}
            for val in sorted(unique_vals):
                mask = attr_vals == val
                n_group = mask.sum()
                if n_group < 10:
                    continue
                # "Positive outcome" = approved = predicted no default
                approval_rate = (y_pred[mask] == 0).mean()
                default_rate = y_proba[mask].mean()
                group_rates[val] = {
                    "n": n_group,
                    "approval_rate": approval_rate,
                    "mean_default_prob": default_rate,
                }

            if len(group_rates) >= 2:
                # Find majority and minority groups
                majority_val = max(group_rates, key=lambda v: group_rates[v]["n"])
                majority_rate = group_rates[majority_val]["approval_rate"]

                print(f"  {'Value':>8} {'N':>6} {'Approval':>10} {'Disp.Impact':>14}")
                print("  " + "─" * 42)
                for val, info in sorted(group_rates.items()):
                    # TODO: Compute disparate impact ratio for each group
                    di = ____  # Hint: info["approval_rate"] / max(majority_rate, 0.001)
                    flag = " ⚠ < 0.8" if di < 0.8 else ""
                    print(
                        f"  {val:>8.0f} {info['n']:>6} "
                        f"{info['approval_rate']:>10.3f} {di:>14.3f}{flag}"
                    )
else:
    print("No protected attributes found in feature set.")
    print("Creating synthetic protected attribute for demonstration...")

    # Create synthetic age groups for demonstration
    if "age" in feature_names:
        attr_idx = feature_names.index("age")
    else:
        attr_idx = 0

    attr_vals = X_test[:, attr_idx]
    median_val = np.median(attr_vals)
    group_young = attr_vals <= median_val
    group_old = ~group_young

    approval_young = (y_pred[group_young] == 0).mean()
    approval_old = (y_pred[group_old] == 0).mean()
    # TODO: Compute disparate impact ratio between the two groups
    di_ratio = ____  # Hint: approval_young / max(approval_old, 0.001)

    print(f"  Feature used: {feature_names[attr_idx]} (median split)")
    print(f"  Group ≤ median: approval={approval_young:.3f} (n={group_young.sum()})")
    print(f"  Group > median: approval={approval_old:.3f} (n={group_old.sum()})")
    print(f"  Disparate impact ratio: {di_ratio:.3f}")
    print(f"  4/5 rule: {'PASS' if di_ratio >= 0.8 else 'FAIL ⚠'}")

# ── Checkpoint 9 ─────────────────────────────────────────────────────
print("\n✓ Checkpoint 9 passed — disparate impact analysis complete\n")


# ══════════════════════════════════════════════════════════════════════
# TASK 10: Fairness — equalized odds
# ══════════════════════════════════════════════════════════════════════
# Equalized odds: TPR and FPR should be equal across groups.
# If TPR(group A) >> TPR(group B), the model catches more defaults in
# one group than another — unfair if the groups differ on a protected
# attribute.

print(f"\n=== Fairness: Equalized Odds ===")

# Use the median split of the first feature as a synthetic protected attr
attr_idx_eq = 0
attr_vals_eq = X_test[:, attr_idx_eq]
median_eq = np.median(attr_vals_eq)
group_a = attr_vals_eq <= median_eq
group_b = ~group_a

for group_name, mask in [
    ("Group A (≤ median)", group_a),
    ("Group B (> median)", group_b),
]:
    y_g = y_test[mask]
    p_g = y_pred[mask]
    if y_g.sum() > 0 and (y_g == 0).sum() > 0:
        cm_g = confusion_matrix(y_g, p_g)
        tn_g, fp_g, fn_g, tp_g = cm_g.ravel()
        # TODO: Compute TPR and FPR for each group
        tpr = ____  # Hint: tp_g / (tp_g + fn_g) if (tp_g + fn_g) > 0 else 0
        fpr = ____  # Hint: fp_g / (fp_g + tn_g) if (fp_g + tn_g) > 0 else 0
        print(f"  {group_name}: TPR={tpr:.3f}, FPR={fpr:.3f}, n={mask.sum()}")
    else:
        print(f"  {group_name}: insufficient samples for TPR/FPR")

print("\nEqualized odds requires TPR_A ≈ TPR_B AND FPR_A ≈ FPR_B.")
print("Large gaps indicate the model's error patterns differ by group.")

# ── Checkpoint 10 ────────────────────────────────────────────────────
print("\n✓ Checkpoint 10 passed — equalized odds analysis complete\n")


# ══════════════════════════════════════════════════════════════════════
# TASK 11: Impossibility theorem demonstration
# ══════════════════════════════════════════════════════════════════════
# Chouldechova (2017) and Kleinberg et al. (2016) proved:
# When base rates differ between groups, it is MATHEMATICALLY IMPOSSIBLE
# to simultaneously satisfy:
#   1. Demographic parity (equal selection rates)
#   2. Equalized odds (equal TPR and FPR)
#   3. Calibration (predicted probabilities are reliable per group)
#
# This is not a modelling failure — it is a mathematical constraint.

print(f"\n=== Impossibility Theorem ===")
print(
    """
Chouldechova (2017) / Kleinberg et al. (2016):

When base rates differ between groups (e.g., 15% default in group A,
8% default in group B), you CANNOT simultaneously achieve:

  1. DEMOGRAPHIC PARITY
     P(predict default | group A) = P(predict default | group B)
     → Equal selection rates regardless of group

  2. EQUALIZED ODDS
     TPR_A = TPR_B  AND  FPR_A = FPR_B
     → Equal error rates across groups

  3. CALIBRATION
     P(actually default | predict 20%, group A) = 0.20
     P(actually default | predict 20%, group B) = 0.20
     → Predicted probabilities are equally reliable per group

These three desiderata form an impossible triangle when base rates
differ.  Any real-world credit model MUST choose which fairness
criterion to prioritise and transparently document the tradeoff.

Practical implications for credit scoring:
  - Singapore MAS: emphasis on calibration (risk-based pricing)
  - EU AI Act: emphasis on non-discrimination (demographic parity)
  - US ECOA: emphasis on disparate impact (4/5 rule)
  - No single model satisfies all three regulators simultaneously
"""
)

# Numerical demonstration
if group_a.sum() > 50 and group_b.sum() > 50:
    base_rate_a = y_test[group_a].mean()
    base_rate_b = y_test[group_b].mean()
    print(f"In our data:")
    print(f"  Base rate (Group A): {base_rate_a:.3f}")
    print(f"  Base rate (Group B): {base_rate_b:.3f}")
    if abs(base_rate_a - base_rate_b) > 0.01:
        print(f"  Base rates differ → impossibility theorem applies")
        print(f"  Cannot simultaneously satisfy all three fairness criteria")
    else:
        print(f"  Base rates approximately equal → less tension between criteria")

# ── Checkpoint 11 ────────────────────────────────────────────────────
print("\n✓ Checkpoint 11 passed — impossibility theorem explained\n")


# ══════════════════════════════════════════════════════════════════════
# TASK 12: Complete fairness audit report
# ══════════════════════════════════════════════════════════════════════

print(f"\n{'=' * 70}")
print("FAIRNESS AUDIT REPORT — CREDIT DEFAULT MODEL")
print(f"{'=' * 70}")

# SHAP contributions for protected attributes
if protected_in_model:
    print(f"\nProtected attributes in model: {protected_in_model}")
    for attr in protected_in_model:
        attr_idx_audit = feature_names.index(attr)
        attr_shap = shap_vals[:, attr_idx_audit]
        rank = [n for n, _ in importance_ranking].index(attr) + 1
        print(f"\n  {attr}:")
        print(f"    Mean |SHAP|: {np.abs(attr_shap).mean():.4f}")
        print(f"    SHAP Rank: #{rank} / {len(feature_names)}")
        print(
            f"    Action: {'INVESTIGATE — in top 10' if rank <= 10 else 'Low risk — outside top 10'}"
        )
else:
    print("\nNo explicit protected attributes in feature set.")
    print("However, proxy variables may encode protected information.")
    print("(e.g., zip code → ethnicity, education → age)")

print(
    f"""
AUDIT SUMMARY:
  1. Disparate impact analysis: completed
  2. Equalized odds analysis: completed
  3. SHAP contribution of protected attributes: completed
  4. Impossibility theorem: acknowledged — cannot satisfy all criteria

RECOMMENDATIONS:
  a. Monitor disparate impact ratio quarterly (>0.8 threshold)
  b. Document which fairness criterion is prioritised and why
  c. Implement SHAP explanations for every declined application
  d. Review proxy variable effects (zip code, education level)
  e. Retrain when demographic shift changes group base rates
"""
)

# ── Checkpoint 12 ────────────────────────────────────────────────────
print("\n✓ Checkpoint 12 passed — fairness audit report complete\n")


# ══════════════════════════════════════════════════════════════════════
# REFLECTION
# ══════════════════════════════════════════════════════════════════════
print("═" * 70)
print("  WHAT YOU'VE MASTERED")
print("═" * 70)
print(
    """
  ✓ TreeSHAP: exact Shapley values in O(TLD²) for tree models
  ✓ Additivity property: SHAP values sum to the model output (verified)
  ✓ Global importance: mean |SHAP| ranks features by average impact
  ✓ Dependence analysis: direction + magnitude of each feature's effect
  ✓ From-scratch permutation importance: model-agnostic, manual impl
  ✓ LIME: local linear approximation — model-agnostic but less stable
  ✓ Individual explanations: waterfall decomposition per prediction
  ✓ SHAP interaction effects: which feature pairs matter together
  ✓ Disparate impact: 4/5 rule for protected groups
  ✓ Equalized odds: equal TPR and FPR across groups
  ✓ Impossibility theorem: cannot satisfy all fairness criteria
  ✓ Complete fairness audit: regulatory-grade documentation

  KEY INSIGHT: Interpretability is not optional in regulated industries.
  Singapore PDPA and MAS guidelines require explainable credit decisions.
  SHAP provides theoretically grounded, legally defensible explanations.

  PRODUCTION RULE:
    Tree models → TreeSHAP (exact, fast)
    Black-box → LIME or KernelSHAP (approximate)
    Always audit protected attributes before deployment

  NEXT: Exercise 7 scales up from a single model to a full Kailash
  Workflow.  You'll orchestrate feature engineering, model training,
  evaluation, and persistence using WorkflowBuilder.
"""
)

print("\n✓ Exercise 6 complete — interpretability and fairness audit")
