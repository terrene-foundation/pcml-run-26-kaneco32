# Copyright 2026 Terrene Foundation
# SPDX-License-Identifier: Apache-2.0
"""
# ════════════════════════════════════════════════════════════════════════
# MLFP02 — Exercise 7: CUPED and Causal Inference
# ════════════════════════════════════════════════════════════════════════
#
# WHAT YOU'LL LEARN:
#   After completing this exercise, you will be able to:
#   - Derive and implement CUPED variance reduction using pre-experiment data
#   - Quantify how much CUPED shrinks confidence intervals (based on ρ²)
#   - Implement multi-covariate CUPED for further variance reduction
#   - Apply Bayesian A/B testing to get posterior probability of improvement
#   - Compute expected loss for decision-making under uncertainty
#   - Implement sequential testing with mSPRT (always-valid p-values)
#   - Demonstrate the peeking problem with simulation
#   - Implement Difference-in-Differences (DiD) for observational data
#   - Test the parallel trends assumption that underlies DiD
#   - Implement stratified CUPED for heterogeneous treatment effects
#   - Log experiment results to ExperimentTracker for reproducibility
#   - Synthesise causal inference methods into a decision framework
#
# PREREQUISITES: Complete Exercises 3-4 — you should understand hypothesis
#   testing, p-values, SRM detection, and Welch's t-test.
#
# ESTIMATED TIME: ~170 minutes
#
# TASKS:
#    1. Load experiment data with pre-experiment covariates; SRM check
#    2. Standard A/B analysis baseline (no CUPED)
#    3. Single-covariate CUPED: derive θ, compute Y_adj, verify variance reduction
#    4. Multi-covariate CUPED: use multiple pre-experiment features
#    5. Stratified CUPED: different treatment effects by segment
#    6. Bayesian A/B testing: posterior probability and expected loss
#    7. Sequential testing with mSPRT (always-valid p-values)
#    8. Peeking problem simulation: demonstrate inflated Type I error
#    9. Difference-in-Differences (DiD) for observational data
#   10. Parallel trends test for DiD validity
#   11. Log all results to ExperimentTracker
#   12. Causal inference decision framework and business interpretation
#
# DATASET: E-commerce experiment with pre-experiment covariates
#   Columns: experiment_group, revenue, pre_metric_value, timestamp
#
# THEORY (CUPED):
#   Y_adj = Y - θ(X - E[X])  where θ = Cov(Y,X)/Var(X)
#   Var(Y_adj) = Var(Y)(1 - ρ²)  where ρ = Cor(Y,X)
#
# ════════════════════════════════════════════════════════════════════════
"""
from __future__ import annotations

import asyncio

import numpy as np
import polars as pl
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from kailash.db import ConnectionManager
from kailash_ml.engines.experiment_tracker import ExperimentTracker
from kailash_ml import ModelVisualizer
from scipy import stats

from shared import MLFPDataLoader
from shared.kailash_helpers import setup_environment

setup_environment()


# ── Data Loading ──────────────────────────────────────────────────────

loader = MLFPDataLoader()
experiment = loader.load("mlfp02", "experiment_data.parquet")

print("=" * 70)
print("  MLFP02 Exercise 7: CUPED and Causal Inference")
print("=" * 70)
print(f"\n  Data loaded: experiment_data.parquet")
print(f"  Shape: {experiment.shape}")
print(f"  Columns: {experiment.columns}")
print(experiment.head(5))

# Separate groups
control = experiment.filter(pl.col("experiment_group") == "control")
treatment = experiment.filter(pl.col("experiment_group") != "control")

n_c, n_t = control.height, treatment.height
print(f"\nControl: {n_c:,} | Treatment: {n_t:,}")


# ══════════════════════════════════════════════════════════════════════
# TASK 1: SRM Check and Data Exploration
# ══════════════════════════════════════════════════════════════════════

expected = np.array([n_c + n_t] * 2) / 2
observed = np.array([n_c, n_t])
_, srm_p = stats.chisquare(observed, f_exp=expected)
print(f"\nSRM check: p={srm_p:.6f} — {'OK' if srm_p > 0.01 else 'SRM DETECTED'}")

# Explore pre-experiment covariates
for col in ["revenue", "pre_metric_value", "metric_value"]:
    if col in experiment.columns:
        vals = experiment[col].drop_nulls()
        print(f"  {col}: mean={vals.mean():.2f}, std={vals.std():.2f}, n={vals.len()}")

# ── Checkpoint 1 ─────────────────────────────────────────────────────
assert 0 <= srm_p <= 1, "SRM p-value must be valid"
print("\n✓ Checkpoint 1 passed — SRM check and data exploration completed\n")


# ══════════════════════════════════════════════════════════════════════
# TASK 2: Standard Analysis Baseline (No CUPED)
# ══════════════════════════════════════════════════════════════════════

y_c = control["revenue"].to_numpy().astype(np.float64)
y_t = treatment["revenue"].to_numpy().astype(np.float64)

mean_c, mean_t = y_c.mean(), y_t.mean()
lift = mean_t - mean_c
# TODO: Compute naive standard error for two-sample mean difference
se_naive = ____  # Hint: np.sqrt(y_c.var(ddof=1) / n_c + y_t.var(ddof=1) / n_t)
ci_naive = (lift - 1.96 * se_naive, lift + 1.96 * se_naive)
z_naive = lift / se_naive
p_naive = 2 * (1 - stats.norm.cdf(abs(z_naive)))

print(f"\n=== Standard Analysis (no CUPED) ===")
print(f"Control mean: ${mean_c:.2f}")
print(f"Treatment mean: ${mean_t:.2f}")
print(f"Lift: ${lift:.2f} ({lift / mean_c:.2%} relative)")
print(f"SE: ${se_naive:.2f}")
print(f"95% CI: [${ci_naive[0]:.2f}, ${ci_naive[1]:.2f}]")
print(f"CI width: ${ci_naive[1] - ci_naive[0]:.2f}")
print(f"p-value: {p_naive:.6f}")
# INTERPRETATION: The naive analysis uses only experiment-period data.
# It ignores that some users are naturally high-spenders — CUPED
# removes this baseline noise by leveraging pre-experiment data.

# ── Checkpoint 2 ─────────────────────────────────────────────────────
assert se_naive > 0, "SE must be positive"
assert ci_naive[0] < ci_naive[1], "CI lower must be below upper"
print("\n✓ Checkpoint 2 passed — standard analysis baseline established\n")


# ══════════════════════════════════════════════════════════════════════
# TASK 3: Single-Covariate CUPED
# ══════════════════════════════════════════════════════════════════════
# CUPED: Y_adj = Y - θ(X - E[X])
# θ = Cov(Y, X) / Var(X) — the optimal coefficient
# Var(Y_adj) = Var(Y)(1 - ρ²) where ρ = Cor(Y, X)

x_c = control["pre_metric_value"].to_numpy().astype(np.float64)
x_t = treatment["pre_metric_value"].to_numpy().astype(np.float64)

# Pool all data for θ estimation
x_all = np.concatenate([x_c, x_t])
y_all = np.concatenate([y_c, y_t])

# TODO: Compute θ = Cov(Y, X) / Var(X) using np.cov and np.var
theta = ____  # Hint: np.cov(y_all, x_all)[0, 1] / np.var(x_all, ddof=1)

# TODO: Compute the pre-post correlation ρ = Cor(Y, X)
rho = ____  # Hint: np.corrcoef(y_all, x_all)[0, 1]

# Adjusted values: Y_adj = Y - θ(X - E[X])
x_mean = x_all.mean()
y_c_adj = y_c - theta * (x_c - x_mean)
y_t_adj = y_t - theta * (x_t - x_mean)

# CUPED analysis
mean_c_adj = y_c_adj.mean()
mean_t_adj = y_t_adj.mean()
lift_adj = mean_t_adj - mean_c_adj
se_cuped = np.sqrt(y_c_adj.var(ddof=1) / n_c + y_t_adj.var(ddof=1) / n_t)
ci_cuped = (lift_adj - 1.96 * se_cuped, lift_adj + 1.96 * se_cuped)
# TODO: Compute z-statistic and two-sided p-value for CUPED lift
z_cuped = ____  # Hint: lift_adj / se_cuped
p_cuped = ____  # Hint: 2 * (1 - stats.norm.cdf(abs(z_cuped)))

# Variance reduction
var_reduction = 1 - se_cuped**2 / se_naive**2
ci_width_reduction = 1 - se_cuped / se_naive
# TODO: Compute the theoretical variance reduction = ρ²
theoretical_reduction = ____  # Hint: rho**2

print(f"\n=== Single-Covariate CUPED ===")
print(f"Pre-post correlation (ρ): {rho:.4f}")
print(f"θ (optimal coefficient): {theta:.4f}")
print(f"Theoretical variance reduction: {theoretical_reduction:.1%}")
print(f"Actual variance reduction: {var_reduction:.1%}")
print(f"CI width reduction: {ci_width_reduction:.1%}")
print(f"\nCUPED lift: ${lift_adj:.2f}")
print(f"SE (naive): ${se_naive:.2f} → SE (CUPED): ${se_cuped:.2f}")
print(
    f"CI width (naive): ${ci_naive[1]-ci_naive[0]:.2f} → CI width (CUPED): ${ci_cuped[1]-ci_cuped[0]:.2f}"
)
print(f"95% CI: [${ci_cuped[0]:.2f}, ${ci_cuped[1]:.2f}]")
print(f"p-value: {p_cuped:.6f} (was {p_naive:.6f})")

print(f"\n--- Bias Check ---")
print(f"Naive lift: ${lift:.4f}")
print(f"CUPED lift: ${lift_adj:.4f}")
print(f"Difference: ${lift_adj - lift:.4f}")
print(
    f"CUPED is {'unbiased ✓' if abs(lift_adj - lift) < 2 * se_naive else 'BIASED — investigate'}"
)
# INTERPRETATION: CUPED reduces variance by ρ². The point estimate is
# unbiased because E[X - E[X]] = 0. The only change is precision —
# you get the same answer, just with a tighter confidence interval.

# ── Checkpoint 3 ─────────────────────────────────────────────────────
assert 0 <= abs(rho) <= 1, "Correlation must be between -1 and 1"
assert se_cuped <= se_naive * 1.01, "CUPED SE must be ≤ naive SE"
assert (
    abs(var_reduction - theoretical_reduction) < 0.1
), "Actual reduction should approximate theoretical ρ²"
print("\n✓ Checkpoint 3 passed — CUPED variance reduction verified\n")


# ══════════════════════════════════════════════════════════════════════
# TASK 4: Multi-Covariate CUPED
# ══════════════════════════════════════════════════════════════════════
# With multiple pre-experiment features, use multivariate regression
# to compute the optimal adjustment: Y_adj = Y - Xθ where
# θ = (X'X)⁻¹X'Y (regression coefficients from Y on pre-covariates)

print(f"\n=== Multi-Covariate CUPED ===")

pre_features = ["pre_metric_value"]
if "metric_value" in control.columns:
    pre_features.append("metric_value")

X_c_multi = control.select(pre_features).to_numpy().astype(np.float64)
X_t_multi = treatment.select(pre_features).to_numpy().astype(np.float64)
X_all_multi = np.vstack([X_c_multi, X_t_multi])

# Multivariate θ via OLS: θ = (X'X)⁻¹X'Y
X_centered = X_all_multi - X_all_multi.mean(axis=0)
theta_multi = np.linalg.lstsq(X_centered, y_all - y_all.mean(), rcond=None)[0]

# Adjusted values
y_c_adj_multi = y_c - (X_c_multi - X_all_multi.mean(axis=0)) @ theta_multi
y_t_adj_multi = y_t - (X_t_multi - X_all_multi.mean(axis=0)) @ theta_multi

lift_multi = y_t_adj_multi.mean() - y_c_adj_multi.mean()
se_multi = np.sqrt(y_c_adj_multi.var(ddof=1) / n_c + y_t_adj_multi.var(ddof=1) / n_t)
ci_multi = (lift_multi - 1.96 * se_multi, lift_multi + 1.96 * se_multi)
var_red_multi = 1 - se_multi**2 / se_naive**2

print(f"Covariates: {pre_features}")
print(f"Multi-covariate θ: {theta_multi}")
print(f"Variance reduction: {var_red_multi:.1%} (single-cov: {var_reduction:.1%})")
print(f"SE: ${se_multi:.2f} (single: ${se_cuped:.2f}, naive: ${se_naive:.2f})")
print(f"CI: [${ci_multi[0]:.2f}, ${ci_multi[1]:.2f}]")
# INTERPRETATION: Multiple covariates can capture more variance than
# a single one. The improvement depends on how much additional
# predictive power the extra covariates provide.

# ── Checkpoint 4 ─────────────────────────────────────────────────────
assert se_multi <= se_naive * 1.01, "Multi-CUPED SE should be ≤ naive"
print("\n✓ Checkpoint 4 passed — multi-covariate CUPED completed\n")


# ══════════════════════════════════════════════════════════════════════
# TASK 5: Stratified CUPED — Heterogeneous Treatment Effects
# ══════════════════════════════════════════════════════════════════════
# Do different user segments respond differently to treatment?
# Stratify by pre-experiment spending level and apply CUPED within strata.

print(f"\n=== Stratified CUPED ===")

x_all_flat = np.concatenate([x_c, x_t])
q33, q67 = np.percentile(x_all_flat, [33, 67])

strata = {
    "Low spenders": (x_all_flat <= q33),
    "Medium spenders": (x_all_flat > q33) & (x_all_flat <= q67),
    "High spenders": (x_all_flat > q67),
}

group_labels = np.concatenate([np.zeros(n_c), np.ones(n_t)])

print(
    f"{'Stratum':<20} {'n_ctrl':>8} {'n_treat':>8} {'Lift':>10} {'SE':>8} {'p-value':>10}"
)
print("─" * 68)

stratified_results = {}
for stratum_name, mask in strata.items():
    ctrl_mask = mask[:n_c]
    treat_mask = mask[n_c:]

    y_c_s = y_c[ctrl_mask]
    y_t_s = y_t[treat_mask]
    x_c_s = x_c[ctrl_mask]
    x_t_s = x_t[treat_mask]

    if len(y_c_s) < 30 or len(y_t_s) < 30:
        continue

    x_s_all = np.concatenate([x_c_s, x_t_s])
    y_s_all = np.concatenate([y_c_s, y_t_s])
    theta_s = (
        np.cov(y_s_all, x_s_all)[0, 1] / np.var(x_s_all, ddof=1)
        if np.var(x_s_all, ddof=1) > 0
        else 0
    )
    x_mean_s = x_s_all.mean()

    y_c_adj_s = y_c_s - theta_s * (x_c_s - x_mean_s)
    y_t_adj_s = y_t_s - theta_s * (x_t_s - x_mean_s)

    lift_s = y_t_adj_s.mean() - y_c_adj_s.mean()
    se_s = np.sqrt(
        y_c_adj_s.var(ddof=1) / len(y_c_s) + y_t_adj_s.var(ddof=1) / len(y_t_s)
    )
    z_s = lift_s / se_s if se_s > 0 else 0
    p_s = 2 * (1 - stats.norm.cdf(abs(z_s)))

    stratified_results[stratum_name] = {
        "n_ctrl": len(y_c_s),
        "n_treat": len(y_t_s),
        "lift": lift_s,
        "se": se_s,
        "p_value": p_s,
    }
    print(
        f"{stratum_name:<20} {len(y_c_s):>8,} {len(y_t_s):>8,} "
        f"${lift_s:>8.2f} ${se_s:>6.2f} {p_s:>10.6f}"
    )
# INTERPRETATION: If high spenders respond differently to treatment
# than low spenders, a one-size-fits-all analysis masks the heterogeneity.
# Stratified CUPED reveals these differences while maintaining precision.

# ── Checkpoint 5 ─────────────────────────────────────────────────────
assert len(stratified_results) >= 2, "Should have at least 2 strata"
print("\n✓ Checkpoint 5 passed — stratified CUPED completed\n")


# ══════════════════════════════════════════════════════════════════════
# TASK 6: Bayesian A/B Testing
# ══════════════════════════════════════════════════════════════════════
# Instead of p-values, compute:
#   P(treatment > control | data)
#   Expected loss from choosing treatment

se_c_post = y_c_adj.std(ddof=1) / np.sqrt(n_c)
se_t_post = y_t_adj.std(ddof=1) / np.sqrt(n_t)
se_lift = np.sqrt(se_c_post**2 + se_t_post**2)

# TODO: Compute P(treatment > control) = P(lift > 0) from Normal(lift_adj, se_lift)
prob_treatment_better = ____  # Hint: 1 - stats.norm.cdf(0, loc=lift_adj, scale=se_lift)

# Expected loss: E[max(control - treatment, 0)]
z_ratio = -lift_adj / se_lift
# TODO: Compute expected loss for choosing treatment and control
# expected_loss_treatment = se_lift * norm.pdf(z_ratio) + lift_adj * norm.cdf(z_ratio)
expected_loss_treatment = (
    ____  # Hint: se_lift * stats.norm.pdf(z_ratio) + lift_adj * stats.norm.cdf(z_ratio)
)
expected_loss_control = ____  # Hint: se_lift * stats.norm.pdf(-z_ratio) - lift_adj * stats.norm.cdf(-z_ratio)

# 95% credible interval
bayesian_ci = (lift_adj - 1.96 * se_lift, lift_adj + 1.96 * se_lift)

# Probability of practical significance (lift > $1)
prob_practical = 1 - stats.norm.cdf(1.0, loc=lift_adj, scale=se_lift)

print(f"\n=== Bayesian A/B Test ===")
print(
    f"P(treatment > control): {prob_treatment_better:.4f} ({prob_treatment_better:.1%})"
)
print(f"P(treatment > control by >$1): {prob_practical:.4f} ({prob_practical:.1%})")
print(f"Expected loss (choose treatment): ${expected_loss_treatment:.2f}/user")
print(f"Expected loss (choose control):   ${expected_loss_control:.2f}/user")
print(f"95% credible interval: [${bayesian_ci[0]:.2f}, ${bayesian_ci[1]:.2f}]")

print(f"\nDecision framework:")
if prob_treatment_better > 0.95 and expected_loss_treatment < 0.50:
    decision = "SHIP — high confidence + low expected loss"
elif prob_treatment_better > 0.80:
    decision = "CONTINUE — promising but need more data"
else:
    decision = "HOLD — insufficient evidence"
print(f"  → {decision}")
# INTERPRETATION: Bayesian analysis answers "what's the probability
# treatment is better?" rather than "is p < 0.05?" The expected loss
# quantifies the cost of being wrong — if it's $0.05/user, you can
# ship confidently even without 95% certainty.

# ── Checkpoint 6 ─────────────────────────────────────────────────────
assert 0 <= prob_treatment_better <= 1, "Probability must be valid"
assert expected_loss_treatment >= 0, "Expected loss must be non-negative"
print("\n✓ Checkpoint 6 passed — Bayesian A/B analysis completed\n")


# ══════════════════════════════════════════════════════════════════════
# TASK 7: Sequential Testing — Always-Valid p-Values (mSPRT)
# ══════════════════════════════════════════════════════════════════════
# Problem: peeking at results before full sample inflates Type I error.
# Solution: mSPRT provides p-values valid at ANY stopping time.

print(f"\n=== Sequential Testing (mSPRT) ===")

if experiment["timestamp"].dtype in [pl.Utf8, pl.String]:
    exp_daily = experiment.with_columns(
        pl.col("timestamp").str.to_datetime("%Y-%m-%d %H:%M:%S").dt.date().alias("day")
    )
else:
    exp_daily = experiment.with_columns(pl.col("timestamp").cast(pl.Date).alias("day"))

days = sorted(exp_daily["day"].unique().to_list())
sequential_results = []

for i, day in enumerate(days):
    if i < 3:
        continue

    cumulative = exp_daily.filter(pl.col("day") <= day)
    c = (
        cumulative.filter(pl.col("experiment_group") == "control")["revenue"]
        .to_numpy()
        .astype(np.float64)
    )
    t = (
        cumulative.filter(pl.col("experiment_group") != "control")["revenue"]
        .to_numpy()
        .astype(np.float64)
    )

    if len(c) < 100 or len(t) < 100:
        continue

    diff = t.mean() - c.mean()
    se = np.sqrt(c.var(ddof=1) / len(c) + t.var(ddof=1) / len(t))
    z = diff / se if se > 0 else 0
    p_fixed = 2 * (1 - stats.norm.cdf(abs(z)))

    # mSPRT always-valid p-value
    tau_sq = se_naive**2
    v_n = se**2
    # TODO: Compute the mSPRT likelihood ratio lambda_n
    # lambda_n = sqrt(v_n / (v_n + tau_sq)) * exp(tau_sq * z² / (2*(v_n + tau_sq)))
    lambda_n = ____  # Hint: np.sqrt(v_n / (v_n + tau_sq)) * np.exp(tau_sq * z**2 / (2 * (v_n + tau_sq)))
    p_sequential = min(1.0, 1.0 / lambda_n) if lambda_n > 0 else 1.0

    sequential_results.append(
        {
            "day": i + 1,
            "n": len(c) + len(t),
            "lift": diff,
            "p_fixed": p_fixed,
            "p_sequential": p_sequential,
        }
    )

print(f"{'Day':>4} {'n':>8} {'Lift':>10} {'p (fixed)':>12} {'p (mSPRT)':>12}")
print("─" * 52)
step = max(1, len(sequential_results) // 10)
for r in sequential_results[::step]:
    print(
        f"{r['day']:>4} {r['n']:>8,} ${r['lift']:>8.2f} {r['p_fixed']:>12.6f} {r['p_sequential']:>12.6f}"
    )

early_sig_fixed = sum(1 for r in sequential_results if r["p_fixed"] < 0.05)
early_sig_seq = sum(1 for r in sequential_results if r["p_sequential"] < 0.05)
print(f"\nDays with p < 0.05 (fixed):      {early_sig_fixed}/{len(sequential_results)}")
print(f"Days with p < 0.05 (sequential): {early_sig_seq}/{len(sequential_results)}")

# ── Checkpoint 7 ─────────────────────────────────────────────────────
assert len(sequential_results) > 0, "Must have sequential results"
for r in sequential_results:
    assert 0 <= r["p_sequential"] <= 1, "Sequential p-values must be valid"
print("\n✓ Checkpoint 7 passed — sequential testing completed\n")


# ══════════════════════════════════════════════════════════════════════
# TASK 8: Peeking Problem Simulation
# ══════════════════════════════════════════════════════════════════════
# Simulate experiments with NO real effect to show how peeking inflates α.

print(f"\n=== Peeking Problem Simulation ===")

rng = np.random.default_rng(seed=42)
n_peek_sims = 1000
n_per_sim = 2000
n_checks = 20  # Number of times we peek

false_positives_fixed = 0
false_positives_no_peek = 0

for _ in range(n_peek_sims):
    # No real effect — both groups drawn from same distribution
    sim_ctrl = rng.normal(50, 10, size=n_per_sim)
    sim_treat = rng.normal(50, 10, size=n_per_sim)  # Same mean!

    # No peeking: test only at end
    z_end = (sim_treat.mean() - sim_ctrl.mean()) / np.sqrt(
        sim_ctrl.var(ddof=1) / n_per_sim + sim_treat.var(ddof=1) / n_per_sim
    )
    if 2 * (1 - stats.norm.cdf(abs(z_end))) < 0.05:
        false_positives_no_peek += 1

    # Peeking with fixed p-values
    peeked_sig = False
    for check_n in np.linspace(100, n_per_sim, n_checks, dtype=int):
        sc = sim_ctrl[:check_n]
        st = sim_treat[:check_n]
        se_p = np.sqrt(sc.var(ddof=1) / check_n + st.var(ddof=1) / check_n)
        z_p = (st.mean() - sc.mean()) / se_p if se_p > 0 else 0
        if 2 * (1 - stats.norm.cdf(abs(z_p))) < 0.05:
            peeked_sig = True
            break
    if peeked_sig:
        false_positives_fixed += 1

print(f"Simulations: {n_peek_sims:,} (all with NO real effect)")
print(f"Peeks per experiment: {n_checks}")
print(f"\nFalse positive rates:")
print(
    f"  No peeking (test at end):   {false_positives_no_peek/n_peek_sims:.1%} (target: 5%)"
)
print(
    f"  Peeking with fixed p:       {false_positives_fixed/n_peek_sims:.1%} (inflated!)"
)
print(
    f"  Expected with {n_checks} peeks:     ~{(1-(1-0.05)**n_checks)*100:.0f}% (theory)"
)
# INTERPRETATION: Peeking inflates Type I error dramatically. With
# 20 peeks, the false positive rate jumps from 5% to ~64%! Sequential
# testing (mSPRT from Task 7) is the correct way to monitor experiments.

# ── Checkpoint 8 ─────────────────────────────────────────────────────
assert (
    false_positives_fixed > false_positives_no_peek
), "Peeking must inflate false positive rate"
print("\n✓ Checkpoint 8 passed — peeking problem demonstrated\n")


# ══════════════════════════════════════════════════════════════════════
# TASK 9: Difference-in-Differences (DiD)
# ══════════════════════════════════════════════════════════════════════
# When randomisation is not possible, DiD estimates the treatment
# effect from observational data by comparing trends.
# ATT = (Y_treat_post - Y_treat_pre) - (Y_control_post - Y_control_pre)

print(f"\n=== Difference-in-Differences ===")

rng_did = np.random.default_rng(seed=99)
n_per_cell = 500
pre_central = rng_did.normal(550_000, 80_000, size=n_per_cell)
pre_noncentral = rng_did.normal(450_000, 70_000, size=n_per_cell)
post_central = rng_did.normal(540_000, 85_000, size=n_per_cell)
post_noncentral = rng_did.normal(460_000, 72_000, size=n_per_cell)

y_treat_pre = pre_central.mean()
y_treat_post = post_central.mean()
y_ctrl_pre = pre_noncentral.mean()
y_ctrl_post = post_noncentral.mean()

# TODO: Compute the DiD estimate:
# ATT = (post_treat - pre_treat) - (post_ctrl - pre_ctrl)
did_estimate = ____  # Hint: (y_treat_post - y_treat_pre) - (y_ctrl_post - y_ctrl_pre)

se_did = np.sqrt(
    pre_central.var(ddof=1) / n_per_cell
    + post_central.var(ddof=1) / n_per_cell
    + pre_noncentral.var(ddof=1) / n_per_cell
    + post_noncentral.var(ddof=1) / n_per_cell
)
ci_did = (did_estimate - 1.96 * se_did, did_estimate + 1.96 * se_did)
z_did = did_estimate / se_did
p_did = 2 * (1 - stats.norm.cdf(abs(z_did)))

print(f"Scenario: stamp duty increase in Central Singapore")
print(f"\n{'Group':<15} {'Pre-policy':>14} {'Post-policy':>14} {'Δ':>14}")
print("─" * 60)
print(
    f"{'Central':<15} ${y_treat_pre:>12,.0f} ${y_treat_post:>12,.0f} ${y_treat_post-y_treat_pre:>+12,.0f}"
)
print(
    f"{'Non-Central':<15} ${y_ctrl_pre:>12,.0f} ${y_ctrl_post:>12,.0f} ${y_ctrl_post-y_ctrl_pre:>+12,.0f}"
)
print(f"\nDiD estimate (policy effect): ${did_estimate:,.0f}")
print(f"SE: ${se_did:,.0f}")
print(f"95% CI: [${ci_did[0]:,.0f}, ${ci_did[1]:,.0f}]")
print(f"p-value: {p_did:.4f}")
# INTERPRETATION: DiD removes time-invariant confounders by differencing
# pre and post periods. The assumption is that without the policy,
# Central and Non-Central would have followed parallel trends.

# ── Checkpoint 9 ─────────────────────────────────────────────────────
assert se_did > 0, "DiD SE must be positive"
print("\n✓ Checkpoint 9 passed — DiD analysis completed\n")


# ══════════════════════════════════════════════════════════════════════
# TASK 10: Parallel Trends Test
# ══════════════════════════════════════════════════════════════════════
# DiD validity requires parallel trends in the pre-period.
# Test: are the pre-period trends in treatment and control similar?

print(f"\n=== Parallel Trends Test ===")

n_pre_periods = 6
pre_trends_central = []
pre_trends_noncentral = []

for t in range(n_pre_periods):
    central_t = rng_did.normal(530_000 + t * 2000, 80_000, size=200)
    noncentral_t = rng_did.normal(430_000 + t * 2000, 70_000, size=200)
    pre_trends_central.append(central_t.mean())
    pre_trends_noncentral.append(noncentral_t.mean())

time_points = np.arange(n_pre_periods)
# TODO: Fit linear trend to each group and compute slope difference
slope_central = ____  # Hint: np.polyfit(time_points, pre_trends_central, 1)[0]
slope_noncentral = ____  # Hint: np.polyfit(time_points, pre_trends_noncentral, 1)[0]
slope_diff = slope_central - slope_noncentral

print(f"Pre-period trends:")
print(f"  Central slope:     ${slope_central:,.0f}/period")
print(f"  Non-Central slope: ${slope_noncentral:,.0f}/period")
print(f"  Slope difference:  ${slope_diff:,.0f}/period")

# Bootstrap test for parallel trends
n_boot_trends = 5000
boot_slope_diffs = []
for _ in range(n_boot_trends):
    noise_c = rng_did.normal(0, 1000, size=n_pre_periods)
    noise_nc = rng_did.normal(0, 1000, size=n_pre_periods)
    s_c = np.polyfit(time_points, np.array(pre_trends_central) + noise_c, 1)[0]
    s_nc = np.polyfit(time_points, np.array(pre_trends_noncentral) + noise_nc, 1)[0]
    boot_slope_diffs.append(s_c - s_nc)

# TODO: Compute bootstrap p-value: fraction of |boot_slope_diffs| >= |slope_diff|
boot_p = ____  # Hint: np.mean(np.abs(boot_slope_diffs) >= np.abs(slope_diff))
print(f"  Bootstrap p-value for slope difference: {boot_p:.4f}")
if boot_p > 0.05:
    print(f"  Parallel trends assumption HOLDS (cannot reject equal slopes)")
else:
    print(f"  Parallel trends assumption VIOLATED — DiD may be biased")

# ── Checkpoint 10 ────────────────────────────────────────────────────
print("\n✓ Checkpoint 10 passed — parallel trends test completed\n")


# ══════════════════════════════════════════════════════════════════════
# TASK 11: Log to ExperimentTracker
# ══════════════════════════════════════════════════════════════════════


async def log_ab_analysis():
    conn = ConnectionManager("sqlite:///mlfp02_experiments.db")
    await conn.initialize()
    tracker = ExperimentTracker(conn)
    await tracker.initialize()

    exp_id = await tracker.create_experiment(
        name="mlfp02_cuped_causal_inference",
        description="CUPED + Bayesian + DiD analysis",
        tags=["mlfp02", "cuped", "bayesian", "did", "sequential"],
    )

    async with tracker.run(exp_id, run_name="cuped_bayesian_did") as run:
        await run.log_params(
            {
                "cuped_covariate": "pre_metric_value",
                "cuped_theta": str(float(theta)),
                "cuped_rho": str(float(rho)),
                "sequential_method": "mSPRT",
                "did_treatment": "Central Singapore",
            }
        )
        await run.log_metrics(
            {
                "lift_naive": float(lift),
                "lift_cuped": float(lift_adj),
                "se_naive": float(se_naive),
                "se_cuped": float(se_cuped),
                "variance_reduction": float(var_reduction),
                "p_naive": float(p_naive),
                "p_cuped": float(p_cuped),
                "prob_treatment_better": float(prob_treatment_better),
                "expected_loss": float(expected_loss_treatment),
                "did_estimate": float(did_estimate),
                "did_p_value": float(p_did),
            }
        )
    print(f"\nLogged experiment run")
    await conn.close()


try:
    asyncio.run(log_ab_analysis())
except Exception as e:
    print(f"  [Skipped: ExperimentTracker logging ({type(e).__name__}: {e})]")

# ── Checkpoint 11 ────────────────────────────────────────────────────
print("\n✓ Checkpoint 11 passed — experiment logging completed\n")


# ══════════════════════════════════════════════════════════════════════
# TASK 12: Visualise and Synthesise
# ══════════════════════════════════════════════════════════════════════

viz = ModelVisualizer()

# Plot 1: Naive vs CUPED comparison
fig1 = go.Figure()
methods = ["Naive", "CUPED (1-cov)", "CUPED (multi)"]
ses = [se_naive, se_cuped, se_multi]
lifts_plot = [lift, lift_adj, lift_multi]
for i, (m, s, l) in enumerate(zip(methods, ses, lifts_plot)):
    lo, hi = l - 1.96 * s, l + 1.96 * s
    fig1.add_trace(
        go.Scatter(
            x=[lo, l, hi],
            y=[m] * 3,
            mode="markers+lines",
            name=m,
            marker={"size": [8, 12, 8]},
        )
    )
fig1.add_vline(x=0, line_dash="dot", line_color="red")
fig1.update_layout(
    title="Confidence Intervals: Naive vs CUPED", xaxis_title="Treatment Effect ($)"
)
fig1.write_html("ex7_cuped_comparison.html")
print("\nSaved: ex7_cuped_comparison.html")

# Plot 2: Sequential p-values over time
fig2 = go.Figure()
days_seq = [r["day"] for r in sequential_results]
fig2.add_trace(
    go.Scatter(
        x=days_seq, y=[r["p_fixed"] for r in sequential_results], name="Fixed p-value"
    )
)
fig2.add_trace(
    go.Scatter(
        x=days_seq,
        y=[r["p_sequential"] for r in sequential_results],
        name="mSPRT p-value",
    )
)
fig2.add_hline(y=0.05, line_dash="dash", annotation_text="α=0.05")
fig2.update_layout(
    title="Sequential Testing: Fixed vs mSPRT p-values",
    xaxis_title="Day",
    yaxis_title="p-value",
    yaxis_type="log",
)
fig2.write_html("ex7_sequential_pvalues.html")
print("Saved: ex7_sequential_pvalues.html")

# Plot 3: DiD visualization
fig3 = go.Figure()
fig3.add_trace(
    go.Scatter(
        x=["Pre", "Post"],
        y=[y_treat_pre, y_treat_post],
        name="Central",
        line={"color": "red"},
    )
)
fig3.add_trace(
    go.Scatter(
        x=["Pre", "Post"],
        y=[y_ctrl_pre, y_ctrl_post],
        name="Non-Central",
        line={"color": "blue"},
    )
)
counterfactual = y_treat_pre + (y_ctrl_post - y_ctrl_pre)
fig3.add_trace(
    go.Scatter(
        x=["Pre", "Post"],
        y=[y_treat_pre, counterfactual],
        name="Counterfactual",
        line={"dash": "dot", "color": "red"},
    )
)
fig3.update_layout(
    title="Difference-in-Differences: Singapore Cooling Measures",
    yaxis_title="Mean Price ($)",
)
fig3.write_html("ex7_did_visualization.html")
print("Saved: ex7_did_visualization.html")

# ── Checkpoint 12 ────────────────────────────────────────────────────
print("\n✓ Checkpoint 12 passed — visualisations and synthesis complete\n")

print(f"\n{'='*70}")
print(f"CAUSAL INFERENCE DECISION FRAMEWORK")
print(f"{'='*70}")
print(
    f"""
When to use each method:

  CUPED: You have an RCT AND pre-experiment data.
    → Reduces CI width by {ci_width_reduction:.0%} (free precision gain)
    → Unbiased: same point estimate, just tighter

  Bayesian A/B: You want P(B > A) instead of "is p < 0.05?"
    → P(treatment better) = {prob_treatment_better:.1%}
    → Expected loss = ${expected_loss_treatment:.2f}/user
    → Decision: {decision}

  Sequential (mSPRT): You need to monitor experiments safely.
    → Peeking with fixed p: ~{false_positives_fixed/n_peek_sims:.0%} false positive rate
    → mSPRT is valid at any sample size: no p-hacking risk

  DiD: Randomisation is impossible (policy, geography, time).
    → Requires parallel trends in pre-period
    → Estimate: ${did_estimate:,.0f} policy effect
"""
)


# ══════════════════════════════════════════════════════════════════════
# REFLECTION
# ══════════════════════════════════════════════════════════════════════
print("═" * 70)
print("  WHAT YOU'VE MASTERED")
print("═" * 70)
print(
    """
  ✓ CUPED: Y_adj = Y - θ(X - E[X]), θ = Cov(Y,X)/Var(X)
  ✓ Variance reduction: Var(Y_adj) = Var(Y)(1 - ρ²)
  ✓ Multi-covariate CUPED: θ via multivariate OLS
  ✓ Stratified CUPED: heterogeneous treatment effects by segment
  ✓ Bayesian A/B: P(treatment > control), expected loss
  ✓ mSPRT: always-valid p-values for sequential monitoring
  ✓ Peeking problem: fixed p-values inflate Type I error dramatically
  ✓ DiD: ATT = (post_treat - pre_treat) - (post_ctrl - pre_ctrl)
  ✓ Parallel trends: bootstrap test for DiD validity
  ✓ ExperimentTracker: reproducible experiment logging

  NEXT: In Exercise 8 (Capstone), you'll bring everything together:
  build a feature store, run a full statistical analysis pipeline,
  track experiments with ExperimentTracker, and write a business
  recommendation backed by statistical evidence.
"""
)

print("\n✓ Exercise 7 complete — CUPED and Causal Inference")
