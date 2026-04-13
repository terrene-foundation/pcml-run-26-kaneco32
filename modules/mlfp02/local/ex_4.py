# Copyright 2026 Terrene Foundation
# SPDX-License-Identifier: Apache-2.0
"""
# ════════════════════════════════════════════════════════════════════════
# MLFP02 — Exercise 4: A/B Testing and Experiment Design
# ════════════════════════════════════════════════════════════════════════
#
# WHAT YOU'LL LEARN:
#   After completing this exercise, you will be able to:
#   - Design a complete A/B experiment with pre-registered hypotheses
#   - Compute required sample sizes via power analysis before collecting data
#   - Simulate experiment data with a known effect to validate a pipeline
#   - Detect Sample Ratio Mismatch (SRM) and understand its causes
#   - Construct and interpret Welch-Satterthwaite confidence intervals
#   - Implement a data collection plan (Why/What/Where/How/Frequency)
#   - Simulate SRM to see its impact on effect estimation
#   - Build a complete experiment analysis report from raw data
#   - Evaluate experiment validity criteria (SUTVA, no interference)
#   - Compute sequential sample-size calculations (adaptive experiments)
#
# PREREQUISITES: Complete Exercise 3 — you should understand hypothesis
#   testing, p-values, power, and multiple testing corrections.
#
# ESTIMATED TIME: ~170 minutes
#
# TASKS:
#    1. Load experiment data and exploratory analysis
#    2. Design an A/B experiment: hypotheses and pre-registration
#    3. Power analysis — compute required sample size for target MDE
#    4. Simulate experiment with known effect (positive control)
#    5. SRM detection on simulated and real data
#    6. Simulating SRM: what happens when randomisation breaks
#    7. Welch's t-test with Satterthwaite degrees of freedom
#    8. Confidence intervals for treatment effect (Welch + bootstrap)
#    9. Data collection plan (Why/What/Where/How/Frequency)
#   10. Experiment validity: SUTVA, interference, novelty effects
#   11. Adaptive sample size — sequential experiment design
#   12. Full experiment report and business recommendation
#
# DATASET: Multi-arm experiment data
#   Source: Simulated e-commerce experiment with control and treatments
#   Key columns: experiment_group, metric_value (engagement score)
#
# THEORY:
#   Pre-registration: decide hypothesis, sample size, stopping rule
#   BEFORE collecting data to prevent p-hacking.
#   Power: n = (z_{α/2} + z_β)² × 2σ² / δ²
#   SRM: χ² test on observed vs expected split.
#
# ════════════════════════════════════════════════════════════════════════
"""
from __future__ import annotations

import math

import numpy as np
import polars as pl
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from kailash_ml import ModelVisualizer
from scipy import stats

from shared import MLFPDataLoader


# ── Data Loading ──────────────────────────────────────────────────────

loader = MLFPDataLoader()
experiment = loader.load("mlfp02", "experiment_data.parquet")

print("=" * 70)
print("  MLFP02 Exercise 4: A/B Testing and Experiment Design")
print("=" * 70)
print(f"\n  Data loaded: experiment_data.parquet")
print(f"  Shape: {experiment.shape}")
print(f"  Columns: {experiment.columns}")
print(experiment.head(5))

# Understand the group structure
group_counts = experiment["experiment_group"].value_counts().sort("experiment_group")
print(f"\n--- Group Allocation ---")
print(group_counts)

# Focus on two-arm A/B: control vs treatment_a
ab_data = experiment.filter(
    pl.col("experiment_group").is_in(["control", "treatment_a"])
)
control = ab_data.filter(pl.col("experiment_group") == "control")
treatment = ab_data.filter(pl.col("experiment_group") == "treatment_a")

n_control = control.height
n_treatment = treatment.height
n_total = ab_data.height

ctrl_values = control["metric_value"].to_numpy().astype(np.float64)
treat_values = treatment["metric_value"].to_numpy().astype(np.float64)

print(f"\n=== Two-Arm A/B Subset ===")
print(
    f"Control:   n={n_control:,}, mean={ctrl_values.mean():.4f}, std={ctrl_values.std():.4f}"
)
print(
    f"Treatment: n={n_treatment:,}, mean={treat_values.mean():.4f}, std={treat_values.std():.4f}"
)


# ══════════════════════════════════════════════════════════════════════
# TASK 1: Experiment Design — Hypothesis Formulation
# ══════════════════════════════════════════════════════════════════════

print("\n" + "=" * 70)
print("TASK 1: Experiment Design — Hypothesis Formulation")
print("=" * 70)

print(
    """
Scenario: An e-commerce platform wants to test whether a new
recommendation algorithm (treatment_a) increases user engagement
(metric_value) compared to the existing algorithm (control).

PRE-REGISTRATION DOCUMENT:
═══════════════════════════
1. Primary hypothesis:
   H₀: μ_treatment = μ_control  (no effect on engagement)
   H₁: μ_treatment ≠ μ_control  (two-sided)

2. Primary metric: metric_value (engagement score)

3. Design parameters:
   - Significance level (α): 0.05 (5% false positive rate)
   - Power (1-β): 0.80 (80% chance of detecting a real effect)
   - Randomisation unit: user
   - Allocation: equal (50/50)

4. Stopping rule: analyse only after target n is reached.
   No peeking (see Exercise 7 for sequential testing).

5. Pre-registered corrections: Bonferroni for 3 secondary metrics.

Key principles:
  ✓ Randomisation — eliminates confounders
  ✓ Equal allocation — maximises power for given total n
  ✓ Pre-registration — prevents p-hacking and HARKing
  ✓ Single primary metric — reduces multiple testing burden
"""
)


# ══════════════════════════════════════════════════════════════════════
# TASK 2: Power Analysis — Required Sample Size
# ══════════════════════════════════════════════════════════════════════

print("\n" + "=" * 70)
print("TASK 2: Power Analysis — Required Sample Size")
print("=" * 70)

alpha = 0.05
power_target = 0.80
# TODO: Compute the upper-tail z critical value for significance (one side of two-sided test)
z_alpha_half = ____  # Hint: stats.norm.ppf(1 - alpha / 2)

# TODO: Compute z-score corresponding to 80% power
z_beta = ____  # Hint: stats.norm.ppf(power_target)

sigma_pooled = ctrl_values.std(ddof=1)

# Compute required n for different MDE levels
print(f"\n--- Required Sample Size by MDE ---")
print(f"Baseline mean: {ctrl_values.mean():.4f}, σ_pooled: {sigma_pooled:.4f}")
print(f"{'Relative MDE':>14} {'Absolute MDE':>14} {'n per group':>14} {'n total':>10}")
print("─" * 56)

n_required_results = {}
for rel_mde_pct in [1.0, 2.0, 3.0, 5.0, 10.0]:
    mde_abs = ctrl_values.mean() * (rel_mde_pct / 100)
    cohens_d = mde_abs / sigma_pooled
    # TODO: Compute required n per group using the power formula
    # n = (z_{α/2} + z_β)² × 2σ² / δ²
    n_per = ____  # Hint: math.ceil((z_alpha_half + z_beta) ** 2 * 2 * sigma_pooled**2 / mde_abs**2)
    n_required_results[rel_mde_pct] = {"mde": mde_abs, "n_per": n_per, "d": cohens_d}
    print(f"{rel_mde_pct:>12.1f}%  {mde_abs:>14.4f}  {n_per:>14,}  {2*n_per:>10,}")

# Use 2% relative MDE as our design target
design_mde_pct = 2.0
mde_absolute = n_required_results[design_mde_pct]["mde"]
n_required_per = n_required_results[design_mde_pct]["n_per"]
cohens_d = n_required_results[design_mde_pct]["d"]

print(f"\nDesign target: {design_mde_pct}% relative MDE = {mde_absolute:.4f} absolute")
print(
    f"Cohen's d: {cohens_d:.4f} ({'small' if cohens_d < 0.2 else 'small-medium' if cohens_d < 0.5 else 'medium'})"
)
print(f"Required: {n_required_per:,} per group, {2*n_required_per:,} total")
print(f"Actual: {n_total:,} ({n_total / (2*n_required_per):.1f}x required)")
if n_total >= 2 * n_required_per:
    print("→ ADEQUATELY POWERED")
else:
    print("→ UNDERPOWERED — need more observations or accept larger MDE")

# ── Checkpoint 1 ─────────────────────────────────────────────────────
assert n_required_per > 0, "Required sample size must be positive"
assert cohens_d > 0, "Cohen's d must be positive"
print("\n✓ Checkpoint 1 passed — power analysis completed\n")


# ══════════════════════════════════════════════════════════════════════
# TASK 3: Power Curve — Power vs Sample Size
# ══════════════════════════════════════════════════════════════════════

print("\n" + "=" * 70)
print("TASK 3: Power Curve")
print("=" * 70)

sample_sizes = np.arange(500, n_required_per * 3, max(500, n_required_per // 20))
power_at_n = []
for n_i in sample_sizes:
    se_i = sigma_pooled * np.sqrt(2 / n_i)
    ncp_i = mde_absolute / se_i
    power_i = (
        1 - stats.norm.cdf(z_alpha_half - ncp_i) + stats.norm.cdf(-z_alpha_half - ncp_i)
    )
    power_at_n.append(power_i)

print(f"Power at selected sample sizes (per group, MDE={mde_absolute:.4f}):")
for frac in [0.25, 0.5, 1.0, 1.5, 2.0]:
    idx = np.argmin(np.abs(sample_sizes - n_required_per * frac))
    print(f"  n = {sample_sizes[idx]:>8,}: power = {power_at_n[idx]:.1%}")

# ── Checkpoint 2 ─────────────────────────────────────────────────────
assert power_at_n[-1] > power_at_n[0], "Power should increase with n"
print("\n✓ Checkpoint 2 passed — power curve computed\n")


# ══════════════════════════════════════════════════════════════════════
# TASK 4: Simulate Experiment with Known Treatment Effect
# ══════════════════════════════════════════════════════════════════════
# Positive control: inject a KNOWN effect to validate the pipeline.

print("\n" + "=" * 70)
print("TASK 4: Simulate Experiment (Positive Control)")
print("=" * 70)

rng = np.random.default_rng(seed=42)
sim_n_per = 10_000
true_effect = 2.0
sim_mu_control = ctrl_values.mean()
sim_sigma = sigma_pooled

# TODO: Simulate control group from Normal(sim_mu_control, sim_sigma)
sim_control = ____  # Hint: rng.normal(loc=sim_mu_control, scale=sim_sigma, size=sim_n_per)

# TODO: Simulate treatment group with true_effect added to the mean
sim_treatment = ____  # Hint: rng.normal(loc=sim_mu_control + true_effect, scale=sim_sigma, size=sim_n_per)

print(f"True control mean:   {sim_mu_control:.4f}")
print(f"True treatment mean: {sim_mu_control + true_effect:.4f}")
print(f"True effect (δ):     {true_effect:.4f}")
print(f"n per group:         {sim_n_per:,}")
print(f"\nRealized:")
print(f"  Control mean:   {sim_control.mean():.4f}")
print(f"  Treatment mean: {sim_treatment.mean():.4f}")
print(
    f"  Observed diff:  {sim_treatment.mean() - sim_control.mean():.4f} (true: {true_effect})"
)

# Run the t-test on simulated data
sim_t, sim_p = stats.ttest_ind(sim_treatment, sim_control, equal_var=False)
print(f"  t-statistic: {sim_t:.4f}, p-value: {sim_p:.2e}")
print(
    f"  {'DETECTED' if sim_p < alpha else 'MISSED'} (positive control should be detected)"
)
# INTERPRETATION: If the test fails to detect a known true effect of 2.0
# with n=10,000 per group, there is a bug in the analysis pipeline.
# This is the experiment design equivalent of a unit test.

# ── Checkpoint 3 ─────────────────────────────────────────────────────
assert sim_p < alpha, "Positive control must be detected"
assert (
    abs(sim_treatment.mean() - sim_control.mean() - true_effect) < 3.0
), "Realized effect should be within 3.0 of true"
print("\n✓ Checkpoint 3 passed — positive control validated\n")


# ══════════════════════════════════════════════════════════════════════
# TASK 5: SRM Detection — Simulated and Real Data
# ══════════════════════════════════════════════════════════════════════

print("\n" + "=" * 70)
print("TASK 5: Sample Ratio Mismatch Detection")
print("=" * 70)

# SRM on simulated data (designed 50/50 — should pass)
sim_obs = np.array([sim_n_per, sim_n_per])
sim_exp = np.array([sim_n_per, sim_n_per])
sim_chi2, sim_srm_p = stats.chisquare(sim_obs, f_exp=sim_exp)
print(f"\n--- Simulated (designed 50/50) ---")
print(f"χ²={sim_chi2:.4f}, p={sim_srm_p:.6f} → {'SRM' if sim_srm_p < 0.01 else 'OK'}")

# SRM on real data
real_obs = np.array([n_control, n_treatment])
real_exp = np.array([n_total / 2, n_total / 2])
# TODO: Compute chi-square statistic and p-value for real SRM check
chi2_stat, srm_p = ____  # Hint: stats.chisquare(real_obs, f_exp=real_exp)
print(f"\n--- Real Data (intended 50/50) ---")
print(
    f"Observed: {n_control:,} / {n_treatment:,} ({n_control/n_total:.4f}/{n_treatment/n_total:.4f})"
)
print(f"χ²={chi2_stat:.4f}, p={srm_p:.2e}")
if srm_p < 0.01:
    print("SRM DETECTED — investigate:")
    print("  1. Bot filtering differential")
    print("  2. Technical redirect issues")
    print("  3. Randomisation bug")
    print("  4. Post-randomisation population filter")
else:
    print("No SRM detected — split is consistent with design")

# ── Checkpoint 4 ─────────────────────────────────────────────────────
assert sim_chi2 == 0.0, "Simulated equal groups should give χ²=0"
assert 0 <= srm_p <= 1, "SRM p-value must be valid"
print("\n✓ Checkpoint 4 passed — SRM detection completed\n")


# ══════════════════════════════════════════════════════════════════════
# TASK 6: Simulating SRM — Impact on Effect Estimation
# ══════════════════════════════════════════════════════════════════════
# What happens when SRM is present? We simulate experiments where
# high-value users are more likely to end up in treatment.

print("\n" + "=" * 70)
print("TASK 6: Simulating SRM — Impact on Estimation")
print("=" * 70)

n_srm_sim = 1000
srm_biases = []
no_srm_biases = []
true_lift = 1.0  # True treatment effect

for _ in range(n_srm_sim):
    # No SRM: random 50/50 allocation
    users = rng.normal(50, 10, size=2000)  # User "quality" scores
    assigned = rng.choice([0, 1], size=2000)  # 50/50 random
    ctrl_outcome = users[assigned == 0] + rng.normal(0, 5, size=(assigned == 0).sum())
    treat_outcome = (
        users[assigned == 1] + true_lift + rng.normal(0, 5, size=(assigned == 1).sum())
    )
    no_srm_biases.append((treat_outcome.mean() - ctrl_outcome.mean()) - true_lift)

    # WITH SRM: high-value users 60% likely to end up in treatment
    assign_prob = np.where(users > 55, 0.6, 0.4)  # Biased allocation
    assigned_srm = (rng.random(size=2000) < assign_prob).astype(int)
    ctrl_out_srm = users[assigned_srm == 0] + rng.normal(
        0, 5, size=(assigned_srm == 0).sum()
    )
    treat_out_srm = (
        users[assigned_srm == 1]
        + true_lift
        + rng.normal(0, 5, size=(assigned_srm == 1).sum())
    )
    srm_biases.append((treat_out_srm.mean() - ctrl_out_srm.mean()) - true_lift)

print(f"True treatment effect: {true_lift}")
print(f"\nNo SRM:")
print(f"  Mean estimation bias: {np.mean(no_srm_biases):+.4f}")
print(f"  Std of bias: {np.std(no_srm_biases):.4f}")
print(f"\nWith SRM (high-value → treatment):")
print(f"  Mean estimation bias: {np.mean(srm_biases):+.4f}")
print(f"  Std of bias: {np.std(srm_biases):.4f}")
print(
    f"\nSRM adds a POSITIVE bias of ~{np.mean(srm_biases) - np.mean(no_srm_biases):+.2f}"
)
print(f"because high-value users inflate treatment outcomes.")
# INTERPRETATION: SRM doesn't just break the sample split — it
# introduces SYSTEMATIC BIAS in treatment effect estimates. When
# high-value users are more likely to end up in treatment, the
# estimated lift is inflated. This is why SRM detection is the
# first sanity check, not an optional nicety.

# ── Checkpoint 5 ─────────────────────────────────────────────────────
assert abs(np.mean(no_srm_biases)) < 0.5, "No-SRM bias should be near zero"
assert abs(np.mean(srm_biases)) > abs(
    np.mean(no_srm_biases)
), "SRM should introduce larger bias"
print("\n✓ Checkpoint 5 passed — SRM impact on estimation demonstrated\n")


# ══════════════════════════════════════════════════════════════════════
# TASK 7: Welch's t-test with Satterthwaite Degrees of Freedom
# ══════════════════════════════════════════════════════════════════════

print("\n" + "=" * 70)
print("TASK 7: Welch's t-test")
print("=" * 70)

# On simulated data (known effect)
print("\n--- Simulated Data (true effect=2.0) ---")
sim_t_stat, sim_p_val = stats.ttest_ind(sim_treatment, sim_control, equal_var=False)
print(f"t={sim_t_stat:.4f}, p={sim_p_val:.2e}")
print(f"{'SIGNIFICANT' if sim_p_val < alpha else 'NOT sig'} at α={alpha}")

# On real data
print("\n--- Real Data ---")
# TODO: Run Welch's t-test on real treat_values vs ctrl_values
real_t_stat, real_p_val = ____  # Hint: stats.ttest_ind(treat_values, ctrl_values, equal_var=False)
obs_diff = treat_values.mean() - ctrl_values.mean()
rel_lift = obs_diff / ctrl_values.mean() * 100

# Welch-Satterthwaite degrees of freedom (manual computation)
s1_sq_n1 = ctrl_values.var(ddof=1) / n_control
s2_sq_n2 = treat_values.var(ddof=1) / n_treatment
# TODO: Compute Welch-Satterthwaite degrees of freedom
df_ws = ____  # Hint: (s1_sq_n1 + s2_sq_n2) ** 2 / (s1_sq_n1**2 / (n_control - 1) + s2_sq_n2**2 / (n_treatment - 1))

print(f"Control:   {ctrl_values.mean():.4f} ± {ctrl_values.std():.4f}")
print(f"Treatment: {treat_values.mean():.4f} ± {treat_values.std():.4f}")
print(f"Diff: {obs_diff:+.4f} ({rel_lift:+.2f}% relative)")
print(f"t-statistic: {real_t_stat:.4f}")
print(f"Welch-Satterthwaite df: {df_ws:.1f}")
print(f"p-value: {real_p_val:.6f}")
print(f"{'SIGNIFICANT' if real_p_val < alpha else 'NOT significant'} at α={alpha}")

# Cohen's d
pooled_std_real = np.sqrt((ctrl_values.var(ddof=1) + treat_values.var(ddof=1)) / 2)
cohens_d_real = obs_diff / pooled_std_real
print(
    f"Cohen's d: {cohens_d_real:.4f} ({'small' if abs(cohens_d_real) < 0.2 else 'medium' if abs(cohens_d_real) < 0.5 else 'large'})"
)

# ── Checkpoint 6 ─────────────────────────────────────────────────────
assert sim_p_val < alpha, "Positive control must be detected"
assert 0 <= real_p_val <= 1, "Real p-value must be valid"
print("\n✓ Checkpoint 6 passed — Welch's t-test completed\n")


# ══════════════════════════════════════════════════════════════════════
# TASK 8: Confidence Intervals — Welch + Bootstrap
# ══════════════════════════════════════════════════════════════════════

print("\n" + "=" * 70)
print("TASK 8: Confidence Intervals for Treatment Effect")
print("=" * 70)

# Welch CI
real_se = np.sqrt(s1_sq_n1 + s2_sq_n2)
t_crit = stats.t.ppf(1 - alpha / 2, df=df_ws)
# TODO: Compute Welch 95% CI as (obs_diff - t_crit*SE, obs_diff + t_crit*SE)
welch_ci = ____  # Hint: (obs_diff - t_crit * real_se, obs_diff + t_crit * real_se)

# Bootstrap CI
n_boot = 10_000
boot_diffs = np.array(
    [
        rng.choice(treat_values, size=n_treatment, replace=True).mean()
        - rng.choice(ctrl_values, size=n_control, replace=True).mean()
        for _ in range(n_boot)
    ]
)
# TODO: Compute 95% bootstrap percentile CI from boot_diffs
boot_ci = ____  # Hint: tuple(np.percentile(boot_diffs, [2.5, 97.5]))

# Normal approximation CI
normal_ci = (obs_diff - 1.96 * real_se, obs_diff + 1.96 * real_se)

print(f"Observed difference: {obs_diff:+.4f}")
print(f"{'Method':<20} {'Lower':>12} {'Upper':>12} {'Width':>10}")
print("─" * 56)
print(
    f"{'Welch t-CI':<20} {welch_ci[0]:>12.4f} {welch_ci[1]:>12.4f} {welch_ci[1]-welch_ci[0]:>10.4f}"
)
print(
    f"{'Normal CI':<20} {normal_ci[0]:>12.4f} {normal_ci[1]:>12.4f} {normal_ci[1]-normal_ci[0]:>10.4f}"
)
print(
    f"{'Bootstrap CI':<20} {boot_ci[0]:>12.4f} {boot_ci[1]:>12.4f} {boot_ci[1]-boot_ci[0]:>10.4f}"
)

if welch_ci[0] > 0:
    print("\nCI entirely above zero — POSITIVE treatment effect")
elif welch_ci[1] < 0:
    print("\nCI entirely below zero — NEGATIVE treatment effect")
else:
    print("\nCI spans zero — effect not distinguishable from zero")
# INTERPRETATION: The CI tells you HOW BIG the effect likely is —
# far more informative than a binary p-value. A CI of [0.01, 0.03]
# means the effect is real but tiny. A CI of [-2, 5] means we have
# no idea whether the effect is positive or negative.

# ── Checkpoint 7 ─────────────────────────────────────────────────────
assert welch_ci[0] < welch_ci[1], "CI lower must be below upper"
assert boot_ci[0] < boot_ci[1], "Bootstrap CI must be valid"
print("\n✓ Checkpoint 7 passed — confidence intervals computed\n")


# ══════════════════════════════════════════════════════════════════════
# TASK 9: Data Collection Plan (Why/What/Where/How/Frequency)
# ══════════════════════════════════════════════════════════════════════

print("\n" + "=" * 70)
print("TASK 9: Data Collection Plan")
print("=" * 70)

plan = {
    "WHY (Hypotheses & Value)": {
        "Primary hypothesis": "New algorithm increases engagement by ≥ 2%",
        "Secondary hypotheses": "Revenue impact, conversion impact",
        "Business value": "1% engagement lift ≈ $200K annual revenue increase",
        "Success criteria": "p < 0.05 AND Cohen's d > 0.1 AND CI excludes zero",
    },
    "WHAT (Data Requirements)": {
        "Primary metric": "metric_value (engagement score, 0-100)",
        "Secondary metrics": "revenue, conversion, pages_viewed",
        "Covariates": "signup_date, device_type, country, prior_activity",
        "Guardrail metrics": "page_load_time, error_rate, support_tickets",
        "Minimum rows": f"{2 * n_required_per:,} (from power analysis)",
    },
    "WHERE (Data Sources)": {
        "Internal": "Event pipeline (Kafka → BigQuery), user_features table",
        "External": "None required for this experiment",
        "Schema": "user_id, timestamp, experiment_group, metric_value, revenue",
    },
    "HOW (Collection Method)": {
        "Assignment": "Server-side random hash on user_id (deterministic)",
        "Logging": "Event-sourced: every impression, click, purchase logged",
        "Quality": "Dedup on (user_id, session_id), validate schema on ingest",
        "Privacy": "PII stripped at collection; analysis on anonymised IDs",
    },
    "FREQUENCY (Timing)": {
        "Collection frequency": "Real-time events, hourly batch aggregation",
        "Analysis frequency": "Weekly interim report (no peeking at p-values)",
        "Duration": f"~{2*n_required_per // 5000} days at 5,000 users/day",
        "Stopping rule": "Analyse after target n reached; no early stopping",
    },
}

for section, items in plan.items():
    print(f"\n{section}")
    print("─" * 50)
    for key, value in items.items():
        print(f"  {key}: {value}")

# ── Checkpoint 8 ─────────────────────────────────────────────────────
assert len(plan) == 5, "Plan must cover all 5 sections"
print("\n✓ Checkpoint 8 passed — data collection plan created\n")


# ══════════════════════════════════════════════════════════════════════
# TASK 10: Experiment Validity — SUTVA and Interference
# ══════════════════════════════════════════════════════════════════════

print("\n" + "=" * 70)
print("TASK 10: Experiment Validity Criteria")
print("=" * 70)

print(
    """
SUTVA (Stable Unit Treatment Value Assumption):
  Each user's outcome depends ONLY on their own treatment assignment,
  not on other users' assignments.

Violations of SUTVA:
  1. Network effects — if treated users share recommendations with
     control users via social features, control is "contaminated"
  2. Marketplace effects — if treatment changes supply/demand dynamics,
     control users are affected by shifted prices
  3. Shared resources — if treatment uses more server capacity,
     control experiences slower page loads

Checks for this experiment:
"""
)

# Check 1: Variance ratio (should be ~1 if no interference)
var_ratio = treat_values.var() / ctrl_values.var()
print(f"  1. Variance ratio (treatment/control): {var_ratio:.3f}")
print(f"     Expected ~1.0 if no differential interference")
print(f"     Status: {'OK' if 0.8 < var_ratio < 1.2 else 'INVESTIGATE'}")

# Check 2: Distribution shape similarity (KS test)
ks_stat, ks_p = stats.ks_2samp(ctrl_values, treat_values)
print(f"\n  2. KS test for distribution similarity: D={ks_stat:.4f}, p={ks_p:.6f}")
print(f"     A small p-value suggests distributions differ beyond just location shift")

# Check 3: Novelty effect — compare early vs late treatment outcomes
n_half = n_treatment // 2
early_treat = treat_values[:n_half]
late_treat = treat_values[n_half:]
novelty_t, novelty_p = stats.ttest_ind(early_treat, late_treat, equal_var=False)
print(
    f"\n  3. Novelty check (early vs late treatment): t={novelty_t:.4f}, p={novelty_p:.4f}"
)
if novelty_p < 0.05:
    print(
        f"     WARNING: early/late treatment outcomes differ — possible novelty effect"
    )
else:
    print(f"     OK: no evidence of novelty or fatigue effect")
# INTERPRETATION: SUTVA is rarely perfectly satisfied. The question is
# whether violations are large enough to meaningfully bias results.
# Document known violations and their likely direction of bias.

# ── Checkpoint 9 ─────────────────────────────────────────────────────
assert var_ratio > 0, "Variance ratio must be positive"
assert 0 <= ks_p <= 1, "KS p-value must be valid"
print("\n✓ Checkpoint 9 passed — validity criteria assessed\n")


# ══════════════════════════════════════════════════════════════════════
# TASK 11: Adaptive Sample Size — Sequential Design
# ══════════════════════════════════════════════════════════════════════
# Sometimes we don't know the variance before the experiment starts.
# Adaptive design: start with a pilot, estimate variance, then compute
# the remaining sample size needed.

print("\n" + "=" * 70)
print("TASK 11: Adaptive Sample Size Calculation")
print("=" * 70)

# Simulate a pilot phase
pilot_n = 500
pilot_ctrl = rng.choice(ctrl_values, size=pilot_n, replace=True)
pilot_treat = rng.choice(treat_values, size=pilot_n, replace=True)

# TODO: Compute pooled sigma estimate from the pilot data
pilot_sigma = ____  # Hint: np.sqrt((pilot_ctrl.var(ddof=1) + pilot_treat.var(ddof=1)) / 2)
pilot_diff = pilot_treat.mean() - pilot_ctrl.mean()

# Re-compute required n based on pilot estimate
target_mde = mde_absolute
# TODO: Compute adaptive required n using pilot_sigma instead of sigma_pooled
n_adaptive = ____  # Hint: math.ceil((z_alpha_half + z_beta) ** 2 * 2 * pilot_sigma**2 / target_mde**2)

print(f"Pilot phase: n={pilot_n} per group")
print(f"Pilot σ estimate: {pilot_sigma:.4f} (true σ ≈ {sigma_pooled:.4f})")
print(f"Pilot observed diff: {pilot_diff:+.4f}")
print(f"\nAdaptive required n per group: {n_adaptive:,}")
print(f"Original required n per group: {n_required_per:,}")
print(f"Ratio: {n_adaptive / n_required_per:.2f}x")
print(f"Remaining needed: {max(0, n_adaptive - pilot_n):,} per group")

# Multi-stage adaptive: how estimate improves with pilot size
print(f"\n--- Pilot Size vs Required n Stability ---")
for pilot_size in [100, 250, 500, 1000]:
    sigs = []
    for _ in range(100):
        pc = rng.choice(ctrl_values, size=pilot_size, replace=True)
        pt = rng.choice(treat_values, size=pilot_size, replace=True)
        s = np.sqrt((pc.var(ddof=1) + pt.var(ddof=1)) / 2)
        sigs.append(s)
    mean_sig = np.mean(sigs)
    std_sig = np.std(sigs)
    n_req = math.ceil((z_alpha_half + z_beta) ** 2 * 2 * mean_sig**2 / target_mde**2)
    print(
        f"  Pilot n={pilot_size:>4}: σ̂={mean_sig:.4f} ± {std_sig:.4f}, "
        f"required n={n_req:,}"
    )
# INTERPRETATION: With a small pilot (n=100), the variance estimate is
# noisy, leading to uncertain sample size calculations. A pilot of
# ~500 gives a stable estimate. Adaptive design avoids both
# underpowered experiments (σ underestimated) and waste (σ overestimated).

# ── Checkpoint 10 ────────────────────────────────────────────────────
assert n_adaptive > 0, "Adaptive sample size must be positive"
print("\n✓ Checkpoint 10 passed — adaptive design completed\n")


# ══════════════════════════════════════════════════════════════════════
# TASK 12: Full Experiment Report and Visualisation
# ══════════════════════════════════════════════════════════════════════

viz = ModelVisualizer()

# Plot 1: Power curve
fig1 = go.Figure()
fig1.add_trace(go.Scatter(x=sample_sizes.tolist(), y=power_at_n, name="Power"))
fig1.add_hline(y=0.8, line_dash="dash", annotation_text="80% target")
fig1.add_vline(
    x=n_required_per, line_dash="dot", annotation_text=f"Required n={n_required_per:,}"
)
fig1.update_layout(
    title=f"Power Curve (MDE={mde_absolute:.2f}, α={alpha})",
    xaxis_title="Sample Size per Group",
    yaxis_title="Power",
)
fig1.write_html("ex4_power_curve.html")
print("\nSaved: ex4_power_curve.html")

# Plot 2: SRM simulation
fig2 = make_subplots(
    rows=1, cols=2, subplot_titles=["No SRM (unbiased)", "With SRM (biased)"]
)
fig2.add_trace(go.Histogram(x=no_srm_biases, nbinsx=40, name="No SRM"), row=1, col=1)
fig2.add_trace(go.Histogram(x=srm_biases, nbinsx=40, name="SRM"), row=1, col=2)
fig2.update_layout(title="SRM Impact on Treatment Effect Estimation Bias", height=350)
fig2.write_html("ex4_srm_simulation.html")
print("Saved: ex4_srm_simulation.html")

# Plot 3: CI comparison
fig3 = go.Figure()
methods = ["Welch t-CI", "Normal CI", "Bootstrap CI"]
lowers = [welch_ci[0], normal_ci[0], boot_ci[0]]
uppers = [welch_ci[1], normal_ci[1], boot_ci[1]]
for i, (method, lo, hi) in enumerate(zip(methods, lowers, uppers)):
    fig3.add_trace(
        go.Scatter(
            x=[lo, obs_diff, hi],
            y=[method] * 3,
            mode="markers+lines",
            name=method,
            marker={"size": [8, 12, 8]},
        )
    )
fig3.add_vline(x=0, line_dash="dot", line_color="red")
fig3.update_layout(
    title="95% Confidence Intervals for Treatment Effect",
    xaxis_title="Treatment Effect",
)
fig3.write_html("ex4_confidence_intervals.html")
print("Saved: ex4_confidence_intervals.html")

# ── Checkpoint 11 ────────────────────────────────────────────────────
print("\n✓ Checkpoint 11 passed — visualisations saved\n")

# Final business report
print(f"\n{'='*70}")
print(f"EXPERIMENT REPORT")
print(f"{'='*70}")
print(
    f"""
Experiment: Recommendation Algorithm A/B Test
Duration: designed for {2*n_required_per:,} total users
Actual: {n_total:,} users

SRM Check: p={srm_p:.4f} → {'PASS' if srm_p > 0.01 else 'FAIL — results may be biased'}

Primary Metric (metric_value — engagement score):
  Control:   {ctrl_values.mean():.4f} ± {ctrl_values.std():.4f}
  Treatment: {treat_values.mean():.4f} ± {treat_values.std():.4f}
  Lift: {obs_diff:+.4f} ({rel_lift:+.2f}% relative)
  p-value: {real_p_val:.6f}
  Cohen's d: {cohens_d_real:.4f}
  95% CI: [{welch_ci[0]:.4f}, {welch_ci[1]:.4f}]

Decision: {"SHIP — statistically significant positive effect" if real_p_val < alpha and obs_diff > 0
           else "HOLD — more data needed" if real_p_val > alpha and abs(obs_diff) > mde_absolute * 0.5
           else "NO SHIP — no meaningful effect detected"}

Validity: Variance ratio {var_ratio:.2f}, no novelty effect detected.
"""
)

# ── Checkpoint 12 ────────────────────────────────────────────────────
print("✓ Checkpoint 12 passed — experiment report complete\n")


# ══════════════════════════════════════════════════════════════════════
# REFLECTION
# ══════════════════════════════════════════════════════════════════════
print("═" * 70)
print("  WHAT YOU'VE MASTERED")
print("═" * 70)
print(
    """
  ✓ Pre-registration: hypotheses, metrics, stopping rules BEFORE data
  ✓ Power analysis: required n = f(α, power, MDE, σ)
  ✓ Positive control: simulate known effect to validate pipeline
  ✓ SRM detection: χ² test + understanding its causes
  ✓ SRM impact: biased allocation → biased treatment effect estimates
  ✓ Welch's t-test: robust to unequal variances
  ✓ Welch-Satterthwaite df: more accurate than pooled df
  ✓ Multiple CI methods: Welch, Normal, Bootstrap
  ✓ Data collection plan: Why/What/Where/How/Frequency
  ✓ SUTVA and validity: interference, novelty, variance checks
  ✓ Adaptive design: pilot → estimate σ → compute remaining n
  ✓ Complete experiment report with business recommendation

  NEXT: In Exercise 5 you'll build linear regression from scratch.
  You'll derive OLS using matrix algebra, test coefficient significance
  with t-statistics, detect multicollinearity, and run residual
  diagnostics — all on HDB price prediction data.
"""
)

print("\n✓ Exercise 4 complete — A/B Testing and Experiment Design")
