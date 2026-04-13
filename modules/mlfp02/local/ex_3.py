# Copyright 2026 Terrene Foundation
# SPDX-License-Identifier: Apache-2.0
"""
# ════════════════════════════════════════════════════════════════════════
# MLFP02 — Exercise 3: Bootstrapping and Hypothesis Testing
# ════════════════════════════════════════════════════════════════════════
#
# WHAT YOU'LL LEARN:
#   After completing this exercise, you will be able to:
#   - Formulate null and alternative hypotheses for business questions
#   - Implement bootstrap resampling from scratch and compute percentile
#     and BCa confidence intervals for any statistic
#   - Compute and correctly interpret p-values (NOT as P(H₀ is true))
#   - Run a two-proportion z-test for conversion rates
#   - Run Mann-Whitney U and Welch's t-test for continuous metrics
#   - Compute statistical power and minimum detectable effect (MDE)
#   - Apply Bonferroni correction (FWER control) and Benjamini-Hochberg
#     (FDR control) for multiple testing
#   - Implement a permutation test as a distribution-free alternative
#   - Simulate false discovery rates to see corrections in action
#   - Compute and interpret effect sizes (Cohen's h, Cohen's d)
#
# PREREQUISITES: Complete Exercise 2 — you should understand MLE,
#   confidence intervals, and how to evaluate statistical models.
#
# ESTIMATED TIME: ~170 minutes
#
# TASKS:
#    1. Load A/B test data and perform sanity checks (SRM)
#    2. Bootstrap resampling from scratch — percentile and BCa CIs
#    3. Power analysis — minimum detectable effect (MDE)
#    4. Power curve — what power do we achieve at different effect sizes?
#    5. Primary hypothesis test — conversion rate (two-proportion z-test)
#    6. Multiple metrics testing (conversion, revenue, AOV, engagement)
#    7. Bonferroni correction (FWER control)
#    8. Benjamini-Hochberg correction (FDR control) with q-values
#    9. Permutation test — distribution-free alternative
#   10. False discovery rate simulation
#   11. Effect size interpretation (Cohen's h and d)
#   12. Visualise results and business interpretation
#
# DATASET: E-commerce A/B test — user-level conversion and revenue data
#   Source: Simulated from real e-commerce patterns
#
# THEORY:
#   - Bonferroni: α_adj = α/m (controls FWER)
#   - BH-FDR: rank p-values, reject if p(k) <= (k/m)α (controls FDR)
#   - Power = 1 - β = P(reject H₀ | H₁ true)
#   - MDE ≈ (z_{α/2} + z_β) × √(p(1-p)(1/n₁ + 1/n₂))
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
ab_data = loader.load("mlfp02", "experiment_data.parquet")

print("=" * 70)
print("  MLFP02 Exercise 3: Bootstrapping and Hypothesis Testing")
print("=" * 70)
print(f"\n  Data loaded: experiment_data.parquet")
print(f"  Shape: {ab_data.shape}")
print(f"  Columns: {ab_data.columns}")
print(ab_data.head(5))

control = ab_data.filter(pl.col("experiment_group") == "control")
treatment = ab_data.filter(pl.col("experiment_group") != "control")

if "converted" not in ab_data.columns:
    ab_data = ab_data.with_columns(
        (pl.col("metric_value") > 0).cast(pl.Int8).alias("converted")
    )
    control = ab_data.filter(pl.col("experiment_group") == "control")
    treatment = ab_data.filter(pl.col("experiment_group") != "control")

n_control = control.height
n_treatment = treatment.height
n_total = ab_data.height

print(f"\nControl:   n = {n_control:,}")
print(f"Treatment: n = {n_treatment:,}")
print(f"Total:     n = {n_total:,}")


# ══════════════════════════════════════════════════════════════════════
# TASK 1: Sanity Checks — Sample Ratio Mismatch (SRM)
# ══════════════════════════════════════════════════════════════════════
# SRM test: if the experiment was 50/50, do observed counts match?
# Uses χ² goodness-of-fit test. SRM indicates randomisation bugs.

expected_ratio = 0.5
expected_control = n_total * expected_ratio
expected_treatment = n_total * (1 - expected_ratio)

observed = np.array([n_control, n_treatment])
expected = np.array([expected_control, expected_treatment])

# TODO: Run chi-square goodness-of-fit test
chi2_stat, srm_p_value = ____  # Hint: stats.chisquare(observed, f_exp=expected)

print(f"\n=== Sample Ratio Mismatch Check ===")
print(f"Expected ratio: {expected_ratio:.0%} / {1 - expected_ratio:.0%}")
print(f"Observed ratio: {n_control / n_total:.4f} / {n_treatment / n_total:.4f}")
print(f"χ² statistic: {chi2_stat:.4f}")
print(f"p-value: {srm_p_value:.6f}")
if srm_p_value < 0.01:
    print("SRM DETECTED — investigate randomisation before trusting results!")
else:
    print("No SRM detected — sample split is consistent with design")

# ── Checkpoint 1 ─────────────────────────────────────────────────────
assert 0 <= srm_p_value <= 1, "SRM p-value must be between 0 and 1"
assert chi2_stat >= 0, "Chi-squared statistic must be non-negative"
assert n_control + n_treatment == n_total, "Control + treatment must equal total"
print("\n✓ Checkpoint 1 passed — SRM check completed\n")


# ══════════════════════════════════════════════════════════════════════
# TASK 2: Bootstrap Resampling from Scratch
# ══════════════════════════════════════════════════════════════════════
# Bootstrap: resample WITH REPLACEMENT, compute statistic, repeat.
# Three CI methods: percentile, normal, BCa (gold standard).

rng = np.random.default_rng(seed=42)
n_bootstrap = 10_000

ctrl_converted = control["converted"].to_numpy().astype(np.float64)
treat_converted = treatment["converted"].to_numpy().astype(np.float64)

# Manual bootstrap loop (from scratch)
boot_diffs = np.zeros(n_bootstrap)
boot_ctrl_rates = np.zeros(n_bootstrap)
boot_treat_rates = np.zeros(n_bootstrap)

for i in range(n_bootstrap):
    boot_ctrl = rng.choice(ctrl_converted, size=n_control, replace=True)
    boot_treat = rng.choice(treat_converted, size=n_treatment, replace=True)
    boot_ctrl_rates[i] = boot_ctrl.mean()
    boot_treat_rates[i] = boot_treat.mean()
    boot_diffs[i] = boot_treat.mean() - boot_ctrl.mean()

# TODO: Compute percentile CI from bootstrap distribution
pctile_ci = ____  # Hint: np.percentile(boot_diffs, [2.5, 97.5])

boot_se = boot_diffs.std()
# TODO: Compute observed conversion rate difference (treatment - control)
observed_diff = ____  # Hint: treat_converted.mean() - ctrl_converted.mean()
normal_boot_ci = (observed_diff - 1.96 * boot_se, observed_diff + 1.96 * boot_se)

# BCa CI using scipy (gold standard)
bca_result = stats.bootstrap(
    (treat_converted, ctrl_converted),
    statistic=lambda t, c: t.mean() - c.mean(),
    n_resamples=n_bootstrap,
    confidence_level=0.95,
    method="BCa",
    random_state=42,
)
bca_ci = (bca_result.confidence_interval.low, bca_result.confidence_interval.high)

print(f"\n=== Bootstrap CIs for Conversion Rate Difference ===")
print(f"Observed difference: {observed_diff:+.6f}")
print(f"Bootstrap SE: {boot_se:.6f}")
print(f"{'Method':<25} {'Lower':>12} {'Upper':>12} {'Width':>12}")
print("─" * 65)
print(
    f"{'Percentile CI':<25} {pctile_ci[0]:>12.6f} {pctile_ci[1]:>12.6f} {pctile_ci[1]-pctile_ci[0]:>12.6f}"
)
print(
    f"{'Normal Boot CI':<25} {normal_boot_ci[0]:>12.6f} {normal_boot_ci[1]:>12.6f} {normal_boot_ci[1]-normal_boot_ci[0]:>12.6f}"
)
print(
    f"{'BCa CI':<25} {bca_ci[0]:>12.6f} {bca_ci[1]:>12.6f} {bca_ci[1]-bca_ci[0]:>12.6f}"
)
# INTERPRETATION: BCa is gold standard as it corrects for both bias
# and acceleration. For symmetric statistics, all three agree.

# ── Checkpoint 2 ─────────────────────────────────────────────────────
assert len(boot_diffs) == n_bootstrap, "Should have n_bootstrap resamples"
assert pctile_ci[0] < pctile_ci[1], "CI lower must be below upper"
assert boot_se > 0, "Bootstrap SE must be positive"
print("\n✓ Checkpoint 2 passed — bootstrap CIs computed from scratch\n")


# ══════════════════════════════════════════════════════════════════════
# TASK 3: Power Analysis — Minimum Detectable Effect
# ══════════════════════════════════════════════════════════════════════
# MDE ≈ (z_{α/2} + z_β) × √(p(1-p)(1/n₁ + 1/n₂))

p_control = ctrl_converted.mean()
alpha = 0.05
power_target = 0.80
z_alpha_half = stats.norm.ppf(1 - alpha / 2)
z_beta = stats.norm.ppf(power_target)

pooled_se = np.sqrt(p_control * (1 - p_control) * (1 / n_control + 1 / n_treatment))
# TODO: Compute the Minimum Detectable Effect
mde = ____  # Hint: (z_alpha_half + z_beta) * pooled_se

print(f"\n=== Power Analysis ===")
print(f"Baseline conversion rate: {p_control:.4f} ({p_control:.2%})")
print(f"α = {alpha}, Power = {power_target:.0%}")
print(f"z_{{α/2}} = {z_alpha_half:.3f}, z_β = {z_beta:.3f}")
print(f"Minimum Detectable Effect (MDE): {mde:.6f} ({mde:.4%} absolute)")
print(f"Relative MDE: {mde / p_control:.2%} of baseline")

# ── Checkpoint 3 ─────────────────────────────────────────────────────
assert 0 < mde < 1, "MDE must be a valid proportion"
assert 0 < p_control < 1, "Baseline must be a valid proportion"
print("\n✓ Checkpoint 3 passed — MDE computed\n")


# ══════════════════════════════════════════════════════════════════════
# TASK 4: Power Curve — Power at Different Effect Sizes and Sample Sizes
# ══════════════════════════════════════════════════════════════════════

print(f"\n=== Power Curves ===")

effect_sizes = np.linspace(0, mde * 3, 100)
powers_by_effect = []
for delta in effect_sizes:
    ncp = delta / pooled_se
    power_val = (
        1 - stats.norm.cdf(z_alpha_half - ncp) + stats.norm.cdf(-z_alpha_half - ncp)
    )
    powers_by_effect.append(power_val)

print(f"\n--- Power by Effect Size (n={n_total:,}) ---")
for mult in [0.5, 1.0, 1.5, 2.0, 3.0]:
    idx = min(int(mult / 3 * 99), 99)
    print(f"  Effect = {effect_sizes[idx]:.4%}: Power = {powers_by_effect[idx]:.1%}")

sample_sizes_power = np.arange(500, n_total * 2, 500)
powers_by_n = []
for n_per in sample_sizes_power:
    se_n = np.sqrt(p_control * (1 - p_control) * 2 / n_per)
    ncp = mde / se_n
    power_val = (
        1 - stats.norm.cdf(z_alpha_half - ncp) + stats.norm.cdf(-z_alpha_half - ncp)
    )
    powers_by_n.append(power_val)

print(f"\n--- Power by Sample Size (effect = MDE = {mde:.4%}) ---")
for frac in [0.25, 0.5, 1.0, 1.5, 2.0]:
    idx = min(int(frac * n_total / 500) - 1, len(sample_sizes_power) - 1)
    idx = max(0, idx)
    print(f"  n = {sample_sizes_power[idx]:>8,}: Power = {powers_by_n[idx]:.1%}")

# ── Checkpoint 4 ─────────────────────────────────────────────────────
assert (
    powers_by_effect[-1] > powers_by_effect[0]
), "Power should increase with effect size"
assert len(powers_by_effect) == len(effect_sizes), "One power per effect size"
print("\n✓ Checkpoint 4 passed — power curves computed\n")


# ══════════════════════════════════════════════════════════════════════
# TASK 5: Primary Hypothesis Test — Conversion Rate
# ══════════════════════════════════════════════════════════════════════

p_treatment = treat_converted.mean()
p_pooled = ab_data["converted"].mean()

# TODO: Compute SE for two-proportion z-test using pooled proportion
se_diff = (
    ____  # Hint: np.sqrt(p_pooled * (1 - p_pooled) * (1 / n_control + 1 / n_treatment))
)
# TODO: Compute z-statistic = (p_treatment - p_control) / se_diff
z_stat = ____  # Hint: (p_treatment - p_control) / se_diff
# TODO: Compute two-tailed p-value from z-statistic
p_value_conversion = ____  # Hint: 2 * (1 - stats.norm.cdf(abs(z_stat)))

cohens_h = 2 * (np.arcsin(np.sqrt(p_treatment)) - np.arcsin(np.sqrt(p_control)))

print(f"\n=== Primary Metric: Conversion Rate ===")
print(f"Control:   {p_control:.4f} ({p_control:.2%})")
print(f"Treatment: {p_treatment:.4f} ({p_treatment:.2%})")
print(f"Absolute lift: {p_treatment - p_control:+.4f}")
print(f"Relative lift: {(p_treatment - p_control) / p_control:+.2%}")
print(f"z-statistic: {z_stat:.4f}")
print(f"p-value: {p_value_conversion:.6f}")
print(f"Cohen's h: {cohens_h:.4f}")
print(
    f"{'SIGNIFICANT' if p_value_conversion < alpha else 'NOT significant'} at α={alpha}"
)
# INTERPRETATION: p-value is NOT P(H₀ is true). Always report effect
# size alongside p-value. A tiny effect with huge n can be highly
# significant yet practically meaningless.

# ── Checkpoint 5 ─────────────────────────────────────────────────────
assert 0 <= p_value_conversion <= 1, "p-value must be between 0 and 1"
assert se_diff > 0, "SE of difference must be positive"
print("\n✓ Checkpoint 5 passed — primary test completed\n")


# ══════════════════════════════════════════════════════════════════════
# TASK 6: Multiple Metrics — Conversion, Revenue, AOV, Engagement
# ══════════════════════════════════════════════════════════════════════
# Testing multiple metrics inflates Type I error.
# P(at least one false positive) = 1 - (1-α)^m

metrics_results = {}

metrics_results["conversion_rate"] = {
    "control": p_control,
    "treatment": p_treatment,
    "stat": z_stat,
    "p_value": p_value_conversion,
    "test": "z-test",
}

rev_control = control["revenue"].to_numpy().astype(np.float64)
rev_treatment = treatment["revenue"].to_numpy().astype(np.float64)
u_stat, p_value_revenue = stats.mannwhitneyu(
    rev_treatment, rev_control, alternative="two-sided"
)
metrics_results["revenue_per_user"] = {
    "control": rev_control.mean(),
    "treatment": rev_treatment.mean(),
    "stat": u_stat,
    "p_value": p_value_revenue,
    "test": "Mann-Whitney U",
}

aov_ctrl = (
    control.filter(pl.col("converted") == 1)["revenue"].to_numpy().astype(np.float64)
)
aov_treat = (
    treatment.filter(pl.col("converted") == 1)["revenue"].to_numpy().astype(np.float64)
)
if len(aov_ctrl) > 1 and len(aov_treat) > 1:
    t_aov, p_aov = stats.ttest_ind(aov_treat, aov_ctrl, equal_var=False)
    metrics_results["avg_order_value"] = {
        "control": aov_ctrl.mean(),
        "treatment": aov_treat.mean(),
        "stat": t_aov,
        "p_value": p_aov,
        "test": "Welch's t-test",
    }

mv_ctrl = control["metric_value"].to_numpy().astype(np.float64)
mv_treat = treatment["metric_value"].to_numpy().astype(np.float64)
# TODO: Run Welch's t-test on metric_value (treatment vs control, unequal variances)
t_mv, p_mv = ____  # Hint: stats.ttest_ind(mv_treat, mv_ctrl, equal_var=False)
metrics_results["metric_value"] = {
    "control": mv_ctrl.mean(),
    "treatment": mv_treat.mean(),
    "stat": t_mv,
    "p_value": p_mv,
    "test": "Welch's t-test",
}

print(f"\n=== Multiple Metric Results (unadjusted) ===")
print(f"{'Metric':<20} {'Control':>12} {'Treatment':>12} {'p-value':>10} {'Sig':>4}")
print("─" * 65)
for name, r in metrics_results.items():
    sig = (
        "***"
        if r["p_value"] < 0.001
        else "**" if r["p_value"] < 0.01 else "*" if r["p_value"] < 0.05 else "ns"
    )
    print(
        f"{name:<20} {r['control']:>12.4f} {r['treatment']:>12.4f} {r['p_value']:>10.6f} {sig:>4}"
    )

m = len(metrics_results)
# TODO: Compute family-wise error rate for m simultaneous tests
fwer = ____  # Hint: 1 - (1 - alpha) ** m
print(f"\nNumber of tests (m): {m}")
print(f"Family-wise error rate (uncorrected): {fwer:.4f} ({fwer:.1%})")
# INTERPRETATION: With m tests at α=0.05, P(at least one false positive)
# exceeds 5%. Multiple testing corrections are essential.

# ── Checkpoint 6 ─────────────────────────────────────────────────────
assert m >= 3, "Should test at least 3 metrics"
assert fwer > alpha, "FWER must exceed single-test α"
print("\n✓ Checkpoint 6 passed — multiple metrics tested\n")


# ══════════════════════════════════════════════════════════════════════
# TASK 7: Bonferroni Correction (FWER Control)
# ══════════════════════════════════════════════════════════════════════
# Most conservative: α_adj = α/m

p_values = np.array([r["p_value"] for r in metrics_results.values()])
metric_names = list(metrics_results.keys())

# TODO: Compute Bonferroni-adjusted significance threshold
bonferroni_alpha = ____  # Hint: alpha / m
bonferroni_significant = p_values < bonferroni_alpha

print(f"\n=== Bonferroni Correction (FWER control) ===")
print(f"Adjusted α = {alpha}/{m} = {bonferroni_alpha:.4f}")
for i, name in enumerate(metric_names):
    sig = "SIGNIFICANT" if bonferroni_significant[i] else "not significant"
    print(f"  {name}: p={p_values[i]:.6f} -> {sig}")
print(f"\nBonferroni rejects: {sum(bonferroni_significant)}/{m}")

# ── Checkpoint 7 ─────────────────────────────────────────────────────
assert bonferroni_alpha < alpha, "Bonferroni α must be smaller than original α"
print("\n✓ Checkpoint 7 passed — Bonferroni correction applied\n")


# ══════════════════════════════════════════════════════════════════════
# TASK 8: Benjamini-Hochberg Correction (FDR Control)
# ══════════════════════════════════════════════════════════════════════
# Controls False Discovery Rate. Less conservative than Bonferroni.
# Procedure: sort p-values, find largest k where p(k) <= (k/m)α

sorted_indices = np.argsort(p_values)
sorted_p = p_values[sorted_indices]
# TODO: Compute BH thresholds for each rank k: (k+1)/m * alpha
bh_thresholds = ____  # Hint: np.array([(k + 1) / m * alpha for k in range(m)])

max_k = -1
for k in range(m):
    if sorted_p[k] <= bh_thresholds[k]:
        max_k = k

bh_significant_mask = np.zeros(m, dtype=bool)
if max_k >= 0:
    bh_significant_mask[sorted_indices[: max_k + 1]] = True

q_values = np.zeros(m)
sorted_q = np.zeros(m)
sorted_q[m - 1] = sorted_p[m - 1]
for k in range(m - 2, -1, -1):
    sorted_q[k] = min(sorted_p[k] * m / (k + 1), sorted_q[k + 1])
q_values[sorted_indices] = sorted_q

print(f"\n=== Benjamini-Hochberg Correction (FDR control) ===")
print(f"Target FDR: {alpha}")
print(
    f"{'Rank':>4} {'Metric':<20} {'p-value':>10} {'Threshold':>10} {'q-value':>10} {'Sig':>6}"
)
print("─" * 65)
for k in range(m):
    orig_idx = sorted_indices[k]
    name = metric_names[orig_idx]
    sig = "YES" if bh_significant_mask[orig_idx] else "no"
    print(
        f"{k+1:>4} {name:<20} {sorted_p[k]:>10.6f} {bh_thresholds[k]:>10.6f} {sorted_q[k]:>10.6f} {sig:>6}"
    )
print(f"\nBH-FDR rejects: {sum(bh_significant_mask)}/{m}")
# INTERPRETATION: BH-FDR allows more discoveries than Bonferroni.
# In exploratory settings, BH-FDR is preferred. In confirmatory settings,
# Bonferroni is safer.

# ── Checkpoint 8 ─────────────────────────────────────────────────────
assert all(0 <= q <= 1 for q in q_values), "All q-values must be valid"
assert all(q_values >= p_values - 1e-10), "q-values must be >= raw p-values"
print("\n✓ Checkpoint 8 passed — BH-FDR correction applied\n")


# ══════════════════════════════════════════════════════════════════════
# TASK 9: Permutation Test — Distribution-Free Alternative
# ══════════════════════════════════════════════════════════════════════
# Algorithm: pool, randomly split, compute statistic, repeat 10K times.

n_permutations = 10_000
all_converted = ab_data["converted"].to_numpy()

perm_diffs = np.zeros(n_permutations)
for i in range(n_permutations):
    perm = rng.permutation(all_converted)
    perm_ctrl_rate = perm[:n_control].mean()
    perm_treat_rate = perm[n_control:].mean()
    perm_diffs[i] = perm_treat_rate - perm_ctrl_rate

# TODO: Compute p-value as fraction of |permuted diffs| >= |observed|
perm_p_value = ____  # Hint: np.mean(np.abs(perm_diffs) >= np.abs(observed_diff))

print(f"\n=== Permutation Test (conversion rate) ===")
print(f"Observed difference: {observed_diff:+.6f}")
print(f"Permutation null: mean={perm_diffs.mean():.6f}, std={perm_diffs.std():.6f}")
print(f"Permutation p-value: {perm_p_value:.6f}")
print(f"Parametric p-value:  {p_value_conversion:.6f}")
print(
    f"Agreement: {'YES' if (perm_p_value < alpha) == (p_value_conversion < alpha) else 'NO'}"
)
# INTERPRETATION: The permutation test makes no distributional assumptions.
# Agreement with parametric tests builds confidence. Disagreement suggests
# parametric assumptions may be violated.

# ── Checkpoint 9 ─────────────────────────────────────────────────────
assert 0 <= perm_p_value <= 1, "Permutation p-value must be valid"
assert len(perm_diffs) == n_permutations, "Should have n_permutations samples"
assert abs(perm_diffs.mean()) < 0.01, "Permutation null should be centred near zero"
print("\n✓ Checkpoint 9 passed — permutation tests completed\n")


# ══════════════════════════════════════════════════════════════════════
# TASK 10: False Discovery Rate Simulation
# ══════════════════════════════════════════════════════════════════════
# Simulate 100 tests where we KNOW only 10 have real effects.

print(f"\n=== False Discovery Rate Simulation ===")

n_sim_tests = 100
n_sim_rounds = 500
n_truly_different = 10

bonf_false_discoveries = []
bh_false_discoveries = []
bonf_true_discoveries = []
bh_true_discoveries = []
raw_false_discoveries = []

for _ in range(n_sim_rounds):
    sim_p_values = np.zeros(n_sim_tests)
    is_true_null = np.ones(n_sim_tests, dtype=bool)

    for j in range(n_sim_tests):
        n_per = 500
        ctrl_sim = rng.binomial(1, 0.10, size=n_per)
        if j < n_truly_different:
            treat_sim = rng.binomial(1, 0.12, size=n_per)
            is_true_null[j] = False
        else:
            treat_sim = rng.binomial(1, 0.10, size=n_per)

        p_c = ctrl_sim.mean()
        p_t = treat_sim.mean()
        p_pool = (ctrl_sim.sum() + treat_sim.sum()) / (2 * n_per)
        se = np.sqrt(p_pool * (1 - p_pool) * 2 / n_per) if 0 < p_pool < 1 else 1e-10
        z = (p_t - p_c) / se if se > 0 else 0
        sim_p_values[j] = 2 * (1 - stats.norm.cdf(abs(z)))

    raw_sig = sim_p_values < alpha
    raw_false = raw_sig & is_true_null
    raw_false_discoveries.append(raw_false.sum())

    bonf_sig = sim_p_values < (alpha / n_sim_tests)
    bonf_false = bonf_sig & is_true_null
    bonf_true = bonf_sig & ~is_true_null
    bonf_false_discoveries.append(bonf_false.sum())
    bonf_true_discoveries.append(bonf_true.sum())

    sorted_idx = np.argsort(sim_p_values)
    sorted_pv = sim_p_values[sorted_idx]
    bh_thresh = np.array([(k + 1) / n_sim_tests * alpha for k in range(n_sim_tests)])
    mk = -1
    for k in range(n_sim_tests):
        if sorted_pv[k] <= bh_thresh[k]:
            mk = k
    bh_sig = np.zeros(n_sim_tests, dtype=bool)
    if mk >= 0:
        bh_sig[sorted_idx[: mk + 1]] = True
    bh_false = bh_sig & is_true_null
    bh_true = bh_sig & ~is_true_null
    bh_false_discoveries.append(bh_false.sum())
    bh_true_discoveries.append(bh_true.sum())

avg_raw_fd = np.mean(raw_false_discoveries)
avg_bonf_fd = np.mean(bonf_false_discoveries)
avg_bonf_td = np.mean(bonf_true_discoveries)
avg_bh_fd = np.mean(bh_false_discoveries)
avg_bh_td = np.mean(bh_true_discoveries)
bonf_fdr = (
    avg_bonf_fd / (avg_bonf_fd + avg_bonf_td) if (avg_bonf_fd + avg_bonf_td) > 0 else 0
)
bh_fdr = avg_bh_fd / (avg_bh_fd + avg_bh_td) if (avg_bh_fd + avg_bh_td) > 0 else 0

print(
    f"Simulation: {n_sim_tests} tests/round, {n_truly_different} truly different, {n_sim_rounds} rounds"
)
print(f"\n{'Method':<15} {'Avg False Disc':>15} {'Avg True Disc':>15} {'FDR':>8}")
print("─" * 55)
print(f"{'Raw (α=0.05)':<15} {avg_raw_fd:>15.1f} {'—':>15} {'—':>8}")
print(f"{'Bonferroni':<15} {avg_bonf_fd:>15.2f} {avg_bonf_td:>15.2f} {bonf_fdr:>8.3f}")
print(f"{'BH-FDR':<15} {avg_bh_fd:>15.2f} {avg_bh_td:>15.2f} {bh_fdr:>8.3f}")

# ── Checkpoint 10 ────────────────────────────────────────────────────
assert (
    avg_raw_fd > avg_bonf_fd
), "Bonferroni should have fewer false discoveries than raw"
assert avg_bh_td >= avg_bonf_td, "BH-FDR should find at least as many true effects"
print("\n✓ Checkpoint 10 passed — FDR simulation validates corrections\n")


# ══════════════════════════════════════════════════════════════════════
# TASK 11: Effect Size Interpretation
# ══════════════════════════════════════════════════════════════════════
# Statistical significance != practical significance.
# Cohen's h (proportions), Cohen's d (means): small=0.2, medium=0.5, large=0.8

print(f"\n=== Effect Size Interpretation ===")

for name, r in metrics_results.items():
    ctrl_val = r["control"]
    treat_val = r["treatment"]

    if name == "conversion_rate":
        effect = 2 * (np.arcsin(np.sqrt(treat_val)) - np.arcsin(np.sqrt(ctrl_val)))
        metric = "Cohen's h"
    else:
        if name == "revenue_per_user":
            s_pool = np.sqrt((rev_control.var(ddof=1) + rev_treatment.var(ddof=1)) / 2)
        elif name == "metric_value":
            s_pool = np.sqrt((mv_ctrl.var(ddof=1) + mv_treat.var(ddof=1)) / 2)
        else:
            s_pool = 1.0
        effect = (treat_val - ctrl_val) / s_pool if s_pool > 0 else 0
        metric = "Cohen's d"

    abs_effect = abs(effect)
    magnitude = (
        "negligible"
        if abs_effect < 0.2
        else "small" if abs_effect < 0.5 else "medium" if abs_effect < 0.8 else "large"
    )
    sig_str = "sig" if r["p_value"] < alpha else "ns"
    print(
        f"  {name:<20}: {metric}={effect:+.4f} ({magnitude}) | p={r['p_value']:.4f} ({sig_str})"
    )

print(f"\nKey: significant != practically important. Always report effect size.")

# ── Checkpoint 11 ────────────────────────────────────────────────────
assert cohens_h is not None, "Cohen's h must be computed"
print("\n✓ Checkpoint 11 passed — effect sizes computed and interpreted\n")


# ══════════════════════════════════════════════════════════════════════
# TASK 12: Visualise and Interpret
# ══════════════════════════════════════════════════════════════════════

viz = ModelVisualizer()

fig1 = viz.histogram(
    boot_diffs,
    title="Bootstrap Distribution: Conversion Rate Difference",
    x_label="Treatment - Control",
)
fig1.add_vline(x=observed_diff, line_dash="dash", annotation_text="Observed")
fig1.add_vline(x=0, line_dash="dot", line_color="red", annotation_text="H0: no effect")
fig1.write_html("ex3_bootstrap_distribution.html")
print("\nSaved: ex3_bootstrap_distribution.html")

fig2 = go.Figure()
fig2.add_trace(go.Scatter(x=effect_sizes, y=powers_by_effect, name="Power"))
fig2.add_hline(y=0.8, line_dash="dash", annotation_text="80% power target")
fig2.add_vline(x=mde, line_dash="dot", annotation_text=f"MDE={mde:.4f}")
fig2.update_layout(
    title="Statistical Power vs Effect Size",
    xaxis_title="Effect Size",
    yaxis_title="Power",
)
fig2.write_html("ex3_power_curve.html")
print("Saved: ex3_power_curve.html")

fig3 = viz.histogram(
    perm_diffs,
    title="Permutation Null Distribution: Conversion Rate Difference",
    x_label="Permuted Difference",
)
fig3.add_vline(x=observed_diff, line_dash="dash", annotation_text="Observed")
fig3.write_html("ex3_permutation_null.html")
print("Saved: ex3_permutation_null.html")

# ── Checkpoint 12 ────────────────────────────────────────────────────
print("\n✓ Checkpoint 12 passed — visualisations saved\n")


# ══════════════════════════════════════════════════════════════════════
# REFLECTION
# ══════════════════════════════════════════════════════════════════════
print(
    """
What you've mastered:
  ✓ Bootstrap from scratch: percentile and BCa CIs
  ✓ Power analysis and MDE for experiment design
  ✓ Two-proportion z-test, Mann-Whitney U, Welch's t-test
  ✓ Multiple testing: FWER vs FDR, Bonferroni vs BH-FDR
  ✓ Permutation test as a distribution-free alternative
  ✓ FDR simulation: proving corrections work as advertised
  ✓ Effect sizes: Cohen's h and d, practical vs statistical significance

Next: In Exercise 4, you'll apply this to full experiment design —
pre-registration, SRM detection, Welch-Satterthwaite CIs, and
adaptive sequential testing.
"""
)
