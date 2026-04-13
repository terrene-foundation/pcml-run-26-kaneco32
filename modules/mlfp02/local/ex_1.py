# Copyright 2026 Terrene Foundation
# SPDX-License-Identifier: Apache-2.0
"""
# ════════════════════════════════════════════════════════════════════════
# MLFP02 — Exercise 1: Probability and Bayesian Thinking
# ════════════════════════════════════════════════════════════════════════
#
# WHAT YOU'LL LEARN:
#   After completing this exercise, you will be able to:
#   - Construct truth tables and compute joint/conditional probabilities
#     from real HDB transaction data
#   - Apply Bayes' theorem to real-world scenarios (medical tests, property
#     valuation)
#   - Compute MLE for Normal distribution parameters and quantify estimation
#     uncertainty via the Cramér-Rao bound
#   - Implement Normal-Normal and Beta-Binomial conjugate priors and derive
#     posterior distributions analytically
#   - Run prior sensitivity analysis and compare credible vs confidence
#     intervals with a repeated-sampling simulation
#   - Compute expected value and demonstrate sampling bias (friendship paradox)
#   - Visualise prior, likelihood, and posterior using ModelVisualizer
#
# PREREQUISITES: Complete M1 — you should be comfortable loading data,
#   computing summary statistics, and reading Polars DataFrames.
#
# ESTIMATED TIME: ~170 minutes
#
# TASKS:
#    1. Load data, compute probability fundamentals (truth tables, joint probs)
#    2. Compute MLE for Normal parameters with Cramér-Rao bound
#    3. Bayes' theorem applied — medical test and HDB valuation scenarios
#    4. Normal-Normal conjugate prior: derive and compute posterior
#    5. Prior sensitivity analysis — sweep prior hyperparameters
#    6. Beta-Binomial conjugate: model HDB transaction success rates
#    7. Credible vs confidence interval — repeated-sampling simulation
#    8. Expected value and sampling bias (friendship paradox simulation)
#    9. Bootstrap confidence intervals (percentile + BCa) for comparison
#   10. Bayesian estimation across flat types — compare data vs prior balance
#   11. Visualise all results with ModelVisualizer
#   12. Business interpretation synthesis
#
# DATASET: HDB resale flat transactions (Singapore)
#   Source: data.gov.sg — public housing resale records
#   Key column: resale_price (SGD)
#
# THEORY:
#   Normal-Normal conjugate: prior μ ~ N(μ₀, σ₀²), likelihood x ~ N(μ, σ²)
#   Posterior: μ|x ~ N(μₙ, σₙ²) where:
#     μₙ = (μ₀/σ₀² + n*x̄/σ²) / (1/σ₀² + n/σ²)
#     σₙ² = 1 / (1/σ₀² + n/σ²)
#
#   Beta-Binomial conjugate: prior p ~ Beta(α, β), likelihood x ~ Binomial(n,p)
#   Posterior: p|x ~ Beta(α + k, β + n - k) where k = number of successes
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
hdb = loader.load("mlfp01", "hdb_resale.parquet")

print("=" * 70)
print("  MLFP02 Exercise 1: Probability and Bayesian Thinking")
print("=" * 70)

# Focus on a specific flat type and recent period for clearer analysis
hdb_recent = hdb.filter(
    (pl.col("flat_type") == "4 ROOM")
    & (pl.col("month").str.to_date("%Y-%m") >= pl.date(2020, 1, 1))
)

prices = hdb_recent["resale_price"].to_numpy().astype(np.float64)
print(f"\n  Data loaded: {len(prices):,} 4-room HDB transactions (2020+)")
print(f"  Price range: ${prices.min():,.0f} – ${prices.max():,.0f}")
print(f"  Sample mean: ${prices.mean():,.0f}")
print(f"  Sample std:  ${prices.std():,.0f}\n")

# ══════════════════════════════════════════════════════════════════════
# BAYESIAN THINKING — THE INTUITION (before the math)
# ══════════════════════════════════════════════════════════════════════
# Imagine you're estimating the price of a 4-room HDB flat. You have:
#   1. A PRIOR belief: "I've been watching the market — ~$500K"
#   2. NEW DATA: 50 recent transactions, compute the sample mean
#   3. A POSTERIOR belief: principled combination of prior + data
#
# With LITTLE data, your prior dominates. With LOTS of data, data wins.
# ══════════════════════════════════════════════════════════════════════


# ══════════════════════════════════════════════════════════════════════
# TASK 1: Probability Fundamentals — Truth Tables and Joint Probabilities
# ══════════════════════════════════════════════════════════════════════
# Key rules:
#   P(A) + P(A') = 1
#   P(A,B) = P(A) × P(B|A)
#   Independent events: P(A,B) = P(A) × P(B)

print("\n" + "=" * 70)
print("TASK 1: Probability Fundamentals")
print("=" * 70)

hdb_all = hdb.filter(pl.col("month").str.to_date("%Y-%m") >= pl.date(2020, 1, 1))
total_n = hdb_all.height

# Event A: transaction is a 4-room flat
n_4room = hdb_all.filter(pl.col("flat_type") == "4 ROOM").height
# TODO: Compute P(4-room) = n_4room / total_n
p_4room = ____  # Hint: n_4room / total_n

# Event B: price above $500K
n_above_500k = hdb_all.filter(pl.col("resale_price") > 500_000).height
# TODO: Compute P(price > $500K)
p_above_500k = ____  # Hint: n_above_500k / total_n

# Joint probability: P(4-room AND price > 500K)
n_4room_and_above = hdb_all.filter(
    (pl.col("flat_type") == "4 ROOM") & (pl.col("resale_price") > 500_000)
).height
# TODO: Compute the joint probability P(A and B)
p_joint = ____  # Hint: n_4room_and_above / total_n

# Conditional: P(price > 500K | 4-room) = P(A,B) / P(A)
# TODO: Compute conditional probability
p_above_given_4room = ____  # Hint: n_4room_and_above / n_4room if n_4room > 0 else 0

# Test independence: compare P(A,B) vs P(A)*P(B)
p_independent = p_4room * p_above_500k

print(f"\n--- Truth Table (empirical) ---")
print(f"Total transactions (2020+): {total_n:,}")
print(f"P(4-room)           = {p_4room:.4f} ({p_4room:.1%})")
print(f"P(price > $500K)    = {p_above_500k:.4f} ({p_above_500k:.1%})")
print(f"P(4-room AND >$500K)= {p_joint:.4f} ({p_joint:.1%})")
print(f"P(>$500K | 4-room)  = {p_above_given_4room:.4f} ({p_above_given_4room:.1%})")
print(f"\n--- Independence Check ---")
print(f"P(A)xP(B) = {p_independent:.4f}")
print(f"P(A,B)    = {p_joint:.4f}")
if abs(p_joint - p_independent) < 0.01:
    print("Events are approximately independent")
else:
    print("Events are NOT independent — flat type affects price probability")
# INTERPRETATION: If flat type and price are not independent, knowing
# the flat type tells you something about the price distribution.

price_cats = hdb_all.with_columns(
    pl.when(pl.col("resale_price") <= 400_000)
    .then(pl.lit("<=400K"))
    .when(pl.col("resale_price") <= 600_000)
    .then(pl.lit("400K-600K"))
    .when(pl.col("resale_price") <= 800_000)
    .then(pl.lit("600K-800K"))
    .otherwise(pl.lit(">800K"))
    .alias("price_band")
)
cross_tab = (
    price_cats.group_by("flat_type", "price_band")
    .agg(pl.len().alias("count"))
    .sort("flat_type", "price_band")
)
print(f"\n--- Cross-Tabulation (sample) ---")
print(cross_tab.head(12))

# ── Checkpoint 1 ─────────────────────────────────────────────────────
assert 0 < p_4room < 1, "P(4-room) must be a valid probability"
assert 0 < p_above_500k < 1, "P(>500K) must be a valid probability"
assert p_joint <= min(p_4room, p_above_500k), "Joint prob cannot exceed marginals"
assert (
    abs(p_above_given_4room - p_joint / p_4room) < 1e-10
), "Conditional probability identity must hold: P(B|A) = P(A,B)/P(A)"
print("\n✓ Checkpoint 1 passed — probability fundamentals computed\n")


# ══════════════════════════════════════════════════════════════════════
# TASK 2: Maximum Likelihood Estimation (MLE)
# ══════════════════════════════════════════════════════════════════════
# For X ~ N(μ, σ²): MLE gives μ̂ = x̄, σ̂² = (1/n)Σ(xᵢ - x̄)²
# Fisher information for Normal: I(μ) = n/σ² → Var(μ̂) ≥ σ²/n

n = len(prices)
mle_mean = prices.mean()
# TODO: Compute MLE variance using ddof=0 (biased estimator)
mle_var = ____  # Hint: prices.var(ddof=0)
mle_std = np.sqrt(mle_var)

# Bessel's correction: unbiased variance uses ddof=1
unbiased_var = prices.var(ddof=1)
unbiased_std = np.sqrt(unbiased_var)

# TODO: Compute Fisher information = n / mle_var
fisher_info = ____  # Hint: n / mle_var
cramer_rao_bound = 1 / fisher_info
mle_se = np.sqrt(cramer_rao_bound)

print(f"\n=== MLE Estimates ===")
print(f"μ̂ = ${mle_mean:,.0f}")
print(f"σ̂ (MLE, ddof=0)     = ${mle_std:,.0f}")
print(f"σ̂ (unbiased, ddof=1) = ${unbiased_std:,.0f}")
print(f"Bias: MLE σ underestimates by ${unbiased_std - mle_std:,.2f}")
print(f"\nFisher information I(μ) = {fisher_info:.4f}")
print(f"Cramér-Rao lower bound: Var(μ̂) >= {cramer_rao_bound:.2f}")
print(f"MLE standard error: ${mle_se:,.2f}")

# ── Checkpoint 2 ─────────────────────────────────────────────────────
assert n > 0, "No data loaded"
assert mle_mean > 0, "MLE mean should be positive"
assert mle_std > 0, "MLE std should be positive"
assert mle_se > 0, "Standard error should be positive"
assert mle_se < mle_std, "SE of mean should be much smaller than std of prices"
assert unbiased_std > mle_std, "Unbiased σ must be > MLE σ (Bessel's correction)"
print("\n✓ Checkpoint 2 passed — MLE estimates and Cramér-Rao bound computed\n")


# ══════════════════════════════════════════════════════════════════════
# TASK 3: Bayes' Theorem — Real-World Applications
# ══════════════════════════════════════════════════════════════════════
# Bayes' theorem: P(A|B) = P(B|A) × P(A) / P(B)

print(f"\n=== Bayes' Theorem Application 1: Medical Test (COVID ART) ===")

sensitivity = 0.85   # P(positive test | infected)
specificity = 0.995  # P(negative test | not infected)
prevalence = 0.02    # P(infected) — base rate in Singapore community

# P(positive test) = P(+|infected)P(infected) + P(+|not infected)P(not infected)
# TODO: Compute marginal probability of a positive test
p_positive = ____  # Hint: sensitivity * prevalence + (1 - specificity) * (1 - prevalence)

# TODO: Apply Bayes' theorem: P(infected | +test) = P(+|infected)*P(infected)/P(+)
p_infected_given_positive = ____  # Hint: (sensitivity * prevalence) / p_positive

p_false_positive = 1 - p_infected_given_positive

print(f"Sensitivity: {sensitivity:.1%}  Specificity: {specificity:.1%}  Prevalence: {prevalence:.1%}")
print(f"P(positive test)           = {p_positive:.4f} ({p_positive:.2%})")
print(f"P(infected | positive test) = {p_infected_given_positive:.4f} ({p_infected_given_positive:.1%})")
print(f"P(false positive)           = {p_false_positive:.4f} ({p_false_positive:.1%})")
# INTERPRETATION: Even with 99.5% specificity, when prevalence is 2%, a
# positive test may still mostly indicate false positives. This is the
# base rate fallacy — ignoring prevalence leads to overconfidence.

print(f"\n--- Effect of Prevalence on P(infected | +test) ---")
for prev in [0.001, 0.01, 0.02, 0.05, 0.10, 0.20, 0.50]:
    p_pos = sensitivity * prev + (1 - specificity) * (1 - prev)
    p_inf = (sensitivity * prev) / p_pos
    print(f"  Prevalence {prev:>5.1%} -> P(infected | +) = {p_inf:.1%}")

print(f"\n=== Bayes' Theorem Application 2: HDB Valuation ===")
bishan_flats = hdb_recent.filter(pl.col("town") == "BISHAN")
if bishan_flats.height > 0:
    p_above_600k_bishan = (
        bishan_flats.filter(pl.col("resale_price") > 600_000).height
        / bishan_flats.height
    )
    mean_bishan = bishan_flats["resale_price"].mean()
    print(f"Bishan 4-room data: {bishan_flats.height} transactions")
    print(f"Mean price: ${mean_bishan:,.0f}")
    print(f"P(price > $600K | Bishan 4-room) = {p_above_600k_bishan:.2%}")
else:
    p_above_600k_bishan = 0.5
    print("No Bishan 4-room data found — using uninformative prior")

# ── Checkpoint 3 ─────────────────────────────────────────────────────
assert 0 < p_infected_given_positive < 1, "Posterior probability must be valid"
assert (
    p_infected_given_positive > prevalence
), "Positive test must increase probability of infection above base rate"
print("\n✓ Checkpoint 3 passed — Bayes' theorem applications computed\n")


# ══════════════════════════════════════════════════════════════════════
# TASK 4: Normal-Normal Conjugate Prior — Posterior Distribution
# ══════════════════════════════════════════════════════════════════════
# Prior: μ ~ N(μ₀, σ₀²), Known σ² (plug-in from MLE)
# Posterior precision = prior precision + data precision

mu_0 = 500_000.0   # Prior mean: $500K
sigma_0 = 100_000.0  # Prior std: moderate uncertainty
sigma_known = mle_std

precision_prior = 1.0 / sigma_0**2
precision_data = n / sigma_known**2
precision_posterior = precision_prior + precision_data
sigma_n_sq = 1.0 / precision_posterior
sigma_n = np.sqrt(sigma_n_sq)

# TODO: Compute posterior mean (precision-weighted combination of prior and data)
# Formula: mu_n = sigma_n_sq * (mu_0 * precision_prior + n * mle_mean / sigma_known**2)
mu_n = ____  # Hint: sigma_n_sq * (mu_0 * precision_prior + n * mle_mean / sigma_known**2)

print(f"\n=== Normal-Normal Conjugate Posterior ===")
print(f"Prior: μ ~ N(μ₀={mu_0:,.0f}, σ₀={sigma_0:,.0f})")
print(f"Posterior: μ|data ~ N(μₙ={mu_n:,.0f}, σₙ={sigma_n:,.2f})")
print(f"Data-to-prior precision ratio: {precision_data / precision_prior:.0f}x")
print(f"  -> Posterior is dominated by {'data' if precision_data > precision_prior else 'prior'}")

# TODO: Compute 95% credible interval bounds
ci_95_lower = ____  # Hint: mu_n - 1.96 * sigma_n
ci_95_upper = ____  # Hint: mu_n + 1.96 * sigma_n
print(f"\n95% Bayesian credible interval: [${ci_95_lower:,.2f}, ${ci_95_upper:,.2f}]")
# INTERPRETATION: A credible interval has a direct probability statement:
# "Given the data, there is a 95% probability that true mean price
# lies in this range." Different from the frequentist CI interpretation.

# ── Checkpoint 4 ─────────────────────────────────────────────────────
assert precision_data > 0, "Data precision must be positive"
assert sigma_n < sigma_0, "Posterior std should be narrower than prior"
assert ci_95_lower < mu_n < ci_95_upper, "Posterior mean must be within its own CI"
assert abs(mu_n - mle_mean) < sigma_0, "Posterior should be close to MLE with large n"
print("\n✓ Checkpoint 4 passed — Normal-Normal posterior computed\n")


# ══════════════════════════════════════════════════════════════════════
# TASK 5: Prior Sensitivity Analysis
# ══════════════════════════════════════════════════════════════════════
# How does the posterior change as we vary the prior hyperparameters?
# Sweep prior mean from $300K to $700K and prior std from $20K to $200K.

print(f"\n=== Prior Sensitivity Analysis ===")
print(f"\n--- Varying prior mean (σ₀ = ${sigma_0:,.0f} fixed) ---")
print(f"{'Prior μ₀':>12} {'Posterior μₙ':>15} {'Shift from MLE':>16} {'Prior Weight':>13}")
print("─" * 60)
for mu_sweep in [300_000, 400_000, 500_000, 600_000, 700_000]:
    prec_pr = 1.0 / sigma_0**2
    prec_dt = n / sigma_known**2
    prec_post = prec_pr + prec_dt
    # TODO: Compute posterior mean for this swept prior mean
    mu_post = ____  # Hint: (mu_sweep * prec_pr + n * mle_mean / sigma_known**2) / prec_post
    prior_wt = prec_pr / prec_post * 100
    print(
        f"${mu_sweep:>10,.0f}  ${mu_post:>13,.0f}  {mu_post - mle_mean:>+14,.0f}  {prior_wt:>11.4f}%"
    )

print(f"\n--- Varying prior std (μ₀ = ${mu_0:,.0f} fixed) ---")
print(f"{'Prior σ₀':>12} {'Posterior μₙ':>15} {'Prior Weight':>13}")
print("─" * 45)
for sigma_sweep in [20_000, 50_000, 100_000, 200_000, 500_000]:
    prec_pr = 1.0 / sigma_sweep**2
    prec_dt = n / sigma_known**2
    prec_post = prec_pr + prec_dt
    mu_post = (mu_0 * prec_pr + n * mle_mean / sigma_known**2) / prec_post
    prior_wt = prec_pr / prec_post * 100
    print(f"${sigma_sweep:>10,.0f}  ${mu_post:>13,.0f}  {prior_wt:>11.4f}%")
# INTERPRETATION: Even a very opinionated prior gets overwhelmed by large n.
# A sensitivity analysis shows stakeholders conclusions are robust to priors.

# ── Checkpoint 5 ─────────────────────────────────────────────────────
for mu_test in [300_000, 700_000]:
    prec_pr = 1.0 / sigma_0**2
    prec_dt = n / sigma_known**2
    mu_post_test = (mu_test * prec_pr + n * mle_mean / sigma_known**2) / (
        prec_pr + prec_dt
    )
    assert (
        abs(mu_post_test - mle_mean) < 5000
    ), "With large n, posterior should be near MLE regardless of prior mean"
print("\n✓ Checkpoint 5 passed — prior sensitivity demonstrates data dominance\n")


# ══════════════════════════════════════════════════════════════════════
# TASK 6: Beta-Binomial Conjugate — Transaction Success Rates
# ══════════════════════════════════════════════════════════════════════
# Prior: p ~ Beta(α, β), E[p] = α/(α+β)
# Posterior: p | data ~ Beta(α + k, β + n - k)

print(f"\n=== Beta-Binomial Conjugate ===")

threshold = 500_000
k_success = int((prices > threshold).sum())
n_trials = len(prices)
empirical_rate = k_success / n_trials

print(f"Successes (price > ${threshold:,}): {k_success:,} / {n_trials:,} = {empirical_rate:.2%}")

alpha_prior = 2.0
beta_prior = 2.0
prior_mean = alpha_prior / (alpha_prior + beta_prior)

# TODO: Compute posterior parameters for the weak prior
# Posterior update: add k successes to alpha, add (n - k) failures to beta
alpha_post_weak = ____  # Hint: alpha_prior + k_success
beta_post_weak = ____   # Hint: beta_prior + (n_trials - k_success)
post_mean_weak = alpha_post_weak / (alpha_post_weak + beta_post_weak)

alpha_strong = 20.0
beta_strong = 80.0
alpha_post_strong = alpha_strong + k_success
beta_post_strong = beta_strong + (n_trials - k_success)
post_mean_strong = alpha_post_strong / (alpha_post_strong + beta_post_strong)

ci_weak = stats.beta.ppf([0.025, 0.975], alpha_post_weak, beta_post_weak)
ci_strong = stats.beta.ppf([0.025, 0.975], alpha_post_strong, beta_post_strong)

print(f"\n--- Weak Prior: Beta({alpha_prior}, {beta_prior}), E[p]={prior_mean:.2f} ---")
print(f"Posterior: Beta({alpha_post_weak:.0f}, {beta_post_weak:.0f})")
print(f"Posterior mean: {post_mean_weak:.4f} ({post_mean_weak:.2%})")
print(f"95% CI: [{ci_weak[0]:.4f}, {ci_weak[1]:.4f}]")
print(f"Empirical rate: {empirical_rate:.4f}")
# INTERPRETATION: Large n overwhelms even a strong prior (Beta(20,80)).
# Use informative priors when you have domain knowledge and small samples.

# ── Checkpoint 6 ─────────────────────────────────────────────────────
assert alpha_post_weak == alpha_prior + k_success, "Posterior alpha update incorrect"
assert beta_post_weak == beta_prior + (
    n_trials - k_success
), "Posterior beta update incorrect"
assert ci_weak[0] < post_mean_weak < ci_weak[1], "Posterior mean must be within CI"
assert (
    abs(post_mean_weak - empirical_rate) < 0.01
), "With weak prior and large n, posterior mean should be near empirical rate"
print("\n✓ Checkpoint 6 passed — Beta-Binomial conjugate computed\n")


# ══════════════════════════════════════════════════════════════════════
# TASK 7: Credible vs Confidence Interval — Repeated Sampling Simulation
# ══════════════════════════════════════════════════════════════════════
# Frequentist CI: "95% of intervals from repeated sampling contain true μ."
# Bayesian credible: "95% probability that μ lies in THIS interval."

print(f"\n=== Credible vs Confidence Interval Simulation ===")

rng = np.random.default_rng(seed=42)
true_mu = mle_mean
true_sigma = mle_std
n_simulations = 1000
sample_size = 100

freq_covers = 0
bayes_covers = 0
freq_widths = []
bayes_widths = []

for _ in range(n_simulations):
    sample = rng.normal(true_mu, true_sigma, size=sample_size)
    xbar = sample.mean()
    se = sample.std(ddof=1) / np.sqrt(sample_size)

    freq_lower = xbar - 1.96 * se
    freq_upper = xbar + 1.96 * se
    if freq_lower <= true_mu <= freq_upper:
        freq_covers += 1
    freq_widths.append(freq_upper - freq_lower)

    prec_pr = 1.0 / sigma_0**2
    prec_dt = sample_size / sample.std(ddof=0) ** 2
    prec_post = prec_pr + prec_dt
    sigma_post = np.sqrt(1.0 / prec_post)
    mu_post = (
        mu_0 * prec_pr + sample_size * xbar / sample.std(ddof=0) ** 2
    ) / prec_post
    bayes_lower = mu_post - 1.96 * sigma_post
    bayes_upper = mu_post + 1.96 * sigma_post
    if bayes_lower <= true_mu <= bayes_upper:
        bayes_covers += 1
    bayes_widths.append(bayes_upper - bayes_lower)

freq_coverage = freq_covers / n_simulations
bayes_coverage = bayes_covers / n_simulations

print(f"Simulations: {n_simulations:,}, sample size: {sample_size}")
print(f"Frequentist CI coverage: {freq_coverage:.1%} (target: 95%)")
print(f"Bayesian credible coverage: {bayes_coverage:.1%}")
print(f"Mean freq CI width: ${np.mean(freq_widths):,.0f}")
print(f"Mean Bayes CI width: ${np.mean(bayes_widths):,.0f}")

# ── Checkpoint 7 ─────────────────────────────────────────────────────
assert (
    0.90 < freq_coverage < 1.0
), f"Frequentist coverage should be near 95%, got {freq_coverage:.1%}"
assert (
    0.90 < bayes_coverage < 1.0
), f"Bayesian coverage should be near 95%, got {bayes_coverage:.1%}"
print("\n✓ Checkpoint 7 passed — coverage simulation validates both methods\n")


# ══════════════════════════════════════════════════════════════════════
# TASK 8: Expected Value and Sampling Bias (Friendship Paradox)
# ══════════════════════════════════════════════════════════════════════
# E[X] = Σ pᵢ × xᵢ
# Sampling bias: popular people appear in more friend lists.

print(f"\n=== Expected Value: HDB Price by Flat Type ===")

flat_type_stats = (
    hdb_all.group_by("flat_type")
    .agg(
        pl.col("resale_price").mean().alias("mean_price"),
        pl.len().alias("count"),
    )
    .sort("flat_type")
)
total_transactions = flat_type_stats["count"].sum()
flat_type_stats = flat_type_stats.with_columns(
    (pl.col("count") / total_transactions).alias("probability")
)
print(flat_type_stats)

# TODO: Compute transaction-weighted expected price: E[price] = Σ P(type) × E[price|type]
expected_price = ____  # Hint: (flat_type_stats["probability"].to_numpy() * flat_type_stats["mean_price"].to_numpy()).sum()
print(f"\nE[price] (transaction-weighted) = ${expected_price:,.0f}")
print(f"Simple average across types     = ${flat_type_stats['mean_price'].mean():,.0f}")
print(f"Difference: ${expected_price - flat_type_stats['mean_price'].mean():,.0f}")

print(f"\n=== Sampling Bias: Friendship Paradox ===")
n_people = 200
degrees = rng.zipf(a=2.0, size=n_people).clip(max=n_people - 1)
avg_degree = degrees.mean()

friend_degrees = []
for person_idx in range(n_people):
    if degrees[person_idx] > 0:
        friend_probs = degrees / degrees.sum()
        friend_idx = rng.choice(n_people, p=friend_probs)
        friend_degrees.append(degrees[friend_idx])

avg_friend_degree = np.mean(friend_degrees)
print(f"Your average number of friends: {avg_degree:.1f}")
print(f"Your friends' average number of friends: {avg_friend_degree:.1f}")
print(f"-> Your friends have {avg_friend_degree / avg_degree:.1f}x more friends than you!")
# INTERPRETATION: Sampling bias affects product reviews, survey responses,
# and click-through rates. Popular items appear disproportionately.

# ── Checkpoint 8 ─────────────────────────────────────────────────────
assert avg_friend_degree > avg_degree, "Friends should have more friends (paradox)"
assert expected_price > 0, "Expected price must be positive"
print("\n✓ Checkpoint 8 passed — expected value and sampling bias demonstrated\n")


# ══════════════════════════════════════════════════════════════════════
# TASK 9: Bootstrap Confidence Intervals for Comparison
# ══════════════════════════════════════════════════════════════════════
# Non-parametric bootstrap: resample with replacement, compute statistic.
# BCa (bias-corrected accelerated) is the gold standard.

n_bootstrap = 10_000

bootstrap_means = np.array(
    [rng.choice(prices, size=n, replace=True).mean() for _ in range(n_bootstrap)]
)

# TODO: Compute percentile CI bounds at 2.5th and 97.5th percentiles
boot_ci_lower = ____  # Hint: np.percentile(bootstrap_means, 2.5)
boot_ci_upper = ____  # Hint: np.percentile(bootstrap_means, 97.5)

bca_result = stats.bootstrap(
    (prices,),
    statistic=np.mean,
    n_resamples=n_bootstrap,
    confidence_level=0.95,
    method="BCa",
    random_state=42,
)
bca_ci_lower = bca_result.confidence_interval.low
bca_ci_upper = bca_result.confidence_interval.high

normal_ci_lower = mle_mean - 1.96 * mle_se
normal_ci_upper = mle_mean + 1.96 * mle_se

print(f"\n=== Confidence / Credible Intervals Comparison ===")
print(f"{'Method':<25} {'Lower':>14} {'Upper':>14} {'Width':>12}")
print("─" * 70)
print(f"{'Normal theory 95% CI':<25} ${normal_ci_lower:>12,.2f} ${normal_ci_upper:>12,.2f} ${normal_ci_upper - normal_ci_lower:>10,.2f}")
print(f"{'Bootstrap percentile CI':<25} ${boot_ci_lower:>12,.2f} ${boot_ci_upper:>12,.2f} ${boot_ci_upper - boot_ci_lower:>10,.2f}")
print(f"{'Bootstrap BCa CI':<25} ${bca_ci_lower:>12,.2f} ${bca_ci_upper:>12,.2f} ${bca_ci_upper - bca_ci_lower:>10,.2f}")
print(f"{'Bayesian 95% credible':<25} ${ci_95_lower:>12,.2f} ${ci_95_upper:>12,.2f} ${ci_95_upper - ci_95_lower:>10,.2f}")

# ── Checkpoint 9 ─────────────────────────────────────────────────────
assert boot_ci_lower < boot_ci_upper, "Bootstrap CI lower must be below upper"
assert bca_ci_lower < bca_ci_upper, "BCa CI lower must be below upper"
assert normal_ci_lower < mle_mean < normal_ci_upper, "MLE mean within normal CI"
assert len(bootstrap_means) == n_bootstrap, "Should have n_bootstrap samples"
print("\n✓ Checkpoint 9 passed — bootstrap CIs computed\n")


# ══════════════════════════════════════════════════════════════════════
# TASK 10: Bayesian Estimation Across Flat Types
# ══════════════════════════════════════════════════════════════════════
# Apply Normal-Normal conjugate to each flat type.
# Less data → prior matters more (Bayesian regularisation).

flat_types = ["2 ROOM", "3 ROOM", "4 ROOM", "5 ROOM", "EXECUTIVE"]

results_by_type = {}
for ft in flat_types:
    subset = hdb.filter(
        (pl.col("flat_type") == ft)
        & (pl.col("month").str.to_date("%Y-%m") >= pl.date(2020, 1, 1))
    )
    if subset.height == 0:
        continue

    p = subset["resale_price"].to_numpy().astype(np.float64)
    n_ft = len(p)
    xbar = p.mean()
    s = p.std()

    prec_prior = 1.0 / sigma_0**2
    prec_data = n_ft / s**2
    prec_post = prec_prior + prec_data
    # TODO: Compute posterior mean for this flat type
    mu_post = ____  # Hint: (mu_0 * prec_prior + n_ft * xbar / s**2) / prec_post
    sigma_post = np.sqrt(1.0 / prec_post)

    results_by_type[ft] = {
        "n": n_ft,
        "mle_mean": xbar,
        "posterior_mean": mu_post,
        "posterior_std": sigma_post,
        "prior_weight": (prec_prior / prec_post) * 100,
        "ci_lower": mu_post - 1.96 * sigma_post,
        "ci_upper": mu_post + 1.96 * sigma_post,
    }

print(f"\n=== Bayesian Estimates by Flat Type ===")
print(f"{'Type':<12} {'n':>8} {'MLE Mean':>12} {'Post Mean':>12} {'Post σ':>10} {'Prior %':>8}")
print("─" * 70)
for ft, r in results_by_type.items():
    print(
        f"{ft:<12} {r['n']:>8,} ${r['mle_mean']:>10,.0f} "
        f"${r['posterior_mean']:>10,.0f} ${r['posterior_std']:>8,.2f} {r['prior_weight']:>7.3f}%"
    )

# ── Checkpoint 10 ────────────────────────────────────────────────────
for ft, r in results_by_type.items():
    assert r["posterior_std"] < sigma_0, f"{ft}: posterior std should shrink from prior"
    assert 0 < r["prior_weight"] < 100, f"{ft}: prior weight must be between 0 and 100%"
print("\n✓ Checkpoint 10 passed — flat type posteriors all valid\n")


# ══════════════════════════════════════════════════════════════════════
# TASK 11: Visualise with ModelVisualizer
# ══════════════════════════════════════════════════════════════════════

viz = ModelVisualizer()

x_range = np.linspace(mu_0 - 3 * sigma_0, mu_0 + 3 * sigma_0, 500)
prior_pdf = stats.norm.pdf(x_range, mu_0, sigma_0)
x_posterior = np.linspace(mu_n - 5 * sigma_n, mu_n + 5 * sigma_n, 500)
posterior_pdf = stats.norm.pdf(x_posterior, mu_n, sigma_n)

fig1 = make_subplots(
    rows=1, cols=2, subplot_titles=["Prior Distribution", "Posterior Distribution"]
)
fig1.add_trace(
    go.Scatter(x=x_range, y=prior_pdf, name="Prior N(500K, 100K²)", line={"color": "blue"}),
    row=1, col=1,
)
fig1.add_trace(
    go.Scatter(
        x=x_posterior, y=posterior_pdf,
        name=f"Posterior N({mu_n:,.0f}, {sigma_n:,.0f}²)", line={"color": "red"},
    ),
    row=1, col=2,
)
fig1.update_layout(title="Prior vs Posterior: 4-Room HDB Mean Price", height=400)
fig1.write_html("ex1_prior_posterior.html")
print("\nSaved: ex1_prior_posterior.html")

# TODO: Create a bootstrap distribution histogram using viz.histogram
fig2 = ____  # Hint: viz.histogram(bootstrap_means, title="Bootstrap Distribution of Sample Mean", x_label="Mean Price ($)")
fig2.write_html("ex1_bootstrap_distribution.html")
print("Saved: ex1_bootstrap_distribution.html")

x_beta = np.linspace(0, 1, 500)
beta_prior_pdf = stats.beta.pdf(x_beta, alpha_prior, beta_prior)
beta_post_pdf = stats.beta.pdf(x_beta, alpha_post_weak, beta_post_weak)

fig3 = go.Figure()
fig3.add_trace(
    go.Scatter(x=x_beta, y=beta_prior_pdf, name="Prior Beta(2,2)",
               line={"color": "blue", "dash": "dash"})
)
fig3.add_trace(
    go.Scatter(x=x_beta, y=beta_post_pdf, name="Posterior", line={"color": "red"})
)
fig3.add_vline(x=empirical_rate, line_dash="dot", annotation_text="Empirical rate")
fig3.update_layout(
    title="Beta-Binomial: P(price > $500K) Prior vs Posterior",
    xaxis_title="Proportion",
    yaxis_title="Density",
)
fig3.write_html("ex1_beta_binomial.html")
print("Saved: ex1_beta_binomial.html")

# ── Checkpoint 11 ────────────────────────────────────────────────────
running_means = np.cumsum(bootstrap_means) / np.arange(1, n_bootstrap + 1)
assert (
    abs(running_means[-1] - mle_mean) < mle_se * 3
), "Bootstrap mean should converge to MLE mean"
print("\n✓ Checkpoint 11 passed — visualisations saved\n")


# ══════════════════════════════════════════════════════════════════════
# TASK 12: Business Interpretation Synthesis
# ══════════════════════════════════════════════════════════════════════

print(f"\n=== Business Interpretation: Property Valuation Insights ===")
print(
    f"""
1. MARKET POSITION: The average 4-room HDB resale price is ${mle_mean:,.0f}
   with a standard error of ${mle_se:,.0f}. Based on {n:,} recent transactions.

2. BAYESIAN ESTIMATE: Data overwhelms the $500K prior — posterior mean
   is ${mu_n:,.0f}. Prior assumptions barely matter with large n.

3. PRICE SEGMENTS: {empirical_rate:.1%} of 4-room transactions close above
   $500K. Beta-Binomial credible interval: [{ci_weak[0]:.2%}, {ci_weak[1]:.2%}].

4. FLAT TYPE VARIATION: For 4-room (abundant data), prior contributes <0.01%.
   For rare types (2-room, Executive), prior contributes more.

5. SAMPLING BIAS: The friendship paradox applies to property markets.
   High-visibility properties skew market perception.
"""
)

# ── Checkpoint 12 ────────────────────────────────────────────────────
assert results_by_type, "Should have computed posteriors for at least one flat type"
print("\n✓ Checkpoint 12 passed — business interpretation complete\n")


# ══════════════════════════════════════════════════════════════════════
# REFLECTION
# ══════════════════════════════════════════════════════════════════════
print("""
What you've mastered:
  ✓ Empirical probability from data (joint, conditional, independence)
  ✓ Bayes' theorem with medical and property scenarios
  ✓ MLE for Normal parameters with Cramér-Rao bound
  ✓ Normal-Normal conjugate prior: computing posteriors analytically
  ✓ Prior sensitivity: showing data dominance with large n
  ✓ Beta-Binomial conjugate: Bayesian model for proportions
  ✓ Credible vs confidence intervals via repeated-sampling simulation
  ✓ Expected value and sampling bias (friendship paradox)

Next: In Exercise 2, you'll dive deeper into parameter estimation —
optimising log-likelihoods numerically, MAP estimation, and AIC/BIC
model selection.
""")
