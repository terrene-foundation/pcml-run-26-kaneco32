# Copyright 2026 Terrene Foundation
# SPDX-License-Identifier: Apache-2.0
"""
# ════════════════════════════════════════════════════════════════════════
# MLFP02 — Exercise 2: Parameter Estimation and Inference
# ════════════════════════════════════════════════════════════════════════
#
# WHAT YOU'LL LEARN:
#   After completing this exercise, you will be able to:
#   - Distinguish population parameters from sample statistics and explain
#     why Bessel's correction (ddof=1) fixes bias in the sample variance
#   - Simulate the Central Limit Theorem: observe how the sampling
#     distribution of the mean becomes Normal regardless of the population
#   - Write and optimise a log-likelihood function using scipy.optimize
#   - Distinguish MLE (no prior) from MAP estimation (MLE + prior)
#   - Compute standard errors and Wald confidence intervals from the Hessian
#   - Construct profile likelihood CIs (more accurate for small n)
#   - Diagnose three failure modes of MLE: small n, multimodal, misspecification
#   - Use AIC/BIC to compare distribution families (Normal vs Student-t)
#   - Implement bootstrap for non-standard statistics (median, trimmed mean)
#   - Visualise likelihood surfaces and sampling distributions
#
# PREREQUISITES: Complete Exercise 1 — you should understand MLE, posterior
#   distributions, and the Normal-Normal conjugate prior.
#
# ESTIMATED TIME: ~170 minutes
#
# TASKS:
#    1. Load Singapore economic data and profile distributions
#    2. Population vs sample: demonstrate Bessel's correction with simulation
#    3. Central Limit Theorem simulation with visualisation
#    4. MLE via log-likelihood optimisation (scipy.optimize)
#    5. Standard errors from Fisher information; Wald and profile LR CIs
#    6. MAP estimation — MLE with a prior; shrinkage demonstration
#    7. MLE failure case 1: small n — biased and high-variance estimates
#    8. MLE failure case 2: multimodal data — mean between modes
#    9. MLE failure case 3: misspecified likelihood — tail risk
#   10. AIC/BIC model comparison across distribution families
#   11. Bootstrap for the median and trimmed mean (non-standard statistics)
#   12. Visualise and interpret all results
#
# DATASET: Singapore economic indicators (GDP, inflation, unemployment)
#   Source: Singapore Department of Statistics (DOS)
#   Key column: gdp_growth_pct (quarterly % change, annualised)
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
from scipy.optimize import minimize

from shared import MLFPDataLoader


# ── Data Loading ──────────────────────────────────────────────────────

loader = MLFPDataLoader()
econ = loader.load("mlfp01", "economic_indicators.csv")

print("=" * 70)
print("  MLFP02 Exercise 2: Parameter Estimation and Inference")
print("=" * 70)
print(
    f"\n  Data loaded: economic_indicators.csv ({econ.shape[0]} rows, {econ.shape[1]} cols)"
)
print(f"  Columns: {econ.columns}")
print(econ.head(8))

gdp_growth = econ["gdp_growth_pct"].drop_nulls().to_numpy().astype(np.float64)
inflation = econ["inflation_rate"].drop_nulls().to_numpy().astype(np.float64)
unemployment = econ["unemployment_rate"].drop_nulls().to_numpy().astype(np.float64)

n_gdp = len(gdp_growth)
print(f"\nGDP growth observations: {n_gdp}")
print(f"GDP growth range: {gdp_growth.min():.2f}% to {gdp_growth.max():.2f}%")
print(f"Sample mean: {gdp_growth.mean():.3f}%")
print(f"Sample std:  {gdp_growth.std():.3f}%")


# ══════════════════════════════════════════════════════════════════════
# TASK 1: Data Profiling — Distribution Shape Assessment
# ══════════════════════════════════════════════════════════════════════
# Before fitting a parametric model, check whether the Normal assumption
# is plausible. Use Shapiro-Wilk, skewness, and excess kurtosis.

# TODO: Run the Shapiro-Wilk normality test on gdp_growth
shapiro_stat, shapiro_p = ____  # Hint: stats.shapiro(gdp_growth)
print(f"\n=== Distribution Shape Assessment ===")
print(f"Shapiro-Wilk test: W={shapiro_stat:.4f}, p={shapiro_p:.4f}")
if shapiro_p > 0.05:
    print("Cannot reject normality — Normal likelihood is plausible")
else:
    print("Normality rejected — consider heavier-tailed distributions")

skew = stats.skew(gdp_growth)
kurt = stats.kurtosis(gdp_growth)  # excess kurtosis (Fisher)
print(f"Skewness: {skew:.3f} (|skew|>1 suggests non-Normal)")
print(f"Excess kurtosis: {kurt:.3f} (>0 -> heavier tails than Normal)")

for name, arr in [
    ("GDP growth", gdp_growth),
    ("Inflation", inflation),
    ("Unemployment", unemployment),
]:
    print(
        f"\n{name}: mean={arr.mean():.3f}, std={arr.std():.3f}, "
        f"min={arr.min():.3f}, max={arr.max():.3f}, n={len(arr)}"
    )

# ── Checkpoint 1 ─────────────────────────────────────────────────────
assert 0 <= shapiro_p <= 1, "Shapiro-Wilk p-value must be between 0 and 1"
assert n_gdp > 10, f"Expected substantial GDP history, got {n_gdp} observations"
print("\n✓ Checkpoint 1 passed — data profiled and normality assessed\n")


# ══════════════════════════════════════════════════════════════════════
# TASK 2: Population vs Sample — Bessel's Correction Simulation
# ══════════════════════════════════════════════════════════════════════
# MLE uses ddof=0 (biased). Unbiased estimator uses ddof=1.
# Expected bias: E[MLE σ̂²] = (n-1)/n × σ²

print(f"\n=== Population vs Sample Variance ===")

rng = np.random.default_rng(seed=42)
true_mu = 3.0
true_sigma = 5.0
n_sim_repeats = 5000
sample_size_demo = 10

mle_vars = []
unbiased_vars = []

for _ in range(n_sim_repeats):
    sample = rng.normal(true_mu, true_sigma, size=sample_size_demo)
    mle_vars.append(sample.var(ddof=0))
    unbiased_vars.append(sample.var(ddof=1))

mle_var_mean = np.mean(mle_vars)
unbiased_var_mean = np.mean(unbiased_vars)
true_var = true_sigma**2

print(f"True σ² = {true_var:.2f}")
print(f"n per sample: {sample_size_demo}, Simulations: {n_sim_repeats:,}")
print(
    f"E[MLE σ̂²] (ddof=0):     {mle_var_mean:.2f} (bias = {mle_var_mean - true_var:+.2f})"
)
print(
    f"E[unbiased s²] (ddof=1): {unbiased_var_mean:.2f} (bias = {unbiased_var_mean - true_var:+.2f})"
)
print(
    f"Ratio MLE/true: {mle_var_mean / true_var:.4f} (expected: {(sample_size_demo-1)/sample_size_demo:.4f})"
)
# INTERPRETATION: The MLE variance systematically underestimates the
# true variance by factor (n-1)/n. For n=10, that's a 10% negative bias.
# Bessel's correction (ddof=1) makes the estimator unbiased on average.

# ── Checkpoint 2 ─────────────────────────────────────────────────────
assert mle_var_mean < true_var, "MLE variance should underestimate on average"
assert (
    abs(unbiased_var_mean - true_var) < true_var * 0.05
), "Unbiased variance should be within 5% of true variance"
print("\n✓ Checkpoint 2 passed — Bessel's correction demonstrated\n")


# ══════════════════════════════════════════════════════════════════════
# TASK 3: Central Limit Theorem — Simulation and Visualisation
# ══════════════════════════════════════════════════════════════════════
# CLT: the sampling distribution of x̄ approaches N(μ, σ²/n) as n → ∞.
# Demonstrate with three population shapes: Exponential, Uniform, Bimodal.

print(f"\n=== Central Limit Theorem Simulation ===")

n_clt_samples = 5000
sample_sizes_clt = [5, 15, 30, 100]

population_exp = rng.exponential(scale=1.0, size=100_000)
population_unif = rng.uniform(0, 10, size=100_000)
pop_bimodal = np.concatenate(
    [rng.normal(2, 0.5, size=50_000), rng.normal(8, 0.5, size=50_000)]
)

populations = {
    "Exponential(1)": population_exp,
    "Uniform(0,10)": population_unif,
    "Bimodal N(2,0.5)+N(8,0.5)": pop_bimodal,
}

for pop_name, pop in populations.items():
    print(f"\n--- {pop_name} ---")
    print(
        f"Population: mean={pop.mean():.3f}, std={pop.std():.3f}, skew={stats.skew(pop):.3f}"
    )

    for n_s in sample_sizes_clt:
        sample_means = np.array(
            [
                rng.choice(pop, size=n_s, replace=True).mean()
                for _ in range(n_clt_samples)
            ]
        )
        sm_mean = sample_means.mean()
        sm_std = sample_means.std()
        sm_skew = stats.skew(sample_means)
        theoretical_se = pop.std() / np.sqrt(n_s)

        sw_stat, sw_p = stats.shapiro(
            rng.choice(sample_means, size=min(500, n_clt_samples), replace=False)
        )
        print(
            f"  n={n_s:>3}: mean(x̄)={sm_mean:.3f}, "
            f"SE(x̄)={sm_std:.3f} (theory: {theoretical_se:.3f}), "
            f"skew={sm_skew:.3f}, Shapiro p={sw_p:.3f}"
        )
# INTERPRETATION: Even for the heavily skewed Exponential, the sampling
# distribution of x̄ becomes approximately Normal by n=30. This is why
# Normal-based confidence intervals work even for non-Normal data.

# ── Checkpoint 3 ─────────────────────────────────────────────────────
exp_means_100 = np.array(
    [rng.choice(population_exp, size=100, replace=True).mean() for _ in range(2000)]
)
_, sw_p_100 = stats.shapiro(rng.choice(exp_means_100, size=200, replace=False))
assert (
    sw_p_100 > 0.01
), f"CLT: sampling distribution of mean should be ~Normal for n=100"
print("\n✓ Checkpoint 3 passed — CLT demonstrated across three population shapes\n")


# ══════════════════════════════════════════════════════════════════════
# TASK 4: MLE via Log-Likelihood Optimisation
# ══════════════════════════════════════════════════════════════════════
# For X ~ N(μ, σ²): ℓ(μ, σ | x) = -n/2 log(2π) - n log(σ) - Σ(xᵢ-μ)²/(2σ²)
# Reparameterize: use log(σ) to enforce σ > 0.


def neg_log_likelihood_normal(params: np.ndarray, x: np.ndarray) -> float:
    """Negative log-likelihood for N(mu, sigma²) — minimized by scipy."""
    mu, log_sigma = params
    sigma = np.exp(log_sigma)
    if sigma <= 0:
        return np.inf
    return -np.sum(stats.norm.logpdf(x, loc=mu, scale=sigma))


mle_mu_analytic = gdp_growth.mean()
mle_sigma_analytic = gdp_growth.std(ddof=0)

# TODO: Set initial params for optimizer: [mean, log(std)]
x0 = ____  # Hint: np.array([gdp_growth.mean(), np.log(gdp_growth.std())])
result_mle = minimize(
    neg_log_likelihood_normal,
    x0,
    args=(gdp_growth,),
    method="L-BFGS-B",
    options={"maxiter": 1000, "ftol": 1e-12},
)

mle_mu_numeric = result_mle.x[0]
# TODO: Extract sigma from log-parameterized optimizer result
mle_sigma_numeric = ____  # Hint: np.exp(result_mle.x[1])

print(f"\n=== MLE Results: GDP Growth Rate ===")
print(f"Analytical MLE:  μ = {mle_mu_analytic:.4f}%, σ = {mle_sigma_analytic:.4f}%")
print(f"Numerical MLE:   μ = {mle_mu_numeric:.4f}%, σ = {mle_sigma_numeric:.4f}%")
print(f"Optimizer converged: {result_mle.success} (message: {result_mle.message})")
print(f"Log-likelihood at MLE: {-result_mle.fun:.4f}")

# ── Checkpoint 4 ─────────────────────────────────────────────────────
assert result_mle.success, "MLE optimizer should converge"
assert (
    abs(mle_mu_numeric - mle_mu_analytic) < 0.01
), "Numerical and analytical MLE should agree"
print("\n✓ Checkpoint 4 passed — MLE computed analytically and numerically\n")


# ══════════════════════════════════════════════════════════════════════
# TASK 5: Standard Errors, Wald CI, and Profile Likelihood CI
# ══════════════════════════════════════════════════════════════════════
# SE(μ̂) = σ/√n, SE(σ̂) = σ/√(2n)
# Wald CI: μ̂ ± z_{α/2} × SE(μ̂)
# Profile LR CI: find μ where 2(ℓ(μ̂) - ℓ(μ)) < χ²(0.95, df=1)

# TODO: Compute SE of the MLE mean estimate
mle_mu_se = ____  # Hint: mle_sigma_numeric / np.sqrt(n_gdp)
# TODO: Compute SE of the MLE sigma estimate
mle_sigma_se = ____  # Hint: mle_sigma_numeric / np.sqrt(2 * n_gdp)

print(f"\n=== Standard Errors and Confidence Intervals ===")
print(f"SE(μ̂) = {mle_mu_se:.4f}%")
print(f"SE(σ̂) = {mle_sigma_se:.4f}%")

# TODO: Compute 95% Wald CI: (mu - 1.96*se, mu + 1.96*se)
mle_mu_ci = (
    ____  # Hint: (mle_mu_numeric - 1.96 * mle_mu_se, mle_mu_numeric + 1.96 * mle_mu_se)
)
print(f"\n95% Wald CI for μ: [{mle_mu_ci[0]:.3f}%, {mle_mu_ci[1]:.3f}%]")

# Profile likelihood CI
lr_threshold = stats.chi2.ppf(0.95, df=1) / 2
profile_loglik = lambda mu: -neg_log_likelihood_normal(
    [mu, np.log(mle_sigma_numeric)], gdp_growth
)
loglik_at_mle = -result_mle.fun

mu_grid = np.linspace(
    mle_mu_numeric - 4 * mle_mu_se, mle_mu_numeric + 4 * mle_mu_se, 500
)
lr_values = np.array([loglik_at_mle - profile_loglik(mu) for mu in mu_grid])
lr_ci_mask = lr_values <= lr_threshold
if lr_ci_mask.any():
    lr_ci = (mu_grid[lr_ci_mask][0], mu_grid[lr_ci_mask][-1])
else:
    lr_ci = mle_mu_ci

print(f"Profile LR 95% CI for μ: [{lr_ci[0]:.3f}%, {lr_ci[1]:.3f}%]")
print(f"Wald CI width:    {mle_mu_ci[1] - mle_mu_ci[0]:.4f}%")
print(f"Profile CI width: {lr_ci[1] - lr_ci[0]:.4f}%")
# INTERPRETATION: Profile LR CI is invariant to reparameterisation and
# better for small n. When they agree, Normal approximation is adequate.

# ── Checkpoint 5 ─────────────────────────────────────────────────────
assert mle_mu_ci[0] < mle_mu_numeric < mle_mu_ci[1], "MLE mean must be inside Wald CI"
assert mle_mu_se > 0, "Standard error must be positive"
print("\n✓ Checkpoint 5 passed — SEs and CIs computed\n")


# ══════════════════════════════════════════════════════════════════════
# TASK 6: MAP Estimation — MLE with a Prior
# ══════════════════════════════════════════════════════════════════════
# MAP = argmax p(θ|x) = argmax [ℓ(θ|x) + log p(θ)]
# Prior: μ ~ N(3.5%, 1.5²%) — typical growth for a small open economy

mu_prior_mean = 3.5  # %
mu_prior_std = 1.5  # %


def neg_map_objective(params: np.ndarray, x: np.ndarray) -> float:
    """Negative MAP objective = NLL + negative log-prior."""
    mu, log_sigma = params
    sigma = np.exp(log_sigma)
    if sigma <= 0:
        return np.inf
    nll = -np.sum(stats.norm.logpdf(x, loc=mu, scale=sigma))
    # TODO: Add the negative log-prior term: -stats.norm.logpdf(mu, loc=mu_prior_mean, scale=mu_prior_std)
    neg_log_prior = (
        ____  # Hint: -stats.norm.logpdf(mu, loc=mu_prior_mean, scale=mu_prior_std)
    )
    return nll + neg_log_prior


result_map = minimize(
    neg_map_objective,
    x0,
    args=(gdp_growth,),
    method="L-BFGS-B",
    options={"maxiter": 1000, "ftol": 1e-12},
)

map_mu = result_map.x[0]
map_sigma = np.exp(result_map.x[1])

print(f"\n=== MAP vs MLE Comparison ===")
print(f"Prior: μ ~ N({mu_prior_mean}, {mu_prior_std}²)")
print(f"n = {n_gdp} observations")
print(f"MLE: μ̂ = {mle_mu_numeric:.4f}%, σ̂ = {mle_sigma_numeric:.4f}%")
print(f"MAP: μ̂ = {map_mu:.4f}%, σ̂ = {map_sigma:.4f}%")
print(f"MAP shrinkage toward prior: {map_mu - mle_mu_numeric:+.4f}%")

print(f"\n--- MAP Shrinkage by Sample Size ---")
for n_small in [3, 5, 10, 20, 50, n_gdp]:
    sample = gdp_growth[: min(n_small, n_gdp)]
    r_mle = minimize(
        neg_log_likelihood_normal,
        [sample.mean(), np.log(sample.std() + 1e-6)],
        args=(sample,),
        method="L-BFGS-B",
    )
    r_map = minimize(
        neg_map_objective,
        [sample.mean(), np.log(sample.std() + 1e-6)],
        args=(sample,),
        method="L-BFGS-B",
    )
    mle_s = r_mle.x[0]
    map_s = r_map.x[0]
    shrinkage_pct = abs(map_s - mle_s) / (abs(mu_prior_mean - mle_s) + 1e-10) * 100
    print(
        f"  n={n_small:>3}: MLE={mle_s:>7.3f}%, MAP={map_s:>7.3f}%, "
        f"shrinkage={map_s - mle_s:+.3f}% ({shrinkage_pct:.1f}% toward prior)"
    )
# INTERPRETATION: MAP shrinks toward the prior. With small n, the prior
# has ~30% influence. With large n, prior barely matters. MAP is
# equivalent to L2 regularisation (Ridge) in regression.

# ── Checkpoint 6 ─────────────────────────────────────────────────────
assert result_map.success, "MAP optimizer should converge"
assert (
    abs(map_mu - mle_mu_numeric) <= abs(mu_prior_mean - mle_mu_numeric) + 0.1
), "MAP should lie between MLE and prior"
print("\n✓ Checkpoint 6 passed — MAP estimation and shrinkage demonstrated\n")


# ══════════════════════════════════════════════════════════════════════
# TASK 7: MLE Failure Case 1 — Small n
# ══════════════════════════════════════════════════════════════════════
# With very small n, MLE estimates are unstable and biased (σ̂ especially).

print(f"\n=== MLE Failure Case 1: Small n ===")

small_ns = [3, 5, 10]
n_trials_per = 2000

for n_s in small_ns:
    mle_mus = []
    mle_sigmas = []
    for _ in range(n_trials_per):
        sample = rng.choice(gdp_growth, size=n_s, replace=False)
        mle_mus.append(sample.mean())
        mle_sigmas.append(sample.std(ddof=0))

    mu_bias = np.mean(mle_mus) - gdp_growth.mean()
    sigma_bias = np.mean(mle_sigmas) - gdp_growth.std(ddof=0)

    print(f"\nn={n_s}: (over {n_trials_per} random subsets)")
    print(f"  μ̂: bias={mu_bias:+.4f}%")
    print(f"  σ̂: bias={sigma_bias:+.4f}% (negative bias confirms MLE underestimates σ)")
# INTERPRETATION: For n=3, MLE σ̂ is biased downward by ~17%. This means
# CIs are too narrow. Remedies: Bessel's correction, MAP, or bootstrap.

# ── Checkpoint 7 ─────────────────────────────────────────────────────
mle_sigmas_3 = [
    rng.choice(gdp_growth, size=3, replace=False).std(ddof=0) for _ in range(1000)
]
assert np.mean(mle_sigmas_3) < gdp_growth.std(
    ddof=0
), "MLE σ̂ should be biased low for n=3"
print("\n✓ Checkpoint 7 passed — small-n bias demonstrated\n")


# ══════════════════════════════════════════════════════════════════════
# TASK 8: MLE Failure Case 2 — Multimodal Data
# ══════════════════════════════════════════════════════════════════════
# A single Normal fitted to bimodal data puts the mean between modes.

print(f"\n=== MLE Failure Case 2: Multimodal Data ===")

pre_covid = rng.normal(loc=4.0, scale=1.2, size=30)
covid_shock = rng.normal(loc=-5.0, scale=3.0, size=10)
bimodal_data = np.concatenate([pre_covid, covid_shock])

# TODO: Compute MLE estimates for the bimodal data
bimodal_mle_mu = ____  # Hint: bimodal_data.mean()
bimodal_mle_sigma = ____  # Hint: bimodal_data.std(ddof=0)

bimodality_coeff = (stats.skew(bimodal_data) ** 2 + 1) / (
    stats.kurtosis(bimodal_data, fisher=True)
    + 3
    * (
        (len(bimodal_data) - 1) ** 2
        / ((len(bimodal_data) - 2) * (len(bimodal_data) - 3))
    )
)

print(f"Bimodal data: {len(bimodal_data)} observations (two economic regimes)")
print(f"Mode 1 (pre-COVID): mean=4.0%, n=30")
print(f"Mode 2 (COVID shock): mean=-5.0%, n=10")
print(f"\nSingle Normal MLE: μ={bimodal_mle_mu:.2f}%, σ={bimodal_mle_sigma:.2f}%")
print(f"Bimodality coefficient: {bimodality_coeff:.3f} (>0.555 -> bimodal)")
print(f"Problem: MLE estimate {bimodal_mle_mu:.2f}% lies BETWEEN the two modes!")
print(f"\nRemedy: Gaussian Mixture Model (GMM) or regime-switching model")

# ── Checkpoint 8 ─────────────────────────────────────────────────────
assert (
    bimodal_mle_mu > -5.0 and bimodal_mle_mu < 4.0
), "MLE mean should fall between the two modes"
print("\n✓ Checkpoint 8 passed — multimodal failure demonstrated\n")


# ══════════════════════════════════════════════════════════════════════
# TASK 9: MLE Failure Case 3 — Misspecified Likelihood
# ══════════════════════════════════════════════════════════════════════
# Fitting Normal to heavy-tailed data underestimates tail risk.

print(f"\n=== MLE Failure Case 3: Misspecified Likelihood ===")

rng_t = np.random.default_rng(seed=77)
shock_data = rng_t.standard_t(df=3, size=100) * 2.0 + 2.5

normal_mle_mu = shock_data.mean()
normal_mle_sigma = shock_data.std(ddof=0)


def neg_ll_t(params: np.ndarray, x: np.ndarray) -> float:
    df, mu, scale = params
    if df <= 0 or scale <= 0:
        return np.inf
    return -np.sum(stats.t.logpdf(x, df=df, loc=mu, scale=scale))


result_t = minimize(
    neg_ll_t,
    [5.0, shock_data.mean(), shock_data.std()],
    args=(shock_data,),
    method="Nelder-Mead",
)
t_df, t_mu, t_scale = result_t.x

print(
    f"{'Percentile':<12} {'Normal':>10} {'t-dist':>10} {'Empirical':>10} {'Normal Error':>12}"
)
print("─" * 60)
for pctile in [90, 95, 99, 99.5]:
    normal_q = stats.norm.ppf(pctile / 100, loc=normal_mle_mu, scale=normal_mle_sigma)
    t_q = stats.t.ppf(pctile / 100, df=t_df, loc=t_mu, scale=t_scale)
    emp_q = np.percentile(shock_data, pctile)
    error = abs(normal_q - emp_q)
    print(
        f"{pctile:>10}th  {normal_q:>10.2f} {t_q:>10.2f} {emp_q:>10.2f} {error:>12.2f}"
    )
print(f"\nt-distribution df = {t_df:.1f} (lower -> heavier tails)")
print(f"Normal systematically underestimates extreme events")
# INTERPRETATION: The Normal model underestimates tail risk at the 99th
# percentile. Using the wrong model means you'll be underprepared for
# crises — the core argument for heavy-tailed models in risk management.

# ── Checkpoint 9 ─────────────────────────────────────────────────────
normal_99 = stats.norm.ppf(0.99, loc=normal_mle_mu, scale=normal_mle_sigma)
t_99 = stats.t.ppf(0.99, df=t_df, loc=t_mu, scale=t_scale)
assert t_99 > normal_99, "t-distribution 99th percentile should exceed Normal's"
print("\n✓ Checkpoint 9 passed — misspecification and tail risk demonstrated\n")


# ══════════════════════════════════════════════════════════════════════
# TASK 10: AIC/BIC Model Comparison
# ══════════════════════════════════════════════════════════════════════
# AIC = 2k - 2ℓ (lower is better, k = number of parameters)
# BIC = k*log(n) - 2ℓ (penalises complexity more for large n)

print(f"\n=== AIC/BIC Model Comparison ===")

fits = {}

# TODO: Compute log-likelihood for the fitted Normal distribution
ll_normal = ____  # Hint: np.sum(stats.norm.logpdf(gdp_growth, mle_mu_numeric, mle_sigma_numeric))
fits["Normal"] = {"k": 2, "ll": ll_normal}

r_t_full = minimize(
    neg_ll_t,
    [5.0, gdp_growth.mean(), gdp_growth.std()],
    args=(gdp_growth,),
    method="Nelder-Mead",
)
ll_t = -r_t_full.fun
fits["Student-t"] = {"k": 3, "ll": ll_t}

sn_params = stats.skewnorm.fit(gdp_growth)
ll_sn = np.sum(stats.skewnorm.logpdf(gdp_growth, *sn_params))
fits["Skew-Normal"] = {"k": 3, "ll": ll_sn}

lap_loc, lap_scale = stats.laplace.fit(gdp_growth)
ll_lap = np.sum(stats.laplace.logpdf(gdp_growth, loc=lap_loc, scale=lap_scale))
fits["Laplace"] = {"k": 2, "ll": ll_lap}

print(f"{'Distribution':<15} {'k':>3} {'Log-lik':>12} {'AIC':>10} {'BIC':>10}")
print("─" * 55)
for name, f in fits.items():
    # TODO: Compute AIC = 2*k - 2*log_likelihood
    aic = ____  # Hint: 2 * f["k"] - 2 * f["ll"]
    # TODO: Compute BIC = k*log(n) - 2*log_likelihood
    bic = ____  # Hint: f["k"] * np.log(n_gdp) - 2 * f["ll"]
    f["aic"] = aic
    f["bic"] = bic
    print(f"{name:<15} {f['k']:>3} {f['ll']:>12.2f} {aic:>10.2f} {bic:>10.2f}")

best_aic = min(fits.items(), key=lambda x: x[1]["aic"])
best_bic = min(fits.items(), key=lambda x: x[1]["bic"])
print(f"\nBest by AIC: {best_aic[0]} (AIC={best_aic[1]['aic']:.2f})")
print(f"Best by BIC: {best_bic[0]} (BIC={best_bic[1]['bic']:.2f})")
# INTERPRETATION: AIC and BIC may disagree. BIC penalises complexity
# more for large n. When they agree, evidence is strong.

# ── Checkpoint 10 ────────────────────────────────────────────────────
assert all(
    f["aic"] is not None for f in fits.values()
), "All AIC values must be computed"
print("\n✓ Checkpoint 10 passed — AIC/BIC model selection completed\n")


# ══════════════════════════════════════════════════════════════════════
# TASK 11: Bootstrap for Non-Standard Statistics
# ══════════════════════════════════════════════════════════════════════
# Bootstrap works for ANY statistic without analytical formulas.

print(f"\n=== Bootstrap for Non-Standard Statistics ===")

n_boot = 10_000

# TODO: Bootstrap the median using np.median
boot_medians = np.array(
    [____ for _ in range(n_boot)]
)  # Hint: np.median(rng.choice(gdp_growth, size=n_gdp, replace=True))
median_ci = np.percentile(boot_medians, [2.5, 97.5])

boot_trimmed = np.array(
    [
        stats.trim_mean(rng.choice(gdp_growth, size=n_gdp, replace=True), 0.1)
        for _ in range(n_boot)
    ]
)
trimmed_ci = np.percentile(boot_trimmed, [2.5, 97.5])

boot_iqrs = np.array(
    [
        np.subtract(
            *np.percentile(rng.choice(gdp_growth, size=n_gdp, replace=True), [75, 25])
        )
        for _ in range(n_boot)
    ]
)
iqr_ci = np.percentile(boot_iqrs, [2.5, 97.5])

boot_means = np.array(
    [rng.choice(gdp_growth, size=n_gdp, replace=True).mean() for _ in range(n_boot)]
)
mean_ci = np.percentile(boot_means, [2.5, 97.5])

print(f"{'Statistic':<18} {'Estimate':>10} {'Boot SE':>10} {'95% CI':>25}")
print("─" * 65)
for name, est, boots in [
    ("Mean", gdp_growth.mean(), boot_means),
    ("Median", np.median(gdp_growth), boot_medians),
    ("10% Trimmed Mean", stats.trim_mean(gdp_growth, 0.1), boot_trimmed),
    ("IQR", np.subtract(*np.percentile(gdp_growth, [75, 25])), boot_iqrs),
]:
    ci = np.percentile(boots, [2.5, 97.5])
    print(
        f"{name:<18} {est:>10.4f} {boots.std():>10.4f} [{ci[0]:>10.4f}, {ci[1]:>10.4f}]"
    )
# INTERPRETATION: Bootstrap SE and theoretical SE agree for the mean
# (validating the bootstrap). For median, trimmed mean, IQR, there's
# no simple formula — bootstrap is the only practical option.

# ── Checkpoint 11 ────────────────────────────────────────────────────
assert (
    median_ci[0] < np.median(gdp_growth) < median_ci[1]
), "Median must be within its bootstrap CI"
assert boot_medians.std() > 0, "Bootstrap SE of median must be positive"
print("\n✓ Checkpoint 11 passed — bootstrap for non-standard statistics\n")


# ══════════════════════════════════════════════════════════════════════
# TASK 12: Visualise and Interpret All Results
# ══════════════════════════════════════════════════════════════════════

viz = ModelVisualizer()

# Plot 1: CLT demonstration
fig1 = make_subplots(rows=1, cols=3, subplot_titles=["n=5", "n=30", "n=100"])
for i, ns in enumerate([5, 30, 100]):
    sample_means = [rng.choice(population_exp, size=ns).mean() for _ in range(3000)]
    fig1.add_trace(
        go.Histogram(
            x=sample_means, nbinsx=40, name=f"n={ns}", opacity=0.7, showlegend=True
        ),
        row=1,
        col=i + 1,
    )
fig1.update_layout(
    title="CLT: Sampling Distribution of Mean from Exponential(1)", height=350
)
fig1.write_html("ex2_clt_demonstration.html")
print("\nSaved: ex2_clt_demonstration.html")

# Plot 2: Profile log-likelihood
ll_values = np.array(
    [
        -neg_log_likelihood_normal([mu, np.log(mle_sigma_numeric)], gdp_growth)
        for mu in mu_grid
    ]
)
fig2 = go.Figure()
fig2.add_trace(go.Scatter(x=mu_grid, y=ll_values, name="Profile log-likelihood"))
fig2.add_vline(x=mle_mu_numeric, line_dash="dash", annotation_text="MLE")
fig2.add_vline(x=lr_ci[0], line_dash="dot", line_color="red")
fig2.add_vline(x=lr_ci[1], line_dash="dot", line_color="red")
fig2.update_layout(
    title="Profile Log-Likelihood: GDP Growth Rate",
    xaxis_title="μ (GDP growth %)",
    yaxis_title="Log-likelihood",
)
fig2.write_html("ex2_profile_loglikelihood.html")
print("Saved: ex2_profile_loglikelihood.html")

# Plot 3: Bimodal data histogram
# TODO: Create a bimodal histogram using viz.histogram
fig3 = ____  # Hint: viz.histogram(bimodal_data, title="Bimodal GDP Growth (Pre-COVID + COVID Shock)", x_label="GDP Growth (%)")
fig3.add_vline(x=bimodal_mle_mu, line_dash="dash", annotation_text="MLE mean")
fig3.write_html("ex2_bimodal_failure.html")
print("Saved: ex2_bimodal_failure.html")

# Plot 4: Bootstrap distributions comparison
fig4 = make_subplots(
    rows=1, cols=2, subplot_titles=["Bootstrap Means", "Bootstrap Medians"]
)
fig4.add_trace(go.Histogram(x=boot_means, nbinsx=50, name="Means"), row=1, col=1)
fig4.add_trace(go.Histogram(x=boot_medians, nbinsx=50, name="Medians"), row=1, col=2)
fig4.update_layout(title="Bootstrap Distributions: Mean vs Median", height=350)
fig4.write_html("ex2_bootstrap_comparison.html")
print("Saved: ex2_bootstrap_comparison.html")

# ── Checkpoint 12 ────────────────────────────────────────────────────
print("\n✓ Checkpoint 12 passed — visualisations saved\n")


# ══════════════════════════════════════════════════════════════════════
# REFLECTION
# ══════════════════════════════════════════════════════════════════════
print(
    """
What you've mastered:
  ✓ Population vs sample: parameter vs statistic
  ✓ Bessel's correction: MLE σ̂ is biased low by (n-1)/n
  ✓ Central Limit Theorem: x̄ -> Normal for Exponential, Uniform, Bimodal
  ✓ Log-likelihood optimisation with scipy.optimize.minimize
  ✓ Wald SE from Fisher information: SE(μ̂) = σ / sqrt(n)
  ✓ Profile likelihood CI — invariant to reparameterisation
  ✓ MAP = MLE + log-prior; shrinks toward prior, fades with large n
  ✓ MLE failure modes: small n, multimodal, misspecification
  ✓ AIC/BIC: penalised likelihood for distribution comparison
  ✓ Bootstrap for any statistic: median, trimmed mean, IQR

Next: In Exercise 3, you'll move to decision-making — formulate
hypotheses, run power analysis, apply multiple testing corrections
(Bonferroni and BH-FDR), and implement a permutation test.
"""
)
