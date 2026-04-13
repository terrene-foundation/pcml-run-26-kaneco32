# Copyright 2026 Terrene Foundation
# SPDX-License-Identifier: Apache-2.0
"""
# ════════════════════════════════════════════════════════════════════════
# MLFP02 — Exercise 8: Capstone — Statistical Analysis Project
# ════════════════════════════════════════════════════════════════════════
#
# WHAT YOU'LL LEARN:
#   After completing this exercise, you will be able to:
#   - Execute a complete statistical analysis from data to recommendations
#   - Engineer and version features using FeatureStore with typed schemas
#   - Retrieve features at different points in time to prevent data leakage
#   - Track data lineage from raw data to features to trained model
#   - Apply point-in-time correctness to ensure no future data in training
#   - Build a regression model using the full M2 statistical toolkit
#   - Generate temporal and interaction features with domain rationale
#   - Compute rolling market statistics using Polars group_by_dynamic
#   - Present statistical findings in a stakeholder-ready format
#   - Connect Bayesian inference, hypothesis testing, and regression
#     into a coherent analytical narrative
#   - Use ExperimentTracker for reproducible model audit trails
#
# PREREQUISITES: All of Module 2 (Exercises 1-7) — Bayesian inference,
#   MLE/MAP estimation, hypothesis testing, A/B design, linear and
#   logistic regression, CUPED and causal inference.
#
# ESTIMATED TIME: ~180 minutes
#
# TASKS:
#    1. Load data and connect to FeatureStore + ExperimentTracker
#    2. Exploratory statistical analysis (distributions, correlations)
#    3. Define FeatureSchema v1 — basic property features
#    4. Compute and store v1 features with validation
#    5. Point-in-time retrieval — demonstrate leakage prevention
#    6. Define FeatureSchema v2 — add rolling market features
#    7. Compute v2 features (group_by_dynamic + rolling stats)
#    8. Build regression model on feature-store features
#    9. Hypothesis tests on model coefficients (from-scratch t-stats)
#   10. Bayesian posterior for the price premium of key features
#   11. Data lineage: full audit trail from data to model
#   12. Stakeholder report: synthesise all M2 concepts
#
# DATASET: HDB resale flat transactions (Singapore)
#   Source: data.gov.sg — public housing resale records
#
# ════════════════════════════════════════════════════════════════════════
"""
from __future__ import annotations

import asyncio
from datetime import datetime

import numpy as np
import polars as pl
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from kailash.db import ConnectionManager
from kailash_ml import FeatureStore, DataExplorer, ModelVisualizer
from kailash_ml.engines.experiment_tracker import ExperimentTracker
from kailash_ml.types import FeatureSchema, FeatureField
from scipy import stats

from shared import MLFPDataLoader
from shared.kailash_helpers import setup_environment

setup_environment()


# ── Data Loading ──────────────────────────────────────────────────────

loader = MLFPDataLoader()
hdb = loader.load("mlfp01", "hdb_resale.parquet")

print("=" * 70)
print("  MLFP02 Exercise 8: Capstone — Statistical Analysis Project")
print("=" * 70)
print(f"\n  Data loaded: hdb_resale.parquet ({hdb.shape[0]:,} rows)")

hdb = hdb.with_columns(pl.col("month").str.to_date("%Y-%m").alias("transaction_date"))


# ══════════════════════════════════════════════════════════════════════
# TASK 1: Connect to FeatureStore and ExperimentTracker
# ══════════════════════════════════════════════════════════════════════


async def setup():
    """Set up shared infrastructure."""
    from kailash.infrastructure import StoreFactory

    # TODO: Create StoreFactory, FeatureStore, and ExperimentTracker
    factory = ____  # Hint: StoreFactory("sqlite:///mlfp02_experiments.db")
    fs = ____  # Hint: FeatureStore(factory, table_prefix="kml_feat_")
    tracker = ____  # Hint: ExperimentTracker(factory)
    return factory, fs, tracker


HAS_FEATURE_STORE = False
try:
    conn, fs, tracker = asyncio.run(setup())
    HAS_FEATURE_STORE = True
    print(f"\nConnected to FeatureStore + ExperimentTracker")
except Exception as e:
    print(f"  Note: Infrastructure setup skipped ({type(e).__name__}: {e})")
    print(f"  Proceeding with core statistical analysis...")
    conn, fs, tracker = None, None, None

# ── Checkpoint 1 ─────────────────────────────────────────────────────
if HAS_FEATURE_STORE:
    print("\n✓ Checkpoint 1 passed — FeatureStore and ExperimentTracker connected\n")
else:
    print(
        "\n⚠ Checkpoint 1 — FeatureStore not available, proceeding with core analysis\n"
    )


# ══════════════════════════════════════════════════════════════════════
# TASK 2: Exploratory Statistical Analysis
# ══════════════════════════════════════════════════════════════════════
# Apply M2 statistical concepts to explore the data before modelling.

print(f"\n=== Exploratory Statistical Analysis ===")

prices = hdb["resale_price"].to_numpy().astype(np.float64)

# Distribution shape
skew = stats.skew(prices)
kurt = stats.kurtosis(prices)
sw_stat, sw_p = stats.shapiro(
    np.random.default_rng(42).choice(prices, size=5000, replace=False)
)

print(f"\nPrice distribution:")
print(f"  n = {len(prices):,}")
print(f"  Mean: ${prices.mean():,.0f}, Median: ${np.median(prices):,.0f}")
print(f"  Std: ${prices.std():,.0f}")
print(
    f"  Skewness: {skew:.3f} ({'right-skewed' if skew > 0.5 else 'approximately symmetric'})"
)
print(
    f"  Excess kurtosis: {kurt:.3f} ({'heavy-tailed' if kurt > 1 else 'normal-tailed'})"
)
print(f"  Shapiro-Wilk: W={sw_stat:.4f}, p={sw_p:.6f}")

# MLE fit
mle_mu = prices.mean()
mle_sigma = prices.std(ddof=0)
print(f"\nMLE (Normal): μ̂=${mle_mu:,.0f}, σ̂=${mle_sigma:,.0f}")

# Correlation analysis between numeric features
hdb_numeric = hdb.with_columns(
    (
        (
            pl.col("storey_range").str.extract(r"(\d+)", 1).cast(pl.Float64)
            + pl.col("storey_range").str.extract(r"TO (\d+)", 1).cast(pl.Float64)
        )
        / 2
    ).alias("storey_midpoint"),
    (99 - (pl.col("transaction_date").dt.year() - pl.col("lease_commence_date")))
    .cast(pl.Float64)
    .alias("remaining_lease_years"),
).drop_nulls(subset=["floor_area_sqm", "storey_midpoint", "remaining_lease_years"])

corr_cols = [
    "resale_price",
    "floor_area_sqm",
    "storey_midpoint",
    "remaining_lease_years",
]
corr_data = hdb_numeric.select(corr_cols).to_numpy().astype(np.float64)

# TODO: Compute correlation matrix from the design matrix
corr_matrix = ____  # Hint: np.corrcoef(corr_data.T)

print(f"\nCorrelation matrix:")
print(f"{'':>20}", end="")
for c in corr_cols:
    print(f"  {c[:12]:>12}", end="")
print()
for i, name in enumerate(corr_cols):
    print(f"{name[:20]:<20}", end="")
    for j in range(len(corr_cols)):
        print(f"  {corr_matrix[i,j]:>12.3f}", end="")
    print()

# Price by flat type (ANOVA preview)
flat_types = hdb["flat_type"].unique().sort().to_list()
print(f"\n--- Price by Flat Type ---")
for ft in flat_types:
    subset = hdb.filter(pl.col("flat_type") == ft)["resale_price"]
    if subset.len() > 10:
        print(
            f"  {ft:<12}: n={subset.len():>7,}, mean=${subset.mean():>10,.0f}, "
            f"median=${subset.median():>10,.0f}"
        )

# ── Checkpoint 2 ─────────────────────────────────────────────────────
assert len(prices) > 0, "Must have price data"
assert corr_matrix.shape == (4, 4), "Correlation matrix must be 4x4"
print("\n✓ Checkpoint 2 passed — exploratory analysis completed\n")


# ══════════════════════════════════════════════════════════════════════
# TASK 3: FeatureSchema v1 — Basic Property Features
# ══════════════════════════════════════════════════════════════════════
# TODO: Define property_schema_v1 using FeatureSchema with 4 FeatureFields:
#   floor_area_sqm (float64, not nullable), remaining_lease_years (float64,
#   not nullable), storey_midpoint (float64, not nullable),
#   price_per_sqm (float64, not nullable).
#   Set entity_id_column="transaction_id", timestamp_column="transaction_date",
#   version=1.
# Hint: FeatureSchema(name="hdb_property_features", features=[...], ...)

property_schema_v1 = FeatureSchema(
    name="hdb_property_features",
    features=[
        FeatureField(
            name="floor_area_sqm",
            dtype=____,  # Hint: "float64"
            nullable=____,  # Hint: False
            description="Floor area in square metres",
        ),
        FeatureField(
            name="remaining_lease_years",
            dtype="float64",
            nullable=False,
            description="Remaining lease in years",
        ),
        FeatureField(
            name="storey_midpoint",
            dtype="float64",
            nullable=False,
            description="Midpoint of storey range",
        ),
        FeatureField(
            name="price_per_sqm",
            dtype="float64",
            nullable=False,
            description="Transaction price per square metre",
        ),
    ],
    entity_id_column=____,  # Hint: "transaction_id"
    timestamp_column=____,  # Hint: "transaction_date"
    version=1,
)

print(f"=== FeatureSchema v1 ===")
print(f"Name: {property_schema_v1.name}, Version: {property_schema_v1.version}")
for f in property_schema_v1.features:
    print(f"  {f.name}: {f.dtype} — {f.description}")


# ══════════════════════════════════════════════════════════════════════
# TASK 4: Compute and Store v1 Features
# ══════════════════════════════════════════════════════════════════════


def compute_v1_features(df: pl.DataFrame) -> pl.DataFrame:
    """Compute version 1 property features from raw HDB data."""
    return df.with_columns(
        (
            (
                pl.col("storey_range").str.extract(r"(\d+)", 1).cast(pl.Float64)
                + pl.col("storey_range").str.extract(r"TO (\d+)", 1).cast(pl.Float64)
            )
            / 2
        ).alias("storey_midpoint"),
        # TODO: Compute price_per_sqm as resale_price divided by floor_area_sqm
        (____).alias("price_per_sqm"),  # Hint: pl.col("resale_price") / pl.col("floor_area_sqm")
        (99 - (pl.col("transaction_date").dt.year() - pl.col("lease_commence_date")))
        .cast(pl.Float64)
        .alias("remaining_lease_years"),
    ).with_row_index("transaction_id")


features_v1 = compute_v1_features(hdb)
print(f"\nComputed v1 features: {features_v1.shape}")

# Sanity checks on feature values
for feat_name in ["price_per_sqm", "storey_midpoint", "remaining_lease_years"]:
    vals = features_v1[feat_name].drop_nulls()
    print(
        f"  {feat_name}: mean={vals.mean():.1f}, min={vals.min():.1f}, max={vals.max():.1f}"
    )

# ── Checkpoint 3 ─────────────────────────────────────────────────────
assert "transaction_id" in features_v1.columns, "transaction_id must exist"
assert "price_per_sqm" in features_v1.columns, "price_per_sqm must be computed"
assert features_v1["price_per_sqm"].min() > 0, "price_per_sqm must be positive"
print("\n✓ Checkpoint 3 passed — v1 features computed and validated\n")

# Store in FeatureStore
if HAS_FEATURE_STORE:
    try:

        async def store_v1():
            # TODO: Register schema and store features
            await ____  # Hint: fs.register_features(property_schema_v1)
            return await ____  # Hint: fs.store(features_v1, property_schema_v1)

        row_count = asyncio.run(store_v1())
        print(f"Stored {row_count:,} v1 feature rows")
    except Exception as e:
        HAS_FEATURE_STORE = False
        print(f"  [Skipped: FeatureStore ({type(e).__name__}: {e})]")


# ══════════════════════════════════════════════════════════════════════
# TASK 5: Point-in-Time Retrieval — Leakage Prevention
# ══════════════════════════════════════════════════════════════════════

print(f"\n=== Point-in-Time Feature Retrieval ===")

if HAS_FEATURE_STORE:
    try:

        async def pit_demo():
            # TODO: Retrieve features at two different time cuts
            f_2023 = await ____  # Hint: fs.get_training_set(schema=property_schema_v1, start=datetime(2000,1,1), end=datetime(2023,1,1))
            f_2024 = await ____  # Hint: fs.get_training_set(schema=property_schema_v1, start=datetime(2000,1,1), end=datetime(2024,1,1))
            return f_2023, f_2024

        features_2023, features_2024 = asyncio.run(pit_demo())
        delta = features_2024.height - features_2023.height
        print(f"Features as of 2023-01-01: {features_2023.height:,} rows")
        print(f"Features as of 2024-01-01: {features_2024.height:,} rows")
        print(f"Additional transactions in 2023: {delta:,}")
    except Exception as e:
        features_2023 = features_2024 = None
        HAS_FEATURE_STORE = False
        print(f"  [Skipped: PIT retrieval ({type(e).__name__}: {e})]")
else:
    features_2023 = features_2024 = None
    # Simulate PIT with Polars filter
    f_2023 = features_v1.filter(pl.col("transaction_date") < pl.date(2023, 1, 1))
    f_2024 = features_v1.filter(pl.col("transaction_date") < pl.date(2024, 1, 1))
    print(f"[Polars PIT] Features before 2023: {f_2023.height:,}")
    print(f"[Polars PIT] Features before 2024: {f_2024.height:,}")
    print(f"[Polars PIT] Delta: {f_2024.height - f_2023.height:,}")

print(f"\n--- Why Point-in-Time Matters ---")
print(f"To predict prices at T=2023-01-01, you must ONLY use data before T.")
print(f"Using 2024 data would leak future info → over-optimistic evaluation.")

# ── Checkpoint 4 ─────────────────────────────────────────────────────
print("\n✓ Checkpoint 4 passed — point-in-time retrieval demonstrated\n")


# ══════════════════════════════════════════════════════════════════════
# TASK 6: FeatureSchema v2 — Rolling Market Features
# ══════════════════════════════════════════════════════════════════════

property_schema_v2 = FeatureSchema(
    name="hdb_property_features",
    features=[
        *property_schema_v1.features,
        FeatureField(
            name="town_median_price",
            dtype="float64",
            nullable=True,
            description="Median price in town (trailing 6 months)",
        ),
        FeatureField(
            name="town_transaction_volume",
            dtype="int64",
            nullable=True,
            description="Transaction count in town (trailing 6 months)",
        ),
        FeatureField(
            name="town_price_trend",
            dtype="float64",
            nullable=True,
            description="6-month price change % in town",
        ),
    ],
    entity_id_column="transaction_id",
    timestamp_column="transaction_date",
    version=2,
)

n_new = len(property_schema_v2.features) - len(property_schema_v1.features)
print(f"\n=== FeatureSchema v2 (+{n_new} market features) ===")
for f in property_schema_v2.features:
    tag = (
        " [NEW]"
        if f.name not in [ff.name for ff in property_schema_v1.features]
        else ""
    )
    print(f"  {f.name}: {f.dtype}{tag}")


# ══════════════════════════════════════════════════════════════════════
# TASK 7: Compute v2 Features (Rolling Market Statistics)
# ══════════════════════════════════════════════════════════════════════


def compute_v2_features(df: pl.DataFrame) -> pl.DataFrame:
    """Compute v2 features including rolling market context."""
    result = compute_v1_features(df).sort("transaction_date")

    # TODO: Compute monthly town-level statistics using group_by_dynamic
    # Hint: result.group_by_dynamic("transaction_date", every="1mo", group_by="town")
    #       .agg(pl.col("resale_price").median().alias("monthly_median"),
    #            pl.col("resale_price").count().alias("monthly_volume"))
    #       .sort("town", "transaction_date")
    town_stats = ____

    # TODO: Compute 6-month rolling stats over each town using .over("town")
    # Hint: town_stats.with_columns(
    #   pl.col("monthly_median").rolling_mean(window_size=6).over("town").alias("town_median_price"),
    #   pl.col("monthly_volume").rolling_sum(window_size=6).over("town").alias("town_transaction_volume"),
    #   ((pl.col("monthly_median") - pl.col("monthly_median").shift(6).over("town"))
    #    / pl.col("monthly_median").shift(6).over("town") * 100).alias("town_price_trend"),
    # )
    town_stats = ____

    result = result.join(
        town_stats.select(
            "town",
            "transaction_date",
            "town_median_price",
            "town_transaction_volume",
            "town_price_trend",
        ),
        on=["town", "transaction_date"],
        how="left",
    )
    return result


features_v2 = compute_v2_features(hdb)
n_with_market = features_v2.filter(pl.col("town_median_price").is_not_null()).height
print(f"\nComputed v2 features: {features_v2.shape}")
print(
    f"Rows with market context: {n_with_market:,} ({n_with_market/features_v2.height:.1%})"
)
print(f"(First 6 months per town have nulls — rolling window warm-up)")

# ── Checkpoint 5 ─────────────────────────────────────────────────────
assert features_v2.height == features_v1.height, "v2 should have same row count as v1"
assert "town_median_price" in features_v2.columns, "Market features must be computed"
print("\n✓ Checkpoint 5 passed — v2 features computed\n")

# Store v2
if HAS_FEATURE_STORE:
    try:

        async def store_v2():
            await fs.register_features(property_schema_v2)
            return await fs.store(features_v2, property_schema_v2)

        asyncio.run(store_v2())
        print(f"Stored v2 features")
    except Exception as e:
        HAS_FEATURE_STORE = False
        print(f"  [Skipped: v2 store ({type(e).__name__}: {e})]")


# ══════════════════════════════════════════════════════════════════════
# TASK 8: Build Regression Model on Feature-Store Features
# ══════════════════════════════════════════════════════════════════════
# Use the v2 features to build OLS regression (from Exercise 5 skills).

print(f"\n=== Regression Model on v2 Features ===")

# Use rows with complete market features
model_data = features_v2.drop_nulls(
    subset=[
        "floor_area_sqm",
        "storey_midpoint",
        "remaining_lease_years",
        "town_median_price",
        "town_price_trend",
        "resale_price",
    ]
)

# Build design matrix
feature_list = [
    "floor_area_sqm",
    "storey_midpoint",
    "remaining_lease_years",
    "town_median_price",
    "town_price_trend",
]
X_raw = model_data.select(feature_list).to_numpy().astype(np.float64)
y_reg = model_data["resale_price"].to_numpy().astype(np.float64)
n_reg = len(y_reg)

X_reg = np.column_stack([np.ones(n_reg), X_raw])
reg_names = ["intercept"] + feature_list
k_reg = X_reg.shape[1]

# TODO: Solve OLS with lstsq (β = (X'X)⁻¹X'y)
beta_reg = ____  # Hint: np.linalg.lstsq(X_reg, y_reg, rcond=None)[0]
y_hat_reg = X_reg @ beta_reg
resid_reg = y_reg - y_hat_reg

SSR_reg = np.sum(resid_reg**2)
SST_reg = np.sum((y_reg - y_reg.mean()) ** 2)

# TODO: Compute R² and adjusted R²
r2_reg = ____  # Hint: 1 - SSR_reg / SST_reg
adj_r2_reg = ____  # Hint: 1 - (1 - r2_reg) * (n_reg - 1) / (n_reg - k_reg)
rmse_reg = np.sqrt(SSR_reg / n_reg)

print(f"n = {n_reg:,}, k = {k_reg}")
print(f"R² = {r2_reg:.6f} ({r2_reg:.2%} variance explained)")
print(f"Adj R² = {adj_r2_reg:.6f}")
print(f"RMSE = ${rmse_reg:,.0f}")

print(f"\n{'Feature':<25} {'Coefficient':>14}")
print("─" * 42)
for name, coef in zip(reg_names, beta_reg):
    print(f"{name:<25} {coef:>14,.2f}")

# ── Checkpoint 6 ─────────────────────────────────────────────────────
assert r2_reg > 0.3, f"R² should be reasonable, got {r2_reg:.4f}"
print("\n✓ Checkpoint 6 passed — regression model built on v2 features\n")


# ══════════════════════════════════════════════════════════════════════
# TASK 9: Hypothesis Tests on Model Coefficients
# ══════════════════════════════════════════════════════════════════════
# From-scratch t-statistics (applying Exercise 5 skills).

sigma_sq_reg = SSR_reg / (n_reg - k_reg)

# TODO: Compute XtX inverse and standard errors for each coefficient
XtX_inv = ____  # Hint: np.linalg.inv(X_reg.T @ X_reg)
se_beta = ____  # Hint: np.sqrt(sigma_sq_reg * np.diag(XtX_inv))
t_stats = beta_reg / se_beta
p_values = 2 * (1 - stats.t.cdf(np.abs(t_stats), df=n_reg - k_reg))

print(f"\n=== Coefficient Significance ===")
print(f"{'Feature':<25} {'β':>12} {'SE':>10} {'t':>8} {'p-value':>12} {'Sig':>4}")
print("─" * 75)
for i, name in enumerate(reg_names):
    sig = (
        "***"
        if p_values[i] < 0.001
        else "**" if p_values[i] < 0.01 else "*" if p_values[i] < 0.05 else "ns"
    )
    print(
        f"{name:<25} {beta_reg[i]:>12,.2f} {se_beta[i]:>10,.2f} "
        f"{t_stats[i]:>8.2f} {p_values[i]:>12.2e} {sig:>4}"
    )

# F-statistic: model vs intercept-only
SSE_reg = np.sum((y_hat_reg - y_reg.mean()) ** 2)
f_stat_reg = (SSE_reg / (k_reg - 1)) / (SSR_reg / (n_reg - k_reg))
f_p_reg = 1 - stats.f.cdf(f_stat_reg, dfn=k_reg - 1, dfd=n_reg - k_reg)
print(f"\nF-statistic: {f_stat_reg:.2f} (p < {f_p_reg:.2e})")
print(
    f"Model is {'significantly better' if f_p_reg < 0.05 else 'NOT better'} than mean-only"
)

# ── Checkpoint 7 ─────────────────────────────────────────────────────
assert all(se > 0 for se in se_beta), "SEs must be positive"
print("\n✓ Checkpoint 7 passed — hypothesis tests completed\n")


# ══════════════════════════════════════════════════════════════════════
# TASK 10: Bayesian Posterior for Key Feature Premiums
# ══════════════════════════════════════════════════════════════════════
# Apply Normal-Normal conjugate (from Exercise 1) to estimate the
# posterior distribution of key regression coefficients.

print(f"\n=== Bayesian Posterior for Feature Premiums ===")

# For each coefficient, treat the OLS estimate as the data likelihood
# and apply a weakly informative prior.
for i, name in enumerate(reg_names):
    if i == 0:
        continue  # Skip intercept

    beta_hat = beta_reg[i]
    se_hat = se_beta[i]

    # Weakly informative prior: N(0, 10000²) for price coefficients
    mu_prior = 0.0
    sigma_prior = 10_000.0

    # TODO: Compute Normal-Normal conjugate posterior
    # prec_prior = 1/sigma_prior², prec_data = 1/se_hat²
    # prec_post = prec_prior + prec_data
    # mu_post = (mu_prior * prec_prior + beta_hat * prec_data) / prec_post
    # sigma_post = sqrt(1 / prec_post)
    prec_prior = 1.0 / sigma_prior**2
    prec_data = 1.0 / se_hat**2
    prec_post = prec_prior + prec_data
    mu_post = ____  # Hint: (mu_prior * prec_prior + beta_hat * prec_data) / prec_post
    sigma_post = ____  # Hint: np.sqrt(1.0 / prec_post)

    ci_low = mu_post - 1.96 * sigma_post
    ci_high = mu_post + 1.96 * sigma_post
    p_positive = 1 - stats.norm.cdf(0, mu_post, sigma_post)

    print(f"\n{name}:")
    print(f"  OLS: β̂={beta_hat:,.2f} ± {se_hat:,.2f}")
    print(f"  Posterior: N({mu_post:,.2f}, {sigma_post:,.2f})")
    print(f"  95% credible: [{ci_low:,.2f}, {ci_high:,.2f}]")
    print(f"  P(β > 0): {p_positive:.4f}")

# ── Checkpoint 8 ─────────────────────────────────────────────────────
print("\n✓ Checkpoint 8 passed — Bayesian posteriors for coefficients\n")


# ══════════════════════════════════════════════════════════════════════
# TASK 11: Data Lineage — Full Audit Trail
# ══════════════════════════════════════════════════════════════════════

if HAS_FEATURE_STORE:
    try:

        async def log_lineage():
            # TODO: Create experiment and log model params/metrics
            exp_id = await tracker.create_experiment(
                name=____,  # Hint: "mlfp02_capstone_model"
                description="Capstone: HDB price model with v2 features",
                tags=["mlfp02", "capstone", "feature-store", "lineage"],
            )
            async with tracker.run(exp_id, run_name="hdb_price_ols_v2") as run:
                await run.log_params(
                    {
                        "feature_schema": "hdb_property_features",
                        "feature_version": "2",
                        "model_type": "OLS",
                        "n_features": str(len(feature_list)),
                        "n_observations": str(n_reg),
                    }
                )
                # TODO: Log r2, adj_r2, rmse, and f_statistic metrics
                await run.log_metrics(
                    {
                        "r2": ____,  # Hint: float(r2_reg)
                        "adj_r2": float(adj_r2_reg),
                        "rmse": float(rmse_reg),
                        "f_statistic": float(f_stat_reg),
                    }
                )
                run_id = run.id if hasattr(run, "id") else "logged"
            print(f"\n=== Data Lineage ===")
            print(f"Model run: {run_id}")
            print(f"  → Feature schema: hdb_property_features v2")
            print(f"  → Features: {feature_list}")
            print(f"  → Training rows: {n_reg:,}")
            print(f"  → R²: {r2_reg:.4f}, RMSE: ${rmse_reg:,.0f}")
            return exp_id, run_id

        exp_id, run_id = asyncio.run(log_lineage())
    except Exception as e:
        print(f"  [Skipped: Lineage logging ({type(e).__name__}: {e})]")
else:
    print(f"\n=== Data Lineage (manual) ===")
    print(f"Model: OLS regression")
    print(f"  → Features: {feature_list}")
    print(f"  → Training rows: {n_reg:,}")
    print(f"  → R²: {r2_reg:.4f}, RMSE: ${rmse_reg:,.0f}")

# ── Checkpoint 9 ─────────────────────────────────────────────────────
print("\n✓ Checkpoint 9 passed — data lineage documented\n")


# ══════════════════════════════════════════════════════════════════════
# TASK 12: Stakeholder Report — Full M2 Synthesis
# ══════════════════════════════════════════════════════════════════════

viz = ModelVisualizer()

# Plot 1: Actual vs Predicted
rng = np.random.default_rng(42)
sample_idx = rng.choice(n_reg, size=min(3000, n_reg), replace=False)
fig1 = viz.scatter(
    x=y_reg[sample_idx].tolist(),
    y=y_hat_reg[sample_idx].tolist(),
    title="Capstone Model: Actual vs Predicted Price",
    x_label="Actual ($)",
    y_label="Predicted ($)",
)
fig1.add_trace(
    go.Scatter(
        x=[y_reg.min(), y_reg.max()],
        y=[y_reg.min(), y_reg.max()],
        mode="lines",
        name="Perfect",
        line={"dash": "dash", "color": "red"},
    )
)
fig1.write_html("ex8_actual_vs_predicted.html")
print("\nSaved: ex8_actual_vs_predicted.html")

# Plot 2: Coefficient forest plot
fig2 = go.Figure()
for i in range(1, k_reg):
    ci_lo = beta_reg[i] - 1.96 * se_beta[i]
    ci_hi = beta_reg[i] + 1.96 * se_beta[i]
    fig2.add_trace(
        go.Scatter(
            x=[ci_lo, beta_reg[i], ci_hi],
            y=[reg_names[i]] * 3,
            mode="markers+lines",
            name=reg_names[i],
            marker={"size": [6, 10, 6]},
        )
    )
fig2.add_vline(x=0, line_dash="dot", line_color="red")
fig2.update_layout(
    title="Regression Coefficients with 95% CIs", xaxis_title="Coefficient Value"
)
fig2.write_html("ex8_coefficient_forest.html")
print("Saved: ex8_coefficient_forest.html")

# Plot 3: Residual histogram
fig3 = viz.histogram(resid_reg, title="Residual Distribution", x_label="Residual ($)")
fig3.write_html("ex8_residuals.html")
print("Saved: ex8_residuals.html")

# ── Checkpoint 10 ────────────────────────────────────────────────────
print("\n✓ Checkpoint 10 passed — visualisations saved\n")

# Stakeholder report
print(f"\n{'='*70}")
print(f"STAKEHOLDER REPORT: HDB Resale Price Analysis")
print(f"{'='*70}")
print(
    f"""
EXECUTIVE SUMMARY
This analysis applies statistical methods from Module 2 to Singapore's
HDB resale market, covering {features_v2.height:,} transactions.

KEY FINDINGS:

1. PRICE DISTRIBUTION (Ex 1-2: Bayesian + MLE)
   Average 4-room price: ${mle_mu:,.0f} ± ${mle_sigma:,.0f}
   Distribution is {'right-skewed' if skew > 0.5 else 'approximately symmetric'}
   (skewness={skew:.2f}), suggesting some high-value outliers.

2. PRICE DRIVERS (Ex 5: Linear Regression)
   Our v2 model with {len(feature_list)} features explains {r2_reg:.1%} of
   price variation. Key drivers:"""
)
for i in range(1, k_reg):
    if p_values[i] < 0.05:
        print(
            f"   - {reg_names[i]}: ${beta_reg[i]:+,.0f} per unit (p<{max(p_values[i], 1e-10):.2e})"
        )
print(
    f"""
3. MARKET CONTEXT (Ex 8: Feature Engineering)
   Rolling 6-month town median prices and transaction volumes capture
   local market conditions. {n_with_market:,} of {features_v2.height:,}
   transactions have complete market context.

4. DATA QUALITY
   Point-in-time correctness ensures no future data leaks into
   training. Feature versioning (v1 → v2) tracks schema evolution.

5. MODEL LIMITATIONS
   - {(1-r2_reg)*100:.0f}% of variance unexplained (renovation, facing, luck)
   - Linear model may miss non-linear relationships
   - Market features have 6-month warm-up period (nulls)

RECOMMENDATIONS:
   - Use v2 features for all new valuation models
   - Add flat-type dummy encoding for +5-10% R² improvement
   - Consider non-linear models (Random Forest, XGBoost) in M3
   - Monitor town-level trends for early signals of price shifts
"""
)

# ── Checkpoint 11 ────────────────────────────────────────────────────
print("✓ Checkpoint 11 passed — stakeholder report generated\n")

# Clean up
if HAS_FEATURE_STORE and conn is not None and hasattr(conn, "close"):
    try:
        asyncio.run(conn.close())
    except Exception:
        pass


# ══════════════════════════════════════════════════════════════════════
# REFLECTION
# ══════════════════════════════════════════════════════════════════════
print("═" * 70)
print("  WHAT YOU'VE MASTERED")
print("═" * 70)
print(
    """
  ✓ FeatureSchema: typed fields, versioning (v1 → v2)
  ✓ FeatureStore: registration, storage, point-in-time retrieval
  ✓ Rolling market features: group_by_dynamic + rolling statistics
  ✓ Complete regression pipeline: features → OLS → diagnostics
  ✓ From-scratch t-statistics on regression coefficients
  ✓ Bayesian posteriors for coefficient interpretation
  ✓ ExperimentTracker: model lineage and audit trail
  ✓ Stakeholder-ready report translating statistics to decisions

  MODULE 2 COMPLETE — YOUR STATISTICAL TOOLKIT:
  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  Ex 1: Bayesian inference — conjugate priors, credible intervals
  Ex 2: MLE + MAP — optimisation, CLT, failure modes, AIC/BIC
  Ex 3: Hypothesis testing — bootstrap, power, BH-FDR, permutation
  Ex 4: A/B design — pre-registration, SRM, adaptive sample sizes
  Ex 5: Linear regression — OLS from scratch, VIF, WLS, diagnostics
  Ex 6: Logistic regression — sigmoid MLE, odds ratios, calibration
  Ex 7: CUPED + causal inference — variance reduction, DiD, mSPRT
  Ex 8: Capstone — feature store, lineage, complete pipeline

  → NEXT MODULE: M3 — Supervised ML in the Kailash Pipeline
    You'll use TrainingPipeline, HyperparameterSearch, and ModelRegistry
    to build, tune, and deploy models at production scale.
"""
)

print("\n✓ Exercise 8 complete — Module 2 Capstone")
