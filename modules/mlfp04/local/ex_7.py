# Copyright 2026 Terrene Foundation
# SPDX-License-Identifier: Apache-2.0
"""
# ════════════════════════════════════════════════════════════════════════
# MLFP04 — Exercise 7: Recommender Systems and Collaborative Filtering
# ════════════════════════════════════════════════════════════════════════
#
# WHAT YOU'LL LEARN:
#   After completing this exercise, you will be able to:
#   - Build content-based, user-based, and item-based recommenders
#   - Implement ALS matrix factorisation from scratch
#   - Evaluate recommenders using RMSE, precision@k, MAP
#   - Build a hybrid recommender combining CF and content-based
#   - Explain implicit vs explicit feedback and SVD++
#   - Visualise learned embeddings (2D PCA projection)
#   - Articulate THE PIVOT: optimisation drives feature discovery
#   - Explain the connection: matrix factorisation -> neural embeddings
#
# PREREQUISITES:
#   - MLFP04 Exercise 3 (PCA/SVD — matrix factorisation is generalised SVD)
#   - MLFP04 Exercise 6 (text embeddings — same concept, different domain)
#
# ESTIMATED TIME: ~150-180 min
#
# TASKS:
#   1.  Generate synthetic user-item rating matrix
#   2.  Content-based filtering using item features
#   3.  User-based collaborative filtering
#   4.  Item-based collaborative filtering
#   5.  Matrix factorisation with ALS from scratch
#   6.  Evaluation: RMSE, precision@k, MAP
#   7.  Hybrid recommender: weighted combination of CF + content
#   8.  Implicit vs explicit feedback, SVD++ concept
#   9.  Visualise learned embeddings
#   10. THE PIVOT: optimisation drives feature discovery
#
# DATASET: Synthetic user-item ratings (100 users x 50 items, 30% observed)
#   True latent dimension: 5 (known ground truth for validation)
#
# ════════════════════════════════════════════════════════════════════════
"""
from __future__ import annotations

import numpy as np
import polars as pl

from kailash_ml import ModelVisualizer
from sklearn.decomposition import PCA


# ══════════════════════════════════════════════════════════════════════
# TASK 1: Generate synthetic user-item rating matrix
# ══════════════════════════════════════════════════════════════════════

rng = np.random.default_rng(seed=42)

N_USERS = 100
N_ITEMS = 50
N_LATENT_TRUE = 5
SPARSITY = 0.30

U_true = rng.normal(0, 1, size=(N_USERS, N_LATENT_TRUE))
V_true = rng.normal(0, 1, size=(N_ITEMS, N_LATENT_TRUE))

R_full = U_true @ V_true.T
R_full = (R_full - R_full.min()) / (R_full.max() - R_full.min()) * 4 + 1
R_full += rng.normal(0, 0.3, size=R_full.shape)
R_full = np.clip(R_full, 1.0, 5.0)

mask = rng.random(size=(N_USERS, N_ITEMS)) < SPARSITY
R_observed = np.where(mask, R_full, np.nan)

n_observed = int(mask.sum())
print("=== Synthetic Rating Matrix ===")
print(f"Users: {N_USERS}, Items: {N_ITEMS}")
print(f"True latent dimension: {N_LATENT_TRUE}")
print(
    f"Observed ratings: {n_observed:,} / {N_USERS * N_ITEMS:,} ({n_observed / (N_USERS * N_ITEMS):.1%})"
)
print(f"Rating range: {np.nanmin(R_observed):.2f} - {np.nanmax(R_observed):.2f}")
print(f"Mean rating: {np.nanmean(R_observed):.2f}")

user_ids = [f"user_{i:03d}" for i in range(N_USERS)]
item_ids = [f"item_{j:02d}" for j in range(N_ITEMS)]

ratings_long = []
for i in range(N_USERS):
    for j in range(N_ITEMS):
        if mask[i, j]:
            ratings_long.append(
                {
                    "user_id": user_ids[i],
                    "item_id": item_ids[j],
                    "rating": round(R_observed[i, j], 1),
                }
            )

ratings_df = pl.DataFrame(ratings_long)
print(f"\nRatings DataFrame: {ratings_df.shape}")

# Create holdout set for evaluation
holdout_frac = 0.2
holdout_mask = mask & (rng.random(size=(N_USERS, N_ITEMS)) < holdout_frac)
train_mask = mask & ~holdout_mask
n_train = int(train_mask.sum())
n_holdout = int(holdout_mask.sum())
print(f"Train: {n_train:,}, Holdout: {n_holdout:,}")

R_train = np.where(train_mask, R_observed, np.nan)

# Item features for content-based
N_ITEM_FEATURES = 8
item_features = rng.random(size=(N_ITEMS, N_ITEM_FEATURES))
item_norms = np.linalg.norm(item_features, axis=1, keepdims=True)
item_features_normed = item_features / np.maximum(item_norms, 1e-10)

# ── Checkpoint 1 ─────────────────────────────────────────────────────
assert n_observed > 0, "Should have observed ratings"
assert n_holdout > 0, "Should have holdout ratings"
assert n_train > 0, "Should have training ratings"
print("\n✓ Checkpoint 1 passed — rating matrix with train/holdout split\n")


# ══════════════════════════════════════════════════════════════════════
# TASK 2: Content-Based Filtering
# ══════════════════════════════════════════════════════════════════════


def content_based_predict(
    R: np.ndarray,
    item_feats: np.ndarray,
    obs_mask: np.ndarray,
) -> np.ndarray:
    """Predict ratings using content-based filtering."""
    n_users, n_items = R.shape
    predictions = np.full((n_users, n_items), np.nan)

    for u in range(n_users):
        rated_idx = np.where(obs_mask[u])[0]
        if len(rated_idx) == 0:
            continue

        ratings_u = np.nan_to_num(R[u, rated_idx], nan=0.0)
        profile = (ratings_u[:, None] * item_feats[rated_idx]).sum(axis=0)
        profile_norm = np.linalg.norm(profile)
        if profile_norm < 1e-10:
            continue
        profile /= profile_norm

        for j in range(n_items):
            feat_norm = np.linalg.norm(item_feats[j])
            if feat_norm < 1e-10:
                continue
            sim = profile @ item_feats[j] / feat_norm
            predictions[u, j] = 1.0 + (sim + 1.0) * 2.0

    return predictions


cb_predictions = content_based_predict(R_train, item_features, train_mask)

cb_errors = []
for i in range(N_USERS):
    for j in range(N_ITEMS):
        if holdout_mask[i, j] and not np.isnan(cb_predictions[i, j]):
            cb_errors.append((R_observed[i, j] - cb_predictions[i, j]) ** 2)

cb_rmse = np.sqrt(np.mean(cb_errors)) if cb_errors else float("inf")
cb_coverage = len(cb_errors) / max(n_holdout, 1)

print("=== Content-Based Filtering ===")
print(f"RMSE (holdout): {cb_rmse:.4f}")
print(f"Coverage: {cb_coverage:.1%}")

# ── Checkpoint 2 ─────────────────────────────────────────────────────
assert cb_rmse > 0, "Content-based RMSE should be positive"
print("\n✓ Checkpoint 2 passed — content-based filtering\n")


# ══════════════════════════════════════════════════════════════════════
# TASK 3: User-Based Collaborative Filtering
# ══════════════════════════════════════════════════════════════════════

K_NEIGHBOURS = 20


def user_similarity_matrix(
    R: np.ndarray, obs_mask: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Compute pairwise cosine similarity between users on mean-centred ratings."""
    n_users = R.shape[0]
    sim = np.zeros((n_users, n_users))

    user_means = np.array(
        [
            np.nanmean(R[u, obs_mask[u]]) if obs_mask[u].any() else 0.0
            for u in range(n_users)
        ]
    )
    R_centred = R.copy()
    for u in range(n_users):
        R_centred[u, obs_mask[u]] -= user_means[u]
    R_centred[~obs_mask] = 0.0

    for u in range(n_users):
        for v in range(u, n_users):
            both = obs_mask[u] & obs_mask[v]
            if not both.any():
                continue
            ru, rv = R_centred[u, both], R_centred[v, both]
            denom = np.linalg.norm(ru) * np.linalg.norm(rv)
            if denom < 1e-10:
                continue
            # TODO: Compute cosine similarity = dot product / (norm_u * norm_v)
            s = ____  # Hint: ru @ rv / denom
            sim[u, v] = s
            sim[v, u] = s

    return sim, user_means


def user_based_cf_predict(
    R: np.ndarray,
    obs_mask: np.ndarray,
    sim: np.ndarray,
    user_means: np.ndarray,
    k: int = 20,
) -> np.ndarray:
    """Predict ratings using user-based collaborative filtering."""
    n_users, n_items = R.shape
    predictions = np.full((n_users, n_items), np.nan)

    for u in range(n_users):
        similarities = sim[u].copy()
        similarities[u] = -np.inf
        top_k = np.argsort(similarities)[-k:]
        top_k = top_k[similarities[top_k] > 0]

        if len(top_k) == 0:
            continue

        for j in range(n_items):
            rated_neighbours = top_k[obs_mask[top_k, j]]
            if len(rated_neighbours) == 0:
                continue
            weights = sim[u, rated_neighbours]
            denom = np.abs(weights).sum()
            if denom < 1e-10:
                continue
            weighted_dev = weights @ (
                R[rated_neighbours, j] - user_means[rated_neighbours]
            )
            predictions[u, j] = user_means[u] + weighted_dev / denom

    return np.clip(predictions, 1.0, 5.0)


print("=== User-Based Collaborative Filtering ===")
user_sim, user_means = user_similarity_matrix(R_train, train_mask)
ubcf_predictions = user_based_cf_predict(
    R_train, train_mask, user_sim, user_means, k=K_NEIGHBOURS
)

ubcf_errors = []
for i in range(N_USERS):
    for j in range(N_ITEMS):
        if holdout_mask[i, j] and not np.isnan(ubcf_predictions[i, j]):
            ubcf_errors.append((R_observed[i, j] - ubcf_predictions[i, j]) ** 2)

ubcf_rmse = np.sqrt(np.mean(ubcf_errors)) if ubcf_errors else float("inf")
ubcf_coverage = len(ubcf_errors) / max(n_holdout, 1)

print(f"RMSE (holdout): {ubcf_rmse:.4f}, Coverage: {ubcf_coverage:.1%}")

# ── Checkpoint 3 ─────────────────────────────────────────────────────
assert ubcf_rmse > 0, "User-based CF RMSE should be positive"
assert user_sim.shape == (N_USERS, N_USERS), "User similarity should be NxN"
print("\n✓ Checkpoint 3 passed — user-based CF\n")


# ══════════════════════════════════════════════════════════════════════
# TASK 4: Item-Based Collaborative Filtering
# ══════════════════════════════════════════════════════════════════════


def item_similarity_matrix(
    R: np.ndarray, obs_mask: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Compute pairwise cosine similarity between items."""
    n_items = R.shape[1]
    sim = np.zeros((n_items, n_items))
    item_means = np.array(
        [
            np.nanmean(R[obs_mask[:, j], j]) if obs_mask[:, j].any() else 0.0
            for j in range(n_items)
        ]
    )
    R_centred = R.copy()
    for j in range(n_items):
        R_centred[obs_mask[:, j], j] -= item_means[j]
    R_centred[~obs_mask] = 0.0

    for i in range(n_items):
        for j in range(i, n_items):
            both = obs_mask[:, i] & obs_mask[:, j]
            if not both.any():
                continue
            ri, rj = R_centred[both, i], R_centred[both, j]
            denom = np.linalg.norm(ri) * np.linalg.norm(rj)
            if denom < 1e-10:
                continue
            s = ri @ rj / denom
            sim[i, j] = s
            sim[j, i] = s
    return sim, item_means


def item_based_cf_predict(
    R: np.ndarray,
    obs_mask: np.ndarray,
    item_sim: np.ndarray,
    k: int = 20,
) -> np.ndarray:
    """Predict ratings using item-based collaborative filtering."""
    n_users, n_items = R.shape
    predictions = np.full((n_users, n_items), np.nan)

    for u in range(n_users):
        rated_items = np.where(obs_mask[u])[0]
        if len(rated_items) == 0:
            continue
        for j in range(n_items):
            sims = item_sim[j, rated_items]
            if len(sims) > k:
                top_idx = np.argsort(sims)[-k:]
            else:
                top_idx = np.arange(len(sims))
            pos_idx = top_idx[sims[top_idx] > 0]
            if len(pos_idx) == 0:
                continue
            weights = sims[pos_idx]
            denom = np.abs(weights).sum()
            if denom < 1e-10:
                continue
            predictions[u, j] = weights @ R[u, rated_items[pos_idx]] / denom

    return np.clip(predictions, 1.0, 5.0)


print("=== Item-Based Collaborative Filtering ===")
item_sim, item_means = item_similarity_matrix(R_train, train_mask)
ibcf_predictions = item_based_cf_predict(R_train, train_mask, item_sim, k=K_NEIGHBOURS)

ibcf_errors = []
for i in range(N_USERS):
    for j in range(N_ITEMS):
        if holdout_mask[i, j] and not np.isnan(ibcf_predictions[i, j]):
            ibcf_errors.append((R_observed[i, j] - ibcf_predictions[i, j]) ** 2)

ibcf_rmse = np.sqrt(np.mean(ibcf_errors)) if ibcf_errors else float("inf")
ibcf_coverage = len(ibcf_errors) / max(n_holdout, 1)

print(f"RMSE (holdout): {ibcf_rmse:.4f}, Coverage: {ibcf_coverage:.1%}")

# ── Checkpoint 4 ─────────────────────────────────────────────────────
assert ibcf_rmse > 0, "Item-based CF RMSE should be positive"
assert item_sim.shape == (N_ITEMS, N_ITEMS), "Item sim should be MxM"
print("\n✓ Checkpoint 4 passed — item-based CF\n")


# ══════════════════════════════════════════════════════════════════════
# TASK 5: Matrix Factorisation with ALS (from scratch)
# ══════════════════════════════════════════════════════════════════════

K_LATENT = 10
LAMBDA_REG = 0.1
N_ITERATIONS = 50


def als_matrix_factorisation(
    R: np.ndarray,
    obs_mask: np.ndarray,
    k: int,
    lam: float,
    n_iter: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray, list[float]]:
    """Alternating Least Squares matrix factorisation."""
    n_users, n_items = R.shape
    U = rng.normal(0, 0.1, size=(n_users, k))
    V = rng.normal(0, 0.1, size=(n_items, k))
    R_safe = np.nan_to_num(R, nan=0.0)
    errors = []
    identity = lam * np.eye(k)

    for iteration in range(n_iter):
        # Fix V, solve for each user
        for u in range(n_users):
            rated = np.where(obs_mask[u])[0]
            if len(rated) == 0:
                continue
            V_u = V[rated]
            r_u = R_safe[u, rated]
            # TODO: Solve ALS update for user u: A = V_u.T @ V_u + identity, b = V_u.T @ r_u
            A = ____  # Hint: V_u.T @ V_u + identity
            b = ____  # Hint: V_u.T @ r_u
            U[u] = np.linalg.solve(A, b)

        # Fix U, solve for each item
        for j in range(n_items):
            raters = np.where(obs_mask[:, j])[0]
            if len(raters) == 0:
                continue
            U_j = U[raters]
            r_j = R_safe[raters, j]
            A = U_j.T @ U_j + identity
            b = U_j.T @ r_j
            V[j] = np.linalg.solve(A, b)

        R_hat = U @ V.T
        residuals = (R_safe - R_hat)[obs_mask]
        rmse = np.sqrt(np.mean(residuals**2))
        errors.append(rmse)

        if iteration % 10 == 0 or iteration == n_iter - 1:
            print(f"  Iteration {iteration:3d}: RMSE = {rmse:.4f}")

    return U, V, errors


print("=== Matrix Factorisation with ALS ===")
print(f"k={K_LATENT}, lambda={LAMBDA_REG}, iterations={N_ITERATIONS}")

U_learned, V_learned, als_errors = als_matrix_factorisation(
    R_train,
    train_mask,
    k=K_LATENT,
    lam=LAMBDA_REG,
    n_iter=N_ITERATIONS,
    rng=rng,
)

# TODO: Compute predicted rating matrix from learned U and V factors
als_R_hat = ____  # Hint: np.clip(U_learned @ V_learned.T, 1.0, 5.0)

als_holdout_errors = []
for i in range(N_USERS):
    for j in range(N_ITEMS):
        if holdout_mask[i, j]:
            als_holdout_errors.append((R_observed[i, j] - als_R_hat[i, j]) ** 2)

als_rmse = np.sqrt(np.mean(als_holdout_errors))
als_coverage = 1.0

print(f"\nHoldout RMSE: {als_rmse:.4f}")
print(f"Coverage: {als_coverage:.0%} (MF predicts all pairs)")

# ── Checkpoint 5 ─────────────────────────────────────────────────────
assert als_rmse < als_errors[0], "ALS should improve over random init"
for i in range(1, len(als_errors)):
    assert (
        als_errors[i] <= als_errors[i - 1] + 0.01
    ), f"ALS RMSE should be non-increasing at step {i}"
print("\n✓ Checkpoint 5 passed — ALS converged with non-increasing RMSE\n")


# ══════════════════════════════════════════════════════════════════════
# TASK 6: Evaluation — RMSE, precision@k, MAP
# ══════════════════════════════════════════════════════════════════════


def precision_at_k(
    predictions: np.ndarray,
    R_true: np.ndarray,
    holdout: np.ndarray,
    k: int = 5,
    threshold: float = 3.5,
) -> float:
    """Compute precision@k: fraction of top-k recommendations that are relevant."""
    precisions = []
    for u in range(predictions.shape[0]):
        holdout_items = np.where(holdout[u])[0]
        if len(holdout_items) == 0:
            continue
        relevant = set(j for j in holdout_items if R_true[u, j] >= threshold)
        if len(relevant) == 0:
            continue
        holdout_pred = [
            (j, predictions[u, j])
            for j in holdout_items
            if not np.isnan(predictions[u, j])
        ]
        holdout_pred.sort(key=lambda x: -x[1])
        recommended = set(j for j, _ in holdout_pred[:k])
        if len(recommended) == 0:
            continue
        precisions.append(len(recommended & relevant) / len(recommended))
    return np.mean(precisions) if precisions else 0.0


def mean_average_precision(
    predictions: np.ndarray,
    R_true: np.ndarray,
    holdout: np.ndarray,
    threshold: float = 3.5,
) -> float:
    """Compute MAP: mean of per-user average precision."""
    aps = []
    for u in range(predictions.shape[0]):
        holdout_items = np.where(holdout[u])[0]
        if len(holdout_items) == 0:
            continue
        relevant = set(j for j in holdout_items if R_true[u, j] >= threshold)
        if len(relevant) == 0:
            continue
        holdout_pred = [
            (j, predictions[u, j])
            for j in holdout_items
            if not np.isnan(predictions[u, j])
        ]
        holdout_pred.sort(key=lambda x: -x[1])

        hits = 0
        sum_precision = 0.0
        for rank, (j, _) in enumerate(holdout_pred, 1):
            if j in relevant:
                hits += 1
                sum_precision += hits / rank
        if hits > 0:
            aps.append(sum_precision / len(relevant))
    return np.mean(aps) if aps else 0.0


print("=== Comprehensive Evaluation ===")
all_predictions = {
    "Content-Based": cb_predictions,
    "User-Based CF": ubcf_predictions,
    "Item-Based CF": ibcf_predictions,
    "ALS MF": als_R_hat,
}

print(f"{'Method':<20} {'RMSE':>8} {'Coverage':>10} {'P@5':>8} {'MAP':>8}")
print("─" * 56)

eval_results = {}
for name, preds in all_predictions.items():
    errors = []
    for i in range(N_USERS):
        for j in range(N_ITEMS):
            if holdout_mask[i, j] and not np.isnan(preds[i, j]):
                errors.append((R_observed[i, j] - preds[i, j]) ** 2)
    rmse = np.sqrt(np.mean(errors)) if errors else float("inf")
    cov = len(errors) / max(n_holdout, 1)
    p5 = precision_at_k(preds, R_observed, holdout_mask, k=5)
    map_score = mean_average_precision(preds, R_observed, holdout_mask)

    eval_results[name] = {"RMSE": rmse, "Coverage": cov, "P@5": p5, "MAP": map_score}
    print(f"{name:<20} {rmse:>8.4f} {cov:>9.1%} {p5:>8.4f} {map_score:>8.4f}")

# ── Checkpoint 6 ─────────────────────────────────────────────────────
assert len(eval_results) == 4, "Should evaluate all 4 methods"
# INTERPRETATION: RMSE measures rating prediction accuracy. Precision@k and
# MAP measure ranking quality (are the top recommendations actually good?).
# For production systems, ranking metrics matter more than RMSE.
print("\n✓ Checkpoint 6 passed — RMSE, P@5, MAP evaluation\n")


# ══════════════════════════════════════════════════════════════════════
# TASK 7: Hybrid recommender
# ══════════════════════════════════════════════════════════════════════

print("=== Hybrid Recommender ===")


def normalise_preds(preds: np.ndarray) -> np.ndarray:
    valid = ~np.isnan(preds)
    result = preds.copy()
    if valid.sum() > 0:
        pmin, pmax = np.nanmin(preds), np.nanmax(preds)
        if pmax > pmin:
            result[valid] = (preds[valid] - pmin) / (pmax - pmin)
    return result


# TODO: Compute MAP-based weights: weight_name = max(MAP, 0.01) / total_map
map_scores = {name: res["MAP"] for name, res in eval_results.items()}
total_map = sum(max(m, 0.01) for m in map_scores.values())
weights = (
    ____  # Hint: {name: max(m, 0.01) / total_map for name, m in map_scores.items()}
)

print(f"Blending weights (MAP-based):")
for name, w in weights.items():
    print(f"  {name:<20}: {w:.3f}")

hybrid_preds = np.zeros((N_USERS, N_ITEMS))
for name, preds in all_predictions.items():
    norm_preds = normalise_preds(preds)
    norm_preds = np.nan_to_num(norm_preds, nan=0.5)
    hybrid_preds += weights[name] * norm_preds

# Scale back to rating range
hybrid_preds = hybrid_preds * 4 + 1
hybrid_preds = np.clip(hybrid_preds, 1.0, 5.0)

# Evaluate hybrid
hybrid_errors = []
for i in range(N_USERS):
    for j in range(N_ITEMS):
        if holdout_mask[i, j]:
            hybrid_errors.append((R_observed[i, j] - hybrid_preds[i, j]) ** 2)

hybrid_rmse = np.sqrt(np.mean(hybrid_errors))
hybrid_p5 = precision_at_k(hybrid_preds, R_observed, holdout_mask, k=5)
hybrid_map = mean_average_precision(hybrid_preds, R_observed, holdout_mask)

print(f"\nHybrid: RMSE={hybrid_rmse:.4f}, P@5={hybrid_p5:.4f}, MAP={hybrid_map:.4f}")

best_single_map = max(r["MAP"] for r in eval_results.values())
print(f"Best single method MAP: {best_single_map:.4f}")
print(f"Hybrid MAP improvement: {hybrid_map - best_single_map:+.4f}")

eval_results["Hybrid"] = {
    "RMSE": hybrid_rmse,
    "Coverage": 1.0,
    "P@5": hybrid_p5,
    "MAP": hybrid_map,
}

# ── Checkpoint 7 ─────────────────────────────────────────────────────
assert hybrid_rmse > 0, "Hybrid RMSE should be positive"
assert abs(sum(weights.values()) - 1.0) < 1e-6, "Weights should sum to 1"
# INTERPRETATION: Hybrid recommenders combine strengths of multiple approaches.
# Content-based handles cold-start; CF captures community preferences; MF
# learns latent structure. The Netflix Prize winner used 107 blended models.
print("\n✓ Checkpoint 7 passed — hybrid recommender\n")


# ══════════════════════════════════════════════════════════════════════
# TASK 8: Implicit vs explicit feedback, SVD++ concept
# ══════════════════════════════════════════════════════════════════════

print("=== Implicit vs Explicit Feedback ===")
print(
    """
Explicit feedback: users provide ratings (1-5 stars).
  - Clear signal but sparse (most users rate few items)
  - Absence of rating is ambiguous (dislike? never seen?)

Implicit feedback: clicks, views, purchases, time spent.
  - Abundant but noisy (a click != a like)
  - No NEGATIVE signal: absence of click != dislike (maybe never seen)

ALS for implicit data (Hu et al. 2008):
  - Treat ALL entries as observed (not just rated ones)
  - Binary preference: p_ui = 1 if user interacted, 0 otherwise
  - Confidence: c_ui = 1 + alpha * count (more interactions = higher confidence)
  - Loss: sum_ALL (c_ui)(p_ui - u^T v)^2 + lambda(||U||^2 + ||V||^2)

SVD++ extends SVD with implicit feedback:
  - r_hat(u,i) = mu + b_u + b_i + q_i^T (p_u + |N(u)|^{-0.5} sum_{j in N(u)} y_j)
  - N(u) = items user interacted with (implicit set)
  - y_j = implicit feedback vectors
  - Captures: "users who clicked on these items tend to have these preferences"

Both ALS-implicit and SVD++ are widely used in production (Spotify, Netflix).
"""
)

# Simulate implicit feedback from our rating data
implicit_interactions = R_observed > 3.0  # "liked" = rated > 3
implicit_counts = (
    np.nan_to_num(R_observed, nan=0) * mask
)  # Use ratings as interaction counts

n_implicit = int(implicit_interactions[mask].sum())
print(f"Simulated implicit interactions (rating > 3): {n_implicit:,} / {n_observed:,}")
print(f"Implicit interaction rate: {n_implicit / n_observed:.1%}")

# ── Checkpoint 8 ─────────────────────────────────────────────────────
assert n_implicit > 0, "Should have some implicit interactions"
# INTERPRETATION: Implicit feedback is the dominant paradigm in modern
# recommender systems. Most users never explicitly rate; they click, view,
# and purchase. Converting these signals to useful training data is critical.
print("\n✓ Checkpoint 8 passed — implicit feedback concepts\n")


# ══════════════════════════════════════════════════════════════════════
# TASK 9: Visualise learned embeddings
# ══════════════════════════════════════════════════════════════════════

viz = ModelVisualizer()

# TODO: Project user embeddings to 2D using PCA
pca_users = ____  # Hint: PCA(n_components=2, random_state=42)
U_2d = ____  # Hint: pca_users.fit_transform(U_learned)

print("=== Embedding Visualisation ===")
print(f"User PCA explained variance: {pca_users.explained_variance_ratio_.sum():.1%}")

user_embed_df = pl.DataFrame(
    {
        "user_id": user_ids,
        "pc1": U_2d[:, 0].tolist(),
        "pc2": U_2d[:, 1].tolist(),
        "avg_rating": [
            float(np.nanmean(R_observed[i, mask[i]])) if mask[i].any() else 0.0
            for i in range(N_USERS)
        ],
    }
)

fig_users = viz.scatter(user_embed_df, x="pc1", y="pc2", color="avg_rating")
fig_users.update_layout(title="User Embeddings (ALS, PCA projection)")
fig_users.write_html("ex7_user_embeddings.html")
print("Saved: ex7_user_embeddings.html")

# Item embeddings
pca_items = PCA(n_components=2, random_state=42)
V_2d = pca_items.fit_transform(V_learned)
print(f"Item PCA explained variance: {pca_items.explained_variance_ratio_.sum():.1%}")

item_embed_df = pl.DataFrame(
    {
        "item_id": item_ids,
        "pc1": V_2d[:, 0].tolist(),
        "pc2": V_2d[:, 1].tolist(),
        "avg_rating": [
            float(np.nanmean(R_observed[mask[:, j], j])) if mask[:, j].any() else 0.0
            for j in range(N_ITEMS)
        ],
    }
)

fig_items = viz.scatter(item_embed_df, x="pc1", y="pc2", color="avg_rating")
fig_items.update_layout(title="Item Embeddings (ALS, PCA projection)")
fig_items.write_html("ex7_item_embeddings.html")

# Convergence plot
fig_conv = viz.training_history({"ALS RMSE": als_errors}, x_label="Iteration")
fig_conv.update_layout(title="ALS Convergence")
fig_conv.write_html("ex7_als_convergence.html")

# Method comparison
comparison_metrics = {
    name: {"RMSE": r["RMSE"], "MAP": r["MAP"]} for name, r in eval_results.items()
}
fig_compare = viz.metric_comparison(comparison_metrics)
fig_compare.update_layout(title="Recommender System Comparison")
fig_compare.write_html("ex7_method_comparison.html")
print(
    "Saved: ex7_item_embeddings.html, ex7_als_convergence.html, ex7_method_comparison.html"
)

# Subspace alignment
from numpy.linalg import svd as np_svd


def subspace_similarity(A: np.ndarray, B: np.ndarray) -> float:
    """Measure alignment via principal angles."""
    Qa, _ = np.linalg.qr(A)
    Qb, _ = np.linalg.qr(B)
    _, sigmas, _ = np_svd(Qa.T @ Qb, full_matrices=False)
    return float(np.mean(np.minimum(sigmas, 1.0)))


user_sub_sim = subspace_similarity(U_learned[:, :N_LATENT_TRUE], U_true)
item_sub_sim = subspace_similarity(V_learned[:, :N_LATENT_TRUE], V_true)

print(f"\nSubspace alignment (1.0 = perfect recovery):")
print(f"  User factors: {user_sub_sim:.3f}")
print(f"  Item factors: {item_sub_sim:.3f}")

# ── Checkpoint 9 ─────────────────────────────────────────────────────
assert user_sub_sim > 0, "User subspace alignment should be positive"
assert U_learned.shape == (N_USERS, K_LATENT), "U should be (N_USERS, K)"
print("\n✓ Checkpoint 9 passed — embeddings visualised and aligned\n")


# ══════════════════════════════════════════════════════════════════════
# TASK 10: THE PIVOT — optimisation drives feature discovery
# ══════════════════════════════════════════════════════════════════════

print("=" * 70)
print("THE PIVOT: Optimisation Drives Feature Discovery")
print("=" * 70)
print(
    """
What we built:  R  ≈  U  *  V^T
  (100x50)      (100x10)  (10x50)

  U = user embeddings, V = item embeddings
  Nobody told the model what those 10 dimensions mean.
  ALS DISCOVERED them by minimising reconstruction error.

Connection to PCA (lesson 4.3):
  PCA: X = U Sigma V^T   (SVD on dense matrix)
  ALS: R ≈ U V^T         (factorisation on sparse matrix)
  Both find low-rank structure by minimising reconstruction error.

THE BRIDGE to Neural Networks (lesson 4.8):
  Neural network hidden activations h = f(Wx + b) are embeddings.
  They are learned by minimising a loss function (backpropagation).

  Matrix Factorisation:  loss = ||R - U V^T||^2
  Neural Network:        loss = ||y - f(Wx + b)||^2

  SAME PRINCIPLE: optimisation discovers features automatically.
  The difference: neural networks add nonlinear activations.

THE COMPLETE CHAIN:
  Association rules (Ex 5) -> co-occurrence features (manual)
  Matrix factorisation (Ex 7) -> latent factors (automatic, linear)
  Neural networks (Ex 8) -> latent representations (automatic, nonlinear)

"You don't engineer features. You engineer a loss function,
 and the optimisation process discovers the features for you."
"""
)

# ── Checkpoint 10 ────────────────────────────────────────────────────
# INTERPRETATION: THE PIVOT is the most important conceptual moment in M4.
# Matrix factorisation learns embeddings by optimisation. Neural networks
# generalise this with nonlinear activation functions. The principle is
# identical: minimise a loss, discover features automatically.
print("\n✓ Checkpoint 10 passed — THE PIVOT articulated\n")

print("=" * 70)
print("Exercise 7 complete — Recommender Systems and THE PIVOT")
print("=" * 70)


# ══════════════════════════════════════════════════════════════════════
# REFLECTION
# ══════════════════════════════════════════════════════════════════════
print("═" * 70)
print("  WHAT YOU'VE MASTERED")
print("═" * 70)
print(
    f"""
  ✓ Content-based: user profile from rated items -> cosine similarity
  ✓ User-based CF: similar users -> borrow preferences
  ✓ Item-based CF: similar items -> more stable (Amazon scale)
  ✓ ALS matrix factorisation: R ≈ U * V^T, non-increasing RMSE
  ✓ Evaluation: RMSE + precision@k + MAP (ranking > rating accuracy)
  ✓ Hybrid recommender: weighted blend of all approaches
  ✓ Implicit feedback: ALS for clicks/views, SVD++ with implicit features
  ✓ Embeddings: learned representations, visualised in 2D

  THE PIVOT — COMPLETE CHAIN:
    Association rules -> co-occurrence (manual)
    Matrix factorisation -> latent factors (automatic, linear)
    Neural networks -> latent representations (automatic, nonlinear)

  NEXT: Exercise 8 completes the USML bridge. You'll build neural networks
  from scratch and see that hidden layer activations ARE embeddings,
  learned by the same principle: minimise a loss, discover features.
"""
)
print("═" * 70)
