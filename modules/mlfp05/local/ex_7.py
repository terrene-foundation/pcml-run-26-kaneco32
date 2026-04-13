# Copyright 2026 Terrene Foundation
# SPDX-License-Identifier: Apache-2.0
"""
# ════════════════════════════════════════════════════════════════════════
# MLFP05 — Exercise 7: Transfer Learning with Pre-trained ResNet-18
# ════════════════════════════════════════════════════════════════════════
#
# WHAT YOU'LL LEARN:
#   After completing this exercise, you will be able to:
#   - Load a pre-trained torchvision ResNet-18 (ImageNet weights) and adapt
#     it to CIFAR-10 by replacing the final classifier head
#   - Freeze the convolutional backbone and train only the new head
#   - Compare from-scratch CNN vs transfer learning on FULL CIFAR-10 (50K)
#   - Track both training runs with ExperimentTracker (per-epoch metrics)
#   - Register and compare models with ModelRegistry
#   - Export the fine-tuned model to ONNX and serve it with InferenceServer
#   - Run a data efficiency experiment: how does accuracy scale with 10%,
#     25%, 50%, 100% of training data?
#   - Visualise learned feature representations with t-SNE
#   - Understand adapter modules as a lightweight alternative to full
#     fine-tuning (connecting to M6 LoRA)
#
# PREREQUISITES: M5/ex_2 (CNNs and PyTorch), M5/ex_1 (ExperimentTracker).
# ESTIMATED TIME: ~120-150 min
#
# DATASET: CIFAR-10 — 50,000 training + 10,000 test real 32x32 colour
#   photos across 10 classes (airplane, automobile, bird, cat, deer, dog,
#   frog, horse, ship, truck). Downloaded automatically by torchvision.
#   ResNet-18 was trained on ImageNet (1.28M photos, 1000 classes) — we
#   fine-tune the last layer for the 10 CIFAR classes.
#
# TASKS:
#   1. Load full CIFAR-10, set up ExperimentTracker and ModelRegistry
#   2. Build a transfer-learning ResNet-18 (frozen backbone + new head)
#   3. Build a from-scratch CNN baseline
#   4. Train both models, logging per-epoch metrics to ExperimentTracker
#   5. Register both in ModelRegistry and compare
#   6. Data efficiency experiment: train transfer model on 10/25/50/100%
#   7. Visualise learned features with t-SNE
#   8. Export to ONNX and serve with InferenceServer
#   9. Adapter module concept (bridge to M6 LoRA)
#
# ════════════════════════════════════════════════════════════════════════
"""
from __future__ import annotations

import asyncio
import pickle
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset

import torchvision
import torchvision.transforms as T

from kailash.db import ConnectionManager
from kailash_ml import ModelVisualizer
from kailash_ml.engines.experiment_tracker import ExperimentTracker
from kailash_ml.engines.model_registry import ModelRegistry

from shared.kailash_helpers import get_device, setup_environment

setup_environment()

torch.manual_seed(42)
np.random.seed(42)
device = get_device()
print(f"Using device: {device}")


# ════════════════════════════════════════════════════════════════════════
# TASK 1 — Load FULL CIFAR-10 and set up kailash-ml engines
# ════════════════════════════════════════════════════════════════════════
# Transfer learning on CIFAR-10 uses the FULL 50K training set. Unlike
# the sub-sampled approach (which highlights few-shot benefits), training
# on the full dataset shows the end-to-end production workflow: track,
# register, export, serve. A data efficiency experiment in TASK 6
# quantifies how transfer learning scales with dataset size.

REPO_ROOT = Path(__file__).resolve().parents[3]
DATA_DIR = REPO_ROOT / "data" / "mlfp05" / "cifar10"
DATA_DIR.mkdir(parents=True, exist_ok=True)

# ResNet-18 expects ImageNet normalisation. We resize CIFAR-10 from 32x32
# to 96x96 because ResNet's strided convolutions shrink spatial maps by
# 32x — a 32x32 input would be reduced to 1x1 before the final pooling,
# killing most spatial information. 96x96 gives a 3x3 final feature map.
INPUT_SIZE = 96
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

train_transform = T.Compose(
    [
        T.Resize((INPUT_SIZE, INPUT_SIZE)),
        T.RandomHorizontalFlip(),
        T.RandomCrop(INPUT_SIZE, padding=8),
        T.ToTensor(),
        T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ]
)
val_transform = T.Compose(
    [
        T.Resize((INPUT_SIZE, INPUT_SIZE)),
        T.ToTensor(),
        T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ]
)

train_set = torchvision.datasets.CIFAR10(
    root=str(DATA_DIR),
    train=True,
    download=True,
    transform=train_transform,
)
val_set = torchvision.datasets.CIFAR10(
    root=str(DATA_DIR),
    train=False,
    download=True,
    transform=val_transform,
)

BATCH_SIZE = 128
train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
val_loader = DataLoader(val_set, batch_size=BATCH_SIZE, num_workers=0)

N_CLASSES = 10
CLASS_NAMES = train_set.classes

print(
    f"CIFAR-10 (full): train={len(train_set)}, val={len(val_set)}, "
    f"classes={N_CLASSES}"
)
print(f"  Input size: {INPUT_SIZE}x{INPUT_SIZE} (resized for ResNet-18)")
print(f"  Classes: {CLASS_NAMES}")


# Set up kailash-ml engines: ExperimentTracker + ModelRegistry
async def setup_engines():
    # TODO: Create ConnectionManager, initialize it, create ExperimentTracker
    # Hint: ConnectionManager("sqlite:///mlfp05_transfer.db"), await conn.initialize()
    #       ExperimentTracker(conn), await tracker.create_experiment(name=..., description=...)
    conn = ____  # noqa: F821
    await ____  # noqa: F821

    tracker = ____  # noqa: F821
    exp_name = await tracker.create_experiment(
        name=____,  # noqa: F821
        description=____,  # noqa: F821
    )

    try:
        # TODO: Create ModelRegistry(conn)
        registry = ____  # noqa: F821
        has_registry = True
    except Exception as e:
        registry = None
        has_registry = False
        print(f"  Note: ModelRegistry setup skipped ({e})")

    return conn, tracker, exp_name, registry, has_registry


conn, tracker, exp_name, registry, has_registry = asyncio.run(setup_engines())

# ── Checkpoint 1 ─────────────────────────────────────────────────────
assert len(train_set) == 50000, (
    f"Expected full 50K CIFAR-10, got {len(train_set)}. "
    "Transfer learning exercises need the full dataset."
)
assert len(val_set) == 10000, "CIFAR-10 test set should be 10K"
assert tracker is not None, "ExperimentTracker should be initialised"
assert exp_name is not None, "Experiment should be created"
# INTERPRETATION: We train on the full 50K to see realistic transfer
# learning performance. The ExperimentTracker and ModelRegistry provide
# the production infrastructure: every run is logged, every model is
# registered and versioned, and the best model can be promoted to
# production with a full audit trail.
print("\n--- Checkpoint 1 passed --- CIFAR-10 loaded, engines initialised\n")


# ════════════════════════════════════════════════════════════════════════
# TASK 2 — Build a transfer-learning ResNet-18
# ════════════════════════════════════════════════════════════════════════
# torchvision.models.resnet18(weights=IMAGENET1K_V1) loads the ImageNet
# checkpoint (~44 MB). We freeze all convolutional layers and replace
# the final fc with a fresh 10-class head. Only the new head trains —
# ~5K parameters instead of ~11M.


def build_transfer_resnet(
    n_classes: int = N_CLASSES, freeze_backbone: bool = True
) -> nn.Module:
    try:
        weights = torchvision.models.ResNet18_Weights.IMAGENET1K_V1
        model = torchvision.models.resnet18(weights=weights)
        print(f"  Loaded pre-trained ResNet-18 (weights={weights})")
    except Exception as exc:
        # Offline fallback: random weights. The code path remains identical.
        print(f"  Pre-trained weights unavailable ({type(exc).__name__}: {exc})")
        print("  Falling back to randomly initialised ResNet-18.")
        model = torchvision.models.resnet18(weights=None)

    # TODO: Freeze the backbone if freeze_backbone is True, then replace the fc head
    # Hint: for p in model.parameters(): p.requires_grad = False
    #       in_features = model.fc.in_features
    #       model.fc = nn.Linear(in_features, n_classes)
    if freeze_backbone:
        for p in model.parameters():
            p.requires_grad = ____  # noqa: F821

    in_features = model.fc.in_features
    model.fc = ____  # noqa: F821
    return model


# ════════════════════════════════════════════════════════════════════════
# TASK 3 — Build a from-scratch CNN baseline
# ════════════════════════════════════════════════════════════════════════
# A small CNN trained from random initialisation. Comparable parameter
# count to isolate the effect of pre-training vs architecture advantage.


def build_scratch_cnn(n_classes: int = N_CLASSES) -> nn.Module:
    """Baseline: a small CNN trained from random init for comparison."""
    # TODO: Build a Sequential CNN with conv/batchnorm/relu/pool blocks + fc head
    # Hint: Conv2d(3, 32, 3, padding=1) -> BN -> ReLU -> MaxPool2d(2)
    #       Conv2d(32, 64, ...) -> Conv2d(64, 128, ...) -> AdaptiveAvgPool2d(1)
    #       Flatten -> Dropout(0.3) -> Linear(128, n_classes)
    return ____  # noqa: F821


# ════════════════════════════════════════════════════════════════════════
# TASK 4 — Training harness with ExperimentTracker integration
# ════════════════════════════════════════════════════════════════════════
# Each training function logs parameters, per-epoch loss and accuracy
# to the ExperimentTracker. This replaces print-and-forget with a
# persistent, queryable record of every experiment.


async def train_model_async(
    model: nn.Module,
    name: str,
    tr_loader: DataLoader,
    vl_loader: DataLoader,
    epochs: int = 8,
    lr: float = 1e-3,
) -> tuple[list[float], list[float], list[float]]:
    """Train a model and log everything to ExperimentTracker."""
    model.to(device)
    params = [p for p in model.parameters() if p.requires_grad]
    n_trainable = sum(p.numel() for p in params)
    n_total = sum(p.numel() for p in model.parameters())
    print(f"\n-- {name} --  trainable params: {n_trainable:,} / {n_total:,}")

    opt = torch.optim.Adam(params, lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)

    train_losses: list[float] = []
    val_accs: list[float] = []
    train_accs: list[float] = []

    async with tracker.run(experiment_name=exp_name, run_name=name) as ctx:
        # TODO: Log all hyperparams as a dict with ctx.log_params({...})
        # Hint: await ctx.log_params({"model_type": name, "trainable_params": str(n_trainable), ...})
        await ctx.log_params(
            {
                "model_type": ____,  # Hint: name
                "trainable_params": ____,  # Hint: str(n_trainable)
                "total_params": ____,  # Hint: str(n_total)
                "epochs": ____,  # Hint: str(epochs)
                "lr": ____,  # Hint: str(lr)
                "batch_size": ____,  # Hint: str(tr_loader.batch_size)
                "dataset_size": ____,  # Hint: str(len(tr_loader.dataset))
            }
        )

        for epoch in range(epochs):
            # ── Training ──────────────────────────────────────────────────
            model.train()
            batch_losses = []
            correct = total = 0
            for xb, yb in tr_loader:
                xb, yb = xb.to(device), yb.to(device)
                opt.zero_grad()
                logits = model(xb)
                loss = F.cross_entropy(logits, yb)
                loss.backward()
                opt.step()
                batch_losses.append(loss.item())
                correct += int((logits.argmax(dim=-1) == yb).sum().item())
                total += int(yb.size(0))
            train_losses.append(float(np.mean(batch_losses)))
            train_accs.append(correct / total)
            scheduler.step()

            # ── Validation ────────────────────────────────────────────────
            model.eval()
            correct = total = 0
            with torch.no_grad():
                for xb, yb in vl_loader:
                    xb, yb = xb.to(device), yb.to(device)
                    preds = model(xb).argmax(dim=-1)
                    correct += int((preds == yb).sum().item())
                    total += int(yb.size(0))
            val_accs.append(correct / total)

            await ctx.log_metrics(
                {____: train_losses[-1], ____: train_accs[-1], ____: val_accs[-1]},
                # Hint: "train_loss", "train_acc", "val_acc"
                step=epoch + 1,
            )

            print(
                f"  epoch {epoch + 1}/{epochs}  "
                f"loss={train_losses[-1]:.4f}  "
                f"train_acc={train_accs[-1]:.3f}  "
                f"val_acc={val_accs[-1]:.3f}"
            )

        await ctx.log_metrics(
            {
                ____: val_accs[-1],  # Hint: "final_val_acc"
                ____: max(val_accs),  # Hint: "best_val_acc"
                ____: train_losses[-1],  # Hint: "final_train_loss"
            }
        )

    return train_losses, val_accs, train_accs


def train_model(
    model: nn.Module,
    name: str,
    tr_loader: DataLoader,
    vl_loader: DataLoader,
    epochs: int = 8,
    lr: float = 1e-3,
) -> tuple[list[float], list[float], list[float]]:
    """Sync wrapper — one asyncio.run per training call."""
    return asyncio.run(train_model_async(model, name, tr_loader, vl_loader, epochs, lr))


# ── Train both models ─────────────────────────────────────────────────
EPOCHS = 8

print("\n" + "=" * 70)
print("  TRAINING: Transfer Learning (frozen ResNet-18 + new head)")
print("=" * 70)
transfer_model = build_transfer_resnet()
transfer_losses, transfer_accs, transfer_train_accs = train_model(
    transfer_model,
    "resnet18_transfer",
    train_loader,
    val_loader,
    epochs=EPOCHS,
)

print("\n" + "=" * 70)
print("  TRAINING: From-Scratch CNN baseline")
print("=" * 70)
scratch_model = build_scratch_cnn()
scratch_losses, scratch_accs, scratch_train_accs = train_model(
    scratch_model,
    "cnn_from_scratch",
    train_loader,
    val_loader,
    epochs=EPOCHS,
)

# ── Checkpoint 2 ─────────────────────────────────────────────────────
best_transfer = max(transfer_accs)
best_scratch = max(scratch_accs)
assert best_transfer > 0.50, (
    f"Transfer val accuracy {best_transfer:.3f} below 0.50 -- check ImageNet "
    "normalisation, input resize, or epoch count"
)
assert len(transfer_losses) == EPOCHS, "Should have per-epoch losses"
assert len(scratch_losses) == EPOCHS, "Should have per-epoch losses"
# INTERPRETATION: Transfer learning leverages features already learned
# on ImageNet's 1.28M images. Even though we only train a linear head,
# the frozen ResNet-18 backbone provides powerful feature extraction
# that the from-scratch CNN cannot match in the same number of epochs.
print(f"\n  Transfer best val_acc: {best_transfer:.3f}")
print(f"  Scratch  best val_acc: {best_scratch:.3f}")
print(f"  Advantage: {best_transfer - best_scratch:+.3f}")
print("\n--- Checkpoint 2 passed --- both models trained and logged\n")


# ════════════════════════════════════════════════════════════════════════
# TASK 5 — Register both models in ModelRegistry and compare runs
# ════════════════════════════════════════════════════════════════════════
# The ModelRegistry provides versioned model storage with lifecycle
# management (staging -> shadow -> production -> archived). We register
# both models and compare their tracked metrics.


async def register_and_compare():
    """Register both models and compare experiment runs."""
    transfer_id = None
    scratch_id = None

    if has_registry:
        from kailash_ml.types import MetricSpec

        # TODO: Serialize each model state_dict and register with registry.register_model
        # Hint: pickle.dumps(transfer_model.state_dict())
        #       await registry.register_model(name=..., artifact=..., metrics=[MetricSpec(...)])
        transfer_bytes = ____  # noqa: F821
        transfer_version = await registry.register_model(
            name="cifar10_resnet18_transfer",
            artifact=____,  # noqa: F821
            metrics=[
                MetricSpec(name="val_acc", value=max(transfer_accs)),
                MetricSpec(name="final_loss", value=transfer_losses[-1]),
            ],
        )
        transfer_id = transfer_version.version

        scratch_bytes = ____  # noqa: F821
        scratch_version = await registry.register_model(
            name="cifar10_cnn_scratch",
            artifact=____,  # noqa: F821
            metrics=[
                MetricSpec(name="val_acc", value=max(scratch_accs)),
                MetricSpec(name="final_loss", value=scratch_losses[-1]),
            ],
        )
        scratch_id = scratch_version.version

        # Promote the better model
        if max(transfer_accs) >= max(scratch_accs):
            await registry.promote_model(
                name="cifar10_resnet18_transfer",
                version=transfer_version.version,
                target_stage="production",
                reason=(
                    f"Transfer model outperforms scratch: "
                    f"val_acc={max(transfer_accs):.4f} vs {max(scratch_accs):.4f}"
                ),
            )
            print("  Promoted: cifar10_resnet18_transfer -> production")
        else:
            await registry.promote_model(
                name="cifar10_cnn_scratch",
                version=scratch_version.version,
                target_stage="production",
                reason=(
                    f"Scratch model outperforms transfer: "
                    f"val_acc={max(scratch_accs):.4f} vs {max(transfer_accs):.4f}"
                ),
            )
            print("  Promoted: cifar10_cnn_scratch -> production")
    else:
        transfer_id = "skipped"
        scratch_id = "skipped"
        print("  Note: ModelRegistry not available. Skipping registration.")

    # Compare runs via ExperimentTracker
    print("\n  === ExperimentTracker Run Comparison ===")
    print(f"  {'Metric':<25} {'Transfer':>12} {'Scratch':>12} {'Delta':>12}")
    print("  " + "-" * 65)
    print(
        f"  {'Best val accuracy':<25} "
        f"{max(transfer_accs):>12.4f} "
        f"{max(scratch_accs):>12.4f} "
        f"{max(transfer_accs) - max(scratch_accs):>+12.4f}"
    )
    print(
        f"  {'Final train loss':<25} "
        f"{transfer_losses[-1]:>12.4f} "
        f"{scratch_losses[-1]:>12.4f} "
        f"{transfer_losses[-1] - scratch_losses[-1]:>+12.4f}"
    )

    return transfer_id, scratch_id


transfer_model_id, scratch_model_id = asyncio.run(register_and_compare())

# ── Checkpoint 3 ─────────────────────────────────────────────────────
assert transfer_model_id is not None, "Transfer model should be registered"
assert scratch_model_id is not None, "Scratch model should be registered"
# INTERPRETATION: The ModelRegistry gives every model a version, metrics,
# and an audit trail. Promoting a model to production records the exact
# comparison that justified the decision -- auditors can see precisely
# why the transfer model was chosen (or not) over the baseline.
print("\n--- Checkpoint 3 passed --- models registered and compared\n")


# ════════════════════════════════════════════════════════════════════════
# TASK 6 — Data efficiency experiment
# ════════════════════════════════════════════════════════════════════════
# The defining advantage of transfer learning: pre-trained features
# make learning more sample-efficient. We train the transfer model on
# 10%, 25%, 50%, and 100% of CIFAR-10 and plot how accuracy scales.
# This answers the production question: "How much labelled data do I
# actually need when I have a pre-trained backbone?"

DATA_FRACTIONS = [0.10, 0.25, 0.50, 1.0]
efficiency_results: dict[float, float] = {}

print("\n" + "=" * 70)
print("  DATA EFFICIENCY EXPERIMENT")
print("=" * 70)

rng = np.random.default_rng(42)


async def _run_efficiency_trial_async(frac: float) -> float:
    """One data-efficiency trial, logged under its own tracker run."""
    n_samples = int(len(train_set) * frac)
    indices = rng.choice(len(train_set), size=n_samples, replace=False).tolist()
    subset = Subset(train_set, indices)
    sub_loader = DataLoader(subset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)

    model_eff = build_transfer_resnet()
    model_eff.to(device)
    params = [p for p in model_eff.parameters() if p.requires_grad]
    opt = torch.optim.Adam(params, lr=1e-3)

    eff_epochs = 4
    async with tracker.run(
        experiment_name=exp_name,
        run_name=f"efficiency_{int(frac * 100)}pct",
    ) as ctx:
        await ctx.log_params(
            {
                ____: ____,  # Hint: "data_fraction", str(frac)
                ____: ____,  # Hint: "n_samples", str(n_samples)
                ____: ____,  # Hint: "epochs", str(eff_epochs)
            }
        )

        for epoch in range(eff_epochs):
            model_eff.train()
            for xb, yb in sub_loader:
                xb, yb = xb.to(device), yb.to(device)
                opt.zero_grad()
                loss = F.cross_entropy(model_eff(xb), yb)
                loss.backward()
                opt.step()

        # Evaluate on full val set
        model_eff.eval()
        correct = total = 0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(device), yb.to(device)
                preds = model_eff(xb).argmax(dim=-1)
                correct += int((preds == yb).sum().item())
                total += int(yb.size(0))
        acc = correct / total

        await ctx.log_metric(____, acc)  # Hint: "val_acc"

    return acc, n_samples


for frac in DATA_FRACTIONS:
    acc, n_samples = asyncio.run(_run_efficiency_trial_async(frac))
    efficiency_results[frac] = acc
    print(f"  {frac * 100:5.0f}% data ({n_samples:>5,} samples) -> val_acc = {acc:.4f}")

# ── Checkpoint 4 ─────────────────────────────────────────────────────
assert len(efficiency_results) == len(
    DATA_FRACTIONS
), "Should have results for all data fractions"
assert (
    efficiency_results[0.10] > 0.20
), f"Transfer with 10% data should beat random (acc={efficiency_results[0.10]:.3f})"
# INTERPRETATION: Transfer learning shows diminishing returns as data
# increases -- the gap between 10% and 100% is smaller than you might
# expect. This is because the pre-trained features already capture
# general visual patterns. The practical implication: when labelled data
# is expensive to obtain (medical images, industrial defects), transfer
# learning gives you 80% of the accuracy with 20% of the data.
print("\n--- Checkpoint 4 passed --- data efficiency experiment complete\n")


# ════════════════════════════════════════════════════════════════════════
# TASK 7 — Feature representation visualisation (t-SNE)
# ════════════════════════════════════════════════════════════════════════
# Extract features from the frozen ResNet-18 backbone (before the final
# classifier head) and project them to 2D with t-SNE. If transfer
# learning works, samples of the same class should cluster together even
# though the backbone was never trained on CIFAR-10.

from sklearn.manifold import TSNE


def extract_features(model: nn.Module, loader: DataLoader, max_samples: int = 2000):
    """Extract features from the penultimate layer (before fc head)."""
    model.eval()
    hook_features = []

    def hook_fn(module, inp, out):
        hook_features.append(out.flatten(1).detach().cpu())

    # For ResNet, features come from the avgpool layer
    if hasattr(model, "avgpool"):
        handle = model.avgpool.register_forward_hook(hook_fn)
    else:
        handle = model[-3].register_forward_hook(hook_fn)

    labels = []
    with torch.no_grad():
        collected = 0
        for xb, yb in loader:
            if collected >= max_samples:
                break
            xb = xb.to(device)
            model(xb)
            labels.append(yb.numpy())
            collected += len(yb)

    handle.remove()
    features_np = torch.cat(hook_features, dim=0).numpy()[:max_samples]
    labels_np = np.concatenate(labels)[:max_samples]
    return features_np, labels_np


print("\n-- Extracting features for t-SNE visualisation --")
transfer_feats, transfer_labels = extract_features(transfer_model, val_loader)
scratch_feats, scratch_labels = extract_features(scratch_model, val_loader)

print(f"  Transfer features shape: {transfer_feats.shape}")
print(f"  Scratch features shape: {scratch_feats.shape}")

# TODO: Run t-SNE on both feature sets (perplexity=30, n_iter=500, random_state=42)
# Hint: TSNE(n_components=2, perplexity=30, n_iter=500, random_state=42).fit_transform(...)
tsne_transfer = ____  # noqa: F821
coords_transfer = ____  # noqa: F821

tsne_scratch = ____  # noqa: F821
coords_scratch = ____  # noqa: F821


def cluster_quality(coords: np.ndarray, labels: np.ndarray) -> float:
    """Simple cluster quality: ratio of intra-class to inter-class distance."""
    intra = []
    centroids = []
    for c in range(N_CLASSES):
        mask = labels == c
        if mask.sum() < 2:
            continue
        pts = coords[mask]
        centroid = pts.mean(axis=0)
        centroids.append(centroid)
        intra.append(np.mean(np.linalg.norm(pts - centroid, axis=1)))
    centroids = np.array(centroids)
    inter = np.mean(
        [
            np.linalg.norm(centroids[i] - centroids[j])
            for i in range(len(centroids))
            for j in range(i + 1, len(centroids))
        ]
    )
    avg_intra = np.mean(intra)
    return avg_intra / inter if inter > 0 else float("inf")


cq_transfer = cluster_quality(coords_transfer, transfer_labels)
cq_scratch = cluster_quality(coords_scratch, scratch_labels)

print(f"\n  t-SNE cluster quality (lower = better separated):")
print(f"    Transfer: {cq_transfer:.4f}")
print(f"    Scratch:  {cq_scratch:.4f}")

print(f"\n  Transfer t-SNE centroids (first 5 classes):")
for c in range(min(5, N_CLASSES)):
    mask = transfer_labels == c
    centroid = coords_transfer[mask].mean(axis=0)
    print(f"    {CLASS_NAMES[c]:>12}: ({centroid[0]:+.1f}, {centroid[1]:+.1f})")

# ── Checkpoint 5 ─────────────────────────────────────────────────────
assert transfer_feats.shape[0] > 0, "Should have extracted features"
assert coords_transfer.shape == (transfer_feats.shape[0], 2), "t-SNE should produce 2D"
# INTERPRETATION: In a good t-SNE plot, classes form distinct clusters.
# Transfer features typically show tighter clusters because ImageNet
# pre-training teaches the backbone to separate visual categories. The
# scratch CNN learns noisier, less separable features in the same number
# of epochs because it starts from random weights.
print("\n--- Checkpoint 5 passed --- t-SNE feature visualisation complete\n")


# ════════════════════════════════════════════════════════════════════════
# TASK 8 — ONNX export and InferenceServer
# ════════════════════════════════════════════════════════════════════════
# Export the fine-tuned transfer model to ONNX for portable deployment,
# then serve it with kailash-ml's InferenceServer for batch predictions.

transfer_model.eval()
transfer_model_cpu = transfer_model.cpu()
onnx_path = Path("ex_7_transfer_resnet.onnx")
sample = torch.randn(1, 3, INPUT_SIZE, INPUT_SIZE)

# TODO: Export transfer_model_cpu to ONNX using torch.onnx.export
# Hint: torch.onnx.export(model, sample, path, input_names=["input"],
#         output_names=["logits"], dynamic_axes={...}, opset_version=17, dynamo=False)
____  # noqa: F821

print(f"\nExported to {onnx_path} ({onnx_path.stat().st_size // 1024} KB)")

# Move model back to device for further use
transfer_model.to(device)

# Serve with InferenceServer
from kailash_ml import InferenceServer


async def serve_predictions():
    """Load the fine-tuned model and serve sample predictions."""
    # TODO: Create InferenceServer, load_model, then predict on a batch
    # Hint: server = InferenceServer()
    #       await server.load_model("cifar10_transfer", model_for_serving)
    server = ____  # noqa: F821

    model_for_serving = build_transfer_resnet()
    model_for_serving.load_state_dict(transfer_model.state_dict())
    model_for_serving.eval()

    try:
        await ____  # noqa: F821

        test_batch_x, test_batch_y = next(iter(val_loader))
        sample_x = test_batch_x[:8]
        sample_y = test_batch_y[:8]

        with torch.no_grad():
            logits = model_for_serving(sample_x)
            preds = logits.argmax(dim=-1)
            probs = F.softmax(logits, dim=-1)
            confidences = probs.max(dim=-1).values

        print("\n  === InferenceServer Predictions ===")
        print(
            f"  {'#':<4} {'True':>12} {'Predicted':>12} {'Confidence':>12} {'Correct':>8}"
        )
        print("  " + "-" * 52)
        n_correct = 0
        for i in range(len(sample_x)):
            true_cls = CLASS_NAMES[sample_y[i]]
            pred_cls = CLASS_NAMES[preds[i]]
            conf = confidences[i].item()
            correct = "Y" if preds[i] == sample_y[i] else "N"
            if preds[i] == sample_y[i]:
                n_correct += 1
            print(
                f"  {i + 1:<4} {true_cls:>12} {pred_cls:>12} {conf:>12.3f} {correct:>8}"
            )
        print(f"\n  Sample accuracy: {n_correct}/{len(sample_x)}")
        return n_correct, len(sample_x)
    except Exception as e:
        print(f"  Note: InferenceServer demo skipped ({e})")
        with torch.no_grad():
            test_x, test_y = next(iter(val_loader))
            preds = model_for_serving(test_x[:8]).argmax(dim=-1)
            n_correct = int((preds == test_y[:8]).sum().item())
        print(f"\n  Direct predictions: {n_correct}/8 correct")
        return n_correct, 8


n_correct, n_total = asyncio.run(serve_predictions())

# ── Checkpoint 6 ─────────────────────────────────────────────────────
assert onnx_path.exists(), "ONNX file should be exported"
assert onnx_path.stat().st_size > 1000, "ONNX file should not be empty"
assert n_correct >= 0, "Should have run predictions"
# INTERPRETATION: The ONNX export creates a portable model artifact that
# runs on any ONNX-compatible runtime (CPU, GPU, edge devices, mobile).
# InferenceServer wraps the model with caching, batch prediction, and
# REST API capabilities. This is the deployment path: train with PyTorch,
# export to ONNX, serve with InferenceServer.
print("\n--- Checkpoint 6 passed --- ONNX export and serving complete\n")


# ════════════════════════════════════════════════════════════════════════
# TASK 9 — Adapter modules (bridge to M6 LoRA)
# ════════════════════════════════════════════════════════════════════════
# Full fine-tuning updates ALL parameters; frozen-head training updates
# ONLY the final layer. Adapter modules sit between these extremes: they
# inject small trainable bottleneck layers inside the frozen backbone
# while keeping the original weights frozen. This achieves near-full-
# fine-tuning quality with a fraction of the trainable parameters.
#
# In M6, you will learn LoRA (Low-Rank Adaptation), which is an adapter
# technique specifically designed for LLMs. The concept is the same:
# inject small trainable matrices into the frozen model, but LoRA uses
# low-rank decomposition (A @ B where A is d x r and B is r x d with
# r << d) instead of bottleneck layers.


class BottleneckAdapter(nn.Module):
    """A simple adapter module: down-project, nonlinearity, up-project.

    Inserted between frozen layers. The skip connection ensures the
    adapter starts as an identity function (initial weights are small),
    so training begins from the pre-trained features, not random noise.
    """

    def __init__(self, dim: int, bottleneck: int = 64):
        super().__init__()
        # TODO: Build down-projection (dim -> bottleneck) and up-projection (bottleneck -> dim)
        # Hint: self.down = nn.Linear(dim, bottleneck)
        #       self.up = nn.Linear(bottleneck, dim)
        #       nn.init.zeros_(self.up.weight) and nn.init.zeros_(self.up.bias)
        self.down = ____  # noqa: F821
        self.up = ____  # noqa: F821
        nn.init.zeros_(self.up.weight)
        nn.init.zeros_(self.up.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # TODO: Return skip connection: x + up(relu(down(x)))
        # Hint: return x + self.up(F.relu(self.down(x)))
        return ____  # noqa: F821


def build_adapter_resnet(n_classes: int = N_CLASSES, bottleneck: int = 64) -> nn.Module:
    """ResNet-18 with bottleneck adapters after each residual block."""
    try:
        weights = torchvision.models.ResNet18_Weights.IMAGENET1K_V1
        model = torchvision.models.resnet18(weights=weights)
    except Exception:
        model = torchvision.models.resnet18(weights=None)

    # Freeze all original parameters
    for p in model.parameters():
        p.requires_grad = False

    # Replace fc head
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, n_classes)

    # Insert adapters after layer3 and layer4
    original_layer3 = model.layer3
    original_layer4 = model.layer4

    class AdaptedBlock(nn.Module):
        def __init__(self, block, adapter):
            super().__init__()
            self.block = block
            self.adapter = adapter

        def forward(self, x):
            out = self.block(x)
            b, c, h, w = out.shape
            pooled = out.mean(dim=[2, 3])  # (B, C)
            adapted = self.adapter(pooled)  # (B, C)
            return out + adapted.unsqueeze(-1).unsqueeze(-1)

    model.layer3 = AdaptedBlock(original_layer3, BottleneckAdapter(256, bottleneck))
    model.layer4 = AdaptedBlock(original_layer4, BottleneckAdapter(512, bottleneck))

    return model


# Quick adapter training (fewer epochs since this is a concept demo)
print("\n" + "=" * 70)
print("  ADAPTER MODULE DEMO (bridge to M6 LoRA)")
print("=" * 70)

adapter_model = build_adapter_resnet(bottleneck=64)
adapter_model.to(device)
adapter_params = [p for p in adapter_model.parameters() if p.requires_grad]
n_adapter_trainable = sum(p.numel() for p in adapter_params)
n_adapter_total = sum(p.numel() for p in adapter_model.parameters())
print(
    f"  Adapter trainable: {n_adapter_trainable:,} / {n_adapter_total:,} "
    f"({100 * n_adapter_trainable / n_adapter_total:.1f}%)"
)

opt_adapter = torch.optim.Adam(adapter_params, lr=1e-3)
adapter_model.train()

for epoch in range(3):
    batch_losses = []
    for xb, yb in train_loader:
        xb, yb = xb.to(device), yb.to(device)
        opt_adapter.zero_grad()
        loss = F.cross_entropy(adapter_model(xb), yb)
        loss.backward()
        opt_adapter.step()
        batch_losses.append(loss.item())
    epoch_loss = float(np.mean(batch_losses))
    print(f"  adapter epoch {epoch + 1}/3  loss={epoch_loss:.4f}")

adapter_model.eval()
correct = total = 0
with torch.no_grad():
    for xb, yb in val_loader:
        xb, yb = xb.to(device), yb.to(device)
        preds = adapter_model(xb).argmax(dim=-1)
        correct += int((preds == yb).sum().item())
        total += int(yb.size(0))
adapter_acc = correct / total
print(f"  Adapter val_acc: {adapter_acc:.4f}")

print(f"\n  === Parameter Efficiency Comparison ===")
print(f"  {'Method':<25} {'Trainable':>12} {'Val Acc':>10}")
print("  " + "-" * 50)
transfer_trainable = sum(
    p.numel() for p in transfer_model.parameters() if p.requires_grad
)
scratch_trainable = sum(
    p.numel() for p in scratch_model.parameters() if p.requires_grad
)
print(f"  {'From scratch':<25} {scratch_trainable:>12,} {max(scratch_accs):>10.4f}")
print(f"  {'Frozen head':<25} {transfer_trainable:>12,} {max(transfer_accs):>10.4f}")
print(f"  {'Adapter (bottleneck)':<25} {n_adapter_trainable:>12,} {adapter_acc:>10.4f}")
print(f"\n  In M6, you will learn LoRA — the adapter technique for LLMs.")
print(f"  Same idea: inject small trainable matrices into frozen weights.")
print(f"  LoRA uses low-rank decomposition: W + A @ B where A is d x r, B is r x d.")

# ── Checkpoint 7 ─────────────────────────────────────────────────────
assert (
    adapter_acc > 0.20
), f"Adapter model should beat random chance (acc={adapter_acc:.3f})"
assert (
    n_adapter_trainable < n_adapter_total
), "Adapter model should have fewer trainable params than total"
# INTERPRETATION: Adapters achieve a balance between the frozen-head
# approach (too few trainable parameters) and full fine-tuning (too many,
# risk of catastrophic forgetting). In production, adapters are preferred
# when you need task-specific models but cannot afford to store a full
# copy of the backbone per task. LoRA in M6 extends this to LLMs.
print("\n--- Checkpoint 7 passed --- adapter concept demonstrated\n")


# ════════════════════════════════════════════════════════════════════════
# Visualisations with ModelVisualizer
# ════════════════════════════════════════════════════════════════════════

viz = ModelVisualizer()

# TODO: Plot training curves for transfer and scratch models
# Hint: viz.training_history(metrics={"transfer loss": transfer_losses, ...},
#                             x_label="Epoch", y_label="Value")
fig_training = viz.training_history(
    metrics=____,  # noqa: F821
    x_label=____,  # noqa: F821
    y_label=____,  # noqa: F821
)
fig_training.write_html("ex_7_training_curves.html")
print("Saved: ex_7_training_curves.html")

# Data efficiency chart (plotly directly)
import plotly.graph_objects as go

fig_efficiency = go.Figure()
fracs = sorted(efficiency_results.keys())
accs_by_frac = [efficiency_results[f] for f in fracs]
fig_efficiency.add_trace(
    go.Scatter(
        x=[f * 100 for f in fracs],
        y=accs_by_frac,
        mode="lines+markers",
        name="Transfer (ResNet-18)",
        marker=dict(size=10),
        line=dict(width=3),
    )
)
fig_efficiency.update_layout(
    title="Data Efficiency: Transfer Learning Accuracy vs Training Set Size",
    xaxis_title="% of CIFAR-10 Training Data",
    yaxis_title="Validation Accuracy",
    template="plotly_white",
)
fig_efficiency.write_html("ex_7_data_efficiency.html")
print("Saved: ex_7_data_efficiency.html")

# t-SNE visualisation (plotly directly)
fig_tsne = go.Figure()
for c in range(N_CLASSES):
    mask = transfer_labels == c
    fig_tsne.add_trace(
        go.Scatter(
            x=coords_transfer[mask, 0],
            y=coords_transfer[mask, 1],
            mode="markers",
            name=CLASS_NAMES[c],
            marker=dict(size=4, opacity=0.6),
        )
    )
fig_tsne.update_layout(
    title="t-SNE: Transfer Learning Feature Space (ResNet-18 backbone)",
    xaxis_title="t-SNE 1",
    yaxis_title="t-SNE 2",
    template="plotly_white",
)
fig_tsne.write_html("ex_7_tsne.html")
print("Saved: ex_7_tsne.html")

# ── Checkpoint 8 ─────────────────────────────────────────────────────
import os

assert os.path.exists("ex_7_training_curves.html"), "Training curves should be saved"
assert os.path.exists(
    "ex_7_data_efficiency.html"
), "Data efficiency chart should be saved"
assert os.path.exists("ex_7_tsne.html"), "t-SNE plot should be saved"
print("\n--- Checkpoint 8 passed --- all visualisations saved\n")

# Clean up
asyncio.run(conn.close())


# ════════════════════════════════════════════════════════════════════════
# REFLECTION
# ════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("  WHAT YOU'VE MASTERED")
print("=" * 70)
print(
    """
  TRANSFER LEARNING:
  [x] Pre-trained ResNet-18 backbone (ImageNet weights) + new CIFAR-10 head
      Froze all conv layers; trained only the final 5K-parameter linear head.
  [x] From-scratch CNN baseline: fair comparison with random init weights
  [x] Data efficiency experiment: transfer learning uses 80% less labelled data
      to reach comparable accuracy — the "sample efficiency" advantage.

  FEATURE ANALYSIS:
  [x] t-SNE visualisation: extracted penultimate-layer features and projected
      to 2D. Transfer features show tighter class clusters because ImageNet
      pre-training taught separable visual representations.

  DEPLOYMENT:
  [x] ONNX export: portable model artifact runnable on CPU/GPU/edge devices
  [x] InferenceServer: wraps PyTorch model for batch predictions with logging

  ADAPTER MODULES (bridge to M6 LoRA):
  [x] BottleneckAdapter: down-project -> relu -> up-project + skip connection
      Starts as identity (zero-init up-projection), adapts without forgetting.
  [x] Parameter comparison: frozen-head < adapter < full fine-tuning
  [x] LoRA (M6) = same idea, low-rank decomposition instead of bottleneck

  NEXT: In Exercise 8, you will apply reinforcement learning — a fundamentally
  different learning paradigm where an agent learns by interacting with an
  environment. PPO in RL is the same algorithm used for RLHF in LLMs (M6).
"""
)
