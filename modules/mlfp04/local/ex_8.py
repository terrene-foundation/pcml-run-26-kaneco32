# Copyright 2026 Terrene Foundation
# SPDX-License-Identifier: Apache-2.0
"""
# ════════════════════════════════════════════════════════════════════════
# MLFP04 — Exercise 8: Deep Learning Foundations
# ════════════════════════════════════════════════════════════════════════
#
# WHAT YOU'LL LEARN:
#   After completing this exercise, you will be able to:
#   - Build a neural network from scratch (forward pass, loss, backprop)
#   - Explain why linear models cannot learn XOR (and hidden layers can)
#   - Implement and compare activation functions (ReLU, GELU, Sigmoid, Tanh)
#   - Apply dropout, batch normalisation, and weight initialisation
#   - Compare optimisers (SGD, Adam, AdamW) on training dynamics
#   - Implement learning rate scheduling (cosine annealing, warmup)
#   - Apply gradient clipping to prevent exploding gradients
#   - Implement early stopping to prevent overfitting
#   - Build a CNN with residual connections (ResBlock)
#   - Export trained models to ONNX with OnnxBridge
#   - Explain representation learning: hidden layers = automated USML
#
# PREREQUISITES:
#   - MLFP04 Exercise 7 (THE PIVOT: matrix factorisation -> neural embeddings)
#   - MLFP03 Exercise 4 (gradient boosting — optimisation-based learning)
#
# ESTIMATED TIME: ~180-210 min (densest lesson in the curriculum)
#
# TASKS:
#   1.  XOR proof: linear model fails, hidden layer succeeds
#   2.  Activation functions: ReLU, Leaky ReLU, GELU, Sigmoid, Tanh
#   3.  Weight initialisation: Xavier vs Kaiming vs zero
#   4.  Build CNN with residual connections (ResBlock)
#   5.  Optimiser comparison: SGD, Adam, AdamW
#   6.  Learning rate scheduling: cosine annealing, warmup
#   7.  Dropout and batch normalisation
#   8.  Gradient clipping and training dynamics monitoring
#   9.  Early stopping implementation
#   10. ONNX export with OnnxBridge
#   11. Loss function taxonomy and the USML bridge
#
# DATASET: Synthetic image data (5000 x 64x64, 5 classes)
#
# ════════════════════════════════════════════════════════════════════════
"""
from __future__ import annotations

import numpy as np
import polars as pl

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import DataLoader, TensorDataset

    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False
    print("  PyTorch not installed. Install with: pip install torch")

from kailash_ml import ModelVisualizer
from kailash_ml.bridge.onnx_bridge import OnnxBridge

from shared import MLFPDataLoader
from shared.kailash_helpers import get_device, setup_environment

setup_environment()

if HAS_TORCH:
    device = get_device()
    print(f"Device: {device}")
else:
    device = None


# ══════════════════════════════════════════════════════════════════════
# TASK 1: XOR proof — linear model fails, hidden layer succeeds
# ══════════════════════════════════════════════════════════════════════
# Forward pass:  z = Wx + b,  y_hat = sigma(z)
# Loss:          L = -y log(y_hat) - (1-y) log(1-y_hat)   (BCE)
# Backward:      dL/dW = (y_hat - y) x',  dL/db = (y_hat - y)
# Update:        W <- W - lr * dL/dW

rng_simple = np.random.default_rng(42)
n_simple = 200
n_feats_simple = 4

X_simple = rng_simple.standard_normal((n_simple, n_feats_simple)).astype(np.float32)
y_simple = ((X_simple[:, 0] > 0) ^ (X_simple[:, 1] > 0)).astype(np.float32)

print("=== XOR Proof: Linear vs Non-Linear ===")
print(f"Task: XOR classification ({n_simple} samples, {n_feats_simple} features)")
print(f"Class balance: {y_simple.mean():.2f}")

if HAS_TORCH:
    # Linear model (no hidden layer)
    # TODO: Create a linear model with n_feats_simple inputs and 1 output
    simple_net = ____  # Hint: nn.Linear(n_feats_simple, 1)
    # TODO: Create an SGD optimiser with lr=0.1
    simple_opt = ____  # Hint: torch.optim.SGD(simple_net.parameters(), lr=0.1)
    simple_crit = nn.BCEWithLogitsLoss()

    X_t = torch.from_numpy(X_simple)
    y_t = torch.from_numpy(y_simple).unsqueeze(1)

    simple_losses = []
    for epoch in range(50):
        simple_opt.zero_grad()
        loss = simple_crit(simple_net(X_t), y_t)
        loss.backward()
        simple_opt.step()
        simple_losses.append(loss.item())

    with torch.no_grad():
        preds = torch.sigmoid(simple_net(X_t)).numpy().flatten()
        acc_linear = ((preds > 0.5) == y_simple).mean()

    print(
        f"Linear (no hidden layer): loss={simple_losses[-1]:.4f}, accuracy={acc_linear:.4f}"
    )
    print(f"  Linear models have a single hyperplane boundary — cannot separate XOR.")

    # Hidden layer model (CAN learn XOR)
    # TODO: Build a Sequential model: Linear(n_feats_simple, 32) -> ReLU ->
    #       Linear(32, 16) -> ReLU -> Linear(16, 1)
    hidden_net = ____  # Hint: nn.Sequential(nn.Linear(...), nn.ReLU(), ...)
    # TODO: Create an Adam optimiser with lr=0.01
    hidden_opt = ____  # Hint: torch.optim.Adam(hidden_net.parameters(), lr=0.01)

    hidden_losses = []
    for epoch in range(100):
        hidden_opt.zero_grad()
        loss = simple_crit(hidden_net(X_t), y_t)
        loss.backward()
        hidden_opt.step()
        hidden_losses.append(loss.item())

    with torch.no_grad():
        preds_h = torch.sigmoid(hidden_net(X_t)).numpy().flatten()
        acc_hidden = ((preds_h > 0.5) == y_simple).mean()

    print(
        f"Hidden layers (32+16 units): loss={hidden_losses[-1]:.4f}, accuracy={acc_hidden:.4f}"
    )
    print(f"\nKey insight:")
    print(f"  Linear: z = Wx + b  (hyperplane boundary)")
    print(f"  ReLU:   max(0, z)   (piecewise-linear, universal approximation)")
    print(f"  Depth:  stacking layers = composing functions = exponential expressivity")

    # ── Checkpoint 1 ─────────────────────────────────────────────────
    assert (
        acc_hidden > acc_linear
    ), f"Hidden layer (acc={acc_hidden:.4f}) should outperform linear (acc={acc_linear:.4f})"
    assert hidden_losses[-1] < hidden_losses[0], "Hidden network loss should decrease"
    # INTERPRETATION: XOR is the canonical proof that linear models are insufficient.
    # One hidden layer with nonlinear activations creates piecewise-linear boundaries.
    print("\n✓ Checkpoint 1 passed — XOR proof: hidden layers beat linear\n")


# ══════════════════════════════════════════════════════════════════════
# TASK 2: Activation functions comparison
# ══════════════════════════════════════════════════════════════════════

if HAS_TORCH:
    print("=== Activation Functions ===")

    x_range = torch.linspace(-3, 3, 100)

    # TODO: Fill in the activations dict — map each name to the nn module
    activations = {
        "ReLU": ____,  # Hint: nn.ReLU()
        "LeakyReLU": ____,  # Hint: nn.LeakyReLU(0.01)
        "GELU": ____,  # Hint: nn.GELU()
        "Sigmoid": ____,  # Hint: nn.Sigmoid()
        "Tanh": ____,  # Hint: nn.Tanh()
        "ELU": ____,  # Hint: nn.ELU(alpha=1.0)
        "Swish/SiLU": ____,  # Hint: nn.SiLU()
    }

    print(f"{'Activation':<15} {'f(-2)':>8} {'f(0)':>8} {'f(2)':>8} {'Range':>20}")
    print("─" * 55)
    for name, act in activations.items():
        vals = act(torch.tensor([-2.0, 0.0, 2.0]))
        print(
            f"{name:<15} {vals[0].item():>8.4f} {vals[1].item():>8.4f} {vals[2].item():>8.4f}",
            end="",
        )
        if name == "Sigmoid":
            print(f" {'(0, 1)':>20}")
        elif name == "Tanh":
            print(f" {'(-1, 1)':>20}")
        elif name in ("ReLU", "LeakyReLU", "ELU"):
            print(f" {'[0/neg, inf)':>20}")
        else:
            print(f" {'(-inf, inf)':>20}")

    print(
        """
Activation selection guide:
  ReLU:        Default for hidden layers. Fast, sparse activations.
               Problem: "dying ReLU" (neurons stuck at 0 if z < 0 always).
  LeakyReLU:   Fixes dying ReLU by allowing small negative slope.
  GELU:        Used in transformers (BERT, GPT). Smooth, probabilistic gate.
  ELU:         Smooth negative side, self-normalising properties.
  Swish/SiLU:  x * sigmoid(x). Used in EfficientNet, smooth like GELU.
  Sigmoid:     Output layer for binary classification (maps to [0, 1]).
  Tanh:        Output layer when you need [-1, 1]. Hidden layer in LSTMs.

  Rule of thumb: ReLU for hidden layers, Sigmoid/Softmax for output.
  Modern architectures (transformers): GELU everywhere.
"""
    )

    # Train with different activations
    act_results = {}
    for act_name, act_fn in [
        ("ReLU", nn.ReLU()),
        ("GELU", nn.GELU()),
        ("Tanh", nn.Tanh()),
        ("Swish", nn.SiLU()),
    ]:
        net = nn.Sequential(
            nn.Linear(n_feats_simple, 32),
            act_fn,
            nn.Linear(32, 16),
            act_fn,
            nn.Linear(16, 1),
        )
        opt = torch.optim.Adam(net.parameters(), lr=0.01)
        losses = []
        for epoch in range(80):
            opt.zero_grad()
            loss = simple_crit(net(X_t), y_t)
            loss.backward()
            opt.step()
            losses.append(loss.item())

        with torch.no_grad():
            acc = ((torch.sigmoid(net(X_t)).numpy().flatten() > 0.5) == y_simple).mean()
        act_results[act_name] = {"final_loss": losses[-1], "accuracy": acc}
        print(f"  {act_name:<10}: loss={losses[-1]:.4f}, accuracy={acc:.4f}")

    # ── Checkpoint 2 ─────────────────────────────────────────────────
    assert len(act_results) == 4, "Should test 4 activation functions"
    assert all(
        r["accuracy"] > 0.5 for r in act_results.values()
    ), "All activations should beat random on XOR"
    print("\n✓ Checkpoint 2 passed — activation function comparison\n")


# ══════════════════════════════════════════════════════════════════════
# TASK 3: Weight initialisation
# ══════════════════════════════════════════════════════════════════════

if HAS_TORCH:
    print("=== Weight Initialisation ===")

    def init_and_train(init_name: str, init_fn) -> dict:
        """Train a network with a specific weight initialisation."""
        net = nn.Sequential(
            nn.Linear(n_feats_simple, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
        )
        for m in net.modules():
            if isinstance(m, nn.Linear):
                init_fn(m.weight)
                nn.init.zeros_(m.bias)

        opt = torch.optim.Adam(net.parameters(), lr=0.01)
        losses = []
        for epoch in range(80):
            opt.zero_grad()
            loss = simple_crit(net(X_t), y_t)
            loss.backward()
            opt.step()
            losses.append(loss.item())
        return {"init_loss": losses[0], "final_loss": losses[-1], "losses": losses}

    init_results = {}
    for name, fn in [
        # TODO: Fill in the four init strategies as (name, lambda) pairs
        ("Xavier/Glorot", ____),  # Hint: lambda w: nn.init.xavier_uniform_(w)
        (
            "Kaiming/He",
            ____,
        ),  # Hint: lambda w: nn.init.kaiming_uniform_(w, nonlinearity="relu")
        ("Normal(0,1)", ____),  # Hint: lambda w: nn.init.normal_(w, 0, 1)
        ("Zeros", ____),  # Hint: lambda w: nn.init.zeros_(w)
    ]:
        result = init_and_train(name, fn)
        init_results[name] = result
        print(
            f"  {name:<15}: init_loss={result['init_loss']:.4f}, final_loss={result['final_loss']:.4f}"
        )

    print(
        """
Weight initialisation guide:
  Xavier/Glorot: Var(w) = 2/(fan_in + fan_out). For Sigmoid/Tanh.
  Kaiming/He:    Var(w) = 2/fan_in. For ReLU (accounts for half-dead neurons).
  Normal(0,1):   Too large — activations explode in deep networks.
  Zeros:         Symmetry problem — all neurons learn the same thing.

  PyTorch default: Kaiming uniform (for Linear layers).
  Rule: Use Kaiming for ReLU, Xavier for Sigmoid/Tanh, default otherwise.
"""
    )

    # ── Checkpoint 3 ─────────────────────────────────────────────────
    assert (
        init_results["Kaiming/He"]["final_loss"]
        < init_results["Zeros"]["final_loss"] + 0.5
    ), "Kaiming init should converge better than zero init"
    print("✓ Checkpoint 3 passed — weight initialisation comparison\n")


# ══════════════════════════════════════════════════════════════════════
# TASK 4: Build CNN with residual connections
# ══════════════════════════════════════════════════════════════════════

if HAS_TORCH:
    loader = MLFPDataLoader()

    n_samples_img = 5000
    n_channels = 1
    img_size = 64
    n_classes = 5

    rng_data = np.random.default_rng(42)
    X_images = rng_data.standard_normal(
        (n_samples_img, n_channels, img_size, img_size)
    ).astype(np.float32)
    y_labels = (rng_data.random((n_samples_img, n_classes)) > 0.85).astype(np.float32)

    print("=== Medical Image Data ===")
    print(f"Images: {X_images.shape} (N, C, H, W)")
    print(f"Labels: {y_labels.shape} (N, classes)")
    print(f"Positive rates per class: {y_labels.mean(axis=0).round(3)}")

    split = int(0.8 * n_samples_img)
    X_train, X_test = X_images[:split], X_images[split:]
    y_train, y_test = y_labels[:split], y_labels[split:]

    train_ds = TensorDataset(torch.from_numpy(X_train), torch.from_numpy(y_train))
    test_ds = TensorDataset(torch.from_numpy(X_test), torch.from_numpy(y_test))
    train_loader = DataLoader(train_ds, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=64)

    class ResBlock(nn.Module):
        """Residual block: skip connection preserves gradient flow."""

        def __init__(self, channels: int):
            super().__init__()
            # TODO: Define conv1: Conv2d(channels, channels, 3, padding=1)
            self.conv1 = ____  # Hint: nn.Conv2d(channels, channels, 3, padding=1)
            # TODO: Define bn1: BatchNorm2d(channels)
            self.bn1 = ____  # Hint: nn.BatchNorm2d(channels)
            # TODO: Define conv2: same shape as conv1
            self.conv2 = ____  # Hint: nn.Conv2d(channels, channels, 3, padding=1)
            # TODO: Define bn2: BatchNorm2d(channels)
            self.bn2 = ____  # Hint: nn.BatchNorm2d(channels)

        def forward(self, x):
            residual = x
            # TODO: Apply conv1 -> bn1 -> relu
            out = ____  # Hint: torch.relu(self.bn1(self.conv1(x)))
            # TODO: Apply conv2 -> bn2 (no activation yet — add residual first)
            out = ____  # Hint: self.bn2(self.conv2(out))
            # TODO: Add residual connection and apply relu
            return ____  # Hint: torch.relu(out + residual)

    class MedicalCNN(nn.Module):
        """CNN for multi-label medical image classification."""

        def __init__(self, n_classes: int = 5, dropout_rate: float = 0.3):
            super().__init__()
            self.features = nn.Sequential(
                nn.Conv2d(1, 32, 3, padding=1),
                nn.BatchNorm2d(32),
                nn.ReLU(),
                nn.MaxPool2d(2),
                ResBlock(32),
                nn.Conv2d(32, 64, 3, padding=1),
                nn.BatchNorm2d(64),
                nn.ReLU(),
                nn.MaxPool2d(2),
                ResBlock(64),
                nn.AdaptiveAvgPool2d(4),
            )
            self.classifier = nn.Sequential(
                nn.Flatten(),
                nn.Linear(64 * 4 * 4, 128),
                nn.ReLU(),
                nn.Dropout(dropout_rate),
                nn.Linear(128, n_classes),
            )

        def forward(self, x):
            return self.classifier(self.features(x))

    model = MedicalCNN(n_classes=n_classes, dropout_rate=0.3).to(device)
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\n=== Model Architecture ===")
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable: {trainable_params:,}")

    # Verify ResBlock
    with torch.no_grad():
        dummy = torch.zeros(1, 32, 16, 16)
        res_block = ResBlock(32)
        out = res_block(dummy)
        assert out.shape == dummy.shape, "ResBlock should preserve dimensions"

    # ── Checkpoint 4 ─────────────────────────────────────────────────
    assert (
        total_params > 1000
    ), f"Model should have substantial params, got {total_params}"
    # INTERPRETATION: The residual connection (out + residual) provides a
    # 'gradient highway' that bypasses the block, preventing vanishing gradients.
    print("\n✓ Checkpoint 4 passed — MedicalCNN with ResBlocks\n")


# ══════════════════════════════════════════════════════════════════════
# TASK 5: Optimiser comparison
# ══════════════════════════════════════════════════════════════════════

if HAS_TORCH:
    print("=== Optimiser Comparison ===")

    def train_for_epochs(model_fn, optimizer_fn, n_epochs: int = 10) -> list[float]:
        """Quick training to compare optimisers."""
        net = model_fn().to(device)
        opt = optimizer_fn(net.parameters())
        crit = nn.BCEWithLogitsLoss()
        losses = []
        for epoch in range(n_epochs):
            net.train()
            epoch_loss = 0.0
            n_batches = 0
            for X_batch, y_batch in train_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                opt.zero_grad()
                loss = crit(net(X_batch), y_batch)
                loss.backward()
                opt.step()
                epoch_loss += loss.item()
                n_batches += 1
            losses.append(epoch_loss / max(n_batches, 1))
        return losses

    opt_results = {}
    for opt_name, opt_fn in [
        # TODO: Define four optimiser lambdas
        ("SGD lr=0.01", ____),  # Hint: lambda p: optim.SGD(p, lr=0.01)
        ("SGD+momentum", ____),  # Hint: lambda p: optim.SGD(p, lr=0.01, momentum=0.9)
        ("Adam lr=1e-3", ____),  # Hint: lambda p: optim.Adam(p, lr=1e-3)
        (
            "AdamW lr=1e-3",
            ____,
        ),  # Hint: lambda p: optim.AdamW(p, lr=1e-3, weight_decay=1e-4)
    ]:
        losses = train_for_epochs(
            lambda: MedicalCNN(n_classes=n_classes),
            opt_fn,
            n_epochs=8,
        )
        opt_results[opt_name] = losses
        print(f"  {opt_name:<20}: init={losses[0]:.4f} -> final={losses[-1]:.4f}")

    print(
        """
Optimiser guide:
  SGD:           Vanilla gradient descent. Slow, needs tuning.
  SGD+momentum:  Accumulates gradient history. Faster convergence.
  Adam:          Adaptive learning rates per parameter. Default choice.
  AdamW:         Adam with decoupled weight decay. Best for regularisation.

  Rule of thumb: Start with AdamW (lr=1e-3, weight_decay=1e-4).
  For fine-tuning pretrained models: SGD+momentum often better.
"""
    )

    # ── Checkpoint 5 ─────────────────────────────────────────────────
    assert all(
        losses[-1] < losses[0] for losses in opt_results.values()
    ), "All optimisers should reduce loss"
    print("✓ Checkpoint 5 passed — optimiser comparison\n")


# ══════════════════════════════════════════════════════════════════════
# TASK 6: Learning rate scheduling
# ══════════════════════════════════════════════════════════════════════

if HAS_TORCH:
    print("=== Learning Rate Scheduling ===")

    criterion = nn.BCEWithLogitsLoss()
    n_epochs = 20

    # Train with cosine annealing
    model = MedicalCNN(n_classes=n_classes).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    # TODO: Create a CosineAnnealingLR scheduler with T_max=n_epochs, eta_min=1e-6
    scheduler = ____  # Hint: optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_epochs, eta_min=1e-6)

    history = {"train_loss": [], "val_loss": [], "lr": [], "grad_norm": []}

    for epoch in range(n_epochs):
        model.train()
        train_losses = []
        epoch_grad_norms = []

        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            logits = model(X_batch)
            loss = criterion(logits, y_batch)
            loss.backward()

            # TODO: Compute gradient norm (L2 norm across all parameters)
            total_norm = 0.0
            for p in model.parameters():
                if p.grad is not None:
                    # TODO: Accumulate squared gradient norms
                    total_norm += ____  # Hint: p.grad.data.norm(2).item() ** 2
            # TODO: Take square root to get the total norm
            total_norm = ____  # Hint: total_norm ** 0.5
            epoch_grad_norms.append(total_norm)

            # TODO: Clip gradients with max_norm=1.0
            ____  # Hint: torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            train_losses.append(loss.item())

        model.eval()
        val_losses = []
        with torch.no_grad():
            for X_batch, y_batch in test_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                val_losses.append(criterion(model(X_batch), y_batch).item())

        history["train_loss"].append(np.mean(train_losses))
        history["val_loss"].append(np.mean(val_losses))
        history["lr"].append(scheduler.get_last_lr()[0])
        history["grad_norm"].append(np.mean(epoch_grad_norms))

        # TODO: Step the scheduler
        ____  # Hint: scheduler.step()

        if (epoch + 1) % 5 == 0:
            print(
                f"Epoch {epoch + 1}/{n_epochs}: "
                f"train={history['train_loss'][-1]:.4f}, "
                f"val={history['val_loss'][-1]:.4f}, "
                f"lr={history['lr'][-1]:.6f}, "
                f"grad={history['grad_norm'][-1]:.4f}"
            )

    print(
        """
LR schedule guide:
  Step decay:         Reduce by factor every N epochs. Simple.
  Cosine annealing:   LR follows cosine curve. Smooth decay.
  Warmup + cosine:    Linear warmup then cosine. For transformers.
  One-cycle:          Ramp up then down. Fast convergence.
  ReduceLROnPlateau:  Reduce when metric stops improving. Adaptive.
"""
    )

    # ── Checkpoint 6 ─────────────────────────────────────────────────
    assert (
        history["train_loss"][-1] < history["train_loss"][0]
    ), "Training loss should decrease"
    assert history["lr"][-1] < history["lr"][0], "Cosine annealing should reduce LR"
    # INTERPRETATION: Cosine annealing: LR(t) = eta_min + 0.5*(eta_max - eta_min)*(1 + cos(pi*t/T)).
    # Large steps early (escape local minima), fine-tuning late.
    print("\n✓ Checkpoint 6 passed — cosine annealing LR schedule\n")


# ══════════════════════════════════════════════════════════════════════
# TASK 7: Dropout and batch normalisation
# ══════════════════════════════════════════════════════════════════════

if HAS_TORCH:
    print("=== Dropout and Batch Normalisation ===")

    # Demonstrate dropout: train vs eval mode
    dropout_layer = nn.Dropout(p=0.5)
    test_input = torch.ones(1, 10)

    # TODO: Set dropout to train mode and apply it
    dropout_layer.train()
    dropped = ____  # Hint: dropout_layer(test_input)
    n_zeros_train = (dropped == 0).sum().item()

    # TODO: Set dropout to eval mode and apply it
    dropout_layer.eval()
    not_dropped = ____  # Hint: dropout_layer(test_input)
    n_zeros_eval = (not_dropped == 0).sum().item()

    print(f"Dropout (p=0.5):")
    print(f"  Train mode: {n_zeros_train}/10 neurons zeroed (random ~50%)")
    print(f"  Eval mode:  {n_zeros_eval}/10 neurons zeroed (should be 0)")
    print(f"  Dropout during training prevents co-adaptation of neurons.")
    print(f"  At eval time, all neurons are active (scaled by 1/(1-p)).")

    # Demonstrate batch normalisation
    # TODO: Create BatchNorm1d with 4 features
    bn = ____  # Hint: nn.BatchNorm1d(4)
    test_batch = torch.randn(16, 4) * 5 + 3  # Mean ~3, Std ~5

    bn.train()
    normalised = bn(test_batch)
    print(f"\nBatch Normalisation:")
    print(
        f"  Before: mean={test_batch.mean(0).detach().numpy().round(2)}, "
        f"std={test_batch.std(0).detach().numpy().round(2)}"
    )
    print(
        f"  After:  mean={normalised.mean(0).detach().numpy().round(2)}, "
        f"std={normalised.std(0).detach().numpy().round(2)}"
    )
    print(f"  BN normalises to mean~0, std~1 per feature across the batch.")
    print(f"  Stabilises training, enables higher learning rates.")

    # Compare: with vs without dropout and BN
    def train_variant(use_dropout: bool, use_bn: bool, label: str) -> list[float]:
        layers = [nn.Conv2d(1, 32, 3, padding=1)]
        if use_bn:
            layers.append(nn.BatchNorm2d(32))
        layers.extend([nn.ReLU(), nn.MaxPool2d(2)])
        layers.append(nn.Conv2d(32, 64, 3, padding=1))
        if use_bn:
            layers.append(nn.BatchNorm2d(64))
        layers.extend(
            [
                nn.ReLU(),
                nn.AdaptiveAvgPool2d(4),
                nn.Flatten(),
                nn.Linear(64 * 4 * 4, 64),
                nn.ReLU(),
            ]
        )
        if use_dropout:
            layers.append(nn.Dropout(0.3))
        layers.append(nn.Linear(64, n_classes))

        net = nn.Sequential(*layers).to(device)
        opt = optim.AdamW(net.parameters(), lr=1e-3)
        crit = nn.BCEWithLogitsLoss()
        val_losses = []
        for epoch in range(10):
            net.train()
            for X_b, y_b in train_loader:
                X_b, y_b = X_b.to(device), y_b.to(device)
                opt.zero_grad()
                crit(net(X_b), y_b).backward()
                opt.step()
            net.eval()
            vl = []
            with torch.no_grad():
                for X_b, y_b in test_loader:
                    X_b, y_b = X_b.to(device), y_b.to(device)
                    vl.append(crit(net(X_b), y_b).item())
            val_losses.append(np.mean(vl))
        return val_losses

    variants = {}
    for use_do, use_bn, label in [
        (False, False, "Baseline"),
        (True, False, "+Dropout"),
        (False, True, "+BatchNorm"),
        (True, True, "+Both"),
    ]:
        vl = train_variant(use_do, use_bn, label)
        variants[label] = vl
        print(f"  {label:<12}: val_loss {vl[0]:.4f} -> {vl[-1]:.4f}")

    # ── Checkpoint 7 ─────────────────────────────────────────────────
    assert n_zeros_train > 0, "Dropout should zero neurons in train mode"
    assert n_zeros_eval == 0, "Dropout should NOT zero neurons in eval mode"
    print("\n✓ Checkpoint 7 passed — dropout and batch normalisation\n")


# ══════════════════════════════════════════════════════════════════════
# TASK 8: Gradient clipping and monitoring
# ══════════════════════════════════════════════════════════════════════

if HAS_TORCH:
    print("=== Gradient Clipping ===")
    print(f"Gradient norm history (from Task 6 training):")
    print(f"  Initial grad norm: {history['grad_norm'][0]:.4f}")
    print(f"  Final grad norm:   {history['grad_norm'][-1]:.4f}")
    print(f"  Max grad norm:     {max(history['grad_norm']):.4f}")
    print(f"  Clip threshold:    1.0")

    n_clipped = sum(1 for g in history["grad_norm"] if g > 1.0)
    print(f"  Epochs with clipping: {n_clipped}/{n_epochs}")

    print(
        """
Gradient clipping prevents exploding gradients:
  Max-norm clipping:  scale all gradients if total norm > threshold
    torch.nn.utils.clip_grad_norm_(params, max_norm=1.0)
  Value clipping:     clip each gradient element to [-value, value]
    torch.nn.utils.clip_grad_value_(params, clip_value=0.5)

  Max-norm is preferred (preserves gradient direction).
  Without clipping, one bad batch can send weights to infinity.
"""
    )

    # ── Checkpoint 8 ─────────────────────────────────────────────────
    assert all(g > 0 for g in history["grad_norm"]), "Gradient norms should be positive"
    print("✓ Checkpoint 8 passed — gradient monitoring\n")


# ══════════════════════════════════════════════════════════════════════
# TASK 9: Early stopping
# ══════════════════════════════════════════════════════════════════════

if HAS_TORCH:
    print("=== Early Stopping ===")

    class EarlyStopping:
        """Monitor validation loss and stop if it doesn't improve."""

        def __init__(self, patience: int = 5, min_delta: float = 0.001):
            self.patience = patience
            self.min_delta = min_delta
            self.counter = 0
            self.best_loss = float("inf")
            self.should_stop = False

        def step(self, val_loss: float) -> bool:
            # TODO: If val_loss improved by at least min_delta, update best_loss and reset counter
            if ____:  # Hint: val_loss < self.best_loss - self.min_delta
                self.best_loss = val_loss
                self.counter = 0
            else:
                # TODO: Increment counter; if patience exceeded, set should_stop = True
                self.counter += ____  # Hint: 1
                if self.counter >= self.patience:
                    self.should_stop = ____  # Hint: True
            return self.should_stop

    # Demonstrate on our training history
    es = EarlyStopping(patience=5, min_delta=0.001)
    stopped_epoch = n_epochs
    for epoch, vl in enumerate(history["val_loss"]):
        if es.step(vl):
            stopped_epoch = epoch
            break

    print(f"Early stopping analysis:")
    print(f"  Best val loss: {es.best_loss:.4f}")
    if stopped_epoch < n_epochs:
        print(
            f"  Would stop at epoch: {stopped_epoch + 1} (saved {n_epochs - stopped_epoch - 1} epochs)"
        )
    else:
        print(f"  No early stop triggered (val loss kept improving within patience)")
    print(f"  Patience: {es.patience} epochs")

    print(
        """
Early stopping prevents overfitting:
  1. Monitor validation loss each epoch
  2. If val_loss hasn't improved for 'patience' epochs, stop
  3. Restore weights from the best epoch
  4. Acts as implicit regularisation (limits effective training time)

  Combine with LR scheduling: reduce LR first, then early stop.
"""
    )

    # ── Checkpoint 9 ─────────────────────────────────────────────────
    assert es.best_loss < float("inf"), "Should find a best val loss"
    print("✓ Checkpoint 9 passed — early stopping\n")


# ══════════════════════════════════════════════════════════════════════
# TASK 10: ONNX export with OnnxBridge
# ══════════════════════════════════════════════════════════════════════

if HAS_TORCH:
    print("=== ONNX Export with OnnxBridge ===")

    # TODO: Create an OnnxBridge instance
    bridge = ____  # Hint: OnnxBridge()
    # TODO: Check model compatibility with check_compatibility(model, framework="pytorch")
    compat = ____  # Hint: bridge.check_compatibility(model, framework="pytorch")
    print(f"Compatible: {compat.compatible}")
    print(f"Confidence: {compat.confidence}")

    # TODO: Export model with bridge.export() — output_path="medical_cnn.onnx"
    export_result = ____  # Hint: bridge.export(model=model, framework="pytorch", output_path="medical_cnn.onnx", n_features=None)

    print(f"Export: {export_result.success}")
    if export_result.success and export_result.onnx_path:
        if export_result.model_size_bytes:
            print(f"Model size: {export_result.model_size_bytes / 1024:.1f} KB")

        # Validate
        sample_input = torch.from_numpy(X_test[:10]).to(device)
        # TODO: Validate ONNX model matches PyTorch output with tolerance=1e-4
        validation = ____  # Hint: bridge.validate(model=model, onnx_path=export_result.onnx_path, sample_input=sample_input, tolerance=1e-4)
        print(f"Valid: {validation.valid}")
        print(f"Max diff: {validation.max_diff:.8f}")

    # ── Checkpoint 10 ────────────────────────────────────────────────
    assert compat.compatible, "MedicalCNN should be ONNX-compatible"
    # INTERPRETATION: ONNX is vendor-neutral. Once exported, the model runs
    # on any ONNX runtime without the PyTorch dependency.
    print("\n✓ Checkpoint 10 passed — ONNX export\n")


# ══════════════════════════════════════════════════════════════════════
# TASK 11: Loss function taxonomy and the USML bridge
# ══════════════════════════════════════════════════════════════════════

if HAS_TORCH:
    print("=== Loss Function Taxonomy ===")
    print(
        """
Regression losses:
  MSE:   mean((y - y_hat)^2).  Standard, sensitive to outliers.
  MAE:   mean(|y - y_hat|).    Robust to outliers.

Classification losses:
  Cross-entropy:  -sum(y * log(y_hat)).  Standard for multi-class.
  Binary CE:      -y*log(y_hat) - (1-y)*log(1-y_hat).  Binary/multi-label.
  Focal loss:     -alpha*(1-p)^gamma * log(p).  Imbalanced classes.

Similarity/metric learning:
  Contrastive:    y*d^2 + (1-y)*max(0, margin-d)^2.  Pairs.
  Triplet:        max(0, d(a,p) - d(a,n) + margin).  Anchor-pos-neg.

Generative/distribution:
  KL divergence:  sum(p * log(p/q)).  Distribution matching.
  Reconstruction: ||x - x_hat||^2.  Autoencoders.

Choosing a loss function:
  The loss function defines WHAT the network learns.
  MSE for regression, CE for classification, triplet for embeddings.
  Focal loss for imbalanced data (fraud, medical).
"""
    )

    # Visualise training dynamics
    # TODO: Create a ModelVisualizer instance
    viz = ____  # Hint: ModelVisualizer()

    # TODO: Create training history chart for train/val loss
    fig_loss = ____  # Hint: viz.training_history({"Train Loss": history["train_loss"], "Val Loss": history["val_loss"]}, x_label="Epoch")
    fig_loss.update_layout(title="Training and Validation Loss")
    fig_loss.write_html("ex8_loss_curves.html")

    # TODO: Create training history chart for learning rate
    fig_lr = ____  # Hint: viz.training_history({"Learning Rate": history["lr"]}, x_label="Epoch")
    fig_lr.update_layout(title="Cosine Annealing LR Schedule")
    fig_lr.write_html("ex8_lr_schedule.html")

    # TODO: Create training history chart for gradient norm
    fig_grad = ____  # Hint: viz.training_history({"Gradient Norm": history["grad_norm"]}, x_label="Epoch")
    fig_grad.update_layout(title="Gradient Norm During Training")
    fig_grad.write_html("ex8_gradient_norms.html")

    print("Saved: ex8_loss_curves.html, ex8_lr_schedule.html, ex8_gradient_norms.html")

    # ── Checkpoint 11 ────────────────────────────────────────────────
    print("\n✓ Checkpoint 11 passed — loss taxonomy and training visualisation\n")
    print("\n✓ Exercise 8 complete — DL foundations and training toolkit")

else:
    print("\n✓ Exercise 8 skipped — PyTorch not installed")

# The USML bridge summary
print(
    """
THE USML BRIDGE — COMPLETE:
  Clustering (no labels)          -> discover groups
  Dimensionality reduction        -> discover axes
  Association rules               -> discover co-occurrence patterns (manual)
  Matrix factorisation            -> discover latent factors (automatic, linear)
  Neural hidden layers            -> discover representations (automatic, nonlinear)
  "Hidden layers ARE USML + error feedback" — the bridge is complete.
"""
)


# ══════════════════════════════════════════════════════════════════════
# REFLECTION
# ══════════════════════════════════════════════════════════════════════
print("═" * 70)
print("  MODULE 4 MASTERY — UNSUPERVISED ML AND DL FOUNDATIONS")
print("═" * 70)
print(
    f"""
  M4 CAPSTONE CHECKLIST:
  ✓ Clustering (Ex 1): K-means, hierarchical, DBSCAN, HDBSCAN, spectral, GMM
  ✓ EM algorithm (Ex 2): derive, implement, verify convergence guarantee
  ✓ Dimensionality reduction (Ex 3): PCA=SVD, Kernel PCA, t-SNE, UMAP
  ✓ Anomaly detection (Ex 4): Z-score, IQR, IsolationForest, LOF, EnsembleEngine
  ✓ Association rules (Ex 5): Apriori from scratch, FP-Growth, rule features
  ✓ NLP / topic modelling (Ex 6): TF-IDF, BM25, NMF, LDA, BERTopic, NPMI
  ✓ Recommender systems (Ex 7): CF, ALS, P@k, MAP, hybrid, THE PIVOT
  ✓ DL foundations (Ex 8): activations, init, optimisers, LR schedule, ONNX

  THIS EXERCISE:
  ✓ XOR proof: linear layers cannot learn nonlinear boundaries
  ✓ Activation functions: ReLU, GELU, Tanh, Swish — when to use each
  ✓ Weight init: Kaiming for ReLU, Xavier for Sigmoid/Tanh
  ✓ ResBlock: skip connections prevent vanishing gradients
  ✓ Optimisers: SGD -> momentum -> Adam -> AdamW (adaptive LR per param)
  ✓ LR scheduling: cosine annealing for smooth decay
  ✓ Dropout: regularisation by random neuron zeroing
  ✓ BatchNorm: stabilise training by normalising activations
  ✓ Gradient clipping: prevent exploding gradients
  ✓ Early stopping: implicit regularisation via validation monitoring
  ✓ OnnxBridge: production-ready model export
  ✓ Loss taxonomy: MSE, CE, focal, triplet, KL divergence

  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  MODULE 5 PREVIEW: DEEP LEARNING IN VISION AND TRANSFER LEARNING
  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

  Module 4 built neural networks from scratch and established the bridge
  from unsupervised ML to deep learning. Module 5 takes those foundations
  and applies them to real-world vision and NLP tasks:

  M5 covers:
  • CNNs for image classification (Fashion-MNIST, CIFAR-10)
  • Transfer learning with pretrained models (ResNet, EfficientNet)
  • NLP transformers: attention, BERT, fine-tuning
  • ONNX deployment via InferenceServer
  • Vision transformers (ViT) and modern architectures

  The CNN architecture from this exercise reappears in M5 as the
  foundation for transfer learning and fine-tuning.
"""
)
print("═" * 70)
