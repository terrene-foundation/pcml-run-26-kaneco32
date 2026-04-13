# Copyright 2026 Terrene Foundation
# SPDX-License-Identifier: Apache-2.0
"""
# ════════════════════════════════════════════════════════════════════════
# MLFP06 — Exercise 2: LLM Fine-Tuning — LoRA, Adapters, and the
#                       Technique Landscape
# ════════════════════════════════════════════════════════════════════════
#
# WHAT YOU'LL LEARN:
#   After completing this exercise, you will be able to:
#   - Implement LoRA from scratch as an nn.Module (understand the maths:
#     W_new = W + A @ B, rank r << d)
#   - Implement adapter layers from scratch (bottleneck: FC -> act -> FC)
#   - Compare LoRA vs adapters across 4 dimensions (param efficiency,
#     implementation complexity, flexibility, modularity)
#   - Survey all 10 fine-tuning techniques and select the right one
#   - Explain model merging techniques (TIES, DARE, SLERP, task arithmetic)
#   - Describe quantisation methods (GPTQ, AWQ, GGUF, QLoRA)
#   - Use kailash-align AlignmentPipeline for SFT with LoRA
#   - Register and version adapters in AdapterRegistry
#
# PREREQUISITES:
#   Exercise 1 (LLM fundamentals, transformer architecture from M5.4).
#   Linear algebra: matrix rank, SVD (M4.3) — LoRA IS low-rank
#   factorisation applied to weight updates.
#
# ESTIMATED TIME: ~180 min
#
# TASKS:
#    1. Load SFT dataset (IMDB instruction-response pairs)
#    2. Implement LoRA layer FROM SCRATCH (nn.Module)
#    3. Implement adapter layer FROM SCRATCH (bottleneck module)
#    4. Compare LoRA vs adapter: parameter count, architecture
#    5. Fine-tuning landscape survey (all 10 techniques)
#    6. Model merging: TIES, DARE, SLERP, task arithmetic
#    7. Quantisation overview: GPTQ, AWQ, GGUF, QLoRA
#    8. AlignmentPipeline SFT training with LoRA
#    9. Register adapter in AdapterRegistry
#   10. Parameter reduction analysis across ranks
#
# DATASET: IMDB sentiment (stanfordnlp/imdb on HuggingFace)
#   25,000 real movie reviews with binary positive/negative labels.
#   Reformatted as SFT instruction-response pairs.  Subsampled to 2,000
#   for fast training.  Split: 90% train / 10% eval.
#   Base model: from env variable SFT_BASE_MODEL.
#
# ════════════════════════════════════════════════════════════════════════
"""
from __future__ import annotations

import math
import os
from pathlib import Path

import polars as pl
import torch
import torch.nn as nn

from kailash_align import AdapterRegistry, AlignmentConfig, AlignmentPipeline

from shared.kailash_helpers import get_device, setup_environment

setup_environment()

device = get_device()
print(f"Compute device: {device}")


# ══════════════════════════════════════════════════════════════════════════
# TASK 1: Load SFT Dataset (IMDB sentiment from HuggingFace)
# ══════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 70)
print("TASK 1: Load SFT Dataset")
print("=" * 70)

CACHE_DIR = Path("data/mlfp06/imdb")
CACHE_DIR.mkdir(parents=True, exist_ok=True)
CACHE_FILE = CACHE_DIR / "imdb_sft_2k.parquet"

if CACHE_FILE.exists():
    print(f"Loading cached IMDB SFT pairs from {CACHE_FILE}")
    sft_data = pl.read_parquet(CACHE_FILE)
else:
    print("Downloading stanfordnlp/imdb from HuggingFace (first run)...")
    from datasets import load_dataset

    ds = load_dataset("stanfordnlp/imdb", split="train")
    ds = ds.shuffle(seed=42).select(range(min(2000, len(ds))))

    label_names = {0: "negative", 1: "positive"}
    rows = []
    for row in ds:
        review = row["text"][:1500]
        sentiment = label_names[row["label"]]
        rows.append(
            {
                "instruction": (
                    "Classify the sentiment of the following movie review as "
                    "either 'positive' or 'negative', then briefly justify "
                    f"your answer.\n\nReview: {review}"
                ),
                "response": (
                    f"Sentiment: {sentiment}. The reviewer expresses a clearly "
                    f"{sentiment} reaction to the film."
                ),
                "text": review,
                "label": sentiment,
            }
        )
    sft_data = pl.DataFrame(rows)
    sft_data.write_parquet(CACHE_FILE)
    print(f"Cached {sft_data.height} SFT pairs to {CACHE_FILE}")

print(f"Shape: {sft_data.shape}")
print(f"Columns: {sft_data.columns}")
print(f"Sample instruction:\n{sft_data['instruction'][0][:300]}...")

n_train = int(sft_data.height * 0.9)
train_data = sft_data[:n_train]
eval_data = sft_data[n_train:]
print(f"Train: {train_data.height}, Eval: {eval_data.height}")

# ── Checkpoint 1 ─────────────────────────────────────────────────────────
assert sft_data.height > 0, "Task 1: SFT dataset should not be empty"
assert "instruction" in sft_data.columns, "Dataset needs 'instruction' column"
assert "response" in sft_data.columns, "Dataset needs 'response' column"
print(f"✓ Checkpoint 1 passed — {sft_data.height} SFT pairs loaded\n")

# INTERPRETATION: SFT data = (instruction, response) pairs.  The model
# learns to follow instructions by maximising P(response | instruction).
# Quality matters more than quantity: 500 high-quality domain-specific
# pairs can outperform 10,000 generic pairs.


# ══════════════════════════════════════════════════════════════════════════
# TASK 2: Implement LoRA Layer FROM SCRATCH (nn.Module)
# ══════════════════════════════════════════════════════════════════════════

print("=" * 70)
print("TASK 2: LoRA Layer — From-Scratch Implementation")
print("=" * 70)

print(
    """
LoRA theory (Hu et al., 2021):
  Full fine-tuning: W_new = W + delta_W   (delta_W is d × d)
  LoRA:             W_new = W + A @ B     (A is d × r, B is r × d)
  rank r << d  =>  A @ B has at most r non-zero singular values.
  Pre-trained W remains FROZEN; only A and B are trained.
  Connects to M4.3 SVD: LoRA IS low-rank factorisation of delta_W.
"""
)


# TODO: Implement class LoRALayer(nn.Module) with:
#   - __init__(self, in_features, out_features, rank=8, alpha=16.0, dropout=0.0)
#       self.rank, self.alpha, self.scaling = alpha / rank
#       self.lora_A = nn.Parameter(torch.empty(in_features, rank))
#       self.lora_B = nn.Parameter(torch.zeros(rank, out_features))
#       self.dropout = nn.Dropout(p=dropout) if dropout > 0 else nn.Identity()
#       call self.reset_parameters()
#   - reset_parameters(self): nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
#   - forward(self, x): dropped = dropout(x); return (dropped @ A @ B) * scaling
# Hint: B is zero-initialised so LoRA starts as identity at training start
____


# TODO: Implement class LoRALinear(nn.Module) that:
#   - Stores a frozen pretrained_linear and a trainable LoRALayer
#   - forward(self, x): return self.linear(x) + self.lora(x)
# Hint: freeze with: for param in self.linear.parameters(): param.requires_grad = False
____


# Demonstrate with a synthetic linear layer
d_model = 512
lora_rank = 8
pretrained = nn.Linear(d_model, d_model).to(device)
lora_linear = LoRALinear(pretrained, rank=lora_rank, alpha=16.0).to(device)

frozen_params = sum(p.numel() for p in lora_linear.linear.parameters())
trainable_params = sum(
    p.numel() for p in lora_linear.lora.parameters() if p.requires_grad
)
x_test = torch.randn(2, 10, d_model, device=device)
y_test = lora_linear(x_test)

print(f"LoRA layer created: d={d_model}, r={lora_rank}")
print(f"  Frozen params:    {frozen_params:,}")
print(f"  Trainable params: {trainable_params:,}")
print(f"  Ratio: {trainable_params / frozen_params:.4%} of original")
print(f"  Output shape:     {y_test.shape}")

# ── Checkpoint 2 ─────────────────────────────────────────────────────────
assert isinstance(lora_linear, LoRALinear), "Task 2: LoRALinear should be created"
assert (
    trainable_params == d_model * lora_rank + lora_rank * d_model
), f"LoRA params should be 2*d*r = {2 * d_model * lora_rank}"
assert y_test.shape == x_test.shape, "Output shape should match input shape"
print("✓ Checkpoint 2 passed — LoRA from-scratch implementation verified\n")

# INTERPRETATION: LoRA trainable params = 2*d*r.
# For d=512, r=8: 8,192 vs 262,144 full params = 3.1%.
# The B=0 init means LoRA starts as identity, then learns the domain delta.


# ══════════════════════════════════════════════════════════════════════════
# TASK 3: Implement Adapter Layer FROM SCRATCH (bottleneck module)
# ══════════════════════════════════════════════════════════════════════════

print("=" * 70)
print("TASK 3: Adapter Layer — From-Scratch Implementation")
print("=" * 70)

print(
    """
Adapter theory (Houlsby et al., 2019):
  Insert a bottleneck module between transformer layers:
    Input (d) -> Down-project (d -> b) -> Activation -> Up-project (b -> d)
  where b << d.  Original transformer weights remain FROZEN.
  Residual connection: output = adapter(x) + x
"""
)


# TODO: Implement class AdapterLayer(nn.Module) with:
#   - __init__(self, d_model, bottleneck_dim=64, dropout=0.1)
#       self.layer_norm = nn.LayerNorm(d_model)
#       self.down_proj = nn.Linear(d_model, bottleneck_dim)
#       self.activation = nn.GELU()
#       self.up_proj = nn.Linear(bottleneck_dim, d_model)
#       self.dropout = nn.Dropout(dropout)
#       zero-init up_proj weight and bias so adapter starts as identity
#   - forward(self, x): residual=x; h = norm->down->act->up->dropout; return h+residual
# Hint: nn.init.zeros_(self.up_proj.weight); nn.init.zeros_(self.up_proj.bias)
____


# TODO: Implement class AdapterTransformerBlock(nn.Module) that:
#   - Wraps original_block (freezes its params) and adds AdapterLayer
#   - forward(self, x): h = block(x); return adapter(h)
# Hint: for param in self.block.parameters(): param.requires_grad = False
____


# Demonstrate adapter
adapter = AdapterLayer(d_model=d_model, bottleneck_dim=64).to(device)
adapter_params = sum(p.numel() for p in adapter.parameters())
y_adapter = adapter(x_test)

print(f"Adapter layer: d={d_model}, bottleneck=64")
print(f"  Adapter params: {adapter_params:,}")
print(f"  Output shape:   {y_adapter.shape}")

# ── Checkpoint 3 ─────────────────────────────────────────────────────────
assert isinstance(adapter, AdapterLayer), "Task 3: AdapterLayer should be created"
assert y_adapter.shape == x_test.shape, "Adapter output shape should match input"
expected_adapter_params = (
    d_model
    + d_model  # LayerNorm weight + bias
    + d_model * 64
    + 64  # down_proj
    + 64 * d_model
    + d_model  # up_proj
)
assert (
    adapter_params == expected_adapter_params
), f"Adapter params mismatch: got {adapter_params}, expected {expected_adapter_params}"
print("✓ Checkpoint 3 passed — adapter from-scratch implementation verified\n")

# INTERPRETATION: Adapter params = 2*d*b + 2*b + 2*d.
# For d=512, b=64: ~66K params vs LoRA r=8 (~8K).  More expressive
# (GELU nonlinearity) but cannot be merged back into frozen weights.


# ══════════════════════════════════════════════════════════════════════════
# TASK 4: Compare LoRA vs Adapter
# ══════════════════════════════════════════════════════════════════════════

print("=" * 70)
print("TASK 4: LoRA vs Adapter Comparison")
print("=" * 70)


# TODO: Build comparison_data dict with key "Dimension" and columns "LoRA", "Adapter"
#   covering 4 dimensions: Parameter Update Mechanism, Parameter Efficiency,
#   Implementation Complexity, Flexibility & Modularity.
#   Then: comparison_df = pl.DataFrame(comparison_data) and print it.
# Hint: include trainable_params and adapter_params in the efficiency row
____


# TODO: Print a parameter efficiency sweep for:
#   LoRA r=4, r=8, r=16, r=32 and Adapter b=32, b=64, b=128 and Full fine-tune
#   showing name, param count, and percentage of d_model*d_model
# Hint: LoRA = 2*d*r; Adapter = 2*d*b + 2*b + 2*d
____

# ── Checkpoint 4 ─────────────────────────────────────────────────────────
assert comparison_df.height == 4, "Task 4: comparison should cover 4 dimensions"
print("\n✓ Checkpoint 4 passed — LoRA vs adapter comparison complete\n")

# INTERPRETATION: LoRA dominates for single-task adaptation (merge A,B into W
# at inference for zero overhead).  Adapters work best when multiple tasks
# share a base model simultaneously (stack adapters, hot-swap at inference).


# ══════════════════════════════════════════════════════════════════════════
# TASK 5: Fine-Tuning Landscape Survey (all 10 techniques)
# ══════════════════════════════════════════════════════════════════════════

print("=" * 70)
print("TASK 5: Fine-Tuning Technique Landscape")
print("=" * 70)


# TODO: Build a list of 10 dicts, each with keys: name, mechanism, params, when
#   Cover: LoRA, Adapter Layers, Prefix Tuning, Prompt Tuning, Full Fine-Tuning,
#   LLRD, QLoRA, Continued Pre-Training, RLHF/PPO, DPO
#   Then: techniques_df = pl.DataFrame(techniques) and print it
# Hint: "params" should be a string like "0.1-5% of full"
____

# ── Checkpoint 5 ─────────────────────────────────────────────────────────
assert techniques_df.height == 10, "Task 5: should survey exactly 10 techniques"
print("\n✓ Checkpoint 5 passed — 10 fine-tuning techniques surveyed\n")


# ══════════════════════════════════════════════════════════════════════════
# TASK 6: Model Merging (TIES, DARE, SLERP, task arithmetic)
# ══════════════════════════════════════════════════════════════════════════

print("=" * 70)
print("TASK 6: Model Merging Techniques")
print("=" * 70)


# TODO: Print an explanation of 4 merging techniques:
#   - Task arithmetic: theta_merged = theta_base + lambda*(theta_ft - theta_base)
#   - TIES: Trim (keep top-k% by magnitude), Elect (majority sign vote), Merge
#   - DARE: randomly drop small task-vector deltas before merging
#   - SLERP: spherical linear interpolation on the unit hypersphere
# Hint: explain what problem each technique solves (e.g. TIES resolves sign conflicts)
____


# TODO: Simulate a toy task arithmetic merge:
#   theta_base = torch.randn(10)
#   theta_task_a = theta_base + 0.1 * torch.randn(10)
#   theta_merged = theta_base + 0.5 * (theta_task_a - theta_base)  # lambda=0.5
#   Print L2 distance: torch.dist(theta_base, theta_merged).item()
# Hint: the distance should be roughly half the distance between base and task_a
____

# ── Checkpoint 6 ─────────────────────────────────────────────────────────
print("✓ Checkpoint 6 passed — model merging techniques explained\n")


# ══════════════════════════════════════════════════════════════════════════
# TASK 7: Quantisation Overview (GPTQ, AWQ, GGUF, QLoRA)
# ══════════════════════════════════════════════════════════════════════════

print("=" * 70)
print("TASK 7: Quantisation Overview")
print("=" * 70)


# TODO: Print an explanation of 4 quantisation methods:
#   - GPTQ: post-training quantisation, layer-by-layer, minimise reconstruction error
#   - AWQ: activation-aware weight quantisation, scales weights by activation magnitudes
#   - GGUF: CPU-friendly format (llama.cpp), mixed precision Q4/Q5/Q8 layers
#   - QLoRA: 4-bit NF4 quantised base + bf16 LoRA adapters + paged optimiser
# Also print a memory table for a 7B model at fp32/fp16/int8/int4 (7B * bytes_per_param)
# Hint: fp32=4B, fp16=2B, int8=1B, int4=0.5B per parameter
____

# ── Checkpoint 7 ─────────────────────────────────────────────────────────
print("✓ Checkpoint 7 passed — quantisation overview complete\n")


# ══════════════════════════════════════════════════════════════════════════
# TASK 8: AlignmentPipeline SFT Training with LoRA
# ══════════════════════════════════════════════════════════════════════════

print("=" * 70)
print("TASK 8: AlignmentPipeline SFT with LoRA")
print("=" * 70)


# TODO: Build and run an SFT pipeline:
#   1. base_model = os.environ.get("SFT_BASE_MODEL", "microsoft/Phi-3-mini-4k-instruct")
#   2. config = AlignmentConfig(
#          method="sft", model_name=base_model,
#          lora_rank=8, lora_alpha=16.0, lora_dropout=0.05,
#          target_modules=["q_proj", "v_proj"],
#          num_train_epochs=1, per_device_train_batch_size=4,
#          max_seq_length=512, output_dir="outputs/mlfp06/sft_lora"
#      )
#   3. train_samples = train_data.select(["instruction", "response"]).to_dicts()
#   4. sft_pipeline = AlignmentPipeline(config)
#   5. metrics = sft_pipeline.train(train_samples)
#   6. Print metrics (loss, perplexity, steps)
# Hint: AlignmentConfig(method="sft", ...) — method is always a lowercase string
____

# ── Checkpoint 8 ─────────────────────────────────────────────────────────
assert sft_pipeline is not None, "Task 8: SFT pipeline should be created"
print("✓ Checkpoint 8 passed — AlignmentPipeline SFT training initiated\n")

# INTERPRETATION: AlignmentPipeline wraps HuggingFace TRL under the hood.
# LoRA injects trainable A,B into attention projections, freezes the rest.


# ══════════════════════════════════════════════════════════════════════════
# TASK 9: Register Adapter in AdapterRegistry
# ══════════════════════════════════════════════════════════════════════════

print("=" * 70)
print("TASK 9: AdapterRegistry — Version and Retrieve Adapters")
print("=" * 70)


# TODO: Register and retrieve the trained adapter:
#   1. registry = AdapterRegistry(base_dir="outputs/mlfp06/adapters")
#   2. adapter_id = registry.register(
#          name="imdb-sentiment-lora",
#          base_model=base_model,
#          adapter_path="outputs/mlfp06/sft_lora",
#          method="sft",
#          metadata={"dataset": "imdb", "rank": 8, "task": "sentiment"}
#      )
#   3. loaded = registry.load(adapter_id)
#   4. Print adapter_id, loaded.name, loaded.method, loaded.metadata
# Hint: registry.register(...) returns a string adapter_id
____

# ── Checkpoint 9 ─────────────────────────────────────────────────────────
assert adapter_id is not None, "Task 9: adapter registration should return an ID"
assert loaded is not None, "Task 9: adapter should be loadable from registry"
print(f"✓ Checkpoint 9 passed — adapter registered with ID: {adapter_id}\n")


# ══════════════════════════════════════════════════════════════════════════
# TASK 10: Parameter Reduction Analysis Across Ranks
# ══════════════════════════════════════════════════════════════════════════

print("=" * 70)
print("TASK 10: Parameter Reduction Analysis")
print("=" * 70)


# TODO: For a realistic 7B model (d_model=4096, n_layers=32, 4 attention projections):
#   full_params = d_model^2 * 4 * n_layers
#   For each rank in [1, 2, 4, 8, 16, 32, 64]:
#       lora_params = 2 * d_model * rank * 4 * n_layers
#       pct = lora_params / full_params * 100
#   Build a polars DataFrame with columns: rank, lora_params, pct_of_full
#   and print it
# Hint: pct will range from ~0.01% (r=1) to ~1.56% (r=64)
____

# ── Checkpoint 10 ─────────────────────────────────────────────────────────
print("✓ Checkpoint 10 passed — parameter reduction analysis complete\n")

# INTERPRETATION: LoRA rank 8 on a 7B model: ~0.1% of full parameters.
# Higher rank = more expressive but diminishing returns after rank 16-32.
# Sweet spot: rank 8-16 for most domain adaptation tasks.


# ══════════════════════════════════════════════════════════════════════════
# REFLECTION
# ══════════════════════════════════════════════════════════════════════════
print("═" * 70)
print("  WHAT YOU'VE MASTERED")
print("═" * 70)
print(
    """
  ✓ LoRA from scratch: A @ B low-rank factorisation, B=0 init, scaling
  ✓ Adapter from scratch: bottleneck d->b->d with GELU + residual
  ✓ LoRA vs adapters: parameter efficiency, merge-ability, expressiveness
  ✓ 10 fine-tuning techniques: when to use LoRA, adapters, full fine-tuning
  ✓ Model merging: task arithmetic, TIES, DARE, SLERP
  ✓ Quantisation: GPTQ, AWQ, GGUF, QLoRA — memory vs quality trade-offs
  ✓ AlignmentPipeline: SFT training with LoRA in one API call
  ✓ AdapterRegistry: versioning, registering, and loading adapters
  ✓ Parameter reduction: rank 8 on 7B model ≈ 0.1% of full parameters

  NEXT: Exercise 3 (DPO/GRPO) moves beyond instruction following.
  DPO directly optimises for human preferences by comparing chosen
  vs rejected responses — no separate reward model needed.
"""
)
