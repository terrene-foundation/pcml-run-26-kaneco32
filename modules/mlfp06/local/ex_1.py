# Copyright 2026 Terrene Foundation
# SPDX-License-Identifier: Apache-2.0
"""
# ════════════════════════════════════════════════════════════════════════
# MLFP06 — Exercise 1: LLM Fundamentals, Prompt Engineering, and
#                       Structured Output
# ════════════════════════════════════════════════════════════════════════
#
# WHAT YOU'LL LEARN:
#   After completing this exercise, you will be able to:
#   - Explain how LLMs are pre-trained (next-token prediction vs masked LM)
#     and why scaling laws matter (parameters, data, compute)
#   - Apply 6 prompt engineering techniques: zero-shot, few-shot, CoT,
#     zero-shot CoT, self-consistency (majority vote), and structured
#     output specification
#   - Use Kaizen Delegate with streaming, events, and cost tracking
#   - Define typed Signatures with InputField / OutputField for structured
#     LLM output extraction
#   - Compare prompting strategies quantitatively (accuracy, cost, latency)
#     and explain when each technique helps
#   - Describe inference-time optimisations: KV-cache, speculative decoding,
#     continuous batching
#
# PREREQUISITES:
#   M5 complete (transformers, attention, positional encoding from M5.4).
#   Understanding that LLMs predict the next token — prompts shift which
#   tokens are likely.  No fine-tuning required; prompting is zero-cost
#   adaptation.
#
# ESTIMATED TIME: ~180 min
#
# TASKS:
#    1. LLM foundations recap (pre-training, scaling laws)
#    2. Zero-shot classification with Delegate
#    3. Few-shot with curated example selection
#    4. Chain-of-thought (CoT) prompting
#    5. Zero-shot CoT ("Let's think step by step")
#    6. Self-consistency (sample multiple CoT paths, majority vote)
#    7. Structured prompting with explicit JSON output format
#    8. Kaizen Signature for type-safe structured extraction
#    9. Quantitative comparison across all prompting strategies
#   10. Inference optimisations (KV-cache, speculative decoding, batching)
#
# DATASET: SST-2 Sentiment (stanfordnlp/sst2 on HuggingFace)
#   The Stanford Sentiment Treebank — real movie review snippets labelled
#   positive (1) or negative (0).  Standard NLP benchmark for evaluating
#   transformer language models.  We use a 200-row subsample for fast
#   prompt-engineering experiments.
#
# ════════════════════════════════════════════════════════════════════════
"""
from __future__ import annotations

import asyncio
import json
import os
import time
from collections import Counter
from pathlib import Path

import polars as pl

from kaizen import InputField, OutputField, Signature
from kaizen_agents import Delegate
from kaizen_agents.agents.specialized.simple_qa import SimpleQAAgent

from shared.kailash_helpers import setup_environment

setup_environment()

model = os.environ.get("DEFAULT_LLM_MODEL", os.environ.get("OPENAI_PROD_MODEL"))
if not model:
    raise EnvironmentError("Set DEFAULT_LLM_MODEL or OPENAI_PROD_MODEL in .env")
print(f"LLM Model: {model}")

# ── Data Loading (SST-2 sentiment from HuggingFace) ─────────────────────

CACHE_DIR = Path("data/mlfp06/sst2")
CACHE_DIR.mkdir(parents=True, exist_ok=True)
CACHE_FILE = CACHE_DIR / "sst2_200.parquet"

if CACHE_FILE.exists():
    print(f"Loading cached SST-2 from {CACHE_FILE}")
    sst2 = pl.read_parquet(CACHE_FILE)
else:
    print("Downloading stanfordnlp/sst2 from HuggingFace (first run)...")
    from datasets import load_dataset

    ds = load_dataset("stanfordnlp/sst2", split="train")
    ds = ds.shuffle(seed=42).select(range(min(200, len(ds))))

    label_names = {0: "negative", 1: "positive"}
    rows = [
        {
            "text": row["sentence"],
            "label": label_names[row["label"]],
            "label_id": row["label"],
        }
        for row in ds
    ]
    sst2 = pl.DataFrame(rows)
    sst2.write_parquet(CACHE_FILE)
    print(f"Cached {sst2.height} SST-2 rows to {CACHE_FILE}")

eval_docs = sst2.head(20)  # first 20 for evaluation across strategies
print(f"Loaded {sst2.height:,} sentences for classification")
print(f"Label distribution: {dict(sst2['label'].value_counts().iter_rows())}")


# ══════════════════════════════════════════════════════════════════════════
# TASK 1: LLM Foundations Recap
# ══════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 70)
print("TASK 1: LLM Foundations Recap")
print("=" * 70)

# TODO: Print a multi-line explanation covering:
#   - Pre-training objectives: GPT (next-token) vs BERT (masked LM)
#   - Scaling laws (Chinchilla): parameters N, data D, compute C
#   - Key model families: autoregressive, encoder-only, encoder-decoder
#   - RLHF overview: SFT → reward model → PPO
# Hint: use print("""...""") with the theory explained in your own words
____


# ══════════════════════════════════════════════════════════════════════════
# TASK 2: Zero-shot classification with Delegate
# ══════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 70)
print("TASK 2: Zero-Shot Classification")
print("=" * 70)

CATEGORIES = ["positive", "negative"]


# TODO: Define an async function zero_shot_classify(text) that:
#   - Creates a Delegate with model=model and max_llm_cost_usd=0.5
#   - Builds a prompt asking for sentiment classification into CATEGORIES
#   - Streams events from delegate.run(prompt) collecting text and cost
#   - Normalises the label to "positive" or "negative"
#   - Returns (label, cost, elapsed)
# Hint: async for event in delegate.run(prompt): check hasattr(event, "text")
____


# TODO: Define async run_zero_shot(docs) that:
#   - Iterates over docs["text"] and docs["label"]
#   - Calls zero_shot_classify for each, computes accuracy and total cost
#   - Prints sample predictions and summary metrics
#   - Returns list of result dicts with keys: pred, true, correct, cost, elapsed
# Hint: acc = sum(r["correct"] for r in results) / len(results)
____


print("\n=== Zero-Shot Classification ===")
zero_shot_results = asyncio.run(run_zero_shot(eval_docs))

# ── Checkpoint 1 ─────────────────────────────────────────────────────────
assert len(zero_shot_results) > 0, "Task 2: zero-shot should produce results"
assert all(
    r["pred"] in CATEGORIES for r in zero_shot_results
), "Predictions must be valid categories"
print("✓ Checkpoint 1 passed — zero-shot classification complete\n")

# INTERPRETATION: Zero-shot asks the model to classify without examples.
# Performance depends on how well the categories align with the model's
# pre-training distribution.  Strengths: fast, cheap, no example curation.
# Weaknesses: may hallucinate categories, inconsistent formatting.


# ══════════════════════════════════════════════════════════════════════════
# TASK 3: Few-shot with curated example selection
# ══════════════════════════════════════════════════════════════════════════

print("=" * 70)
print("TASK 3: Few-Shot Classification")
print("=" * 70)

FEW_SHOT_EXAMPLES = [
    {
        "text": "an absolute masterpiece of storytelling and visual style.",
        "category": "positive",
    },
    {
        "text": "a tedious and predictable mess from start to finish.",
        "category": "negative",
    },
    {
        "text": "delightfully clever, with performances that elevate every scene.",
        "category": "positive",
    },
    {
        "text": "fails to land a single emotional beat in over two hours.",
        "category": "negative",
    },
]


# TODO: Define async few_shot_classify(text) that:
#   - Creates a Delegate with model=model and max_llm_cost_usd=0.5
#   - Builds a prompt that shows FEW_SHOT_EXAMPLES before the new review
#   - Streams events, collects response and cost
#   - Normalises label to "positive" or "negative"
#   - Returns (label, cost, elapsed)
# Hint: format examples as "Review: ...\nSentiment: ..." before the new text
____


# TODO: Define async run_few_shot(docs) that mirrors run_zero_shot
#   - Iterates, calls few_shot_classify, tracks accuracy/cost
#   - Returns list of result dicts with same keys as zero_shot
# Hint: same structure as run_zero_shot
____


print("\n=== Few-Shot Classification (4 examples) ===")
few_shot_results = asyncio.run(run_few_shot(eval_docs))

# ── Checkpoint 2 ─────────────────────────────────────────────────────────
assert len(few_shot_results) > 0, "Task 3: few-shot should produce results"
print("✓ Checkpoint 2 passed — few-shot classification complete\n")

# INTERPRETATION: Few-shot provides examples of the desired behaviour.
# The model sees input->output pairs and applies the pattern to new inputs.
# Key decisions: how many examples (3-8 typically), selection (diverse,
# representative), ordering (experiment — no universal best order).


# ══════════════════════════════════════════════════════════════════════════
# TASK 4: Chain-of-Thought (CoT) Prompting
# ══════════════════════════════════════════════════════════════════════════

print("=" * 70)
print("TASK 4: Chain-of-Thought Prompting")
print("=" * 70)


# TODO: Define async cot_classify(text) that:
#   - Creates a Delegate with model=model, max_llm_cost_usd=0.5
#   - Builds a prompt with explicit step-by-step reasoning instructions:
#       1. Identify opinion words and their valence
#       2. Assess overall tone
#       3. Consider sarcasm
#       4. State final classification
#   - Streams response, extracts label from last line of reasoning
#   - Returns (label, reasoning, cost, elapsed)
# Hint: check reasoning.lower().split("\n")[-1] for the final label
____


# TODO: Define async run_cot(docs) similar to run_zero_shot but:
#   - Calls cot_classify and stores "reasoning" in each result dict
#   - Prints a reasoning excerpt for the first 3 docs
# Hint: results must contain key "reasoning" for Checkpoint 3
____


print("\n=== Chain-of-Thought Classification ===")
cot_results = asyncio.run(run_cot(eval_docs))

# ── Checkpoint 3 ─────────────────────────────────────────────────────────
assert len(cot_results) > 0, "Task 4: CoT should produce results"
assert all(
    "reasoning" in r for r in cot_results
), "Each CoT result should contain reasoning"
print("✓ Checkpoint 3 passed — chain-of-thought classification complete\n")

# INTERPRETATION: CoT forces step-by-step reasoning before answering.
# Trade-off: more tokens (higher cost, higher latency), but better for
# ambiguous cases.  The reasoning trace is auditable.


# ══════════════════════════════════════════════════════════════════════════
# TASK 5: Zero-Shot CoT ("Let's think step by step")
# ══════════════════════════════════════════════════════════════════════════

print("=" * 70)
print("TASK 5: Zero-Shot CoT")
print("=" * 70)


# TODO: Define async zero_shot_cot_classify(text) that:
#   - Creates a Delegate with model=model, max_llm_cost_usd=0.5
#   - Appends "Let's think step by step." to the classification prompt
#     WITHOUT any explicit reasoning template
#   - Extracts label by counting "positive" vs "negative" occurrences
#   - Returns (label, reasoning, cost, elapsed)
# Hint: lower.count("negative") > lower.count("positive") => "negative"
____


# TODO: Define async run_zero_shot_cot(docs) similar to run_cot
# Hint: return list with pred, true, correct, cost, elapsed
____


print("\n=== Zero-Shot CoT ('Let's think step by step') ===")
zs_cot_results = asyncio.run(run_zero_shot_cot(eval_docs))

# ── Checkpoint 4 ─────────────────────────────────────────────────────────
assert len(zs_cot_results) > 0, "Task 5: zero-shot CoT should produce results"
print("✓ Checkpoint 4 passed — zero-shot CoT classification complete\n")

# INTERPRETATION: "Let's think step by step" is the simplest CoT trigger.
# No manually crafted reasoning template, yet it often improves accuracy
# over plain zero-shot — the model generates its own chain of thought.


# ══════════════════════════════════════════════════════════════════════════
# TASK 6: Self-Consistency (sample multiple CoT paths, majority vote)
# ══════════════════════════════════════════════════════════════════════════

print("=" * 70)
print("TASK 6: Self-Consistency (Majority Vote)")
print("=" * 70)

N_SAMPLES = 3  # Number of independent CoT samples per query


# TODO: Define async self_consistency_classify(text) that:
#   - Calls cot_classify N_SAMPLES times for the same text
#   - Collects all votes (labels)
#   - Uses Counter to find the majority vote
#   - Returns (majority_label, votes_list, total_cost, elapsed)
# Hint: Counter(votes).most_common(1)[0][0] gives the majority label
____


# TODO: Define async run_self_consistency(docs) that:
#   - Operates on docs.head(10) (cost control)
#   - Calls self_consistency_classify per document
#   - Stores "votes" in result dict
#   - Prints the vote breakdown for first 5 docs
# Hint: results must contain key "votes" for Checkpoint 5
____


print(f"\n=== Self-Consistency ({N_SAMPLES} CoT samples, majority vote) ===")
sc_results = asyncio.run(run_self_consistency(eval_docs))

# ── Checkpoint 5 ─────────────────────────────────────────────────────────
assert len(sc_results) > 0, "Task 6: self-consistency should produce results"
assert all(
    "votes" in r for r in sc_results
), "Each result should record individual votes"
print("✓ Checkpoint 5 passed — self-consistency classification complete\n")

# INTERPRETATION: Self-consistency samples multiple INDEPENDENT reasoning
# paths and uses majority vote.  It combats the randomness of any single
# CoT sample.  Cost scales linearly with N_SAMPLES.  Diminishing returns
# beyond N=5 for binary classification, N=7-9 for multi-class.


# ══════════════════════════════════════════════════════════════════════════
# TASK 7: Structured Prompting (explicit JSON output format)
# ══════════════════════════════════════════════════════════════════════════

print("=" * 70)
print("TASK 7: Structured Prompting (JSON Output Spec)")
print("=" * 70)


# TODO: Define async structured_prompt_classify(text) that:
#   - Creates a Delegate with model=model, max_llm_cost_usd=0.5
#   - Prompts the model to return a JSON object with fields:
#       "sentiment": "positive" or "negative"
#       "confidence": float 0.0-1.0
#       "key_words": list of up to 5 sentiment-signal words
#       "reasoning": one-sentence explanation
#   - Parses JSON by finding first { ... } block in response
#   - Returns (parsed_dict, cost, elapsed) — fallback dict on parse failure
# Hint: start = response.index("{"); end = response.rindex("}") + 1
____


# TODO: Define async run_structured_prompt(docs) that:
#   - Operates on docs.head(10)
#   - Extracts "sentiment" from parsed dict as prediction
#   - Stores "parsed" in result dict
#   - Prints formatted JSON for first 3 docs
# Hint: results must contain key "parsed" for Checkpoint 6
____


print("\n=== Structured Prompting (JSON output) ===")
structured_prompt_results = asyncio.run(run_structured_prompt(eval_docs))

# ── Checkpoint 6 ─────────────────────────────────────────────────────────
assert (
    len(structured_prompt_results) > 0
), "Task 7: structured prompting should produce results"
assert all(
    "parsed" in r for r in structured_prompt_results
), "Results should contain parsed JSON"
print("✓ Checkpoint 6 passed — structured prompting complete\n")

# INTERPRETATION: Structured prompting specifies the output format explicitly.
# It's a halfway house between free-form text and Signature enforcement:
# the model usually complies, but there's no type system guarantee.
# JSON parsing can fail on malformed output.  Use Signatures for production.


# ══════════════════════════════════════════════════════════════════════════
# TASK 8: Kaizen Signature for Type-Safe Structured Extraction
# ══════════════════════════════════════════════════════════════════════════

print("=" * 70)
print("TASK 8: Kaizen Signature — Type-Safe Structured Extraction")
print("=" * 70)


# TODO: Define a ReviewExtraction Signature class that:
#   - Has InputField: review_text (str) — the movie review to analyse
#   - Has OutputFields:
#       sentiment (str): "Exactly one of: positive, negative"
#       confidence (float): "Classification confidence 0.0 to 1.0"
#       key_phrases (list[str]): up to 5 phrases signalling sentiment
#       targets (list[str]): aspects evaluated (acting, plot, visuals, pacing)
#       tone (str): emotional tone (enthusiastic, measured, disappointed, etc.)
# Hint: class ReviewExtraction(Signature): with InputField / OutputField
____


# TODO: Define async run_signature_extraction(docs) that:
#   - Creates a SimpleQAAgent with signature=ReviewExtraction, model=model,
#     max_llm_cost_usd=1.0
#   - Operates on docs.head(10)
#   - Calls agent.run(review_text=text[:800]) per review
#   - Prints sentiment, confidence, key_phrases[:3], targets[:3], tone
#     for the first 3 reviews
#   - Returns list of result objects
# Hint: result = await agent.run(review_text=text[:800])
____


print("\n=== Signature Extraction ===")
signature_results = asyncio.run(run_signature_extraction(eval_docs))

# ── Checkpoint 7 ─────────────────────────────────────────────────────────
assert len(signature_results) > 0, "Task 8: Signature extraction should produce results"
sample = signature_results[0]
assert hasattr(sample, "sentiment"), "Result should have 'sentiment' field"
assert hasattr(sample, "confidence"), "Result should have 'confidence' field"
assert 0 <= sample.confidence <= 1, "Confidence should be in [0, 1]"
assert hasattr(sample, "tone"), "Result should have 'tone' field"
print(
    "✓ Checkpoint 7 passed — Signature extraction: "
    f"sentiment='{sample.sentiment}', confidence={sample.confidence:.2f}\n"
)

# INTERPRETATION: Signatures guarantee structure.  Unlike structured
# prompting (Task 7), the output is typed — downstream code accesses
# result.sentiment, not result["sentiment"].  If the model returns
# something malformed, Kaizen retries or raises a typed error.


# ══════════════════════════════════════════════════════════════════════════
# TASK 9: Quantitative Comparison Across All Strategies
# ══════════════════════════════════════════════════════════════════════════

print("=" * 70)
print("TASK 9: Quantitative Strategy Comparison")
print("=" * 70)


# TODO: Define compute_metrics(results, name) that:
#   - Computes accuracy as sum(correct) / n
#   - Sums total_cost from results
#   - Computes avg_latency_s
#   - Returns dict: strategy, n, accuracy, total_cost, avg_latency_s
# Hint: use r.get("cost", 0.0) and r.get("elapsed", 0.0) for safety
____


# TODO: Build a strategy_metrics list by calling compute_metrics on:
#   zero_shot_results, few_shot_results, cot_results,
#   zs_cot_results, sc_results, structured_prompt_results
# Then build comparison_df = pl.DataFrame(strategy_metrics) and print it
# Hint: strategy names: "Zero-Shot", "Few-Shot (4 examples)", etc.
____

print("\nKey insights:")
print("  - Zero-shot: cheapest, fastest, but may hallucinate categories")
print("  - Few-shot: better consistency via examples; cost of longer prompt")
print("  - CoT: forces reasoning; best for ambiguous cases; highest latency")
print("  - Zero-shot CoT: lightweight reasoning trigger; good cost/quality ratio")
print(
    f"  - Self-consistency: majority vote over N={N_SAMPLES} paths; highest accuracy, N× cost"
)
print("  - Structured JSON: format control, but fragile (parsing failures)")
print("  - Signature: type-safe, retries on failure — production standard")

# ── Checkpoint 8 ─────────────────────────────────────────────────────────
assert comparison_df.height >= 6, "Task 9: comparison should cover all strategies"
print("\n✓ Checkpoint 8 passed — strategy comparison table generated\n")

# INTERPRETATION: The comparison reveals the cost-quality Pareto frontier.
# For production: Signature (type safety) + few-shot (consistency).
# For research: self-consistency (highest accuracy) at N× cost.
# For speed: zero-shot (cheapest, fastest, good enough for simple tasks).


# ══════════════════════════════════════════════════════════════════════════
# TASK 10: Inference Optimisations
# ══════════════════════════════════════════════════════════════════════════

print("=" * 70)
print("TASK 10: Inference Optimisations")
print("=" * 70)

# TODO: Print a multi-line explanation covering:
#   - KV-Cache: what is cached, why it reduces compute from O(n^2) to O(n)
#   - Speculative Decoding: draft model + target model verification, speedup
#   - Continuous Batching: vs traditional batching, GPU utilisation
#   - Flash Attention: kernel fusion, memory O(n) vs O(n^2)
# Hint: use print("""...""") with your own clear explanations
____

# ── Checkpoint 9 ─────────────────────────────────────────────────────────
print("✓ Checkpoint 9 passed — inference optimisations explained\n")

# INTERPRETATION: These optimisations are invisible to the prompt engineer
# but critical to production deployment:
# KV-cache:           every serving framework uses it; 10× throughput
# Speculative decode: 2-3× latency reduction with zero quality loss
# Continuous batching: keeps GPU utilisation >90% in production
# Flash attention:     enables long context (128K+) on consumer hardware


# ══════════════════════════════════════════════════════════════════════════
# REFLECTION
# ══════════════════════════════════════════════════════════════════════════
print("═" * 70)
print("  WHAT YOU'VE MASTERED")
print("═" * 70)
print(
    """
  ✓ LLM foundations: pre-training (next-token, masked LM), scaling laws,
    RLHF pipeline (SFT → reward model → PPO)
  ✓ Zero-shot: task description only; cheapest, fastest, least consistent
  ✓ Few-shot: examples steer the model; better consistency and formatting
  ✓ Chain-of-thought: "Think step by step" elicits intermediate reasoning
  ✓ Zero-shot CoT: append trigger phrase; lightweight reasoning at low cost
  ✓ Self-consistency: majority vote over N independent CoT paths; highest
    accuracy for ambiguous tasks; N× cost
  ✓ Structured prompting: JSON output spec; convenient but fragile parsing
  ✓ Kaizen Signature: typed, validated, retryable — production standard
  ✓ Kaizen Delegate: streaming LLM calls with cost budget enforcement
  ✓ Inference optimisations: KV-cache, speculative decoding, continuous
    batching, flash attention

  Prompting cost vs quality hierarchy:
    Zero-shot (cheapest) < Few-shot < Zero-shot CoT < CoT
    < Self-consistency (N×) < Signature (type-safe, production)

  NEXT: Exercise 2 (Fine-Tuning) goes beyond prompting.  Instead of
  giving the model better instructions, you change the model weights
  using low-rank matrix decomposition (LoRA) and adapter layers — both
  implemented from scratch.
"""
)
