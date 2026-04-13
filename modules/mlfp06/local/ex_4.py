# Copyright 2026 Terrene Foundation
# SPDX-License-Identifier: Apache-2.0
"""
# ════════════════════════════════════════════════════════════════════════
# MLFP06 — Exercise 4: RAG Systems — Chunking, Retrieval, and Evaluation
# ════════════════════════════════════════════════════════════════════════
#
# WHAT YOU'LL LEARN:
#   After completing this exercise, you will be able to:
#   - Implement 4 chunking strategies (fixed, sentence, paragraph, semantic)
#     and explain the trade-offs of each
#   - Build dense retrieval (cosine similarity on embeddings), sparse
#     retrieval (BM25), and hybrid retrieval (RRF fusion)
#   - Implement a cross-encoder re-ranker for precision improvement
#   - Evaluate RAG quality using RAGAS metrics (faithfulness, answer
#     relevance, context relevance, context recall)
#   - Implement HyDE (Hypothetical Document Embeddings) and measure its
#     retrieval improvement
#   - Build a complete end-to-end RAG pipeline with Kaizen Delegate
#
# PREREQUISITES:
#   Exercise 1 (Delegate, prompt engineering).  M4.6 (NLP, embeddings,
#   BM25 keyword search).  Understanding that LLMs have a knowledge
#   cutoff and cannot access documents unless injected into the prompt.
#
# ESTIMATED TIME: ~180 min
#
# TASKS:
#    1. Load RAG corpus (real Q&A dataset from HuggingFace)
#    2. Implement 4 chunking strategies
#    3. Generate dense embeddings via Delegate
#    4. Implement BM25 sparse retrieval from scratch
#    5. Build hybrid retrieval (dense + sparse + RRF)
#    6. Implement cross-encoder re-ranking
#    7. RAGAS evaluation framework
#    8. Implement HyDE (Hypothetical Document Embeddings)
#    9. Full RAG pipeline: retrieve -> rerank -> generate
#   10. Compare retrieval strategies quantitatively
#
# DATASET: neural-bridge/rag-dataset-12000 (HuggingFace)
#   12,000 real RAG question-answer-context triples.  Each row provides
#   the source context, a real question, and a ground-truth answer.
#   We use contexts as the retrieval corpus and questions as test queries.
#
# ════════════════════════════════════════════════════════════════════════
"""
from __future__ import annotations

import asyncio
import math
import os
import re
from collections import Counter
from pathlib import Path

import polars as pl

from kaizen_agents import Delegate

from shared.kailash_helpers import setup_environment

setup_environment()

model = os.environ.get("DEFAULT_LLM_MODEL", os.environ.get("OPENAI_PROD_MODEL"))
if not model:
    raise EnvironmentError("Set DEFAULT_LLM_MODEL or OPENAI_PROD_MODEL in .env")
print(f"LLM Model: {model}")

# ── Data Loading ─────────────────────────────────────────────────────────

CACHE_DIR = Path("data/mlfp06/rag")
CACHE_DIR.mkdir(parents=True, exist_ok=True)
CACHE_FILE = CACHE_DIR / "rag_corpus_1k.parquet"

if CACHE_FILE.exists():
    print(f"Loading cached RAG corpus from {CACHE_FILE}")
    corpus = pl.read_parquet(CACHE_FILE)
else:
    print("Downloading neural-bridge/rag-dataset-12000 from HuggingFace...")
    from datasets import load_dataset

    ds = load_dataset("neural-bridge/rag-dataset-12000", split="train")
    ds = ds.shuffle(seed=42).select(range(min(1000, len(ds))))
    rows = [
        {
            "section": f"doc_{i:04d}",
            "text": row["context"],
            "question": row["question"],
            "answer": row["answer"],
        }
        for i, row in enumerate(ds)
    ]
    corpus = pl.DataFrame(rows)
    corpus.write_parquet(CACHE_FILE)
    print(f"Cached {corpus.height} documents to {CACHE_FILE}")

print(f"Loaded {corpus.height:,} documents")
print(f"Columns: {corpus.columns}")

# Separate corpus texts and evaluation questions
doc_texts = corpus["text"].to_list()
eval_questions = corpus.head(20)["question"].to_list()
eval_answers = corpus.head(20)["answer"].to_list()


# ══════════════════════════════════════════════════════════════════════════
# TASK 1: Load RAG Corpus
# ══════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 70)
print("TASK 1: RAG Corpus Loaded")
print("=" * 70)

# ── Checkpoint 1 ─────────────────────────────────────────────────────────
assert corpus.height > 0, "Task 1: corpus should not be empty"
assert "text" in corpus.columns, "Corpus needs 'text' column"
assert "question" in corpus.columns, "Corpus needs 'question' column"
print(f"✓ Checkpoint 1 passed — {corpus.height} documents loaded\n")


# ══════════════════════════════════════════════════════════════════════════
# TASK 2: Implement 4 Chunking Strategies
# ══════════════════════════════════════════════════════════════════════════

print("=" * 70)
print("TASK 2: Chunking Strategies")
print("=" * 70)


# TODO: Implement fixed_chunk(text, chunk_size=512, overlap=50) that:
#   - Splits text into chunks of chunk_size characters with overlap
#   - Returns list of chunk strings
# Hint: for i in range(0, len(text), chunk_size - overlap): chunks.append(text[i:i+chunk_size])
____


# TODO: Implement sentence_chunk(text, max_sentences=5) that:
#   - Splits on sentence boundaries (". ", "! ", "? ")
#   - Groups max_sentences sentences per chunk
#   - Returns list of chunk strings
# Hint: re.split(r'(?<=[.!?])\s+', text) splits at sentence boundaries
____


# TODO: Implement paragraph_chunk(text) that:
#   - Splits on blank lines ("\n\n")
#   - Strips whitespace, discards empty chunks
#   - Returns list of paragraph strings
# Hint: [p.strip() for p in text.split("\n\n") if p.strip()]
____


# TODO: Implement semantic_chunk(text, target_size=400, threshold=0.3) that:
#   - Computes running cosine similarity between consecutive sentences
#   - Starts a new chunk when similarity drops below threshold
#   - Returns list of chunk strings
#   Note: use simple word-overlap Jaccard similarity (no embeddings needed here)
# Hint: jaccard(a, b) = |intersection(words_a, words_b)| / |union(words_a, words_b)|
____


# Demonstrate on a sample document
sample_doc = doc_texts[0]
fixed_chunks = fixed_chunk(sample_doc)
sentence_chunks = sentence_chunk(sample_doc)
para_chunks = paragraph_chunk(sample_doc)
semantic_chunks = semantic_chunk(sample_doc)

print(f"Document length: {len(sample_doc)} chars")
print(f"Fixed chunking:    {len(fixed_chunks)} chunks")
print(f"Sentence chunking: {len(sentence_chunks)} chunks")
print(f"Paragraph chunking:{len(para_chunks)} chunks")
print(f"Semantic chunking: {len(semantic_chunks)} chunks")

# ── Checkpoint 2 ─────────────────────────────────────────────────────────
assert len(fixed_chunks) > 0, "Task 2: fixed chunking should produce chunks"
assert len(sentence_chunks) > 0, "Task 2: sentence chunking should produce chunks"
assert len(para_chunks) > 0, "Task 2: paragraph chunking should produce chunks"
assert len(semantic_chunks) > 0, "Task 2: semantic chunking should produce chunks"
print("✓ Checkpoint 2 passed — 4 chunking strategies implemented\n")

# INTERPRETATION: Fixed chunking is fast but splits mid-sentence.
# Sentence/paragraph chunking preserves semantic units.
# Semantic chunking groups similar content, ideal for coherent retrieval.


# ══════════════════════════════════════════════════════════════════════════
# TASK 3: Generate Dense Embeddings via Delegate
# ══════════════════════════════════════════════════════════════════════════

print("=" * 70)
print("TASK 3: Dense Embeddings")
print("=" * 70)


# TODO: Implement async embed_texts(texts, batch_size=10) that:
#   - Creates a Delegate with model=model, max_llm_cost_usd=2.0
#   - For each text, asks the LLM to produce a dense representation as a
#     list of 16 floats (simulated embeddings for cost control)
#   - Actually: use a deterministic hash-based embedding for speed
#     (map each word to a position in a 64-dim vector, normalise)
#   - Returns list of embedding vectors (list of floats, length 64)
# Hint: import hashlib; hash each word to a bucket, accumulate, normalise
____


# TODO: Implement cosine_similarity(vec_a, vec_b) returning a float in [-1, 1]
# Hint: dot(a,b) / (norm(a) * norm(b)); handle zero-norm case
____


# TODO: Implement dense_retrieve(query, doc_embeddings, doc_texts, top_k=5) that:
#   - Embeds the query (synchronously, use a simple hash embedding)
#   - Computes cosine similarity to all doc_embeddings
#   - Returns top_k (doc_text, score) pairs
# Hint: sorted by score descending, take [:top_k]
____


# Build embeddings for first 100 docs (cost control)
corpus_subset = doc_texts[:100]

print("Building hash embeddings for 100 documents...")
doc_embeddings = [embed_texts([t])[0] for t in corpus_subset]
print(f"Embedding shape: {len(doc_embeddings[0])} dims")

test_query = eval_questions[0]
dense_results = dense_retrieve(test_query, doc_embeddings, corpus_subset)
print(f"\nQuery: {test_query[:80]}...")
print(f"Top-3 dense results:")
for i, (text, score) in enumerate(dense_results[:3]):
    print(f"  {i+1}. score={score:.3f}: {text[:100]}...")

# ── Checkpoint 3 ─────────────────────────────────────────────────────────
assert len(doc_embeddings) == 100, "Task 3: should have 100 embeddings"
assert len(doc_embeddings[0]) == 64, "Task 3: embeddings should be 64-dim"
assert len(dense_results) == 5, "Task 3: dense_retrieve should return top_k=5"
print("✓ Checkpoint 3 passed — dense embeddings and retrieval working\n")

# INTERPRETATION: Dense retrieval captures semantic similarity via vector
# space proximity.  Hash-based embeddings are a teaching approximation;
# production uses transformer encoders (E5, BGE, OpenAI text-embedding-3).


# ══════════════════════════════════════════════════════════════════════════
# TASK 4: BM25 Sparse Retrieval from Scratch
# ══════════════════════════════════════════════════════════════════════════

print("=" * 70)
print("TASK 4: BM25 Sparse Retrieval")
print("=" * 70)

print(
    """
BM25 formula:
  score(q, d) = sum_{t in q} IDF(t) * (tf(t,d) * (k1+1)) / (tf(t,d) + k1*(1-b+b*|d|/avgdl))

  IDF(t) = log((N - df(t) + 0.5) / (df(t) + 0.5) + 1)
  k1 = 1.5 (term frequency saturation)
  b  = 0.75 (document length normalisation)
  N  = total documents
  df(t) = documents containing term t
  |d| / avgdl = document length relative to corpus average
"""
)


# TODO: Implement class BM25Index with:
#   - __init__(self, docs, k1=1.5, b=0.75):
#       Tokenise all docs (lowercase, split on non-alphanumeric)
#       Compute: df (term -> doc count), idf, avgdl
#   - score(self, query, doc_idx) -> float: compute BM25 score
#   - retrieve(self, query, top_k=5) -> list[(doc_text, score)]:
#       Score all docs, return top_k by descending score
# Hint: IDF(t) = log((N - df[t] + 0.5) / (df[t] + 0.5) + 1)
____


bm25 = BM25Index(corpus_subset)
sparse_results = bm25.retrieve(test_query)
print(f"\nQuery: {test_query[:80]}...")
print(f"Top-3 BM25 results:")
for i, (text, score) in enumerate(sparse_results[:3]):
    print(f"  {i+1}. score={score:.3f}: {text[:100]}...")

# ── Checkpoint 4 ─────────────────────────────────────────────────────────
assert isinstance(bm25, BM25Index), "Task 4: BM25Index should be created"
assert len(sparse_results) == 5, "Task 4: BM25 should return top_k=5 results"
print("✓ Checkpoint 4 passed — BM25 from scratch implemented\n")

# INTERPRETATION: BM25 is keyword-based: it rewards exact term matches,
# normalised by document length.  Fast and effective for fact retrieval.
# Weakness: misses paraphrases and synonyms that dense retrieval handles.


# ══════════════════════════════════════════════════════════════════════════
# TASK 5: Hybrid Retrieval (Dense + Sparse + RRF)
# ══════════════════════════════════════════════════════════════════════════

print("=" * 70)
print("TASK 5: Hybrid Retrieval with RRF")
print("=" * 70)


# TODO: Implement reciprocal_rank_fusion(rankings, k=60) that:
#   - Takes a list of ranked doc lists (each list = [(doc_text, score)])
#   - Computes RRF score for each doc: sum(1 / (k + rank_i))
#   - Returns unified ranked list of (doc_text, rrf_score)
# Hint: for each ranker, assign rank 1,2,3,... to its results
____


# TODO: Implement hybrid_retrieve(query, doc_embeddings, bm25_index, doc_texts,
#                                   top_k=5, alpha=0.5) that:
#   - Gets dense_results = dense_retrieve(query, doc_embeddings, doc_texts, top_k*2)
#   - Gets sparse_results = bm25_index.retrieve(query, top_k*2)
#   - Fuses with RRF and returns top_k results
# Hint: pass [dense_results, sparse_results] to reciprocal_rank_fusion
____


hybrid_results = hybrid_retrieve(test_query, doc_embeddings, bm25, corpus_subset)
print(f"\nQuery: {test_query[:80]}...")
print(f"Top-3 Hybrid (RRF) results:")
for i, (text, score) in enumerate(hybrid_results[:3]):
    print(f"  {i+1}. rrf_score={score:.4f}: {text[:100]}...")

# ── Checkpoint 5 ─────────────────────────────────────────────────────────
assert len(hybrid_results) == 5, "Task 5: hybrid should return top_k=5 results"
print("✓ Checkpoint 5 passed — hybrid retrieval with RRF implemented\n")

# INTERPRETATION: RRF combines rankings from multiple retrievers without
# needing to normalise scores (which are on incompatible scales).
# A document appearing at rank 1 in both dense and sparse scores highest.


# ══════════════════════════════════════════════════════════════════════════
# TASK 6: Cross-Encoder Re-ranking
# ══════════════════════════════════════════════════════════════════════════

print("=" * 70)
print("TASK 6: Cross-Encoder Re-ranking")
print("=" * 70)


# TODO: Define async cross_encode_rerank(query, candidates, top_k=3) that:
#   - Creates a Delegate with model=model, max_llm_cost_usd=1.0
#   - For each candidate, asks the LLM to score relevance to query (0-10)
#   - Returns top_k (doc_text, score) pairs sorted by relevance score
# Hint: prompt = f"Rate relevance of this passage to the query 0-10.\nQuery: {query}\nPassage: {passage[:500]}\nScore:"
____


print("\n=== Cross-Encoder Re-ranking ===")
reranked = asyncio.run(cross_encode_rerank(test_query, hybrid_results))
print(f"After re-ranking (top 3):")
for i, (text, score) in enumerate(reranked):
    print(f"  {i+1}. score={score:.1f}: {text[:100]}...")

# ── Checkpoint 6 ─────────────────────────────────────────────────────────
assert len(reranked) <= 3, "Task 6: re-ranker should return at most top_k=3"
print("✓ Checkpoint 6 passed — cross-encoder re-ranking complete\n")

# INTERPRETATION: Cross-encoders jointly process query+passage, giving
# more precise relevance scores than bi-encoder dot products.  Cost:
# one LLM call per candidate.  Use bi-encoder (fast) then cross-encoder
# (precise) in a two-stage pipeline.


# ══════════════════════════════════════════════════════════════════════════
# TASK 7: RAGAS Evaluation Framework
# ══════════════════════════════════════════════════════════════════════════

print("=" * 70)
print("TASK 7: RAGAS Evaluation")
print("=" * 70)


# TODO: Define async ragas_evaluate(question, context, answer, ground_truth,
#                                    budget=0.5) that:
#   Uses a Delegate to compute 4 RAGAS metrics (0-1 each):
#   - faithfulness: is the answer supported by the context? (no hallucination)
#   - answer_relevance: does the answer address the question?
#   - context_relevance: does the context contain info to answer the question?
#   - context_recall: does the context cover all info in ground_truth?
#   Ask the LLM for all 4 scores as JSON in one call
#   Returns dict with the 4 scores + overall (mean of all 4)
# Hint: prompt for JSON {"faithfulness":..., "answer_relevance":..., ...}
____


# Evaluate on first 5 questions
print("\n=== RAGAS Evaluation (5 questions) ===")
ragas_scores = []
for i in range(5):
    q = eval_questions[i]
    gt = eval_answers[i]
    # Use top hybrid result as context
    ctx_results = hybrid_retrieve(q, doc_embeddings, bm25, corpus_subset, top_k=1)
    ctx = ctx_results[0][0] if ctx_results else ""
    # Simulate an answer
    ans = f"Based on the context: {ctx[:200]}..."
    score = asyncio.run(ragas_evaluate(q, ctx, ans, gt))
    ragas_scores.append(score)
    print(
        f"  Q{i+1}: faithfulness={score['faithfulness']:.2f}, "
        f"answer_rel={score['answer_relevance']:.2f}, "
        f"overall={score['overall']:.2f}"
    )

avg_ragas = {
    k: sum(s[k] for s in ragas_scores) / len(ragas_scores) for k in ragas_scores[0]
}
print(f"\nAverage RAGAS: {avg_ragas}")

# ── Checkpoint 7 ─────────────────────────────────────────────────────────
assert len(ragas_scores) == 5, "Task 7: should evaluate 5 questions"
assert all("overall" in s for s in ragas_scores), "Each score needs 'overall'"
print("✓ Checkpoint 7 passed — RAGAS evaluation complete\n")

# INTERPRETATION: RAGAS decomposes RAG quality into 4 axes.
# Low faithfulness = the model hallucinated (answer not in context).
# Low context_relevance = retrieval is noisy (retrieved wrong docs).
# Fix retrieval first (context_relevance), then generation (faithfulness).


# ══════════════════════════════════════════════════════════════════════════
# TASK 8: HyDE (Hypothetical Document Embeddings)
# ══════════════════════════════════════════════════════════════════════════

print("=" * 70)
print("TASK 8: HyDE — Hypothetical Document Embeddings")
print("=" * 70)

print(
    """
HyDE (Gao et al., 2022):
  Problem: query and document are in different embedding spaces.
    Query: short, interrogative
    Document: long, declarative — different vocabulary and structure.
  Solution:
    1. Generate a HYPOTHETICAL document that would answer the query
       (using the LLM — no retrieval needed yet)
    2. Embed the hypothetical document (same space as real docs)
    3. Retrieve real docs similar to the hypothetical document
  Why it works: hypothetical doc uses the vocabulary and style of
  real documents, closing the query-document embedding gap.
"""
)


# TODO: Define async hyde_retrieve(query, doc_embeddings, doc_texts,
#                                   top_k=5) that:
#   1. Creates a Delegate and generates a hypothetical answer to the query
#      (50-100 words, declarative, as if it came from a Wikipedia article)
#   2. Embeds the hypothetical answer using embed_texts([hyp_doc])[0]
#   3. Computes cosine similarity to all doc_embeddings
#   4. Returns top_k (doc_text, score) pairs
# Hint: the hypothetical doc should NOT be retrieved — just used for embedding
____


print("\n=== HyDE vs Standard Dense Retrieval ===")
hyde_results = asyncio.run(hyde_retrieve(test_query, doc_embeddings, corpus_subset))
print(f"Query: {test_query[:80]}...")
print(f"HyDE top-3:")
for i, (text, score) in enumerate(hyde_results[:3]):
    print(f"  {i+1}. score={score:.3f}: {text[:100]}...")

# ── Checkpoint 8 ─────────────────────────────────────────────────────────
assert len(hyde_results) == 5, "Task 8: HyDE should return top_k=5 results"
print("✓ Checkpoint 8 passed — HyDE retrieval implemented\n")

# INTERPRETATION: HyDE typically improves recall for factual queries
# where the user's question phrasing differs from document vocabulary.
# Downside: one extra LLM call per query (the hypothetical generation).


# ══════════════════════════════════════════════════════════════════════════
# TASK 9: Full RAG Pipeline (retrieve -> rerank -> generate)
# ══════════════════════════════════════════════════════════════════════════

print("=" * 70)
print("TASK 9: Full End-to-End RAG Pipeline")
print("=" * 70)


# TODO: Define async rag_pipeline(question, doc_embeddings, bm25_index,
#                                  doc_texts, top_k=5) that:
#   1. Retrieve: hybrid_retrieve(question, doc_embeddings, bm25_index, doc_texts, top_k)
#   2. Rerank: cross_encode_rerank(question, retrieved_docs, top_k=3)
#   3. Build context: join top-3 reranked passages with "\n\n---\n\n"
#   4. Generate: Delegate with prompt:
#       "Answer the question using ONLY the provided context. If not in context, say so.
#        Context: {context}
#        Question: {question}
#        Answer:"
#   5. Returns dict: question, context, answer, sources (list of top-3 texts)
# Hint: max_context_chars = 2000 to control token cost
____


print("\n=== Full RAG Pipeline ===")
rag_result = asyncio.run(rag_pipeline(test_query, doc_embeddings, bm25, corpus_subset))
print(f"Question: {rag_result['question'][:80]}...")
print(f"Answer:   {rag_result['answer'][:300]}...")
print(f"Sources:  {len(rag_result['sources'])} passages used")

# ── Checkpoint 9 ─────────────────────────────────────────────────────────
assert "answer" in rag_result, "Task 9: pipeline should produce an answer"
assert "sources" in rag_result, "Task 9: pipeline should return sources"
assert len(rag_result["sources"]) > 0, "Task 9: should retrieve at least one source"
print("✓ Checkpoint 9 passed — full RAG pipeline working\n")


# ══════════════════════════════════════════════════════════════════════════
# TASK 10: Compare Retrieval Strategies Quantitatively
# ══════════════════════════════════════════════════════════════════════════

print("=" * 70)
print("TASK 10: Retrieval Strategy Comparison")
print("=" * 70)


# TODO: Define evaluate_retrieval(strategy_name, retrieve_fn, questions,
#                                   answers, doc_texts, top_k=5) that:
#   - For each question, retrieves top_k documents using retrieve_fn
#   - Checks if the ground-truth answer words appear in the retrieved texts
#   - Computes recall@k = fraction of questions where answer found in top_k
#   - Returns dict: strategy, recall_at_k, avg_retrieved_chars
# Hint: any(answer.lower()[:30] in " ".join(texts).lower()) counts as a hit
____


# Evaluate all strategies on 10 questions
strategies = [
    ("Dense", lambda q: dense_retrieve(q, doc_embeddings, corpus_subset)),
    ("BM25", lambda q: bm25.retrieve(q)),
    ("Hybrid-RRF", lambda q: hybrid_retrieve(q, doc_embeddings, bm25, corpus_subset)),
]

strategy_results = []
for name, fn in strategies:
    result = evaluate_retrieval(
        name, fn, eval_questions[:10], eval_answers[:10], corpus_subset
    )
    strategy_results.append(result)
    print(f"  {name}: recall@5={result['recall_at_k']:.1%}")

comparison_df = pl.DataFrame(strategy_results)
print(f"\n{comparison_df}")

# ── Checkpoint 10 ─────────────────────────────────────────────────────────
assert comparison_df.height == 3, "Task 10: should compare 3 strategies"
print("\n✓ Checkpoint 10 passed — retrieval strategy comparison complete\n")

# INTERPRETATION: Hybrid-RRF typically beats both dense and sparse alone.
# Dense handles paraphrases; BM25 handles exact keywords.  Together they
# cover each other's weaknesses.


# ══════════════════════════════════════════════════════════════════════════
# REFLECTION
# ══════════════════════════════════════════════════════════════════════════
print("═" * 70)
print("  WHAT YOU'VE MASTERED")
print("═" * 70)
print(
    """
  ✓ 4 chunking strategies: fixed, sentence, paragraph, semantic
  ✓ Dense retrieval: hash embeddings, cosine similarity, top-k search
  ✓ BM25 from scratch: IDF, term frequency normalisation, k1/b parameters
  ✓ Hybrid retrieval: RRF fusion of dense + sparse rankings
  ✓ Cross-encoder re-ranking: LLM-scored precision improvement
  ✓ RAGAS metrics: faithfulness, answer relevance, context relevance, recall
  ✓ HyDE: hypothetical document generation to bridge query-doc gap
  ✓ Full RAG pipeline: retrieve → rerank → generate → evaluate
  ✓ Strategy comparison: recall@k across dense, BM25, hybrid

  RAG design principles:
    Start with hybrid retrieval (best recall)
    Add re-ranking for precision (top-3 after top-10 retrieval)
    Measure faithfulness to catch hallucinations
    Use HyDE when queries and documents use different vocabulary

  NEXT: Exercise 5 (Agents) gives the LLM the ability to ACT — calling
  tools, reasoning in a loop, and deciding when it has enough information.
"""
)
