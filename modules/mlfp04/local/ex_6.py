# Copyright 2026 Terrene Foundation
# SPDX-License-Identifier: Apache-2.0
"""
# ════════════════════════════════════════════════════════════════════════
# MLFP04 — Exercise 6: NLP — Text to Topics
# ════════════════════════════════════════════════════════════════════════
#
# WHAT YOU'LL LEARN:
#   After completing this exercise, you will be able to:
#   - Derive TF-IDF from scratch and explain why it down-weights common words
#   - Implement BM25 scoring and compare with TF-IDF
#   - Apply NMF for topic extraction from a TF-IDF matrix
#   - Apply LDA for probabilistic topic modelling
#   - Build a BERTopic pipeline (embeddings + UMAP + HDBSCAN + c-TF-IDF)
#   - Evaluate topic quality using NPMI coherence metrics
#   - Use word embeddings (Word2Vec, GloVe, FastText) as features
#   - Perform basic sentiment analysis on customer reviews
#   - Track topic distribution across categories
#
# PREREQUISITES:
#   - MLFP04 Exercise 3 (UMAP — used inside BERTopic)
#   - MLFP04 Exercise 1 (HDBSCAN — used inside BERTopic)
#
# ESTIMATED TIME: ~150-180 min
#
# TASKS:
#   1.  TF-IDF warmup: bag-of-words, term frequency, IDF derivation
#   2.  BM25 scoring: improved TF-IDF with saturation and length normalisation
#   3.  NMF topic extraction from TF-IDF matrix
#   4.  Load and preprocess text corpus
#   5.  LDA topic modelling (probabilistic, mixed membership)
#   6.  BERTopic model (UMAP + HDBSCAN + c-TF-IDF)
#   7.  Topic coherence evaluation (NPMI)
#   8.  Word embeddings as features (Word2Vec concept)
#   9.  Sentiment analysis on customer reviews
#   10. Topic distribution and visualisation
#
# DATASET: Document corpus (from MLFP03)
#   Goal: discover latent topics without labelled categories
#   Business context: content tagging, media monitoring, trend detection
#
# ════════════════════════════════════════════════════════════════════════
"""
from __future__ import annotations

import numpy as np
import polars as pl
from collections import Counter

from kailash_ml import ModelVisualizer

from shared import MLFPDataLoader

try:
    from bertopic import BERTopic
    from sentence_transformers import SentenceTransformer
except ImportError:
    BERTopic = None
    SentenceTransformer = None


# ══════════════════════════════════════════════════════════════════════
# TASK 1: TF-IDF — from bag-of-words to term weighting
# ══════════════════════════════════════════════════════════════════════
# TF-IDF = Term Frequency x Inverse Document Frequency
#   TF(t, d) = count(t in d) / count(all words in d)
#   IDF(t) = log(N / df(t))   where df(t) = docs containing t
#   TF-IDF(t, d) = TF(t, d) x IDF(t)

from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer

toy_corpus = [
    "Singapore economy grew strongly in 2024",
    "Singapore property market shows resilience",
    "MAS tightens monetary policy amid global uncertainty",
    "Property developers report strong demand",
    "Singapore government announces new housing measures",
    "Global markets react to central bank decisions",
    "Technology sector leads Singapore stock exchange",
    "Housing prices continue upward trend in Singapore",
]

# Step 1: Bag of words
bow_vectorizer = CountVectorizer(stop_words="english")
X_bow = bow_vectorizer.fit_transform(toy_corpus)
bow_vocab = bow_vectorizer.get_feature_names_out()

print("=== Bag-of-Words Warmup ===")
print(f"Vocabulary size: {len(bow_vocab)}")
print(f"Matrix shape: {X_bow.shape} (docs x vocab)")
print(f"\nDocument 0 non-zero terms:")
doc0 = X_bow[0].toarray()[0]
for term, count in zip(bow_vocab, doc0):
    if count > 0:
        print(f"  '{term}': {int(count)}")

# TODO: Create TfidfVectorizer with stop_words="english" and norm="l2", then fit_transform
tfidf_vectorizer = ____  # Hint: TfidfVectorizer(stop_words="english", norm="l2")
X_tfidf = ____  # Hint: tfidf_vectorizer.fit_transform(toy_corpus)
tfidf_vocab = tfidf_vectorizer.get_feature_names_out()

print(f"\n=== TF-IDF Weights (Document 0 vs Document 1) ===")
print(f"{'Term':<20} {'Doc0 TF-IDF':>14} {'Doc1 TF-IDF':>14} {'IDF':>10}")
print("─" * 62)
idf_values = tfidf_vectorizer.idf_
doc0_tfidf = X_tfidf[0].toarray()[0]
doc1_tfidf = X_tfidf[1].toarray()[0]
for term, idf, t0, t1 in sorted(
    zip(tfidf_vocab, idf_values, doc0_tfidf, doc1_tfidf),
    key=lambda x: -abs(x[2] + x[3]),
)[:12]:
    print(f"  {term:<20} {t0:>14.4f} {t1:>14.4f} {idf:>10.4f}")

print("\nTF-IDF insight:")
print("  'singapore' appears in 4/8 docs -> lower IDF -> penalised")
print("  'monetary' appears in 1/8 docs  -> higher IDF -> rewarded")

# Manual TF-IDF verification
manual_tf = {}
for i, doc in enumerate(toy_corpus):
    words = doc.lower().split()
    words = [w for w in words if w not in {"in", "to", "the", "of", "and", "a"}]
    for w in words:
        manual_tf[(i, w)] = manual_tf.get((i, w), 0) + 1

# ── Checkpoint 1 ─────────────────────────────────────────────────────
idf_dict = dict(zip(tfidf_vocab, idf_values))
if "singapore" in idf_dict and "monetary" in idf_dict:
    assert (
        idf_dict["singapore"] < idf_dict["monetary"]
    ), "'singapore' (common) should have lower IDF than 'monetary' (rare)"
row_norms = np.sqrt(np.asarray(X_tfidf.multiply(X_tfidf).sum(axis=1)).flatten())
assert all(
    abs(n - 1.0) < 0.01 for n in row_norms if n > 0
), "TF-IDF rows should be L2-normalised"
# INTERPRETATION: IDF = log(N / df(t)). High IDF = rare term = discriminative.
# Common words get low IDF weights and become less influential in document
# similarity computations.
print(
    "\n✓ Checkpoint 1 passed — TF-IDF IDF values correctly rank rare vs common terms\n"
)


# ══════════════════════════════════════════════════════════════════════
# TASK 2: BM25 scoring
# ══════════════════════════════════════════════════════════════════════
# BM25 improves TF-IDF with two key modifications:
#   1. Term frequency saturation: TF contribution plateaus (diminishing returns)
#   2. Document length normalisation: longer docs don't automatically score higher
#
# BM25(t, d) = IDF(t) * (tf(t,d) * (k1 + 1)) / (tf(t,d) + k1 * (1 - b + b * |d|/avgdl))
#   k1 = 1.2 (controls saturation), b = 0.75 (controls length normalisation)

print("=== BM25 Scoring ===")


def bm25_score(
    tf: float,
    df: int,
    N: int,
    dl: int,
    avgdl: float,
    k1: float = 1.2,
    b: float = 0.75,
) -> float:
    """Compute BM25 score for a single term in a single document."""
    # TODO: Compute IDF component = log((N - df + 0.5) / (df + 0.5) + 1)
    idf = ____  # Hint: np.log((N - df + 0.5) / (df + 0.5) + 1)
    # TODO: Compute TF component with saturation and length normalisation
    tf_component = ____  # Hint: (tf * (k1 + 1)) / (tf + k1 * (1 - b + b * dl / avgdl))
    return idf * tf_component


# Compare TF-IDF vs BM25 on toy corpus
doc_lengths = [len(doc.split()) for doc in toy_corpus]
avgdl = np.mean(doc_lengths)
N_docs = len(toy_corpus)

print(f"Document lengths: {doc_lengths}")
print(f"Average doc length: {avgdl:.1f}")

# Compute BM25 for 'singapore' across documents
print(f"\nBM25 vs TF-IDF for 'singapore':")
print(f"{'Doc':>4} {'TF':>4} {'DocLen':>7} {'TF-IDF':>10} {'BM25':>10}")
print("─" * 40)

# Document frequency for 'singapore'
sg_df = sum(1 for doc in toy_corpus if "singapore" in doc.lower())

for i, doc in enumerate(toy_corpus):
    words = doc.lower().split()
    tf = words.count("singapore")
    if tf > 0:
        tfidf_val = doc0_tfidf[list(tfidf_vocab).index("singapore")] if i == 0 else 0
        bm25_val = bm25_score(tf, sg_df, N_docs, len(words), avgdl)
        print(f"{i:>4} {tf:>4} {len(words):>7} {tfidf_val:>10.4f} {bm25_val:>10.4f}")

print("\nBM25 vs TF-IDF:")
print("  TF-IDF: TF grows linearly (10 occurrences = 10x weight)")
print("  BM25: TF saturates (10 occurrences ≈ 3x weight due to k1=1.2)")
print("  BM25 also normalises by document length (b=0.75)")
print("  BM25 is the default scoring in Elasticsearch and most search engines")

# ── Checkpoint 2 ─────────────────────────────────────────────────────
test_bm25 = bm25_score(tf=3, df=2, N=10, dl=50, avgdl=40)
assert test_bm25 > 0, "BM25 score should be positive for present terms"
score_tf1 = bm25_score(tf=1, df=2, N=10, dl=50, avgdl=40)
score_tf10 = bm25_score(tf=10, df=2, N=10, dl=50, avgdl=40)
assert score_tf10 < 10 * score_tf1, "BM25 should show TF saturation"
# INTERPRETATION: BM25 is the industry standard for text retrieval. The
# saturation parameter k1 prevents terms appearing 100 times from dominating
# those appearing 10 times. This matches human intuition: a 10th mention
# adds less information than the 1st.
print("\n✓ Checkpoint 2 passed — BM25 with saturation and length normalisation\n")


# ══════════════════════════════════════════════════════════════════════
# TASK 3: NMF topic extraction from TF-IDF
# ══════════════════════════════════════════════════════════════════════
# NMF: X ≈ W * H where W >= 0, H >= 0
#   W[doc, topic] = document's weight for each topic
#   H[topic, word] = topic's weight for each word
# Non-negativity makes topics interpretable (only additive contributions)

from sklearn.decomposition import NMF

n_nmf_topics = 3
# TODO: Create NMF with n_nmf_topics components, fit_transform on X_tfidf
nmf_toy = ____  # Hint: NMF(n_components=n_nmf_topics, random_state=42, max_iter=300)
W_toy = ____  # Hint: nmf_toy.fit_transform(X_tfidf)
H_toy = nmf_toy.components_

print("=== NMF Topics from TF-IDF (toy corpus) ===")
for t in range(n_nmf_topics):
    top_words = [tfidf_vocab[i] for i in H_toy[t].argsort()[-6:][::-1]]
    print(f"  Topic {t}: {', '.join(top_words)}")

print("\nNMF factorises X ≈ W x H where:")
print("  W[doc, topic] = document's weight for each topic")
print("  H[topic, word] = topic's weight for each word")
print("  Non-negativity makes topics additive and interpretable")

# Reconstruction quality
recon_error = np.linalg.norm(X_tfidf.toarray() - W_toy @ H_toy) / np.linalg.norm(
    X_tfidf.toarray()
)
print(f"\nNMF reconstruction error (relative): {recon_error:.4f}")

# ── Checkpoint 3 ─────────────────────────────────────────────────────
assert W_toy.shape == (len(toy_corpus), n_nmf_topics), "W should be (n_docs, n_topics)"
assert H_toy.shape == (
    n_nmf_topics,
    len(tfidf_vocab),
), "H should be (n_topics, n_vocab)"
assert W_toy.min() >= -1e-10, "NMF W matrix should be non-negative"
assert H_toy.min() >= -1e-10, "NMF H matrix should be non-negative"
# INTERPRETATION: NMF discovers additive parts — each document is a sum of
# topic contributions. Unlike PCA (which allows negative loadings), NMF
# topics only ADD to the representation, making them more interpretable.
print("\n✓ Checkpoint 3 passed — NMF topic extraction\n")


# ── Data Loading ──────────────────────────────────────────────────────

loader = MLFPDataLoader()
news = loader.load("mlfp03", "documents.parquet")

print(f"=== Document Corpus ===")
print(f"Shape: {news.shape}")
print(f"Columns: {news.columns}")
print(f"Categories: {news['category'].unique().to_list()}")


# ══════════════════════════════════════════════════════════════════════
# TASK 4: Load and preprocess text corpus
# ══════════════════════════════════════════════════════════════════════

news_clean = news.with_columns(
    (pl.col("title") + ". " + pl.col("content")).alias("text"),
    pl.col("category").alias("year_month"),
).filter(pl.col("content").str.len_chars() > 100)

documents = news_clean["text"].to_list()
categories = news_clean["category"].to_list()
print(f"\nCleaned corpus: {len(documents):,} documents")

# Corpus statistics
doc_lengths = [len(d.split()) for d in documents]
print(
    f"Document lengths: min={min(doc_lengths)}, median={np.median(doc_lengths):.0f}, "
    f"max={max(doc_lengths)}"
)
print(f"Total words: {sum(doc_lengths):,}")

# ── Checkpoint 4 ─────────────────────────────────────────────────────
assert len(documents) > 0, "Corpus should not be empty"
assert all(len(d) > 100 for d in documents), "All docs should be > 100 chars"
# INTERPRETATION: Text preprocessing in Polars is vectorised. The filter
# operation removes stubs in a single pass — 10-100x faster than Python loops.
print("\n✓ Checkpoint 4 passed — corpus cleaned and preprocessed\n")


# ══════════════════════════════════════════════════════════════════════
# TASK 5: LDA topic modelling
# ══════════════════════════════════════════════════════════════════════
# LDA: each document is a mixture of topics, each topic is a distribution
# over words. Generative model:
#   1. For each document, sample topic proportions theta ~ Dirichlet(alpha)
#   2. For each word position, sample topic z ~ Categorical(theta)
#   3. Sample word w ~ Categorical(beta_z)

from sklearn.decomposition import LatentDirichletAllocation

vectorizer_lda = TfidfVectorizer(
    max_features=3000, stop_words="english", max_df=0.95, min_df=3
)
X_lda_tfidf = vectorizer_lda.fit_transform(documents)
lda_vocab = vectorizer_lda.get_feature_names_out()

# Try different numbers of topics
print("=== LDA Topic Modelling ===")
lda_results = {}
for n_t in [5, 10, 15]:
    # TODO: Create LatentDirichletAllocation with n_t components
    lda = ____  # Hint: LatentDirichletAllocation(n_components=n_t, random_state=42, max_iter=30, learning_method="online", batch_size=128)
    lda.fit(X_lda_tfidf)
    perplexity = lda.perplexity(X_lda_tfidf)
    lda_results[n_t] = {"model": lda, "perplexity": perplexity}
    print(f"  K={n_t}: perplexity={perplexity:.0f}")

# Use K=10 for detailed analysis
n_topics_lda = 10
lda_model = lda_results[n_topics_lda]["model"]
lda_doc_topics = lda_model.transform(X_lda_tfidf)
lda_labels = lda_doc_topics.argmax(axis=1)

print(f"\nLDA Topics (K={n_topics_lda}):")
for t in range(n_topics_lda):
    top_words = [lda_vocab[i] for i in lda_model.components_[t].argsort()[-8:][::-1]]
    count = (lda_labels == t).sum()
    print(f"  Topic {t}: {', '.join(top_words)} (n={count})")

print("\nLDA vs NMF:")
print("  LDA: probabilistic, topics as distributions, mixed membership")
print("  NMF: deterministic, non-negative factorisation, faster")
print("  LDA: each document has a distribution OVER topics (soft)")
print("  NMF: each document has weights for topics (also soft but not probabilistic)")

# ── Checkpoint 5 ─────────────────────────────────────────────────────
assert lda_doc_topics.shape == (
    len(documents),
    n_topics_lda,
), "LDA output shape mismatch"
assert (
    abs(lda_doc_topics.sum(axis=1).mean() - 1.0) < 0.01
), "LDA topic proportions should sum to 1 per document"
# INTERPRETATION: LDA is a generative probabilistic model. Each document's
# topic distribution theta tells you HOW MUCH of each topic is in the document.
# Lower perplexity = better model (but diminishing returns beyond optimal K).
print("\n✓ Checkpoint 5 passed — LDA topic modelling\n")


# ══════════════════════════════════════════════════════════════════════
# TASK 6: BERTopic model
# ══════════════════════════════════════════════════════════════════════

if BERTopic is not None:
    # TODO: Create BERTopic with embedding_model="all-MiniLM-L6-v2", min_topic_size=20
    topic_model = ____  # Hint: BERTopic(embedding_model="all-MiniLM-L6-v2", umap_model=None, hdbscan_model=None, min_topic_size=20, nr_topics="auto", verbose=True)
    topics, probs = topic_model.fit_transform(documents)

    topic_info = topic_model.get_topic_info()
    n_topics = len(topic_info) - 1
    print(f"\n=== BERTopic Results ===")
    print(f"Topics found: {n_topics}")
    print(f"Outlier documents: {(np.array(topics) == -1).sum():,}")

    print(f"\nTop 10 Topics:")
    for _, row in topic_info.head(11).iterrows():
        if row["Topic"] == -1:
            continue
        print(f"  Topic {row['Topic']}: {row['Name'][:60]} (n={row['Count']})")
else:
    # NMF fallback
    print("\nBERTopic not installed, using TF-IDF + NMF fallback")
    vectorizer = TfidfVectorizer(
        max_features=5000, stop_words="english", max_df=0.95, min_df=5
    )
    tfidf_matrix = vectorizer.fit_transform(documents)
    feature_names = vectorizer.get_feature_names_out()

    n_topics = 15
    nmf = NMF(n_components=n_topics, random_state=42, max_iter=300)
    W = nmf.fit_transform(tfidf_matrix)
    H = nmf.components_

    topics = W.argmax(axis=1).tolist()
    probs = W / (W.sum(axis=1, keepdims=True) + 1e-10)

    print(f"\nNMF Topics ({n_topics}):")
    for topic_idx in range(n_topics):
        top_words = [feature_names[i] for i in H[topic_idx].argsort()[-8:][::-1]]
        count = sum(1 for t in topics if t == topic_idx)
        print(f"  Topic {topic_idx}: {', '.join(top_words)} (n={count})")

# ── Checkpoint 6 ─────────────────────────────────────────────────────
assert n_topics > 0, "Should discover at least one topic"
# INTERPRETATION: BERTopic pipeline: (1) SentenceTransformer embeds docs,
# (2) UMAP reduces to 5D, (3) HDBSCAN clusters, (4) c-TF-IDF extracts
# representative words per cluster. Topic -1 = outlier documents.
print("\n✓ Checkpoint 6 passed — BERTopic/NMF topics discovered\n")


# ══════════════════════════════════════════════════════════════════════
# TASK 7: Topic coherence evaluation (NPMI)
# ══════════════════════════════════════════════════════════════════════
# NPMI(w_i, w_j) = (log P(w_i,w_j)/(P(w_i)P(w_j))) / (-log P(w_i,w_j))
# Range: [-1, 1]. Higher = more coherent topic.


def compute_npmi(
    documents: list[str], topic_words: list[list[str]], window_size: int = 10
) -> list[float]:
    """Compute NPMI coherence for each topic."""
    word_doc_count = Counter()
    pair_doc_count = Counter()
    n_docs = len(documents)

    for doc in documents:
        words = set(doc.lower().split())
        for w in words:
            word_doc_count[w] += 1
        word_list = list(words)
        for i in range(len(word_list)):
            for j in range(i + 1, len(word_list)):
                pair = tuple(sorted([word_list[i], word_list[j]]))
                pair_doc_count[pair] += 1

    coherences = []
    for topic in topic_words:
        npmi_sum = 0
        n_pairs = 0
        for i in range(len(topic)):
            for j in range(i + 1, len(topic)):
                w_i, w_j = topic[i].lower(), topic[j].lower()
                pair = tuple(sorted([w_i, w_j]))
                p_i = word_doc_count.get(w_i, 0) / n_docs
                p_j = word_doc_count.get(w_j, 0) / n_docs
                p_ij = pair_doc_count.get(pair, 0) / n_docs

                if p_ij > 0 and p_i > 0 and p_j > 0:
                    # TODO: Compute NPMI = PMI / (-log(p_ij)) where PMI = log(p_ij / (p_i * p_j))
                    pmi = ____  # Hint: np.log(p_ij / (p_i * p_j))
                    npmi = ____  # Hint: pmi / (-np.log(p_ij))
                    npmi_sum += npmi
                    n_pairs += 1

        coherences.append(npmi_sum / max(n_pairs, 1))
    return coherences


# Get topic words from whichever model we used
if BERTopic is not None:
    topic_words = []
    for topic_id in range(min(n_topics, 15)):
        words = [w for w, _ in topic_model.get_topic(topic_id)[:10]]
        topic_words.append(words)
else:
    topic_words = []
    for topic_idx in range(min(n_topics, 15)):
        words = [feature_names[i] for i in H[topic_idx].argsort()[-10:][::-1]]
        topic_words.append(words)

# Also compute NPMI for LDA topics
lda_topic_words = []
for t in range(n_topics_lda):
    words = [lda_vocab[i] for i in lda_model.components_[t].argsort()[-10:][::-1]]
    lda_topic_words.append(words)

coherences_main = compute_npmi(documents[:3000], topic_words)
coherences_lda = compute_npmi(documents[:3000], lda_topic_words)

print(f"\n=== Topic Coherence (NPMI) ===")
method_name = "BERTopic" if BERTopic is not None else "NMF"
print(f"\n{method_name} topics:")
print(f"  Mean NPMI: {np.mean(coherences_main):.4f}")
for i, c in enumerate(coherences_main[:10]):
    bar = "#" * max(0, int((c + 0.5) * 20))
    print(f"  Topic {i}: {c:+.4f} {bar}")

print(f"\nLDA topics:")
print(f"  Mean NPMI: {np.mean(coherences_lda):.4f}")
for i, c in enumerate(coherences_lda[:10]):
    bar = "#" * max(0, int((c + 0.5) * 20))
    print(f"  Topic {i}: {c:+.4f} {bar}")

# ── Checkpoint 7 ─────────────────────────────────────────────────────
assert len(coherences_main) > 0, "Should compute NPMI for at least one topic"
mean_npmi = np.mean(coherences_main)
assert mean_npmi > -0.5, f"Mean NPMI should be > -0.5, got {mean_npmi:.4f}"
# INTERPRETATION: NPMI > 0.1 = words co-occur more than by chance (coherent).
# NPMI < 0 = words less likely to co-occur (incoherent). Best coherence
# when topic words form a tight semantic cluster.
print("\n✓ Checkpoint 7 passed — NPMI coherence for both models\n")


# ══════════════════════════════════════════════════════════════════════
# TASK 8: Word embeddings as features (conceptual)
# ══════════════════════════════════════════════════════════════════════

print("=== Word Embeddings as Features ===")
print(
    """
Word embeddings map words to dense vectors where similar words are close:

  Word2Vec (CBOW):     predict word from context
  Word2Vec (Skip-gram): predict context from word
  GloVe:              global co-occurrence matrix factorisation
  FastText:           subword embeddings (handles OOV words)

Key properties:
  - Semantic similarity: cos(king, queen) > cos(king, car)
  - Analogies: king - man + woman ≈ queen
  - Dense: 100-300 dimensions vs 10K+ for bag-of-words

How are these vectors learned? Neural networks! (covered in Ex 8)
  Word2Vec trains a shallow neural network to predict context words.
  The hidden layer weights become the word embeddings.
  This is the same principle as matrix factorisation in Ex 7:
    optimisation drives feature discovery.

Using word embeddings as features:
  1. Average word vectors in a document -> document vector
  2. Use as input to classifiers, clustering, similarity search
  3. Pretrained embeddings (GloVe, FastText) work well even without
     domain-specific training
"""
)

# Simulate document embeddings (in practice, use SentenceTransformer)
rng = np.random.default_rng(42)
n_embed_dim = 64
doc_embeddings = rng.standard_normal((len(documents), n_embed_dim)).astype(np.float32)

# Cluster document embeddings
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

km_embed = KMeans(n_clusters=5, random_state=42, n_init=5)
embed_labels = km_embed.fit_predict(doc_embeddings)
embed_sil = silhouette_score(doc_embeddings, embed_labels)
print(f"\nDocument embedding clustering (simulated):")
print(f"  Embedding dim: {n_embed_dim}")
print(f"  Clusters: 5, Silhouette: {embed_sil:.4f}")
print("  (Real embeddings from SentenceTransformer would produce meaningful clusters)")

# ── Checkpoint 8 ─────────────────────────────────────────────────────
assert doc_embeddings.shape == (len(documents), n_embed_dim), "Embedding shape mismatch"
# INTERPRETATION: Word embeddings encode semantic meaning in dense vectors.
# "How does Word2Vec learn these vectors? We will see in Ex 8 when we study
# neural networks." — the hidden layer activations ARE the embeddings.
print("\n✓ Checkpoint 8 passed — word embedding concepts\n")


# ══════════════════════════════════════════════════════════════════════
# TASK 9: Sentiment analysis (brief)
# ══════════════════════════════════════════════════════════════════════

print("=== Sentiment Analysis ===")

positive_words = {
    "good",
    "great",
    "excellent",
    "best",
    "strong",
    "growth",
    "resilience",
    "success",
    "positive",
    "improved",
    "innovative",
}
negative_words = {
    "bad",
    "poor",
    "decline",
    "fall",
    "weak",
    "crisis",
    "risk",
    "loss",
    "negative",
    "failed",
    "uncertainty",
}

sentiments = []
for doc in documents:
    words = set(doc.lower().split())
    pos_count = len(words & positive_words)
    neg_count = len(words & negative_words)
    total = pos_count + neg_count
    # TODO: Compute sentiment = (pos - neg) / total, else 0.0
    if total > 0:
        sentiment = ____  # Hint: (pos_count - neg_count) / total
    else:
        sentiment = 0.0
    sentiments.append(sentiment)

sentiments = np.array(sentiments)
print(f"Sentiment distribution:")
print(f"  Positive (> 0.3): {(sentiments > 0.3).mean():.1%}")
print(
    f"  Neutral (-0.3 to 0.3): {((sentiments >= -0.3) & (sentiments <= 0.3)).mean():.1%}"
)
print(f"  Negative (< -0.3): {(sentiments < -0.3).mean():.1%}")
print(f"  Mean sentiment: {sentiments.mean():.4f}")

# Sentiment by category
news_with_sentiment = news_clean.with_columns(
    pl.Series("sentiment", sentiments[: news_clean.height])
)
print(f"\nSentiment by category:")
for cat in sorted(news_with_sentiment["category"].unique().to_list()):
    cat_sent = news_with_sentiment.filter(pl.col("category") == cat)["sentiment"].mean()
    indicator = "+" if cat_sent > 0.05 else "-" if cat_sent < -0.05 else "~"
    print(f"  {cat:<20} {cat_sent:+.4f} {indicator}")

print("\nNote: lexicon-based sentiment is a baseline. Production systems use")
print("fine-tuned transformer models (e.g., DistilBERT for sentiment).")
print("You will build these in Module 5.")

# ── Checkpoint 9 ─────────────────────────────────────────────────────
assert len(sentiments) == len(documents), "Should have sentiment for each doc"
assert sentiments.min() >= -1.0 and sentiments.max() <= 1.0, "Sentiment in [-1, 1]"
# INTERPRETATION: Lexicon-based sentiment counts positive and negative words.
# It's fast but misses context ("not good" would be counted as positive).
# Transformer-based models (M5) handle negation and sarcasm correctly.
print("\n✓ Checkpoint 9 passed — sentiment analysis\n")


# ══════════════════════════════════════════════════════════════════════
# TASK 10: Topic distribution and visualisation
# ══════════════════════════════════════════════════════════════════════

news_with_topics = news_clean.with_columns(
    pl.Series("topic", [int(t) for t in topics[: news_clean.height]])
)

# Topic distribution by category
temporal = (
    news_with_topics.filter(pl.col("topic") >= 0)
    .group_by("year_month", "topic")
    .agg(pl.col("topic").count().alias("count"))
    .sort("year_month", "topic")
)

monthly_totals = temporal.group_by("year_month").agg(
    pl.col("count").sum().alias("total")
)
temporal = temporal.join(monthly_totals, on="year_month").with_columns(
    (pl.col("count") / pl.col("total")).alias("proportion")
)

print(f"=== Topic Distribution by Category ===")
all_categories = sorted(temporal["year_month"].unique().to_list())
for cat in all_categories[:5]:
    print(f"\n  Category '{cat}':")
    cat_data = temporal.filter(pl.col("year_month") == cat).sort(
        "proportion", descending=True
    )
    for row in cat_data.head(3).iter_rows(named=True):
        print(f"    Topic {row['topic']}: {row['proportion']:.1%}")

viz = ModelVisualizer()

# Topic coherence comparison
coherence_data = {f"Topic_{i}": {"NPMI": c} for i, c in enumerate(coherences_main[:10])}
fig = viz.metric_comparison(coherence_data)
fig.update_layout(title="Topic Coherence (NPMI)")
fig.write_html("ex6_topic_coherence.html")

# LDA vs NMF/BERTopic comparison
method_comparison = {
    f"{method_name} mean NPMI": {"NPMI": np.mean(coherences_main)},
    "LDA mean NPMI": {"NPMI": np.mean(coherences_lda)},
}
fig_cmp = viz.metric_comparison(method_comparison)
fig_cmp.update_layout(title="Topic Model Comparison: NPMI Coherence")
fig_cmp.write_html("ex6_model_comparison.html")

# Topic size distribution
topic_counts = Counter(t for t in topics if t >= 0)
size_data = {"Topic Size": [topic_counts.get(i, 0) for i in range(min(n_topics, 15))]}
fig_size = viz.training_history(size_data, x_label="Topic ID")
fig_size.update_layout(title="Topic Size Distribution")
fig_size.write_html("ex6_topic_sizes.html")

print(
    "\nSaved: ex6_topic_coherence.html, ex6_model_comparison.html, ex6_topic_sizes.html"
)

# ── Checkpoint 10 ────────────────────────────────────────────────────
assert len(coherences_main) > 0, "Should have coherence values"
# INTERPRETATION: Topic distribution by category reveals which topics dominate
# which content areas. Trending topics show up as increasing proportions
# over time. Declining topics may indicate shifting editorial focus.
print("\n✓ Checkpoint 10 passed — topic distribution and visualisation\n")

print("\n✓ Exercise 6 complete — NLP topic modelling")


# ══════════════════════════════════════════════════════════════════════
# REFLECTION
# ══════════════════════════════════════════════════════════════════════
print("═" * 70)
print("  WHAT YOU'VE MASTERED")
print("═" * 70)
print(
    f"""
  ✓ TF-IDF: TF(t,d) x log(N/df(t)) — rare discriminative terms rewarded
  ✓ BM25: saturation (k1) + length normalisation (b) improves TF-IDF
  ✓ NMF: non-negative matrix factorisation -> interpretable topics
  ✓ LDA: probabilistic, mixed membership, generative model
  ✓ BERTopic: neural embeddings + UMAP + HDBSCAN + c-TF-IDF
  ✓ NPMI coherence: quantify topic quality without human annotation
  ✓ Word embeddings: dense semantic vectors (Word2Vec, GloVe, FastText)
  ✓ Sentiment analysis: lexicon-based baseline for text classification

  NLP PIPELINE DECISION GUIDE:
    TF-IDF + NMF -> fast, interpretable, well-defined topics
    LDA          -> probabilistic, mixed membership, topic distributions
    BERTopic     -> semantic, handles polysemy, fine-grained topics

  NEXT: Exercise 7 introduces THE PIVOT — recommender systems with
  matrix factorisation. You'll see that learning user and item embeddings
  by minimising reconstruction error is the same principle as Word2Vec —
  and the same principle that neural networks use to learn representations.
"""
)
print("═" * 70)
