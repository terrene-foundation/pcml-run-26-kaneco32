# Copyright 2026 Terrene Foundation
# SPDX-License-Identifier: Apache-2.0
"""
# ════════════════════════════════════════════════════════════════════════
# MLFP04 — Exercise 5: Association Rules and Market Basket Analysis
# ════════════════════════════════════════════════════════════════════════
#
# WHAT YOU'LL LEARN:
#   After completing this exercise, you will be able to:
#   - Implement the Apriori algorithm from scratch with pruning
#   - Compute support, confidence, lift, and conviction for association rules
#   - Compare Apriori and FP-Growth implementations
#   - Filter actionable rules using three-threshold criteria
#   - Interpret rules with business meaning (cross-category, complements)
#   - Engineer rule-based features that improve a supervised classifier
#   - Explain the connection: co-occurrence -> matrix factorisation -> neural nets
#   - Use discovered patterns as features for supervised models
#
# PREREQUISITES:
#   - MLFP04 Exercise 1 (clustering — pattern discovery without labels)
#   - MLFP03 Exercise 1 (feature engineering — rules as domain features)
#
# ESTIMATED TIME: ~150-180 min
#
# TASKS:
#   1.  Generate synthetic Singapore retail transaction data
#   2.  Implement Apriori from scratch: frequent itemsets, candidate gen, pruning
#   3.  Compute support, confidence, lift, conviction for discovered rules
#   4.  Compare with mlxtend FP-Growth
#   5.  Filter and rank actionable rules (three-threshold criteria)
#   6.  Interpret rules with business meaning and category analysis
#   7.  Engineer features from discovered rules for classification
#   8.  Train and compare baseline vs rule-enhanced supervised models
#   9.  Feature importance analysis: which rule features matter most
#   10. Forward connections: co-occurrence -> matrix factorisation -> neural nets
#
# DATASET: Synthetic Singapore retail transactions (2500 transactions, 25 products)
#
# THEORY:
#   Support:    supp(X) = count(X) / N
#   Confidence: conf(X -> Y) = supp(X u Y) / supp(X)
#   Lift:       lift(X -> Y) = conf(X -> Y) / supp(Y)
#               Lift > 1 = positive association (surprise)
#   Conviction: conv(X -> Y) = (1 - supp(Y)) / (1 - conf(X -> Y))
#
# ════════════════════════════════════════════════════════════════════════
"""
from __future__ import annotations

from collections import defaultdict
from itertools import combinations

import numpy as np
import polars as pl
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from kailash_ml import ModelVisualizer

try:
    from mlxtend.frequent_patterns import apriori as mlx_apriori
    from mlxtend.frequent_patterns import association_rules as mlx_association_rules
    from mlxtend.frequent_patterns import fpgrowth as mlx_fpgrowth
except ImportError:
    mlx_apriori = None
    mlx_fpgrowth = None
    mlx_association_rules = None


# ══════════════════════════════════════════════════════════════════════
# TASK 1: Generate synthetic Singapore retail transaction data
# ══════════════════════════════════════════════════════════════════════

rng = np.random.default_rng(42)

PRODUCTS = [
    "bread",
    "butter",
    "milk",
    "eggs",
    "rice",
    "noodles",
    "soy_sauce",
    "cooking_oil",
    "chicken",
    "fish",
    "coffee",
    "tea",
    "sugar",
    "condensed_milk",
    "biscuits",
    "chips",
    "soft_drink",
    "beer",
    "wine",
    "tissue",
    "shampoo",
    "soap",
    "detergent",
    "toothpaste",
    "bananas",
]

N_TRANSACTIONS = 2500

BUNDLES = [
    (["bread", "butter", "eggs"], 0.15),
    (["coffee", "condensed_milk", "sugar"], 0.12),
    (["rice", "chicken", "soy_sauce"], 0.10),
    (["noodles", "eggs", "soy_sauce"], 0.08),
    (["beer", "chips"], 0.09),
    (["milk", "biscuits"], 0.07),
    (["shampoo", "soap", "toothpaste"], 0.06),
    (["tea", "sugar", "biscuits"], 0.05),
    (["wine", "chips", "biscuits"], 0.04),
    (["cooking_oil", "rice", "fish"], 0.06),
    (["detergent", "tissue", "soap"], 0.05),
    (["bananas", "milk", "eggs"], 0.05),
]

transactions: list[set[str]] = []
for _ in range(N_TRANSACTIONS):
    basket: set[str] = set()
    for bundle_items, prob in BUNDLES:
        if rng.random() < prob:
            for item in bundle_items:
                if rng.random() < 0.85:
                    basket.add(item)
    n_random = rng.poisson(2)
    random_items = rng.choice(PRODUCTS, size=min(n_random, 5), replace=False)
    basket.update(random_items)
    if len(basket) > 0:
        transactions.append(basket)

print("=== Synthetic Retail Transactions ===")
print(f"Transactions: {len(transactions):,}")
print(f"Products: {len(PRODUCTS)}")
avg_basket = np.mean([len(t) for t in transactions])
print(f"Avg basket size: {avg_basket:.1f} items")
print(f"\nSample transactions:")
for i in range(5):
    print(f"  Txn {i}: {sorted(transactions[i])}")

# Product frequency analysis
product_freq = defaultdict(int)
for txn in transactions:
    for item in txn:
        product_freq[item] += 1

print(f"\nProduct frequency (top 10):")
for item, count in sorted(product_freq.items(), key=lambda x: -x[1])[:10]:
    print(f"  {item:<20} {count:>5} ({count / len(transactions):.1%})")

# ── Checkpoint 1 ─────────────────────────────────────────────────────
assert len(transactions) >= 2000, "Should have at least 2000 transactions"
assert avg_basket > 2, "Average basket should have > 2 items"
# INTERPRETATION: Realistic transaction data has co-purchase patterns (bundles)
# plus random noise. The bundles simulate real behaviour: breakfast items bought
# together, kopi ingredients together, household items together.
print("\n✓ Checkpoint 1 passed — transaction data generated\n")


# ══════════════════════════════════════════════════════════════════════
# TASK 2: Implement Apriori from scratch
# ══════════════════════════════════════════════════════════════════════
# Apriori principle: if an itemset is infrequent, all supersets are too.
# This allows aggressive pruning of the candidate space.


def _generate_candidates(
    prev_level: list[frozenset[str]],
    k: int,
) -> list[frozenset[str]]:
    """Generate candidate k-itemsets with Apriori pruning."""
    prev_set = set(prev_level)
    candidates: set[frozenset[str]] = set()

    for i, a in enumerate(prev_level):
        for b in prev_level[i + 1 :]:
            union = a | b
            if len(union) == k:
                all_subsets_frequent = all(
                    union - frozenset([item]) in prev_set for item in union
                )
                if all_subsets_frequent:
                    candidates.add(union)

    return list(candidates)


def apriori(
    transactions: list[set[str]],
    min_support: float,
) -> dict[frozenset[str], float]:
    """Apriori algorithm for frequent itemset mining."""
    n = len(transactions)
    min_count = min_support * n

    # Step 1: Count single-item support
    item_counts: dict[str, int] = defaultdict(int)
    for txn in transactions:
        for item in txn:
            item_counts[item] += 1

    # Step 2: L1 — frequent 1-itemsets
    freq_itemsets: dict[frozenset[str], float] = {}
    current_level: list[frozenset[str]] = []
    for item, count in item_counts.items():
        if count >= min_count:
            fs = frozenset([item])
            # TODO: Compute support = count / n and store in freq_itemsets
            freq_itemsets[fs] = ____  # Hint: count / n
            current_level.append(fs)

    print(f"\n  L1: {len(current_level)} frequent items (min_support={min_support})")

    k = 2
    while current_level:
        candidates = _generate_candidates(current_level, k)
        if not candidates:
            break

        candidate_counts: dict[frozenset[str], int] = defaultdict(int)
        for txn in transactions:
            txn_frozen = frozenset(txn)
            for candidate in candidates:
                # TODO: Increment count if candidate is a subset of the transaction
                if ____:  # Hint: candidate.issubset(txn_frozen)
                    candidate_counts[candidate] += 1

        current_level = []
        for candidate, count in candidate_counts.items():
            if count >= min_count:
                freq_itemsets[candidate] = count / n
                current_level.append(candidate)

        print(f"  L{k}: {len(current_level)} frequent {k}-itemsets")
        k += 1

    return freq_itemsets


print("=== Apriori Algorithm (from scratch) ===")
min_sup = 0.03
frequent_itemsets = apriori(transactions, min_support=min_sup)
print(f"\nTotal frequent itemsets: {len(frequent_itemsets)}")

# ── Checkpoint 2 ─────────────────────────────────────────────────────
assert len(frequent_itemsets) > 0, "Should find at least one frequent itemset"
n = len(transactions)
for itemset, support in list(frequent_itemsets.items())[:5]:
    actual_count = sum(1 for t in transactions if itemset.issubset(frozenset(t)))
    actual_support = actual_count / n
    assert (
        abs(actual_support - support) < 0.005
    ), f"Support {support:.4f} doesn't match actual {actual_support:.4f}"
    assert (
        support >= min_sup - 0.001
    ), f"Itemset with support {support:.4f} should meet min_support={min_sup}"
# INTERPRETATION: The Apriori principle is anti-monotone: if {bread, butter} is
# infrequent, no superset can be frequent. For 25 products, there are 2^25 = 33M
# possible itemsets; Apriori checks only ~1000 candidates.
print(
    "\n✓ Checkpoint 2 passed — Apriori found frequent itemsets with correct support\n"
)

sorted_itemsets = sorted(frequent_itemsets.items(), key=lambda x: -x[1])
print(f"\nTop 15 frequent itemsets:")
print(f"  {'Itemset':<45} {'Support':>8}")
print("  " + "-" * 55)
for itemset, support in sorted_itemsets[:15]:
    items_str = ", ".join(sorted(itemset))
    print(f"  {items_str:<45} {support:>8.4f}")


# ══════════════════════════════════════════════════════════════════════
# TASK 3: Compute association rules — support, confidence, lift, conviction
# ══════════════════════════════════════════════════════════════════════


def generate_rules(
    freq_itemsets: dict[frozenset[str], float],
    min_confidence: float = 0.5,
) -> list[dict]:
    """Generate association rules from frequent itemsets."""
    rules = []
    for itemset, support in freq_itemsets.items():
        if len(itemset) < 2:
            continue
        items = list(itemset)
        for r in range(1, len(items)):
            for antecedent_tuple in combinations(items, r):
                antecedent = frozenset(antecedent_tuple)
                consequent = itemset - antecedent

                supp_ant = freq_itemsets.get(antecedent)
                supp_con = freq_itemsets.get(consequent)
                if supp_ant is None or supp_con is None:
                    continue

                # TODO: Compute confidence = support / supp_ant
                confidence = ____  # Hint: support / supp_ant
                if confidence < min_confidence:
                    continue

                # TODO: Compute lift = confidence / supp_con
                lift = ____  # Hint: confidence / supp_con

                # TODO: Compute conviction = (1 - supp_con) / (1 - confidence)
                # Use float("inf") when confidence == 1.0
                conviction = (
                    ____  # Hint: (1 - supp_con) / (1 - confidence)
                    if confidence < 1.0
                    else float("inf")
                )

                rules.append(
                    {
                        "antecedent": antecedent,
                        "consequent": consequent,
                        "support": support,
                        "confidence": confidence,
                        "lift": lift,
                        "conviction": conviction,
                    }
                )
    return rules


min_conf = 0.3
rules = generate_rules(frequent_itemsets, min_confidence=min_conf)
rules_by_lift = sorted(rules, key=lambda r: -r["lift"])

# ── Checkpoint 3 ─────────────────────────────────────────────────────
assert len(rules) > 0, "Should generate at least one rule"
for rule in rules[:5]:
    assert 0 <= rule["confidence"] <= 1.0, f"Confidence must be in [0,1]"
    assert rule["lift"] > 0, f"Lift must be positive"
# INTERPRETATION: Confidence = P(Y|X), Lift > 1 = positive association.
# The most actionable rules have both high confidence AND high lift.
print("\n✓ Checkpoint 3 passed — association rules with valid metrics\n")

print(f"=== Association Rules ===")
print(f"Rules found: {len(rules)} (min_confidence={min_conf})")
print(f"\nTop 20 rules by lift:")
print(
    f"  {'Antecedent':<25} {'->':>3} {'Consequent':<20} {'Supp':>6} {'Conf':>6} {'Lift':>6} {'Conv':>6}"
)
print("  " + "-" * 90)
for rule in rules_by_lift[:20]:
    ant = ", ".join(sorted(rule["antecedent"]))
    con = ", ".join(sorted(rule["consequent"]))
    conv_str = (
        f"{rule['conviction']:.2f}" if rule["conviction"] != float("inf") else "inf"
    )
    print(
        f"  {ant:<25} {'->':>3} {con:<20} "
        f"{rule['support']:>6.3f} {rule['confidence']:>6.3f} "
        f"{rule['lift']:>6.2f} {conv_str:>6}"
    )


# ══════════════════════════════════════════════════════════════════════
# TASK 4: FP-Growth comparison (mlxtend)
# ══════════════════════════════════════════════════════════════════════

all_items = sorted(PRODUCTS)
rows = []
for txn in transactions:
    row = {item: item in txn for item in all_items}
    rows.append(row)

txn_df = pl.DataFrame(rows)
print(f"\n=== Transaction Matrix ===")
print(f"Shape: {txn_df.shape} (transactions x products)")
print(f"Density: {txn_df.select(pl.all().mean()).to_numpy().mean():.2%}")

if mlx_fpgrowth is not None:
    txn_pd = txn_df.to_pandas()
    # TODO: Run FP-Growth with min_support=min_sup, use_colnames=True
    fp_frequent = (
        ____  # Hint: mlx_fpgrowth(txn_pd, min_support=min_sup, use_colnames=True)
    )
    fp_rules = mlx_association_rules(
        fp_frequent, metric="confidence", min_threshold=min_conf
    )

    print(f"\n=== FP-Growth (mlxtend) ===")
    print(f"Frequent itemsets: {len(fp_frequent)}")
    print(f"Association rules: {len(fp_rules)}")
    print(f"\nComparison:")
    print(f"  Apriori (scratch): {len(frequent_itemsets)} itemsets, {len(rules)} rules")
    print(f"  FP-Growth (mlxtend): {len(fp_frequent)} itemsets, {len(fp_rules)} rules")

    # Top-10 overlap check
    our_top = set()
    for r in rules_by_lift[:10]:
        our_top.add((frozenset(r["antecedent"]), frozenset(r["consequent"])))
    fp_rules_sorted = fp_rules.sort_values("lift", ascending=False)
    fp_top = set()
    for _, row in fp_rules_sorted.head(10).iterrows():
        fp_top.add((frozenset(row["antecedents"]), frozenset(row["consequents"])))
    overlap = len(our_top & fp_top)
    print(
        f"\n  Top-10 rule overlap: {overlap}/10 — {'agree' if overlap >= 7 else 'diverge'}"
    )
else:
    print("\nmlxtend not installed — skipping FP-Growth comparison")

# ── Checkpoint 4 ─────────────────────────────────────────────────────
# INTERPRETATION: FP-Growth uses a compressed FP-tree — no candidate generation.
# Much faster than Apriori for large datasets. Both should produce identical
# frequent itemsets given the same min_support threshold.
print("\n✓ Checkpoint 4 passed — FP-Growth comparison (if available)\n")


# ══════════════════════════════════════════════════════════════════════
# TASK 5: Filter and rank actionable rules
# ══════════════════════════════════════════════════════════════════════

# TODO: Filter rules to those passing all three thresholds:
# support >= 0.03, confidence >= 0.4, lift > 1.5
actionable_rules = ____  # Hint: [r for r in rules if r["support"] >= 0.03 and r["confidence"] >= 0.4 and r["lift"] > 1.5]
actionable_rules.sort(key=lambda r: -r["lift"])

# ── Checkpoint 5 ─────────────────────────────────────────────────────
assert len(actionable_rules) > 0, "At least one rule should pass all thresholds"
assert actionable_rules[0]["lift"] > 1.5, "Top rule lift should exceed 1.5"
# INTERPRETATION: Three-threshold filter implements actionability: support
# ensures enough data, confidence ensures reliability, lift ensures surprise.
print("\n✓ Checkpoint 5 passed — actionable rules filtered\n")

print(f"=== Actionable Rules (supp>=0.03, conf>=0.4, lift>1.5) ===")
print(f"Rules passing: {len(actionable_rules)}")
print(
    f"\n  {'Antecedent':<25} {'->':>3} {'Consequent':<20} {'Supp':>6} {'Conf':>6} {'Lift':>6}"
)
print("  " + "-" * 85)
for rule in actionable_rules[:15]:
    ant = ", ".join(sorted(rule["antecedent"]))
    con = ", ".join(sorted(rule["consequent"]))
    print(
        f"  {ant:<25} {'->':>3} {con:<20} "
        f"{rule['support']:>6.3f} {rule['confidence']:>6.3f} {rule['lift']:>6.2f}"
    )


# ══════════════════════════════════════════════════════════════════════
# TASK 6: Interpret rules with business meaning
# ══════════════════════════════════════════════════════════════════════

CATEGORY_MAP = {
    "bread": "breakfast",
    "butter": "breakfast",
    "eggs": "breakfast",
    "milk": "dairy",
    "condensed_milk": "dairy",
    "coffee": "beverage",
    "tea": "beverage",
    "soft_drink": "beverage",
    "sugar": "pantry",
    "rice": "pantry",
    "cooking_oil": "pantry",
    "soy_sauce": "pantry",
    "noodles": "pantry",
    "chicken": "protein",
    "fish": "protein",
    "beer": "alcohol",
    "wine": "alcohol",
    "chips": "snack",
    "biscuits": "snack",
    "bananas": "fruit",
    "shampoo": "personal_care",
    "soap": "personal_care",
    "toothpaste": "personal_care",
    "tissue": "household",
    "detergent": "household",
}

# Cross-category analysis
cross_category_rules = []
within_category_rules = []

for rule in actionable_rules:
    ant_cats = set(CATEGORY_MAP.get(item, "other") for item in rule["antecedent"])
    con_cats = set(CATEGORY_MAP.get(item, "other") for item in rule["consequent"])
    if ant_cats & con_cats:
        within_category_rules.append(rule)
    else:
        cross_category_rules.append(rule)

print(f"\n=== Business Interpretation ===")
print(f"Cross-category rules: {len(cross_category_rules)}")
print(f"Within-category rules: {len(within_category_rules)}")

for i, rule in enumerate(actionable_rules[:10]):
    ant_items = sorted(rule["antecedent"])
    con_items = sorted(rule["consequent"])
    ant_cats = set(CATEGORY_MAP.get(item, "other") for item in ant_items)
    con_cats = set(CATEGORY_MAP.get(item, "other") for item in con_items)

    if ant_cats == con_cats:
        rel_type = "within-category complement"
    elif ant_cats & con_cats:
        rel_type = "cross-category with overlap"
    else:
        rel_type = "cross-category association"

    print(f"\n  Rule {i + 1}: {' + '.join(ant_items)} -> {' + '.join(con_items)}")
    print(f"    Lift={rule['lift']:.2f}, Conf={rule['confidence']:.1%}")
    print(f"    Categories: {', '.join(ant_cats)} -> {', '.join(con_cats)}")
    print(f"    Type: {rel_type}")
    if rule["lift"] > 3.0:
        print(f"    Action: STRONG pairing — co-locate on shelf, bundle discount")
    elif rule["lift"] > 2.0:
        print(f"    Action: Moderate pairing — cross-sell recommendation")
    else:
        print(f"    Action: Mild pairing — personalised suggestion")

# ── Checkpoint 6 ─────────────────────────────────────────────────────
assert len(cross_category_rules) + len(within_category_rules) == len(
    actionable_rules
), "All actionable rules should be categorised"
# INTERPRETATION: Cross-category rules (e.g., breakfast -> beverage) are more
# valuable for store layout and marketing. Within-category rules (e.g., personal
# care items together) are expected. The surprise is in the cross-category links.
print("\n✓ Checkpoint 6 passed — business interpretation with category analysis\n")


# ══════════════════════════════════════════════════════════════════════
# TASK 7: Engineer features from discovered rules
# ══════════════════════════════════════════════════════════════════════

basket_sizes = np.array([len(t) for t in transactions])
high_value_threshold = 6
labels = (basket_sizes >= high_value_threshold).astype(int)

print(f"=== Feature Engineering from Rules ===")
print(f"Target: high-value shopper (basket >= {high_value_threshold} items)")
print(f"Positive rate: {labels.mean():.1%}")

X_baseline = txn_df.to_numpy().astype(np.float64)

top_rules_for_features = actionable_rules[:20]

rule_features: list[dict[str, int]] = []
for txn in transactions:
    txn_set = frozenset(txn)
    row: dict[str, int] = {}

    rules_triggered = 0
    total_lift = 0.0

    for idx, rule in enumerate(top_rules_for_features):
        ant_present = int(rule["antecedent"].issubset(txn_set))
        full_present = int((rule["antecedent"] | rule["consequent"]).issubset(txn_set))

        ant_name = "_".join(sorted(rule["antecedent"]))
        con_name = "_".join(sorted(rule["consequent"]))
        row[f"rule_{idx}_ant_{ant_name}"] = ant_present
        row[f"rule_{idx}_full_{ant_name}_to_{con_name}"] = full_present

        if full_present:
            rules_triggered += 1
            total_lift += rule["lift"]

    row["n_rules_triggered"] = rules_triggered
    row["total_rule_lift"] = int(total_lift * 100)
    row["max_rule_lift"] = int(
        max(
            (
                r["lift"]
                for r in top_rules_for_features
                if (r["antecedent"] | r["consequent"]).issubset(txn_set)
            ),
            default=0,
        )
        * 100
    )
    rule_features.append(row)

rule_df = pl.DataFrame(rule_features).fill_null(0)
X_rules = rule_df.to_numpy().astype(np.float64)
# TODO: Horizontally stack X_baseline and X_rules to create combined features
X_combined = ____  # Hint: np.hstack([X_baseline, X_rules])

print(f"Baseline features: {X_baseline.shape[1]} (product presence)")
print(f"Rule features: {X_rules.shape[1]} (from {len(top_rules_for_features)} rules)")
print(f"Combined features: {X_combined.shape[1]}")

# ── Checkpoint 7 ─────────────────────────────────────────────────────
assert X_combined.shape[1] > X_baseline.shape[1], "Combined should have more features"
assert X_rules.shape[1] > 0, "Should generate at least some rule features"
# INTERPRETATION: Rule-based features encode co-purchase patterns as explicit
# signals. The n_rules_triggered and total_rule_lift features capture the
# overall bundle-buying behaviour of each customer.
print("\n✓ Checkpoint 7 passed — rule-based features engineered\n")


# ══════════════════════════════════════════════════════════════════════
# TASK 8: Train and compare baseline vs rule-enhanced models
# ══════════════════════════════════════════════════════════════════════

X_base_train, X_base_test, y_train, y_test = train_test_split(
    X_baseline, labels, test_size=0.3, random_state=42, stratify=labels
)
X_rules_train, X_rules_test, _, _ = train_test_split(
    X_rules, labels, test_size=0.3, random_state=42, stratify=labels
)
X_comb_train, X_comb_test, _, _ = train_test_split(
    X_combined, labels, test_size=0.3, random_state=42, stratify=labels
)

scaler_base = StandardScaler()
scaler_rules = StandardScaler()
scaler_comb = StandardScaler()

X_base_train_s = scaler_base.fit_transform(X_base_train)
X_base_test_s = scaler_base.transform(X_base_test)
X_rules_train_s = scaler_rules.fit_transform(X_rules_train)
X_rules_test_s = scaler_rules.transform(X_rules_test)
X_comb_train_s = scaler_comb.fit_transform(X_comb_train)
X_comb_test_s = scaler_comb.transform(X_comb_test)

results: dict[str, dict[str, float]] = {}

print("=== Model Comparison: Logistic Regression ===")
for name, X_tr, X_te in [
    ("Baseline (products)", X_base_train_s, X_base_test_s),
    ("Rules only", X_rules_train_s, X_rules_test_s),
    ("Combined", X_comb_train_s, X_comb_test_s),
]:
    # TODO: Train a LogisticRegression, predict, and compute accuracy, F1, AUC
    lr = ____  # Hint: LogisticRegression(max_iter=1000, random_state=42)
    lr.fit(X_tr, y_train)
    y_pred = lr.predict(X_te)
    y_proba = lr.predict_proba(X_te)[:, 1]
    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_proba)
    results[f"LR: {name}"] = {"Accuracy": acc, "F1": f1, "AUC_ROC": auc}
    print(f"  {name:<25} Acc={acc:.4f}  F1={f1:.4f}  AUC={auc:.4f}")

print(f"\n=== Model Comparison: Random Forest ===")
for name, X_tr, X_te in [
    ("Baseline (products)", X_base_train, X_base_test),
    ("Rules only", X_rules_train, X_rules_test),
    ("Combined", X_comb_train, X_comb_test),
]:
    # TODO: Train a RandomForestClassifier with n_estimators=200, max_depth=10
    rf = ____  # Hint: RandomForestClassifier(n_estimators=200, max_depth=10, random_state=42, n_jobs=-1)
    rf.fit(X_tr, y_train)
    y_pred = rf.predict(X_te)
    y_proba = rf.predict_proba(X_te)[:, 1]
    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_proba)
    results[f"RF: {name}"] = {"Accuracy": acc, "F1": f1, "AUC_ROC": auc}
    print(f"  {name:<25} Acc={acc:.4f}  F1={f1:.4f}  AUC={auc:.4f}")

# ── Checkpoint 8 ─────────────────────────────────────────────────────
lr_base_auc = results.get("LR: Baseline (products)", {}).get("AUC_ROC", 0)
lr_comb_auc = results.get("LR: Combined", {}).get("AUC_ROC", 0)
rf_base_auc = results.get("RF: Baseline (products)", {}).get("AUC_ROC", 0)
assert lr_base_auc > 0.5, "Baseline LR should beat random"
assert rf_base_auc > 0.5, "Baseline RF should beat random"
assert (
    lr_comb_auc >= lr_base_auc - 0.05
), "Rule-enhanced LR should not significantly regress"
# INTERPRETATION: Rule-based features encode co-purchase patterns as explicit
# signals. For simple models (LR), the rules provide structure the model
# cannot learn from raw product presence alone.
print("\n✓ Checkpoint 8 passed — baseline vs rule-enhanced models compared\n")


# ══════════════════════════════════════════════════════════════════════
# TASK 9: Feature importance analysis
# ══════════════════════════════════════════════════════════════════════

rf_combined = RandomForestClassifier(
    n_estimators=200, max_depth=10, random_state=42, n_jobs=-1
)
rf_combined.fit(X_comb_train, y_train)
importances = rf_combined.feature_importances_

product_importance = importances[: X_baseline.shape[1]].sum()
rule_importance = importances[X_baseline.shape[1] :].sum()
total_importance = product_importance + rule_importance

print(f"=== Feature Importance Attribution ===")
print(f"Product features contribute: {product_importance / total_importance:.1%}")
print(f"Rule features contribute:    {rule_importance / total_importance:.1%}")

all_feature_names = all_items + list(rule_df.columns)
top_features_idx = np.argsort(importances)[::-1][:15]
print(f"\nTop 15 features (combined model):")
for idx in top_features_idx:
    fname = all_feature_names[idx] if idx < len(all_feature_names) else f"feature_{idx}"
    ftype = "product" if idx < X_baseline.shape[1] else "rule"
    print(f"  [{ftype:>7}] {fname:<50} {importances[idx]:.4f}")

# ── Checkpoint 9 ─────────────────────────────────────────────────────
assert (
    product_importance + rule_importance > 0.99
), "Total importance should sum to ~1.0"
# INTERPRETATION: If rule features appear in the top-15 by importance, they
# capture signal that raw product presence misses. The n_rules_triggered
# feature often ranks high because it summarises bundle-buying behaviour.
print("\n✓ Checkpoint 9 passed — feature importance analysis\n")


# ══════════════════════════════════════════════════════════════════════
# TASK 10: Forward connections and visualisation
# ══════════════════════════════════════════════════════════════════════

print(f"=== Forward Connections ===")
print(
    """
Association rules are the MANUAL approach to discovering co-occurrence features.

The FORWARD CONNECTION through the rest of Module 4:
  1. Association rules (Ex 5): discover co-occurrence features MANUALLY
     - You define support/confidence/lift thresholds
     - You select which rules become features
     - Explicit, interpretable, but requires domain expertise

  2. Matrix factorisation (Ex 7): discover latent factors AUTOMATICALLY
     - R ≈ U * V^T learns user and item embeddings
     - Same co-occurrence structure, but learned by optimisation
     - Less interpretable, but discovers patterns you'd never specify

  3. Neural networks (Ex 8): discover NONLINEAR latent representations
     - Hidden layers learn arbitrary feature combinations
     - Backpropagation replaces manual threshold selection
     - Most powerful, least interpretable

This progression:
  Manual rules -> Linear factorisation -> Nonlinear neural nets
  is the arc of Module 4: from explicit to learned feature discovery.
"""
)

viz = ModelVisualizer()

fig = viz.metric_comparison(results)
fig.update_layout(title="Baseline vs Rule-Enhanced Model Comparison")
fig.write_html("ex5_model_comparison.html")
print("Saved: ex5_model_comparison.html")

rules_for_plot = pl.DataFrame(
    {
        "support": [r["support"] for r in rules_by_lift[:100]],
        "confidence": [r["confidence"] for r in rules_by_lift[:100]],
        "lift": [r["lift"] for r in rules_by_lift[:100]],
        "rule": [
            f"{', '.join(sorted(r['antecedent']))} -> {', '.join(sorted(r['consequent']))}"
            for r in rules_by_lift[:100]
        ],
    }
)

fig_scatter = viz.scatter(rules_for_plot, x="support", y="confidence", color="lift")
fig_scatter.update_layout(
    title="Association Rules: Support vs Confidence (colour=Lift)"
)
fig_scatter.write_html("ex5_rules_scatter.html")
print("Saved: ex5_rules_scatter.html")

# ── Checkpoint 10 ────────────────────────────────────────────────────
# INTERPRETATION: The scatter plot reveals the tradeoff: high-support rules
# tend to have lower lift (common patterns are less surprising). The most
# valuable rules are in the upper-right quadrant: high support AND high lift.
print("\n✓ Checkpoint 10 passed — visualisation and forward connections\n")

print("\n✓ Exercise 5 complete — association rules and market basket analysis")


# ══════════════════════════════════════════════════════════════════════
# REFLECTION
# ══════════════════════════════════════════════════════════════════════
print("═" * 70)
print("  WHAT YOU'VE MASTERED")
print("═" * 70)
print(
    f"""
  ✓ Apriori: anti-monotone pruning eliminates supersets of infrequent sets
  ✓ Support, confidence, lift, conviction: four axes of rule quality
  ✓ FP-Growth: compressed FP-tree, no candidate generation, faster
  ✓ Three-threshold filter: support + confidence + lift = actionability
  ✓ Category analysis: cross-category vs within-category rules
  ✓ Feature engineering: co-purchase patterns as supervised features
  ✓ Model comparison: rule features add value for simpler models

  THE FORWARD CONNECTION:
    Association rules    -> co-occurrence features manually
    Matrix factorisation -> latent factors automatically (Ex 7)
    Neural networks      -> nonlinear feature combinations (Ex 8)

  NEXT: Exercise 6 moves to unstructured text. You'll derive TF-IDF
  from scratch, apply NMF and BERTopic topic modelling, and evaluate
  topic quality using NPMI coherence metrics.
"""
)
print("═" * 70)
