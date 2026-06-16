# CROSS-POLLINATION.md — ternary-search-rs

> **Conservation Law Connection:** Search allocates γ across the crate space

## Role in the Conservation Law

`ternary-search-rs` provides semantic search over 1,150 crate repositories using
384-dimensional BGE embeddings. In the conservation law framework:

- **Search quality** = γ (finding the right crate is productive)
- **Search overhead** = η (ranking, scoring, routing computation)
- **Index size** contributes to C (larger index = more total capacity needed)

The ternary voting principle applies to search result ranking:
- **+1:** Relevant (contributes to γ — use this crate)
- **0:** Neutral (ambiguous — explore further or discard)
- **−1:** Irrelevant (contributes to η — noise in the index)

Search results with ternary scoring naturally implement the CLT cancellation:
irrelevant results (−1) cancel against relevant results (+1), leaving the
neutral (0) boundary cases for human judgment.

## delta-clt Verification Results

With 1,012 crates indexed (384-dim vectors), the search system processes queries
against a population that is large enough for δ(n) to be small: δ(1012) ≈ 0.031.

This means search η (ranking overhead) should be ≈ 3.1% of total search budget —
the system is well into the regime where the conservation law predicts high efficiency.

The live experiment showed that small populations (n=6) have high drift. This
validates the need for large indexes: a search system with only 6 crates would
have 18.6% search overhead, while 1,012 crates achieves <4%.

## Cross-Repo Connections

### → superinstance-core
`superinstance-core` provides ECS entity management. Search results can be
registered as entities with component metadata, enabling fleet-wide discovery.

**Shared:** Both manage collections of entities/components.
**Different:** `search-rs` is about finding things by similarity; `core` is about
managing things once found.

### → ternary-fleet
Fleet components need to discover each other. `ternary-search-rs` provides the
crate discovery layer for fleet composition — "which crate implements X?" is
a search query.

**Shared:** Both serve the fleet ecosystem.
**Different:** `fleet` runs components; `search-rs` finds them.

### → ternary-svm
Both are ML-style systems over vector spaces. `ternary-search-rs` uses cosine
similarity in embedding space; `ternary-svm` uses hyperplane separation in
ternary feature space. They could be composed: SVM classifies search results
into relevant/neutral/irrelevant (ternary output).

**Shared:** Both operate on vector representations. Both produce ternary classifications.
**Different:** Search is similarity-based (nearest neighbor); SVM is boundary-based (margin).

## Fleet Position

```
┌──────────────────────────────────────────────────┐
│  ternary-search-rs — THE DISCOVERY LAYER          │
│                                                   │
│  1,012 crates × 384-dim embeddings                │
│  δ(1012) ≈ 0.031 → 97% search efficiency          │
│                                                   │
│  Query → Trit scoring:                            │
│    +1: relevant (γ contribution)                  │
│     0: neutral (human judgment needed)            │
│    −1: irrelevant (η noise, cancels in aggregate) │
│                                                   │
│  Pairs with:                                      │
│  ├─ superinstance-core (entity registration)      │
│  ├─ ternary-fleet (component discovery)           │
│  └─ ternary-svm (result classification)           │
└──────────────────────────────────────────────────┘
```

