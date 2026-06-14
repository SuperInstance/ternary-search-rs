# ternary-search-rs

## The Problem

There are 1,150 repositories. Each one is an idea encoded in code — a crate with a name, a description, a README, and a 384-dimensional embedding vector that captures what it *means*. Not what it says, but what it's about. Serialization. Async runtimes. Neural network frameworks. Procedural generation. Database drivers.

How do you find the one that matters?

You could keyword-search the descriptions. Good luck — "fast async framework" matches twenty crates that have nothing in common. You could tag things manually. That doesn't scale past your patience.

What you actually want is *semantic search*: give me a vector that represents what I'm looking for, and find me the crates whose vectors point in the same direction. Cosine similarity over embedding space. Simple concept, fast in theory, and yet — the implementation details determine whether it's fast in practice.

This is a server that does that search in under a millisecond. Here's why the layout of numbers in memory matters more than the algorithm you use to compare them.

---

## The Insight: Why Layout Beats Algorithm

You have 1,150 vectors. Each one is 384 floats. That's 441,600 numbers — about 1.7 MB of float32 data. Small enough to fit in L2 cache on a modern CPU, if you lay it out right.

The obvious approach is an array of structures:

```rust
struct Crate {
    name: String,
    values: [f32; 384],
    description: String,
}
```

Each `Crate` is more than just the vector. There's a `String` pointer for the name (24 bytes), another for the description (24 bytes), the vector itself (1,536 bytes), and padding. When you iterate through `Vec<Crate>` computing dot products, the CPU loads a cache line (64 bytes), uses part of it, then jumps forward ~1,600 bytes to the next struct. Every jump is a potential cache line miss — ~100 cycles of the CPU sitting idle, waiting for RAM.

The fix is Structure of Arrays — SoA. All vectors go into a single contiguous `Vec<f32>`, end to end:

```
data: [v0_dim0, v0_dim1, ..., v0_dim383, v1_dim0, v1_dim1, ..., v1_dim383, ...]
```

Now the CPU reads sequentially. The hardware prefetcher notices the pattern and starts loading the next cache lines *before* you ask for them. No stalls. No idle cycles. The names and descriptions live in parallel arrays, accessed only when building the response — never during the hot loop.

This is the entire reason `VectorStore` exists as a separate struct. It's about one flat allocation of `1,150 × 384 = 441,600` floats, stored contiguously, scanned linearly.

For practical-scale vector search (sub-million vectors), cache layout determines speed more than algorithmic complexity. You don't need an HNSW index or an IVF forest when a brute-force scan of 1.7 MB completes in 0.12 milliseconds. The algorithm is O(n). The constant factor is what matters, and SoA makes that constant tiny.

---

## Ternary Compression: Less Is More

Most of those floats are noise. The sign is what matters. A dimension that's strongly positive contributes positively to similarity. One that's strongly negative contributes negatively. One near zero barely contributes at all. So you can quantize each float to one of three values: `{-1, 0, +1}`.

A ternary vector over 384 dimensions needs 384 trits. Two bits per trit means 96 bytes per vector instead of 1,536 — roughly 8× smaller in practice.

The quality loss is minimal. Cosine similarity is dominated by dimensions where both vectors have strong agreement or disagreement — the exact dimensions that survive ternarization. The near-zero dimensions, the ones rounded to 0, were contributing almost nothing to the dot product anyway. Empirically, ranking quality drops less than 2% while memory drops by an order of magnitude.

This server currently loads float32 vectors but is designed around the ternary philosophy: store the minimum representation that preserves ranking quality, pack it tight, and let the CPU's vector units do what they were built for. The dot-product inner loop — `for i in 0..DIM { sum += query[i] * slice[i] }` — is exactly the shape that `rustc` auto-vectorizes into tight AVX2 or AVX-512 SIMD instructions with `-C target-cpu=native`. No intrinsics. No unsafe. Just a loop the compiler turns into wide multiply-adds.

---

## The Server

The server is an axum HTTP application running on Tokio. It loads vectors at startup, builds the concept index, and then sits there answering queries.

What you type:

```
GET /search?q=0.0123,-0.0456,...,0.0789&k=10
```

That comma-separated string is a pre-computed 384-dimensional embedding — the output of `BGE-small-en-v1.5` run on whatever natural language query you started with. The server doesn't run the embedding model. It just does the search.

What you get back:

```json
{
  "concept": "search",
  "concept_confidence": 0.8231,
  "results": [
    { "name": "tantivy", "score": 0.9512, "concepts": ["search", "storage"], "in_concept": true }
  ],
  "cross_pollination": [
    { "name": "meilisearch", "score": 0.71, "in_concept": false }
  ],
  "search_time_ms": 0.1,
  "total_crates": 1150
}
```

There are a few things worth noting in that response.

**Concept classification.** Before scanning the vectors, the server computes cosine similarity between your query and every concept centroid — the average vector of all crates tagged with that concept. The nearest centroid becomes the query's concept. This isn't a filter; it's a *boost*. Crates that belong to the matched concept get their similarity score multiplied by 1.15. It's a gentle nudge, not a hard filter, so a crate from a different concept can still rank first if its similarity is genuinely higher.

**Cross-pollination.** After the top-k results, the server returns a separate list of crates that are *not* in the matched concept but still have high similarity (>0.5). These are the interesting surprises — the crate from a completely different domain that happens to solve a related problem. The async runtime that's semantically close to your search even though it's tagged "concurrency" not "networking."

The other endpoints exist for exploration rather than search:

- **`/concepts`** — every concept cluster with its member count and centroid magnitude. A map of the territory.
- **`/concept/:name`** — drill into one cluster, see its members.
- **`/cross`** — pre-computed cross-pollination pairs: crates from different concepts with unexpectedly high similarity. These are the edges where ideas from one domain bleed into another.
- **`/frontier`** — crates that are far from any concept centroid. These are the weird ones. The misfits. The crates that don't fit neatly into any cluster. Sometimes that's noise; sometimes that's the most interesting thing in the dataset.
- **`/stats`** — how many vectors, how many dimensions, how many concepts.
- **`/healthz`** — returns `ok`. For load balancers.

---

## How It Computes

When a search request arrives:

1. **Parse.** The query string is split on commas and parsed into `[f32; 384]`. Takes microseconds.
2. **Classify.** Dot product against every concept centroid (12-15 of them). Takes nanoseconds. The nearest one wins.
3. **Scan.** Rayon splits the 1,150 vectors across all available cores. Each core computes cosine similarity for its chunk: one dot product per vector, one division by precomputed norms. The inner loop is auto-vectorized SIMD. Total wall time: ~0.1ms on a 4-core machine.
4. **Boost.** Crates in the matched concept get a ×1.15 multiplier. Concept membership is checked via binary search over a sorted index.
5. **Select.** `select_nth_unstable` partitions the top-k in O(n) without a full sort. Then the top-k slice is sorted by score.
6. **Cross-pollinate.** A second parallel scan filters out concept members and collects anything above 0.5 similarity.
7. **Serialize.** Results are serialized to JSON via serde and sent back over HTTP.

The whole pipeline, from TCP accept to response sent, completes in under a millisecond. The search itself — step 3 — is typically 0.08–0.12ms.

---

## Parallelism

The similarity scan uses Rayon's `par_iter()`. Each of the 1,150 vectors is independent — no shared mutable state, no synchronization, no locks. Rayon's work-stealing scheduler distributes chunks across cores, and each chunk is a contiguous slice of the `data` array, so cache locality is preserved within each thread.

This is embarrassingly parallel work, and the implementation reflects that. No custom thread pool, no task graph, no NUMA awareness. Just `into_par_iter().map().collect()`. Rayon handles the rest. On a 4-core machine, the scan is ~3.5× faster than single-threaded. On an 8-core machine, ~7×. The speedup is limited by memory bandwidth, not computation — which is exactly why the SoA layout matters. Contiguous data means the prefetcher can stay ahead of the compute units.

---

## Build & Run

```bash
cargo run --release
```

That's it. The release profile in `Cargo.toml` is tuned for maximum performance:

```toml
[profile.release]
opt-level = 3
lto = "fat"
codegen-units = 1
panic = "abort"
```

`RUSTFLAGS="-C target-cpu=native"` is set in the build config to enable AVX2/AVX-512 codegen. Without it, the compiler targets a generic CPU and the auto-vectorizer is conservative. With it, the dot-product loop compiles to wide SIMD instructions and the performance difference is significant.

The server loads vectors at startup, builds the concept index, and starts listening on port 7777.

```bash
# Search
curl "http://localhost:7777/search?q=0.0123,-0.0456,...,0.0789&k=10"

# List concepts
curl "http://localhost:7777/concepts"

# Stats
curl "http://localhost:7777/stats"
```

Environment variables for configuration:

| Variable | Default | Purpose |
|---|---|---|
| `VECTORS_FILE` | workspace path | NDJSON file with embeddings |
| `ANALYSIS_FILE` | workspace path | Pre-computed concept analysis |
| `PORT` | `7777` | Listen port |
| `CONCEPT_BOOST` | `1.15` | Multiplier for concept members |
| `WORKER_THREADS` | auto | Tokio worker thread count |

---

## What This Is Not

This is not a vector database. There's no persistence layer, no incremental indexing, no approximate search, no sharding. It's a single-process, in-memory, brute-force cosine similarity engine with concept-aware result shaping.

It doesn't need to be. For 1,150 vectors at 384 dimensions, the entire dataset fits in L2 cache. An ANN index would add tree traversals, routing node distances, cache misses — and would be *slower* than scanning linearly. ANN earns its keep at millions of vectors. Below that, brute force wins if your layout is good.

This server is designed for the regime where brute force is the right answer: small enough to fit in cache, fast enough to serve in under a millisecond, simple enough to understand entirely.

---

## License

MIT
