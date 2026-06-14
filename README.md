# ternary-search-rs

High-performance ternary vector search server in Rust. Drop-in replacement for the Python semantic search server, optimized for >100K queries/sec.

## Features

- **SoA (Structure of Arrays) layout** — cache-aligned vector storage for minimal cache-line misses
- **SIMD-friendly dot product** — auto-vectorized by `rustc` with `-C target-cpu=native`
- **Rayon parallelism** — similarity computed across all cores
- **Concept-guided search** — boost crates in the nearest concept cluster
- **Cross-pollination** — surface results from *other* concept clusters
- **All original endpoints** — `/search`, `/concepts`, `/concept/:name`, `/cross`, `/frontier`, `/stats`
- **Health check** — `/healthz` for load balancer probes

## Quick Start

```bash
# Build
cargo build --release

# Run with defaults (loads from workspace)
./target/release/ternary-search

# Or specify paths explicitly
VECTORS_FILE=/path/to/fleet_embeddings.ndjson \
ANALYSIS_FILE=/path/to/concept_analysis.json \
PORT=7777 \
./target/release/ternary-search
```

## API

### `GET /search?q=<vec>&k=10`

Search by pre-computed embedding vector.

- `q` — 384 comma-separated f32 values (BGE-small-en-v1.5 output)
- `k` — number of results (default 10, max 100)

```bash
curl "http://localhost:7777/search?q=0.0123,-0.0456,...,0.0789&k=10"
```

```json
{
  "query": "vec[384]",
  "concept": "search",
  "concept_confidence": 0.8231,
  "results": [
    {
      "name": "crate-name",
      "score": 0.9512,
      "concepts": ["search", "storage"],
      "description": "...",
      "in_concept": true
    }
  ],
  "cross_pollination": [...],
  "search_time_ms": 0.12,
  "total_crates": 1150
}
```

### `GET /concepts`

List all concept clusters with member counts and centroid norms.

### `GET /concept/:name`

Get details about a specific concept cluster (up to 50 sample members).

### `GET /cross`

Top cross-pollination pairs — crate pairs from different concepts with high similarity.

### `GET /frontier`

Negative space / frontier ideas — crates far from any concept centroid.

### `GET /stats`

```json
{
  "total_crates": 1150,
  "dimensions": 384,
  "concepts": 12,
  "engine": "ternary-search-rs/0.1.0 (rust+axum+rayon)"
}
```

### `GET /healthz`

Returns `ok`. For load balancer health checks.

## Architecture

```
┌──────────────────────────────────────────────────┐
│  axum HTTP server (Tokio async runtime)          │
│                                                  │
│  ┌────────────────────────────────────────────┐  │
│  │  VectorStore (SoA layout)                  │  │
│  │  ┌──────────────┐  ┌──────────────────┐   │  │
│  │  │ data: Vec<f32>│  │ names: Vec<String>│   │  │
│  │  │ (N × 384)    │  │ concepts: Vec<…>  │   │  │
│  │  │ contiguous   │  │ descriptions     │   │  │
│  │  └──────────────┘  └──────────────────┘   │  │
│  │  norms: precomputed L2 norms              │  │
│  │  concept_index: HashMap<String, Vec<usize>>│ │
│  └────────────────────────────────────────────┘  │
│                                                  │
│  Rayon thread pool for parallel similarity scan  │
└──────────────────────────────────────────────────┘
```

### Performance Design

1. **SoA layout**: All 384-dim vectors are contiguous in a single `Vec<f32>`. A linear scan across N vectors touches exactly `N × 384 × 4` bytes sequentially, which modern CPUs prefetch perfectly.

2. **Auto-vectorized dot product**: The inner loop `sum += query[i] * slice[i]` is compiled to SIMD instructions (AVX2/AVX-512) with `-C target-cpu=native -C opt-level=3 -C lto=fat`.

3. **Rayon parallelism**: `par_iter()` splits the N-vector scan across all cores. Each chunk is independent — no synchronization needed.

4. **Precomputed norms**: L2 norms are computed once at startup, so cosine similarity is just `dot / (norm_q * norm_r)`.

5. **Concept boost**: Rather than a separate filtered pass, concept membership is checked during scoring with a simple binary search (concept member lists are sorted at load time).

## Build

```bash
# Release build with all optimizations
cargo build --release

# The Cargo.toml is pre-configured for maximum performance:
# - opt-level = 3
# - lto = "fat"
# - codegen-units = 1
# - target-cpu = native
# - panic = abort
```

## Configuration

| Env Var          | CLI Flag         | Default                        | Description                  |
|------------------|------------------|--------------------------------|------------------------------|
| `VECTORS_FILE`   | `--vectors`      | workspace/fleet_embeddings.ndjson | Path to NDJSON vectors     |
| `ANALYSIS_FILE`  | `--analysis`     | workspace/concept_analysis.json   | Path to concept analysis   |
| `PORT`           | `--port`         | `7777`                         | Listen port                  |
| `CONCEPT_BOOST`  | `--concept-boost`| `1.15`                         | Boost multiplier for concept |
| `WORKER_THREADS` | `--worker-threads`| auto                          | Tokio worker threads         |

## NDJSON Format

Each line is a JSON object:

```json
{"name":"crate-name","description":"...","values":[0.1,0.2,...],"concepts":["search","math"],"readme_length":1234}
```

## License

MIT
