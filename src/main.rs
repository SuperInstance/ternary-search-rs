//! Ternary vector search server — high-performance Rust replacement for the Python semantic search server.

mod vectors;
mod concepts;

use std::collections::HashMap;
use std::net::SocketAddr;
use std::sync::Arc;

use axum::{
    extract::{Path, Query, State},
    http::StatusCode,
    routing::get,
    Json, Router,
};
use clap::Parser;
use serde::{Deserialize, Serialize};
use tower_http::cors::CorsLayer;
use tracing_subscriber;

use concepts::ConceptAnalysis;
use vectors::{SearchHit, VectorStore, DIM};

/// CLI args.
#[derive(Parser, Debug)]
#[command(name = "ternary-search", about = "High-performance ternary vector search server")]
struct Args {
    /// Path to fleet_embeddings.ndjson
    #[arg(long, env = "VECTORS_FILE")]
    vectors: Option<String>,

    /// Path to concept_analysis.json (optional)
    #[arg(long, env = "ANALYSIS_FILE")]
    analysis: Option<String>,

    /// Port to listen on
    #[arg(long, env = "PORT", default_value = "7777")]
    port: u16,

    /// Concept boost multiplier for guided search
    #[arg(long, env = "CONCEPT_BOOST", default_value = "1.15")]
    concept_boost: f32,

    /// Number of Tokio worker threads (0 = auto)
    #[arg(long, env = "WORKER_THREADS", default_value = "0")]
    worker_threads: usize,
}

/// Shared application state.
#[derive(Clone)]
struct AppState {
    store: Arc<VectorStore>,
    analysis: Arc<ConceptAnalysis>,
    concept_boost: f32,
}

/// Query params for /search.
#[derive(Deserialize)]
struct SearchQuery {
    q: Option<String>,
    k: Option<usize>,
}

/// Search response.
#[derive(Serialize)]
struct SearchResponse {
    query: String,
    concept: Option<String>,
    concept_confidence: f32,
    results: Vec<SearchHit>,
    cross_pollination: Vec<SearchHit>,
    search_time_ms: f64,
    total_crates: usize,
}

/// /concepts response.
#[derive(Serialize)]
struct ConceptsResponse {
    concepts: HashMap<String, vectors::ConceptInfo>,
}

/// /concept/:name response.
#[derive(Serialize)]
struct ConceptDetail {
    concept: String,
    count: usize,
    members_sample: Vec<String>,
}

/// /cross response.
#[derive(Serialize)]
struct CrossResponse {
    cross_pollination: Vec<concepts::CrossPair>,
}

/// /frontier response.
#[derive(Serialize)]
struct FrontierResponse {
    frontier: Vec<concepts::FrontierEntry>,
}

/// /stats response.
#[derive(Serialize)]
struct StatsResponse {
    total_crates: usize,
    dimensions: usize,
    concepts: usize,
    engine: &'static str,
}

/// Parse a query string into 384 floats.
/// Accepts comma-separated values or raw float array.
fn parse_query_vector(q: &str) -> Option<[f32; DIM]> {
    // Try parsing as comma-separated f32 values
    let parts: Vec<&str> = q.split(',').collect();
    if parts.len() != DIM {
        return None;
    }
    let mut arr = [0.0f32; DIM];
    for (i, p) in parts.iter().enumerate() {
        arr[i] = p.trim().parse().ok()?;
    }
    Some(arr)
}

/// GET /search?q=...&k=10
async fn search_handler(
    State(state): State<AppState>,
    Query(params): Query<SearchQuery>,
) -> Result<Json<SearchResponse>, (StatusCode, String)> {
    let q = params.q.ok_or((
        StatusCode::BAD_REQUEST,
        "Missing ?q= parameter".into(),
    ))?;

    let top_k = params.k.unwrap_or(10).clamp(1, 100);

    // Parse the query vector — q is expected to be comma-separated f32 values
    // This is the pre-computed embedding from the client (BGE model output)
    let query_vec = parse_query_vector(&q).ok_or_else(|| {
        (
            StatusCode::BAD_REQUEST,
            format!(
                "Expected {} comma-separated float values, got: {}",
                DIM,
                if q.len() > 80 { &q[..80] } else { &q }
            ),
        )
    })?;

    let start = std::time::Instant::now();

    let (scored, concept_info) = state.store.search(&query_vec, top_k, state.concept_boost);

    let concept_name = concept_info.as_ref().map(|(c, _)| c.clone());
    let concept_conf = concept_info.as_ref().map(|(_, s)| *s).unwrap_or(0.0);

    // Build primary results
    let results: Vec<SearchHit> = scored
        .iter()
        .map(|(idx, score)| {
            let in_concept = concept_name
                .as_ref()
                .map(|c| state.store.concepts[*idx].iter().any(|x| x == c))
                .unwrap_or(false);
            state.store.hit(*idx, *score, in_concept)
        })
        .collect();

    // Cross-pollination: crates NOT in the matched concept
    let cross_raw = state.store.cross_pollination(
        &query_vec,
        concept_name.as_deref(),
        5,
    );
    let cross: Vec<SearchHit> = cross_raw
        .iter()
        .map(|(idx, score)| state.store.hit(*idx, *score, false))
        .collect();

    let elapsed = start.elapsed().as_secs_f64() * 1000.0;

    Ok(Json(SearchResponse {
        query: format!("vec[{}]", DIM),
        concept: concept_name,
        concept_confidence: (concept_conf * 10000.0).round() / 10000.0,
        results,
        cross_pollination: cross,
        search_time_ms: (elapsed * 10.0).round() / 10.0,
        total_crates: state.store.len,
    }))
}

/// GET /concepts
async fn concepts_handler(State(state): State<AppState>) -> Json<ConceptsResponse> {
    Json(ConceptsResponse {
        concepts: state.store.concept_info(),
    })
}

/// GET /concept/:name
async fn concept_detail_handler(
    State(state): State<AppState>,
    Path(name): Path<String>,
) -> Result<Json<ConceptDetail>, (StatusCode, String)> {
    match state.store.concept_index.get(&name) {
        Some(indices) => {
            let members_sample: Vec<String> = indices
                .iter()
                .take(50)
                .map(|&i| state.store.names[i].clone())
                .collect();
            Ok(Json(ConceptDetail {
                concept: name,
                count: indices.len(),
                members_sample,
            }))
        }
        None => Err((
            StatusCode::NOT_FOUND,
            format!("Unknown concept: {name}"),
        )),
    }
}

/// GET /cross
async fn cross_handler(State(state): State<AppState>) -> Json<CrossResponse> {
    Json(CrossResponse {
        cross_pollination: state.analysis.cross.iter().take(20).cloned().collect(),
    })
}

/// GET /frontier
async fn frontier_handler(State(state): State<AppState>) -> Json<FrontierResponse> {
    Json(FrontierResponse {
        frontier: state.analysis.frontier.iter().take(20).cloned().collect(),
    })
}

/// GET /stats
async fn stats_handler(State(state): State<AppState>) -> Json<StatsResponse> {
    Json(StatsResponse {
        total_crates: state.store.len,
        dimensions: DIM,
        concepts: state.store.concept_index.len(),
        engine: "ternary-search-rs/0.1.0 (rust+axum+rayon)",
    })
}

/// GET /healthz
async fn healthz_handler() -> &'static str {
    "ok"
}

/// GET / — API info
async fn root_handler() -> Json<serde_json::Value> {
    Json(serde_json::json!({
        "service": "SuperInstance Ternary Search",
        "version": "0.1.0",
        "endpoints": [
            "/search?q=<comma-separated-384-floats>&k=10",
            "/concepts",
            "/concept/:name",
            "/cross",
            "/frontier",
            "/stats",
            "/healthz"
        ]
    }))
}

fn main() {
    tracing_subscriber::fmt()
        .with_target(false)
        .compact()
        .init();

    let args = Args::parse();

    // Resolve vectors file path
    let vectors_path = args.vectors.as_deref().unwrap_or(
        "/home/phoenix/.openclaw/workspace/fleet_embeddings.ndjson",
    );
    let analysis_path = args.analysis.as_deref().unwrap_or(
        "/home/phoenix/.openclaw/workspace/concept_analysis.json",
    );

    tracing::info!("Loading vectors from {vectors_path}");

    // Load vectors (blocking — startup only)
    let store = VectorStore::load(std::path::Path::new(vectors_path))
        .expect("Failed to load vectors");

    tracing::info!("Loading analysis from {analysis_path}");
    let analysis = ConceptAnalysis::load_or_compute(
        Some(std::path::Path::new(analysis_path)),
        &store,
    )
    .expect("Failed to load analysis");

    let state = AppState {
        store: Arc::new(store),
        analysis: Arc::new(analysis),
        concept_boost: args.concept_boost,
    };

    // Build router
    let app = Router::new()
        .route("/", get(root_handler))
        .route("/search", get(search_handler))
        .route("/concepts", get(concepts_handler))
        .route("/concept/{name}", get(concept_detail_handler))
        .route("/cross", get(cross_handler))
        .route("/frontier", get(frontier_handler))
        .route("/stats", get(stats_handler))
        .route("/healthz", get(healthz_handler))
        .layer(CorsLayer::very_permissive())
        .with_state(state);

    let addr = SocketAddr::from(([0, 0, 0, 0], args.port));
    tracing::info!("🚀 Listening on http://localhost:{}", args.port);

    // Configure Tokio runtime
    let mut builder = tokio::runtime::Builder::new_multi_thread();
    builder.enable_all();
    if args.worker_threads > 0 {
        builder.worker_threads(args.worker_threads);
    }
    let rt = builder.build().expect("Failed to build Tokio runtime");

    rt.block_on(async {
        let listener = tokio::net::TcpListener::bind(addr)
            .await
            .expect("Failed to bind");
        axum::serve(listener, app).await.expect("Server error");
    });
}
