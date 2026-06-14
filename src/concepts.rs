//! Concept clustering, cross-pollination pairs, and frontier (negative space).
//! Precomputed at startup from the analysis file or computed from scratch.

use rayon::prelude::*;
use serde::{Deserialize, Serialize};


use crate::vectors::{VectorStore, DIM};

/// A cross-pollination pair from the analysis file.
#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct CrossPair {
    pub a: String,
    pub b: String,
    pub similarity: f64,
    pub concepts_a: Vec<String>,
    pub concepts_b: Vec<String>,
}

/// A frontier (negative space) entry from the analysis file.
#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct FrontierEntry {
    pub name: String,
    pub distance: f64,
    pub nearest_concept: String,
    pub concepts: Vec<String>,
}

/// Analysis data loaded from concept_analysis.json.
#[derive(Debug, Deserialize, Default)]
pub struct AnalysisFile {
    #[serde(default)]
    pub cross_pollination: Vec<CrossPair>,
    #[serde(default, rename = "negative_space")]
    pub frontier: Vec<FrontierEntry>,
}

/// The combined concept analysis structure.
pub struct ConceptAnalysis {
    pub cross: Vec<CrossPair>,
    pub frontier: Vec<FrontierEntry>,
}

impl ConceptAnalysis {
    /// Load from concept_analysis.json. If the file doesn't exist, compute cross-pollination
    /// from the vector store.
    pub fn load_or_compute(
        path: Option<&std::path::Path>,
        store: &VectorStore,
    ) -> Result<Self, String> {
        if let Some(p) = path {
            if p.exists() {
                let content = std::fs::read_to_string(p)
                    .map_err(|e| format!("read {}: {e}", p.display()))?;
                let file: AnalysisFile =
                    serde_json::from_str(&content).map_err(|e| format!("parse analysis: {e}"))?;
                tracing::info!(
                    "Loaded analysis: {} cross pairs, {} frontier entries",
                    file.cross_pollination.len(),
                    file.frontier.len()
                );
                return Ok(Self {
                    cross: file.cross_pollination,
                    frontier: file.frontier,
                });
            }
        }

        // Compute cross-pollination from scratch
        tracing::info!("Computing cross-pollination pairs from vector store...");
        let cross = Self::compute_cross(store);
        tracing::info!("Found {} cross-pollination pairs", cross.len());

        Ok(Self {
            cross,
            frontier: Vec::new(),
        })
    }

    /// Compute top cross-pollination pairs: highest similarity across different concepts.
    fn compute_cross(store: &VectorStore) -> Vec<CrossPair> {
        // For each concept pair, find the most similar crate pair
        let concepts: Vec<&String> = store.concept_centroids.keys().collect();
        let mut pairs: Vec<CrossPair> = Vec::new();

        for i in 0..concepts.len() {
            for j in (i + 1)..concepts.len() {
                let members_a = store.concept_index.get(concepts[i]);
                let members_b = store.concept_index.get(concepts[j]);
                if members_a.is_none() || members_b.is_none() {
                    continue;
                }
                let members_a = members_a.unwrap();
                let members_b = members_b.unwrap();

                // Find best cross pair between these two concepts
                let best = members_a
                    .par_iter()
                    .map(|&a| {
                        // For each a, find best b in members_b
                        let mut best_b = (0usize, 0.0f32);
                        for &b in members_b {
                            let sim: f32 = (0..DIM)
                                .map(|d| store.data[a * DIM + d] * store.data[b * DIM + d])
                                .sum();
                            if sim > best_b.1 {
                                best_b = (b, sim);
                            }
                        }
                        (a, best_b.0, best_b.1)
                    })
                    .max_by(|x, y| {
                        x.2.partial_cmp(&y.2).unwrap_or(std::cmp::Ordering::Equal)
                    });

                if let Some((a, b, sim)) = best {
                    if sim > 0.5 {
                        pairs.push(CrossPair {
                            a: store.names[a].clone(),
                            b: store.names[b].clone(),
                            similarity: (sim as f64 * 10000.0).round() / 10000.0,
                            concepts_a: store.concepts[a].iter().take(4).cloned().collect(),
                            concepts_b: store.concepts[b].iter().take(4).cloned().collect(),
                        });
                    }
                }
            }
        }

        pairs.sort_by(|a, b| b.similarity.partial_cmp(&a.similarity).unwrap_or(std::cmp::Ordering::Equal));
        pairs
    }
}
