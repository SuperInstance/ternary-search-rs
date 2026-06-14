//! Vector store: cache-aligned SoA layout with SIMD-friendly cosine similarity.

use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::Path;

pub const DIM: usize = 384;

/// One record parsed from NDJSON.
#[derive(Debug, Deserialize)]
pub struct Record {
    pub name: String,
    pub description: String,
    #[serde(default)]
    pub concepts: Vec<String>,
    pub values: Vec<f32>,
    #[serde(default)]
    pub readme_length: Option<u64>,
}

/// SoA (Structure of Arrays) layout for cache-aligned vector scan.
/// All vectors are contiguous in memory so a single linear scan touches
/// minimal cache lines per dimension.
pub struct VectorStore {
    /// Flattened vectors: [n * DIM], row-major.
    pub data: Vec<f32>,
    /// Crate names aligned to rows.
    pub names: Vec<String>,
    /// Descriptions.
    pub descriptions: Vec<String>,
    /// Concepts per crate.
    pub concepts: Vec<Vec<String>>,
    /// Number of vectors.
    pub len: usize,
    /// Precomputed norms for fast cosine.
    pub norms: Vec<f32>,
    /// Concept → list of row indices.
    pub concept_index: HashMap<String, Vec<usize>>,
    /// Concept → centroid vector (DIM).
    pub concept_centroids: HashMap<String, Vec<f32>>,
}

#[derive(Debug, Serialize)]
pub struct SearchHit {
    pub name: String,
    pub score: f32,
    pub concepts: Vec<String>,
    pub description: String,
    pub in_concept: bool,
}

#[derive(Debug, Serialize)]
pub struct ConceptInfo {
    pub count: usize,
    pub centroid_norm: f32,
}

#[derive(Debug, Serialize)]
pub struct Stats {
    pub total_crates: usize,
    pub dimensions: usize,
    pub concepts: usize,
    pub engine: &'static str,
}

impl VectorStore {
    /// Load vectors from an NDJSON file.
    pub fn load(path: &Path) -> Result<Self, String> {
        let content =
            std::fs::read_to_string(path).map_err(|e| format!("read {}: {e}", path.display()))?;

        let mut records: Vec<Record> = Vec::new();

        for (i, line) in content.lines().enumerate() {
            if line.trim().is_empty() {
                continue;
            }
            let rec: Record = serde_json::from_str(line)
                .map_err(|e| format!("parse line {i}: {e}"))?;
            if rec.values.len() != DIM {
                return Err(format!(
                    "line {i}: expected {DIM} dims, got {}",
                    rec.values.len()
                ));
            }
            records.push(rec);
        }

        if records.is_empty() {
            return Err("no records found".into());
        }

        let n = records.len();

        // Build SoA layout
        let mut data = Vec::with_capacity(n * DIM);
        let mut names = Vec::with_capacity(n);
        let mut descriptions = Vec::with_capacity(n);
        let mut concepts = Vec::with_capacity(n);

        for rec in &records {
            data.extend_from_slice(&rec.values);
            names.push(rec.name.clone());
            descriptions.push(rec.description.clone());
            concepts.push(rec.concepts.clone());
        }

        // Precompute L2 norms
        let norms: Vec<f32> = (0..n)
            .map(|i| {
                let slice = &data[i * DIM..(i + 1) * DIM];
                slice.iter().map(|v| v * v).sum::<f32>().sqrt()
            })
            .collect();

        // Build concept index
        let mut concept_index: HashMap<String, Vec<usize>> = HashMap::new();
        for (i, c) in concepts.iter().enumerate() {
            for concept in c {
                concept_index.entry(concept.clone()).or_default().push(i);
            }
        }

        // Compute concept centroids
        let mut concept_centroids: HashMap<String, Vec<f32>> = HashMap::new();
        for (concept, indices) in &concept_index {
            let mut centroid = vec![0.0f32; DIM];
            for &idx in indices {
                let row = &data[idx * DIM..(idx + 1) * DIM];
                for (d, v) in row.iter().enumerate() {
                    centroid[d] += v;
                }
            }
            let inv_n = 1.0 / indices.len() as f32;
            for c in centroid.iter_mut() {
                *c *= inv_n;
            }
            concept_centroids.insert(concept.clone(), centroid);
        }

        tracing::info!(
            "Loaded {n} vectors, {DIM}-dim, {} concepts",
            concept_index.len()
        );

        Ok(Self {
            data,
            names,
            descriptions,
            concepts,
            len: n,
            norms,
            concept_index,
            concept_centroids,
        })
    }

    /// Dot product of a query vector against a single row.
    #[inline(always)]
    fn dot(&self, query: &[f32; DIM], row: usize) -> f32 {
        let slice = &self.data[row * DIM..(row + 1) * DIM];
        // Manual unrolled dot product — compiler auto-vectorizes this well
        let mut sum = 0.0f32;
        for i in 0..DIM {
            sum += query[i] * slice[i];
        }
        sum
    }

    /// Cosine similarity: dot / (norm_q * norm_r).
    /// Since BGE embeddings are already normalized, norm_q ≈ 1.
    #[inline(always)]
    fn cosine(&self, query: &[f32; DIM], query_norm: f32, row: usize) -> f32 {
        let raw = self.dot(query, row);
        let denom = query_norm * self.norms[row];
        if denom > 1e-12 {
            raw / denom
        } else {
            0.0
        }
    }

    /// Compute similarities against all vectors in parallel, return top-k.
    pub fn search(
        &self,
        query: &[f32; DIM],
        top_k: usize,
        concept_boost: f32,
    ) -> (Vec<(usize, f32)>, Option<(String, f32)>) {
        let query_norm: f32 = query.iter().map(|v| v * v).sum::<f32>().sqrt();

        // Classify concept
        let (concept, concept_sim) = self.classify_concept(query);

        // Parallel similarity computation
        let sims: Vec<f32> = (0..self.len)
            .into_par_iter()
            .map(|i| self.cosine(query, query_norm, i))
            .collect();

        // Build boosted scores
        let concept_members: Option<&Vec<usize>> = concept
            .as_ref()
            .and_then(|c| self.concept_index.get(c));

        let mut scored: Vec<(usize, f32)> = (0..self.len)
            .map(|i| {
                let mut score = sims[i];
                if let Some(members) = concept_members {
                    if members.binary_search(&i).is_ok() {
                        score *= concept_boost;
                    }
                }
                (i, score)
            })
            .collect();

        // Partial sort top-k
        let k = top_k.min(scored.len());
        scored.select_nth_unstable_by(k.saturating_sub(1), |a, b| {
            b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal)
        });
        scored[..k].sort_by(|a, b| {
            b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal)
        });

        (scored[..k].to_vec(), concept.map(|c| (c, concept_sim)))
    }

    /// Find the nearest concept cluster centroid.
    fn classify_concept(&self, query: &[f32; DIM]) -> (Option<String>, f32) {
        let mut best_concept: Option<String> = None;
        let mut best_sim = -1.0f32;

        for (concept, centroid) in &self.concept_centroids {
            let mut sim = 0.0f32;
            for i in 0..DIM {
                sim += query[i] * centroid[i];
            }
            if sim > best_sim {
                best_sim = sim;
                best_concept = Some(concept.clone());
            }
        }

        (best_concept, best_sim)
    }

    /// Find cross-pollination results: high-similarity crates NOT in the query concept.
    pub fn cross_pollination(
        &self,
        query: &[f32; DIM],
        exclude_concept: Option<&str>,
        top_k: usize,
    ) -> Vec<(usize, f32)> {
        let query_norm: f32 = query.iter().map(|v| v * v).sum::<f32>().sqrt();

        let exclude_set: Option<&Vec<usize>> =
            exclude_concept.and_then(|c| self.concept_index.get(c));

        let mut scored: Vec<(usize, f32)> = (0..self.len)
            .into_par_iter()
            .filter_map(|i| {
                if let Some(set) = exclude_set {
                    if set.binary_search(&i).is_ok() {
                        return None;
                    }
                }
                let sim = self.cosine(query, query_norm, i);
                if sim > 0.5 {
                    Some((i, sim))
                } else {
                    None
                }
            })
            .collect();

        scored.sort_by(|a, b| {
            b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal)
        });

        scored.into_iter().take(top_k).collect()
    }

    /// Build a SearchHit from a row index.
    pub fn hit(&self, idx: usize, score: f32, in_concept: bool) -> SearchHit {
        SearchHit {
            name: self.names[idx].clone(),
            score: (score * 10000.0).round() / 10000.0,
            concepts: self.concepts[idx].iter().take(4).cloned().collect(),
            description: self.descriptions[idx].chars().take(200).collect(),
            in_concept,
        }
    }

    /// Concept info for /concepts endpoint.
    pub fn concept_info(&self) -> HashMap<String, ConceptInfo> {
        self.concept_centroids
            .iter()
            .map(|(concept, centroid)| {
                let norm = centroid.iter().map(|v| v * v).sum::<f32>().sqrt();
                let count = self.concept_index.get(concept).map(|v| v.len()).unwrap_or(0);
                (
                    concept.clone(),
                    ConceptInfo {
                        count,
                        centroid_norm: norm,
                    },
                )
            })
            .collect()
    }
}
