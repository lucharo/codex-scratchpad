//! Feature extraction for EternaFold
//!
//! ## What are Features in ML-based RNA Folding?
//!
//! Unlike thermodynamic models that use physics-based energy functions,
//! machine learning models use **features** - numerical representations
//! of RNA properties that help predict structure.
//!
//! ### CONTRAfold/EternaFold Features Include:
//!
//! 1. **Base Pair Features**
//!    - Type: AU, GC, GU
//!    - Position in sequence (terminal vs internal)
//!    - Sequence context (neighboring bases)
//!
//! 2. **Loop Features**
//!    - Loop type (hairpin, bulge, internal, multi)
//!    - Loop size
//!    - Closing base pairs
//!    - Sequence composition
//!
//! 3. **Structural Features**
//!    - Stacking (adjacent pairs)
//!    - Coaxial stacking
//!    - Dangling ends
//!    - Terminal mismatches
//!
//! 4. **Experimental Features** (EternaFold specific)
//!    - SHAPE reactivity data
//!    - Binding affinity information
//!    - Multi-task learning signals
//!
//! ## This Implementation
//!
//! We implement a subset of features for educational purposes.
//! A full EternaFold has hundreds of feature templates.

use crate::common::{RnaSequence, Base, BasePair};

/// Feature vector for a structure
///
/// Rust learning: Using a HashMap would be more flexible,
/// but we use a struct for clarity and type safety
#[derive(Debug, Clone)]
pub struct FeatureVector {
    /// Number of AU pairs
    pub au_pairs: usize,

    /// Number of GC pairs
    pub gc_pairs: usize,

    /// Number of GU pairs
    pub gu_pairs: usize,

    /// Number of stacked pairs (adjacent pairs)
    pub stacked_pairs: usize,

    /// Number of hairpin loops
    pub hairpin_loops: usize,

    /// Total hairpin loop size
    pub hairpin_size: usize,

    /// Number of bulge loops
    pub bulge_loops: usize,

    /// Number of internal loops
    pub internal_loops: usize,

    /// Number of multi-loops
    pub multi_loops: usize,
}

impl FeatureVector {
    /// Create empty feature vector
    pub fn new() -> Self {
        FeatureVector {
            au_pairs: 0,
            gc_pairs: 0,
            gu_pairs: 0,
            stacked_pairs: 0,
            hairpin_loops: 0,
            hairpin_size: 0,
            bulge_loops: 0,
            internal_loops: 0,
            multi_loops: 0,
        }
    }

    /// Convert to a flat vector of f64 values
    ///
    /// This would be used as input to a linear model: score = weights · features
    pub fn to_vec(&self) -> Vec<f64> {
        vec![
            self.au_pairs as f64,
            self.gc_pairs as f64,
            self.gu_pairs as f64,
            self.stacked_pairs as f64,
            self.hairpin_loops as f64,
            self.hairpin_size as f64,
            self.bulge_loops as f64,
            self.internal_loops as f64,
            self.multi_loops as f64,
        ]
    }
}

impl Default for FeatureVector {
    fn default() -> Self {
        Self::new()
    }
}

/// Feature extractor
///
/// Rust learning: This is a stateless struct (no fields)
/// It just namespaces related functions
pub struct FeatureExtractor;

impl FeatureExtractor {
    /// Extract features from a sequence and base pairs
    ///
    /// Rust learning: `sequence` and `pairs` are borrowed (read-only)
    pub fn extract(sequence: &RnaSequence, pairs: &[BasePair]) -> FeatureVector {
        let mut features = FeatureVector::new();

        // Count pair types
        for pair in pairs {
            if let (Some(base_i), Some(base_j)) = (sequence.get(pair.i), sequence.get(pair.j)) {
                match (base_i, base_j) {
                    (Base::A, Base::U) | (Base::U, Base::A) => features.au_pairs += 1,
                    (Base::G, Base::C) | (Base::C, Base::G) => features.gc_pairs += 1,
                    (Base::G, Base::U) | (Base::U, Base::G) => features.gu_pairs += 1,
                    _ => {}
                }
            }
        }

        // Count stacked pairs
        features.stacked_pairs = Self::count_stacked_pairs(pairs);

        // Analyze loop structure
        let loop_info = Self::analyze_loops(sequence, pairs);
        features.hairpin_loops = loop_info.hairpin_count;
        features.hairpin_size = loop_info.hairpin_total_size;
        features.bulge_loops = loop_info.bulge_count;
        features.internal_loops = loop_info.internal_count;
        features.multi_loops = loop_info.multi_count;

        features
    }

    /// Count stacked pairs (adjacent pairs)
    ///
    /// Two pairs (i,j) and (i+1,j-1) are stacked
    fn count_stacked_pairs(pairs: &[BasePair]) -> usize {
        let mut count = 0;

        for pair1 in pairs {
            for pair2 in pairs {
                if pair1.i + 1 == pair2.i && pair1.j == pair2.j + 1 {
                    count += 1;
                }
            }
        }

        count
    }

    /// Analyze loop structure
    ///
    /// Rust learning: Returns a custom struct with loop information
    fn analyze_loops(_sequence: &RnaSequence, pairs: &[BasePair]) -> LoopInfo {
        // Simplified loop analysis
        // A full implementation would:
        // 1. Build a tree structure from base pairs
        // 2. Identify loop types by analyzing the tree
        // 3. Extract detailed loop properties

        let mut info = LoopInfo::default();

        // Simple heuristic: count hairpins as pairs with large gaps
        for pair in pairs {
            let loop_size = pair.j - pair.i - 1;

            // Check if any pair is inside this one
            let has_inner_pair = pairs.iter().any(|other| {
                other.i > pair.i && other.j < pair.j
            });

            if !has_inner_pair && loop_size >= 3 {
                // Likely a hairpin
                info.hairpin_count += 1;
                info.hairpin_total_size += loop_size;
            }
        }

        info
    }
}

/// Information about loops in a structure
#[derive(Debug, Clone, Default)]
struct LoopInfo {
    hairpin_count: usize,
    hairpin_total_size: usize,
    bulge_count: usize,
    internal_count: usize,
    multi_count: usize,
}

/// Feature template system (advanced)
///
/// In a full CONTRAfold/EternaFold implementation, features are generated
/// from templates that capture complex patterns.
///
/// For example, a template might be:
/// "Hairpin loop of size 4 closed by GC pair with sequence GAAA"
///
/// The model learns weights for thousands of such templates.
#[allow(dead_code)]
pub struct FeatureTemplate {
    name: String,
    // Template matching logic would go here
}

impl FeatureTemplate {
    #[allow(dead_code)]
    pub fn new(name: &str) -> Self {
        FeatureTemplate {
            name: name.to_string(),
        }
    }

    // Methods to match templates against structures would go here
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_feature_extraction() {
        let seq = RnaSequence::from_str("GGGGAAAACCCC").unwrap();

        // Simple hairpin structure
        let pairs = vec![
            BasePair::new(0, 11),
            BasePair::new(1, 10),
            BasePair::new(2, 9),
            BasePair::new(3, 8),
        ];

        let features = FeatureExtractor::extract(&seq, &pairs);

        assert_eq!(features.gc_pairs, 4);
        assert!(features.hairpin_loops > 0);
    }

    #[test]
    fn test_stacked_pairs() {
        let pairs = vec![
            BasePair::new(0, 10),
            BasePair::new(1, 9),
            BasePair::new(2, 8),
        ];

        let count = FeatureExtractor::count_stacked_pairs(&pairs);
        assert_eq!(count, 2);  // (0,10)-(1,9) and (1,9)-(2,8)
    }
}
