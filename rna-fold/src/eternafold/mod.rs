//! EternaFold: ML-based RNA folding trained on experimental data
//!
//! ## Algorithm Overview
//!
//! Unlike traditional thermodynamic models, EternaFold uses **machine learning**
//! to predict RNA structures based on experimental data from the Eterna project.
//!
//! ### Key Innovations
//!
//! 1. **Multitask Learning**: Trains on three objectives simultaneously
//!    - Structure prediction (base pairing)
//!    - Chemical probing data (SHAPE)
//!    - Binding affinity prediction
//!
//! 2. **Experimental Data**: Uses real folding data from citizen scientists
//!    - 100,000+ RNA designs from Eterna players
//!    - Actual chemical probing experiments
//!    - Much more diverse than traditional datasets
//!
//! 3. **CONTRAfold-SE Base**: Built on discriminative model, not thermodynamic
//!    - Learns parameters from data, not physics
//!    - Can capture non-thermodynamic effects
//!
//! ## This Implementation
//!
//! We'll build a simplified version that demonstrates:
//! - Feature extraction from RNA sequences
//! - Parameter-based scoring (simplified ML model)
//! - Dynamic programming for structure prediction
//! - Partition function calculations
//!
//! A full implementation would include:
//! - Complete CONTRAfold feature set (hundreds of features)
//! - Gradient-based parameter learning
//! - SHAPE data integration
//! - Binding affinity prediction

mod features;
mod dp;

pub use features::FeatureExtractor;
pub use dp::DynamicProgramming;

use crate::common::{RnaSequence, SecondaryStructure, BasePair};

/// EternaFold predictor
///
/// Rust learning: This struct holds the learned parameters
#[derive(Debug, Clone)]
pub struct EternaFold {
    /// Model parameters (weights for features)
    /// In a real implementation, these would be learned from data
    parameters: Parameters,
}

/// Model parameters
///
/// Rust learning: Struct with named fields (like a C++ struct or Python dataclass)
#[derive(Debug, Clone)]
struct Parameters {
    /// Weight for Watson-Crick pairs
    watson_crick_weight: f64,

    /// Weight for GU wobble pairs
    wobble_weight: f64,

    /// Weight for stacking (adjacent pairs)
    stacking_weight: f64,

    /// Weight for hairpin loops
    hairpin_weight: f64,

    /// Weight for bulge loops
    bulge_weight: f64,

    /// Weight for internal loops
    internal_weight: f64,

    /// Weight for multi-loops
    multi_weight: f64,

    /// Bonus for terminal AU pairs (they're less stable)
    terminal_au_penalty: f64,
}

impl Parameters {
    /// Create default parameters
    ///
    /// In a real EternaFold, these would be learned from training data
    /// These values are simplified approximations
    fn default() -> Self {
        Parameters {
            watson_crick_weight: -3.0,  // Strong bonus for GC/AU pairs
            wobble_weight: -1.5,          // Weaker bonus for GU pairs
            stacking_weight: -2.0,        // Bonus for stacking
            hairpin_weight: 0.5,          // Reduced penalty for hairpins
            bulge_weight: 1.0,
            internal_weight: 1.0,
            multi_weight: 1.5,
            terminal_au_penalty: 0.3,     // Small penalty for terminal AU
        }
    }
}

impl EternaFold {
    /// Create a new EternaFold predictor with default parameters
    pub fn new() -> Self {
        EternaFold {
            parameters: Parameters::default(),
        }
    }

    /// Create with custom parameters
    ///
    /// This would be used after training on data
    pub fn with_parameters(parameters: Parameters) -> Self {
        EternaFold { parameters }
    }

    /// Predict RNA secondary structure using dynamic programming
    ///
    /// Rust learning: `&self` borrows the predictor, `&RnaSequence` borrows the input
    pub fn fold(&self, sequence: &RnaSequence) -> SecondaryStructure {
        let n = sequence.len();

        if n == 0 {
            return SecondaryStructure::new(sequence.clone(), vec![], 0.0);
        }

        // DP table: dp[i][j] = best energy for subsequence from i to j
        // Rust learning: vec![vec![0.0; n]; n] creates a 2D vector (like a matrix)
        let mut dp = vec![vec![0.0; n]; n];

        // Traceback table to reconstruct structure
        // Option<(usize, usize)> can be None or Some((i, j))
        let mut trace = vec![vec![None; n]; n];

        // Fill DP table
        // Process all possible subsequences in increasing order of length
        for length in 4..=n {  // Minimum length 4 for a hairpin
            for i in 0..=n.saturating_sub(length) {
                let j = i + length - 1;

                // Option 1: i is unpaired
                if i + 1 <= j {
                    let energy = dp[i + 1][j];
                    if energy < dp[i][j] {
                        dp[i][j] = energy;
                        trace[i][j] = None;
                    }
                }

                // Option 2: i pairs with some k in (i, j]
                for k in i + 1..=j {
                    if self.can_pair(sequence, i, k) {
                        let pair_energy = self.pair_score(sequence, i, k);

                        let mut total_energy = pair_energy;

                        // Add energy from unpaired regions
                        if i + 1 < k {
                            total_energy += dp[i + 1][k - 1];
                        }
                        if k + 1 <= j {
                            total_energy += dp[k + 1][j];
                        }

                        if total_energy < dp[i][j] {
                            dp[i][j] = total_energy;
                            trace[i][j] = Some((i, k));
                        }
                    }
                }
            }
        }

        // Traceback to reconstruct structure
        let pairs = self.traceback(sequence, &trace, 0, n - 1);

        SecondaryStructure::new(sequence.clone(), pairs, dp[0][n - 1])
    }

    /// Check if two positions can form a base pair
    fn can_pair(&self, sequence: &RnaSequence, i: usize, j: usize) -> bool {
        const MIN_LOOP_LENGTH: usize = 3;

        if j <= i + MIN_LOOP_LENGTH {
            return false;
        }

        if let (Some(base_i), Some(base_j)) = (sequence.get(i), sequence.get(j)) {
            base_i.can_pair(base_j)
        } else {
            false
        }
    }

    /// Score a base pair using learned parameters
    ///
    /// This is where the "machine learning" happens (in a simplified way)
    fn pair_score(&self, sequence: &RnaSequence, i: usize, j: usize) -> f64 {
        let base_i = sequence.get(i).unwrap();
        let base_j = sequence.get(j).unwrap();

        let mut score = 0.0;

        // Feature 1: Type of base pair
        if base_i.can_pair_watson_crick(base_j) {
            score += self.parameters.watson_crick_weight;
        } else if base_i.can_pair_wobble(base_j) {
            score += self.parameters.wobble_weight;
        }

        // Feature 2: Stacking (are the adjacent positions also paired?)
        // This would be checked during traceback in a full implementation

        // Feature 3: Terminal AU penalty
        use crate::common::Base;
        if matches!((base_i, base_j), (Base::A, Base::U) | (Base::U, Base::A)) {
            score += self.parameters.terminal_au_penalty;
        }

        // Feature 4: Hairpin loop size
        let loop_size = j - i - 1;
        if loop_size >= 3 {
            score += self.parameters.hairpin_weight * self.loop_score(loop_size);
        }

        score
    }

    /// Score a loop based on its size
    fn loop_score(&self, size: usize) -> f64 {
        match size {
            0..=2 => 10.0,     // Too small
            3 => 5.0,          // Minimum
            4..=6 => 1.0,      // Optimal
            7..=10 => 1.5,     // Good
            _ => 2.0 + (size as f64 - 10.0) * 0.1,  // Linear penalty
        }
    }

    /// Traceback to reconstruct base pairs from DP table
    ///
    /// Rust learning: This is a recursive function that uses pattern matching
    fn traceback(
        &self,
        _sequence: &RnaSequence,
        trace: &[Vec<Option<(usize, usize)>>],
        i: usize,
        j: usize,
    ) -> Vec<BasePair> {
        if i >= j {
            return vec![];
        }

        let mut pairs = vec![];

        if let Some((pair_i, pair_k)) = trace[i][j] {
            // Found a pair
            pairs.push(BasePair::new(pair_i, pair_k));

            // Recurse on unpaired regions
            if pair_i + 1 < pair_k {
                pairs.extend(self.traceback(_sequence, trace, pair_i + 1, pair_k - 1));
            }
            if pair_k + 1 <= j {
                pairs.extend(self.traceback(_sequence, trace, pair_k + 1, j));
            }
        } else if i + 1 <= j {
            // No pair at i, continue
            pairs.extend(self.traceback(_sequence, trace, i + 1, j));
        }

        pairs
    }

    /// Calculate partition function (sum over all structures)
    ///
    /// This is used for base pair probability calculations
    /// Similar to DP but sums probabilities instead of finding minimum
    pub fn partition_function(&self, sequence: &RnaSequence) -> f64 {
        let n = sequence.len();
        if n == 0 {
            return 1.0;
        }

        // Simplified version - full implementation would use inside-outside algorithm
        let structure = self.fold(sequence);
        (-structure.energy).exp()  // Boltzmann weight
    }
}

impl Default for EternaFold {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_eternafold_simple() {
        let seq = RnaSequence::from_str("GGGGAAAACCCC").unwrap();
        let folder = EternaFold::new();
        let structure = folder.fold(&seq);

        println!("{}", structure);
        assert!(!structure.pairs.is_empty());
    }

    #[test]
    fn test_partition_function() {
        let seq = RnaSequence::from_str("GCGC").unwrap();
        let folder = EternaFold::new();
        let z = folder.partition_function(&seq);

        assert!(z > 0.0, "Partition function should be positive");
    }
}
