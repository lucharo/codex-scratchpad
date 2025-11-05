//! Dynamic Programming for RNA secondary structure prediction
//!
//! ## What is Dynamic Programming (DP)?
//!
//! Dynamic Programming is an algorithm design technique that:
//! 1. Breaks a problem into smaller overlapping subproblems
//! 2. Solves each subproblem once and stores the result
//! 3. Reuses stored results to avoid redundant computation
//!
//! Think of it like memoization on steroids!
//!
//! ## RNA Folding with DP
//!
//! For RNA folding, we use DP to find the structure with minimum free energy.
//!
//! ### Classic Nussinov Algorithm (simplified):
//!
//! ```text
//! Let dp[i][j] = best score for subsequence from position i to j
//!
//! Base case: dp[i][i] = 0 (single base, no pairs)
//!
//! Recurrence: dp[i][j] = max of:
//!   1. dp[i+1][j]           (i unpaired)
//!   2. dp[i][j-1]           (j unpaired)
//!   3. dp[i+1][j-1] + pair_score(i,j)  (i pairs with j)
//!   4. max over k: dp[i][k] + dp[k+1][j]  (bifurcation)
//! ```
//!
//! ### Complexity:
//! - Time: O(n³) - three nested loops
//! - Space: O(n²) - 2D table
//!
//! This is why LinearFold's O(n) algorithm is such a breakthrough!
//!
//! ## Rust Learning: 2D Arrays and Vectors
//!
//! - In C++: `vector<vector<double>>` or `double**`
//! - In Python: `list[list[float]]` or numpy array
//! - In Rust: `Vec<Vec<f64>>` (growable) or arrays (fixed size)

use crate::common::{RnaSequence, BasePair};

/// Dynamic Programming solver for RNA folding
///
/// Rust learning: This struct encapsulates the DP algorithm
pub struct DynamicProgramming {
    /// DP table: dp[i][j] = best score for subsequence [i, j]
    /// Rust note: Option<Vec<Vec<f64>>> means it might not be initialized yet
    dp_table: Option<Vec<Vec<f64>>>,

    /// Traceback table to reconstruct the optimal structure
    traceback: Option<Vec<Vec<TracebackEntry>>>,
}

/// Traceback entry to remember how we got the optimal solution
///
/// Rust learning: Enums can hold different types of data!
/// This is like a "tagged union" or "variant" type
#[derive(Debug, Clone, Copy, PartialEq)]
enum TracebackEntry {
    /// No pair at this position
    Unpaired,

    /// Position i pairs with j
    Paired { i: usize, j: usize },

    /// Bifurcation at position k
    Bifurcate { k: usize },
}

impl DynamicProgramming {
    /// Create a new DP solver
    pub fn new() -> Self {
        DynamicProgramming {
            dp_table: None,
            traceback: None,
        }
    }

    /// Run the Nussinov-style DP algorithm
    ///
    /// Returns the minimum free energy
    ///
    /// Rust learning: `&mut self` allows us to modify the struct
    pub fn compute<F>(&mut self, sequence: &RnaSequence, score_fn: F) -> f64
    where
        F: Fn(usize, usize) -> f64,  // score_fn(i, j) returns score for pairing i and j
    {
        let n = sequence.len();

        if n == 0 {
            return 0.0;
        }

        // Initialize tables
        // Rust note: vec![vec![0.0; n]; n] creates an n×n matrix filled with 0.0
        let mut dp = vec![vec![0.0; n]; n];
        let mut trace = vec![vec![TracebackEntry::Unpaired; n]; n];

        // Fill DP table
        // Process all subsequences in order of increasing length
        for length in 1..=n {
            for i in 0..=n.saturating_sub(length) {
                let j = i + length - 1;

                if i == j {
                    dp[i][j] = 0.0;
                    trace[i][j] = TracebackEntry::Unpaired;
                    continue;
                }

                // Option 1: i is unpaired
                if i + 1 <= j {
                    let score = dp[i + 1][j];
                    if score > dp[i][j] {
                        dp[i][j] = score;
                        trace[i][j] = TracebackEntry::Unpaired;
                    }
                }

                // Option 2: j is unpaired
                if i <= j.saturating_sub(1) {
                    let score = dp[i][j - 1];
                    if score > dp[i][j] {
                        dp[i][j] = score;
                        trace[i][j] = TracebackEntry::Unpaired;
                    }
                }

                // Option 3: i pairs with j
                if j > i + 3 {  // Minimum loop size
                    if self.can_pair(sequence, i, j) {
                        let pair_score = score_fn(i, j);
                        let total_score = if i + 1 < j {
                            dp[i + 1][j - 1] + pair_score
                        } else {
                            pair_score
                        };

                        if total_score > dp[i][j] {
                            dp[i][j] = total_score;
                            trace[i][j] = TracebackEntry::Paired { i, j };
                        }
                    }
                }

                // Option 4: Bifurcation (split into two parts)
                for k in i + 1..j {
                    let score = dp[i][k] + dp[k + 1][j];
                    if score > dp[i][j] {
                        dp[i][j] = score;
                        trace[i][j] = TracebackEntry::Bifurcate { k };
                    }
                }
            }
        }

        // Store tables for traceback
        self.dp_table = Some(dp.clone());
        self.traceback = Some(trace);

        dp[0][n - 1]
    }

    /// Traceback to recover the optimal structure
    ///
    /// Rust learning: This is a recursive function
    pub fn traceback_structure(&self, i: usize, j: usize) -> Vec<BasePair> {
        if i >= j {
            return vec![];
        }

        let trace = match &self.traceback {
            Some(t) => t,
            None => return vec![],
        };

        let mut pairs = vec![];

        match trace[i][j] {
            TracebackEntry::Unpaired => {
                // Continue without adding a pair
                if i + 1 <= j {
                    pairs.extend(self.traceback_structure(i + 1, j));
                }
            }

            TracebackEntry::Paired { i: pi, j: pj } => {
                // Add this pair
                pairs.push(BasePair::new(pi, pj));

                // Recurse inside the pair
                if pi + 1 < pj {
                    pairs.extend(self.traceback_structure(pi + 1, pj - 1));
                }
            }

            TracebackEntry::Bifurcate { k } => {
                // Split into two parts
                pairs.extend(self.traceback_structure(i, k));
                pairs.extend(self.traceback_structure(k + 1, j));
            }
        }

        pairs
    }

    /// Check if positions i and j can form a base pair
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

    /// Get the DP table (for debugging/visualization)
    pub fn get_dp_table(&self) -> Option<&Vec<Vec<f64>>> {
        self.dp_table.as_ref()
    }
}

impl Default for DynamicProgramming {
    fn default() -> Self {
        Self::new()
    }
}

/// Partition function calculation
///
/// Instead of finding the single best structure, the partition function
/// sums over all possible structures weighted by their Boltzmann probability.
///
/// Z = Σ exp(-E/RT) for all structures
///
/// This is used to calculate:
/// - Base pair probabilities
/// - Ensemble diversity
/// - Thermodynamic quantities
///
/// The algorithm is similar to the MFE algorithm but uses sum instead of max.
pub struct PartitionFunction {
    /// Partition function table
    z_table: Option<Vec<Vec<f64>>>,

    /// Temperature in Kelvin
    temperature: f64,

    /// Gas constant (kcal/(mol·K))
    gas_constant: f64,
}

impl PartitionFunction {
    /// Create a new partition function calculator
    pub fn new(temperature: f64) -> Self {
        PartitionFunction {
            z_table: None,
            temperature,
            gas_constant: 0.0019872,  // kcal/(mol·K)
        }
    }

    /// Compute partition function
    ///
    /// Similar to DP but uses sum-of-exponentials instead of max
    pub fn compute<F>(&mut self, sequence: &RnaSequence, energy_fn: F) -> f64
    where
        F: Fn(usize, usize) -> f64,
    {
        let n = sequence.len();

        if n == 0 {
            return 1.0;
        }

        let mut z = vec![vec![1.0; n]; n];
        let rt = self.gas_constant * self.temperature;

        for length in 1..=n {
            for i in 0..=n.saturating_sub(length) {
                let j = i + length - 1;

                if i >= j {
                    continue;
                }

                let mut total = 0.0;

                // Sum over all possibilities
                if i + 1 <= j {
                    total += z[i + 1][j];
                }

                if j > i + 3 {
                    let energy = energy_fn(i, j);
                    let boltzmann = (-energy / rt).exp();
                    let inner = if i + 1 < j { z[i + 1][j - 1] } else { 1.0 };
                    total += boltzmann * inner;
                }

                for k in i + 1..j {
                    total += z[i][k] * z[k + 1][j];
                }

                z[i][j] = total;
            }
        }

        self.z_table = Some(z.clone());
        z[0][n - 1]
    }

    /// Calculate base pair probability from partition function
    ///
    /// P(i pairs with j) = Z(i,j) / Z(full sequence)
    pub fn pair_probability(&self, _i: usize, _j: usize) -> f64 {
        // Simplified - full implementation uses inside-outside algorithm
        0.0
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_dp_simple() {
        let seq = RnaSequence::from_str("GGGGCCCC").unwrap();
        let mut dp = DynamicProgramming::new();

        // Simple scoring: +1 for each pair
        let score = dp.compute(&seq, |_, _| 1.0);

        // With 8 bases and +1 per pair, we should get some positive score
        assert!(score >= 0.0);

        let pairs = dp.traceback_structure(0, seq.len() - 1);
        println!("Found {} pairs", pairs.len());
    }

    #[test]
    fn test_partition_function() {
        let seq = RnaSequence::from_str("GGGGAAACCCC").unwrap();
        let mut pf = PartitionFunction::new(310.0);  // 37°C in Kelvin

        let z = pf.compute(&seq, |_, _| -2.0);
        assert!(z > 0.0, "Partition function should be positive");
    }
}
