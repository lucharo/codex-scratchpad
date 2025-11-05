//! LinearFold: Linear-time RNA folding with beam search
//!
//! ## Algorithm Overview
//!
//! Traditional RNA folding algorithms (like Zuker's algorithm) have O(n³) time complexity
//! because they use dynamic programming that considers all possible substructures.
//!
//! **LinearFold achieves O(n) time by:**
//! 1. Processing the sequence in a single direction (5' to 3')
//! 2. Using beam search to keep only the top-k most promising states
//! 3. Pruning away unlikely structures as we go
//!
//! ## Key Innovation: Beam Search
//!
//! Instead of exploring all possible structures, we:
//! - Keep a "beam" of the best B candidates at each position (default B=100)
//! - Discard low-scoring candidates
//! - This trades some accuracy for dramatic speedup
//!
//! ## Rust Learning: Module Organization
//!
//! - `mod.rs` is the module entry point (like __init__.py in Python)
//! - Submodules can be in separate files

mod beam;
mod energy;

pub use beam::BeamSearcher;
pub use energy::SimpleEnergyModel;

use crate::common::{RnaSequence, SecondaryStructure, BasePair};
use std::collections::BinaryHeap;
use ordered_float::OrderedFloat;

/// State in the beam search
///
/// Rust learning: This struct represents a partial folding state as we process the sequence
#[derive(Debug, Clone)]
struct State {
    /// Positions that are currently unpaired (can pair in future)
    /// Rust note: Vec<usize> is like std::vector<size_t> in C++
    unpaired: Vec<usize>,

    /// Base pairs formed so far
    pairs: Vec<BasePair>,

    /// Energy score (lower is better)
    /// Rust note: f64 is a 64-bit float
    energy: f64,
}

impl State {
    /// Create an empty initial state
    fn new() -> Self {
        State {
            unpaired: Vec::new(),
            pairs: Vec::new(),
            energy: 0.0,
        }
    }

    /// Create a new state by adding an unpaired base
    fn with_unpaired(&self, position: usize) -> Self {
        let mut new_state = self.clone();
        new_state.unpaired.push(position);
        new_state
    }

    /// Create a new state by pairing the current position with an earlier unpaired position
    fn with_pair(&self, i: usize, j: usize, energy_delta: f64) -> Self {
        let mut new_state = self.clone();

        // Remove the paired position from unpaired list
        new_state.unpaired.retain(|&pos| pos != i);

        // Add the new pair
        new_state.pairs.push(BasePair::new(i, j));

        // Update energy
        new_state.energy += energy_delta;

        new_state
    }
}

/// Implement ordering for states (for the priority queue)
///
/// Rust learning: Implementing traits is how we define behavior
/// This is similar to implementing interfaces in C++ or protocols in Python
impl Ord for State {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        // Lower energy is better, so we reverse the comparison
        // Rust note: OrderedFloat makes f64 comparable (since NaN complicates things)
        OrderedFloat(other.energy).cmp(&OrderedFloat(self.energy))
    }
}

impl PartialOrd for State {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl PartialEq for State {
    fn eq(&self, other: &Self) -> bool {
        OrderedFloat(self.energy) == OrderedFloat(other.energy)
    }
}

impl Eq for State {}

/// LinearFold algorithm implementation
///
/// Rust learning: This is a "struct" that holds configuration
pub struct LinearFold {
    /// Beam size (number of states to keep)
    beam_size: usize,

    /// Energy model for scoring structures
    energy_model: SimpleEnergyModel,
}

impl LinearFold {
    /// Create a new LinearFold instance
    ///
    /// Rust learning: This is a "constructor" (associated function)
    pub fn new(beam_size: usize) -> Self {
        LinearFold {
            beam_size,
            energy_model: SimpleEnergyModel::new(),
        }
    }

    /// Create with default beam size (100)
    pub fn default() -> Self {
        Self::new(100)
    }

    /// Predict RNA secondary structure
    ///
    /// Rust learning: `&self` borrows the struct (like const this* in C++)
    /// `&RnaSequence` borrows the sequence without taking ownership
    pub fn fold(&self, sequence: &RnaSequence) -> SecondaryStructure {
        let n = sequence.len();

        // Initialize with empty state
        // Rust note: BinaryHeap is a max-heap (like std::priority_queue)
        let mut beam: BinaryHeap<State> = BinaryHeap::new();
        beam.push(State::new());

        // Process sequence from 5' to 3' (left to right)
        for j in 0..n {
            let mut next_beam = BinaryHeap::new();

            // For each state in the current beam
            while let Some(state) = beam.pop() {
                // Option 1: Leave position j unpaired
                next_beam.push(state.with_unpaired(j));

                // Option 2: Pair position j with an earlier unpaired position
                for &i in &state.unpaired {
                    // Check if bases can pair and meet minimum loop length
                    if self.can_pair(sequence, i, j) {
                        let energy_delta = self.energy_model.pair_energy(sequence, i, j);
                        next_beam.push(state.with_pair(i, j, energy_delta));
                    }
                }
            }

            // Prune beam to keep only top candidates
            beam = self.prune_beam(next_beam, self.beam_size);
        }

        // Return the best final state
        let best_state = beam.pop().unwrap_or(State::new());

        SecondaryStructure::new(
            sequence.clone(),
            best_state.pairs,
            best_state.energy,
        )
    }

    /// Check if two positions can form a base pair
    fn can_pair(&self, sequence: &RnaSequence, i: usize, j: usize) -> bool {
        // Minimum loop length (at least 3 unpaired bases between pairs)
        const MIN_LOOP_LENGTH: usize = 3;

        if j <= i + MIN_LOOP_LENGTH {
            return false;
        }

        // Check if bases are complementary
        if let (Some(base_i), Some(base_j)) = (sequence.get(i), sequence.get(j)) {
            base_i.can_pair(base_j)
        } else {
            false
        }
    }

    /// Prune beam to keep only top-k states
    ///
    /// Rust learning: This shows how to work with iterators
    fn prune_beam(&self, beam: BinaryHeap<State>, max_size: usize) -> BinaryHeap<State> {
        let mut states: Vec<State> = beam.into_iter().collect();
        states.sort_by(|a, b| a.cmp(b));
        states.truncate(max_size);
        states.into_iter().collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_linearfold_simple() {
        // Simple hairpin: GGGGAAAACCCC should fold into ((((....))))
        let seq = RnaSequence::from_str("GGGGAAAACCCC").unwrap();
        let folder = LinearFold::default();
        let structure = folder.fold(&seq);

        println!("{}", structure);
        assert!(!structure.pairs.is_empty());
    }

    #[test]
    fn test_linearfold_short() {
        let seq = RnaSequence::from_str("GCACGACG").unwrap();
        let folder = LinearFold::new(10);
        let structure = folder.fold(&seq);

        println!("{}", structure);
        // Just verify it completes without panic
        assert_eq!(structure.sequence.len(), seq.len());
    }
}
