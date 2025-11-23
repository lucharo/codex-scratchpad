//! Beam search implementation for LinearFold
//!
//! ## What is Beam Search?
//!
//! Beam search is a heuristic search algorithm that explores a graph by expanding
//! the most promising nodes, but limits the number of nodes kept in memory.
//!
//! Think of it like:
//! - **Breadth-first search**: Explores all possibilities (too slow)
//! - **Greedy search**: Explores only the best option (misses good solutions)
//! - **Beam search**: Explores the top-K best options (good balance!)
//!
//! ## Why Beam Search for RNA?
//!
//! RNA folding has exponentially many possible structures.
//! Beam search lets us:
//! 1. Find good structures quickly
//! 2. Control memory usage (keep only top-K states)
//! 3. Avoid getting stuck in local optima (keep multiple candidates)
//!
//! ## Rust Learning: Generics and Traits
//!
//! This module shows advanced Rust features:
//! - Generic types: `<T>` (like templates in C++ or generics in Python)
//! - Trait bounds: `T: Ord` (like type constraints)
//! - Lifetimes: `'a` (ensures references are valid)

use std::collections::BinaryHeap;
use std::cmp::Ordering;

/// A scored item in the beam
///
/// Rust learning: This is a generic struct that can hold any type T
/// Think of it like `template<typename T>` in C++ or `Generic[T]` in Python
#[derive(Debug, Clone)]
pub struct Scored<T> {
    pub item: T,
    pub score: f64,
}

impl<T> Scored<T> {
    pub fn new(item: T, score: f64) -> Self {
        Scored { item, score }
    }
}

/// Implement ordering for the priority queue
///
/// Rust learning: We're implementing the Ord trait for any Scored<T>
/// This allows Scored items to be compared and sorted
impl<T> Ord for Scored<T> {
    fn cmp(&self, other: &Self) -> Ordering {
        // Lower score is better, so reverse the comparison
        // Note: We use partial_cmp because f64 can be NaN
        other.score
            .partial_cmp(&self.score)
            .unwrap_or(Ordering::Equal)
    }
}

impl<T> PartialOrd for Scored<T> {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl<T> PartialEq for Scored<T> {
    fn eq(&self, other: &Self) -> bool {
        self.score == other.score
    }
}

impl<T> Eq for Scored<T> {}

/// Beam searcher
///
/// Rust learning: This is a generic type that works with any T
pub struct BeamSearcher<T> {
    /// Maximum beam size
    beam_size: usize,

    /// Current beam (priority queue of scored items)
    /// Rust note: BinaryHeap keeps items sorted
    beam: BinaryHeap<Scored<T>>,
}

impl<T: Clone> BeamSearcher<T> {
    /// Create a new beam searcher
    pub fn new(beam_size: usize) -> Self {
        BeamSearcher {
            beam_size,
            beam: BinaryHeap::new(),
        }
    }

    /// Add an item to the beam
    ///
    /// Rust learning: `&mut self` means this method can modify the struct
    /// (like a non-const method in C++)
    pub fn push(&mut self, item: T, score: f64) {
        self.beam.push(Scored::new(item, score));
    }

    /// Prune the beam to keep only top-k items
    pub fn prune(&mut self) {
        if self.beam.len() <= self.beam_size {
            return;
        }

        // Convert to sorted vector
        let mut items: Vec<_> = self.beam.drain().collect();
        items.sort();
        items.truncate(self.beam_size);

        // Rebuild heap
        self.beam = items.into_iter().collect();
    }

    /// Get the best item from the beam
    ///
    /// Rust learning: Returns `Option<T>` to handle empty beam
    pub fn best(&mut self) -> Option<T> {
        self.beam.pop().map(|scored| scored.item)
    }

    /// Get all items in the beam
    ///
    /// Rust learning: Returns a Vec (owned data)
    pub fn all_items(&self) -> Vec<T> {
        self.beam.iter().map(|scored| scored.item.clone()).collect()
    }

    /// Check if beam is empty
    pub fn is_empty(&self) -> bool {
        self.beam.is_empty()
    }

    /// Get current beam size
    pub fn len(&self) -> usize {
        self.beam.len()
    }

    /// Clear the beam
    pub fn clear(&mut self) {
        self.beam.clear();
    }
}

/// Iterator support for BeamSearcher
///
/// Rust learning: This allows using `for item in beam { ... }`
impl<T> Iterator for BeamSearcher<T> {
    type Item = Scored<T>;

    fn next(&mut self) -> Option<Self::Item> {
        self.beam.pop()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_beam_searcher() {
        let mut beam: BeamSearcher<i32> = BeamSearcher::new(3);

        // Add some items (lower score is better)
        beam.push(10, 10.0);  // bad
        beam.push(1, 1.0);    // good
        beam.push(5, 0.5);    // better
        beam.push(20, 20.0);  // worst

        // Prune to top 3
        beam.prune();

        // Should keep only 3 items
        assert_eq!(beam.len(), 3);

        // All items should be present (we don't care about order for this test)
        assert!(!beam.is_empty());
    }

    #[test]
    fn test_beam_ordering() {
        let mut beam: BeamSearcher<i32> = BeamSearcher::new(5);

        beam.push(1, 5.0);
        beam.push(2, 3.0);
        beam.push(3, 1.0);

        // Should pop in order of score (lowest first)
        assert_eq!(beam.best(), Some(3));
        assert_eq!(beam.best(), Some(2));
        assert_eq!(beam.best(), Some(1));
    }
}
