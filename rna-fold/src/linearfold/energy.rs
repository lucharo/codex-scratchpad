//! Energy models for RNA folding
//!
//! ## What is Free Energy in RNA Folding?
//!
//! RNA molecules fold to minimize free energy (measured in kcal/mol).
//! Lower energy = more stable structure.
//!
//! Real energy models (like Vienna RNA) use:
//! - Experimentally measured thermodynamic parameters
//! - Loop energies (hairpin, bulge, internal, multi-loops)
//! - Stacking energies (adjacent base pairs)
//! - Temperature-dependent calculations
//!
//! This simplified model uses:
//! - Basic stacking bonuses for adjacent pairs
//! - Watson-Crick vs wobble pair preferences
//! - Simple loop penalties

use crate::common::{RnaSequence, Base};

/// Simplified energy model for educational purposes
///
/// Real implementations use complex thermodynamic models.
/// This version captures the key concepts without all the parameters.
///
/// Rust learning: Structs can be empty if they just namespace functions
pub struct SimpleEnergyModel {
    // In a real implementation, this would hold thermodynamic parameters
}

impl SimpleEnergyModel {
    /// Create a new energy model
    pub fn new() -> Self {
        SimpleEnergyModel {}
    }

    /// Calculate energy contribution of a base pair
    ///
    /// Rust learning: `&self` means this is a method, not a static function
    pub fn pair_energy(&self, sequence: &RnaSequence, i: usize, j: usize) -> f64 {
        // Get the bases (with safe Option handling)
        let base_i = match sequence.get(i) {
            Some(b) => b,
            None => return 0.0,
        };

        let base_j = match sequence.get(j) {
            Some(b) => b,
            None => return 0.0,
        };

        // Base pairing energy
        let mut energy = if base_i.can_pair_watson_crick(base_j) {
            -3.0  // Watson-Crick pairs are more stable
        } else if base_i.can_pair_wobble(base_j) {
            -1.0  // G-U wobble pairs are less stable
        } else {
            5.0   // Invalid pair (high penalty)
        };

        // Stacking bonus: adjacent pairs are more stable
        if i > 0 && j < sequence.len() - 1 {
            if let (Some(prev_i), Some(next_j)) = (sequence.get(i - 1), sequence.get(j + 1)) {
                if prev_i.can_pair(next_j) {
                    energy -= 2.0;  // Stacking bonus
                }
            }
        }

        energy
    }

    /// Calculate loop energy
    ///
    /// Different loop types have different energies:
    /// - Hairpin loops: Small loops are penalized
    /// - Interior loops: Asymmetry increases energy
    /// - Multi-loops: More unpaired bases = higher energy
    pub fn loop_energy(&self, loop_size: usize) -> f64 {
        match loop_size {
            0..=3 => 5.0,      // Too small, penalize heavily
            4 => 3.0,          // Minimum stable hairpin
            5..=6 => 2.5,      // Good hairpin size
            7..=10 => 3.0,     // Larger hairpin
            _ => 3.0 + (loop_size as f64 - 10.0) * 0.1,  // Penalty grows slowly
        }
    }

    /// Calculate total energy for a set of base pairs
    ///
    /// This is a simplified version. Real implementations consider:
    /// - Loop types (hairpin, bulge, internal, multi)
    /// - Nearest-neighbor parameters
    /// - Dangling ends
    /// - Special cases (GU closure, etc.)
    pub fn total_energy(&self, sequence: &RnaSequence) -> f64 {
        // For now, just sum up individual pair energies
        // A complete implementation would analyze loop structure
        0.0
    }
}

impl Default for SimpleEnergyModel {
    fn default() -> Self {
        Self::new()
    }
}

/// Advanced energy model placeholder
///
/// This would include full thermodynamic parameters like:
/// - Turner energy parameters (2004 or 1999 parameter set)
/// - Temperature-dependent calculations
/// - Detailed loop energy models
/// - Coaxial stacking
/// - Terminal mismatches
///
/// See Vienna RNA package for a complete implementation
pub struct ViennaEnergyModel {
    temperature: f64,
}

impl ViennaEnergyModel {
    #[allow(dead_code)]
    pub fn new(temperature: f64) -> Self {
        ViennaEnergyModel { temperature }
    }

    // Full implementation would go here...
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_pair_energy() {
        let seq = RnaSequence::from_str("GCGC").unwrap();
        let model = SimpleEnergyModel::new();

        // G-C is a Watson-Crick pair (strong)
        let energy = model.pair_energy(&seq, 0, 3);
        assert!(energy < 0.0, "G-C pair should have negative energy");
    }

    #[test]
    fn test_loop_energy() {
        let model = SimpleEnergyModel::new();

        // Hairpin loop of size 4 (minimum stable)
        let energy = model.loop_energy(4);
        assert!(energy > 0.0, "Loops should have positive energy");
    }
}
