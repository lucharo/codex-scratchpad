//! Common data structures for RNA folding algorithms
//!
//! This module defines the fundamental types used across all folding algorithms.
//!
//! ## Rust Learning Notes for Python/C++ Developers
//!
//! ### Structs vs Classes
//! - Rust uses `struct` for data + `impl` blocks for methods (like C++)
//! - No inheritance; composition and traits instead (think Python protocols)
//!
//! ### Enums
//! - Much more powerful than C++ enums or Python Enum
//! - Can hold data (like tagged unions in C++)
//!
//! ### derive Macros
//! - `#[derive(Debug, Clone, ...)]` auto-implements traits
//! - Similar to Python's @dataclass decorator

use std::fmt;

/// RNA nucleotide base
///
/// In Rust, enums can represent a fixed set of values (like Python Enum or C++ enum class).
/// The `derive` attribute automatically implements common functionality:
/// - `Debug`: Allows printing with {:?}
/// - `Clone`: Allows copying the value
/// - `Copy`: Allows implicit copying (since it's small and simple)
/// - `PartialEq, Eq`: Allows comparing with == and !=
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Base {
    A,  // Adenine
    C,  // Cytosine
    G,  // Guanine
    U,  // Uracil (instead of Thymine in DNA)
}

impl Base {
    /// Check if two bases can form a Watson-Crick base pair
    /// - A pairs with U
    /// - G pairs with C
    ///
    /// Rust note: `&self` is like `this` in C++ or `self` in Python
    pub fn can_pair_watson_crick(&self, other: &Base) -> bool {
        matches!(
            (self, other),
            (Base::A, Base::U) | (Base::U, Base::A) | (Base::G, Base::C) | (Base::C, Base::G)
        )
    }

    /// Check if two bases can form a wobble pair (G-U)
    pub fn can_pair_wobble(&self, other: &Base) -> bool {
        matches!(
            (self, other),
            (Base::G, Base::U) | (Base::U, Base::G)
        )
    }

    /// Check if two bases can pair (Watson-Crick or wobble)
    pub fn can_pair(&self, other: &Base) -> bool {
        self.can_pair_watson_crick(other) || self.can_pair_wobble(other)
    }

    /// Parse a character into a Base
    ///
    /// Returns `Option<Base>`:
    /// - `Some(base)` if valid
    /// - `None` if invalid
    ///
    /// This is Rust's way of handling "nullable" values without null pointers!
    pub fn from_char(c: char) -> Option<Base> {
        match c.to_ascii_uppercase() {
            'A' => Some(Base::A),
            'C' => Some(Base::C),
            'G' => Some(Base::G),
            'U' | 'T' => Some(Base::U),  // Accept both U and T
            _ => None,
        }
    }

    /// Convert a base to its character representation
    pub fn to_char(&self) -> char {
        match self {
            Base::A => 'A',
            Base::C => 'C',
            Base::G => 'G',
            Base::U => 'U',
        }
    }
}

/// RNA sequence
///
/// Rust note: `Vec<T>` is like std::vector<T> in C++ or list[T] in Python
/// It's a growable array that owns its data.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RnaSequence {
    /// The bases in the sequence
    pub bases: Vec<Base>,
}

impl RnaSequence {
    /// Create a new RNA sequence from a string
    ///
    /// Returns `Result<RnaSequence, String>`:
    /// - `Ok(sequence)` if successful
    /// - `Err(message)` if there's an error
    ///
    /// This is Rust's way of explicit error handling (no exceptions by default)
    pub fn from_str(s: &str) -> Result<Self, String> {
        let bases: Result<Vec<Base>, _> = s
            .chars()
            .filter(|c| !c.is_whitespace())  // Skip whitespace
            .map(|c| Base::from_char(c).ok_or_else(|| format!("Invalid base: {}", c)))
            .collect();

        Ok(RnaSequence { bases: bases? })
    }

    /// Get the length of the sequence
    pub fn len(&self) -> usize {
        self.bases.len()
    }

    /// Check if the sequence is empty
    pub fn is_empty(&self) -> bool {
        self.bases.is_empty()
    }

    /// Get a base at a specific position
    ///
    /// Returns `Option<&Base>`:
    /// - `Some(&base)` if position is valid
    /// - `None` if out of bounds
    ///
    /// Rust note: &Base is a "reference" - like a pointer but safe!
    pub fn get(&self, index: usize) -> Option<&Base> {
        self.bases.get(index)
    }

    /// Convert to string representation
    pub fn to_string(&self) -> String {
        self.bases.iter().map(|b| b.to_char()).collect()
    }
}

/// Implement Display trait for nice printing
/// This is like __str__ in Python or operator<< in C++
impl fmt::Display for RnaSequence {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.to_string())
    }
}

/// A base pair in RNA secondary structure
///
/// Represents a pair of positions that form a bond
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct BasePair {
    /// 5' position (smaller index)
    pub i: usize,
    /// 3' position (larger index)
    pub j: usize,
}

impl BasePair {
    /// Create a new base pair
    ///
    /// Ensures i < j by swapping if needed
    pub fn new(i: usize, j: usize) -> Self {
        if i < j {
            BasePair { i, j }
        } else {
            BasePair { i: j, j: i }
        }
    }

    /// Check if two base pairs are nested (one inside the other)
    pub fn is_nested(&self, other: &BasePair) -> bool {
        (self.i < other.i && other.i < self.j && other.j < self.j)
            || (other.i < self.i && self.i < other.j && self.j < other.j)
    }

    /// Check if two base pairs conflict (pseudoknot)
    pub fn conflicts_with(&self, other: &BasePair) -> bool {
        // Conflicting if they cross: i < k < j < l
        (self.i < other.i && other.i < self.j && self.j < other.j)
            || (other.i < self.i && self.i < other.j && other.j < self.j)
    }
}

/// RNA secondary structure
///
/// Represents the predicted folding structure as a set of base pairs
#[derive(Debug, Clone, PartialEq)]
pub struct SecondaryStructure {
    /// The RNA sequence
    pub sequence: RnaSequence,
    /// Base pairs in the structure
    pub pairs: Vec<BasePair>,
    /// Free energy of the structure (in kcal/mol)
    pub energy: f64,
}

impl SecondaryStructure {
    /// Create a new secondary structure
    pub fn new(sequence: RnaSequence, pairs: Vec<BasePair>, energy: f64) -> Self {
        SecondaryStructure {
            sequence,
            pairs,
            energy,
        }
    }

    /// Convert to dot-bracket notation
    ///
    /// - `.` represents unpaired base
    /// - `(` represents opening of a pair
    /// - `)` represents closing of a pair
    ///
    /// Example: `(((...)))` represents a hairpin loop
    pub fn to_dot_bracket(&self) -> String {
        let len = self.sequence.len();
        let mut structure = vec!['.'; len];

        for pair in &self.pairs {
            structure[pair.i] = '(';
            structure[pair.j] = ')';
        }

        structure.iter().collect()
    }

    /// Parse from dot-bracket notation
    pub fn from_dot_bracket(sequence: RnaSequence, dot_bracket: &str, energy: f64) -> Result<Self, String> {
        if sequence.len() != dot_bracket.len() {
            return Err("Sequence and structure length mismatch".to_string());
        }

        let mut pairs = Vec::new();
        let mut stack = Vec::new();

        for (i, c) in dot_bracket.chars().enumerate() {
            match c {
                '(' => stack.push(i),
                ')' => {
                    if let Some(j) = stack.pop() {
                        pairs.push(BasePair::new(j, i));
                    } else {
                        return Err("Unmatched closing bracket".to_string());
                    }
                }
                '.' => {}
                _ => return Err(format!("Invalid character in dot-bracket: {}", c)),
            }
        }

        if !stack.is_empty() {
            return Err("Unmatched opening bracket".to_string());
        }

        Ok(SecondaryStructure::new(sequence, pairs, energy))
    }
}

impl fmt::Display for SecondaryStructure {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "{}\n{} ({:.2} kcal/mol)",
            self.sequence.to_string(),
            self.to_dot_bracket(),
            self.energy
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_base_pairing() {
        assert!(Base::A.can_pair(&Base::U));
        assert!(Base::G.can_pair(&Base::C));
        assert!(Base::G.can_pair(&Base::U));  // Wobble pair
        assert!(!Base::A.can_pair(&Base::C));
    }

    #[test]
    fn test_rna_sequence() {
        let seq = RnaSequence::from_str("AUGC").unwrap();
        assert_eq!(seq.len(), 4);
        assert_eq!(seq.to_string(), "AUGC");
    }

    #[test]
    fn test_dot_bracket() {
        let seq = RnaSequence::from_str("GGGGAAACCCC").unwrap();
        let structure = SecondaryStructure::from_dot_bracket(seq, "((((...))))", -5.0).unwrap();
        assert_eq!(structure.pairs.len(), 4);
        assert_eq!(structure.to_dot_bracket(), "((((...))))");
    }
}
