//! # RNA Fold: Educational Implementation of RNA Folding Algorithms
//!
//! This library implements two state-of-the-art RNA secondary structure prediction algorithms:
//! - **LinearFold**: Linear-time (O(n)) approximate folding using beam search
//! - **EternaFold**: Machine learning-based folding trained on experimental data
//!
//! ## For Python/C++ Developers Learning Rust
//!
//! Key Rust concepts you'll encounter:
//! - **Ownership**: Rust's memory safety without garbage collection
//! - **Borrowing**: References (&) and mutable references (&mut)
//! - **Pattern matching**: Like switch/case but much more powerful
//! - **Option<T>**: Rust's way of handling nullable values (no null pointers!)
//! - **Result<T, E>**: Explicit error handling (no exceptions by default)
//! - **Traits**: Similar to interfaces in C++/Python protocols
//!
//! ## Module Organization
//!
//! - `common`: Shared data structures (RNA sequences, base pairs, structures)
//! - `linearfold`: LinearFold algorithm implementation
//! - `eternafold`: EternaFold algorithm implementation
//! - `python`: Python bindings (when compiled with `python` feature)

// Declare submodules
pub mod common;
pub mod linearfold;
pub mod eternafold;

// Re-export main types for convenience
pub use common::{RnaSequence, SecondaryStructure, BasePair};
pub use linearfold::LinearFold;
pub use eternafold::EternaFold;

// Python bindings (only compiled when the "python" feature is enabled)
#[cfg(feature = "python")]
pub mod python;
