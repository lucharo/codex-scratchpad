//! Python bindings for the RNA folding library
//!
//! This module uses PyO3 to expose Rust functions to Python.
//!
//! ## Rust Learning: PyO3 and Python Bindings
//!
//! PyO3 is a Rust library that allows you to:
//! 1. Call Rust code from Python (what we're doing here)
//! 2. Call Python code from Rust
//! 3. Create native Python modules entirely in Rust
//!
//! ### Key Concepts:
//!
//! - `#[pyfunction]`: Marks a Rust function to be exposed to Python
//! - `#[pyclass]`: Marks a Rust struct to be exposed as a Python class
//! - `#[pymethods]`: Marks impl block with methods for a Python class
//! - `#[pymodule]`: Defines a Python module
//! - `PyResult<T>`: Python-compatible Result type
//!
//! ### Memory Management:
//!
//! PyO3 handles the conversion between Rust and Python types:
//! - Rust String ↔ Python str
//! - Rust Vec<T> ↔ Python list
//! - Rust f64 ↔ Python float
//! - etc.
//!
//! Python's garbage collector and Rust's ownership work together seamlessly!

use pyo3::prelude::*;
use crate::{RnaSequence, SecondaryStructure, LinearFold, EternaFold};

/// Python-facing wrapper for SecondaryStructure
///
/// Rust learning: #[pyclass] makes this available as a Python class
#[pyclass(name = "SecondaryStructure")]
pub struct PySecondaryStructure {
    /// The underlying Rust structure
    inner: SecondaryStructure,
}

#[pymethods]
impl PySecondaryStructure {
    /// Get the RNA sequence as a string
    #[getter]
    fn sequence(&self) -> String {
        self.inner.sequence.to_string()
    }

    /// Get the dot-bracket notation
    #[getter]
    fn structure(&self) -> String {
        self.inner.to_dot_bracket()
    }

    /// Get the free energy in kcal/mol
    #[getter]
    fn energy(&self) -> f64 {
        self.inner.energy
    }

    /// Get the number of base pairs
    #[getter]
    fn num_pairs(&self) -> usize {
        self.inner.pairs.len()
    }

    /// String representation for Python
    fn __str__(&self) -> String {
        format!("{}", self.inner)
    }

    /// Representation for Python
    fn __repr__(&self) -> String {
        format!(
            "SecondaryStructure(sequence='{}', structure='{}', energy={:.2})",
            self.inner.sequence.to_string(),
            self.inner.to_dot_bracket(),
            self.inner.energy
        )
    }
}

/// Python wrapper for LinearFold
#[pyclass(name = "LinearFold")]
pub struct PyLinearFold {
    inner: LinearFold,
}

#[pymethods]
impl PyLinearFold {
    /// Create a new LinearFold predictor
    ///
    /// Args:
    ///     beam_size: Number of states to keep in the beam (default: 100)
    ///
    /// Returns:
    ///     A new LinearFold predictor
    ///
    /// Example:
    ///     >>> folder = LinearFold(beam_size=100)
    #[new]
    #[pyo3(signature = (beam_size=100))]
    fn new(beam_size: usize) -> Self {
        PyLinearFold {
            inner: LinearFold::new(beam_size),
        }
    }

    /// Predict RNA secondary structure
    ///
    /// Args:
    ///     sequence: RNA sequence string (A, C, G, U)
    ///
    /// Returns:
    ///     SecondaryStructure object with prediction
    ///
    /// Raises:
    ///     ValueError: If sequence contains invalid characters
    ///
    /// Example:
    ///     >>> folder = LinearFold()
    ///     >>> structure = folder.fold("GGGGAAAACCCC")
    ///     >>> print(structure.structure)
    ///     ((((....))))
    fn fold(&self, sequence: &str) -> PyResult<PySecondaryStructure> {
        let seq = RnaSequence::from_str(sequence)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(e))?;

        let structure = self.inner.fold(&seq);

        Ok(PySecondaryStructure { inner: structure })
    }
}

/// Python wrapper for EternaFold
#[pyclass(name = "EternaFold")]
pub struct PyEternaFold {
    inner: EternaFold,
}

#[pymethods]
impl PyEternaFold {
    /// Create a new EternaFold predictor
    ///
    /// Returns:
    ///     A new EternaFold predictor with default parameters
    ///
    /// Example:
    ///     >>> folder = EternaFold()
    #[new]
    fn new() -> Self {
        PyEternaFold {
            inner: EternaFold::new(),
        }
    }

    /// Predict RNA secondary structure
    ///
    /// Args:
    ///     sequence: RNA sequence string (A, C, G, U)
    ///
    /// Returns:
    ///     SecondaryStructure object with prediction
    ///
    /// Raises:
    ///     ValueError: If sequence contains invalid characters
    ///
    /// Example:
    ///     >>> folder = EternaFold()
    ///     >>> structure = folder.fold("GGGGAAAACCCC")
    ///     >>> print(f"Energy: {structure.energy:.2f} kcal/mol")
    fn fold(&self, sequence: &str) -> PyResult<PySecondaryStructure> {
        let seq = RnaSequence::from_str(sequence)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(e))?;

        let structure = self.inner.fold(&seq);

        Ok(PySecondaryStructure { inner: structure })
    }

    /// Calculate partition function for a sequence
    ///
    /// The partition function sums over all possible structures weighted by
    /// their Boltzmann probability. It's used to calculate base pair probabilities.
    ///
    /// Args:
    ///     sequence: RNA sequence string (A, C, G, U)
    ///
    /// Returns:
    ///     Partition function value (float)
    ///
    /// Example:
    ///     >>> folder = EternaFold()
    ///     >>> z = folder.partition_function("GGGGAAAACCCC")
    fn partition_function(&self, sequence: &str) -> PyResult<f64> {
        let seq = RnaSequence::from_str(sequence)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(e))?;

        Ok(self.inner.partition_function(&seq))
    }
}

/// The Python module definition
///
/// This is the entry point when Python imports the module
///
/// Rust learning: #[pymodule] creates a Python module
/// The function name becomes the module name
#[pymodule]
fn rna_fold(m: &Bound<'_, PyModule>) -> PyResult<()> {
    // Add classes
    m.add_class::<PyLinearFold>()?;
    m.add_class::<PyEternaFold>()?;
    m.add_class::<PySecondaryStructure>()?;

    // Add module documentation
    m.add("__doc__", "RNA secondary structure prediction using LinearFold and EternaFold algorithms")?;
    m.add("__version__", env!("CARGO_PKG_VERSION"))?;

    Ok(())
}

/// Convenience function for quick predictions (Python)
///
/// This would be exposed as a top-level function in Python
#[pyfunction]
#[allow(dead_code)]
fn predict_linearfold(sequence: &str, beam_size: Option<usize>) -> PyResult<PySecondaryStructure> {
    let beam_size = beam_size.unwrap_or(100);
    let folder = PyLinearFold::new(beam_size);
    folder.fold(sequence)
}

#[pyfunction]
#[allow(dead_code)]
fn predict_eternafold(sequence: &str) -> PyResult<PySecondaryStructure> {
    let folder = PyEternaFold::new();
    folder.fold(sequence)
}
