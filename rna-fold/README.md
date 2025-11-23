# RNA Fold: Educational Implementation of RNA Structure Prediction

[![Rust](https://img.shields.io/badge/rust-1.70%2B-orange.svg)](https://www.rust-lang.org/)
[![Python](https://img.shields.io/badge/python-3.8%2B-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

An educational implementation of two state-of-the-art RNA secondary structure prediction algorithms: **LinearFold** and **EternaFold**. This project is designed for learning:

- 🧬 **How RNA folding algorithms work** by implementing them
- 🦀 **How to write Rust** coming from Python/C++ backgrounds
- 🐍 **How to connect Rust to Python** using PyO3 bindings
- 📚 **Modern bioinformatics algorithms** through hands-on code

## Table of Contents

1. [Background: RNA Secondary Structure Prediction](#background)
2. [The Algorithms](#the-algorithms)
   - [LinearFold](#linearfold)
   - [EternaFold](#eternafold)
3. [Getting Started with Rust](#getting-started-with-rust)
4. [Using the Library](#using-the-library)
   - [Rust](#usage-rust)
   - [Python](#usage-python)
5. [Rust Concepts for Python/C++ Developers](#rust-concepts)
6. [Project Structure](#project-structure)
7. [References](#references)

---

## Background: RNA Secondary Structure Prediction {#background}

### What is RNA Secondary Structure?

RNA molecules fold into complex 3D shapes that determine their function. The secondary structure represents which bases pair with each other:

```
Sequence:  GGGGAAAACCCC
Structure: ((((....))))
           ||||    ||||
           G-C pairs form stems
           A's form a hairpin loop
```

### Why Predict Structure?

- **Drug design**: Many drugs target RNA structures
- **Synthetic biology**: Design RNA with specific functions
- **Understanding biology**: Structure determines function

### The Prediction Problem

Given an RNA sequence (A, C, G, U), predict which bases pair to minimize free energy.

**Challenges:**
- Exponentially many possible structures (2^n possibilities!)
- Complex thermodynamics
- Need for speed (genomes have billions of bases)

---

## The Algorithms {#the-algorithms}

### LinearFold {#linearfold}

**LinearFold: Linear-Time Approximate RNA Folding**

#### Key Innovation

Traditional algorithms (like Zuker's) use dynamic programming with **O(n³)** time complexity. LinearFold achieves **O(n)** time using:

1. **Directional Processing**: Scan 5' to 3' (left to right) instead of considering all substrings
2. **Beam Search**: Keep only top-K candidate structures instead of exploring everything
3. **Pruning**: Discard unlikely structures early

#### How It Works

```
For each position j from 0 to n:
    For each candidate structure in beam:
        Option 1: Leave j unpaired
        Option 2: Pair j with earlier unpaired position i
            - Check if i,j can pair (complementary bases)
            - Score the pairing
            - Add to next beam
    Prune beam to keep top-K candidates
```

The beam size (default 100) controls accuracy vs speed tradeoff.

#### Papers & References

- **Paper**: [LinearFold: linear-time approximate RNA folding by 5'-to-3' dynamic programming and beam search](https://academic.oup.com/bioinformatics/article/35/14/i295/5529205) (Bioinformatics, 2019)
- **Authors**: He Zhang, Liang Huang, et al.
- **GitHub**: https://github.com/LinearFold/LinearFold
- **Complexity**: O(n) time, O(n) space
- **Accuracy**: Competitive with O(n³) methods on most sequences

### EternaFold {#eternafold}

**EternaFold: ML-Based Folding Trained on Experimental Data**

#### Key Innovation

Instead of using physics-based thermodynamic models, EternaFold uses **machine learning** trained on real experimental data from the Eterna project (a citizen science game where players design RNA).

#### Key Features

1. **Multitask Learning**: Trains on three objectives simultaneously
   - Structure prediction (base pairing)
   - Chemical probing data (SHAPE reactivity)
   - Binding affinity prediction

2. **Experimental Training Data**:
   - 100,000+ RNA designs from Eterna players
   - Actual chemical probing experiments
   - More diverse than traditional thermodynamic datasets

3. **CONTRAfold-SE Base**:
   - Discriminative model (learns from data)
   - Not purely thermodynamic (captures real-world effects)
   - Hundreds of learned parameters

#### How It Works

```
Extract features from sequence:
    - Base pair types (AU, GC, GU)
    - Stacking patterns
    - Loop structures
    - Sequence context

Score = Σ (weight_i × feature_i)

Use dynamic programming to find structure with best score
```

#### Papers & References

- **Paper**: [RNA secondary structure packages evaluated and improved by high-throughput experiments](https://pubmed.ncbi.nlm.nih.gov/36192461/) (Nature Methods, 2022)
- **Authors**: Wayment-Steele et al.
- **GitHub**: https://github.com/eternagame/EternaFold
- **Eterna Project**: https://eternagame.org
- **Training Data**: Crowd-sourced RNA designs with experimental validation

---

## Getting Started with Rust {#getting-started-with-rust}

### What is Rust?

Rust is a systems programming language that offers:
- **Memory safety** without garbage collection
- **Performance** comparable to C/C++
- **Modern tooling** and package management
- **Fearless concurrency**

### Installation

#### Option 1: Using Pixi (Recommended)

[Pixi](https://prefix.dev/docs/pixi/overview) is a modern package manager that handles both Rust and Python:

```bash
# Install pixi (if not already installed)
curl -fsSL https://pixi.sh/install.sh | bash

# Navigate to project directory
cd rna-fold

# Install all dependencies (Rust, Python, etc.)
pixi install

# Activate the environment
pixi shell
```

#### Option 2: Using rustup (Traditional)

```bash
# Install Rust toolchain
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh

# Install Python and maturin separately
pip install maturin pytest
```

### Building and Running Rust Code

```bash
# Build the project
cargo build

# Run tests
cargo test

# Build optimized version
cargo build --release

# Run with logging
RUST_LOG=info cargo run

# Check code without building
cargo check

# Format code
cargo fmt

# Lint code
cargo clippy
```

### Project Structure

```
rna-fold/
├── Cargo.toml          # Rust package manifest (like package.json or setup.py)
├── pixi.toml           # Pixi environment configuration
├── pyproject.toml      # Python package configuration
├── src/
│   ├── lib.rs          # Library root (exports public API)
│   ├── common.rs       # Shared data structures
│   ├── linearfold/     # LinearFold implementation
│   │   ├── mod.rs      # Module entry (like __init__.py)
│   │   ├── beam.rs     # Beam search algorithm
│   │   └── energy.rs   # Energy scoring
│   ├── eternafold/     # EternaFold implementation
│   │   ├── mod.rs
│   │   ├── features.rs # Feature extraction
│   │   └── dp.rs       # Dynamic programming
│   └── python.rs       # Python bindings (PyO3)
└── python/
    ├── example.py      # Python usage examples
    └── tests/          # Python tests
```

---

## Using the Library {#using-the-library}

### Rust {#usage-rust}

```rust
use rna_fold::{RnaSequence, LinearFold, EternaFold};

fn main() {
    // Parse RNA sequence
    let sequence = RnaSequence::from_str("GGGGAAAACCCC").unwrap();

    // LinearFold prediction
    let linearfold = LinearFold::new(100); // beam size
    let structure = linearfold.fold(&sequence);

    println!("LinearFold:");
    println!("{}", structure);
    // Output:
    // GGGGAAAACCCC
    // ((((....))))  (-12.5 kcal/mol)

    // EternaFold prediction
    let eternafold = EternaFold::new();
    let structure = eternafold.fold(&sequence);

    println!("\nEternaFold:");
    println!("{}", structure);
}
```

### Python {#usage-python}

First, build the Python module:

```bash
# Development install (rebuild on code changes)
maturin develop --features python

# Or using pixi
pixi run build-python
```

Then use from Python:

```python
from rna_fold import LinearFold, EternaFold

# LinearFold prediction
folder = LinearFold(beam_size=100)
structure = folder.fold("GGGGAAAACCCC")

print(f"Sequence:  {structure.sequence}")
print(f"Structure: {structure.structure}")
print(f"Energy:    {structure.energy:.2f} kcal/mol")
print(f"Pairs:     {structure.num_pairs}")

# EternaFold prediction
folder = EternaFold()
structure = folder.fold("GGGGAAAACCCC")

# Calculate partition function
z = folder.partition_function("GGGGAAAACCCC")
print(f"Partition function: {z:.2e}")
```

Run the examples:

```bash
# Python example
python python/example.py

# Python tests
pytest python/tests/

# Or using pixi
pixi run test-python
```

---

## Rust Concepts for Python/C++ Developers {#rust-concepts}

### Ownership and Borrowing

**The Big Idea**: Rust tracks who "owns" each piece of data. No garbage collector needed!

```rust
// Python equivalent: s = "hello"
let s = String::from("hello");  // s owns the string

// Python: t = s (both point to same string)
// Rust: s is moved to t, s no longer valid!
let t = s;
// println!("{}", s);  // ERROR: s was moved

// Solution: Clone or borrow
let s = String::from("hello");
let t = s.clone();     // Deep copy (like Python copy.deepcopy)
let r = &s;            // Borrow (like Python reference, but compiler-checked!)
```

**Key Rules:**
1. Each value has exactly one owner
2. When owner goes out of scope, value is dropped (freed)
3. You can have many immutable borrows (&T) OR one mutable borrow (&mut T)

**Python analogy**: Like reference counting, but compile-time checked!

**C++ analogy**: Like std::unique_ptr, but built into the language!

### Option and Result: No Null Pointers!

```rust
// Python: might return None
// C++: might return nullptr
// Rust: Option<T>

fn find_base(seq: &RnaSequence, index: usize) -> Option<&Base> {
    seq.get(index)  // Returns Some(&base) or None
}

// Use with pattern matching
match find_base(&seq, 10) {
    Some(base) => println!("Found: {:?}", base),
    None => println!("Out of bounds!"),
}

// Or use methods
let base = find_base(&seq, 10).unwrap_or(&Base::A);
```

```rust
// Python: might raise exception
// C++: might throw exception
// Rust: Result<T, E>

fn parse_sequence(s: &str) -> Result<RnaSequence, String> {
    // Returns Ok(sequence) or Err(message)
}

// Use with ?  operator (like Python's "try")
let seq = parse_sequence("ACGU")?;  // Returns early if error
```

### Traits: Like Interfaces

```rust
// Python: class that implements a protocol
// C++: class that implements an interface
// Rust: struct that implements a trait

trait Foldable {
    fn fold(&self, sequence: &RnaSequence) -> SecondaryStructure;
}

impl Foldable for LinearFold {
    fn fold(&self, sequence: &RnaSequence) -> SecondaryStructure {
        // Implementation
    }
}
```

### Structs and Enums

```rust
// Python: @dataclass
// C++: struct with methods
// Rust: struct + impl

struct RnaSequence {
    bases: Vec<Base>,  // Like Python list[Base]
}

impl RnaSequence {
    fn new() -> Self { ... }         // Constructor (static method)
    fn len(&self) -> usize { ... }   // Method (like self in Python)
}
```

```rust
// Python: Enum (basic) or Union types
// C++: enum class or std::variant
// Rust: enum (can hold data!)

enum Base {
    A,
    C,
    G,
    U,
}

// Powerful: enums can have data
enum Result<T, E> {
    Ok(T),      // Success case holds a T
    Err(E),     // Error case holds an E
}
```

### Pattern Matching

```rust
// Like Python match (3.10+) but more powerful
// Like C++ switch but works on complex types

match base {
    Base::A => println!("Adenine"),
    Base::C | Base::G => println!("Strong pair"),
    _ => println!("Other"),
}

// Destructuring
match result {
    Ok(value) => println!("Success: {}", value),
    Err(e) => println!("Error: {}", e),
}
```

### Generics

```rust
// Python: Generic[T]
// C++: template<typename T>
// Rust: <T>

struct BeamSearcher<T> {
    items: Vec<T>,
}

impl<T> BeamSearcher<T> {
    fn push(&mut self, item: T) { ... }
}

// With trait bounds
fn find_best<T: Ord>(items: Vec<T>) -> Option<T> { ... }
```

### Memory Management Comparison

| Concept | Python | C++ | Rust |
|---------|--------|-----|------|
| Allocation | Automatic (GC) | Manual (`new`) or RAII | Automatic (ownership) |
| Deallocation | GC | Manual (`delete`) or RAII | Automatic (drop) |
| References | Reference counting | Pointers/references | Borrowed references |
| Safety | Runtime checks | Unsafe (UB possible) | Compile-time checks |
| Performance | Slower (GC overhead) | Fast | Fast (zero-cost abstractions) |

---

## Python Bindings with PyO3 {#python-bindings}

### How PyO3 Works

PyO3 creates a bridge between Rust and Python:

```rust
use pyo3::prelude::*;

// Mark struct as Python class
#[pyclass]
struct LinearFold {
    beam_size: usize,
}

// Mark methods as Python methods
#[pymethods]
impl LinearFold {
    #[new]
    fn new(beam_size: usize) -> Self {
        LinearFold { beam_size }
    }

    fn fold(&self, sequence: &str) -> PyResult<SecondaryStructure> {
        // Rust code here
        Ok(result)
    }
}

// Create Python module
#[pymodule]
fn rna_fold(m: &PyModule) -> PyResult<()> {
    m.add_class::<LinearFold>()?;
    Ok(())
}
```

### Building Python Extensions

```bash
# Development build (fast, for testing)
maturin develop --features python

# Release build (optimized, for production)
maturin build --release --features python

# Create wheel for distribution
maturin build --release --features python -o dist/
```

### Type Conversions

| Rust Type | Python Type | Notes |
|-----------|-------------|-------|
| `String` | `str` | Automatic conversion |
| `Vec<T>` | `list` | Automatic conversion |
| `HashMap<K,V>` | `dict` | Automatic conversion |
| `f64` | `float` | Automatic conversion |
| `usize` | `int` | Automatic conversion |
| `Option<T>` | `Optional[T]` | `None` maps to `None` |
| `Result<T,E>` | Exception | `Err` raises Python exception |

---

## Performance Notes

### LinearFold

- **Time**: O(n) with beam search (vs O(n³) for exact algorithms)
- **Space**: O(n × beam_size)
- **Beam size tradeoff**:
  - Larger beam = more accurate but slower
  - Smaller beam = faster but less accurate
  - Default 100 is a good balance

### EternaFold

- **Time**: O(n³) for dynamic programming
- **Space**: O(n²) for DP table
- **More accurate** on sequences similar to Eterna training data
- **Use when**: Accuracy matters more than speed

### Benchmarking

```bash
# Rust benchmarks
cargo bench

# Python benchmarks
pytest python/tests/ --benchmark
```

---

## Learning Resources

### RNA Folding

- [ViennaRNA Package](https://www.tbi.univie.ac.at/RNA/) - Classic thermodynamic folding
- [Eterna](https://eternagame.org) - RNA design game with real experiments
- [Rfam](https://rfam.xfam.org/) - RNA families database

### Rust

- [The Rust Book](https://doc.rust-lang.org/book/) - Official Rust tutorial
- [Rust by Example](https://doc.rust-lang.org/rust-by-example/) - Learn by doing
- [Rustlings](https://github.com/rust-lang/rustlings) - Interactive exercises

### PyO3

- [PyO3 Guide](https://pyo3.rs/) - Official PyO3 documentation
- [Maturin](https://www.maturin.rs/) - Build Python wheels from Rust

---

## References {#references}

### LinearFold

- **Paper**: He Zhang, Liang Huang. "LinearFold: linear-time approximate RNA folding by 5'-to-3' dynamic programming and beam search." *Bioinformatics*, Volume 35, Issue 14, July 2019, Pages i295–i304. [DOI:10.1093/bioinformatics/btz375](https://doi.org/10.1093/bioinformatics/btz375)
- **arXiv**: https://arxiv.org/abs/2001.04020
- **GitHub**: https://github.com/LinearFold/LinearFold

### EternaFold

- **Paper**: Hannah K. Wayment-Steele et al. "RNA secondary structure packages evaluated and improved by high-throughput experiments." *Nature Methods*, 2022. [PubMed](https://pubmed.ncbi.nlm.nih.gov/36192461/)
- **GitHub**: https://github.com/eternagame/EternaFold
- **Eterna Project**: https://eternagame.org

### Classic RNA Folding

- **Nussinov Algorithm**: Ruth Nussinov, Ann B. Jacobson. "Fast algorithm for predicting the secondary structure of single-stranded RNA." *PNAS*, 1980.
- **Zuker Algorithm**: Michael Zuker, Patrick Stiegler. "Optimal computer folding of large RNA sequences using thermodynamics and auxiliary information." *Nucleic Acids Research*, 1981.
- **CONTRAfold**: Chuong B. Do et al. "CONTRAfold: RNA secondary structure prediction without physics-based models." *Bioinformatics*, 2006.

---

## License

MIT License - see LICENSE file for details.

---

## Contributing

This is an educational project! Contributions welcome:

- Improve code comments and documentation
- Add more examples
- Implement additional features from the papers
- Optimize performance
- Add visualizations

---

## Acknowledgments

- **LinearFold team** (Liang Huang et al.) for the algorithm
- **EternaFold team** and Eterna community for experimental data
- **Rust community** for excellent tooling and documentation
- **PyO3 developers** for making Rust-Python interop smooth

---

## FAQ

**Q: Is this production-ready?**

A: No! This is an educational implementation. For production use, see the official LinearFold and EternaFold packages.

**Q: How accurate is it?**

A: Less accurate than the full implementations due to simplifications, but captures the core algorithms.

**Q: Can I use this for my research?**

A: This is for learning. For research, use the official implementations linked above.

**Q: Why Rust?**

A: Rust offers performance comparable to C++ with safety guarantees, making it ideal for learning systems programming concepts.

**Q: How do I visualize structures?**

A: The dot-bracket notation can be visualized using tools like [forna](http://rna.tbi.univie.ac.at/forna/) or [VARNA](http://varna.lri.fr/).

---

**Happy RNA Folding! 🧬🦀🐍**
