# Quick Start Guide

Get up and running with RNA Fold in 5 minutes!

## Prerequisites

Choose one of:

**Option A: Pixi (Recommended)**
```bash
curl -fsSL https://pixi.sh/install.sh | bash
```

**Option B: Manual**
```bash
# Install Rust
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh

# Install Python packages
pip install maturin pytest
```

## Installation

```bash
# Clone or download the repository
cd rna-fold

# With Pixi
pixi install
pixi shell

# Or with manual setup
cargo build
```

## Quick Tests

### Test Rust Implementation

```bash
# Run tests
cargo test

# Run example
cargo run --example basic_usage

# Run benchmark (release mode for speed)
cargo run --example benchmark --release
```

### Test Python Bindings

```bash
# Build Python module
maturin develop --features python

# Run Python example
python python/example.py

# Run Python tests
pytest python/tests/ -v
```

## Your First RNA Prediction

### Rust

Create a file `my_rna.rs`:

```rust
use rna_fold::{RnaSequence, LinearFold};

fn main() {
    let seq = RnaSequence::from_str("GGGGAAAACCCC").unwrap();
    let folder = LinearFold::new(100);
    let structure = folder.fold(&seq);
    println!("{}", structure);
}
```

Run with:
```bash
cargo run
```

### Python

Create a file `my_rna.py`:

```python
from rna_fold import LinearFold

folder = LinearFold(beam_size=100)
structure = folder.fold("GGGGAAAACCCC")

print(f"Sequence:  {structure.sequence}")
print(f"Structure: {structure.structure}")
print(f"Energy:    {structure.energy:.2f} kcal/mol")
```

Run with:
```bash
python my_rna.py
```

## Next Steps

- Read the [README](README.md) for detailed documentation
- Explore the code in `src/` to learn Rust concepts
- Check out `python/example.py` for more Python usage
- Read the inline comments to understand the algorithms

## Common Issues

**"rna_fold module not found" in Python**
```bash
maturin develop --features python
```

**Tests failing**
```bash
# Make sure you're in the project directory
cd rna-fold

# Clean and rebuild
cargo clean
cargo build
cargo test
```

**Slow performance**
```bash
# Always use --release for benchmarks
cargo run --example benchmark --release
```

## Getting Help

- Check the [README](README.md) for detailed info
- Look at the [examples](examples/) directory
- Read the inline code documentation
- Open an issue on GitHub

Happy RNA folding! 🧬
