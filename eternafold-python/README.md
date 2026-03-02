# eternafold-python

Python bindings for [EternaFold](https://github.com/eternagame/EternaFold), an RNA secondary structure prediction tool.

## Installation

```bash
pip install eternafold
```

## Quick Start

```python
import eternafold

# Predict MEA structure
structure = eternafold.predict("GGGGGAAAAAACCCCC")
print(structure)  # (((((......))))

# Compute log partition function
log_z = eternafold.pfunc("GGGGGAAAAAACCCCC")

# Fold: predict structure and compute energy
structure, energy = eternafold.fold("GGGGGAAAAAACCCCC")

# Energy of a specific structure
energy = eternafold.energy_of_structure("GGGGGAAAAAACCCCC", "(((((......)))))")

# Base pair probability matrix
bpp_matrix = eternafold.bpps("GGGGGAAAAAACCCCC")
```

## Arnie Integration

```python
import eternafold
eternafold.configure_arnie()

# Now Arnie can find EternaFold automatically
from arnie.pfunc import pfunc
Z = pfunc("GGGGGAAAAAACCCCC", package="eternafold")
```

## License

BSD-3-Clause (see vendor/eternafold/LICENSE)
