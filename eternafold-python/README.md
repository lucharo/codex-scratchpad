# eternafold-python

Python bindings for [EternaFold](https://github.com/eternagame/EternaFold), an RNA secondary structure prediction tool.

Installs directly via `pip` / `uv` — no conda or pixi environment required. Works with downstream libraries like [Arnie](https://github.com/DasLab/arnie) without any modification to their source code.

## Installation

```bash
pip install eternafold
```

## Quick Start

```python
import eternafold

structure = eternafold.fold("GGGGGAAAAAACCCCC")
```

## Arnie Integration

```python
import eternafold
eternafold.configure_arnie()

# Arnie finds EternaFold automatically — no arnie.rc changes needed
from arnie.pfunc import pfunc
Z = pfunc("GGGGGAAAAAACCCCC", package="eternafold")
```

## Development

Requires [uv](https://docs.astral.sh/uv/):

```bash
uv sync
uv run pytest
```

## License

BSD-3-Clause (see [EternaFold LICENSE](https://github.com/eternagame/EternaFold/blob/master/LICENSE))
