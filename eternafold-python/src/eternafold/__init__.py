"""EternaFold: Python bindings for RNA secondary structure prediction.

Example usage::

    import eternafold

    # Predict MEA structure
    structure = eternafold.predict("GGGGGAAAAAACCCCC")

    # Compute log partition function
    log_z = eternafold.pfunc("GGGGGAAAAAACCCCC")

    # Fold: predict structure and compute its energy
    structure, energy = eternafold.fold("GGGGGAAAAAACCCCC")

    # Energy of a specific structure
    energy = eternafold.energy_of_structure("GGGGGAAAAAACCCCC", "(((((......)))))")

    # Base pair probability matrix
    bpp_matrix = eternafold.bpps("GGGGGAAAAAACCCCC")

    # Arnie compatibility
    eternafold.configure_arnie()
"""

from __future__ import annotations

import os
from pathlib import Path

from eternafold._core import pfunc as _pfunc
from eternafold._core import predict as _predict
from eternafold._core import bpps as _bpps

__version__ = "0.1.0"
__all__ = [
    "pfunc",
    "predict",
    "fold",
    "energy_of_structure",
    "bpps",
    "get_binary_dir",
    "get_params_dir",
    "get_default_params_file",
    "configure_arnie",
]

_PACKAGE_DIR = Path(__file__).parent


def get_binary_dir() -> str:
    """Return the directory containing the contrafold CLI binary.

    This is useful for configuring Arnie or other tools that shell out
    to the contrafold command-line interface.
    """
    return str(_PACKAGE_DIR / "bin")


def get_params_dir() -> str:
    """Return the directory containing bundled EternaFold parameter files."""
    return str(_PACKAGE_DIR / "parameters")


def get_default_params_file() -> str:
    """Return the path to the default EternaFold parameter file."""
    return str(_PACKAGE_DIR / "parameters" / "EternaFoldParams.v1")


def configure_arnie() -> None:
    """Set environment variables so Arnie can find EternaFold.

    After calling this, Arnie's ``pfunc(seq, package='eternafold')``
    and related functions will work without additional configuration.

    Sets ``ETERNAFOLD_PATH`` to the directory containing the bundled
    ``contrafold`` binary, and ``ETERNAFOLD_PARAMETERS`` to the default
    parameter file.
    """
    os.environ["ETERNAFOLD_PATH"] = get_binary_dir()
    os.environ["ETERNAFOLD_PARAMETERS"] = get_default_params_file()


def pfunc(
    sequence: str,
    constraints: str = "?",
    *,
    param_file: str = "",
) -> float:
    """Compute the log partition function for an RNA sequence.

    Args:
        sequence: RNA sequence (ACGU characters).
        constraints: Structure constraints in dot-bracket notation.
            Use ``'?'`` for fully unconstrained (default).
        param_file: Path to a custom parameter file. If empty,
            uses the compiled-in defaults.

    Returns:
        Log partition coefficient.
    """
    return _pfunc(sequence, constraints, param_file)


def predict(
    sequence: str,
    constraints: str = "?",
    *,
    param_file: str = "",
    gamma: float = 6.0,
) -> str:
    """Predict the MEA secondary structure for an RNA sequence.

    Args:
        sequence: RNA sequence (ACGU characters).
        constraints: Structure constraints in dot-bracket notation.
            Use ``'?'`` for fully unconstrained (default).
        param_file: Path to a custom parameter file. If empty,
            uses the compiled-in defaults.
        gamma: Sensitivity/specificity tradeoff parameter (default 6.0).
            Higher values emphasize sensitivity.

    Returns:
        Predicted structure in dot-bracket notation.
    """
    return _predict(sequence, constraints, param_file, gamma)


def fold(
    sequence: str,
    constraints: str = "?",
    *,
    param_file: str = "",
    gamma: float = 6.0,
) -> tuple[str, float]:
    """Predict structure and compute its free energy estimate.

    Convenience function that calls :func:`predict` and
    :func:`energy_of_structure`.

    Args:
        sequence: RNA sequence (ACGU characters).
        constraints: Structure constraints in dot-bracket notation.
        param_file: Path to a custom parameter file.
        gamma: Sensitivity/specificity tradeoff parameter.

    Returns:
        Tuple of (structure, energy) where structure is a dot-bracket
        string and energy is the log-ratio of constrained to
        unconstrained partition functions.
    """
    structure = predict(sequence, constraints, param_file=param_file, gamma=gamma)
    energy = energy_of_structure(sequence, structure, param_file=param_file)
    return structure, energy


def energy_of_structure(
    sequence: str,
    structure: str,
    *,
    param_file: str = "",
) -> float:
    """Compute the free energy estimate of a specific structure.

    Returns the difference in log partition function between the
    constrained (given structure) and unconstrained ensembles.

    Args:
        sequence: RNA sequence (ACGU characters).
        structure: Structure in dot-bracket notation.
        param_file: Path to a custom parameter file.

    Returns:
        Energy estimate (log Z_constrained - log Z_unconstrained).
    """
    log_z = _pfunc(sequence, "?", param_file)
    log_z_constrained = _pfunc(sequence, structure, param_file)
    return log_z_constrained - log_z


def bpps(
    sequence: str,
    constraints: str = "?",
    *,
    param_file: str = "",
) -> list[list[float]]:
    """Compute the base pair probability matrix for an RNA sequence.

    Args:
        sequence: RNA sequence (ACGU characters).
        constraints: Structure constraints in dot-bracket notation.
        param_file: Path to a custom parameter file.

    Returns:
        NxN matrix of base pair probabilities as a list of lists.
    """
    return _bpps(sequence, constraints, param_file)
