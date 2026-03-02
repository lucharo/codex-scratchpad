"""Tests for the eternafold Python package."""

import os
import pytest

import eternafold


class TestPfunc:
    """Test partition function computation."""

    def test_basic_sequence(self):
        """pfunc returns a finite float for a simple sequence."""
        result = eternafold.pfunc("GGGGGAAAAAACCCCC")
        assert isinstance(result, float)
        assert result != 0.0

    def test_with_structure_constraint(self):
        """pfunc with a fully specified structure constraint."""
        result = eternafold.pfunc("GGGGGAAAAAACCCCC", "(((((......)))))")
        assert isinstance(result, float)

    def test_unconstrained_default(self):
        """pfunc with default '?' constraints."""
        r1 = eternafold.pfunc("GGGGGAAAAAACCCCC")
        r2 = eternafold.pfunc("GGGGGAAAAAACCCCC", "?")
        assert r1 == r2


class TestPredict:
    """Test MEA structure prediction."""

    def test_basic_prediction(self):
        """predict returns a dot-bracket string of correct length."""
        seq = "GGGGGAAAAAACCCCC"
        result = eternafold.predict(seq)
        assert isinstance(result, str)
        assert len(result) == len(seq)
        assert all(c in ".()[]{}<>" for c in result)

    def test_hairpin(self):
        """A simple hairpin sequence should form a stem-loop."""
        seq = "GGGGGAAAAAACCCCC"
        result = eternafold.predict(seq)
        # Should have paired bases (parentheses)
        assert "(" in result
        assert ")" in result

    def test_single_stranded(self):
        """A poly-A sequence should be mostly unpaired."""
        seq = "AAAAAAAAAAAAAAA"
        result = eternafold.predict(seq)
        # Mostly dots for homopolymer
        dot_count = result.count(".")
        assert dot_count >= len(seq) // 2


class TestFold:
    """Test the fold convenience function."""

    def test_returns_tuple(self):
        """fold returns a (structure, energy) tuple."""
        seq = "GGGGGAAAAAACCCCC"
        result = eternafold.fold(seq)
        assert isinstance(result, tuple)
        assert len(result) == 2
        structure, energy = result
        assert isinstance(structure, str)
        assert isinstance(energy, float)

    def test_structure_matches_predict(self):
        """fold's structure should match predict's output."""
        seq = "GGGGGAAAAAACCCCC"
        structure_from_fold, _ = eternafold.fold(seq)
        structure_from_predict = eternafold.predict(seq)
        assert structure_from_fold == structure_from_predict


class TestEnergyOfStructure:
    """Test energy of structure computation."""

    def test_structured_vs_unstructured(self):
        """A good structure should have negative energy (favorable)."""
        seq = "GGGGGAAAAAACCCCC"
        energy = eternafold.energy_of_structure(seq, "(((((......)))))")
        assert isinstance(energy, float)
        # The MEA structure should have favorable (negative) energy
        assert energy < 0


class TestBpps:
    """Test base pair probability matrix computation."""

    def test_matrix_shape(self):
        """bpps returns an NxN matrix."""
        seq = "GGGGGAAAAAACCCCC"
        matrix = eternafold.bpps(seq)
        n = len(seq)
        assert len(matrix) == n
        for row in matrix:
            assert len(row) == n

    def test_symmetric(self):
        """The bpp matrix should be symmetric."""
        seq = "GGGGGAAAAAACCCCC"
        matrix = eternafold.bpps(seq)
        n = len(seq)
        for i in range(n):
            for j in range(n):
                assert abs(matrix[i][j] - matrix[j][i]) < 1e-6

    def test_probabilities_in_range(self):
        """All probabilities should be in [0, 1]."""
        seq = "GGGGGAAAAAACCCCC"
        matrix = eternafold.bpps(seq)
        for row in matrix:
            for p in row:
                assert 0.0 <= p <= 1.0 + 1e-6


class TestArnieIntegration:
    """Test Arnie compatibility helpers."""

    def test_get_binary_dir(self):
        """get_binary_dir returns a string path."""
        path = eternafold.get_binary_dir()
        assert isinstance(path, str)
        assert "bin" in path

    def test_get_params_dir(self):
        """get_params_dir returns a string path."""
        path = eternafold.get_params_dir()
        assert isinstance(path, str)
        assert "parameters" in path

    def test_get_default_params_file(self):
        """get_default_params_file returns a path containing EternaFoldParams."""
        path = eternafold.get_default_params_file()
        assert "EternaFoldParams" in path

    def test_configure_arnie(self):
        """configure_arnie sets expected environment variables."""
        eternafold.configure_arnie()
        assert "ETERNAFOLD_PATH" in os.environ
        assert "ETERNAFOLD_PARAMETERS" in os.environ
        assert "bin" in os.environ["ETERNAFOLD_PATH"]
        assert "EternaFoldParams" in os.environ["ETERNAFOLD_PARAMETERS"]
