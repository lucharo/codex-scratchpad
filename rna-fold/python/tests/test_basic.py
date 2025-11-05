"""
Basic tests for the rna_fold Python bindings
"""

import pytest

try:
    from rna_fold import LinearFold, EternaFold, SecondaryStructure
except ImportError:
    pytest.skip("rna_fold not installed, run: maturin develop --features python", allow_module_level=True)


class TestLinearFold:
    """Tests for LinearFold algorithm"""

    def test_create_folder(self):
        """Test creating a LinearFold instance"""
        folder = LinearFold(beam_size=50)
        assert folder is not None

    def test_fold_simple_sequence(self):
        """Test folding a simple sequence"""
        folder = LinearFold()
        structure = folder.fold("GGGGAAAACCCC")

        assert structure.sequence == "GGGGAAAACCCC"
        assert len(structure.structure) == len(structure.sequence)
        assert structure.energy <= 0.0  # Energy should be negative (stable)
        assert structure.num_pairs >= 0

    def test_fold_longer_sequence(self):
        """Test folding a longer sequence"""
        folder = LinearFold(beam_size=100)
        # Simple tRNA-like sequence
        seq = "GCGGAUUUAGCUCAGUUGGGAGAGC"
        structure = folder.fold(seq)

        assert structure.sequence == seq
        assert len(structure.structure) == len(seq)

    def test_invalid_sequence(self):
        """Test that invalid sequences raise errors"""
        folder = LinearFold()

        with pytest.raises(ValueError):
            folder.fold("GGGXAAACCC")  # X is not a valid base

    def test_structure_representation(self):
        """Test string representation of structure"""
        folder = LinearFold()
        structure = folder.fold("GGGGAAAACCCC")

        str_repr = str(structure)
        assert "GGGGAAAACCCC" in str_repr

        repr_str = repr(structure)
        assert "SecondaryStructure" in repr_str


class TestEternaFold:
    """Tests for EternaFold algorithm"""

    def test_create_folder(self):
        """Test creating an EternaFold instance"""
        folder = EternaFold()
        assert folder is not None

    def test_fold_simple_sequence(self):
        """Test folding a simple sequence"""
        folder = EternaFold()
        structure = folder.fold("GGGGAAAACCCC")

        assert structure.sequence == "GGGGAAAACCCC"
        assert len(structure.structure) == len(structure.sequence)
        assert structure.num_pairs >= 0

    def test_partition_function(self):
        """Test partition function calculation"""
        folder = EternaFold()
        z = folder.partition_function("GGGGAAAACCCC")

        assert z > 0.0  # Partition function should be positive

    def test_invalid_sequence(self):
        """Test that invalid sequences raise errors"""
        folder = EternaFold()

        with pytest.raises(ValueError):
            folder.fold("INVALID123")


class TestComparison:
    """Compare LinearFold and EternaFold predictions"""

    def test_same_length_predictions(self):
        """Both algorithms should produce same-length structures"""
        seq = "GGGGAAAACCCC"

        lf = LinearFold()
        ef = EternaFold()

        structure_lf = lf.fold(seq)
        structure_ef = ef.fold(seq)

        assert len(structure_lf.structure) == len(structure_ef.structure)
        assert len(structure_lf.structure) == len(seq)

    def test_valid_dot_bracket(self):
        """Both algorithms should produce valid dot-bracket notation"""
        seq = "GGGGAAAACCCC"

        lf = LinearFold()
        ef = EternaFold()

        for folder in [lf, ef]:
            structure = folder.fold(seq)

            # Check that structure only contains valid characters
            assert all(c in "()." for c in structure.structure)

            # Check that brackets are balanced
            open_count = structure.structure.count("(")
            close_count = structure.structure.count(")")
            assert open_count == close_count


class TestEdgeCases:
    """Test edge cases and special inputs"""

    def test_empty_like_sequence(self):
        """Test very short sequences"""
        folder = LinearFold()

        # Very short sequence (might not form pairs)
        structure = folder.fold("GC")
        assert len(structure.structure) == 2

    def test_all_same_base(self):
        """Test sequence with all same bases"""
        folder = LinearFold()

        # All A's - cannot pair with themselves
        structure = folder.fold("AAAAAAAA")
        assert structure.num_pairs == 0  # No pairs possible

    def test_case_insensitive(self):
        """Test that both upper and lower case work"""
        folder = LinearFold()

        # Both should work
        structure1 = folder.fold("GGGGAAAACCCC")
        structure2 = folder.fold("ggggaaaacccc")

        # Should produce same structure
        assert structure1.structure == structure2.structure


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
