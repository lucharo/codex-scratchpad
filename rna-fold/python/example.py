"""
Example usage of the rna_fold library

This demonstrates how to use the LinearFold and EternaFold algorithms
from Python to predict RNA secondary structures.
"""

try:
    from rna_fold import LinearFold, EternaFold
except ImportError:
    print("Error: rna_fold module not installed")
    print("Run: maturin develop --features python")
    print("Or: pip install -e .")
    exit(1)


def main():
    """Main example function"""

    # Example RNA sequences
    sequences = {
        "Simple hairpin": "GGGGAAAACCCC",
        "tRNA-like": "GCGGAUUUAGCUCAGUUGGGAGAGCGCCAGACUGAAGAUCUGGAGGUCCUGUGUUCGAUCCACAGAAUUCGCA",
        "Small stem-loop": "CGCGAAAAAAGCGCG",
    }

    print("=" * 80)
    print("RNA Secondary Structure Prediction Examples")
    print("=" * 80)

    for name, seq in sequences.items():
        print(f"\n{name}:")
        print(f"Sequence: {seq}")
        print(f"Length: {len(seq)} nucleotides")

        # LinearFold prediction
        print("\n--- LinearFold (beam search, O(n) time) ---")
        folder_lf = LinearFold(beam_size=100)
        structure_lf = folder_lf.fold(seq)

        print(f"Sequence:  {structure_lf.sequence}")
        print(f"Structure: {structure_lf.structure}")
        print(f"Energy:    {structure_lf.energy:.2f} kcal/mol")
        print(f"Pairs:     {structure_lf.num_pairs}")

        # EternaFold prediction
        print("\n--- EternaFold (ML-based, trained on experimental data) ---")
        folder_ef = EternaFold()
        structure_ef = folder_ef.fold(seq)

        print(f"Sequence:  {structure_ef.sequence}")
        print(f"Structure: {structure_ef.structure}")
        print(f"Energy:    {structure_ef.energy:.2f} kcal/mol")
        print(f"Pairs:     {structure_ef.num_pairs}")

        # Partition function
        z = folder_ef.partition_function(seq)
        print(f"Partition function: {z:.2e}")

        print("\n" + "-" * 80)

    # Demonstrate error handling
    print("\n" + "=" * 80)
    print("Error Handling Example")
    print("=" * 80)

    try:
        folder = LinearFold()
        # Invalid sequence (contains 'X')
        structure = folder.fold("GGGXAAACCC")
    except ValueError as e:
        print(f"✓ Caught expected error: {e}")

    print("\nDone!")


if __name__ == "__main__":
    main()
