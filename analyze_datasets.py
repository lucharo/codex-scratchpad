#!/usr/bin/env python3
"""
Exploratory analysis of ASO Atlas vs ASOptimizer datasets.
Run after ./setup_data.sh to clone the data repositories.
"""

import sys
sys.path.insert(0, 'aso_atlas')

import pandas as pd

def load_datasets():
    """Load both ASO datasets."""
    print("Loading ASO Atlas...")
    atlas_df = pd.read_pickle('aso_atlas/data/aso_atlas.pkl')

    print("Loading ASOptimizer...")
    asopt_df = pd.read_csv('ASOptimizer/dataset/experiments_with_smiles.csv', low_memory=False)

    return atlas_df, asopt_df


def analyze_aso_atlas(df):
    """Detailed analysis of ASO Atlas dataset."""
    print("\n" + "="*60)
    print("ASO ATLAS DATASET ANALYSIS")
    print("="*60)

    print(f"\nShape: {df.shape}")
    print(f"Columns ({len(df.columns)}):")
    for col in df.columns:
        print(f"  - {col}: {df[col].dtype}")

    print(f"\nUnique genes: {df['target_gene'].nunique()}")
    print(f"Unique sequences: {df['aso_sequence_5_to_3'].nunique()}")

    print(f"\n--- Efficacy Stats ---")
    print(df['inhibition_percent'].describe())

    print(f"\n--- Top 10 Genes by ASO count ---")
    print(df['target_gene'].value_counts().head(10))

    print(f"\n--- Cell Lines ---")
    print(df['cell_line'].value_counts().head(10))

    print(f"\n--- Sequence Length Distribution ---")
    df['seq_length'] = df['aso_sequence_5_to_3'].str.len()
    print(df['seq_length'].value_counts().sort_index())

    # Chemistry breakdown
    print(f"\n--- Chemistry Breakdown ---")
    df['has_MOE'] = df['chemistry'].astype(str).str.contains('MOE')
    df['has_cEt'] = df['chemistry'].astype(str).str.contains('cEt')
    df['has_LNA'] = df['chemistry'].astype(str).str.contains('LNA')
    print(f"MOE-containing: {df['has_MOE'].sum()}")
    print(f"cEt-containing: {df['has_cEt'].sum()}")
    print(f"LNA-containing: {df['has_LNA'].sum()}")

    print(f"\n--- Sample Chemistry Entries ---")
    for i in range(3):
        print(f"  {df['chemistry'].iloc[i]}")

    return df


def analyze_asoptimizer(df):
    """Detailed analysis of ASOptimizer dataset."""
    print("\n" + "="*60)
    print("ASOPTIMIZER DATASET ANALYSIS")
    print("="*60)

    print(f"\nShape: {df.shape}")
    print(f"Columns: {list(df.columns)}")

    print(f"\nUnique genes: {df['Target_gene'].nunique()}")
    print(f"Unique sequences: {df['Sequence'].nunique()}")

    print(f"\n--- Efficacy Stats ---")
    print(df['Inhibition(%)'].describe())

    print(f"\n--- Top 10 Genes by ASO count ---")
    print(df['Target_gene'].value_counts().head(10))

    print(f"\n--- Cell Lines ---")
    print(df['Cell_line'].value_counts().head(10))

    print(f"\n--- Modification Types ---")
    print(df['Modification'].value_counts())

    print(f"\n--- Chemical Patterns (top 15) ---")
    print(df['Chemical_Pattern'].value_counts().head(15))

    print(f"\n--- Sequence Length Distribution ---")
    print(df['seq_length'].value_counts().sort_index())

    return df


def compare_datasets(atlas_df, asopt_df):
    """Compare overlaps between datasets."""
    print("\n" + "="*60)
    print("DATASET COMPARISON")
    print("="*60)

    # Sequence overlap
    atlas_seqs = set(atlas_df['aso_sequence_5_to_3'].str.upper())
    asopt_seqs = set(asopt_df['Sequence'].str.upper())

    print("\n--- Sequence Overlap ---")
    print(f"Unique sequences in ASO Atlas: {len(atlas_seqs)}")
    print(f"Unique sequences in ASOptimizer: {len(asopt_seqs)}")
    print(f"Sequences in both: {len(atlas_seqs & asopt_seqs)}")
    print(f"Only in ASO Atlas: {len(atlas_seqs - asopt_seqs)}")
    print(f"Only in ASOptimizer: {len(asopt_seqs - atlas_seqs)}")

    # Gene overlap
    atlas_genes = set(atlas_df['target_gene'].str.upper().dropna())
    asopt_genes = set(asopt_df['Target_gene'].str.upper().dropna())

    print(f"\n--- Gene Overlap ---")
    print(f"Unique genes in ASO Atlas: {len(atlas_genes)}")
    print(f"Unique genes in ASOptimizer: {len(asopt_genes)}")
    print(f"Genes in both: {len(atlas_genes & asopt_genes)}")
    print(f"Overlap genes: {atlas_genes & asopt_genes}")
    print(f"Only in ASOptimizer: {asopt_genes - atlas_genes}")

    # Efficacy correlation for overlapping sequences
    print(f"\n--- Efficacy Correlation (same sequences) ---")
    atlas_df['seq_upper'] = atlas_df['aso_sequence_5_to_3'].str.upper()
    asopt_df['seq_upper'] = asopt_df['Sequence'].str.upper()

    merged = atlas_df.merge(asopt_df, on='seq_upper', suffixes=('_atlas', '_asopt'))
    print(f"Merged records (same sequence): {len(merged)}")

    if len(merged) > 0:
        corr = merged['inhibition_percent'].corr(merged['Inhibition(%)'])
        print(f"Pearson correlation: {corr:.3f}")

        print(f"\nSample comparison (same sequence, different measurements):")
        sample_cols = ['seq_upper', 'inhibition_percent', 'Inhibition(%)',
                       'target_gene', 'Target_gene', 'cell_line', 'Cell_line']
        available = [c for c in sample_cols if c in merged.columns]
        print(merged[available].head(10).to_string())


def analyze_efficacy_by_features(atlas_df, asopt_df):
    """Analyze what features correlate with high efficacy."""
    print("\n" + "="*60)
    print("EFFICACY ANALYSIS BY FEATURES")
    print("="*60)

    # ASO Atlas: efficacy by chemistry type
    print("\n--- ASO Atlas: Mean Efficacy by Chemistry ---")
    atlas_df['has_MOE'] = atlas_df['chemistry'].astype(str).str.contains('MOE')
    atlas_df['has_cEt'] = atlas_df['chemistry'].astype(str).str.contains('cEt')

    moe_only = atlas_df[atlas_df['has_MOE'] & ~atlas_df['has_cEt']]['inhibition_percent'].mean()
    cet_only = atlas_df[~atlas_df['has_MOE'] & atlas_df['has_cEt']]['inhibition_percent'].mean()
    both = atlas_df[atlas_df['has_MOE'] & atlas_df['has_cEt']]['inhibition_percent'].mean()

    print(f"MOE only: {moe_only:.1f}%")
    print(f"cEt only: {cet_only:.1f}%")
    print(f"MOE + cEt: {both:.1f}%")

    # ASOptimizer: efficacy by modification type
    print("\n--- ASOptimizer: Mean Efficacy by Modification ---")
    print(asopt_df.groupby('Modification')['Inhibition(%)'].agg(['mean', 'std', 'count']).round(1))

    # Efficacy by sequence length
    print("\n--- ASO Atlas: Mean Efficacy by Length ---")
    atlas_df['seq_length'] = atlas_df['aso_sequence_5_to_3'].str.len()
    print(atlas_df.groupby('seq_length')['inhibition_percent'].agg(['mean', 'count']).round(1))

    print("\n--- ASOptimizer: Mean Efficacy by Length ---")
    print(asopt_df.groupby('seq_length')['Inhibition(%)'].agg(['mean', 'count']).round(1))


def nucleotide_analysis(atlas_df):
    """Analyze nucleotide composition vs efficacy."""
    print("\n" + "="*60)
    print("NUCLEOTIDE COMPOSITION ANALYSIS")
    print("="*60)

    # High vs low efficacy sequences
    high_eff = atlas_df[atlas_df['inhibition_percent'] >= 70]['aso_sequence_5_to_3']
    low_eff = atlas_df[atlas_df['inhibition_percent'] <= 30]['aso_sequence_5_to_3']

    def calc_composition(sequences):
        all_seq = ''.join(sequences.str.upper())
        total = len(all_seq)
        return {nt: all_seq.count(nt) / total * 100 for nt in 'ACGT'}

    high_comp = calc_composition(high_eff)
    low_comp = calc_composition(low_eff)

    print(f"\nNucleotide composition (%):")
    print(f"{'Nucleotide':<12} {'High Eff (≥70%)':<18} {'Low Eff (≤30%)':<18} {'Difference':<12}")
    print("-" * 60)
    for nt in 'ACGT':
        diff = high_comp[nt] - low_comp[nt]
        print(f"{nt:<12} {high_comp[nt]:<18.1f} {low_comp[nt]:<18.1f} {diff:+.1f}")


if __name__ == "__main__":
    # Load datasets
    atlas_df, asopt_df = load_datasets()

    # Individual analyses
    atlas_df = analyze_aso_atlas(atlas_df)
    asopt_df = analyze_asoptimizer(asopt_df)

    # Comparison
    compare_datasets(atlas_df, asopt_df)

    # Feature analysis
    analyze_efficacy_by_features(atlas_df, asopt_df)

    # Nucleotide analysis
    nucleotide_analysis(atlas_df)

    print("\n" + "="*60)
    print("Analysis complete!")
    print("="*60)
