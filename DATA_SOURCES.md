# ASO Dataset Sources

This analysis compares two major antisense oligonucleotide (ASO) datasets.

## ASO Atlas
- **Repository**: https://github.com/barneyhill/aso_atlas
- **Size**: 190,927 ASO records from 417 USPTO patents
- **Genes**: 343 unique target genes
- **Format**: Pickle file (`data/aso_atlas.pkl`)
- **Key columns**: `aso_sequence_5_to_3`, `inhibition_percent`, `chemistry`, `target_gene`, `cell_line`
- **Chemistry format**: Position-based modifications (MOE, cEt, PS positions)

## ASOptimizer
- **Repository**: https://github.com/Spidercores/ASOptimizer
- **Paper**: [PMC11066473](https://pmc.ncbi.nlm.nih.gov/articles/PMC11066473/)
- **Size**: 36,890 ASO records from patents/literature via Lens.org
- **Genes**: 22 unique target genes
- **Format**: CSV (`dataset/experiments_with_smiles.csv`)
- **Key columns**: `Sequence`, `Inhibition(%)`, `Modification`, `Chemical_Pattern`, `Smiles`
- **Chemistry format**: SMILES molecular structures + pattern strings (e.g., `CCCddddddddddCCC`)

## Dataset Overlap

| Metric | ASO Atlas | ASOptimizer | Overlap |
|--------|-----------|-------------|---------|
| Total records | 190,927 | 36,890 | - |
| Unique sequences | 165,555 | 15,470 | 13,779 |
| Unique genes | 343 | 22 | 11 |

### Shared Genes
MALAT1, HSD17B13, DGAT2, UBE3A, HIF1A, IRF4, APOL1, IRF5, SNCA, PKK, YAP1

## Setup

```bash
./setup_data.sh
```

## Regenerate Report

```bash
pip install pandas plotly
python generate_report.py
```
