# ASO Atlas vs ASOptimizer: Dataset Compatibility Analysis

Can [ASO Atlas](https://github.com/barneyhill/aso_atlas) (190K ASO records from USPTO patents) be used to expand [ASOptimizer](https://github.com/Spidercores/ASOptimizer)'s training data (37K records)?

## Quick Answer

**Yes, with caveats:**
- 89% of ASOptimizer sequences already exist in ASO Atlas (same patent sources)
- ~19K records are true duplicates (identical efficacy values)
- ASO Atlas adds **151K new sequences** but lacks SMILES molecular structures
- SMILES can be generated from ASO Atlas's position-level modification data using RDKit

## Files

| File | Description |
|------|-------------|
| `aso_compatibility_summary.ipynb` | **Start here.** Concise 5-question analysis for decision-making |
| `aso_atlas_vs_asoptimizer_report.html` | Interactive HTML report with Plotly charts |
| `generate_report.py` | Script to regenerate the HTML report |
| `setup_data.sh` | Downloads both datasets |

## Setup

```bash
cd aso-dataset-analysis
./setup_data.sh          # clones both repos (~160MB)
pip install pandas matplotlib
jupyter notebook aso_compatibility_summary.ipynb
```

## Key Numbers

| Metric | ASO Atlas | ASOptimizer | Overlap |
|--------|-----------|-------------|---------|
| Records | 190,927 | 36,890 | - |
| Unique sequences | 165,555 | 15,470 | 13,779 |
| Genes | 343 | 22 | 11 |
| True duplicates | - | - | ~19,000 |
| Efficacy correlation (full match) | - | - | r = 0.66 |
| Has SMILES | No | Yes | - |
