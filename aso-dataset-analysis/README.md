# ASO Atlas vs ASOptimizer: Dataset Compatibility Analysis

Can [ASO Atlas](https://github.com/barneyhill/aso_atlas) (190K ASO records from USPTO patents) be used to expand [ASOptimizer](https://github.com/Spidercores/ASOptimizer)'s training data (37K records)?

**[View Interactive Report](https://raw.githack.com/lucharo/codex-scratchpad/claude/aso-dataset-analysis-report-06su9/aso-dataset-analysis/aso_atlas_vs_asoptimizer_report.html)**

## Quick Answer

**Yes, with caveats:**
- 89% of ASOptimizer sequences already exist in ASO Atlas (same patent sources)
- ~19K records are true duplicates (identical efficacy values)
- ASO Atlas adds **151K new sequences** but lacks SMILES molecular structures
- SMILES can be generated from ASO Atlas's position-level modification data using RDKit

## View Interactive HTML Report or Simplified Action-Biased EDA Notebook

| Resource | Link | Description |
|----------|------|-------------|
| **Interactive HTML Report** | [View Report](https://raw.githack.com/lucharo/codex-scratchpad/claude/aso-dataset-analysis-report-06su9/aso-dataset-analysis/aso_atlas_vs_asoptimizer_report.html) | Plotly charts, tables, key findings - no setup required |
| **Marimo EDA Notebook** | [View Source](https://github.com/lucharo/codex-scratchpad/blob/claude/aso-dataset-analysis-report-06su9/aso-dataset-analysis/aso_compatibility_summary.py) | Action-biased analysis answering 5 key questions |
| **Jupyter Version** | [View Notebook](https://github.com/lucharo/codex-scratchpad/blob/claude/aso-dataset-analysis-report-06su9/aso-dataset-analysis/aso_compatibility_summary.ipynb) | Same analysis in Jupyter format |

## Usage

```bash
# Quick start - run everything as a pipeline
cd aso-dataset-analysis
./setup_data.sh              # download datasets to ./aso_atlas and ./ASOptimizer
uv sync                      # install dependencies
uv run marimo run aso_compatibility_summary.py   # run as script (no UI)
uv run marimo edit aso_compatibility_summary.py  # interactive notebook
```

Or download data to /tmp:
```bash
DATA_DIR=/tmp ./setup_data.sh
```

Regenerate the HTML report:
```bash
uv run python generate_report.py
```

## Files

| File | Description |
|------|-------------|
| `aso_compatibility_summary.py` | **Start here.** Marimo notebook - concise 5-question analysis |
| `aso_compatibility_summary.ipynb` | Same analysis as Jupyter notebook |
| `aso_atlas_vs_asoptimizer_report.html` | Interactive HTML report with Plotly charts |
| `generate_report.py` | Script to regenerate the HTML report |
| `setup_data.sh` | Downloads both datasets |

## Key Numbers

| Metric | ASO Atlas | ASOptimizer | Overlap |
|--------|-----------|-------------|---------|
| Records | 190,927 | 36,890 | - |
| Unique sequences | 165,555 | 15,470 | 13,779 |
| Genes | 343 | 22 | 11 |
| True duplicates | - | - | ~19,000 |
| Efficacy correlation (full match) | - | - | r = 0.66 |
| Has SMILES | No | Yes | - |
