# UK Train Reliability Stats

A lean marimo notebook app for checking historical performance of UK train routes.

## Quick Start

```bash
cd uk-train-stats

# Run the app (auto-installs dependencies)
uvx marimo run train_stats.py

# Or edit interactively
uvx marimo edit train_stats.py
```

## Structure

```
uk-train-stats/
├── train_stats.py   # Main marimo notebook (UI + plots)
├── hsp_api.py       # HSP API client for live data
├── demo_data.py     # Synthetic data generator
├── stats.py         # Statistics calculations
└── README.md
```

## Features

- **Route selection**: Choose origin/destination from common UK stations
- **Time-slot analysis**: Performance for specific departure times (30-min buckets)
- **Historical stats**: 7/30/365 day performance summaries
- **Day-of-week breakdown**: See which days perform best/worst
- **Delay distribution**: CDF chart with refund threshold markers
- **Heatmap**: Visual guide for when to travel

## Data

**Demo mode** (default): Uses realistic synthetic data - just click "Load Data".

**Live data**: Register at [opendata.nationalrail.co.uk](https://opendata.nationalrail.co.uk), enable HSP access, then integrate credentials into `hsp_api.py`.

## Metrics

| Status | Definition |
|--------|------------|
| On Time | ≤5 min late |
| Late | 5-29 min late |
| Very Late | 30+ min late |
| Delay Repay | 15+ min (compensation eligible) |

## Refund Thresholds

| Delay | Typical Compensation |
|-------|---------------------|
| 15-29 min | 25% |
| 30-59 min | 50% |
| 60+ min | 100% |

*Varies by operator.*

## Tech

- **marimo**: Reactive notebook
- **Polars**: DataFrames
- **Altair**: Charts
