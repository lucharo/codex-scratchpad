# UK Train Reliability Stats

A lean marimo notebook app for checking historical performance of UK train routes.

## Quick Start

```bash
# Run the app (auto-installs dependencies)
uvx marimo run train_stats.py

# Or edit interactively
uvx marimo edit train_stats.py
```

## Features

- **Route selection**: Choose any origin/destination pair from common UK stations
- **Time-slot analysis**: Check performance for specific departure times
- **Historical stats**: View 7-day, 30-day, and 365-day performance metrics
- **Day-of-week breakdown**: See which days have the best/worst performance
- **Delay distribution**: Cumulative probability charts showing delay likelihood
- **Refund probability**: See your chances of qualifying for Delay Repay compensation
- **Heatmap**: Visual guide for when to travel to avoid delays

## Data Sources

### Demo Mode (default)
Leave API credentials blank to use realistic simulated data.

### Live Data (HSP API)
1. Register at [opendata.nationalrail.co.uk](https://opendata.nationalrail.co.uk)
2. Enable HSP access in your account settings
3. Enter your credentials in the app

Or set environment variables:
```bash
export NR_EMAIL="your-email@example.com"
export NR_PASSWORD="your-password"
```

## Metrics

| Metric | Definition |
|--------|------------|
| On Time | Arrived within 5 minutes of schedule |
| Late | Arrived 5-29 minutes after schedule |
| Very Late | Arrived 30+ minutes late |
| Delay Repay | 15+ minutes late (eligible for compensation) |

## Refund Thresholds (typical)

| Delay | Compensation |
|-------|--------------|
| 15-29 min | 25% of single fare |
| 30-59 min | 50% of single fare |
| 60+ min | Full refund |

*Note: Varies by train operator. Check your TOC's Delay Repay scheme.*

## Tech Stack

- **marimo**: Reactive notebook/app framework
- **DuckDB**: Embedded analytics database
- **Polars**: Fast DataFrame library
- **Altair**: Declarative visualizations
