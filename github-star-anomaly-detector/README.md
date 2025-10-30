# Fake Stars Detector

A full-stack application for detecting potentially fake GitHub stars using time-series anomaly detection combined with account verification. Visit `fakestars.com/owner/repo` to analyze any GitHub repository.

## 🚀 Full-Stack Application

**Vision:** Visit `/pytorch/pytorch` and instantly see:
- Anomalous regions in star growth
- Account verification for suspicious periods
- Real-time fake star percentage estimate
- Beautiful, minimal UI

### Quick Start

```bash
# Terminal 1: Start backend
cd github-star-anomaly-detector
uv run uvicorn api.main:app --reload

# Terminal 2: Start frontend
cd web
npm install
npm run dev

# Visit http://localhost:3000
```

## Architecture

### Two-Layer Detection System

**Layer 1: Region Anomaly Detection**
- Detects anomalous REGIONS in star growth rate (not just individual points)
- Uses ensemble of 4 algorithms (Z-Score, Moving Average, Rate Change, Isolation Forest)
- Identifies continuous time periods with suspicious growth patterns

**Layer 2: Account Verification**
- Async verification of accounts that starred during anomalous periods
- Checks for bot-like characteristics:
  - Account age (<90 days)
  - No repositories or activity
  - Generic profile (no name, bio, company)
  - Zero followers (isolated account)
- Returns confidence score and fake percentage estimate

### Components

**Backend (FastAPI + Python)**
- `api/main.py` - FastAPI server with `/analyze/{owner}/{repo}` endpoint
- `account_verifier.py` - Async account verification with parallel checks
- `region_detector.py` - Identifies continuous anomalous regions
- `data_fetcher.py` - GitHub API integration
- `anomaly_methods.py` - 5 detection algorithms with unified interface

**Frontend (Next.js 15 + Tailwind)**
- `web/` - Minimal, elegant UI
- Home page with search
- Dynamic route `/[owner]/[repo]` for analysis results
- Real-time loading states
- Responsive cards showing regions and fake star estimates

## Features

- **5 Detection Algorithms with Unified Interface**:
  - Z-Score based statistical anomaly detection
  - Moving Average deviation detection
  - Rate of Change analysis
  - Isolation Forest (ML-based)
  - Ensemble method using voting across all algorithms

- **Account Verification**: Async verification of GitHub accounts for bot-like characteristics
- **Region Detection**: Identifies continuous anomalous time periods, not just individual points
- **Interactive Marimo Notebooks**: Built with reactive cells and Altair visualizations
- **Realistic Synthetic Data**: Generate data with multiple growth patterns (linear, exponential, logarithmic, viral) and seasonality
- **Real GitHub Data**: Fetch actual repository star history via GitHub API with proper error handling
- **Confidence Scores**: Each detection includes a confidence metric (0.0-1.0)
- **Extensible Architecture**: Easy to add new detection methods via strategy pattern

## Installation

This project uses `uv` for dependency management:

```bash
# Clone the repository
cd github-star-anomaly-detector

# Dependencies are managed by uv
# Run notebooks directly:
uv run marimo edit data_exploration.py
# or
uv run marimo edit anomaly_detector.py
```

## Usage

### 1. Data Exploration Notebook

Explore GitHub star data and visualize time series patterns:

```bash
uv run marimo edit data_exploration.py
```

**Features:**
- Toggle between synthetic and real data sources
- Input fields for GitHub owner/repo
- Optional GitHub API token for higher rate limits
- Visualize cumulative stars and daily growth
- Basic statistical analysis with anomaly preview

### 2. Anomaly Detector Notebook

Interactive anomaly detection with configurable parameters:

```bash
uv run marimo edit anomaly_detector.py
```

**Features:**
- Select detection method from dropdown
- Configure data parameters (days, base rate, anomaly strength)
- Choose growth pattern (linear, exponential, logarithmic, viral)
- Real-time visualization with anomaly highlighting
- Method comparison view for ensemble detector
- Confidence scores for each detection

### 3. Using the Modules Programmatically

```python
from data_fetcher import GitHubStarFetcher, generate_synthetic_star_data
from anomaly_methods import get_detector, DetectionConfig

# Generate realistic synthetic data
synthetic_df = generate_synthetic_star_data(
    n_days=365,
    base_rate=10.0,
    anomaly_days=[50, 150, 250],
    anomaly_multiplier=5.0,
    growth_pattern="exponential",  # or "linear", "logarithmic", "viral"
    add_seasonality=True  # Adds weekend effects
)

# Use any detection method
detector = get_detector('ensemble')  # or 'zscore', 'moving_avg', etc.
result = detector.detect(synthetic_df)
anomalies = result.filter(result['is_anomaly'] == True)

# Or customize configuration
config = DetectionConfig(
    zscore_threshold=3.0,
    ensemble_min_votes=3  # Require 3/4 methods to agree
)
detector = get_detector('ensemble', config)

# Fetch real data from GitHub
fetcher = GitHubStarFetcher(token="your_github_token")
stars_df = fetcher.fetch_stars("owner", "repo", max_pages=10)
time_series = fetcher.create_time_series(stars_df, freq="1d")
```

## Anomaly Detection Methods

All methods implement a unified interface and return consistent results with confidence scores.

### 1. Z-Score Method
Detects values that deviate significantly from the mean using standard deviations.

**Best for**: Identifying extreme outliers in normally distributed data
**Configuration**: `zscore_threshold` (default: 2.5)

### 2. Moving Average Method
Identifies points that deviate from a rolling average and standard deviation.

**Best for**: Capturing deviations from recent trends
**Configuration**: `moving_avg_window` (default: 7), `moving_avg_threshold` (default: 2.0)

### 3. Rate of Change Method
Detects sudden spikes or drops in the rate of star growth.

**Best for**: Identifying sudden changes in growth patterns
**Configuration**: `rate_change_threshold` (default: 3.0)

### 4. Isolation Forest
Machine learning-based anomaly detection that isolates anomalies in feature space.

**Best for**: Complex patterns and non-linear relationships
**Configuration**: `isolation_forest_contamination` (default: 0.05)

### 5. Ensemble Method
Combines all methods using voting. A point is anomalous if ≥N methods agree.

**Best for**: Reducing false positives and improving overall accuracy
**Configuration**: `ensemble_min_votes` (default: 2)

**Why 2 votes?** This balances sensitivity (catching true anomalies) with specificity (avoiding false positives). Requiring just 1 vote catches everything but includes noise; requiring 3-4 votes is very conservative but might miss subtler anomalies.

## Data Sources

### GitHub API
- Provides star creation timestamps with user information
- Rate limited (60 requests/hour without auth, 5000 with auth)
- Each request returns up to 100 stargazers
- Includes automatic rate limit handling and retries
- Proper error handling for 404, 403, and network errors

### Synthetic Data Generation
Multiple realistic growth patterns:
- **Linear**: Steady organic growth
- **Exponential**: Viral growth (0.3% daily increase)
- **Logarithmic**: Growth with saturation
- **Viral**: Gaussian spike (early viral moment followed by decay)

Features:
- Configurable base rate and noise (Poisson distribution)
- Weekly seasonality (weekends have 30% fewer stars)
- Precise anomaly injection on specified days
- Reproducible with seed parameter

## Project Structure

```
github-star-anomaly-detector/
├── README.md                    # This file
├── pyproject.toml              # Project dependencies
├── data_fetcher.py             # Data fetching and generation module
├── anomaly_methods.py          # Detection algorithms with unified interface
├── data_exploration.py         # Marimo notebook for data exploration
├── anomaly_detector.py         # Marimo notebook for anomaly detection
└── test_anomaly_detection.py  # Comprehensive test suite
```

## Architecture

### Strategy Pattern for Detectors
All detection methods inherit from `AnomalyDetector` base class:
- Unified `detect()` interface
- Consistent return format (DataFrame with `is_anomaly` and `confidence` columns)
- Easy to extend with new methods
- Factory function `get_detector()` for instantiation

### Benefits of the Design
1. **Simple notebook cells**: Detection logic is just 3 lines
2. **Testable**: Each detector can be tested independently
3. **Configurable**: Centralized configuration via `DetectionConfig`
4. **Extensible**: Add new detectors by implementing the interface

## Testing

Run the comprehensive test suite:

```bash
uv run python test_anomaly_detection.py
```

Tests include:
- Individual method validation
- Ensemble voting logic
- Detection across different growth patterns
- Confidence score verification

All tests validate with injected anomalies at known positions.

## Verification

All notebooks pass `marimo check`:

```bash
uv run marimo check data_exploration.py  # ✓ PASS
uv run marimo check anomaly_detector.py  # ✓ PASS
```

All tests pass:
```bash
uv run python test_anomaly_detection.py  # ✓ 5/5 tests passed
```

## Use Cases

### Detecting Bot-Generated Stars
Sudden spikes in star counts that deviate from organic growth patterns can indicate:
- Bot activity or fake engagement
- Coordinated starring campaigns
- Paid marketing campaigns
- Going viral on social media (legitimate spike)

### Identifying Growth Patterns
- Steady organic growth vs. artificial inflation
- Launch spikes after product releases
- Conference/presentation effects
- Feature release impacts
- Seasonal variations

### Research Applications
- Study repository popularity dynamics
- Compare growth patterns across projects
- Identify influential events in project history
- Validate marketing campaign effectiveness

## Dependencies

- **marimo**: Interactive notebook environment
- **polars**: Fast data manipulation
- **altair**: Declarative visualization
- **numpy**: Numerical computations
- **scipy**: Scientific computing and statistics
- **scikit-learn**: Machine learning (Isolation Forest)
- **httpx**: HTTP client for API requests with timeout handling

## Design Principles

This project follows clean code principles:
- **Separation of Concerns**: Data fetching, detection logic, and visualization are separate
- **Strategy Pattern**: Unified interface for all detection methods
- **Functional Programming**: Pure functions for detection algorithms
- **Type Hints**: Clear function signatures with Optional types
- **Error Handling**: Specific exceptions with informative messages
- **Testability**: Each component can be tested independently

## Contributing

This project follows the guidelines in the root `AGENTS.md`:
- Uses `uv` for package management
- Prefers Marimo over Jupyter notebooks
- Uses Polars over Pandas for data manipulation
- Visualization preference: Altair > plotnine > seaborn > matplotlib

## License

MIT License - See repository root for details
