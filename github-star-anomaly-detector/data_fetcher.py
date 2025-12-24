"""
GitHub Star Data Fetcher

Fetches GitHub repository star history using the GitHub API.
The GitHub API provides star events with timestamps, allowing us to
reconstruct the star count over time.
"""

import httpx
import polars as pl
from datetime import datetime
from typing import Optional
import time


class GitHubAPIError(Exception):
    """Exception raised for GitHub API errors."""
    pass


class RateLimitError(GitHubAPIError):
    """Exception raised when rate limit is exceeded."""
    pass


class GitHubStarFetcher:
    """Fetches GitHub repository star history."""

    def __init__(self, token: Optional[str] = None):
        """
        Initialize the fetcher.

        Args:
            token: Optional GitHub personal access token for higher rate limits
        """
        self.base_url = "https://api.github.com"
        self.headers = {
            "Accept": "application/vnd.github.v3.star+json",
        }
        if token:
            self.headers["Authorization"] = f"token {token}"

    def fetch_stars(
        self,
        owner: str,
        repo: str,
        max_pages: int = 10,
        retry_on_rate_limit: bool = True
    ) -> pl.DataFrame:
        """
        Fetch star history for a GitHub repository.

        Args:
            owner: Repository owner
            repo: Repository name
            max_pages: Maximum number of pages to fetch (each page has up to 100 stars)
            retry_on_rate_limit: If True, wait and retry when rate limited

        Returns:
            Polars DataFrame with columns: starred_at, user

        Raises:
            GitHubAPIError: For API errors (404, 403, etc.)
            RateLimitError: When rate limit is exceeded and retry_on_rate_limit is False
        """
        stars = []
        page = 1

        with httpx.Client(timeout=30.0) as client:
            while page <= max_pages:
                url = f"{self.base_url}/repos/{owner}/{repo}/stargazers"
                params = {"page": page, "per_page": 100}

                try:
                    response = client.get(url, headers=self.headers, params=params)

                    # Handle rate limiting
                    if response.status_code == 429:
                        if retry_on_rate_limit:
                            reset_time = int(response.headers.get('X-RateLimit-Reset', 0))
                            wait_time = max(reset_time - time.time(), 60)
                            print(f"Rate limited. Waiting {wait_time:.0f} seconds...")
                            time.sleep(wait_time)
                            continue
                        else:
                            raise RateLimitError("GitHub API rate limit exceeded")

                    # Handle other errors
                    if response.status_code == 404:
                        raise GitHubAPIError(f"Repository {owner}/{repo} not found")
                    elif response.status_code == 403:
                        raise GitHubAPIError("Access forbidden. Check your API token permissions.")
                    elif response.status_code != 200:
                        raise GitHubAPIError(f"API error: {response.status_code} - {response.text}")

                    data = response.json()

                    if not data:
                        break

                    for item in data:
                        stars.append({
                            "starred_at": item["starred_at"],
                            "user": item["user"]["login"]
                        })

                    print(f"Fetched page {page}, total stars: {len(stars)}")
                    page += 1

                except httpx.TimeoutException:
                    raise GitHubAPIError("Request timed out. Please try again.")
                except httpx.NetworkError as e:
                    raise GitHubAPIError(f"Network error: {str(e)}")

        if not stars:
            return pl.DataFrame(schema={"starred_at": pl.Datetime, "user": pl.Utf8})

        # Convert to Polars DataFrame
        df = pl.DataFrame(stars)
        df = df.with_columns(
            pl.col("starred_at").str.to_datetime()
        ).sort("starred_at")

        return df

    def create_time_series(self, stars_df: pl.DataFrame, freq: str = "1d") -> pl.DataFrame:
        """
        Convert star events to a time series with cumulative star counts.

        Args:
            stars_df: DataFrame from fetch_stars
            freq: Frequency for resampling (e.g., "1d" for daily, "1h" for hourly)

        Returns:
            Polars DataFrame with columns: date, cumulative_stars, new_stars
        """
        if len(stars_df) == 0:
            return pl.DataFrame()

        # Add a count column
        df = stars_df.with_columns(pl.lit(1).alias("count"))

        # Group by date and count
        df = df.group_by_dynamic("starred_at", every=freq).agg(
            pl.col("count").sum().alias("new_stars")
        )

        # Calculate cumulative stars
        df = df.with_columns(
            pl.col("new_stars").cum_sum().alias("cumulative_stars")
        ).rename({"starred_at": "date"})

        return df


def generate_synthetic_star_data(
    n_days: int = 365,
    base_rate: float = 10.0,
    anomaly_days: Optional[list[int]] = None,
    anomaly_multiplier: float = 5.0,
    growth_pattern: str = "linear",
    add_seasonality: bool = True,
    seed: int = 42
) -> pl.DataFrame:
    """
    Generate synthetic GitHub star data for testing.

    Args:
        n_days: Number of days to generate
        base_rate: Base number of stars per day
        anomaly_days: List of days (0-indexed) to inject anomalies
        anomaly_multiplier: Multiplier for anomaly days
        growth_pattern: One of "linear", "exponential", "logarithmic", "viral"
        add_seasonality: Add weekly seasonality (weekends have fewer stars)
        seed: Random seed for reproducibility

    Returns:
        Polars DataFrame with columns: date, new_stars, cumulative_stars
    """
    import numpy as np

    if anomaly_days is None:
        anomaly_days = [100, 200, 300]

    dates = pl.date_range(
        start=datetime(2024, 1, 1),
        end=datetime(2024, 12, 31),
        interval="1d",
        eager=True
    )[:n_days]

    # Generate base star counts with some noise
    np.random.seed(seed)
    new_stars = np.random.poisson(base_rate, n_days).astype(float)

    # Add growth trend based on pattern
    if growth_pattern == "linear":
        # Steady linear growth
        trend = np.linspace(0, base_rate * 0.5, n_days)
        new_stars = new_stars + trend

    elif growth_pattern == "exponential":
        # Exponential growth (viral growth)
        growth_rate = 0.003  # 0.3% daily growth
        trend = base_rate * np.exp(growth_rate * np.arange(n_days)) - base_rate
        new_stars = new_stars + trend

    elif growth_pattern == "logarithmic":
        # Logarithmic growth (saturation)
        trend = base_rate * np.log1p(np.arange(n_days) / 30)
        new_stars = new_stars + trend

    elif growth_pattern == "viral":
        # Viral spike: exponential growth followed by decay
        peak_day = n_days // 3
        x = np.arange(n_days) - peak_day
        # Gaussian-like spike
        viral_curve = base_rate * 3 * np.exp(-0.001 * x**2)
        new_stars = new_stars + viral_curve

    # Add weekly seasonality (if enabled)
    if add_seasonality:
        # Weekends (Saturday=5, Sunday=6) have 30% fewer stars
        day_of_week = np.arange(n_days) % 7
        weekend_effect = np.where((day_of_week == 5) | (day_of_week == 6), 0.7, 1.0)
        new_stars = new_stars * weekend_effect

    # Inject anomalies
    for day in anomaly_days:
        if day < n_days:
            new_stars[day] *= anomaly_multiplier

    # Ensure non-negative values
    new_stars = np.maximum(new_stars, 0)

    # Create DataFrame
    df = pl.DataFrame({
        "date": dates,
        "new_stars": new_stars
    })

    df = df.with_columns(
        pl.col("new_stars").cum_sum().alias("cumulative_stars")
    )

    return df
