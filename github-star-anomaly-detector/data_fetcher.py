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

    def fetch_stars(self, owner: str, repo: str, max_pages: int = 10) -> pl.DataFrame:
        """
        Fetch star history for a GitHub repository.

        Args:
            owner: Repository owner
            repo: Repository name
            max_pages: Maximum number of pages to fetch (each page has up to 100 stars)

        Returns:
            Polars DataFrame with columns: starred_at, user
        """
        stars = []
        page = 1

        with httpx.Client(timeout=30.0) as client:
            while page <= max_pages:
                url = f"{self.base_url}/repos/{owner}/{repo}/stargazers"
                params = {"page": page, "per_page": 100}

                response = client.get(url, headers=self.headers, params=params)

                if response.status_code != 200:
                    print(f"Error: {response.status_code}")
                    break

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

        # Convert to Polars DataFrame
        df = pl.DataFrame(stars)

        if len(df) > 0:
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
    anomaly_multiplier: float = 5.0
) -> pl.DataFrame:
    """
    Generate synthetic GitHub star data for testing.

    Args:
        n_days: Number of days to generate
        base_rate: Base number of stars per day
        anomaly_days: List of days (0-indexed) to inject anomalies
        anomaly_multiplier: Multiplier for anomaly days

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
    np.random.seed(42)
    new_stars = np.random.poisson(base_rate, n_days).astype(float)

    # Add trend (gradual growth)
    trend = np.linspace(0, base_rate * 0.5, n_days)
    new_stars = new_stars + trend

    # Inject anomalies
    for day in anomaly_days:
        if day < n_days:
            new_stars[day] *= anomaly_multiplier

    # Create DataFrame
    df = pl.DataFrame({
        "date": dates,
        "new_stars": new_stars
    })

    df = df.with_columns(
        pl.col("new_stars").cum_sum().alias("cumulative_stars")
    )

    return df
