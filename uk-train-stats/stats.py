"""Statistics calculations for train performance data."""

from dataclasses import dataclass
from datetime import date, timedelta

import polars as pl


@dataclass
class PerformanceStats:
    """Aggregated performance statistics."""
    total_services: int
    on_time_pct: float
    late_pct: float
    very_late_pct: float
    cancelled_pct: float
    delay_repay_pct: float
    avg_delay_mins: float | None
    median_delay_mins: float | None

    def to_dict(self, period: str) -> dict:
        """Convert to dict for table display."""
        if self.total_services == 0:
            return {"Period": period, "Services": 0}
        return {
            "Period": period,
            "Services": self.total_services,
            "On Time": f"{self.on_time_pct:.1f}%",
            "Late": f"{self.late_pct:.1f}%",
            "Very Late": f"{self.very_late_pct:.1f}%",
            "Cancelled": f"{self.cancelled_pct:.1f}%",
            "Delay Repay": f"{self.delay_repay_pct:.1f}%",
            "Avg Delay": f"{self.avg_delay_mins:.1f} min" if self.avg_delay_mins else "N/A",
            "Median": f"{self.median_delay_mins:.0f} min" if self.median_delay_mins else "N/A",
        }


def filter_by_date_range(df: pl.DataFrame, days_back: int) -> pl.DataFrame:
    """Filter DataFrame to services within the last N days."""
    cutoff = date.today() - timedelta(days=days_back)
    return df.filter(pl.col("run_date") >= cutoff)


def filter_by_weekday(df: pl.DataFrame, weekday: int) -> pl.DataFrame:
    """Filter DataFrame to services on a specific weekday (0=Monday)."""
    return df.filter(pl.col("weekday") == weekday)


def filter_by_time_slot(df: pl.DataFrame, dep_slot: str) -> pl.DataFrame:
    """Filter DataFrame to services in a specific departure time slot."""
    return df.filter(pl.col("dep_slot") == dep_slot)


def calculate_stats(df: pl.DataFrame) -> PerformanceStats:
    """Calculate performance statistics from a DataFrame."""
    if df.is_empty():
        return PerformanceStats(0, 0, 0, 0, 0, 0, None, None)

    total = len(df)
    non_cancelled = df.filter(~pl.col("cancelled"))

    return PerformanceStats(
        total_services=total,
        on_time_pct=100 * len(df.filter(pl.col("status") == "on_time")) / total,
        late_pct=100 * len(df.filter(pl.col("status") == "late")) / total,
        very_late_pct=100 * len(df.filter(pl.col("status") == "very_late")) / total,
        cancelled_pct=100 * len(df.filter(pl.col("status") == "cancelled")) / total,
        delay_repay_pct=100 * len(df.filter(pl.col("delay_repay_eligible"))) / total,
        avg_delay_mins=non_cancelled["arr_delay_mins"].mean() if not non_cancelled.is_empty() else None,
        median_delay_mins=non_cancelled["arr_delay_mins"].median() if not non_cancelled.is_empty() else None,
    )


def get_route_stats(df: pl.DataFrame, days_back: int) -> PerformanceStats:
    """Get stats for a route over the last N days."""
    filtered = filter_by_date_range(df, days_back)
    return calculate_stats(filtered)


def get_weekday_stats(df: pl.DataFrame, weekday: int, days_back: int = 365) -> PerformanceStats:
    """Get stats for a specific weekday over the last N days."""
    filtered = filter_by_date_range(df, days_back)
    filtered = filter_by_weekday(filtered, weekday)
    return calculate_stats(filtered)


def get_time_slot_stats(df: pl.DataFrame, dep_slot: str, days_back: int = 365) -> PerformanceStats:
    """Get stats for a specific time slot over the last N days."""
    filtered = filter_by_date_range(df, days_back)
    filtered = filter_by_time_slot(filtered, dep_slot)
    return calculate_stats(filtered)


def get_delay_distribution(df: pl.DataFrame, days_back: int = 365) -> pl.DataFrame:
    """Get delay values for distribution analysis (excludes cancelled)."""
    filtered = filter_by_date_range(df, days_back)
    return filtered.filter(
        (~pl.col("cancelled")) & (pl.col("arr_delay_mins").is_not_null())
    ).select("arr_delay_mins").sort("arr_delay_mins")


def get_heatmap_data(df: pl.DataFrame, days_back: int = 365) -> pl.DataFrame:
    """Get average delay by weekday and hour for heatmap."""
    filtered = filter_by_date_range(df, days_back)
    return (
        filtered
        .filter(~pl.col("cancelled"))
        .group_by(["weekday", "dep_hour"])
        .agg([
            pl.col("arr_delay_mins").mean().alias("avg_delay"),
            pl.len().alias("n_services"),
        ])
        .filter(pl.col("n_services") >= 5)
    )


def calculate_delay_probabilities(df: pl.DataFrame) -> dict[str, float]:
    """
    Calculate probability of delays at key thresholds.

    Returns dict with keys: on_time, under_15, under_30, under_60
    (as percentages)
    """
    if df.is_empty():
        return {"on_time": 0, "under_15": 0, "under_30": 0, "under_60": 0}

    n = len(df)
    return {
        "on_time": 100 * len(df.filter(pl.col("arr_delay_mins") <= 5)) / n,
        "under_15": 100 * len(df.filter(pl.col("arr_delay_mins") <= 15)) / n,
        "under_30": 100 * len(df.filter(pl.col("arr_delay_mins") <= 30)) / n,
        "under_60": 100 * len(df.filter(pl.col("arr_delay_mins") <= 60)) / n,
    }
