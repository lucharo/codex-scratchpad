"""
Region Detection Module

Identifies anomalous REGIONS in star growth, not just individual points.
A region is a continuous time period with anomalous behavior.
"""

import polars as pl
from datetime import datetime
from dataclasses import dataclass
from anomaly_methods import get_detector, DetectionConfig


@dataclass
class AnomalousRegion:
    """Represents a continuous anomalous region in star growth."""

    start_date: datetime
    end_date: datetime
    duration_days: int
    stars_in_region: int
    avg_daily_stars: float
    growth_rate_change: float  # % change from baseline
    confidence: float  # Average confidence from detector
    detected_by: list[str]  # Which methods detected this


def identify_regions(
    df: pl.DataFrame,
    method: str = "ensemble",
    config: DetectionConfig = None,
    min_region_days: int = 1,
    merge_gap_days: int = 2,
) -> list[AnomalousRegion]:
    """
    Identify continuous anomalous regions in star growth.

    Args:
        df: DataFrame with star data (from data_fetcher)
        method: Detection method to use
        config: Optional detection configuration
        min_region_days: Minimum days for a region to be reported
        merge_gap_days: Merge regions separated by less than this many days

    Returns:
        List of AnomalousRegion objects
    """
    # Detect anomalies
    detector = get_detector(method, config)
    result_df = detector.detect(df)

    # Get anomalous points
    anomalies = result_df.filter(pl.col("is_anomaly"))

    if len(anomalies) == 0:
        return []

    # Sort by date
    anomalies = anomalies.sort("date")

    # Group consecutive dates into regions
    regions = []
    current_region_start = None
    current_region_indices = []

    for i, row in enumerate(anomalies.iter_rows(named=True)):
        if current_region_start is None:
            # Start new region
            current_region_start = row["date"]
            current_region_indices = [i]
        else:
            # Check if this point is close to the last one
            days_since_last = (
                row["date"] - anomalies[current_region_indices[-1]]["date"][0]
            ).days

            if days_since_last <= merge_gap_days:
                # Continue current region
                current_region_indices.append(i)
            else:
                # End current region and start new one
                regions.append(
                    _create_region(anomalies, current_region_indices, result_df)
                )
                current_region_start = row["date"]
                current_region_indices = [i]

    # Don't forget the last region
    if current_region_indices:
        regions.append(_create_region(anomalies, current_region_indices, result_df))

    # Filter out regions that are too short
    regions = [r for r in regions if r.duration_days >= min_region_days]

    return regions


def _create_region(
    anomalies: pl.DataFrame, indices: list[int], full_df: pl.DataFrame
) -> AnomalousRegion:
    """Helper to create an AnomalousRegion from anomaly indices."""
    region_data = anomalies[indices]

    start_date = region_data["date"].min()
    end_date = region_data["date"].max()

    # Calculate duration
    duration = (end_date - start_date).days + 1

    # Get stars in this date range from full dataset
    region_full = full_df.filter(
        (pl.col("date") >= start_date) & (pl.col("date") <= end_date)
    )

    stars_in_region = int(region_full["new_stars"].sum())
    avg_daily_stars = region_full["new_stars"].mean()

    # Calculate growth rate change
    # Compare to baseline (overall average excluding this region)
    baseline_avg = full_df.filter(
        (pl.col("date") < start_date) | (pl.col("date") > end_date)
    )["new_stars"].mean()

    if baseline_avg > 0:
        growth_rate_change = ((avg_daily_stars - baseline_avg) / baseline_avg) * 100
    else:
        growth_rate_change = 0.0

    # Get average confidence
    avg_confidence = region_data["confidence"].mean()

    # Determine which methods detected this (for ensemble)
    detected_by = []
    if "is_anomaly_zscore" in region_data.columns:
        if region_data["is_anomaly_zscore"].any():
            detected_by.append("zscore")
        if region_data["is_anomaly_movingaverage"].any():
            detected_by.append("moving_avg")
        if region_data["is_anomaly_ratechange"].any():
            detected_by.append("rate_change")
        if region_data["is_anomaly_isolationforest"].any():
            detected_by.append("isolation_forest")

    return AnomalousRegion(
        start_date=start_date,
        end_date=end_date,
        duration_days=duration,
        stars_in_region=stars_in_region,
        avg_daily_stars=avg_daily_stars,
        growth_rate_change=growth_rate_change,
        confidence=avg_confidence,
        detected_by=detected_by if detected_by else ["unknown"],
    )
