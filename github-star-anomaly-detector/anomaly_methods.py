"""
Anomaly Detection Methods for GitHub Star Data

This module provides a unified interface for different anomaly detection algorithms.
Each detector implements a common interface and returns consistent results.
"""

import polars as pl
import numpy as np
from scipy import stats
from sklearn.ensemble import IsolationForest
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional


@dataclass
class DetectionConfig:
    """Configuration for anomaly detection methods."""

    zscore_threshold: float = 2.5
    moving_avg_window: int = 7
    moving_avg_threshold: float = 2.0
    rate_change_threshold: float = 3.0
    isolation_forest_contamination: float = 0.05
    ensemble_min_votes: int = (
        2  # Minimum methods that must agree for combined detection
    )


class AnomalyDetector(ABC):
    """Base class for all anomaly detection methods."""

    def __init__(self, config: Optional[DetectionConfig] = None):
        """Initialize detector with configuration."""
        self.config = config or DetectionConfig()

    @abstractmethod
    def detect(self, df: pl.DataFrame) -> pl.DataFrame:
        """
        Detect anomalies in the data.

        Args:
            df: DataFrame with 'new_stars' column

        Returns:
            DataFrame with additional columns:
            - is_anomaly: boolean indicating if the point is an anomaly
            - confidence: float between 0 and 1 indicating confidence
        """
        pass

    @property
    @abstractmethod
    def name(self) -> str:
        """Human-readable name of the detector."""
        pass

    @property
    @abstractmethod
    def description(self) -> str:
        """Description of the detection method."""
        pass


class ZScoreDetector(AnomalyDetector):
    """Detect anomalies using Z-score (standard deviations from mean)."""

    @property
    def name(self) -> str:
        return "Z-Score"

    @property
    def description(self) -> str:
        return "Detects values that deviate significantly from the mean using standard deviations"

    def detect(self, df: pl.DataFrame) -> pl.DataFrame:
        """Detect anomalies using Z-score."""
        new_stars = df["new_stars"].to_numpy()
        z_scores = np.abs(stats.zscore(new_stars))

        # Normalize z-scores to confidence (0-1)
        # confidence = min(z_score / (threshold * 2), 1.0)
        confidence = np.minimum(z_scores / (self.config.zscore_threshold * 2), 1.0)

        return df.with_columns(
            [
                pl.Series("z_score", z_scores),
                pl.Series("is_anomaly", z_scores > self.config.zscore_threshold),
                pl.Series("confidence", confidence),
            ]
        )


class MovingAverageDetector(AnomalyDetector):
    """Detect anomalies by comparing to moving average and standard deviation."""

    @property
    def name(self) -> str:
        return "Moving Average"

    @property
    def description(self) -> str:
        return "Identifies points that deviate from a rolling average (captures trend deviations)"

    def detect(self, df: pl.DataFrame) -> pl.DataFrame:
        """Detect anomalies using moving average."""
        result = df.with_columns(
            [
                pl.col("new_stars")
                .rolling_mean(window_size=self.config.moving_avg_window)
                .alias("moving_avg"),
                pl.col("new_stars")
                .rolling_std(window_size=self.config.moving_avg_window)
                .alias("moving_std"),
            ]
        )

        # Calculate deviation from moving average
        result = result.with_columns(
            [
                (
                    (pl.col("new_stars") - pl.col("moving_avg")) / pl.col("moving_std")
                ).alias("ma_deviation")
            ]
        )

        # Calculate confidence
        ma_deviation_abs = result["ma_deviation"].abs().fill_null(0).to_numpy()
        confidence = np.minimum(
            ma_deviation_abs / (self.config.moving_avg_threshold * 2), 1.0
        )

        result = result.with_columns(
            [
                (pl.col("ma_deviation").abs() > self.config.moving_avg_threshold).alias(
                    "is_anomaly"
                ),
                pl.Series("confidence", confidence),
            ]
        )

        return result


class RateChangeDetector(AnomalyDetector):
    """Detect anomalies based on sudden changes in the rate of growth."""

    @property
    def name(self) -> str:
        return "Rate of Change"

    @property
    def description(self) -> str:
        return "Detects sudden spikes or drops in the rate of star growth"

    def detect(self, df: pl.DataFrame) -> pl.DataFrame:
        """Detect anomalies based on rate of change."""
        result = df.with_columns([pl.col("new_stars").diff().alias("rate_change")])

        # Calculate z-score of rate changes
        rate_changes = result["rate_change"].fill_null(0).to_numpy()
        rate_z_scores = np.abs(stats.zscore(rate_changes))

        # Calculate confidence
        confidence = np.minimum(
            rate_z_scores / (self.config.rate_change_threshold * 2), 1.0
        )

        result = result.with_columns(
            [
                pl.Series("rate_z_score", rate_z_scores),
                pl.Series(
                    "is_anomaly", rate_z_scores > self.config.rate_change_threshold
                ),
                pl.Series("confidence", confidence),
            ]
        )

        return result


class IsolationForestDetector(AnomalyDetector):
    """Machine learning-based anomaly detection using Isolation Forest."""

    @property
    def name(self) -> str:
        return "Isolation Forest"

    @property
    def description(self) -> str:
        return "ML-based detector that isolates anomalies in feature space (good for complex patterns)"

    def detect(self, df: pl.DataFrame) -> pl.DataFrame:
        """Detect anomalies using Isolation Forest."""
        features = df.select(["new_stars"]).to_numpy()

        iso_forest = IsolationForest(
            contamination=self.config.isolation_forest_contamination, random_state=42
        )
        predictions = iso_forest.fit_predict(features)
        scores = iso_forest.score_samples(features)

        # Convert scores to confidence (0-1)
        # Isolation forest scores are negative, more negative = more anomalous
        normalized_scores = (scores - scores.min()) / (scores.max() - scores.min())
        confidence = 1.0 - normalized_scores  # Invert so anomalies have high confidence

        return df.with_columns(
            [
                pl.Series("is_anomaly", predictions == -1),
                pl.Series("anomaly_score", scores),
                pl.Series("confidence", confidence),
            ]
        )


class EnsembleDetector(AnomalyDetector):
    """
    Ensemble detector that combines multiple methods.

    Uses voting: a point is an anomaly if at least `min_votes` detectors agree.
    Confidence is the average confidence across all detectors.
    """

    def __init__(self, config: Optional[DetectionConfig] = None):
        """Initialize ensemble with all individual detectors."""
        super().__init__(config)
        self.detectors = [
            ZScoreDetector(config),
            MovingAverageDetector(config),
            RateChangeDetector(config),
            IsolationForestDetector(config),
        ]

    @property
    def name(self) -> str:
        return "Ensemble (Combined)"

    @property
    def description(self) -> str:
        return f"Combines all methods with voting (requires ≥{self.config.ensemble_min_votes} methods to agree)"

    def detect(self, df: pl.DataFrame) -> pl.DataFrame:
        """Detect anomalies using ensemble voting."""
        result = df.clone()
        confidence_cols = []

        # Run all detectors
        for i, detector in enumerate(self.detectors):
            detected = detector.detect(df)
            detector_name = detector.__class__.__name__.replace("Detector", "").lower()

            result = result.with_columns(
                [
                    detected["is_anomaly"].alias(f"is_anomaly_{detector_name}"),
                    detected["confidence"].alias(f"confidence_{detector_name}"),
                ]
            )
            confidence_cols.append(f"confidence_{detector_name}")

        # Count votes
        result = result.with_columns(
            [
                (
                    pl.col("is_anomaly_zscore").cast(pl.Int32)
                    + pl.col("is_anomaly_movingaverage").cast(pl.Int32)
                    + pl.col("is_anomaly_ratechange").cast(pl.Int32)
                    + pl.col("is_anomaly_isolationforest").cast(pl.Int32)
                ).alias("vote_count"),
            ]
        )

        # Calculate average confidence
        avg_confidence = sum(pl.col(c) for c in confidence_cols) / len(confidence_cols)

        result = result.with_columns(
            [
                (pl.col("vote_count") >= self.config.ensemble_min_votes).alias(
                    "is_anomaly"
                ),
                avg_confidence.alias("confidence"),
            ]
        )

        return result


# Registry of all available detectors
DETECTORS = {
    "zscore": ZScoreDetector,
    "moving_avg": MovingAverageDetector,
    "rate_change": RateChangeDetector,
    "isolation_forest": IsolationForestDetector,
    "ensemble": EnsembleDetector,
}


def get_detector(
    method: str, config: Optional[DetectionConfig] = None
) -> AnomalyDetector:
    """
    Factory function to get a detector by name.

    Args:
        method: Name of the detection method
        config: Optional configuration

    Returns:
        Initialized detector instance

    Raises:
        ValueError: If method name is not recognized
    """
    if method not in DETECTORS:
        available = ", ".join(DETECTORS.keys())
        raise ValueError(f"Unknown method '{method}'. Available methods: {available}")

    return DETECTORS[method](config)
