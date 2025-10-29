"""
Quick test script to validate anomaly detection works correctly.
"""

import numpy as np
from data_fetcher import generate_synthetic_star_data
from scipy import stats
import polars as pl


def test_zscore_detection():
    """Test Z-score based anomaly detection."""
    # Generate data with known anomalies
    df = generate_synthetic_star_data(
        n_days=100,
        base_rate=10.0,
        anomaly_days=[20, 50, 80],
        anomaly_multiplier=5.0
    )

    # Calculate z-scores
    new_stars = df['new_stars'].to_numpy()
    z_scores = np.abs(stats.zscore(new_stars))

    # Detect anomalies
    threshold = 2.5
    anomalies = df.with_columns([
        pl.Series("z_score", z_scores),
        pl.Series("is_anomaly", z_scores > threshold)
    ])

    detected = anomalies.filter(pl.col('is_anomaly') == True)

    print(f"Test: Z-Score Anomaly Detection")
    print(f"  Days generated: {len(df)}")
    print(f"  Anomalies injected: 3 (days 20, 50, 80)")
    print(f"  Anomalies detected: {len(detected)}")
    print(f"  Detected days: {detected['date'].to_list()}")

    # Check if we detected the injected anomalies
    detected_day_indices = [
        (date - df['date'][0]).days for date in detected['date'].to_list()
    ]

    # We should detect at least 2 of the major anomalies
    matches = sum(1 for d in detected_day_indices if d in [20, 50, 80])
    success = matches >= 2
    print(f"  Detected injected anomalies: {matches}/3")
    print(f"  Status: {'✓ PASS' if success else '✗ FAIL'}")

    return success


def test_moving_average_detection():
    """Test moving average based anomaly detection."""
    df = generate_synthetic_star_data(
        n_days=100,
        base_rate=10.0,
        anomaly_days=[30, 60],
        anomaly_multiplier=6.0
    )

    # Moving average detection
    window = 7
    threshold = 2.0

    result = df.with_columns([
        pl.col('new_stars').rolling_mean(window_size=window).alias('moving_avg'),
        pl.col('new_stars').rolling_std(window_size=window).alias('moving_std')
    ])

    result = result.with_columns([
        ((pl.col('new_stars') - pl.col('moving_avg')) / pl.col('moving_std')).alias('ma_deviation')
    ])

    result = result.with_columns([
        (pl.col('ma_deviation').abs() > threshold).alias('is_anomaly')
    ])

    detected = result.filter(pl.col('is_anomaly') == True)

    print(f"\nTest: Moving Average Anomaly Detection")
    print(f"  Days generated: {len(df)}")
    print(f"  Anomalies injected: 2 (days 30, 60)")
    print(f"  Anomalies detected: {len(detected)}")

    success = len(detected) >= 2
    print(f"  Status: {'✓ PASS' if success else '✗ FAIL'}")

    return success


def test_isolation_forest():
    """Test Isolation Forest anomaly detection."""
    from sklearn.ensemble import IsolationForest

    df = generate_synthetic_star_data(
        n_days=200,
        base_rate=15.0,
        anomaly_days=[40, 100, 160],
        anomaly_multiplier=4.0
    )

    # Isolation Forest
    features = df.select(['new_stars']).to_numpy()
    iso_forest = IsolationForest(contamination=0.05, random_state=42)
    predictions = iso_forest.fit_predict(features)

    anomalies = df.with_columns([
        pl.Series("is_anomaly", predictions == -1)
    ])

    detected = anomalies.filter(pl.col('is_anomaly') == True)

    print(f"\nTest: Isolation Forest Anomaly Detection")
    print(f"  Days generated: {len(df)}")
    print(f"  Anomalies injected: 3 (days 40, 100, 160)")
    print(f"  Anomalies detected: {len(detected)}")

    success = len(detected) >= 3
    print(f"  Status: {'✓ PASS' if success else '✗ FAIL'}")

    return success


if __name__ == "__main__":
    print("=" * 60)
    print("GitHub Star Anomaly Detector - Test Suite")
    print("=" * 60)

    results = []
    results.append(test_zscore_detection())
    results.append(test_moving_average_detection())
    results.append(test_isolation_forest())

    print("\n" + "=" * 60)
    print(f"Overall: {sum(results)}/{len(results)} tests passed")
    print("=" * 60)

    if all(results):
        print("✓ All tests passed!")
        exit(0)
    else:
        print("✗ Some tests failed")
        exit(1)
