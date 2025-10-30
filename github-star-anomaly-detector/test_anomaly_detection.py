"""
Test suite for GitHub Star Anomaly Detection

Tests all detection methods with synthetic data to validate functionality.
"""

from data_fetcher import generate_synthetic_star_data
from anomaly_methods import get_detector, DetectionConfig


def test_zscore_detection():
    """Test Z-score based anomaly detection."""
    # Generate data with known anomalies
    df = generate_synthetic_star_data(
        n_days=100,
        base_rate=10.0,
        anomaly_days=[20, 50, 80],
        anomaly_multiplier=5.0,
        growth_pattern="linear",
        add_seasonality=False,
        seed=42
    )

    # Detect anomalies
    detector = get_detector('zscore')
    result = detector.detect(df)
    detected = result.filter(result['is_anomaly'] == True)

    # Check if we detected the injected anomalies
    detected_day_indices = [
        (date - df['date'][0]).days for date in detected['date'].to_list()
    ]

    matches = sum(1 for d in detected_day_indices if d in [20, 50, 80])

    print(f"Test: Z-Score Anomaly Detection")
    print(f"  Days generated: {len(df)}")
    print(f"  Anomalies injected: 3 (days 20, 50, 80)")
    print(f"  Anomalies detected: {len(detected)}")
    print(f"  Detected injected anomalies: {matches}/3")

    # We should detect at least 2 of the major anomalies
    success = matches >= 2
    print(f"  Status: {'✓ PASS' if success else '✗ FAIL'}")

    return success


def test_moving_average_detection():
    """Test moving average based anomaly detection."""
    df = generate_synthetic_star_data(
        n_days=100,
        base_rate=10.0,
        anomaly_days=[30, 60],
        anomaly_multiplier=6.0,
        growth_pattern="linear",
        add_seasonality=False,
        seed=42
    )

    detector = get_detector('moving_avg')
    result = detector.detect(df)
    detected = result.filter(result['is_anomaly'] == True)

    print(f"\nTest: Moving Average Anomaly Detection")
    print(f"  Days generated: {len(df)}")
    print(f"  Anomalies injected: 2 (days 30, 60)")
    print(f"  Anomalies detected: {len(detected)}")

    success = len(detected) >= 2
    print(f"  Status: {'✓ PASS' if success else '✗ FAIL'}")

    return success


def test_isolation_forest():
    """Test Isolation Forest anomaly detection."""
    df = generate_synthetic_star_data(
        n_days=200,
        base_rate=15.0,
        anomaly_days=[40, 100, 160],
        anomaly_multiplier=4.0,
        growth_pattern="linear",
        add_seasonality=False,
        seed=42
    )

    detector = get_detector('isolation_forest')
    result = detector.detect(df)
    detected = result.filter(result['is_anomaly'] == True)

    print(f"\nTest: Isolation Forest Anomaly Detection")
    print(f"  Days generated: {len(df)}")
    print(f"  Anomalies injected: 3 (days 40, 100, 160)")
    print(f"  Anomalies detected: {len(detected)}")

    success = len(detected) >= 3
    print(f"  Status: {'✓ PASS' if success else '✗ FAIL'}")

    return success


def test_ensemble_detection():
    """Test Ensemble detector that combines all methods."""
    df = generate_synthetic_star_data(
        n_days=150,
        base_rate=12.0,
        anomaly_days=[30, 75, 120],
        anomaly_multiplier=5.0,
        growth_pattern="linear",
        add_seasonality=False,
        seed=42
    )

    # Test with custom configuration
    config = DetectionConfig(ensemble_min_votes=2)
    detector = get_detector('ensemble', config)
    result = detector.detect(df)
    detected = result.filter(result['is_anomaly'] == True)

    print(f"\nTest: Ensemble Anomaly Detection")
    print(f"  Days generated: {len(df)}")
    print(f"  Anomalies injected: 3 (days 30, 75, 120)")
    print(f"  Anomalies detected: {len(detected)}")
    print(f"  Ensemble requires >= {config.ensemble_min_votes} methods to agree")

    # Ensemble should detect at least 2 anomalies
    success = len(detected) >= 2
    print(f"  Status: {'✓ PASS' if success else '✗ FAIL'}")

    return success


def test_realistic_growth_patterns():
    """Test detection with different growth patterns."""
    patterns = ["linear", "exponential", "logarithmic", "viral"]
    successes = []

    print(f"\nTest: Detection with Realistic Growth Patterns")

    for pattern in patterns:
        df = generate_synthetic_star_data(
            n_days=200,
            base_rate=10.0,
            anomaly_days=[50, 150],
            anomaly_multiplier=5.0,
            growth_pattern=pattern,
            add_seasonality=True,
            seed=42
        )

        detector = get_detector('ensemble')
        result = detector.detect(df)
        detected = result.filter(result['is_anomaly'] == True)

        # Should detect at least one anomaly
        pattern_success = len(detected) >= 1
        successes.append(pattern_success)

        print(f"  {pattern:12s}: {len(detected):2d} anomalies detected - {'✓' if pattern_success else '✗'}")

    overall_success = all(successes)
    print(f"  Status: {'✓ PASS' if overall_success else '✗ FAIL'}")

    return overall_success


if __name__ == "__main__":
    print("=" * 60)
    print("GitHub Star Anomaly Detector - Test Suite")
    print("=" * 60)

    results = []
    results.append(test_zscore_detection())
    results.append(test_moving_average_detection())
    results.append(test_isolation_forest())
    results.append(test_ensemble_detection())
    results.append(test_realistic_growth_patterns())

    print("\n" + "=" * 60)
    print(f"Overall: {sum(results)}/{len(results)} tests passed")
    print("=" * 60)

    if all(results):
        print("✓ All tests passed!")
        exit(0)
    else:
        print("✗ Some tests failed")
        exit(1)
