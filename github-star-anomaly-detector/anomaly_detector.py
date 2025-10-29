"""
GitHub Star Anomaly Detector

This notebook implements and visualizes anomaly detection algorithms
for GitHub star time series data.
"""

import marimo

__generated_with = "0.17.2"
app = marimo.App(width="medium")


@app.cell
def __():
    import marimo as mo
    import polars as pl
    import altair as alt
    import numpy as np
    from data_fetcher import generate_synthetic_star_data
    from sklearn.ensemble import IsolationForest
    from scipy import stats
    from scipy.signal import find_peaks
    return (
        IsolationForest,
        alt,
        find_peaks,
        generate_synthetic_star_data,
        mo,
        np,
        pl,
        stats,
    )


@app.cell
def __(mo):
    mo.md(
        """
        # GitHub Star Anomaly Detector

        This notebook implements multiple anomaly detection methods to identify
        unusual patterns in GitHub repository star growth.

        ## Anomaly Detection Methods

        1. **Statistical (Z-Score)**: Detects values that deviate significantly from the mean
        2. **Moving Average**: Identifies points that deviate from a rolling average
        3. **Isolation Forest**: ML-based anomaly detection
        4. **Rate of Change**: Detects sudden spikes or drops in the rate of stars
        """
    )
    return


@app.cell
def __(generate_synthetic_star_data, mo):
    # Configuration
    n_days_slider = mo.ui.slider(
        start=100,
        stop=500,
        value=365,
        label="Number of days"
    )

    base_rate_slider = mo.ui.slider(
        start=5,
        stop=50,
        value=10,
        label="Base stars per day"
    )

    anomaly_mult_slider = mo.ui.slider(
        start=2,
        stop=10,
        value=5,
        label="Anomaly multiplier"
    )

    mo.md(f"""
    ## Data Configuration

    {n_days_slider}

    {base_rate_slider}

    {anomaly_mult_slider}
    """)
    return anomaly_mult_slider, base_rate_slider, n_days_slider


@app.cell
def __(
    anomaly_mult_slider,
    base_rate_slider,
    generate_synthetic_star_data,
    n_days_slider,
):
    # Generate synthetic data
    star_df = generate_synthetic_star_data(
        n_days=n_days_slider.value,
        base_rate=float(base_rate_slider.value),
        anomaly_days=[50, 150, 250, 320],
        anomaly_multiplier=float(anomaly_mult_slider.value)
    )
    return (star_df,)


@app.cell
def __(mo, np, pl, star_df, stats):
    # Method 1: Z-Score based anomaly detection
    def detect_anomalies_zscore(df: pl.DataFrame, threshold: float = 2.5) -> pl.DataFrame:
        """Detect anomalies using Z-score."""
        new_stars = df['new_stars'].to_numpy()
        z_scores = np.abs(stats.zscore(new_stars))

        return df.with_columns([
            pl.Series("z_score", z_scores),
            pl.Series("is_anomaly_zscore", z_scores > threshold)
        ])

    # Method 2: Moving Average based anomaly detection
    def detect_anomalies_moving_avg(
        df: pl.DataFrame,
        window: int = 7,
        threshold: float = 2.0
    ) -> pl.DataFrame:
        """Detect anomalies using moving average and standard deviation."""
        result = df.with_columns([
            pl.col('new_stars').rolling_mean(window_size=window).alias('moving_avg'),
            pl.col('new_stars').rolling_std(window_size=window).alias('moving_std')
        ])

        # Calculate deviation from moving average
        result = result.with_columns([
            ((pl.col('new_stars') - pl.col('moving_avg')) / pl.col('moving_std')).alias('ma_deviation')
        ])

        # Mark anomalies
        result = result.with_columns([
            (pl.col('ma_deviation').abs() > threshold).alias('is_anomaly_ma')
        ])

        return result

    # Method 3: Rate of Change based anomaly detection
    def detect_anomalies_rate_change(df: pl.DataFrame, threshold: float = 3.0) -> pl.DataFrame:
        """Detect anomalies based on rate of change."""
        result = df.with_columns([
            pl.col('new_stars').diff().alias('rate_change')
        ])

        # Calculate z-score of rate changes
        rate_changes = result['rate_change'].fill_null(0).to_numpy()
        rate_z_scores = np.abs(stats.zscore(rate_changes))

        result = result.with_columns([
            pl.Series("rate_z_score", rate_z_scores),
            pl.Series("is_anomaly_rate", rate_z_scores > threshold)
        ])

        return result

    mo.md("### Anomaly Detection Methods Defined")
    return (
        detect_anomalies_moving_avg,
        detect_anomalies_rate_change,
        detect_anomalies_zscore,
    )


@app.cell
def __(mo):
    # Select detection method
    method_selector = mo.ui.dropdown(
        options={
            "zscore": "Z-Score",
            "moving_avg": "Moving Average",
            "rate_change": "Rate of Change",
            "isolation_forest": "Isolation Forest",
            "combined": "Combined (All Methods)"
        },
        value="combined",
        label="Detection Method"
    )

    mo.md(f"""
    ## Select Anomaly Detection Method

    {method_selector}
    """)
    return (method_selector,)


@app.cell
def __(
    IsolationForest,
    detect_anomalies_moving_avg,
    detect_anomalies_rate_change,
    detect_anomalies_zscore,
    method_selector,
    np,
    pl,
    star_df,
):
    # Apply selected method
    if method_selector.value == "zscore":
        result_df = detect_anomalies_zscore(star_df)
        anomaly_col = 'is_anomaly_zscore'
    elif method_selector.value == "moving_avg":
        result_df = detect_anomalies_moving_avg(star_df)
        anomaly_col = 'is_anomaly_ma'
    elif method_selector.value == "rate_change":
        result_df = detect_anomalies_rate_change(star_df)
        anomaly_col = 'is_anomaly_rate'
    elif method_selector.value == "isolation_forest":
        # Isolation Forest
        features = star_df.select(['new_stars']).to_numpy()
        iso_forest = IsolationForest(contamination=0.05, random_state=42)
        predictions = iso_forest.fit_predict(features)
        result_df = star_df.with_columns([
            pl.Series("is_anomaly_iso", predictions == -1)
        ])
        anomaly_col = 'is_anomaly_iso'
    else:  # combined
        # Apply all methods
        result_df = detect_anomalies_zscore(star_df)
        result_df = detect_anomalies_moving_avg(result_df)
        result_df = detect_anomalies_rate_change(result_df)

        # Isolation Forest
        features = result_df.select(['new_stars']).to_numpy()
        iso_forest = IsolationForest(contamination=0.05, random_state=42)
        predictions = iso_forest.fit_predict(features)
        result_df = result_df.with_columns([
            pl.Series("is_anomaly_iso", predictions == -1)
        ])

        # Combined: anomaly if detected by at least 2 methods
        result_df = result_df.with_columns([
            (
                pl.col('is_anomaly_zscore').cast(pl.Int32) +
                pl.col('is_anomaly_ma').cast(pl.Int32) +
                pl.col('is_anomaly_rate').cast(pl.Int32) +
                pl.col('is_anomaly_iso').cast(pl.Int32)
            ).alias('anomaly_count'),
        ])

        result_df = result_df.with_columns([
            (pl.col('anomaly_count') >= 2).alias('is_anomaly_combined')
        ])

        anomaly_col = 'is_anomaly_combined'

    # Get anomaly subset
    anomalies = result_df.filter(pl.col(anomaly_col) == True)
    return anomalies, anomaly_col, features, iso_forest, predictions, result_df


@app.cell
def __(anomalies, mo, result_df):
    mo.md(f"""
    ## Results

    - **Total Days:** {len(result_df)}
    - **Anomalies Detected:** {len(anomalies)}
    - **Anomaly Rate:** {len(anomalies) / len(result_df) * 100:.2f}%
    """)
    return


@app.cell
def __(alt, anomalies, mo, result_df):
    # Visualization: Time series with anomalies highlighted
    base_chart = alt.Chart(result_df.to_pandas()).mark_line().encode(
        x=alt.X('date:T', title='Date'),
        y=alt.Y('new_stars:Q', title='New Stars per Day'),
        tooltip=['date:T', 'new_stars:Q', 'cumulative_stars:Q']
    ).properties(
        width=800,
        height=400,
        title='GitHub Stars with Anomalies Highlighted'
    )

    # Anomaly points
    anomaly_chart = alt.Chart(anomalies.to_pandas()).mark_circle(
        size=100,
        color='red',
        opacity=0.7
    ).encode(
        x='date:T',
        y='new_stars:Q',
        tooltip=['date:T', 'new_stars:Q']
    )

    combined_chart = (base_chart + anomaly_chart).interactive()
    mo.ui.altair_chart(combined_chart)
    return anomaly_chart, base_chart, combined_chart


@app.cell
def __(alt, mo, result_df):
    # Cumulative stars chart
    cumulative_chart = alt.Chart(result_df.to_pandas()).mark_area(
        line={'color': 'darkblue'},
        color=alt.Gradient(
            gradient='linear',
            stops=[
                alt.GradientStop(color='white', offset=0),
                alt.GradientStop(color='darkblue', offset=1)
            ],
            x1=0, x2=0, y1=1, y2=0
        )
    ).encode(
        x=alt.X('date:T', title='Date'),
        y=alt.Y('cumulative_stars:Q', title='Cumulative Stars'),
        tooltip=['date:T', 'cumulative_stars:Q']
    ).properties(
        width=800,
        height=300,
        title='Cumulative Star Growth'
    ).interactive()

    mo.ui.altair_chart(cumulative_chart)
    return (cumulative_chart,)


@app.cell
def __(anomalies, mo):
    mo.md("### Detected Anomalies")
    if len(anomalies) > 0:
        mo.ui.table(anomalies.select(['date', 'new_stars', 'cumulative_stars']))
    else:
        mo.md("*No anomalies detected with current settings.*")
    return


@app.cell
def __(method_selector, mo, result_df):
    if method_selector.value == "combined":
        mo.md("### Method Comparison")
        comparison = result_df.select([
            'date',
            'new_stars',
            'is_anomaly_zscore',
            'is_anomaly_ma',
            'is_anomaly_rate',
            'is_anomaly_iso',
            'anomaly_count'
        ])
        mo.ui.table(comparison.head(20))
    return (comparison,)


@app.cell
def __(mo):
    mo.md(
        """
        ## Interpretation

        - **Red circles** indicate detected anomalies
        - **Z-Score**: Good for detecting outliers in normally distributed data
        - **Moving Average**: Captures deviations from recent trends
        - **Rate of Change**: Detects sudden spikes or drops
        - **Isolation Forest**: ML-based, good for complex patterns
        - **Combined**: More robust, reduces false positives
        """
    )
    return


if __name__ == "__main__":
    app.run()
