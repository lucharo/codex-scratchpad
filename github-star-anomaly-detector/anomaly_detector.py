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
    from data_fetcher import generate_synthetic_star_data
    from anomaly_methods import get_detector, DetectionConfig, DETECTORS

    return (
        DETECTORS,
        DetectionConfig,
        alt,
        generate_synthetic_star_data,
        get_detector,
        mo,
        pl,
    )


@app.cell
def __(mo):
    mo.md(
        """
        # GitHub Star Anomaly Detector

        This notebook implements multiple anomaly detection methods to identify
        unusual patterns in GitHub repository star growth.

        ## Anomaly Detection Methods

        1. **Z-Score**: Detects values that deviate significantly from the mean
        2. **Moving Average**: Identifies points that deviate from a rolling average
        3. **Rate of Change**: Detects sudden spikes or drops in the rate of stars
        4. **Isolation Forest**: ML-based anomaly detection
        5. **Ensemble**: Combines all methods with voting for robust detection
        """
    )
    return


@app.cell
def __(mo):
    # Data configuration
    n_days_slider = mo.ui.slider(start=100, stop=500, value=365, label="Number of days")

    base_rate_slider = mo.ui.slider(
        start=5, stop=50, value=10, label="Base stars per day"
    )

    anomaly_mult_slider = mo.ui.slider(
        start=2, stop=10, value=5, label="Anomaly multiplier"
    )

    growth_pattern = mo.ui.dropdown(
        options=["linear", "exponential", "logarithmic", "viral"],
        value="linear",
        label="Growth pattern",
    )

    mo.md(f"""
    ## Data Configuration

    {n_days_slider}

    {base_rate_slider}

    {anomaly_mult_slider}

    {growth_pattern}
    """)
    return anomaly_mult_slider, base_rate_slider, growth_pattern, n_days_slider


@app.cell
def __(
    anomaly_mult_slider,
    base_rate_slider,
    generate_synthetic_star_data,
    growth_pattern,
    n_days_slider,
):
    # Generate synthetic data
    star_df = generate_synthetic_star_data(
        n_days=n_days_slider.value,
        base_rate=float(base_rate_slider.value),
        anomaly_days=[50, 150, 250, 320],
        anomaly_multiplier=float(anomaly_mult_slider.value),
        growth_pattern=growth_pattern.value,
        add_seasonality=True,
    )
    return (star_df,)


@app.cell
def __(DETECTORS, mo):
    # Select detection method
    method_selector = mo.ui.dropdown(
        options={
            k: v().__class__.__name__.replace("Detector", "")
            for k, v in DETECTORS.items()
        },
        value="ensemble",
        label="Detection Method",
    )

    mo.md(f"""
    ## Select Anomaly Detection Method

    {method_selector}
    """)
    return (method_selector,)


@app.cell
def __(get_detector, method_selector, pl, star_df):
    # Apply selected detection method (this is now just 3 lines!)
    detector = get_detector(method_selector.value)
    result_df = detector.detect(star_df)
    anomalies = result_df.filter(pl.col("is_anomaly"))
    return anomalies, detector, result_df


@app.cell
def __(anomalies, detector, mo, result_df):
    mo.md(f"""
    ## Results

    **Method:** {detector.name}

    **Description:** {detector.description}

    - **Total Days:** {len(result_df)}
    - **Anomalies Detected:** {len(anomalies)}
    - **Anomaly Rate:** {len(anomalies) / len(result_df) * 100:.2f}%
    """)
    return


@app.cell
def __(alt, anomalies, mo, result_df):
    # Visualization: Time series with anomalies highlighted
    base_chart = (
        alt.Chart(result_df.to_pandas())
        .mark_line()
        .encode(
            x=alt.X("date:T", title="Date"),
            y=alt.Y("new_stars:Q", title="New Stars per Day"),
            tooltip=["date:T", "new_stars:Q", "cumulative_stars:Q"],
        )
        .properties(
            width=800, height=400, title="GitHub Stars with Anomalies Highlighted"
        )
    )

    # Anomaly points
    anomaly_chart = (
        alt.Chart(anomalies.to_pandas())
        .mark_circle(size=100, color="red", opacity=0.7)
        .encode(
            x="date:T",
            y="new_stars:Q",
            tooltip=["date:T", "new_stars:Q", "confidence:Q"],
        )
    )

    combined_chart = (base_chart + anomaly_chart).interactive()
    mo.ui.altair_chart(combined_chart)
    return anomaly_chart, base_chart, combined_chart


@app.cell
def __(alt, mo, result_df):
    # Cumulative stars chart
    cumulative_chart = (
        alt.Chart(result_df.to_pandas())
        .mark_area(
            line={"color": "darkblue"},
            color=alt.Gradient(
                gradient="linear",
                stops=[
                    alt.GradientStop(color="white", offset=0),
                    alt.GradientStop(color="darkblue", offset=1),
                ],
                x1=0,
                x2=0,
                y1=1,
                y2=0,
            ),
        )
        .encode(
            x=alt.X("date:T", title="Date"),
            y=alt.Y("cumulative_stars:Q", title="Cumulative Stars"),
            tooltip=["date:T", "cumulative_stars:Q"],
        )
        .properties(width=800, height=300, title="Cumulative Star Growth")
        .interactive()
    )

    mo.ui.altair_chart(cumulative_chart)
    return (cumulative_chart,)


@app.cell
def __(anomalies, mo):
    mo.md("### Detected Anomalies")
    if len(anomalies) > 0:
        display_cols = ["date", "new_stars", "cumulative_stars", "confidence"]
        # Only show columns that exist
        available_cols = [c for c in display_cols if c in anomalies.columns]
        mo.ui.table(anomalies.select(available_cols))
    else:
        mo.md("*No anomalies detected with current settings.*")
    return available_cols, display_cols


@app.cell
def __(method_selector, mo, result_df):
    if method_selector.value == "ensemble":
        mo.md("### Method Comparison")
        comparison_cols = [
            "date",
            "new_stars",
            "is_anomaly_zscore",
            "is_anomaly_movingaverage",
            "is_anomaly_ratechange",
            "is_anomaly_isolationforest",
            "vote_count",
            "confidence",
        ]
        # Only show columns that exist
        available_comparison_cols = [
            c for c in comparison_cols if c in result_df.columns
        ]
        mo.ui.table(result_df.select(available_comparison_cols).head(20))
    return available_comparison_cols, comparison_cols


@app.cell
def __(mo):
    mo.md(
        """
        ## Interpretation

        - **Red circles** indicate detected anomalies
        - **Confidence** shows how certain the detector is (0.0 - 1.0)
        - **Z-Score**: Good for detecting outliers in normally distributed data
        - **Moving Average**: Captures deviations from recent trends
        - **Rate of Change**: Detects sudden spikes or drops
        - **Isolation Forest**: ML-based, good for complex patterns
        - **Ensemble**: More robust, reduces false positives by requiring multiple methods to agree
        """
    )
    return


if __name__ == "__main__":
    app.run()
