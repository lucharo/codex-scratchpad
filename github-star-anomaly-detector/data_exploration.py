"""
Data Exploration Notebook for GitHub Star Data

This notebook explores GitHub star data and visualizes the time series.
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
    from data_fetcher import GitHubStarFetcher, generate_synthetic_star_data
    return GitHubStarFetcher, alt, generate_synthetic_star_data, mo, np, pl


@app.cell
def __(mo):
    mo.md(
        """
        # GitHub Star Data Exploration

        This notebook explores GitHub star data and visualizes time series patterns.

        ## Data Sources

        We can either:
        1. Fetch real data from GitHub API (requires API token for large repos)
        2. Use synthetic data for testing anomaly detection algorithms
        """
    )
    return


@app.cell
def __(mo):
    # Create a toggle to choose between real and synthetic data
    data_source = mo.ui.radio(
        options=["synthetic", "real"],
        value="synthetic",
        label="Data Source"
    )
    data_source
    return (data_source,)


@app.cell
def __(generate_synthetic_star_data, data_source):
    # Generate or fetch data based on selection
    if data_source.value == "synthetic":
        # Generate synthetic data with anomalies
        star_df = generate_synthetic_star_data(
            n_days=365,
            base_rate=10.0,
            anomaly_days=[50, 150, 250, 320],
            anomaly_multiplier=5.0
        )
        data_info = "Synthetic data generated with 4 anomalies injected"
    else:
        # For real data, we'll create a placeholder
        # Users can modify this to fetch from their desired repo
        star_df = None
        data_info = "Real data fetching not implemented yet. Please use synthetic data or modify the notebook."
    return data_info, star_df


@app.cell
def __(data_info, mo):
    mo.md(f"**Data Info:** {data_info}")
    return


@app.cell
def __(mo, star_df):
    if star_df is not None:
        mo.md(f"""
        ## Data Summary

        - **Total Days:** {len(star_df)}
        - **Total Stars:** {star_df['cumulative_stars'][-1]:.0f}
        - **Average Stars per Day:** {star_df['new_stars'].mean():.2f}
        - **Max Stars in a Day:** {star_df['new_stars'].max():.0f}
        """)
    return


@app.cell
def __(mo, star_df):
    if star_df is not None:
        mo.ui.table(star_df.head(10))
    return


@app.cell
def __(alt, mo, star_df):
    if star_df is not None:
        # Visualize cumulative stars
        chart1 = alt.Chart(star_df.to_pandas()).mark_line().encode(
            x=alt.X('date:T', title='Date'),
            y=alt.Y('cumulative_stars:Q', title='Cumulative Stars'),
            tooltip=['date:T', 'cumulative_stars:Q', 'new_stars:Q']
        ).properties(
            width=700,
            height=300,
            title='Cumulative Stars Over Time'
        ).interactive()

        mo.ui.altair_chart(chart1)
    return (chart1,)


@app.cell
def __(alt, mo, star_df):
    if star_df is not None:
        # Visualize daily new stars
        chart2 = alt.Chart(star_df.to_pandas()).mark_bar().encode(
            x=alt.X('date:T', title='Date'),
            y=alt.Y('new_stars:Q', title='New Stars per Day'),
            tooltip=['date:T', 'new_stars:Q']
        ).properties(
            width=700,
            height=300,
            title='Daily New Stars'
        ).interactive()

        mo.ui.altair_chart(chart2)
    return (chart2,)


@app.cell
def __(mo, np, star_df):
    if star_df is not None:
        # Calculate basic statistics
        new_stars = star_df['new_stars'].to_numpy()
        mean_stars = np.mean(new_stars)
        std_stars = np.std(new_stars)

        # Simple threshold-based anomaly detection (for exploration)
        threshold = mean_stars + 2 * std_stars
        potential_anomalies = star_df.filter(pl.col('new_stars') > threshold)

        mo.md(f"""
        ## Simple Statistical Analysis

        - **Mean:** {mean_stars:.2f}
        - **Std Dev:** {std_stars:.2f}
        - **Threshold (mean + 2*std):** {threshold:.2f}
        - **Days exceeding threshold:** {len(potential_anomalies)}
        """)
    return mean_stars, new_stars, potential_anomalies, std_stars, threshold


@app.cell
def __(mo, potential_anomalies):
    if potential_anomalies is not None and len(potential_anomalies) > 0:
        mo.md("### Potential Anomalies (simple threshold method)")
        mo.ui.table(potential_anomalies)
    return


if __name__ == "__main__":
    app.run()
