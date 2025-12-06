# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "marimo",
#     "polars==1.18.0",
#     "altair==5.4.1",
#     "httpx==0.28.1",
# ]
# ///

import marimo

__generated_with = "0.10.0"
app = marimo.App(width="medium")


@app.cell(hide_code=True)
def _():
    import marimo as mo
    import polars as pl
    import altair as alt
    return mo, pl, alt


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    # UK Train Reliability Stats

    Check historical performance for any UK train route.

    **Definitions:** On time (≤5 min) · Late (5-29 min) · Very late (30+ min) · Delay Repay (15+ min)
    """)
    return


# =============================================================================
# STATIONS
# =============================================================================

@app.cell
def _():
    STATIONS = {
        "CDF": "Cardiff Central", "PAD": "London Paddington", "EUS": "London Euston",
        "KGX": "London Kings Cross", "VIC": "London Victoria", "WAT": "London Waterloo",
        "BHM": "Birmingham New St", "MAN": "Manchester Picc", "LDS": "Leeds",
        "EDB": "Edinburgh", "GLC": "Glasgow Central", "BRI": "Bristol TM",
        "RDG": "Reading", "OXF": "Oxford", "SWI": "Swindon", "NPT": "Newport",
    }
    return (STATIONS,)


# =============================================================================
# ROUTE SELECTION
# =============================================================================

@app.cell
def _():
    # Time slots for departure time selection (30-min buckets)
    TIME_SLOTS = [""] + [f"{h:02d}:{m:02d}" for h in range(5, 24) for m in [0, 30]]
    return (TIME_SLOTS,)


@app.cell
def _(mo, STATIONS, TIME_SLOTS):
    station_options = {f"{code} - {name}": code for code, name in sorted(STATIONS.items(), key=lambda x: x[1])}
    time_options = {"(any time)": ""} | {t: t for t in TIME_SLOTS if t}

    origin_dropdown = mo.ui.dropdown(options=station_options, value="CDF - Cardiff Central", label="Origin")
    dest_dropdown = mo.ui.dropdown(options=station_options, value="PAD - London Paddington", label="Destination")
    time_dropdown = mo.ui.dropdown(options=time_options, value="(any time)", label="Departure Time")

    mo.hstack([origin_dropdown, dest_dropdown, time_dropdown], justify="start", gap=2)
    return origin_dropdown, dest_dropdown, time_dropdown, station_options, time_options


@app.cell
def _(origin_dropdown, dest_dropdown, time_dropdown):
    origin_crs = origin_dropdown.value
    dest_crs = dest_dropdown.value
    selected_time = time_dropdown.value if time_dropdown.value else None
    return origin_crs, dest_crs, selected_time


@app.cell(hide_code=True)
def _(mo, origin_crs, dest_crs, STATIONS):
    mo.md(f"### Route: **{STATIONS.get(origin_crs, origin_crs)}** → **{STATIONS.get(dest_crs, dest_crs)}**")
    return


# =============================================================================
# DATA LOADING
# =============================================================================

@app.cell
def _(mo):
    mo.md("*Using demo data. For live data, integrate HSP API credentials.*")
    return


@app.cell
def _(mo):
    load_button = mo.ui.run_button(label="Load Data")
    load_button
    return (load_button,)


@app.cell
def _(mo, load_button, origin_crs, dest_crs):
    from demo_data import generate_services

    mo.stop(not load_button.value)

    with mo.status.spinner("Generating data..."):
        services_df = generate_services(origin_crs, dest_crs, days=365)

    mo.callout(mo.md(f"Loaded **{len(services_df):,}** services"), kind="info")
    return (services_df,)


# =============================================================================
# PERFORMANCE TABLES
# =============================================================================

@app.cell(hide_code=True)
def _(mo):
    mo.md("---\n## Performance Statistics")
    return


@app.cell
def _(mo, services_df):
    from stats import get_route_stats

    stats_7d = get_route_stats(services_df, 7)
    stats_30d = get_route_stats(services_df, 30)
    stats_365d = get_route_stats(services_df, 365)

    route_table = [
        stats_7d.to_dict("Last 7 days"),
        stats_30d.to_dict("Last 30 days"),
        stats_365d.to_dict("Last 365 days"),
    ]
    mo.ui.table(route_table, selection=None)
    return stats_7d, stats_30d, stats_365d, route_table


@app.cell(hide_code=True)
def _(mo):
    mo.md("### By Day of Week")
    return


@app.cell
def _(mo, services_df):
    from stats import get_weekday_stats

    DAYS = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    weekday_table = [
        {"Day": DAYS[wd], **get_weekday_stats(services_df, wd).to_dict("")}
        for wd in range(7)
        if get_weekday_stats(services_df, wd).total_services > 0
    ]
    # Remove redundant "Period" key
    for row in weekday_table:
        row.pop("Period", None)

    mo.ui.table(weekday_table, selection=None)
    return DAYS, weekday_table


# =============================================================================
# TIME SLOT ANALYSIS
# =============================================================================

@app.cell(hide_code=True)
def _(mo, selected_time):
    slot_header = f"### Time Slot: **{selected_time}**" if selected_time else "### Time Slot\n*Select a time above*"
    mo.md(slot_header)
    return


@app.cell
def _(mo, selected_time, services_df):
    from stats import get_time_slot_stats

    if selected_time:
        # Parse time string "HH:MM" to get slot
        _hour, _minute = map(int, selected_time.split(":"))
        _slot_start = (_minute // 30) * 30
        dep_slot = f"{_hour:02d}:{_slot_start:02d}-{_hour:02d}:{_slot_start + 29:02d}"

        slot_stats = get_time_slot_stats(services_df, dep_slot)

        if slot_stats.total_services > 0:
            slot_output = mo.md(f"""
**{dep_slot}** ({slot_stats.total_services} services)

| Metric | Value |
|--------|-------|
| On Time | {slot_stats.on_time_pct:.1f}% |
| Late | {slot_stats.late_pct:.1f}% |
| Very Late | {slot_stats.very_late_pct:.1f}% |
| Cancelled | {slot_stats.cancelled_pct:.1f}% |
| **Delay Repay** | **{slot_stats.delay_repay_pct:.1f}%** |
| Avg Delay | {slot_stats.avg_delay_mins:.1f} min |
            """)
        else:
            slot_output = mo.callout(mo.md("No data for this slot"), kind="warn")
    else:
        slot_output = None
        dep_slot = None

    slot_output
    return (dep_slot,)


# =============================================================================
# DELAY DISTRIBUTION
# =============================================================================

@app.cell(hide_code=True)
def _(mo):
    mo.md("""
---
## Delay Distribution & Refund Probability

Thresholds: **15 min** (25% refund) · **30 min** (50%) · **60 min** (full)
    """)
    return


@app.cell
def _(services_df, pl, alt):
    from stats import get_delay_distribution

    delays = get_delay_distribution(services_df, 365)

    if not delays.is_empty():
        _n = len(delays)
        cdf_data = delays.with_columns([
            (pl.lit(1).cum_sum() / _n * 100).alias("cumulative_pct")
        ]).to_pandas()

        cdf_line = alt.Chart(cdf_data).mark_line(color="#2563eb", strokeWidth=2).encode(
            x=alt.X("arr_delay_mins:Q", title="Delay (min)", scale=alt.Scale(domain=[-10, 60])),
            y=alt.Y("cumulative_pct:Q", title="Cumulative %", scale=alt.Scale(domain=[0, 100]))
        )

        thresholds = [
            {"x": 5, "label": "On Time", "color": "#22c55e"},
            {"x": 15, "label": "Delay Repay 25%", "color": "#f59e0b"},
            {"x": 30, "label": "50% Refund", "color": "#ef4444"},
        ]
        rules = alt.Chart(alt.Data(values=thresholds)).mark_rule(strokeDash=[4, 4]).encode(
            x="x:Q", color=alt.Color("color:N", scale=None)
        )

        delay_cdf_chart = (cdf_line + rules).properties(width=550, height=300, title="Cumulative Delay Distribution")
    else:
        delay_cdf_chart = None

    delay_cdf_chart
    return delays, delay_cdf_chart


@app.cell
def _(delays, mo):
    from stats import calculate_delay_probabilities

    if not delays.is_empty():
        probs = calculate_delay_probabilities(delays)

        prob_output = mo.md(f"""
### Delay Probabilities

| Scenario | Probability |
|----------|-------------|
| On time (≤5 min) | **{probs['on_time']:.1f}%** |
| Under 15 min | {probs['under_15']:.1f}% |
| Under 30 min | {probs['under_30']:.1f}% |
| **15+ min (Delay Repay)** | **{100 - probs['under_15']:.1f}%** |
| **30+ min** | **{100 - probs['under_30']:.1f}%** |
        """)
    else:
        prob_output = None

    prob_output
    return (prob_output,)


# =============================================================================
# HISTOGRAM
# =============================================================================

@app.cell
def _(delays, alt):
    if not delays.is_empty():
        hist_data = delays.to_pandas()

        delay_histogram = alt.Chart(hist_data).mark_bar(color="#3b82f6", opacity=0.7).encode(
            x=alt.X("arr_delay_mins:Q", bin=alt.Bin(step=5, extent=[-10, 60]), title="Delay (min)"),
            y=alt.Y("count():Q", title="Services")
        ).properties(width=550, height=200, title="Delay Histogram")
    else:
        delay_histogram = None

    delay_histogram
    return (delay_histogram,)


# =============================================================================
# HEATMAP
# =============================================================================

@app.cell(hide_code=True)
def _(mo):
    mo.md("---\n## When to Travel: Delay Heatmap")
    return


@app.cell
def _(services_df, alt):
    from stats import get_heatmap_data

    heatmap_df = get_heatmap_data(services_df, 365)

    if not heatmap_df.is_empty():
        _df = heatmap_df.to_pandas()
        _df["day"] = _df["weekday"].map({0: "Mon", 1: "Tue", 2: "Wed", 3: "Thu", 4: "Fri", 5: "Sat", 6: "Sun"})

        delay_heatmap = alt.Chart(_df).mark_rect().encode(
            x=alt.X("dep_hour:O", title="Hour"),
            y=alt.Y("day:O", title="Day", sort=["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]),
            color=alt.Color("avg_delay:Q", scale=alt.Scale(scheme="reds", domain=[0, 20]), title="Avg Delay"),
            tooltip=["day:N", "dep_hour:O", alt.Tooltip("avg_delay:Q", format=".1f")]
        ).properties(width=450, height=180, title="Average Delay by Day & Hour")
    else:
        delay_heatmap = None

    delay_heatmap
    return (delay_heatmap,)


# =============================================================================
# FOOTER
# =============================================================================

@app.cell(hide_code=True)
def _(mo):
    mo.md("""
---
*Data: Demo (synthetic) · [Register for HSP API](https://opendata.nationalrail.co.uk)*
    """)
    return


if __name__ == "__main__":
    app.run()
