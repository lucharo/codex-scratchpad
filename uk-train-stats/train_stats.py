# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "marimo",
#     "duckdb==1.1.3",
#     "polars==1.18.0",
#     "altair==5.4.1",
#     "requests==2.32.3",
#     "httpx==0.28.1",
# ]
# ///

import marimo

__generated_with = "0.10.0"
app = marimo.App(width="medium")


@app.cell(hide_code=True)
def _():
    import marimo as mo
    return (mo,)


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    # UK Train Reliability Stats

    Check historical performance for any UK train route. Data sourced from
    National Rail's Historic Service Performance (HSP) API.

    **Definitions:**
    - **On time**: Arrived within 5 minutes of schedule
    - **Late**: Arrived 5-29 minutes after schedule
    - **Very late**: Arrived 30+ minutes late
    - **Delay Repay eligible**: 15+ minutes late (most TOCs) or 30+ minutes (some routes)
    """)
    return


@app.cell
def _():
    import duckdb
    import polars as pl
    import altair as alt
    from datetime import datetime, date, timedelta
    import json
    import os
    from pathlib import Path
    return duckdb, pl, alt, datetime, date, timedelta, json, os, Path


@app.cell
def _(Path):
    # Database setup - stored alongside notebook
    DB_PATH = Path(__file__).parent / "train_performance.duckdb" if "__file__" in dir() else Path("train_performance.duckdb")
    return (DB_PATH,)


@app.cell
def _(duckdb, DB_PATH):
    # Initialize database with schema
    def init_db():
        _conn = duckdb.connect(str(DB_PATH))
        _conn.execute("""
            CREATE TABLE IF NOT EXISTS services (
                -- Unique identifier
                rid VARCHAR PRIMARY KEY,

                -- Service identifiers
                service_uid VARCHAR,
                run_date DATE,

                -- Operator
                toc_code VARCHAR,
                toc_name VARCHAR,

                -- Route info
                origin_crs VARCHAR,
                origin_name VARCHAR,
                dest_crs VARCHAR,
                dest_name VARCHAR,

                -- Scheduled times
                scheduled_dep TIMESTAMP,
                scheduled_arr TIMESTAMP,

                -- Actual times (NULL if cancelled)
                actual_dep TIMESTAMP,
                actual_arr TIMESTAMP,

                -- Delays in minutes (negative = early)
                dep_delay_mins INTEGER,
                arr_delay_mins INTEGER,

                -- Status flags
                cancelled BOOLEAN DEFAULT FALSE,
                part_cancelled BOOLEAN DEFAULT FALSE,

                -- Derived fields for fast querying
                route_key VARCHAR GENERATED ALWAYS AS (origin_crs || '_' || dest_crs) STORED,
                weekday INTEGER,  -- 0=Monday, 6=Sunday
                dep_hour INTEGER,
                dep_slot VARCHAR,  -- e.g. '18:00-18:29'

                -- Categorization
                status VARCHAR,  -- 'on_time', 'late', 'very_late', 'cancelled'
                delay_repay_eligible BOOLEAN,

                -- Metadata
                fetched_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );

            -- Indexes for common query patterns
            CREATE INDEX IF NOT EXISTS idx_route_date ON services(origin_crs, dest_crs, run_date);
            CREATE INDEX IF NOT EXISTS idx_route_weekday ON services(route_key, weekday, run_date);
            CREATE INDEX IF NOT EXISTS idx_route_slot ON services(route_key, dep_slot, run_date);
            CREATE INDEX IF NOT EXISTS idx_scheduled_dep ON services(origin_crs, dest_crs, scheduled_dep);
        """)
        _conn.close()
        return True

    init_db()
    return (init_db,)


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## Select Route and Time
    """)
    return


@app.cell
def _():
    # Common UK stations for quick selection
    STATIONS = {
        "CDF": "Cardiff Central",
        "PAD": "London Paddington",
        "EUS": "London Euston",
        "KGX": "London Kings Cross",
        "VIC": "London Victoria",
        "WAT": "London Waterloo",
        "BHM": "Birmingham New Street",
        "MAN": "Manchester Piccadilly",
        "LDS": "Leeds",
        "EDB": "Edinburgh Waverley",
        "GLC": "Glasgow Central",
        "BRI": "Bristol Temple Meads",
        "NCL": "Newcastle",
        "LIV": "Liverpool Lime Street",
        "SHF": "Sheffield",
        "NOT": "Nottingham",
        "RDG": "Reading",
        "OXF": "Oxford",
        "CBG": "Cambridge",
        "NRW": "Norwich",
        "SWI": "Swindon",
        "NPT": "Newport (South Wales)",
        "BTN": "Brighton",
        "SOU": "Southampton Central",
        "PLY": "Plymouth",
        "EXD": "Exeter St Davids",
    }
    return (STATIONS,)


@app.cell
def _(mo, STATIONS):
    # Create station selection dropdowns
    station_options = {f"{code} - {name}": code for code, name in sorted(STATIONS.items(), key=lambda x: x[1])}

    origin_dropdown = mo.ui.dropdown(
        options=station_options,
        value="CDF - Cardiff Central",
        label="Origin Station"
    )

    dest_dropdown = mo.ui.dropdown(
        options=station_options,
        value="PAD - London Paddington",
        label="Destination Station"
    )

    time_input = mo.ui.time(
        value=None,
        label="Departure Time (optional)"
    )

    mo.hstack([origin_dropdown, dest_dropdown, time_input], justify="start", gap=2)
    return origin_dropdown, dest_dropdown, time_input, station_options


@app.cell
def _(origin_dropdown, dest_dropdown, time_input):
    # Extract selected values
    origin_crs = origin_dropdown.value
    dest_crs = dest_dropdown.value
    selected_time = time_input.value
    return origin_crs, dest_crs, selected_time


@app.cell(hide_code=True)
def _(mo, origin_crs, dest_crs, STATIONS):
    route_display = f"**{STATIONS.get(origin_crs, origin_crs)}** to **{STATIONS.get(dest_crs, dest_crs)}**"
    mo.md(f"### Route: {route_display}")
    return (route_display,)


# ============================================================================
# DATA FETCHING - HSP API
# ============================================================================

@app.cell
def _():
    import httpx
    import base64
    return (httpx, base64)


@app.cell
def _(mo):
    mo.md("""
    ## API Credentials

    Enter your National Rail Data Portal credentials to fetch live data.
    Register at [opendata.nationalrail.co.uk](https://opendata.nationalrail.co.uk)
    and enable HSP access.

    *Leave blank to use demo data.*
    """)
    return


@app.cell
def _(mo, os):
    # API credentials input (with env var fallback)
    api_email = mo.ui.text(
        value=os.environ.get("NR_EMAIL", ""),
        label="Email",
        kind="password"
    )
    api_password = mo.ui.text(
        value=os.environ.get("NR_PASSWORD", ""),
        label="Password",
        kind="password"
    )

    mo.hstack([api_email, api_password], gap=2)
    return api_email, api_password


@app.cell
def _(httpx, base64, date, timedelta, datetime):
    HSP_BASE_URL = "https://hsp-prod.rockshore.net/api/v1"

    def fetch_service_metrics(email: str, password: str, origin: str, dest: str,
                              from_date: date, to_date: date,
                              from_time: str = "0000", to_time: str = "2359",
                              days: str = "WEEKDAY") -> dict:
        """Fetch service metrics from HSP API."""
        auth = base64.b64encode(f"{email}:{password}".encode()).decode()

        payload = {
            "from_loc": origin,
            "to_loc": dest,
            "from_time": from_time,
            "to_time": to_time,
            "from_date": from_date.strftime("%Y-%m-%d"),
            "to_date": to_date.strftime("%Y-%m-%d"),
            "days": days,
            "tolerance": ["0", "5", "10", "15", "30"]  # Multiple tolerance levels
        }

        response = httpx.post(
            f"{HSP_BASE_URL}/serviceMetrics",
            json=payload,
            headers={
                "Authorization": f"Basic {auth}",
                "Content-Type": "application/json"
            },
            timeout=30.0
        )
        response.raise_for_status()
        return response.json()

    def fetch_service_details(email: str, password: str, rids: list[str]) -> list[dict]:
        """Fetch detailed service info for specific RIDs."""
        auth = base64.b64encode(f"{email}:{password}".encode()).decode()

        results = []
        for rid in rids[:100]:  # Limit to avoid rate limiting
            payload = {"rid": rid}
            try:
                response = httpx.post(
                    f"{HSP_BASE_URL}/serviceDetails",
                    json=payload,
                    headers={
                        "Authorization": f"Basic {auth}",
                        "Content-Type": "application/json"
                    },
                    timeout=30.0
                )
                response.raise_for_status()
                results.append(response.json())
            except Exception:
                continue
        return results

    return fetch_service_metrics, fetch_service_details, HSP_BASE_URL


@app.cell
def _(date, timedelta, datetime, pl):
    def generate_demo_data(origin: str, dest: str, days: int = 365) -> pl.DataFrame:
        """Generate realistic demo data for testing."""
        import random
        random.seed(42)

        records = []
        base_date = date.today() - timedelta(days=days)

        # Typical departure times for a major route
        dep_times = ["06:15", "07:00", "07:30", "08:00", "08:30", "09:00", "10:00",
                     "11:00", "12:00", "13:00", "14:00", "15:00", "16:00", "17:00",
                     "17:30", "18:00", "18:30", "19:00", "20:00", "21:00", "22:00"]

        journey_mins = 120 if (origin == "CDF" and dest == "PAD") else 90

        for day_offset in range(days):
            current_date = base_date + timedelta(days=day_offset)
            weekday = current_date.weekday()

            # Fewer services on weekends
            times_today = dep_times if weekday < 5 else dep_times[::2]

            for dep_time in times_today:
                rid = f"demo_{origin}_{dest}_{current_date}_{dep_time.replace(':', '')}"

                hour, minute = map(int, dep_time.split(":"))
                scheduled_dep = datetime.combine(current_date, datetime.min.time().replace(hour=hour, minute=minute))
                scheduled_arr = scheduled_dep + timedelta(minutes=journey_mins)

                # Simulate delays - peak times more likely to be delayed
                is_peak = (7 <= hour <= 9) or (17 <= hour <= 19)

                # ~5% cancellation rate, higher on certain days
                if random.random() < (0.08 if weekday == 0 else 0.05):  # Monday blues
                    cancelled = True
                    actual_dep = None
                    actual_arr = None
                    arr_delay = None
                else:
                    cancelled = False

                    # Delay distribution: mostly on time, long tail
                    if random.random() < 0.65:
                        arr_delay = random.randint(-2, 4)  # On time
                    elif random.random() < 0.85:
                        arr_delay = random.randint(5, 15)  # Minor delay
                    elif random.random() < 0.95:
                        arr_delay = random.randint(16, 30)  # Moderate delay
                    else:
                        arr_delay = random.randint(31, 90)  # Major delay

                    # Peak times: add extra delay
                    if is_peak:
                        arr_delay += random.randint(0, 10)

                    actual_arr = scheduled_arr + timedelta(minutes=arr_delay)
                    actual_dep = scheduled_dep + timedelta(minutes=max(0, arr_delay - 5))

                # Categorize
                if cancelled:
                    status = "cancelled"
                    delay_repay = True
                elif arr_delay <= 5:
                    status = "on_time"
                    delay_repay = False
                elif arr_delay <= 29:
                    status = "late"
                    delay_repay = arr_delay >= 15
                else:
                    status = "very_late"
                    delay_repay = True

                dep_slot = f"{hour:02d}:{(minute // 30) * 30:02d}-{hour:02d}:{(minute // 30) * 30 + 29:02d}"

                records.append({
                    "rid": rid,
                    "service_uid": f"GW{random.randint(1000, 9999)}",
                    "run_date": current_date,
                    "toc_code": "GW",
                    "toc_name": "Great Western Railway",
                    "origin_crs": origin,
                    "origin_name": "Origin",
                    "dest_crs": dest,
                    "dest_name": "Destination",
                    "scheduled_dep": scheduled_dep,
                    "scheduled_arr": scheduled_arr,
                    "actual_dep": actual_dep,
                    "actual_arr": actual_arr,
                    "dep_delay_mins": int((actual_dep - scheduled_dep).total_seconds() / 60) if actual_dep else None,
                    "arr_delay_mins": arr_delay,
                    "cancelled": cancelled,
                    "part_cancelled": False,
                    "weekday": weekday,
                    "dep_hour": hour,
                    "dep_slot": dep_slot,
                    "status": status,
                    "delay_repay_eligible": delay_repay,
                })

        return pl.DataFrame(records)

    return (generate_demo_data,)


@app.cell
def _(mo):
    fetch_button = mo.ui.run_button(label="Fetch/Refresh Data")
    fetch_button
    return (fetch_button,)


@app.cell
def _(mo, fetch_button, api_email, api_password, origin_crs, dest_crs,
      date, timedelta, generate_demo_data, duckdb, DB_PATH, pl):

    mo.stop(not fetch_button.value)

    # Check if we have credentials
    use_demo = not (api_email.value and api_password.value)

    if use_demo:
        with mo.status.spinner("Generating demo data..."):
            df = generate_demo_data(origin_crs, dest_crs, days=365)

            # Store in database
            _conn = duckdb.connect(str(DB_PATH))

            # Clear existing data for this route and insert new
            _conn.execute("""
                DELETE FROM services
                WHERE origin_crs = ? AND dest_crs = ?
            """, [origin_crs, dest_crs])

            _conn.execute("INSERT INTO services SELECT * FROM df")
            _conn.close()

        data_status = mo.callout(
            mo.md(f"**Demo data loaded**: {len(df):,} services for {origin_crs} → {dest_crs}"),
            kind="info"
        )
    else:
        data_status = mo.callout(
            mo.md("**Live API**: Would fetch from HSP API (implement based on your credentials)"),
            kind="warn"
        )

    data_status
    return (data_status, use_demo)


# ============================================================================
# STATISTICS CALCULATIONS
# ============================================================================

@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ---
    ## Performance Statistics
    """)
    return


@app.cell
def _(duckdb, DB_PATH, date, timedelta, pl):
    def get_route_stats(origin: str, dest: str, days_back: int) -> pl.DataFrame:
        """Get aggregated stats for a route over N days."""
        cutoff = date.today() - timedelta(days=days_back)

        _conn = duckdb.connect(str(DB_PATH), read_only=True)
        result = _conn.execute("""
            SELECT
                COUNT(*) as total_services,
                SUM(CASE WHEN status = 'on_time' THEN 1 ELSE 0 END) as on_time,
                SUM(CASE WHEN status = 'late' THEN 1 ELSE 0 END) as late,
                SUM(CASE WHEN status = 'very_late' THEN 1 ELSE 0 END) as very_late,
                SUM(CASE WHEN status = 'cancelled' THEN 1 ELSE 0 END) as cancelled,
                SUM(CASE WHEN delay_repay_eligible THEN 1 ELSE 0 END) as delay_repay_eligible,
                AVG(arr_delay_mins) FILTER (WHERE NOT cancelled) as avg_delay,
                MEDIAN(arr_delay_mins) FILTER (WHERE NOT cancelled) as median_delay,
                PERCENTILE_CONT(0.9) WITHIN GROUP (ORDER BY arr_delay_mins) FILTER (WHERE NOT cancelled) as p90_delay
            FROM services
            WHERE origin_crs = ? AND dest_crs = ? AND run_date >= ?
        """, [origin, dest, cutoff]).pl()
        _conn.close()
        return result

    return (get_route_stats,)


@app.cell
def _(duckdb, DB_PATH, date, timedelta, pl):
    def get_weekday_stats(origin: str, dest: str, weekday: int, days_back: int = 365) -> pl.DataFrame:
        """Get stats for a specific weekday over the past year."""
        cutoff = date.today() - timedelta(days=days_back)

        _conn = duckdb.connect(str(DB_PATH), read_only=True)
        result = _conn.execute("""
            SELECT
                COUNT(*) as total_services,
                SUM(CASE WHEN status = 'on_time' THEN 1 ELSE 0 END) as on_time,
                SUM(CASE WHEN status = 'late' THEN 1 ELSE 0 END) as late,
                SUM(CASE WHEN status = 'very_late' THEN 1 ELSE 0 END) as very_late,
                SUM(CASE WHEN status = 'cancelled' THEN 1 ELSE 0 END) as cancelled,
                SUM(CASE WHEN delay_repay_eligible THEN 1 ELSE 0 END) as delay_repay_eligible,
                AVG(arr_delay_mins) FILTER (WHERE NOT cancelled) as avg_delay,
                MEDIAN(arr_delay_mins) FILTER (WHERE NOT cancelled) as median_delay
            FROM services
            WHERE origin_crs = ? AND dest_crs = ? AND weekday = ? AND run_date >= ?
        """, [origin, dest, weekday, cutoff]).pl()
        _conn.close()
        return result

    return (get_weekday_stats,)


@app.cell
def _(duckdb, DB_PATH, date, timedelta, pl):
    def get_timeslot_stats(origin: str, dest: str, dep_slot: str, days_back: int = 365) -> pl.DataFrame:
        """Get stats for a specific time slot."""
        cutoff = date.today() - timedelta(days=days_back)

        _conn = duckdb.connect(str(DB_PATH), read_only=True)
        result = _conn.execute("""
            SELECT
                COUNT(*) as total_services,
                SUM(CASE WHEN status = 'on_time' THEN 1 ELSE 0 END) as on_time,
                SUM(CASE WHEN status = 'late' THEN 1 ELSE 0 END) as late,
                SUM(CASE WHEN status = 'very_late' THEN 1 ELSE 0 END) as very_late,
                SUM(CASE WHEN status = 'cancelled' THEN 1 ELSE 0 END) as cancelled,
                SUM(CASE WHEN delay_repay_eligible THEN 1 ELSE 0 END) as delay_repay_eligible,
                AVG(arr_delay_mins) FILTER (WHERE NOT cancelled) as avg_delay,
                MEDIAN(arr_delay_mins) FILTER (WHERE NOT cancelled) as median_delay
            FROM services
            WHERE origin_crs = ? AND dest_crs = ? AND dep_slot = ? AND run_date >= ?
        """, [origin, dest, dep_slot, cutoff]).pl()
        _conn.close()
        return result

    return (get_timeslot_stats,)


@app.cell
def _(duckdb, DB_PATH, date, timedelta, pl):
    def get_delay_distribution(origin: str, dest: str, days_back: int = 365) -> pl.DataFrame:
        """Get delay distribution for histogram/CDF."""
        cutoff = date.today() - timedelta(days=days_back)

        _conn = duckdb.connect(str(DB_PATH), read_only=True)
        result = _conn.execute("""
            SELECT
                arr_delay_mins,
                cancelled,
                status,
                delay_repay_eligible
            FROM services
            WHERE origin_crs = ? AND dest_crs = ? AND run_date >= ?
            ORDER BY arr_delay_mins
        """, [origin, dest, cutoff]).pl()
        _conn.close()
        return result

    return (get_delay_distribution,)


@app.cell
def _(mo, origin_crs, dest_crs, get_route_stats):
    # Calculate stats for different time periods
    stats_7d = get_route_stats(origin_crs, dest_crs, 7)
    stats_30d = get_route_stats(origin_crs, dest_crs, 30)
    stats_365d = get_route_stats(origin_crs, dest_crs, 365)

    def format_stats_row(stats, period: str) -> dict:
        if stats.is_empty() or stats["total_services"][0] == 0:
            return {"Period": period, "Services": 0}

        _total = stats["total_services"][0]
        return {
            "Period": period,
            "Services": int(_total),
            "On Time": f"{100 * stats['on_time'][0] / _total:.1f}%",
            "Late": f"{100 * stats['late'][0] / _total:.1f}%",
            "Very Late": f"{100 * stats['very_late'][0] / _total:.1f}%",
            "Cancelled": f"{100 * stats['cancelled'][0] / _total:.1f}%",
            "Delay Repay": f"{100 * stats['delay_repay_eligible'][0] / _total:.1f}%",
            "Avg Delay": f"{stats['avg_delay'][0]:.1f} min" if stats['avg_delay'][0] else "N/A",
            "Median": f"{stats['median_delay'][0]:.0f} min" if stats['median_delay'][0] else "N/A",
        }

    route_stats_table = [
        format_stats_row(stats_7d, "Last 7 days"),
        format_stats_row(stats_30d, "Last 30 days"),
        format_stats_row(stats_365d, "Last 365 days"),
    ]

    mo.ui.table(route_stats_table, selection=None, label="Route Performance Summary")
    return stats_7d, stats_30d, stats_365d, route_stats_table, format_stats_row


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ### Performance by Day of Week
    """)
    return


@app.cell
def _(mo, origin_crs, dest_crs, get_weekday_stats):
    WEEKDAY_NAMES = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]

    weekday_rows = []
    for wd in range(7):
        stats = get_weekday_stats(origin_crs, dest_crs, wd)
        if stats.is_empty() or stats["total_services"][0] == 0:
            continue
        _total = stats["total_services"][0]
        weekday_rows.append({
            "Day": WEEKDAY_NAMES[wd],
            "Services": int(_total),
            "On Time": f"{100 * stats['on_time'][0] / _total:.1f}%",
            "Late": f"{100 * stats['late'][0] / _total:.1f}%",
            "Cancelled": f"{100 * stats['cancelled'][0] / _total:.1f}%",
            "Delay Repay": f"{100 * stats['delay_repay_eligible'][0] / _total:.1f}%",
            "Avg Delay": f"{stats['avg_delay'][0]:.1f} min" if stats['avg_delay'][0] else "N/A",
        })

    mo.ui.table(weekday_rows, selection=None, label="Performance by Day of Week (Last Year)")
    return WEEKDAY_NAMES, weekday_rows


# ============================================================================
# TIME SLOT ANALYSIS
# ============================================================================

@app.cell(hide_code=True)
def _(mo, selected_time):
    time_slot_header = mo.md(f"""
    ### Time Slot Analysis{f': **{selected_time}**' if selected_time else ''}

    {'*Select a departure time above to see time-specific stats*' if not selected_time else ''}
    """)
    time_slot_header
    return (time_slot_header,)


@app.cell
def _(mo, selected_time, origin_crs, dest_crs, get_timeslot_stats):
    if selected_time:
        # Calculate time slot (30-min buckets)
        _hour = selected_time.hour
        _slot_start = (selected_time.minute // 30) * 30
        selected_dep_slot = f"{_hour:02d}:{_slot_start:02d}-{_hour:02d}:{_slot_start + 29:02d}"

        slot_stats = get_timeslot_stats(origin_crs, dest_crs, selected_dep_slot)

        if not slot_stats.is_empty() and slot_stats["total_services"][0] > 0:
            _total = slot_stats["total_services"][0]
            slot_info = mo.md(f"""
            **Time slot: {selected_dep_slot}** ({int(_total)} services in past year)

            | Metric | Value |
            |--------|-------|
            | On Time (≤5 min) | {100 * slot_stats['on_time'][0] / _total:.1f}% |
            | Late (5-29 min) | {100 * slot_stats['late'][0] / _total:.1f}% |
            | Very Late (30+ min) | {100 * slot_stats['very_late'][0] / _total:.1f}% |
            | Cancelled | {100 * slot_stats['cancelled'][0] / _total:.1f}% |
            | **Delay Repay Eligible** | **{100 * slot_stats['delay_repay_eligible'][0] / _total:.1f}%** |
            | Average Delay | {slot_stats['avg_delay'][0]:.1f} min |
            | Median Delay | {slot_stats['median_delay'][0]:.0f} min |
            """)
        else:
            slot_info = mo.callout(mo.md("No data for this time slot"), kind="warn")
            selected_dep_slot = None
    else:
        slot_info = None
        selected_dep_slot = None

    slot_info
    return (selected_dep_slot,)


# ============================================================================
# DELAY DISTRIBUTION CHART
# ============================================================================

@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ---
    ## Delay Distribution & Refund Probability

    The chart below shows the cumulative probability of delays. Key thresholds:
    - **15 minutes**: Delay Repay threshold for most TOCs (25% refund)
    - **30 minutes**: Higher compensation tier (50% refund)
    - **60 minutes**: Full refund eligibility
    """)
    return


@app.cell
def _(origin_crs, dest_crs, get_delay_distribution, pl):
    delay_data = get_delay_distribution(origin_crs, dest_crs, 365)

    # Filter out cancelled and null delays
    delay_data_valid = delay_data.filter(
        (pl.col("cancelled") == False) &
        (pl.col("arr_delay_mins").is_not_null())
    )
    return delay_data, delay_data_valid


@app.cell
def _(delay_data_valid, pl, alt):
    if delay_data_valid.is_empty():
        delay_chart = None
    else:
        # Calculate cumulative distribution
        sorted_delays = delay_data_valid.sort("arr_delay_mins")
        _n = len(sorted_delays)

        cdf_data = sorted_delays.with_columns([
            (pl.lit(1).cum_sum() / _n * 100).alias("cumulative_pct")
        ]).select(["arr_delay_mins", "cumulative_pct"]).to_pandas()

        # Base CDF line
        cdf_line = alt.Chart(cdf_data).mark_line(color="#2563eb", strokeWidth=2).encode(
            x=alt.X("arr_delay_mins:Q", title="Arrival Delay (minutes)", scale=alt.Scale(domain=[-10, 60])),
            y=alt.Y("cumulative_pct:Q", title="Cumulative % of Services", scale=alt.Scale(domain=[0, 100]))
        )

        # Threshold lines
        thresholds = [
            {"x": 5, "label": "On Time (5 min)", "color": "#22c55e"},
            {"x": 15, "label": "Delay Repay 25%", "color": "#f59e0b"},
            {"x": 30, "label": "Delay Repay 50%", "color": "#ef4444"},
            {"x": 60, "label": "Full Refund", "color": "#7c3aed"},
        ]

        rules = alt.Chart(alt.Data(values=thresholds)).mark_rule(strokeDash=[5, 5]).encode(
            x="x:Q",
            color=alt.Color("color:N", scale=None),
            tooltip=["label:N"]
        )

        # Annotations
        annotations = alt.Chart(alt.Data(values=thresholds)).mark_text(
            align="left", dx=5, dy=-5, fontSize=10
        ).encode(
            x="x:Q",
            y=alt.value(10),
            text="label:N",
            color=alt.Color("color:N", scale=None)
        )

        delay_chart = (cdf_line + rules + annotations).properties(
            width=600,
            height=350,
            title="Cumulative Delay Distribution (Past Year)"
        ).configure_axis(
            grid=True
        )

    delay_chart
    return (delay_chart,)


@app.cell
def _(delay_data_valid, mo, pl):
    if not delay_data_valid.is_empty():
        # Calculate probabilities at key thresholds
        _n = len(delay_data_valid)

        prob_on_time = len(delay_data_valid.filter(pl.col("arr_delay_mins") <= 5)) / _n * 100
        prob_under_15 = len(delay_data_valid.filter(pl.col("arr_delay_mins") <= 15)) / _n * 100
        prob_under_30 = len(delay_data_valid.filter(pl.col("arr_delay_mins") <= 30)) / _n * 100
        prob_under_60 = len(delay_data_valid.filter(pl.col("arr_delay_mins") <= 60)) / _n * 100

        prob_table = mo.md(f"""
        ### Delay Probabilities

        | Scenario | Probability | Refund Status |
        |----------|-------------|---------------|
        | On time (≤5 min late) | **{prob_on_time:.1f}%** | No refund |
        | Less than 15 min late | {prob_under_15:.1f}% | No refund |
        | Less than 30 min late | {prob_under_30:.1f}% | May qualify for 25% |
        | Less than 60 min late | {prob_under_60:.1f}% | May qualify for 50% |
        | **15+ min late (Delay Repay)** | **{100 - prob_under_15:.1f}%** | 25%+ refund |
        | **30+ min late** | **{100 - prob_under_30:.1f}%** | 50%+ refund |
        | **60+ min late** | **{100 - prob_under_60:.1f}%** | Full refund |
        """)
    else:
        prob_table = mo.callout(mo.md("No delay data available"), kind="warn")

    prob_table
    return (prob_table,)


# ============================================================================
# HISTOGRAM VIEW
# ============================================================================

@app.cell
def _(delay_data_valid, alt):
    if not delay_data_valid.is_empty():
        hist_data = delay_data_valid.select("arr_delay_mins").to_pandas()

        histogram = alt.Chart(hist_data).mark_bar(color="#3b82f6", opacity=0.7).encode(
            x=alt.X("arr_delay_mins:Q",
                    bin=alt.Bin(step=5, extent=[-10, 60]),
                    title="Arrival Delay (minutes)"),
            y=alt.Y("count():Q", title="Number of Services")
        ).properties(
            width=600,
            height=250,
            title="Delay Distribution Histogram"
        )

        # Add mean line
        mean_val = hist_data["arr_delay_mins"].mean()
        mean_rule = alt.Chart(alt.Data(values=[{"mean": mean_val}])).mark_rule(
            color="red", strokeWidth=2, strokeDash=[5, 5]
        ).encode(x="mean:Q")

        mean_label = alt.Chart(alt.Data(values=[{"mean": mean_val, "label": f"Mean: {mean_val:.1f} min"}])).mark_text(
            align="left", dx=5, color="red"
        ).encode(x="mean:Q", y=alt.value(20), text="label:N")

        delay_histogram = histogram + mean_rule + mean_label
    else:
        delay_histogram = None

    delay_histogram
    return (delay_histogram,)


# ============================================================================
# HEATMAP: DELAY BY HOUR AND WEEKDAY
# ============================================================================

@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ---
    ## When to Travel: Delay Heatmap

    Average arrival delay by day and time. Darker = more delayed.
    """)
    return


@app.cell
def _(duckdb, DB_PATH, origin_crs, dest_crs, date, timedelta, alt):
    _cutoff = date.today() - timedelta(days=365)

    _conn = duckdb.connect(str(DB_PATH), read_only=True)
    heatmap_data = _conn.execute("""
        SELECT
            weekday,
            dep_hour,
            AVG(arr_delay_mins) FILTER (WHERE NOT cancelled) as avg_delay,
            COUNT(*) as n_services
        FROM services
        WHERE origin_crs = ? AND dest_crs = ? AND run_date >= ?
        GROUP BY weekday, dep_hour
        HAVING COUNT(*) >= 5
    """, [origin_crs, dest_crs, _cutoff]).pl()
    _conn.close()

    if not heatmap_data.is_empty():
        heatmap_df = heatmap_data.to_pandas()
        heatmap_df["day_name"] = heatmap_df["weekday"].map({
            0: "Mon", 1: "Tue", 2: "Wed", 3: "Thu", 4: "Fri", 5: "Sat", 6: "Sun"
        })

        heatmap = alt.Chart(heatmap_df).mark_rect().encode(
            x=alt.X("dep_hour:O", title="Departure Hour"),
            y=alt.Y("day_name:O", title="Day", sort=["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]),
            color=alt.Color("avg_delay:Q",
                           scale=alt.Scale(scheme="reds", domain=[0, 20]),
                           title="Avg Delay (min)"),
            tooltip=["day_name:N", "dep_hour:O",
                    alt.Tooltip("avg_delay:Q", format=".1f", title="Avg Delay"),
                    alt.Tooltip("n_services:Q", title="Services")]
        ).properties(
            width=500,
            height=200,
            title="Average Delay by Day and Hour"
        )
    else:
        heatmap = None

    heatmap
    return (heatmap,)


# ============================================================================
# FOOTER
# ============================================================================

@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ---

    *Data definitions:*
    - **On time**: Arrived within 5 minutes of scheduled time
    - **Late**: Arrived 5-29 minutes after scheduled time
    - **Very late**: Arrived 30+ minutes late
    - **Delay Repay**: Services where you may claim compensation (15+ min for most TOCs)

    *Data source: National Rail HSP API or demo data*
    """)
    return


if __name__ == "__main__":
    app.run()
