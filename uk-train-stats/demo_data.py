"""Generate realistic demo data for testing without API access."""

import random
from datetime import date, datetime, timedelta

import polars as pl


def generate_services(
    origin: str,
    dest: str,
    days: int = 365,
    seed: int = 42,
) -> pl.DataFrame:
    """
    Generate realistic synthetic train service data.

    Args:
        origin: Origin station CRS code
        dest: Destination station CRS code
        days: Number of days of history to generate
        seed: Random seed for reproducibility

    Returns:
        DataFrame with columns: rid, run_date, toc_code, origin_crs, dest_crs,
        scheduled_dep, scheduled_arr, actual_arr, arr_delay_mins, cancelled,
        weekday, dep_hour, dep_slot, status, delay_repay_eligible
    """
    random.seed(seed)

    base_date = date.today() - timedelta(days=days)
    journey_mins = _get_journey_duration(origin, dest)
    dep_times = _get_departure_times()

    records = []
    for day_offset in range(days):
        current_date = base_date + timedelta(days=day_offset)
        weekday = current_date.weekday()
        times_today = dep_times if weekday < 5 else dep_times[::2]

        for dep_time in times_today:
            record = _generate_single_service(
                origin, dest, current_date, dep_time, journey_mins, weekday
            )
            records.append(record)

    return pl.DataFrame(records)


def _get_journey_duration(origin: str, dest: str) -> int:
    """Return typical journey duration in minutes for a route."""
    # Cardiff to London is about 2 hours
    if (origin == "CDF" and dest == "PAD") or (origin == "PAD" and dest == "CDF"):
        return 120
    return 90


def _get_departure_times() -> list[str]:
    """Return typical departure times for a major route."""
    return [
        "06:15", "07:00", "07:30", "08:00", "08:30", "09:00", "10:00",
        "11:00", "12:00", "13:00", "14:00", "15:00", "16:00", "17:00",
        "17:30", "18:00", "18:30", "19:00", "20:00", "21:00", "22:00",
    ]


def _generate_single_service(
    origin: str,
    dest: str,
    run_date: date,
    dep_time: str,
    journey_mins: int,
    weekday: int,
) -> dict:
    """Generate a single service record with realistic delay patterns."""
    hour, minute = map(int, dep_time.split(":"))
    scheduled_dep = datetime.combine(
        run_date, datetime.min.time().replace(hour=hour, minute=minute)
    )
    scheduled_arr = scheduled_dep + timedelta(minutes=journey_mins)

    cancelled, arr_delay = _simulate_delay(hour, weekday)

    if cancelled:
        actual_arr = None
        status = "cancelled"
        delay_repay = True
    else:
        actual_arr = scheduled_arr + timedelta(minutes=arr_delay)
        status, delay_repay = _categorize_delay(arr_delay)

    return {
        "rid": f"{origin}_{dest}_{run_date}_{dep_time.replace(':', '')}",
        "run_date": run_date,
        "toc_code": "GW",
        "origin_crs": origin,
        "dest_crs": dest,
        "scheduled_dep": scheduled_dep,
        "scheduled_arr": scheduled_arr,
        "actual_arr": actual_arr,
        "arr_delay_mins": arr_delay,
        "cancelled": cancelled,
        "weekday": weekday,
        "dep_hour": hour,
        "dep_slot": f"{hour:02d}:{(minute // 30) * 30:02d}-{hour:02d}:{(minute // 30) * 30 + 29:02d}",
        "status": status,
        "delay_repay_eligible": delay_repay,
    }


def _simulate_delay(hour: int, weekday: int) -> tuple[bool, int | None]:
    """
    Simulate cancellation and delay based on time and day patterns.

    Returns:
        Tuple of (cancelled, delay_minutes)
    """
    is_peak = (7 <= hour <= 9) or (17 <= hour <= 19)
    is_monday = weekday == 0

    # Cancellation: ~5% base, higher on Mondays
    cancel_prob = 0.08 if is_monday else 0.05
    if random.random() < cancel_prob:
        return True, None

    # Delay distribution: mostly on time, long tail
    r = random.random()
    if r < 0.65:
        delay = random.randint(-2, 4)
    elif r < 0.85:
        delay = random.randint(5, 15)
    elif r < 0.95:
        delay = random.randint(16, 30)
    else:
        delay = random.randint(31, 90)

    # Peak times get extra delay
    if is_peak:
        delay += random.randint(0, 10)

    return False, delay


def _categorize_delay(arr_delay: int) -> tuple[str, bool]:
    """
    Categorize delay into status and refund eligibility.

    Returns:
        Tuple of (status, delay_repay_eligible)
    """
    if arr_delay <= 5:
        return "on_time", False
    elif arr_delay <= 29:
        return "late", arr_delay >= 15
    else:
        return "very_late", True
