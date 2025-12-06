"""Historic Service Performance (HSP) API client.

National Rail's HSP API provides historical train performance data.
Register at https://opendata.nationalrail.co.uk and enable HSP access.
"""

import base64
from datetime import date
from dataclasses import dataclass

import httpx

HSP_BASE_URL = "https://hsp-prod.rockshore.net/api/v1"


@dataclass
class HSPCredentials:
    """API credentials for HSP."""
    email: str
    password: str

    @property
    def auth_header(self) -> str:
        """Return Base64-encoded authorization header value."""
        return base64.b64encode(f"{self.email}:{self.password}".encode()).decode()

    def is_valid(self) -> bool:
        """Check if credentials are non-empty."""
        return bool(self.email and self.password)


def fetch_service_metrics(
    credentials: HSPCredentials,
    origin: str,
    dest: str,
    from_date: date,
    to_date: date,
    from_time: str = "0000",
    to_time: str = "2359",
    days: str = "WEEKDAY",
    tolerances: list[str] | None = None,
) -> dict:
    """
    Fetch aggregated service metrics from HSP API.

    Args:
        credentials: HSP API credentials
        origin: Origin station CRS code (e.g., "CDF")
        dest: Destination station CRS code (e.g., "PAD")
        from_date: Start date for query
        to_date: End date for query
        from_time: Start time in HHMM format (default "0000")
        to_time: End time in HHMM format (default "2359")
        days: Day filter - "WEEKDAY", "SATURDAY", or "SUNDAY"
        tolerances: List of tolerance values in minutes (default ["0", "5", "10", "15", "30"])

    Returns:
        API response as dict containing service metrics
    """
    if tolerances is None:
        tolerances = ["0", "5", "10", "15", "30"]

    payload = {
        "from_loc": origin,
        "to_loc": dest,
        "from_time": from_time,
        "to_time": to_time,
        "from_date": from_date.strftime("%Y-%m-%d"),
        "to_date": to_date.strftime("%Y-%m-%d"),
        "days": days,
        "tolerance": tolerances,
    }

    response = httpx.post(
        f"{HSP_BASE_URL}/serviceMetrics",
        json=payload,
        headers={
            "Authorization": f"Basic {credentials.auth_header}",
            "Content-Type": "application/json",
        },
        timeout=30.0,
    )
    response.raise_for_status()
    return response.json()


def fetch_service_details(
    credentials: HSPCredentials,
    rid: str,
) -> dict:
    """
    Fetch detailed information for a specific service.

    Args:
        credentials: HSP API credentials
        rid: Service RID (unique identifier for a service on a specific day)

    Returns:
        API response as dict containing service details
    """
    payload = {"rid": rid}

    response = httpx.post(
        f"{HSP_BASE_URL}/serviceDetails",
        json=payload,
        headers={
            "Authorization": f"Basic {credentials.auth_header}",
            "Content-Type": "application/json",
        },
        timeout=30.0,
    )
    response.raise_for_status()
    return response.json()


def fetch_service_details_batch(
    credentials: HSPCredentials,
    rids: list[str],
    max_services: int = 100,
) -> list[dict]:
    """
    Fetch details for multiple services.

    Args:
        credentials: HSP API credentials
        rids: List of service RIDs
        max_services: Maximum number of services to fetch (to avoid rate limiting)

    Returns:
        List of service detail dicts (skips failed requests)
    """
    results = []
    for rid in rids[:max_services]:
        try:
            details = fetch_service_details(credentials, rid)
            results.append(details)
        except httpx.HTTPError:
            continue
    return results
