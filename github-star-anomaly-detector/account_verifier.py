"""
Account Verification Module

Analyzes GitHub accounts to detect potentially fake or bot accounts.
All checks are async for performance.
"""

import httpx
import asyncio
from datetime import datetime
from typing import Optional
from dataclasses import dataclass


@dataclass
class AccountScore:
    """Score for a single account with breakdown of checks."""

    username: str
    is_suspicious: bool
    confidence: float  # 0.0 - 1.0
    checks: dict[str, bool]
    reasons: list[str]


@dataclass
class RegionAnalysis:
    """Analysis results for an anomalous region."""

    start_date: datetime
    end_date: datetime
    total_stars: int
    suspicious_count: int
    suspicious_percentage: float
    accounts_analyzed: int
    sample_suspicious_accounts: list[str]


class AccountVerifier:
    """Verifies GitHub accounts to detect fake/bot accounts."""

    def __init__(self, token: Optional[str] = None):
        """
        Initialize verifier.

        Args:
            token: GitHub API token for higher rate limits
        """
        self.base_url = "https://api.github.com"
        self.headers = {
            "Accept": "application/vnd.github.v3+json",
        }
        if token:
            self.headers["Authorization"] = f"token {token}"

    async def verify_account(
        self, username: str, client: httpx.AsyncClient
    ) -> AccountScore:
        """
        Verify a single account with multiple async checks.

        Args:
            username: GitHub username to check
            client: Async HTTP client

        Returns:
            AccountScore with suspiciousness assessment
        """
        checks = {}
        reasons = []

        try:
            # Fetch user data
            user_url = f"{self.base_url}/users/{username}"
            user_response = await client.get(user_url, headers=self.headers)

            if user_response.status_code != 200:
                # If we can't fetch user, assume suspicious
                return AccountScore(
                    username=username,
                    is_suspicious=True,
                    confidence=0.5,
                    checks={"fetch_failed": True},
                    reasons=["Failed to fetch user data"],
                )

            user_data = user_response.json()

            # Check 1: Account age
            created_at = datetime.fromisoformat(
                user_data["created_at"].replace("Z", "+00:00")
            )
            account_age_days = (datetime.now(created_at.tzinfo) - created_at).days
            is_young = account_age_days < 90  # Less than 3 months
            checks["is_young_account"] = is_young
            if is_young:
                reasons.append(f"Account only {account_age_days} days old")

            # Check 2: No repositories
            public_repos = user_data.get("public_repos", 0)
            has_no_repos = public_repos == 0
            checks["has_no_repos"] = has_no_repos
            if has_no_repos:
                reasons.append("No public repositories")

            # Check 3: No followers (isolated account)
            followers = user_data.get("followers", 0)
            is_isolated = followers == 0
            checks["is_isolated"] = is_isolated
            if is_isolated:
                reasons.append("No followers")

            # Check 4: Generic profile (no name, bio, or company)
            has_generic_profile = (
                not user_data.get("name")
                and not user_data.get("bio")
                and not user_data.get("company")
            )
            checks["has_generic_profile"] = has_generic_profile
            if has_generic_profile:
                reasons.append("Generic profile (no name, bio, or company)")

            # Check 5: Fetch contribution activity (expensive, but valuable)
            # For MVP, we'll skip this to save API calls
            # Could check: events, commits, PRs, issues

            # Calculate suspiciousness
            suspicious_flags = sum(
                [
                    is_young,
                    has_no_repos,
                    is_isolated,
                    has_generic_profile,
                ]
            )

            # Confidence based on number of flags
            confidence = min(suspicious_flags / 4.0, 1.0)

            # Suspicious if 2+ flags
            is_suspicious = suspicious_flags >= 2

            return AccountScore(
                username=username,
                is_suspicious=is_suspicious,
                confidence=confidence,
                checks=checks,
                reasons=reasons,
            )

        except Exception as e:
            # On error, return low-confidence suspicious
            return AccountScore(
                username=username,
                is_suspicious=True,
                confidence=0.3,
                checks={"error": True},
                reasons=[f"Error checking account: {str(e)}"],
            )

    async def verify_accounts_batch(
        self, usernames: list[str], max_concurrent: int = 10
    ) -> list[AccountScore]:
        """
        Verify multiple accounts concurrently.

        Args:
            usernames: List of GitHub usernames
            max_concurrent: Maximum concurrent requests

        Returns:
            List of AccountScore objects
        """
        semaphore = asyncio.Semaphore(max_concurrent)

        async def verify_with_semaphore(username: str, client: httpx.AsyncClient):
            async with semaphore:
                return await self.verify_account(username, client)

        async with httpx.AsyncClient(timeout=30.0) as client:
            tasks = [verify_with_semaphore(username, client) for username in usernames]
            return await asyncio.gather(*tasks)

    async def analyze_region(
        self,
        stargazers_in_region: list[tuple[datetime, str]],
        sample_size: Optional[int] = None,
    ) -> RegionAnalysis:
        """
        Analyze accounts that starred during a specific region.

        Args:
            stargazers_in_region: List of (starred_at, username) tuples
            sample_size: Optional limit on accounts to check (for large regions)

        Returns:
            RegionAnalysis with fake account statistics
        """
        if not stargazers_in_region:
            return RegionAnalysis(
                start_date=datetime.now(),
                end_date=datetime.now(),
                total_stars=0,
                suspicious_count=0,
                suspicious_percentage=0.0,
                accounts_analyzed=0,
                sample_suspicious_accounts=[],
            )

        # Get date range
        dates = [s[0] for s in stargazers_in_region]
        start_date = min(dates)
        end_date = max(dates)

        # Sample if needed
        usernames = [s[1] for s in stargazers_in_region]
        total_stars = len(usernames)

        if sample_size and len(usernames) > sample_size:
            # Random sample for large regions
            import random

            usernames = random.sample(usernames, sample_size)

        # Verify accounts
        scores = await self.verify_accounts_batch(usernames)

        # Calculate statistics
        suspicious_accounts = [s for s in scores if s.is_suspicious]
        suspicious_count = len(suspicious_accounts)
        accounts_analyzed = len(scores)

        # Extrapolate to full region if sampled
        if sample_size and accounts_analyzed < total_stars:
            suspicious_percentage = (suspicious_count / accounts_analyzed) * 100
            estimated_suspicious = int(
                (suspicious_count / accounts_analyzed) * total_stars
            )
        else:
            suspicious_percentage = (suspicious_count / total_stars) * 100
            estimated_suspicious = suspicious_count

        # Get sample usernames
        sample_usernames = [s.username for s in suspicious_accounts[:5]]

        return RegionAnalysis(
            start_date=start_date,
            end_date=end_date,
            total_stars=total_stars,
            suspicious_count=estimated_suspicious,
            suspicious_percentage=suspicious_percentage,
            accounts_analyzed=accounts_analyzed,
            sample_suspicious_accounts=sample_usernames,
        )


def detect_account_clusters(accounts: list[AccountScore]) -> list[list[str]]:
    """
    Detect clusters of similar accounts (potential bot networks).

    Args:
        accounts: List of AccountScore objects

    Returns:
        List of clusters (each cluster is a list of usernames)
    """
    # Simple clustering based on creation date and naming patterns
    # For MVP, just flag this as a future feature
    # Could use:
    # - Similar creation dates (within hours)
    # - Similar naming patterns (e.g., user1234, user1235)
    # - Same follower/following patterns

    # TODO: Implement clustering algorithm
    return []
