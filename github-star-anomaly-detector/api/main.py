"""
FastAPI Backend for Fake Stars Detector

Provides endpoint to analyze GitHub repositories for fake stars.
"""

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import sys
import os

# Add parent directory to path to import modules
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data_fetcher import GitHubStarFetcher, GitHubAPIError
from region_detector import identify_regions
from account_verifier import AccountVerifier
from anomaly_methods import DetectionConfig

app = FastAPI(
    title="Fake Stars Detector API",
    description="Detect potentially fake GitHub stars using time-series anomaly detection and account verification",
    version="1.0.0",
)

# Configure CORS for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, restrict this
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class RegionResult(BaseModel):
    """Result for a single anomalous region."""

    start_date: str
    end_date: str
    duration_days: int
    stars_in_region: int
    avg_daily_stars: float
    growth_rate_change: float
    confidence: float
    detected_by: list[str]
    # Account verification results
    suspicious_count: int
    suspicious_percentage: float
    accounts_analyzed: int
    sample_suspicious_accounts: list[str]


class AnalysisResult(BaseModel):
    """Complete analysis result for a repository."""

    owner: str
    repo: str
    total_stars: int
    total_days: int
    baseline_avg_stars: float
    anomalous_regions: list[RegionResult]
    overall_suspicious_percentage: float
    estimated_fake_stars: int


@app.get("/")
async def root():
    """Health check endpoint."""
    return {
        "status": "ok",
        "service": "Fake Stars Detector API",
        "version": "1.0.0",
    }


@app.get("/analyze/{owner}/{repo}", response_model=AnalysisResult)
async def analyze_repo(
    owner: str,
    repo: str,
    token: str = Query(None, description="GitHub API token for higher rate limits"),
    max_pages: int = Query(10, description="Maximum pages of stars to fetch"),
    sample_size: int = Query(
        50, description="Sample size for account verification per region"
    ),
):
    """
    Analyze a GitHub repository for fake stars.

    This endpoint:
    1. Fetches star history from GitHub
    2. Detects anomalous regions in growth rate
    3. Verifies accounts that starred during anomalous periods
    4. Returns comprehensive analysis

    Args:
        owner: Repository owner
        repo: Repository name
        token: Optional GitHub API token
        max_pages: Maximum pages to fetch (100 stars per page)
        sample_size: How many accounts to check per region

    Returns:
        AnalysisResult with detected regions and fake percentage estimates
    """
    try:
        # Step 1: Fetch star history
        fetcher = GitHubStarFetcher(token=token)
        stars_raw = fetcher.fetch_stars(owner, repo, max_pages=max_pages)

        if len(stars_raw) == 0:
            raise HTTPException(
                status_code=404, detail=f"No stars found for {owner}/{repo}"
            )

        # Convert to time series
        time_series = fetcher.create_time_series(stars_raw, freq="1d")

        # Step 2: Detect anomalous regions
        config = DetectionConfig(ensemble_min_votes=2)
        regions = identify_regions(
            time_series,
            method="ensemble",
            config=config,
            min_region_days=1,
            merge_gap_days=2,
        )

        if not regions:
            # No anomalies detected
            return AnalysisResult(
                owner=owner,
                repo=repo,
                total_stars=int(time_series["cumulative_stars"][-1]),
                total_days=len(time_series),
                baseline_avg_stars=time_series["new_stars"].mean(),
                anomalous_regions=[],
                overall_suspicious_percentage=0.0,
                estimated_fake_stars=0,
            )

        # Step 3: Verify accounts in each region
        verifier = AccountVerifier(token=token)
        region_results = []

        for region in regions:
            # Get stargazers in this date range
            stargazers_in_region = [
                (row["starred_at"], row["user"])
                for row in stars_raw.iter_rows(named=True)
                if region.start_date <= row["starred_at"] <= region.end_date
            ]

            # Analyze accounts (async)
            account_analysis = await verifier.analyze_region(
                stargazers_in_region, sample_size=sample_size
            )

            region_results.append(
                RegionResult(
                    start_date=region.start_date.isoformat(),
                    end_date=region.end_date.isoformat(),
                    duration_days=region.duration_days,
                    stars_in_region=region.stars_in_region,
                    avg_daily_stars=region.avg_daily_stars,
                    growth_rate_change=region.growth_rate_change,
                    confidence=region.confidence,
                    detected_by=region.detected_by,
                    suspicious_count=account_analysis.suspicious_count,
                    suspicious_percentage=account_analysis.suspicious_percentage,
                    accounts_analyzed=account_analysis.accounts_analyzed,
                    sample_suspicious_accounts=account_analysis.sample_suspicious_accounts,
                )
            )

        # Calculate overall statistics
        total_stars = int(time_series["cumulative_stars"][-1])
        total_suspicious = sum(r.suspicious_count for r in region_results)
        overall_suspicious_pct = (
            (total_suspicious / total_stars * 100) if total_stars > 0 else 0.0
        )

        return AnalysisResult(
            owner=owner,
            repo=repo,
            total_stars=total_stars,
            total_days=len(time_series),
            baseline_avg_stars=time_series["new_stars"].mean(),
            anomalous_regions=region_results,
            overall_suspicious_percentage=overall_suspicious_pct,
            estimated_fake_stars=total_suspicious,
        )

    except GitHubAPIError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
