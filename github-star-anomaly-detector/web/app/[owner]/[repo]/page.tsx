"use client";

import { useEffect, useState } from "react";
import { useParams } from "next/navigation";

interface RegionResult {
  start_date: string;
  end_date: string;
  duration_days: number;
  stars_in_region: number;
  avg_daily_stars: number;
  growth_rate_change: number;
  confidence: number;
  detected_by: string[];
  suspicious_count: number;
  suspicious_percentage: number;
  accounts_analyzed: number;
  sample_suspicious_accounts: string[];
}

interface AnalysisResult {
  owner: string;
  repo: string;
  total_stars: number;
  total_days: number;
  baseline_avg_stars: number;
  anomalous_regions: RegionResult[];
  overall_suspicious_percentage: number;
  estimated_fake_stars: number;
}

export default function AnalyzePage() {
  const params = useParams();
  const { owner, repo } = params;

  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [data, setData] = useState<AnalysisResult | null>(null);

  useEffect(() => {
    async function fetchAnalysis() {
      try {
        setLoading(true);
        setError(null);

        const response = await fetch(
          `http://localhost:8000/analyze/${owner}/${repo}`
        );

        if (!response.ok) {
          const errorData = await response.json();
          throw new Error(errorData.detail || "Failed to analyze repository");
        }

        const result = await response.json();
        setData(result);
      } catch (err) {
        setError(err instanceof Error ? err.message : "An error occurred");
      } finally {
        setLoading(false);
      }
    }

    if (owner && repo) {
      fetchAnalysis();
    }
  }, [owner, repo]);

  if (loading) {
    return (
      <div className="min-h-screen flex items-center justify-center">
        <div className="text-center">
          <div className="inline-block animate-spin rounded-full h-12 w-12 border-b-2 border-blue-600 mb-4"></div>
          <p className="text-gray-600">
            Analyzing {owner}/{repo}...
          </p>
          <p className="text-sm text-gray-500 mt-2">
            This may take a minute as we fetch star history and verify accounts
          </p>
        </div>
      </div>
    );
  }

  if (error) {
    return (
      <div className="min-h-screen flex items-center justify-center p-8">
        <div className="max-w-2xl w-full bg-red-50 border border-red-200 rounded-lg p-6">
          <h2 className="text-xl font-semibold text-red-900 mb-2">Error</h2>
          <p className="text-red-700">{error}</p>
          <a
            href="/"
            className="mt-4 inline-block text-blue-600 hover:underline"
          >
            ← Back to home
          </a>
        </div>
      </div>
    );
  }

  if (!data) {
    return null;
  }

  return (
    <div className="min-h-screen p-8">
      <div className="max-w-6xl mx-auto">
        {/* Header */}
        <div className="mb-8">
          <a href="/" className="text-blue-600 hover:underline text-sm mb-4 inline-block">
            ← Back to home
          </a>
          <h1 className="text-4xl font-bold text-gray-900 mb-2">
            {data.owner}/{data.repo}
          </h1>
          <p className="text-gray-600">
            Analysis of {data.total_stars.toLocaleString()} stars over{" "}
            {data.total_days} days
          </p>
        </div>

        {/* Summary Cards */}
        <div className="grid grid-cols-1 md:grid-cols-4 gap-6 mb-8">
          <div className="bg-white rounded-lg shadow-sm border border-gray-200 p-6">
            <div className="text-sm text-gray-600 mb-1">Total Stars</div>
            <div className="text-3xl font-bold text-gray-900">
              {data.total_stars.toLocaleString()}
            </div>
          </div>

          <div className="bg-white rounded-lg shadow-sm border border-gray-200 p-6">
            <div className="text-sm text-gray-600 mb-1">Baseline Avg/Day</div>
            <div className="text-3xl font-bold text-gray-900">
              {data.baseline_avg_stars.toFixed(1)}
            </div>
          </div>

          <div className="bg-white rounded-lg shadow-sm border border-gray-200 p-6">
            <div className="text-sm text-gray-600 mb-1">Anomalous Regions</div>
            <div className="text-3xl font-bold text-blue-600">
              {data.anomalous_regions.length}
            </div>
          </div>

          <div className="bg-white rounded-lg shadow-sm border border-red-200 p-6 bg-red-50">
            <div className="text-sm text-red-700 mb-1">Estimated Fake Stars</div>
            <div className="text-3xl font-bold text-red-600">
              {data.estimated_fake_stars.toLocaleString()}
            </div>
            <div className="text-xs text-red-600 mt-1">
              {data.overall_suspicious_percentage.toFixed(1)}% of total
            </div>
          </div>
        </div>

        {/* Anomalous Regions */}
        {data.anomalous_regions.length > 0 ? (
          <div className="space-y-6">
            <h2 className="text-2xl font-bold text-gray-900">
              Anomalous Regions Detected
            </h2>

            {data.anomalous_regions.map((region, index) => (
              <div
                key={index}
                className="bg-white rounded-lg shadow-sm border border-gray-200 p-6"
              >
                <div className="flex justify-between items-start mb-4">
                  <div>
                    <h3 className="text-lg font-semibold text-gray-900 mb-1">
                      Region #{index + 1}
                    </h3>
                    <p className="text-sm text-gray-600">
                      {new Date(region.start_date).toLocaleDateString()} -{" "}
                      {new Date(region.end_date).toLocaleDateString()} (
                      {region.duration_days} days)
                    </p>
                  </div>
                  <div className="flex gap-2">
                    {region.detected_by.map((method) => (
                      <span
                        key={method}
                        className="px-2 py-1 bg-blue-100 text-blue-700 text-xs rounded"
                      >
                        {method}
                      </span>
                    ))}
                  </div>
                </div>

                <div className="grid grid-cols-2 md:grid-cols-4 gap-4 mb-4">
                  <div>
                    <div className="text-xs text-gray-600">Stars in Region</div>
                    <div className="text-xl font-semibold text-gray-900">
                      {region.stars_in_region}
                    </div>
                  </div>

                  <div>
                    <div className="text-xs text-gray-600">Avg/Day</div>
                    <div className="text-xl font-semibold text-gray-900">
                      {region.avg_daily_stars.toFixed(1)}
                    </div>
                  </div>

                  <div>
                    <div className="text-xs text-gray-600">Growth Change</div>
                    <div className={`text-xl font-semibold ${
                      region.growth_rate_change > 0 ? 'text-red-600' : 'text-green-600'
                    }`}>
                      {region.growth_rate_change > 0 ? '+' : ''}
                      {region.growth_rate_change.toFixed(1)}%
                    </div>
                  </div>

                  <div>
                    <div className="text-xs text-gray-600">Confidence</div>
                    <div className="text-xl font-semibold text-gray-900">
                      {(region.confidence * 100).toFixed(0)}%
                    </div>
                  </div>
                </div>

                <div className="border-t border-gray-200 pt-4">
                  <h4 className="font-semibold text-gray-900 mb-2">
                    Account Verification Results
                  </h4>
                  <div className="grid grid-cols-2 md:grid-cols-3 gap-4 mb-3">
                    <div>
                      <div className="text-xs text-gray-600">Accounts Analyzed</div>
                      <div className="text-lg font-semibold text-gray-900">
                        {region.accounts_analyzed}
                      </div>
                    </div>

                    <div>
                      <div className="text-xs text-gray-600">Suspicious Accounts</div>
                      <div className="text-lg font-semibold text-red-600">
                        {region.suspicious_count}
                      </div>
                    </div>

                    <div>
                      <div className="text-xs text-gray-600">Suspicious %</div>
                      <div className="text-lg font-semibold text-red-600">
                        {region.suspicious_percentage.toFixed(1)}%
                      </div>
                    </div>
                  </div>

                  {region.sample_suspicious_accounts.length > 0 && (
                    <div>
                      <div className="text-xs text-gray-600 mb-2">
                        Sample Suspicious Accounts:
                      </div>
                      <div className="flex flex-wrap gap-2">
                        {region.sample_suspicious_accounts.map((username) => (
                          <a
                            key={username}
                            href={`https://github.com/${username}`}
                            target="_blank"
                            rel="noopener noreferrer"
                            className="text-xs px-2 py-1 bg-gray-100 text-gray-700 rounded hover:bg-gray-200"
                          >
                            @{username}
                          </a>
                        ))}
                      </div>
                    </div>
                  )}
                </div>
              </div>
            ))}
          </div>
        ) : (
          <div className="bg-green-50 border border-green-200 rounded-lg p-8 text-center">
            <h2 className="text-xl font-semibold text-green-900 mb-2">
              No Anomalies Detected
            </h2>
            <p className="text-green-700">
              This repository shows normal, organic growth patterns with no suspicious activity detected.
            </p>
          </div>
        )}
      </div>
    </div>
  );
}
