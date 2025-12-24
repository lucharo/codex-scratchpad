"use client";

import { useState } from "react";
import { useRouter } from "next/navigation";

export default function Home() {
  const [input, setInput] = useState("");
  const router = useRouter();

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();

    // Parse input (can be full URL or owner/repo)
    let owner = "";
    let repo = "";

    if (input.includes("github.com")) {
      // Parse from URL
      const match = input.match(/github\.com\/([^\/]+)\/([^\/\?#]+)/);
      if (match) {
        owner = match[1];
        repo = match[2];
      }
    } else if (input.includes("/")) {
      // Already in owner/repo format
      [owner, repo] = input.split("/");
    }

    if (owner && repo) {
      router.push(`/${owner}/${repo}`);
    }
  };

  return (
    <div className="min-h-screen flex flex-col items-center justify-center p-8">
      <div className="max-w-2xl w-full space-y-8">
        <div className="text-center">
          <h1 className="text-5xl font-bold text-gray-900 mb-4">
            Fake Stars Detector
          </h1>
          <p className="text-lg text-gray-600">
            Analyze GitHub repositories for potentially fake stars using
            anomaly detection and account verification
          </p>
        </div>

        <form onSubmit={handleSubmit} className="mt-8">
          <div className="flex gap-2">
            <input
              type="text"
              value={input}
              onChange={(e) => setInput(e.target.value)}
              placeholder="owner/repo or github.com/owner/repo"
              className="flex-1 px-4 py-3 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent outline-none"
            />
            <button
              type="submit"
              className="px-6 py-3 bg-blue-600 text-white rounded-lg hover:bg-blue-700 transition-colors font-medium"
            >
              Analyze
            </button>
          </div>
        </form>

        <div className="mt-12 grid grid-cols-1 md:grid-cols-3 gap-6">
          <div className="p-6 bg-white rounded-lg shadow-sm border border-gray-200">
            <h3 className="font-semibold text-gray-900 mb-2">
              Time-Series Analysis
            </h3>
            <p className="text-sm text-gray-600">
              Detects anomalous regions in star growth patterns
            </p>
          </div>

          <div className="p-6 bg-white rounded-lg shadow-sm border border-gray-200">
            <h3 className="font-semibold text-gray-900 mb-2">
              Account Verification
            </h3>
            <p className="text-sm text-gray-600">
              Analyzes accounts for bot-like characteristics
            </p>
          </div>

          <div className="p-6 bg-white rounded-lg shadow-sm border border-gray-200">
            <h3 className="font-semibold text-gray-900 mb-2">
              Live Analysis
            </h3>
            <p className="text-sm text-gray-600">
              Real-time checking with no caching required
            </p>
          </div>
        </div>

        <div className="text-center text-sm text-gray-500 mt-12">
          <p>
            Made with{" "}
            <a
              href="https://github.com/lucharo/codex-scratchpad"
              className="text-blue-600 hover:underline"
            >
              Claude Code
            </a>
          </p>
        </div>
      </div>
    </div>
  );
}
