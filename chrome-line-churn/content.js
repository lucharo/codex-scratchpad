/**
 * Line Churn for GitHub - Content Script
 * Highlights lines based on how frequently they've been modified.
 *
 * Limitation: Uses GitHub API which is rate-limited (60/hr unauth, 5000/hr with token)
 */

(async function() {
  'use strict';

  // Parse current GitHub file URL
  const match = window.location.pathname.match(/^\/([^/]+)\/([^/]+)\/blob\/([^/]+)\/(.+)$/);
  if (!match) return;

  const [, owner, repo, branch, filePath] = match;
  const cacheKey = `churn:${owner}/${repo}/${filePath}`;

  // Check cache first (stored for 1 hour)
  const cached = await getCache(cacheKey);
  if (cached) {
    applyChurn(cached);
    return;
  }

  // Show loading indicator
  showStatus('Analyzing churn...');

  try {
    const churn = await analyzeChurn(owner, repo, filePath);
    if (churn) {
      await setCache(cacheKey, churn);
      applyChurn(churn);
      showStatus(`Max churn: ${churn.maxChurn}`, 3000);
    }
  } catch (err) {
    console.error('Line Churn error:', err);
    showStatus('Error: ' + err.message, 5000);
  }
})();

/**
 * Fetch commit history and analyze line churn
 */
async function analyzeChurn(owner, repo, filePath) {
  // Get commits that touched this file (limited to 100 for API efficiency)
  const commitsUrl = `https://api.github.com/repos/${owner}/${repo}/commits?path=${encodeURIComponent(filePath)}&per_page=100`;
  const commitsRes = await fetch(commitsUrl);

  if (!commitsRes.ok) {
    if (commitsRes.status === 403) throw new Error('API rate limited');
    throw new Error(`GitHub API error: ${commitsRes.status}`);
  }

  const commits = await commitsRes.json();
  if (commits.length === 0) return null;

  // Fetch diffs for each commit (limit to first 30 to avoid rate limits)
  const maxCommits = Math.min(commits.length, 30);
  const diffs = [];

  for (let i = 0; i < maxCommits; i++) {
    const sha = commits[i].sha;
    const diffUrl = `https://api.github.com/repos/${owner}/${repo}/commits/${sha}`;
    const diffRes = await fetch(diffUrl, {
      headers: { 'Accept': 'application/vnd.github.v3.diff' }
    });

    if (diffRes.ok) {
      diffs.push(await diffRes.text());
    }

    // Small delay to be nice to the API
    await sleep(100);
  }

  const diffHistory = diffs.join('\n');

  // Get current file content from the page
  const lines = getCurrentLines();
  if (!lines) return null;

  // Count churn for each line
  const churnData = [];
  let maxChurn = 0;

  for (let i = 0; i < lines.length; i++) {
    const content = lines[i].trim();
    let count = 0;

    if (content.length >= 3) {
      count = countInDiffs(content, diffHistory);
    }

    maxChurn = Math.max(maxChurn, count);
    churnData.push({ line: i, count, normalized: 0 });
  }

  // Normalize
  if (maxChurn > 0) {
    for (const d of churnData) {
      d.normalized = d.count / maxChurn;
    }
  }

  return { lines: churnData, maxChurn };
}

/**
 * Count occurrences of a line in diff history
 */
function countInDiffs(lineContent, diffHistory) {
  if (!lineContent) return 0;

  // Escape regex special chars
  const escaped = lineContent.replace(/[.*+?^${}()|[\]\\]/g, '\\$&');

  // Count additions and removals
  const addPattern = new RegExp(`\\n\\+${escaped}(?=\\n|$)`, 'g');
  const removePattern = new RegExp(`\\n-${escaped}(?=\\n|$)`, 'g');

  const adds = (diffHistory.match(addPattern) || []).length;
  const removes = (diffHistory.match(removePattern) || []).length;

  return adds + removes;
}

/**
 * Get current file lines from GitHub's DOM
 */
function getCurrentLines() {
  // GitHub's new code view
  const codeLines = document.querySelectorAll('.react-code-lines .react-code-text');
  if (codeLines.length > 0) {
    return Array.from(codeLines).map(el => el.textContent || '');
  }

  // GitHub's classic code view
  const blobCode = document.querySelectorAll('.blob-code-inner');
  if (blobCode.length > 0) {
    return Array.from(blobCode).map(el => el.textContent || '');
  }

  return null;
}

/**
 * Apply churn highlighting to the page
 */
function applyChurn(data) {
  // Try new GitHub UI first
  let rows = document.querySelectorAll('.react-code-lines > div');

  // Fall back to classic UI
  if (rows.length === 0) {
    rows = document.querySelectorAll('tr.js-file-line');
  }

  for (const item of data.lines) {
    if (item.count === 0) continue;

    const row = rows[item.line];
    if (!row) continue;

    const color = getColor(item.normalized);
    row.style.backgroundColor = color;
  }
}

/**
 * Generate heat color (green → red)
 */
function getColor(normalized) {
  const opacity = normalized * 0.3;
  const r = Math.round(255 * Math.min(normalized * 2, 1));
  const g = Math.round(255 * Math.min((1 - normalized) * 2, 1));
  return `rgba(${r}, ${g}, 50, ${opacity})`;
}

/**
 * Show status message
 */
function showStatus(message, duration = 0) {
  let el = document.getElementById('line-churn-status');
  if (!el) {
    el = document.createElement('div');
    el.id = 'line-churn-status';
    document.body.appendChild(el);
  }
  el.textContent = message;
  el.style.display = 'block';

  if (duration > 0) {
    setTimeout(() => { el.style.display = 'none'; }, duration);
  }
}

// Cache helpers using chrome.storage
async function getCache(key) {
  return new Promise(resolve => {
    chrome.storage.local.get(key, result => {
      const item = result[key];
      if (item && Date.now() - item.time < 3600000) { // 1 hour TTL
        resolve(item.data);
      } else {
        resolve(null);
      }
    });
  });
}

async function setCache(key, data) {
  return new Promise(resolve => {
    chrome.storage.local.set({ [key]: { data, time: Date.now() } }, resolve);
  });
}

function sleep(ms) {
  return new Promise(resolve => setTimeout(resolve, ms));
}
