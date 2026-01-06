/**
 * Worn Out Lines - GitHub Chrome Extension
 * Highlights lines based on how frequently they've been modified.
 *
 * Public repos only. Rate limit: 60 requests/hour.
 */

(async function() {
  'use strict';

  const match = window.location.pathname.match(/^\/([^/]+)\/([^/]+)\/blob\/([^/]+)\/(.+)$/);
  if (!match) return;

  const [, owner, repo, , filePath] = match;
  const cacheKey = `churn:${owner}/${repo}/${filePath}`;

  // Check cache (1 hour TTL)
  const cached = await getCache(cacheKey);
  if (cached) {
    applyChurn(cached);
    return;
  }

  showStatus('Analyzing churn...');

  try {
    const churn = await analyzeChurn(owner, repo, filePath);
    if (churn) {
      await setCache(cacheKey, churn);
      applyChurn(churn);
      showStatus(`Max churn: ${churn.maxChurn}`, 3000);
    } else {
      showStatus('No history found', 3000);
    }
  } catch (err) {
    console.error('Worn Out Lines:', err);
    showStatus(err.message, 5000);
  }
})();

async function analyzeChurn(owner, repo, filePath) {
  const commitsUrl = `https://api.github.com/repos/${owner}/${repo}/commits?path=${encodeURIComponent(filePath)}&per_page=100`;
  const commitsRes = await fetch(commitsUrl);

  if (commitsRes.status === 403) throw new Error('Rate limited (60/hr)');
  if (commitsRes.status === 404) throw new Error('Private repo or not found');
  if (!commitsRes.ok) throw new Error(`API error: ${commitsRes.status}`);

  const commits = await commitsRes.json();
  if (!commits.length) return null;

  // Fetch diffs (limit to 30 commits)
  const diffs = [];
  for (let i = 0; i < Math.min(commits.length, 30); i++) {
    const res = await fetch(
      `https://api.github.com/repos/${owner}/${repo}/commits/${commits[i].sha}`,
      { headers: { Accept: 'application/vnd.github.v3.diff' } }
    );
    if (res.ok) diffs.push(await res.text());
    await new Promise(r => setTimeout(r, 50));
  }

  const diffHistory = diffs.join('\n');
  const lines = getLines();
  if (!lines) return null;

  // Count churn per line
  let maxChurn = 0;
  const data = lines.map((content, i) => {
    const trimmed = content.trim();
    const count = trimmed.length >= 3 ? countInDiffs(trimmed, diffHistory) : 0;
    maxChurn = Math.max(maxChurn, count);
    return { line: i, count, normalized: 0 };
  });

  if (maxChurn > 0) data.forEach(d => d.normalized = d.count / maxChurn);

  return { lines: data, maxChurn };
}

function countInDiffs(line, diffs) {
  const esc = line.replace(/[.*+?^${}()|[\]\\]/g, '\\$&');
  const adds = (diffs.match(new RegExp(`\\n\\+${esc}(?=\\n|$)`, 'g')) || []).length;
  const dels = (diffs.match(new RegExp(`\\n-${esc}(?=\\n|$)`, 'g')) || []).length;
  return adds + dels;
}

function getLines() {
  let els = document.querySelectorAll('.react-code-lines .react-code-text');
  if (els.length) return Array.from(els).map(e => e.textContent || '');
  els = document.querySelectorAll('.blob-code-inner');
  if (els.length) return Array.from(els).map(e => e.textContent || '');
  return null;
}

function applyChurn({ lines }) {
  let rows = document.querySelectorAll('.react-code-lines > div');
  if (!rows.length) rows = document.querySelectorAll('tr.js-file-line');

  for (const { line, count, normalized } of lines) {
    if (!count || !rows[line]) continue;
    const r = Math.round(255 * Math.min(normalized * 2, 1));
    const g = Math.round(255 * Math.min((1 - normalized) * 2, 1));
    rows[line].style.backgroundColor = `rgba(${r}, ${g}, 50, ${normalized * 0.3})`;
  }
}

function showStatus(msg, duration = 0) {
  let el = document.getElementById('wol-status');
  if (!el) {
    el = document.createElement('div');
    el.id = 'wol-status';
    el.style.cssText = 'position:fixed;bottom:20px;right:20px;background:#24292e;color:#fff;padding:8px 16px;border-radius:6px;z-index:9999;font:14px system-ui';
    document.body.appendChild(el);
  }
  el.textContent = msg;
  el.style.display = 'block';
  if (duration) setTimeout(() => el.style.display = 'none', duration);
}

const getCache = key => new Promise(r =>
  chrome.storage.local.get(key, res => {
    const item = res[key];
    r(item && Date.now() - item.time < 3600000 ? item.data : null);
  })
);

const setCache = (key, data) => new Promise(r =>
  chrome.storage.local.set({ [key]: { data, time: Date.now() } }, r)
);
