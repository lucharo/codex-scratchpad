/**
 * Worn Out Lines - GitHub Chrome Extension
 * Highlights lines based on how frequently they've been modified.
 *
 * Rate limits: 60/hr without token, 5000/hr with token
 * Set token via: chrome.storage.sync.set({ githubToken: 'your_token' })
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
    const token = await getToken();
    const churn = await analyzeChurn(owner, repo, filePath, token);
    if (churn) {
      await setCache(cacheKey, churn);
      applyChurn(churn);
      showStatus(`Max churn: ${churn.maxChurn}`, 3000);
    }
  } catch (err) {
    console.error('Worn Out Lines error:', err);
    showStatus('Error: ' + err.message, 5000);
  }
})();

async function analyzeChurn(owner, repo, filePath, token) {
  const headers = token ? { Authorization: `token ${token}` } : {};

  // Get commits for this file
  const commitsUrl = `https://api.github.com/repos/${owner}/${repo}/commits?path=${encodeURIComponent(filePath)}&per_page=100`;
  const commitsRes = await fetch(commitsUrl, { headers });

  if (!commitsRes.ok) {
    if (commitsRes.status === 403) throw new Error('Rate limited - add GitHub token');
    throw new Error(`API error: ${commitsRes.status}`);
  }

  const commits = await commitsRes.json();
  if (commits.length === 0) return null;

  // Fetch diffs (limit to 30 commits)
  const diffs = [];
  for (let i = 0; i < Math.min(commits.length, 30); i++) {
    const diffRes = await fetch(
      `https://api.github.com/repos/${owner}/${repo}/commits/${commits[i].sha}`,
      { headers: { ...headers, Accept: 'application/vnd.github.v3.diff' } }
    );
    if (diffRes.ok) diffs.push(await diffRes.text());
    await sleep(50);
  }

  const diffHistory = diffs.join('\n');
  const lines = getCurrentLines();
  if (!lines) return null;

  // Count churn
  const churnData = [];
  let maxChurn = 0;

  for (let i = 0; i < lines.length; i++) {
    const content = lines[i].trim();
    const count = content.length >= 3 ? countInDiffs(content, diffHistory) : 0;
    maxChurn = Math.max(maxChurn, count);
    churnData.push({ line: i, count, normalized: 0 });
  }

  if (maxChurn > 0) {
    for (const d of churnData) d.normalized = d.count / maxChurn;
  }

  return { lines: churnData, maxChurn };
}

function countInDiffs(line, diffs) {
  if (!line) return 0;
  const esc = line.replace(/[.*+?^${}()|[\]\\]/g, '\\$&');
  const adds = (diffs.match(new RegExp(`\\n\\+${esc}(?=\\n|$)`, 'g')) || []).length;
  const dels = (diffs.match(new RegExp(`\\n-${esc}(?=\\n|$)`, 'g')) || []).length;
  return adds + dels;
}

function getCurrentLines() {
  // New GitHub UI
  let els = document.querySelectorAll('.react-code-lines .react-code-text');
  if (els.length) return Array.from(els).map(e => e.textContent || '');
  // Classic UI
  els = document.querySelectorAll('.blob-code-inner');
  if (els.length) return Array.from(els).map(e => e.textContent || '');
  return null;
}

function applyChurn(data) {
  let rows = document.querySelectorAll('.react-code-lines > div');
  if (!rows.length) rows = document.querySelectorAll('tr.js-file-line');

  for (const { line, count, normalized } of data.lines) {
    if (count === 0 || !rows[line]) continue;
    const r = Math.round(255 * Math.min(normalized * 2, 1));
    const g = Math.round(255 * Math.min((1 - normalized) * 2, 1));
    rows[line].style.backgroundColor = `rgba(${r}, ${g}, 50, ${normalized * 0.3})`;
  }
}

function showStatus(msg, duration = 0) {
  let el = document.getElementById('worn-out-lines-status');
  if (!el) {
    el = document.createElement('div');
    el.id = 'worn-out-lines-status';
    el.style.cssText = 'position:fixed;bottom:20px;right:20px;background:#24292e;color:#fff;padding:8px 16px;border-radius:6px;font-size:14px;z-index:9999';
    document.body.appendChild(el);
  }
  el.textContent = msg;
  el.style.display = 'block';
  if (duration > 0) setTimeout(() => el.style.display = 'none', duration);
}

async function getToken() {
  return new Promise(r => chrome.storage.sync.get('githubToken', res => r(res.githubToken)));
}

async function getCache(key) {
  return new Promise(r => {
    chrome.storage.local.get(key, res => {
      const item = res[key];
      r(item && Date.now() - item.time < 3600000 ? item.data : null);
    });
  });
}

async function setCache(key, data) {
  return new Promise(r => chrome.storage.local.set({ [key]: { data, time: Date.now() } }, r));
}

function sleep(ms) { return new Promise(r => setTimeout(r, ms)); }
