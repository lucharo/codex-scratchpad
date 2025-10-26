# How to Apply the c2c Implementation Patch

## Overview
This patch contains 3 commits implementing the c2c (Claude Code to Claude Code) MCP server with trust-based permission system.

## Patch File
- **File**: `c2c-implementation.patch` (260K)
- **Commits**: 3 commits
- **Branch**: `claude/implement-mcp-server-011CUNizPM4bGfUGt3215M2p`

## Apply the Patch

### Option 1: Apply with git am (Recommended)
This preserves all commit messages and authorship:

```bash
# Make sure you're on the right branch
git checkout claude/implement-mcp-server-011CUNizPM4bGfUGt3215M2p

# Apply the patch
git am < c2c-implementation.patch

# Verify
git log --oneline -5
```

### Option 2: Apply with git apply (Alternative)
This applies changes without creating commits:

```bash
# Apply the patch
git apply c2c-implementation.patch

# Then manually commit
git add .
git commit -m "Add c2c MCP server implementation"
```

## What's Included

### Commit 1: Initial c2c MCP Server (bb7f06d)
- Session management with git worktree isolation
- Process management for Claude Code sub-agents
- 7 MCP tools for session control
- 53 comprehensive tests
- 17 files, 3,067+ lines

### Commit 2: Trust-Based Permission System (adfb4ec)
- Permission request/approval system
- Auto-approve/deny/escalate logic based on policies
- 5 new MCP tools for permission management
- Built-in security policies
- 15 new tests (68 total)
- 5 files, 1,089+ lines

### Commit 3: Rename to c2c (4fe8070)
- Renamed from cc2cc to c2c throughout
- Updated all imports, CLI commands, documentation
- All 68 tests passing
- 20 files changed

## Verify After Applying

```bash
# Check commits
git log --oneline -3

# Should show:
# 4fe8070 Rename project from cc2cc to c2c
# adfb4ec Add trust-based permission system for sub-agent control
# bb7f06d Add cc2cc MCP server for Claude Code session management

# Run tests
cd c2c
uv sync
uv run pytest -v

# Should see: 68 passed
```

## Push to Remote

```bash
git push -u origin claude/implement-mcp-server-011CUNizPM4bGfUGt3215M2p
```

## Troubleshooting

### If patch fails to apply
```bash
# Check current state
git status

# If there are conflicts, you can:
# 1. Manually resolve conflicts
# 2. Or try applying in 3-way merge mode
git am -3 < c2c-implementation.patch
```

### If you need to undo
```bash
# Undo the last applied patch
git am --abort

# Or reset to before applying
git reset --hard HEAD~3
```

## Summary

The c2c MCP server enables:
- ✅ Claude Code to delegate to sub-agents in isolated worktrees
- ✅ Trust-based permission system (auto-approve safe, deny dangerous, escalate risky)
- ✅ 68 comprehensive tests, all passing
- ✅ Production-ready, modular, type-safe architecture

This is the differentiating feature: **agents are all about trust**, and c2c makes that trust explicit through a sophisticated permission policy system.
