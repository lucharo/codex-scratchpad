# C2C-Dev MCP Server

Minimal Claude Code to Claude Code bidirectional agent communication.

## Install

```bash
git clone <repo-url>
cd c2c
uv sync
claude mcp add c2c-dev -- uv run python c2c_dev.py
```

## Tools

- `create_conversation(task)` - Create new agent conversation
- `send_message(session_id, message)` - Send message to agent
- `get_conversation(session_id)` - Get conversation history
- `end_conversation(session_id)` - End conversation
- `get_active_sessions()` - List active sessions
- `clear_all_sessions()` - End all conversations

Conversations saved in JSONL at `~/.claude/projects/c2c-agent-conversations/`
