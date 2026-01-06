# Tree of Knowledge (ToK)

A chat application with **text-based conversation branching** powered by Claude via the Claude Agent SDK.

## Key Feature: Text-Level Branching

Unlike traditional chat branching (at message boundaries), Tree of Knowledge allows you to:

1. **Highlight any text** within Claude's response
2. **Ask a follow-up question** about that specific text
3. **Create a new branch** that carries full context up to that point

This matches how we naturally think while reading - specific phrases trigger new questions.

## Architecture

```
tok/
├── backend/           # Python FastAPI + Claude Agent SDK
│   ├── app/
│   │   ├── api/       # REST endpoints
│   │   ├── models/    # SQLAlchemy models
│   │   ├── services/  # Claude SDK integration
│   │   └── core/      # Config, database
│   └── pyproject.toml
│
└── frontend/          # React + TypeScript + Tailwind
    ├── src/
    │   ├── components/
    │   │   ├── Sidebar/    # Tree navigation
    │   │   ├── Chat/       # Messages, streaming
    │   │   └── Input/      # Chat input, branching
    │   ├── hooks/          # useStreaming, useTextSelection
    │   ├── lib/            # API client
    │   └── types/          # TypeScript types
    └── package.json
```

## Features

- **Text-based branching** - Select any text to start a branch
- **Streaming responses** - Real-time response display
- **Thinking mode** - Collapsible extended thinking blocks
- **Tool calling** - Visual display of tool usage (Bash, Read, Write, etc.)
- **File attachments** - Upload images and documents
- **Tree navigation** - "New Tree" and "All Trees" sidebar

## Prerequisites

- Python 3.10+
- Node.js 18+
- Claude Code runtime installed
- Anthropic API key

## Setup

### 1. Install Claude Code Runtime

```bash
curl -fsSL https://claude.ai/install.sh | bash
```

### 2. Backend Setup

```bash
cd tok/backend

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # or .venv\Scripts\activate on Windows

# Install dependencies
pip install -e .

# Set environment variables
export ANTHROPIC_API_KEY=your-api-key

# Run the server
uvicorn app.main:app --reload --port 8000
```

### 3. Frontend Setup

```bash
cd tok/frontend

# Install dependencies
npm install

# Start dev server
npm run dev
```

### 4. Open the App

Visit http://localhost:5173

## API Endpoints

### Trees
- `GET /api/trees` - List all trees
- `POST /api/trees` - Create new tree
- `GET /api/trees/{id}` - Get tree with branches
- `DELETE /api/trees/{id}` - Delete tree

### Branches
- `GET /api/trees/{id}/branches` - List branches
- `POST /api/trees/{id}/branches` - Create branch from text selection

### Messages
- `GET /api/trees/{id}/branches/{bid}/messages` - List messages
- `POST /api/trees/{id}/branches/{bid}/messages` - Send message (streaming SSE)

### Uploads
- `POST /api/upload` - Upload file/image

## Data Model

```
Tree (conversation container)
  └── Branch (conversation thread)
        ├── BranchOrigin (highlighted text + prompt)
        └── Message[]
              ├── ToolCall[] (tool usage records)
              └── Attachment[] (files/images)
```

## Claude Agent SDK Integration

The app uses `ClaudeSDKClient` for multi-turn conversations:

```python
async with ClaudeSDKClient(options=ClaudeAgentOptions(
    allowed_tools=["Read", "Write", "Bash", "WebSearch"],
)) as client:
    await client.query(prompt)
    async for message in client.receive_response():
        # Handle TextBlock, ThinkingBlock, ToolUseBlock, etc.
```

## Tech Stack

**Backend:**
- FastAPI
- Claude Agent SDK
- SQLAlchemy (async)
- SQLite (default) / PostgreSQL

**Frontend:**
- React 18
- TypeScript
- Tailwind CSS
- Vite
- React Router
- Lucide Icons

## License

MIT
