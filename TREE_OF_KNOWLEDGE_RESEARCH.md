# Tree of Knowledge - Research & Architecture Document

## Executive Summary

**Tree of Knowledge** is a chat application with text-based conversation branching powered by Claude via the Claude Agent SDK. Unlike traditional branching (at message boundaries), users can highlight **any section of text** within a response and start a new conversation branch from that point, carrying full context.

---

## Part 1: Claude Agent SDK for Python

### Installation

```bash
# 1. Install Claude Code runtime
curl -fsSL https://claude.ai/install.sh | bash

# 2. Install Python SDK
pip install claude-agent-sdk

# 3. Set API credentials
export ANTHROPIC_API_KEY=your-api-key
```

### Two Core Interaction Patterns

| Pattern | Use Case | Session Handling |
|---------|----------|------------------|
| `query()` | One-off tasks | Fresh session each call |
| `ClaudeSDKClient` | Multi-turn chat | Maintains conversation state |

### ClaudeSDKClient for Chat Applications

```python
from claude_agent_sdk import ClaudeSDKClient, ClaudeAgentOptions, AssistantMessage, TextBlock, ThinkingBlock, ToolUseBlock, ToolResultBlock

async with ClaudeSDKClient(options=ClaudeAgentOptions()) as client:
    await client.connect()

    # First message
    await client.query("Hello, let's discuss Python")
    async for message in client.receive_response():
        if isinstance(message, AssistantMessage):
            for block in message.content:
                if isinstance(block, TextBlock):
                    print(block.text)

    # Follow-up (maintains context)
    await client.query("What about async patterns?")
    async for message in client.receive_response():
        # Claude remembers previous context
        pass
```

### Message Types Reference

| Type | Purpose | Key Fields |
|------|---------|------------|
| `AssistantMessage` | Claude's response container | `content: list[ContentBlock]`, `model: str` |
| `TextBlock` | Plain text response | `text: str` |
| `ThinkingBlock` | Extended thinking content | `thinking: str`, `signature: str` |
| `ToolUseBlock` | Tool invocation | `id: str`, `name: str`, `input: dict` |
| `ToolResultBlock` | Tool execution result | `tool_use_id: str`, `content: str`, `is_error: bool` |
| `ResultMessage` | Session completion | `session_id: str`, `duration_ms: int`, `total_cost_usd: float` |

### Streaming Pattern for Chat UI

```python
async def stream_response(client: ClaudeSDKClient, prompt: str):
    """Stream response with real-time updates for UI"""
    await client.query(prompt)

    async for message in client.receive_response():
        if isinstance(message, AssistantMessage):
            for block in message.content:
                if isinstance(block, ThinkingBlock):
                    yield {"type": "thinking", "content": block.thinking}
                elif isinstance(block, TextBlock):
                    yield {"type": "text", "content": block.text}
                elif isinstance(block, ToolUseBlock):
                    yield {"type": "tool_use", "name": block.name, "input": block.input}
                elif isinstance(block, ToolResultBlock):
                    yield {"type": "tool_result", "content": block.content, "error": block.is_error}
        elif isinstance(message, ResultMessage):
            yield {"type": "complete", "session_id": message.session_id}
```

### Built-in Tools Available

| Tool | Description |
|------|-------------|
| `Read` | Read files (text, images, PDFs) |
| `Write` | Create new files |
| `Edit` | Modify existing files |
| `Bash` | Execute shell commands |
| `Glob` | Find files by pattern |
| `Grep` | Search file contents |
| `WebSearch` | Internet search |
| `WebFetch` | Fetch and parse URLs |
| `Task` | Delegate to subagents |

### Session Management for Branching

```python
# Capture session state for branching
session_id = None

async for message in client.receive_messages():
    if isinstance(message, ResultMessage):
        session_id = message.session_id
        break

# Resume from captured session (potential branching point)
async for message in query(
    prompt="New branch query",
    options=ClaudeAgentOptions(resume=session_id)
):
    pass
```

### ClaudeAgentOptions Configuration

```python
ClaudeAgentOptions(
    allowed_tools=["Read", "Write", "Bash", "WebSearch"],
    permission_mode="acceptEdits",  # or "plan", "bypassPermissions"
    system_prompt="Custom system instructions",
    max_turns=50,
    mcp_servers={"name": McpServerConfig(...)},
    hooks={"PreToolUse": [...], "PostToolUse": [...]},
    cwd="/path/to/workspace"
)
```

---

## Part 2: Similar Projects Analysis

### 1. ChatGPT Branching (OpenAI - Sept 2025)

**How it works:** Hover over any message → click branch → new chat window opens from that point.

**Key features:**
- Context preservation (original thread unchanged)
- Visual labels showing fork origin
- Full conversation continuity in branches

**Limitation:** Branching at message level only, not text selection.

**Source:** [OpenAI ChatGPT Branching](https://wphtaccess.com/2026/01/02/openai-releases-branching-chats-in-chatgpt-a-complete-overview/)

---

### 2. tldraw/branching-chat-template

**GitHub:** [tldraw/branching-chat-template](https://github.com/tldraw/branching-chat-template)

**Tech Stack:**
- Frontend: React + TypeScript + tldraw (visual canvas)
- Backend: Cloudflare Workers + Durable Objects
- AI: Vercel AI SDK (provider-agnostic)

**Architecture:**
- Infinite canvas with draggable message nodes
- Port-based connections between nodes
- Context built by traversing connected message chain
- Real-time streaming responses

**Key Pattern:** Visual node-based conversation graph.

---

### 3. GitChat

**GitHub:** [DrustZ/GitChat](https://github.com/DrustZ/GitChat)

**Concept:** Git-like version control for conversations.
- Branch conversations
- Merge conversation threads
- Modify/rewire chat history
- Flowchart-like message structure

**Innovation:** Treating messages as commits with branching/merging capabilities.

---

### 4. oMyTree

**GitHub:** [isbeingto/oMyTree](https://github.com/isbeingto/oMyTree)

**Features:**
- AI chats → visual conversation tree
- Each message = a node
- Each follow-up = new branch
- Jump between branches
- See thinking evolution visually

---

### 5. ChatTree (aadityaubhat)

**GitHub:** [aadityaubhat/ChatTree](https://github.com/aadityaubhat/ChatTree)

**Concept:** Non-linear conversations where you can branch from any user message.
- Tree structure instead of linear thread
- POC/experimental

---

### 6. Vercel AI SDK - useBranchingChat Hook

**GitHub PR:** [vercel/ai#5085](https://github.com/vercel/ai/pull/5085)

**Features:**
- Tree-structured conversations
- Branching, editing, retrying messages
- ChatGPT/Claude-like UX
- Library-level support

---

### 7. LibreChat (Full-Featured Reference)

**GitHub:** [danny-avila/LibreChat](https://github.com/danny-avila/LibreChat) (20K+ stars)

**Relevant Features:**
- ✅ **Fork Messages & Conversations** for context control
- ✅ **Edit, Resubmit, Continue Messages** with branching
- ✅ File uploads (images, documents)
- ✅ Code Interpreter (sandboxed execution)
- ✅ MCP integration for tools
- ✅ Streaming responses
- ✅ Multi-model support
- ✅ Message search

**Tech Stack:** TypeScript (70%), Node.js backend, Docker-ready

---

## Part 3: Feature Comparison Matrix

| Feature | Tree of Knowledge | ChatGPT | LibreChat | tldraw Template |
|---------|-------------------|---------|-----------|-----------------|
| Text-level branching | ✅ **Unique** | ❌ | ❌ | ❌ |
| Message-level branching | ✅ | ✅ | ✅ | ✅ |
| Visual tree view | Planned | ❌ | ❌ | ✅ |
| Context preservation | ✅ | ✅ | ✅ | ✅ |
| Highlighted text indicator | ✅ **Unique** | ❌ | ❌ | ❌ |
| File attachments | ✅ | ✅ | ✅ | ❌ |
| Image support | ✅ | ✅ | ✅ | ❌ |
| Tool calling visualization | ✅ | ❌ | ✅ | ❌ |
| Thinking mode (collapsible) | ✅ | ❌ | ❌ | ❌ |
| Streaming | ✅ | ✅ | ✅ | ✅ |
| Claude Agent SDK | ✅ | ❌ | ❌ | ❌ |

---

## Part 4: Proposed Architecture

### System Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                        Frontend (React/Next.js)              │
├─────────────────────────────────────────────────────────────┤
│  Sidebar          │  Chat Area                              │
│  ┌─────────────┐  │  ┌─────────────────────────────────────┐│
│  │ + New Tree  │  │  │ Message Stream                      ││
│  │             │  │  │  ┌─────────────────────────────────┐││
│  │ All Trees   │  │  │  │ Assistant Response              │││
│  │  ├─ Tree 1  │  │  │  │ [selectable text for branching] │││
│  │  │  ├─ b1   │  │  │  └─────────────────────────────────┘││
│  │  │  └─ b2   │  │  │                                     ││
│  │  └─ Tree 2  │  │  │  ┌─────────────────────────────────┐││
│  │             │  │  │  │ Tool Use: Read file.py          │││
│  └─────────────┘  │  │  │ [collapsible details]           │││
│                   │  │  └─────────────────────────────────┘││
│                   │  │                                     ││
│                   │  │  ┌─────────────────────────────────┐││
│                   │  │  │ 💭 Thinking... [collapsed]      │││
│                   │  │  └─────────────────────────────────┘││
│                   │  └─────────────────────────────────────┘│
│                   │  ┌─────────────────────────────────────┐│
│                   │  │ Input: [message] [📎] [🖼️] [Send]  ││
│                   │  └─────────────────────────────────────┘│
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                    Backend (Python FastAPI)                  │
├─────────────────────────────────────────────────────────────┤
│  Endpoints:                                                  │
│  - POST /trees                    Create new tree            │
│  - GET  /trees                    List all trees             │
│  - GET  /trees/{id}               Get tree with branches     │
│  - POST /trees/{id}/messages      Send message (streaming)   │
│  - POST /trees/{id}/branch        Create branch from text    │
│  - POST /upload                   Upload files/images        │
│                                                              │
│  Claude Agent SDK Integration:                               │
│  ┌─────────────────────────────────────────────────────────┐│
│  │ ClaudeSDKClient                                         ││
│  │  - Session management per conversation                  ││
│  │  - Streaming message handling                           ││
│  │  - Tool execution (Read, Write, Bash, etc.)            ││
│  │  - Thinking block processing                            ││
│  └─────────────────────────────────────────────────────────┘│
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                       Database (SQLite/PostgreSQL)           │
├─────────────────────────────────────────────────────────────┤
│  Tables:                                                     │
│  - trees (id, name, created_at, updated_at)                 │
│  - branches (id, tree_id, parent_branch_id, name, ...)      │
│  - messages (id, branch_id, role, content, position, ...)   │
│  - branch_origins (branch_id, source_message_id,            │
│                    highlight_start, highlight_end,           │
│                    highlighted_text)                         │
│  - attachments (id, message_id, file_path, file_type)       │
└─────────────────────────────────────────────────────────────┘
```

### Data Model for Text-Based Branching

```python
# Key innovation: storing exact text selection for branching

class BranchOrigin(BaseModel):
    branch_id: str
    source_message_id: str
    source_branch_id: str
    highlight_start: int      # Character offset start
    highlight_end: int        # Character offset end
    highlighted_text: str     # The actual highlighted text
    user_prompt: str          # User's question about highlighted text

class Message(BaseModel):
    id: str
    branch_id: str
    role: Literal["user", "assistant"]
    content: str
    thinking: Optional[str]   # Collapsible thinking content
    tool_calls: List[ToolCall]
    position: int             # Order in conversation
    created_at: datetime
    # For assistant messages, track if text was highlighted for branching
    branch_children: List[str]  # IDs of branches created from this message

class Branch(BaseModel):
    id: str
    tree_id: str
    parent_branch_id: Optional[str]  # None for root
    name: str
    origin: Optional[BranchOrigin]   # How this branch was created
    messages: List[Message]
    created_at: datetime
```

### Text Selection Branching Flow

```
1. User reads assistant response in Branch A
2. User highlights text: "The key insight is that async/await..."
3. UI shows "Branch from selection" button
4. User clicks, enters prompt: "Can you explain this more deeply?"
5. Backend:
   a. Creates new Branch B with parent = Branch A
   b. Stores BranchOrigin with highlight positions and text
   c. Builds context: all messages in Branch A up to highlight point
   d. Sends to Claude:
      - Full context
      - Special marker: "USER HIGHLIGHTED: 'The key insight is that async/await...'"
      - User's prompt
6. Frontend:
   a. Shows Branch B in tree view
   b. In Branch B, first shows highlighted text in special block
   c. Streams Claude's response
```

### Frontend Components

```
src/
├── components/
│   ├── Sidebar/
│   │   ├── NewTreeButton.tsx
│   │   ├── TreeList.tsx
│   │   └── BranchTree.tsx          # Collapsible tree view
│   ├── Chat/
│   │   ├── MessageList.tsx
│   │   ├── Message.tsx
│   │   ├── SelectableText.tsx      # Handles text selection
│   │   ├── BranchIndicator.tsx     # Shows "branched from X"
│   │   ├── ThinkingBlock.tsx       # Collapsible thinking
│   │   ├── ToolCallBlock.tsx       # Tool use visualization
│   │   └── StreamingResponse.tsx
│   ├── Input/
│   │   ├── ChatInput.tsx
│   │   ├── FileUpload.tsx
│   │   └── BranchPrompt.tsx        # Prompt for new branch
│   └── shared/
│       └── ...
├── hooks/
│   ├── useTextSelection.ts         # Track text selection
│   ├── useStreaming.ts             # SSE/WebSocket streaming
│   └── useBranching.ts             # Branch operations
└── ...
```

### Backend API with Streaming

```python
from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from claude_agent_sdk import ClaudeSDKClient, ClaudeAgentOptions

app = FastAPI()

@app.post("/trees/{tree_id}/messages")
async def send_message(tree_id: str, request: MessageRequest):
    """Stream response from Claude"""

    async def generate():
        async with ClaudeSDKClient(options=ClaudeAgentOptions(
            allowed_tools=["Read", "Write", "Bash", "WebSearch"],
        )) as client:

            # Build context from conversation history
            context = await build_context(tree_id, request.branch_id)

            await client.query(context + request.message)

            async for message in client.receive_response():
                if isinstance(message, AssistantMessage):
                    for block in message.content:
                        if isinstance(block, ThinkingBlock):
                            yield f"data: {json.dumps({'type': 'thinking', 'content': block.thinking})}\n\n"
                        elif isinstance(block, TextBlock):
                            yield f"data: {json.dumps({'type': 'text', 'content': block.text})}\n\n"
                        elif isinstance(block, ToolUseBlock):
                            yield f"data: {json.dumps({'type': 'tool_use', 'name': block.name, 'input': block.input})}\n\n"

    return StreamingResponse(generate(), media_type="text/event-stream")

@app.post("/trees/{tree_id}/branch")
async def create_branch(tree_id: str, request: BranchRequest):
    """Create a new branch from highlighted text"""

    # Store the branch origin
    branch = await create_branch_record(
        tree_id=tree_id,
        parent_branch_id=request.source_branch_id,
        origin=BranchOrigin(
            source_message_id=request.source_message_id,
            source_branch_id=request.source_branch_id,
            highlight_start=request.highlight_start,
            highlight_end=request.highlight_end,
            highlighted_text=request.highlighted_text,
            user_prompt=request.prompt
        )
    )

    # Build special context for branched conversation
    context = await build_branch_context(branch)

    # Return branch info, then client calls /messages endpoint
    return {"branch_id": branch.id, "tree_id": tree_id}
```

---

## Part 5: Unique Differentiators

### 1. Text-Level Branching (Novel Feature)
No existing tool allows branching from highlighted text within a message. This matches natural reading/thinking patterns where specific phrases trigger questions.

### 2. Visual Branch Origin
When viewing a branch, clearly show:
- The highlighted text that started it
- Visual connection to parent conversation
- Breadcrumb navigation between branches

### 3. Claude Agent SDK Native
Full tool calling, thinking mode, and streaming - not just basic chat API.

### 4. Tree Metaphor
"New Tree" and "All Trees" instead of chat terminology - reinforces the branching mental model.

---

## Part 6: Implementation Priorities

### Phase 1: Core Chat (MVP)
- [ ] Basic chat interface with streaming
- [ ] Message history persistence
- [ ] Thinking mode display (collapsible)
- [ ] Tool calling visualization

### Phase 2: Text Branching (Unique Value)
- [ ] Text selection detection
- [ ] Branch creation from selection
- [ ] Branch origin display
- [ ] Tree view sidebar

### Phase 3: Full Features
- [ ] File/image attachments
- [ ] Multiple trees management
- [ ] Branch navigation & merging
- [ ] Export conversations

---

## Sources

### Claude Agent SDK
- [Agent SDK Overview](https://platform.claude.com/docs/en/agent-sdk/overview)
- [Python API Reference](https://platform.claude.com/docs/en/agent-sdk/python)
- [GitHub: claude-agent-sdk-python](https://github.com/anthropics/claude-agent-sdk-python)

### Similar Projects
- [tldraw/branching-chat-template](https://github.com/tldraw/branching-chat-template) - Visual branching with tldraw
- [DrustZ/GitChat](https://github.com/DrustZ/GitChat) - Git-like chat branching
- [isbeingto/oMyTree](https://github.com/isbeingto/oMyTree) - Conversation tree visualization
- [aadityaubhat/ChatTree](https://github.com/aadityaubhat/ChatTree) - Non-linear conversations
- [Vercel AI useBranchingChat](https://github.com/vercel/ai/pull/5085) - Library-level branching hook
- [LibreChat](https://github.com/danny-avila/LibreChat) - Full-featured open source chat

### Industry Context
- [ChatGPT Branching Feature](https://medium.com/@CherryZhouTech/chatgpt-launches-branched-chats-effortless-multi-threaded-conversations-d188b90bd78b)
- [UX of Branching Conversations](https://medium.com/@nikivergis/ai-chat-tools-dont-match-how-we-actually-think-exploring-the-ux-of-branching-conversations-259107496afb)
- [TypingMind](https://www.typingmind.com/) - Premium chat UI reference
- [LibreChat 2025 Roadmap](https://www.librechat.ai/blog/2025-02-20_2025_roadmap)
