#!/usr/bin/env python3
"""
C2C-Dev: Minimal Working MVP for Bidirectional Agent Communication
A thin MCP wrapper around the Agent SDK demonstrating your pipe dream.
"""

import asyncio
import uuid
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List
from dataclasses import dataclass
from claude_agent_sdk import ClaudeSDKClient, ClaudeAgentOptions, HookMatcher
from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import Tool
import sys
import os

# Add the src directory to the path to import conversation_storage
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))
from c2c.conversation_storage import ConversationStorage

# Global session storage - super simple!
active_sessions: Dict[str, Dict[str, Any]] = {}
conversation_history: Dict[str, List[Dict[str, str]]] = {}

# Initialize persistent storage
conversation_storage = ConversationStorage()

# Message timing controls
message_timestamps: Dict[str, float] = {}
MIN_MESSAGE_INTERVAL = 2.0  # Minimum seconds between messages

# Enhanced conversation turn structure
@dataclass
class ConversationTurn:
    """A complete conversation turn with user input, agent response, and tool usage"""
    turn_id: str
    timestamp: str
    user_input: str
    agent_response: str
    tools_used: List[Dict[str, Any]]
    session_id: str

# Turn storage for active sessions
active_turns: Dict[str, ConversationTurn] = {}

# Global tool log buffer for current session
current_tool_logs: List[str] = []

async def log_tool_use(input_data: dict, tool_use_id: str, context: dict) -> dict:
    """Log all tool usage for auditing."""
    tool_name = input_data.get('tool_name', 'unknown')
    tool_input = input_data.get('tool_input', {})

    # Create XML-wrapped log entry
    timestamp = datetime.now().isoformat()

    if tool_name == 'Write':
        file_path = tool_input.get('file_path', 'unknown')
        log_entry = f'<tool name="{tool_name}" action="write" file="{file_path}" timestamp="{timestamp}">'
        log_entry += f'Tool: {tool_name} - Writing file: {file_path}'
        log_entry += f'</tool>'
    elif tool_name == 'Read':
        file_path = tool_input.get('file_path', 'unknown')
        log_entry = f'<tool name="{tool_name}" action="read" file="{file_path}" timestamp="{timestamp}">'
        log_entry += f'Tool: {tool_name} - Reading file: {file_path}'
        log_entry += f'</tool>'
    elif tool_name == 'Edit':
        file_path = tool_input.get('file_path', 'unknown')
        log_entry = f'<tool name="{tool_name}" action="edit" file="{file_path}" timestamp="{timestamp}">'
        log_entry += f'Tool: {tool_name} - Editing file: {file_path}'
        log_entry += f'</tool>'
    elif tool_name == 'Bash':
        command = tool_input.get('command', 'unknown')
        # Escape special characters in XML
        safe_command = command.replace('"', '&quot;').replace('<', '&lt;').replace('>', '&gt;').replace('&', '&amp;')
        log_entry = f'<tool name="{tool_name}" action="bash" command="{safe_command}" timestamp="{timestamp}">'
        log_entry += f'Tool: {tool_name} - Running command: {command}'
        log_entry += f'</tool>'
    else:
        log_entry = f'<tool name="{tool_name}" action="unknown" timestamp="{timestamp}">'
        log_entry += f'Tool: {tool_name}'
        log_entry += f'</tool>'

    current_tool_logs.append(log_entry)
    print(log_entry)

    return {}

def _build_conversation_context(session_id: str, new_message: str) -> str:
    """Build conversation context string for Agent SDK to prevent repeated responses"""
    context_parts = []

    # Add conversation history to provide context
    if session_id in conversation_history:
        history = conversation_history[session_id]

        # Include last few exchanges for context (limit to prevent token overflow)
        recent_messages = history[-6:] if len(history) > 6 else history

        for msg in recent_messages:
            role = "User" if msg["role"] == "user" else "Assistant"
            context_parts.append(f"{role}: {msg['message']}")

    # Add the new message
    context_parts.append(f"User: {new_message}")

    # Add instruction to continue conversation naturally
    context_parts.append("\nPlease continue the conversation naturally, building upon our previous discussion. Avoid repeating information you've already shared.")

    return "\n\n".join(context_parts)

# MCP Server Setup
app = Server("c2c-dev")

@app.list_tools()
async def list_tools():
    return [
        Tool(
            name="create_conversation",
            description="Create a new bidirectional agent conversation",
            inputSchema={
                "type": "object",
                "properties": {
                    "task": {
                        "type": "string",
                        "description": "Initial task to start the conversation"
                    }
                },
                "required": ["task"]
            }
        ),
        Tool(
            name="send_message",
            description="Send a message to an active conversation",
            inputSchema={
                "type": "object",
                "properties": {
                    "session_id": {
                        "type": "string",
                        "description": "Session ID of the conversation"
                    },
                    "message": {
                        "type": "string",
                        "description": "Message to send"
                    }
                },
                "required": ["session_id", "message"]
            }
        ),
        Tool(
            name="get_conversation",
            description="Get the conversation history",
            inputSchema={
                "type": "object",
                "properties": {
                    "session_id": {
                        "type": "string",
                        "description": "Session ID of the conversation"
                    }
                },
                "required": ["session_id"]
            }
        ),
        Tool(
            name="end_conversation",
            description="End an active conversation",
            inputSchema={
                "type": "object",
                "properties": {
                    "session_id": {
                        "type": "string",
                        "description": "Session ID of the conversation"
                    }
                },
                "required": ["session_id"]
            }
        ),
        Tool(
            name="list_conversations",
            description="List all stored conversations",
            inputSchema={
                "type": "object",
                "properties": {},
                "required": []
            }
        ),
        Tool(
            name="search_conversations",
            description="Search stored conversations by content",
            inputSchema={
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Search query"
                    }
                },
                "required": ["query"]
            }
        )
    ]

async def create_conversation(task: str) -> str:
    """Core Function 1: Create new bidirectional conversation"""
    session_id = f"c2c-{uuid.uuid4().hex[:8]}"

    try:
        # Add timeout protection
        return await asyncio.wait_for(_create_conversation_internal(session_id, task), timeout=30)

    except asyncio.TimeoutError:
        raise Exception(f"Conversation creation timed out after 30 seconds")
    except Exception as e:
        raise Exception(f"Failed to create conversation: {str(e)}")

async def _create_conversation_internal(session_id: str, task: str) -> str:
    """Internal conversation creation with proper error handling"""
    try:
        # Create Agent SDK client
        options = ClaudeAgentOptions(
            cwd=Path.cwd(),
            continue_conversation=True,
            permission_mode="acceptEdits",
            allowed_tools=[
                "Read",      # Read file contents
                "Write",     # Create new files and write content
                "Edit",      # Modify existing files
                "Bash",      # Execute bash commands
                "Glob",      # Find files by pattern
                "Grep"       # Search within file contents
            ],
            hooks={
                'PreToolUse': [
                    HookMatcher(hooks=[log_tool_use])
                ]
            }
        )

        client = ClaudeSDKClient(options)
        await client.connect()

        # Store session
        active_sessions[session_id] = {
            "client": client,
            "created_at": datetime.now(),
            "status": "active",
            "tool_logs": []
        }

        conversation_history[session_id] = []

        # Send initial task (async - don't wait for response)
        await client.query(task)

        # Store initial user message in conversation history
        conversation_history[session_id].extend([
            {"role": "user", "message": task, "timestamp": datetime.now().isoformat()}
        ])

        # Save to persistent storage with initial message
        try:
            conversation_id = conversation_storage.save_conversation(
                session_id=session_id,
                messages=conversation_history[session_id],
                initial_task=task,
                status="active"
            )
            # Store conversation_id in session data for later reference
            active_sessions[session_id]["conversation_id"] = conversation_id
        except Exception as storage_error:
            # Log storage error but don't fail the conversation creation
            print(f"Warning: Failed to save conversation to storage: {storage_error}")

        # Return session ID immediately (async behavior)
        print(f"[DEBUG] Async conversation created: {session_id}")
        return session_id

    except Exception as e:
        # Clean up on error
        if session_id in active_sessions:
            try:
                client = active_sessions[session_id]["client"]
                await client.disconnect()
            except:
                pass
            del active_sessions[session_id]
        if session_id in conversation_history:
            del conversation_history[session_id]
        raise

async def send_message(session_id: str, message: str) -> str:
    """Core Function 2: Send message to active conversation"""
    if session_id not in active_sessions:
        raise Exception(f"Session not found: {session_id}")

    try:
        # Rate limiting: Check minimum interval between messages
        current_time = asyncio.get_event_loop().time()
        last_message_time = message_timestamps.get(session_id, 0)

        if current_time - last_message_time < MIN_MESSAGE_INTERVAL:
            wait_time = MIN_MESSAGE_INTERVAL - (current_time - last_message_time)
            await asyncio.sleep(wait_time)

        # Add timeout protection
        result = await asyncio.wait_for(_send_message_internal(session_id, message), timeout=30)

        # Update timestamp
        message_timestamps[session_id] = asyncio.get_event_loop().time()

        return result

    except asyncio.TimeoutError:
        raise Exception(f"Message sending timed out after 30 seconds")
    except Exception as e:
        raise Exception(f"Failed to send message: {str(e)}")

async def _send_message_internal(session_id: str, message: str) -> str:
    """Internal message sending with proper error handling"""
    try:
        client = active_sessions[session_id]["client"]

        # Build conversation context for the Agent SDK
        conversation_context = _build_conversation_context(session_id, message)

        # Send message with conversation context
        await client.query(conversation_context)

        # Collect response with proper Unicode handling
        response_parts = []
        try:
            response_complete = False
            message_count = 0
            max_messages = 10  # Prevent infinite loops

            async for response_message in client.receive_response():
                message_count += 1

                # Check for different message types
                if hasattr(response_message, 'content') and isinstance(response_message.content, list):
                    for block in response_message.content:
                        if hasattr(block, 'text'):
                            # Handle Unicode/emoji characters properly
                            text = str(block.text) if block.text else ""
                            if text.strip():  # Only add non-empty text
                                response_parts.append(text)
                                response_complete = True

                # Check for ResultMessage which indicates completion
                if hasattr(response_message, 'subtype') and response_message.subtype in ['success', 'error']:
                    break

                # Stop if we have content or reached message limit
                if response_complete or message_count >= max_messages:
                    break

        except Exception as response_error:
            # Fallback if response collection fails
            response_parts.append(f"Error collecting response: {str(response_error)}")

        # Fix string literal issue - use proper newline
        response = "\n".join(response_parts)

        # Integrate tool logs into response
        if current_tool_logs:
            tool_summary = "\n".join(current_tool_logs)
            response_with_tools = f"{tool_summary}\n\n{response}"
            # Clear tool logs after using them
            current_tool_logs.clear()
        else:
            response_with_tools = response

        # Store in conversation history with safe JSON serialization
        conversation_history[session_id].extend([
            {"role": "user", "message": message, "timestamp": datetime.now().isoformat()},
            {"role": "agent", "message": response_with_tools, "timestamp": datetime.now().isoformat()}
        ])

        # Update persistent storage
        try:
            conversation_id = active_sessions[session_id].get("conversation_id")
            if conversation_id:
                # Add the new messages to storage
                conversation_storage.add_message_to_conversation(conversation_id, "user", message)
                conversation_storage.add_message_to_conversation(conversation_id, "agent", response_with_tools)
        except Exception as storage_error:
            # Log storage error but don't fail the message sending
            print(f"Warning: Failed to update conversation in storage: {storage_error}")

        return response_with_tools

    except Exception as e:
        raise

async def get_conversation(session_id: str) -> List[Dict[str, str]]:
    """Core Function 3: Get conversation history"""
    if session_id not in conversation_history:
        raise Exception(f"Session not found: {session_id}")

    return conversation_history[session_id]

async def end_conversation(session_id: str) -> bool:
    """Core Function 4: End conversation"""
    if session_id not in active_sessions:
        raise Exception(f"Session not found: {session_id}")

    try:
        # Add timeout protection
        return await asyncio.wait_for(_end_conversation_internal(session_id), timeout=10)

    except asyncio.TimeoutError:
        raise Exception(f"Conversation ending timed out after 10 seconds")
    except Exception as e:
        raise Exception(f"Failed to end conversation: {str(e)}")

async def _end_conversation_internal(session_id: str) -> bool:
    """Internal conversation ending with proper error handling"""
    try:
        # Disconnect client
        client = active_sessions[session_id]["client"]
        await client.disconnect()

        # Mark conversation as completed in persistent storage
        try:
            conversation_id = active_sessions[session_id].get("conversation_id")
            if conversation_id:
                conversation_storage.update_conversation_status(conversation_id, "completed")
        except Exception as storage_error:
            # Log storage error but don't fail the conversation ending
            print(f"Warning: Failed to mark conversation as completed in storage: {storage_error}")

        # Clean up
        del active_sessions[session_id]
        del conversation_history[session_id]

        return True

    except Exception as e:
        # Force cleanup even if disconnect fails
        if session_id in active_sessions:
            del active_sessions[session_id]
        if session_id in conversation_history:
            del conversation_history[session_id]
        raise

@app.call_tool()
async def call_tool(name: str, arguments: dict):
    try:
        if name == "create_conversation":
            session_id = await create_conversation(arguments["task"])

            return {
                "content": [{
                    "type": "text",
                    "text": f"""Async conversation created successfully!

Session ID: {session_id}
Task: {arguments['task']}

The agent is now processing your task in the background. Use send_message with session_id '{session_id}' to continue the conversation and get the agent's response."""
                }]
            }

        elif name == "send_message":
            response = await send_message(arguments["session_id"], arguments["message"])

            return {
                "content": [{
                    "type": "text",
                    "text": f"""Message sent and response received!

Agent's Response:
{response[:400]}...

Use get_conversation to see the full conversation history."""
                }]
            }

        elif name == "get_conversation":
            history = await get_conversation(arguments["session_id"])

            conversation_text = ""
            for i, msg in enumerate(history, 1):
                role = "You" if msg["role"] == "user" else "Agent"
                content = msg["message"][:200] + "..." if len(msg["message"]) > 200 else msg["message"]
                conversation_text += f"{i}. {role}: {content}\\n\\n"

            return {
                "content": [{
                    "type": "text",
                    "text": f"""📚 Full Conversation ({len(history)} messages):

{conversation_text}"""
                }]
            }

        elif name == "end_conversation":
            success = await end_conversation(arguments["session_id"])

            return {
                "content": [{
                    "type": "text",
                    "text": f"Conversation {arguments['session_id']} ended successfully!"
                }]
            }

        elif name == "list_conversations":
            conversations = conversation_storage.list_conversations()
            stats = conversation_storage.get_conversation_stats()

            conversation_list = ""
            for i, conv in enumerate(conversations[:10], 1):  # Show first 10
                metadata = conv["metadata"]
                conversation_list += f"{i}. {metadata['conversation_id'][:12]}... - {metadata['initial_task'][:60]}...\n"
                conversation_list += f"   Status: {metadata['status']}, Messages: {metadata['message_count']}, Updated: {metadata['last_updated'][:19]}\n\n"

            return {
                "content": [{
                    "type": "text",
                    "text": f"""📚 Stored Conversations ({stats['total_conversations']} total):

{conversation_list}
Storage Location: {stats['storage_location']}
Status Counts: {stats['status_counts']}"""
                }]
            }

        elif name == "search_conversations":
            results = conversation_storage.search_conversations(arguments["query"])

            search_results = ""
            for i, conv in enumerate(results[:5], 1):  # Show first 5 results
                metadata = conv["metadata"]
                matches = conv.get("matches", [])
                search_results += f"{i}. {metadata['conversation_id'][:12]}... - {metadata['initial_task'][:50]}...\n"
                search_results += f"   Matches found: {len(matches)}\n"
                for match in matches[:3]:  # Show first 3 matches
                    search_results += f"   • {match[:80]}...\n"
                search_results += "\n"

            return {
                "content": [{
                    "type": "text",
                    "text": f"""Search Results for '{arguments['query']}' ({len(results)} found):

{search_results}"""
                }]
            }

        else:
            raise ValueError(f"Unknown tool: {name}")

    except Exception as e:
        return {
            "content": [{
                "type": "text",
                "text": f"Error: {str(e)}"
            }]
        }

async def main():
    """Run the C2C-Dev MCP server"""
    async with stdio_server() as streams:
        await app.run(streams[0], streams[1], app.create_initialization_options())

if __name__ == "__main__":
    print("🚀 Starting C2C-Dev MCP Server...")
    print("🎯 Minimal Working MVP for Bidirectional Agent Communication")
    asyncio.run(main())
