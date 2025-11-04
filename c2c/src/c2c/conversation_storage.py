"""Conversation Storage Module for C2C - Persistent conversation management"""

import json
import os
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, asdict
import uuid

@dataclass
class ConversationMetadata:
    """Metadata for a conversation"""
    conversation_id: str
    session_id: str
    created_at: str
    last_updated: str
    message_count: int
    initial_task: str
    status: str  # "active", "completed", "ended"

@dataclass
class ConversationMessage:
    """Individual message in a conversation"""
    role: str  # "user" or "agent"
    message: str
    timestamp: str
    message_id: Optional[str] = None

class ConversationStorage:
    """Manages persistent storage of C2C conversations"""

    def __init__(self):
        # Create conversations directory in user's home directory
        self.conversations_dir = Path.home() / ".c2c" / "conversations"
        self.conversations_dir.mkdir(parents=True, exist_ok=True)

    def save_conversation(self, session_id: str, messages: List[Dict[str, Any]], initial_task: str, status: str = "active") -> str:
        """Save a conversation to persistent storage"""
        conversation_id = f"conv_{uuid.uuid4().hex[:12]}"

        metadata = ConversationMetadata(
            conversation_id=conversation_id,
            session_id=session_id,
            created_at=messages[0].get("timestamp", datetime.now().isoformat()) if messages else datetime.now().isoformat(),
            last_updated=datetime.now().isoformat(),
            message_count=len(messages),
            initial_task=initial_task,
            status=status
        )

        # Convert messages to proper format
        formatted_messages = []
        for msg in messages:
            formatted_msg = ConversationMessage(
                role=msg["role"],
                message=msg["message"],
                timestamp=msg["timestamp"],
                message_id=msg.get("message_id")
            )
            formatted_messages.append(asdict(formatted_msg))

        # Create conversation data
        conversation_data = {
            "metadata": asdict(metadata),
            "messages": formatted_messages
        }

        # Save to JSON-L file
        conversation_file = self.conversations_dir / f"{conversation_id}.jsonl"
        with open(conversation_file, 'w', encoding='utf-8') as f:
            json.dump(conversation_data, f, ensure_ascii=False, indent=2)

        return conversation_id

    def load_conversation(self, conversation_id: str) -> Optional[Dict[str, Any]]:
        """Load a conversation from storage"""
        conversation_file = self.conversations_dir / f"{conversation_id}.jsonl"

        if not conversation_file.exists():
            return None

        try:
            with open(conversation_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        except (json.JSONDecodeError, FileNotFoundError):
            return None

    def list_conversations(self) -> List[Dict[str, Any]]:
        """List all stored conversations"""
        conversations = []

        for file_path in self.conversations_dir.glob("*.jsonl"):
            try:
                conversation_data = self.load_conversation(file_path.stem)
                if conversation_data:
                    conversations.append(conversation_data)
            except Exception:
                # Skip corrupted files
                continue

        # Sort by last updated (most recent first)
        conversations.sort(key=lambda x: x["metadata"]["last_updated"], reverse=True)
        return conversations

    def update_conversation_status(self, conversation_id: str, status: str) -> bool:
        """Update conversation status"""
        conversation_data = self.load_conversation(conversation_id)
        if not conversation_data:
            return False

        conversation_data["metadata"]["status"] = status
        conversation_data["metadata"]["last_updated"] = datetime.now().isoformat()

        conversation_file = self.conversations_dir / f"{conversation_id}.jsonl"
        with open(conversation_file, 'w', encoding='utf-8') as f:
            json.dump(conversation_data, f, ensure_ascii=False, indent=2)

        return True

    def delete_conversation(self, conversation_id: str) -> bool:
        """Delete a conversation from storage"""
        conversation_file = self.conversations_dir / f"{conversation_id}.jsonl"

        if conversation_file.exists():
            conversation_file.unlink()
            return True

        return False

    def add_message_to_conversation(self, conversation_id: str, role: str, message: str) -> bool:
        """Add a new message to an existing conversation"""
        conversation_data = self.load_conversation(conversation_id)
        if not conversation_data:
            return False

        new_message = {
            "role": role,
            "message": message,
            "timestamp": datetime.now().isoformat(),
            "message_id": f"msg_{uuid.uuid4().hex[:8]}"
        }

        conversation_data["messages"].append(new_message)
        conversation_data["metadata"]["message_count"] = len(conversation_data["messages"])
        conversation_data["metadata"]["last_updated"] = datetime.now().isoformat()

        # Update status to active if not already
        if conversation_data["metadata"]["status"] == "completed":
            conversation_data["metadata"]["status"] = "active"

        conversation_file = self.conversations_dir / f"{conversation_id}.jsonl"
        with open(conversation_file, 'w', encoding='utf-8') as f:
            json.dump(conversation_data, f, ensure_ascii=False, indent=2)

        return True

    def search_conversations(self, query: str) -> List[Dict[str, Any]]:
        """Search conversations by content"""
        results = []
        query_lower = query.lower()

        for conversation in self.list_conversations():
            matches = []

            # Search in metadata
            if query_lower in conversation["metadata"]["initial_task"].lower():
                matches.append(f"Task: {conversation['metadata']['initial_task']}")

            # Search in messages
            for message in conversation["messages"]:
                if query_lower in message["message"].lower():
                    matches.append(f"{message['role']}: {message['message'][:100]}...")

            if matches:
                result = conversation.copy()
                result["matches"] = matches
                results.append(result)

        return results

    def get_conversation_stats(self) -> Dict[str, Any]:
        """Get statistics about stored conversations"""
        conversations = self.list_conversations()

        total_conversations = len(conversations)
        total_messages = sum(conv["metadata"]["message_count"] for conv in conversations)

        status_counts = {}
        for conv in conversations:
            status = conv["metadata"]["status"]
            status_counts[status] = status_counts.get(status, 0) + 1

        return {
            "total_conversations": total_conversations,
            "total_messages": total_messages,
            "status_counts": status_counts,
            "storage_location": str(self.conversations_dir)
        }