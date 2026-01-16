"""
Agent Session for MLOps Agent - Session management with experiment snapshots.
"""

import json
import logging
import os
from datetime import datetime
from typing import Optional, Dict, Any, List
from pathlib import Path

logger = logging.getLogger(__name__)


class AgentSession:
    """
    Manages a single agent session with experiment tracking.
    Handles persistence and recovery of session state.
    """
    
    def __init__(
        self,
        session_id: str,
        original_query: str,
        project_path: Optional[str] = None,
        profile: str = "default"
    ):
        self.session_id = session_id
        self.original_query = original_query
        self.project_path = project_path
        self.profile = profile
        
        # Session metadata
        self.created_at = datetime.utcnow().isoformat()
        self.updated_at = self.created_at
        self.completed_at: Optional[str] = None
        self.status = "active"  # active, completed, failed, paused
        
        # Conversation history for LLM context
        self.messages: List[Dict[str, Any]] = []
        
        # Experiment snapshots (checkpoints)
        self.snapshots: List[Dict[str, Any]] = []
        
        # Session logs directory
        self.logs_dir = Path(__file__).parent.parent / "memory" / "session_logs"
        self.logs_dir.mkdir(parents=True, exist_ok=True)
    
    def add_message(self, role: str, content: str, metadata: Optional[Dict] = None):
        """Add a message to conversation history."""
        message = {
            "role": role,
            "content": content,
            "timestamp": datetime.utcnow().isoformat()
        }
        if metadata:
            message["metadata"] = metadata
        
        self.messages.append(message)
        self.updated_at = datetime.utcnow().isoformat()
    
    def add_tool_call(self, tool_name: str, args: Dict, result: Dict):
        """Record a tool call in session history."""
        self.add_message(
            role="tool",
            content=json.dumps(result, default=str)[:1000],
            metadata={
                "tool": tool_name,
                "args": args,
                "success": result.get("success", False)
            }
        )
    
    def create_snapshot(self, context_snapshot: Dict, label: str = "auto"):
        """Create an experiment snapshot/checkpoint."""
        snapshot = {
            "id": len(self.snapshots),
            "label": label,
            "timestamp": datetime.utcnow().isoformat(),
            "context": context_snapshot,
            "message_count": len(self.messages)
        }
        self.snapshots.append(snapshot)
        return snapshot["id"]
    
    def get_latest_snapshot(self) -> Optional[Dict]:
        """Get the most recent snapshot."""
        return self.snapshots[-1] if self.snapshots else None
    
    def get_conversation_for_llm(self, max_messages: int = 20) -> List[Dict]:
        """Get recent conversation history formatted for LLM."""
        recent = self.messages[-max_messages:] if len(self.messages) > max_messages else self.messages
        
        formatted = []
        for msg in recent:
            if msg["role"] == "tool":
                # Format tool calls as assistant messages
                formatted.append({
                    "role": "assistant",
                    "content": f"[Tool: {msg.get('metadata', {}).get('tool', 'unknown')}]\n{msg['content']}"
                })
            else:
                formatted.append({
                    "role": msg["role"],
                    "content": msg["content"]
                })
        
        return formatted
    
    def mark_completed(self, success: bool = True):
        """Mark session as completed."""
        self.status = "completed" if success else "failed"
        self.completed_at = datetime.utcnow().isoformat()
        self.updated_at = self.completed_at
    
    def to_json(self) -> Dict[str, Any]:
        """Serialize session to JSON-compatible dict."""
        return {
            "session_id": self.session_id,
            "original_query": self.original_query,
            "project_path": self.project_path,
            "profile": self.profile,
            "status": self.status,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "completed_at": self.completed_at,
            "message_count": len(self.messages),
            "snapshot_count": len(self.snapshots),
            "messages": self.messages[-10:],  # Last 10 messages
            "latest_snapshot": self.get_latest_snapshot()
        }
    
    def save(self):
        """Save session to disk."""
        session_file = self.logs_dir / f"{self.session_id}.json"
        with open(session_file, "w") as f:
            json.dump(self.to_json(), f, indent=2, default=str)
    
    @classmethod
    def load(cls, session_id: str) -> Optional["AgentSession"]:
        """Load session from disk."""
        logs_dir = Path(__file__).parent.parent / "memory" / "session_logs"
        session_file = logs_dir / f"{session_id}.json"
        
        if not session_file.exists():
            return None
        
        with open(session_file, "r") as f:
            data = json.load(f)
        
        session = cls(
            session_id=data["session_id"],
            original_query=data["original_query"],
            project_path=data.get("project_path"),
            profile=data.get("profile", "default")
        )
        session.status = data["status"]
        session.created_at = data["created_at"]
        session.updated_at = data["updated_at"]
        session.completed_at = data.get("completed_at")
        session.messages = data.get("messages", [])
        
        return session
    
    def get_experiment_history(self) -> List[Dict]:
        """Extract experiment-related events from session."""
        history = []
        
        for msg in self.messages:
            metadata = msg.get("metadata", {})
            if metadata.get("tool") in ["log_mlflow_metrics", "check_accuracy_threshold", 
                                         "suggest_improvements", "update_hydra_config"]:
                history.append({
                    "timestamp": msg["timestamp"],
                    "tool": metadata["tool"],
                    "args": metadata.get("args", {}),
                    "success": metadata.get("success", False)
                })
        
        return history


class SessionManager:
    """Manages multiple agent sessions."""
    
    def __init__(self):
        self.sessions: Dict[str, AgentSession] = {}
        self.logs_dir = Path(__file__).parent.parent / "memory" / "session_logs"
        self.logs_dir.mkdir(parents=True, exist_ok=True)
    
    def create_session(
        self,
        session_id: str,
        query: str,
        project_path: Optional[str] = None,
        profile: str = "default"
    ) -> AgentSession:
        """Create a new session."""
        session = AgentSession(
            session_id=session_id,
            original_query=query,
            project_path=project_path,
            profile=profile
        )
        self.sessions[session_id] = session
        return session
    
    def get_session(self, session_id: str) -> Optional[AgentSession]:
        """Get session by ID, loading from disk if needed."""
        if session_id in self.sessions:
            return self.sessions[session_id]
        
        # Try loading from disk
        session = AgentSession.load(session_id)
        if session:
            self.sessions[session_id] = session
        return session
    
    def list_sessions(self, status: Optional[str] = None) -> List[Dict]:
        """List all sessions, optionally filtered by status."""
        sessions = []

        for session_file in self.logs_dir.glob("*.json"):
            try:
                with open(session_file, "r") as f:
                    data = json.load(f)

                if status is None or data.get("status") == status:
                    sessions.append({
                        "session_id": data["session_id"],
                        "query": data["original_query"][:50],
                        "status": data["status"],
                        "created_at": data["created_at"]
                    })
            except json.JSONDecodeError as e:
                logger.warning(f"Failed to parse session file {session_file}: {e}")
                continue
            except KeyError as e:
                logger.warning(f"Missing required field in session file {session_file}: {e}")
                continue
            except Exception as e:
                logger.warning(f"Error loading session file {session_file}: {e}")
                continue

        return sorted(sessions, key=lambda x: x["created_at"], reverse=True)
