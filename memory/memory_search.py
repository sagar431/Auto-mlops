"""
Memory Search Module for MLOps Agent.
Searches past experiment sessions for relevant context.
Supports fuzzy matching on queries, metrics, and configurations.
Now uses database instead of JSON files for persistence.
"""

from typing import Any

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import Session, selectinload

from db.models import AgentSession
from db.session import get_async_session

try:
    from rapidfuzz import fuzz

    RAPIDFUZZ_AVAILABLE = True
except ImportError:
    RAPIDFUZZ_AVAILABLE = False


class MemorySearch:
    """
    Search past MLOps Agent sessions for relevant context.
    Enables learning from previous experiments.
    Uses database queries instead of JSON files.
    """

    def __init__(self, db: Session | None = None):
        """
        Initialize MemorySearch with database session.

        Args:
            db: SQLAlchemy Session instance. If None, data must be loaded explicitly.
        """
        self.db = db
        self._index_data: list[dict] | None = None

    @property
    def index_data(self) -> list[dict]:
        """Lazy load sessions from database."""
        if self._index_data is None:
            self._index_data = self._load_sessions()
        return self._index_data

    def _load_sessions(self) -> list[dict]:
        """Load all session logs from database."""
        if self.db is None:
            return []

        stmt = (
            select(AgentSession)
            .options(selectinload(AgentSession.experiment_state))
            .order_by(AgentSession.created_at.desc())
        )
        sessions = list(self.db.execute(stmt).scalars().all())

        all_sessions = []
        for session in sessions:
            exp_state = session.experiment_state
            session_entry = {
                "session_id": session.session_id,
                "original_query": session.original_query or "",
                "final_summary": session.result or "",
                "status": session.status,
                "goal_achieved": session.status == "completed",
                "timestamp": session.created_at.isoformat() if session.created_at else "",
                "experiment_state": exp_state.to_dict() if exp_state else {},
                "project_path": session.project_path or "",
            }
            all_sessions.append(session_entry)

        return all_sessions

    def search_memory(self, query: str, top_k: int = 3, threshold: int = 40) -> list[dict]:
        """
        Search for similar past sessions using fuzzy matching.

        Args:
            query: Search query (user's request)
            top_k: Maximum number of results to return
            threshold: Minimum similarity score (0-100)

        Returns:
            List of matching session summaries
        """
        if not self.index_data:
            return []

        if not RAPIDFUZZ_AVAILABLE:
            return self._simple_search(query, top_k)

        query_lower = query.lower().strip()
        scored = []

        for entry in self.index_data:
            orig_query = entry.get("original_query", "").lower()
            summary = entry.get("final_summary", "").lower()

            query_score = fuzz.token_set_ratio(query_lower, orig_query)
            summary_score = fuzz.token_set_ratio(query_lower, summary)

            combined_score = int(0.7 * query_score + 0.3 * summary_score)

            if entry.get("goal_achieved", False):
                combined_score = min(100, combined_score + 5)

            if combined_score >= threshold:
                scored.append((combined_score, entry))

        scored.sort(key=lambda x: x[0], reverse=True)

        return [self._format_result(score, entry) for score, entry in scored[:top_k]]

    def _simple_search(self, query: str, top_k: int) -> list[dict]:
        """Simple substring-based search fallback."""
        query_lower = query.lower()
        matches = []

        for entry in self.index_data:
            orig_query = entry.get("original_query", "").lower()
            if query_lower in orig_query or any(word in orig_query for word in query_lower.split()):
                matches.append((50, entry))

        return [self._format_result(score, entry) for score, entry in matches[:top_k]]

    def _format_result(self, score: int, entry: dict) -> dict:
        """Format a search result for output."""
        exp_state = entry.get("experiment_state", {})

        return {
            "score": score,
            "session_id": entry["session_id"],
            "original_query": entry["original_query"],
            "summary": entry["final_summary"][:500] if entry["final_summary"] else "",
            "timestamp": entry["timestamp"],
            "goal_achieved": entry.get("goal_achieved", False),
            "best_accuracy": exp_state.get("best_accuracy"),
            "target_accuracy": exp_state.get("target_accuracy"),
            "improvement_attempts": exp_state.get("improvement_attempt", 0),
            "stage": exp_state.get("stage", "unknown"),
        }

    def search_by_metric(
        self, metric_name: str = "accuracy", min_value: float = 0.0, top_k: int = 5
    ) -> list[dict]:
        """
        Search for sessions with best metrics.

        Args:
            metric_name: Metric to search by (accuracy, loss, etc.)
            min_value: Minimum metric value
            top_k: Maximum number of results

        Returns:
            List of sessions sorted by metric
        """
        results = []

        for entry in self.index_data:
            exp_state = entry.get("experiment_state", {})
            metric_value = None

            if metric_name == "accuracy":
                metric_value = exp_state.get("best_accuracy") or exp_state.get("current_accuracy")
            elif metric_name == "loss":
                metric_value = exp_state.get("current_loss")

            if metric_value is not None and metric_value >= min_value:
                results.append((metric_value, entry))

        results.sort(key=lambda x: x[0], reverse=True)

        return [
            {"metric_value": value, **self._format_result(100, entry)}
            for value, entry in results[:top_k]
        ]

    def search_by_config(self, config_key: str, config_value: Any) -> list[dict]:
        """
        Search for sessions with specific configuration.

        Args:
            config_key: Configuration key to search
            config_value: Expected value

        Returns:
            List of matching sessions
        """
        results = []

        for entry in self.index_data:
            exp_state = entry.get("experiment_state", {})
            current_config = exp_state.get("current_config", {})

            if config_key in current_config and current_config[config_key] == config_value:
                results.append(self._format_result(100, entry))

        return results

    def get_improvement_history(self, experiment_name: str | None = None) -> list[dict]:
        """
        Get improvement attempts history for learning.

        Args:
            experiment_name: Optional filter by experiment name

        Returns:
            List of improvement attempts with configs and results
        """
        history = []

        for entry in self.index_data:
            exp_state = entry.get("experiment_state", {})

            if experiment_name and exp_state.get("experiment_name") != experiment_name:
                continue

            for attempt in exp_state.get("improvement_history", []):
                history.append(
                    {
                        "session_id": entry["session_id"],
                        "experiment_name": exp_state.get("experiment_name"),
                        "attempt": attempt.get("attempt"),
                        "config_changes": attempt.get("config_changes", {}),
                        "accuracy_before": attempt.get("accuracy_before"),
                        "accuracy_after": attempt.get("accuracy_after"),
                        "improvement": (
                            attempt.get("accuracy_after", 0) - attempt.get("accuracy_before", 0)
                        ),
                        "timestamp": attempt.get("timestamp"),
                    }
                )

        history.sort(key=lambda x: x.get("improvement", 0), reverse=True)

        return history


class AsyncMemorySearch:
    """
    Async version of MemorySearch for use with async database sessions.
    """

    def __init__(self, db: AsyncSession | None = None):
        """
        Initialize AsyncMemorySearch with async database session.

        Args:
            db: SQLAlchemy AsyncSession instance.
        """
        self.db = db
        self._index_data: list[dict] | None = None

    async def _load_sessions(self) -> list[dict]:
        """Load all session logs from database asynchronously."""
        if self.db is None:
            return []

        stmt = (
            select(AgentSession)
            .options(selectinload(AgentSession.experiment_state))
            .order_by(AgentSession.created_at.desc())
        )
        result = await self.db.execute(stmt)
        sessions = list(result.scalars().all())

        all_sessions = []
        for session in sessions:
            exp_state = session.experiment_state
            session_entry = {
                "session_id": session.session_id,
                "original_query": session.original_query or "",
                "final_summary": session.result or "",
                "status": session.status,
                "goal_achieved": session.status == "completed",
                "timestamp": session.created_at.isoformat() if session.created_at else "",
                "experiment_state": exp_state.to_dict() if exp_state else {},
                "project_path": session.project_path or "",
            }
            all_sessions.append(session_entry)

        return all_sessions

    async def load_data(self) -> None:
        """Explicitly load session data from database."""
        self._index_data = await self._load_sessions()

    @property
    def index_data(self) -> list[dict]:
        """Get loaded session data. Must call load_data() first."""
        if self._index_data is None:
            return []
        return self._index_data

    async def search_memory(self, query: str, top_k: int = 3, threshold: int = 40) -> list[dict]:
        """
        Search for similar past sessions using fuzzy matching.

        Args:
            query: Search query (user's request)
            top_k: Maximum number of results to return
            threshold: Minimum similarity score (0-100)

        Returns:
            List of matching session summaries
        """
        if self._index_data is None:
            await self.load_data()

        if not self.index_data:
            return []

        if not RAPIDFUZZ_AVAILABLE:
            return self._simple_search(query, top_k)

        query_lower = query.lower().strip()
        scored = []

        for entry in self.index_data:
            orig_query = entry.get("original_query", "").lower()
            summary = entry.get("final_summary", "").lower()

            query_score = fuzz.token_set_ratio(query_lower, orig_query)
            summary_score = fuzz.token_set_ratio(query_lower, summary)

            combined_score = int(0.7 * query_score + 0.3 * summary_score)

            if entry.get("goal_achieved", False):
                combined_score = min(100, combined_score + 5)

            if combined_score >= threshold:
                scored.append((combined_score, entry))

        scored.sort(key=lambda x: x[0], reverse=True)

        return [self._format_result(score, entry) for score, entry in scored[:top_k]]

    def _simple_search(self, query: str, top_k: int) -> list[dict]:
        """Simple substring-based search fallback."""
        query_lower = query.lower()
        matches = []

        for entry in self.index_data:
            orig_query = entry.get("original_query", "").lower()
            if query_lower in orig_query or any(word in orig_query for word in query_lower.split()):
                matches.append((50, entry))

        return [self._format_result(score, entry) for score, entry in matches[:top_k]]

    def _format_result(self, score: int, entry: dict) -> dict:
        """Format a search result for output."""
        exp_state = entry.get("experiment_state", {})

        return {
            "score": score,
            "session_id": entry["session_id"],
            "original_query": entry["original_query"],
            "summary": entry["final_summary"][:500] if entry["final_summary"] else "",
            "timestamp": entry["timestamp"],
            "goal_achieved": entry.get("goal_achieved", False),
            "best_accuracy": exp_state.get("best_accuracy"),
            "target_accuracy": exp_state.get("target_accuracy"),
            "improvement_attempts": exp_state.get("improvement_attempt", 0),
            "stage": exp_state.get("stage", "unknown"),
        }

    async def search_by_metric(
        self, metric_name: str = "accuracy", min_value: float = 0.0, top_k: int = 5
    ) -> list[dict]:
        """
        Search for sessions with best metrics.

        Args:
            metric_name: Metric to search by (accuracy, loss, etc.)
            min_value: Minimum metric value
            top_k: Maximum number of results

        Returns:
            List of sessions sorted by metric
        """
        if self._index_data is None:
            await self.load_data()

        results = []

        for entry in self.index_data:
            exp_state = entry.get("experiment_state", {})
            metric_value = None

            if metric_name == "accuracy":
                metric_value = exp_state.get("best_accuracy") or exp_state.get("current_accuracy")
            elif metric_name == "loss":
                metric_value = exp_state.get("current_loss")

            if metric_value is not None and metric_value >= min_value:
                results.append((metric_value, entry))

        results.sort(key=lambda x: x[0], reverse=True)

        return [
            {"metric_value": value, **self._format_result(100, entry)}
            for value, entry in results[:top_k]
        ]

    async def search_by_config(self, config_key: str, config_value: Any) -> list[dict]:
        """
        Search for sessions with specific configuration.

        Args:
            config_key: Configuration key to search
            config_value: Expected value

        Returns:
            List of matching sessions
        """
        if self._index_data is None:
            await self.load_data()

        results = []

        for entry in self.index_data:
            exp_state = entry.get("experiment_state", {})
            current_config = exp_state.get("current_config", {})

            if config_key in current_config and current_config[config_key] == config_value:
                results.append(self._format_result(100, entry))

        return results

    async def get_improvement_history(self, experiment_name: str | None = None) -> list[dict]:
        """
        Get improvement attempts history for learning.

        Args:
            experiment_name: Optional filter by experiment name

        Returns:
            List of improvement attempts with configs and results
        """
        if self._index_data is None:
            await self.load_data()

        history = []

        for entry in self.index_data:
            exp_state = entry.get("experiment_state", {})

            if experiment_name and exp_state.get("experiment_name") != experiment_name:
                continue

            for attempt in exp_state.get("improvement_history", []):
                history.append(
                    {
                        "session_id": entry["session_id"],
                        "experiment_name": exp_state.get("experiment_name"),
                        "attempt": attempt.get("attempt"),
                        "config_changes": attempt.get("config_changes", {}),
                        "accuracy_before": attempt.get("accuracy_before"),
                        "accuracy_after": attempt.get("accuracy_after"),
                        "improvement": (
                            attempt.get("accuracy_after", 0) - attempt.get("accuracy_before", 0)
                        ),
                        "timestamp": attempt.get("timestamp"),
                    }
                )

        history.sort(key=lambda x: x.get("improvement", 0), reverse=True)

        return history


def search_past_experiments(query: str, top_k: int = 3) -> list[dict]:
    """
    Quick search for past experiments using database.

    Args:
        query: Search query
        top_k: Maximum results

    Returns:
        List of matching experiments
    """
    from db import get_session

    with get_session() as db:
        ms = MemorySearch(db)
        return ms.search_memory(query, top_k=top_k)


async def async_search_past_experiments(query: str, top_k: int = 3) -> list[dict]:
    """
    Async quick search for past experiments using database.

    Args:
        query: Search query
        top_k: Maximum results

    Returns:
        List of matching experiments
    """
    async with get_async_session() as db:
        ms = AsyncMemorySearch(db)
        return await ms.search_memory(query, top_k=top_k)


if __name__ == "__main__":
    from db import get_session, init_db

    init_db()

    with get_session() as db:
        ms = MemorySearch(db)

        print(f"Loaded {len(ms.index_data)} sessions")

        if ms.index_data:
            user_query = input("Enter your search query: ")
            results = ms.search_memory(user_query)

            for i, r in enumerate(results, 1):
                print(f"\n--- Result {i} (Score: {r['score']}) ---")
                print(f"Query: {r['original_query']}")
                print(f"Accuracy: {r.get('best_accuracy', 'N/A')}")
                print(f"Summary: {r['summary'][:200]}...")
        else:
            print("No sessions found. Run some experiments first!")
