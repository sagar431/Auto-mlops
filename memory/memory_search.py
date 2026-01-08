"""
Memory Search Module for MLOps Agent.
Searches past experiment sessions for relevant context.
Supports fuzzy matching on queries, metrics, and configurations.
"""

import json
from pathlib import Path
from typing import List, Dict, Any, Optional

try:
    from rapidfuzz import fuzz
    RAPIDFUZZ_AVAILABLE = True
except ImportError:
    RAPIDFUZZ_AVAILABLE = False


class MemorySearch:
    """
    Search past MLOps Agent sessions for relevant context.
    Enables learning from previous experiments.
    """

    def __init__(self, base_dir: str = "memory/session_logs"):
        """
        Initialize MemorySearch with session logs directory.

        Args:
            base_dir: Path to session logs directory
        """
        self.base_dir = Path(base_dir)
        self.index_data = self._load_sessions()

    def _load_sessions(self) -> List[Dict]:
        """Load all session logs from storage."""
        all_sessions = []

        if not self.base_dir.exists():
            return all_sessions

        for session_file in self.base_dir.rglob("*.json"):
            # Skip step files
            if "_steps" in session_file.name:
                continue

            try:
                with open(session_file, "r", encoding="utf-8") as f:
                    data = json.load(f)

                if isinstance(data, dict):
                    session_entry = {
                        "file": str(session_file),
                        "session_id": data.get("session_id", session_file.stem),
                        "original_query": data.get("original_query", ""),
                        "final_summary": data.get("final_summary", ""),
                        "status": data.get("status", "unknown"),
                        "goal_achieved": data.get("goal_achieved", False),
                        "timestamp": data.get("timestamp", ""),
                        # MLOps-specific fields
                        "experiment_state": data.get("experiment_state", {}),
                        "project_path": data.get("project_path", "")
                    }
                    all_sessions.append(session_entry)

            except Exception as e:
                print(f"[WARN] Failed to load session {session_file}: {e}")

        return all_sessions

    def search_memory(
        self,
        query: str,
        top_k: int = 3,
        threshold: int = 40
    ) -> List[Dict]:
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
            # Fallback to simple substring matching
            return self._simple_search(query, top_k)

        query_lower = query.lower().strip()
        scored = []

        for entry in self.index_data:
            orig_query = entry.get("original_query", "").lower()
            summary = entry.get("final_summary", "").lower()

            # Calculate similarity scores
            query_score = fuzz.token_set_ratio(query_lower, orig_query)
            summary_score = fuzz.token_set_ratio(query_lower, summary)

            # Weighted combination (query more important)
            combined_score = int(0.7 * query_score + 0.3 * summary_score)

            # Boost successful sessions
            if entry.get("goal_achieved", False):
                combined_score = min(100, combined_score + 5)

            if combined_score >= threshold:
                scored.append((combined_score, entry))

        # Sort by score descending
        scored.sort(key=lambda x: x[0], reverse=True)

        return [
            self._format_result(score, entry)
            for score, entry in scored[:top_k]
        ]

    def _simple_search(self, query: str, top_k: int) -> List[Dict]:
        """Simple substring-based search fallback."""
        query_lower = query.lower()
        matches = []

        for entry in self.index_data:
            orig_query = entry.get("original_query", "").lower()
            if query_lower in orig_query or any(
                word in orig_query for word in query_lower.split()
            ):
                matches.append((50, entry))

        return [
            self._format_result(score, entry)
            for score, entry in matches[:top_k]
        ]

    def _format_result(self, score: int, entry: Dict) -> Dict:
        """Format a search result for output."""
        exp_state = entry.get("experiment_state", {})

        return {
            "score": score,
            "session_id": entry["session_id"],
            "original_query": entry["original_query"],
            "summary": entry["final_summary"][:500] if entry["final_summary"] else "",
            "timestamp": entry["timestamp"],
            "goal_achieved": entry.get("goal_achieved", False),
            # MLOps-specific context
            "best_accuracy": exp_state.get("best_accuracy"),
            "target_accuracy": exp_state.get("target_accuracy"),
            "improvement_attempts": exp_state.get("improvement_attempt", 0),
            "stage": exp_state.get("stage", "unknown")
        }

    def search_by_metric(
        self,
        metric_name: str = "accuracy",
        min_value: float = 0.0,
        top_k: int = 5
    ) -> List[Dict]:
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

        # Sort by metric descending
        results.sort(key=lambda x: x[0], reverse=True)

        return [
            {
                "metric_value": value,
                **self._format_result(100, entry)
            }
            for value, entry in results[:top_k]
        ]

    def search_by_config(
        self,
        config_key: str,
        config_value: Any
    ) -> List[Dict]:
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

    def get_improvement_history(self, experiment_name: str = None) -> List[Dict]:
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
                history.append({
                    "session_id": entry["session_id"],
                    "experiment_name": exp_state.get("experiment_name"),
                    "attempt": attempt.get("attempt"),
                    "config_changes": attempt.get("config_changes", {}),
                    "accuracy_before": attempt.get("accuracy_before"),
                    "accuracy_after": attempt.get("accuracy_after"),
                    "improvement": (attempt.get("accuracy_after", 0) -
                                  attempt.get("accuracy_before", 0)),
                    "timestamp": attempt.get("timestamp")
                })

        # Sort by improvement descending
        history.sort(key=lambda x: x.get("improvement", 0), reverse=True)

        return history


# Convenience function for quick search
def search_past_experiments(query: str, top_k: int = 3) -> List[Dict]:
    """
    Quick search for past experiments.

    Args:
        query: Search query
        top_k: Maximum results

    Returns:
        List of matching experiments
    """
    return MemorySearch().search_memory(query, top_k=top_k)


if __name__ == "__main__":
    # Test the memory search
    ms = MemorySearch()

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
