"""
Unit tests for the memory search module.

Tests session search, metric-based search, and improvement history retrieval.
"""

from datetime import datetime
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from memory.memory_search import (
    AsyncMemorySearch,
    MemorySearch,
)


class TestMemorySearch:
    """Tests for MemorySearch class."""

    @pytest.fixture
    def mock_db(self):
        """Create a mock database session."""
        db = MagicMock()
        return db

    @pytest.fixture
    def mock_sessions(self):
        """Create mock session data."""
        mock_exp_state = MagicMock()
        mock_exp_state.to_dict.return_value = {
            "best_accuracy": 0.95,
            "target_accuracy": 0.90,
            "improvement_attempt": 2,
            "stage": "completed",
            "current_accuracy": 0.95,
            "current_config": {"learning_rate": 0.001},
            "improvement_history": [
                {
                    "attempt": 1,
                    "config_changes": {"learning_rate": 0.01},
                    "accuracy_before": 0.80,
                    "accuracy_after": 0.90,
                    "timestamp": "2024-01-01T00:00:00",
                },
                {
                    "attempt": 2,
                    "config_changes": {"learning_rate": 0.001},
                    "accuracy_before": 0.90,
                    "accuracy_after": 0.95,
                    "timestamp": "2024-01-02T00:00:00",
                },
            ],
            "experiment_name": "test_experiment",
        }

        mock_session1 = MagicMock()
        mock_session1.session_id = "session-001"
        mock_session1.original_query = "Train a cat classifier"
        mock_session1.result = "Successfully trained model with 95% accuracy"
        mock_session1.status = "completed"
        mock_session1.created_at = datetime(2024, 1, 1, 12, 0, 0)
        mock_session1.project_path = "/path/to/project"
        mock_session1.experiment_state = mock_exp_state

        mock_session2 = MagicMock()
        mock_session2.session_id = "session-002"
        mock_session2.original_query = "Set up MLflow tracking"
        mock_session2.result = "MLflow configured successfully"
        mock_session2.status = "completed"
        mock_session2.created_at = datetime(2024, 1, 2, 12, 0, 0)
        mock_session2.project_path = "/path/to/project2"
        mock_session2.experiment_state = None

        return [mock_session1, mock_session2]

    def test_init_with_db(self, mock_db):
        """Test MemorySearch initialization with database."""
        ms = MemorySearch(db=mock_db)
        assert ms.db == mock_db
        assert ms._index_data is None

    def test_init_without_db(self):
        """Test MemorySearch initialization without database."""
        ms = MemorySearch(db=None)
        assert ms.db is None
        assert ms._index_data is None

    def test_load_sessions_no_db(self):
        """Test loading sessions when no database provided."""
        ms = MemorySearch(db=None)
        sessions = ms._load_sessions()
        assert sessions == []

    def test_load_sessions_with_db(self, mock_db, mock_sessions):
        """Test loading sessions from database."""
        mock_result = MagicMock()
        mock_result.scalars.return_value.all.return_value = mock_sessions
        mock_db.execute.return_value = mock_result

        ms = MemorySearch(db=mock_db)
        sessions = ms._load_sessions()

        assert len(sessions) == 2
        assert sessions[0]["session_id"] == "session-001"
        assert sessions[0]["original_query"] == "Train a cat classifier"
        assert sessions[0]["goal_achieved"] is True
        assert sessions[1]["session_id"] == "session-002"

    def test_index_data_lazy_loading(self, mock_db, mock_sessions):
        """Test lazy loading of index data."""
        mock_result = MagicMock()
        mock_result.scalars.return_value.all.return_value = mock_sessions
        mock_db.execute.return_value = mock_result

        ms = MemorySearch(db=mock_db)
        # First access triggers load
        data = ms.index_data
        assert len(data) == 2

        # Second access uses cached data
        data2 = ms.index_data
        assert data2 is data

    def test_search_memory_empty(self):
        """Test search with no data."""
        ms = MemorySearch(db=None)
        ms._index_data = []
        results = ms.search_memory("test query")
        assert results == []

    @patch("memory.memory_search.RAPIDFUZZ_AVAILABLE", True)
    @patch("memory.memory_search.fuzz")
    def test_search_memory_with_rapidfuzz(self, mock_fuzz, mock_db, mock_sessions):
        """Test search using rapidfuzz."""
        mock_fuzz.token_set_ratio.return_value = 80

        mock_result = MagicMock()
        mock_result.scalars.return_value.all.return_value = mock_sessions
        mock_db.execute.return_value = mock_result

        ms = MemorySearch(db=mock_db)
        results = ms.search_memory("cat classifier", top_k=3, threshold=40)

        assert len(results) > 0
        assert "score" in results[0]
        assert "session_id" in results[0]

    @patch("memory.memory_search.RAPIDFUZZ_AVAILABLE", False)
    def test_search_memory_fallback(self, mock_db, mock_sessions):
        """Test search using fallback method."""
        mock_result = MagicMock()
        mock_result.scalars.return_value.all.return_value = mock_sessions
        mock_db.execute.return_value = mock_result

        ms = MemorySearch(db=mock_db)
        results = ms.search_memory("cat", top_k=3)

        # Should use simple search
        assert isinstance(results, list)

    def test_simple_search(self):
        """Test simple substring search fallback."""
        ms = MemorySearch(db=None)
        ms._index_data = [
            {
                "session_id": "s1",
                "original_query": "train a cat classifier",
                "final_summary": "completed",
                "timestamp": "2024-01-01",
                "goal_achieved": True,
                "experiment_state": {},
            },
            {
                "session_id": "s2",
                "original_query": "setup mlflow",
                "final_summary": "done",
                "timestamp": "2024-01-02",
                "goal_achieved": True,
                "experiment_state": {},
            },
        ]

        results = ms._simple_search("cat", top_k=3)
        assert len(results) >= 1
        assert results[0]["session_id"] == "s1"

    def test_format_result(self):
        """Test result formatting."""
        ms = MemorySearch(db=None)
        entry = {
            "session_id": "test-session",
            "original_query": "test query",
            "final_summary": "test summary",
            "timestamp": "2024-01-01T00:00:00",
            "goal_achieved": True,
            "experiment_state": {
                "best_accuracy": 0.95,
                "target_accuracy": 0.90,
                "improvement_attempt": 2,
                "stage": "completed",
            },
        }

        result = ms._format_result(85, entry)

        assert result["score"] == 85
        assert result["session_id"] == "test-session"
        assert result["original_query"] == "test query"
        assert result["goal_achieved"] is True
        assert result["best_accuracy"] == 0.95
        assert result["improvement_attempts"] == 2

    def test_search_by_metric_accuracy(self):
        """Test searching by accuracy metric."""
        ms = MemorySearch(db=None)
        ms._index_data = [
            {
                "session_id": "s1",
                "original_query": "query1",
                "final_summary": "summary1",
                "timestamp": "2024-01-01",
                "goal_achieved": True,
                "experiment_state": {"best_accuracy": 0.95},
            },
            {
                "session_id": "s2",
                "original_query": "query2",
                "final_summary": "summary2",
                "timestamp": "2024-01-02",
                "goal_achieved": True,
                "experiment_state": {"best_accuracy": 0.80},
            },
            {
                "session_id": "s3",
                "original_query": "query3",
                "final_summary": "summary3",
                "timestamp": "2024-01-03",
                "goal_achieved": True,
                "experiment_state": {"current_accuracy": 0.85},
            },
        ]

        results = ms.search_by_metric("accuracy", min_value=0.85, top_k=5)
        assert len(results) == 2
        # Should be sorted by value descending
        assert results[0]["metric_value"] == 0.95

    def test_search_by_metric_loss(self):
        """Test searching by loss metric."""
        ms = MemorySearch(db=None)
        ms._index_data = [
            {
                "session_id": "s1",
                "original_query": "query1",
                "final_summary": "summary1",
                "timestamp": "2024-01-01",
                "goal_achieved": True,
                "experiment_state": {"current_loss": 0.1},
            },
        ]

        results = ms.search_by_metric("loss", min_value=0.05, top_k=5)
        assert len(results) == 1
        assert results[0]["metric_value"] == 0.1

    def test_search_by_config(self):
        """Test searching by configuration."""
        ms = MemorySearch(db=None)
        ms._index_data = [
            {
                "session_id": "s1",
                "original_query": "query1",
                "final_summary": "summary1",
                "timestamp": "2024-01-01",
                "goal_achieved": True,
                "experiment_state": {"current_config": {"learning_rate": 0.001, "batch_size": 32}},
            },
            {
                "session_id": "s2",
                "original_query": "query2",
                "final_summary": "summary2",
                "timestamp": "2024-01-02",
                "goal_achieved": True,
                "experiment_state": {"current_config": {"learning_rate": 0.01, "batch_size": 64}},
            },
        ]

        results = ms.search_by_config("learning_rate", 0.001)
        assert len(results) == 1
        assert results[0]["session_id"] == "s1"

    def test_search_by_config_no_match(self):
        """Test config search with no matches."""
        ms = MemorySearch(db=None)
        ms._index_data = [
            {
                "session_id": "s1",
                "original_query": "query1",
                "final_summary": "summary1",
                "timestamp": "2024-01-01",
                "goal_achieved": True,
                "experiment_state": {"current_config": {}},
            },
        ]

        results = ms.search_by_config("nonexistent_key", "value")
        assert len(results) == 0

    def test_get_improvement_history(self):
        """Test retrieving improvement history."""
        ms = MemorySearch(db=None)
        ms._index_data = [
            {
                "session_id": "s1",
                "original_query": "query1",
                "final_summary": "summary1",
                "timestamp": "2024-01-01",
                "goal_achieved": True,
                "experiment_state": {
                    "experiment_name": "exp1",
                    "improvement_history": [
                        {
                            "attempt": 1,
                            "config_changes": {"lr": 0.01},
                            "accuracy_before": 0.80,
                            "accuracy_after": 0.90,
                            "timestamp": "2024-01-01",
                        },
                        {
                            "attempt": 2,
                            "config_changes": {"lr": 0.001},
                            "accuracy_before": 0.90,
                            "accuracy_after": 0.95,
                            "timestamp": "2024-01-02",
                        },
                    ],
                },
            },
        ]

        history = ms.get_improvement_history()
        assert len(history) == 2
        # Sorted by improvement (descending)
        assert history[0]["improvement"] > history[1]["improvement"]

    def test_get_improvement_history_filtered(self):
        """Test retrieving improvement history filtered by experiment."""
        ms = MemorySearch(db=None)
        ms._index_data = [
            {
                "session_id": "s1",
                "original_query": "query1",
                "final_summary": "summary1",
                "timestamp": "2024-01-01",
                "goal_achieved": True,
                "experiment_state": {
                    "experiment_name": "exp1",
                    "improvement_history": [
                        {
                            "attempt": 1,
                            "config_changes": {},
                            "accuracy_before": 0.80,
                            "accuracy_after": 0.90,
                        }
                    ],
                },
            },
            {
                "session_id": "s2",
                "original_query": "query2",
                "final_summary": "summary2",
                "timestamp": "2024-01-02",
                "goal_achieved": True,
                "experiment_state": {
                    "experiment_name": "exp2",
                    "improvement_history": [
                        {
                            "attempt": 1,
                            "config_changes": {},
                            "accuracy_before": 0.70,
                            "accuracy_after": 0.85,
                        }
                    ],
                },
            },
        ]

        history = ms.get_improvement_history(experiment_name="exp1")
        assert len(history) == 1
        assert history[0]["experiment_name"] == "exp1"


class TestAsyncMemorySearch:
    """Tests for AsyncMemorySearch class."""

    @pytest.fixture
    def mock_async_db(self):
        """Create a mock async database session."""
        db = AsyncMock()
        return db

    @pytest.fixture
    def mock_sessions(self):
        """Create mock session data."""
        mock_exp_state = MagicMock()
        mock_exp_state.to_dict.return_value = {
            "best_accuracy": 0.95,
            "target_accuracy": 0.90,
            "improvement_attempt": 2,
            "stage": "completed",
        }

        mock_session = MagicMock()
        mock_session.session_id = "session-001"
        mock_session.original_query = "Train a cat classifier"
        mock_session.result = "Successfully trained model"
        mock_session.status = "completed"
        mock_session.created_at = datetime(2024, 1, 1, 12, 0, 0)
        mock_session.project_path = "/path/to/project"
        mock_session.experiment_state = mock_exp_state

        return [mock_session]

    def test_init_with_db(self, mock_async_db):
        """Test AsyncMemorySearch initialization."""
        ms = AsyncMemorySearch(db=mock_async_db)
        assert ms.db == mock_async_db
        assert ms._index_data is None

    def test_index_data_before_load(self):
        """Test index_data returns empty list before loading."""
        ms = AsyncMemorySearch(db=None)
        assert ms.index_data == []

    @pytest.mark.asyncio
    async def test_load_sessions_no_db(self):
        """Test loading sessions without database."""
        ms = AsyncMemorySearch(db=None)
        sessions = await ms._load_sessions()
        assert sessions == []

    @pytest.mark.asyncio
    async def test_load_sessions_with_db(self, mock_async_db, mock_sessions):
        """Test loading sessions from async database."""
        mock_result = MagicMock()
        mock_result.scalars.return_value.all.return_value = mock_sessions
        mock_async_db.execute.return_value = mock_result

        ms = AsyncMemorySearch(db=mock_async_db)
        sessions = await ms._load_sessions()

        assert len(sessions) == 1
        assert sessions[0]["session_id"] == "session-001"

    @pytest.mark.asyncio
    async def test_load_data(self, mock_async_db, mock_sessions):
        """Test explicit data loading."""
        mock_result = MagicMock()
        mock_result.scalars.return_value.all.return_value = mock_sessions
        mock_async_db.execute.return_value = mock_result

        ms = AsyncMemorySearch(db=mock_async_db)
        await ms.load_data()

        assert len(ms.index_data) == 1

    @pytest.mark.asyncio
    async def test_search_memory_empty(self):
        """Test async search with no data."""
        ms = AsyncMemorySearch(db=None)
        ms._index_data = []
        results = await ms.search_memory("test query")
        assert results == []

    @pytest.mark.asyncio
    @patch("memory.memory_search.RAPIDFUZZ_AVAILABLE", True)
    @patch("memory.memory_search.fuzz")
    async def test_search_memory_with_rapidfuzz(self, mock_fuzz, mock_async_db, mock_sessions):
        """Test async search using rapidfuzz."""
        mock_fuzz.token_set_ratio.return_value = 80

        mock_result = MagicMock()
        mock_result.scalars.return_value.all.return_value = mock_sessions
        mock_async_db.execute.return_value = mock_result

        ms = AsyncMemorySearch(db=mock_async_db)
        results = await ms.search_memory("cat classifier", top_k=3, threshold=40)

        assert len(results) > 0

    @pytest.mark.asyncio
    @patch("memory.memory_search.RAPIDFUZZ_AVAILABLE", False)
    async def test_search_memory_fallback(self, mock_async_db, mock_sessions):
        """Test async search using fallback method."""
        mock_result = MagicMock()
        mock_result.scalars.return_value.all.return_value = mock_sessions
        mock_async_db.execute.return_value = mock_result

        ms = AsyncMemorySearch(db=mock_async_db)
        results = await ms.search_memory("cat", top_k=3)

        assert isinstance(results, list)

    def test_simple_search(self):
        """Test simple substring search."""
        ms = AsyncMemorySearch(db=None)
        ms._index_data = [
            {
                "session_id": "s1",
                "original_query": "train a cat classifier",
                "final_summary": "completed",
                "timestamp": "2024-01-01",
                "goal_achieved": True,
                "experiment_state": {},
            },
        ]

        results = ms._simple_search("cat", top_k=3)
        assert len(results) == 1

    def test_format_result(self):
        """Test result formatting."""
        ms = AsyncMemorySearch(db=None)
        entry = {
            "session_id": "test-session",
            "original_query": "test query",
            "final_summary": "test summary",
            "timestamp": "2024-01-01",
            "goal_achieved": True,
            "experiment_state": {
                "best_accuracy": 0.95,
                "improvement_attempt": 1,
                "stage": "completed",
            },
        }

        result = ms._format_result(90, entry)
        assert result["score"] == 90
        assert result["session_id"] == "test-session"

    @pytest.mark.asyncio
    async def test_search_by_metric(self, mock_async_db, mock_sessions):
        """Test async metric-based search."""
        mock_result = MagicMock()
        mock_result.scalars.return_value.all.return_value = mock_sessions
        mock_async_db.execute.return_value = mock_result

        ms = AsyncMemorySearch(db=mock_async_db)
        results = await ms.search_by_metric("accuracy", min_value=0.90, top_k=5)

        assert isinstance(results, list)

    @pytest.mark.asyncio
    async def test_search_by_config(self, mock_async_db, mock_sessions):
        """Test async config-based search."""
        mock_sessions[0].experiment_state.to_dict.return_value = {
            "current_config": {"learning_rate": 0.001}
        }
        mock_result = MagicMock()
        mock_result.scalars.return_value.all.return_value = mock_sessions
        mock_async_db.execute.return_value = mock_result

        ms = AsyncMemorySearch(db=mock_async_db)
        results = await ms.search_by_config("learning_rate", 0.001)

        assert isinstance(results, list)

    @pytest.mark.asyncio
    async def test_get_improvement_history(self, mock_async_db, mock_sessions):
        """Test async improvement history retrieval."""
        mock_sessions[0].experiment_state.to_dict.return_value = {
            "experiment_name": "exp1",
            "improvement_history": [
                {
                    "attempt": 1,
                    "config_changes": {},
                    "accuracy_before": 0.80,
                    "accuracy_after": 0.90,
                }
            ],
        }
        mock_result = MagicMock()
        mock_result.scalars.return_value.all.return_value = mock_sessions
        mock_async_db.execute.return_value = mock_result

        ms = AsyncMemorySearch(db=mock_async_db)
        history = await ms.get_improvement_history()

        assert isinstance(history, list)


class TestSearchPastExperiments:
    """Tests for convenience functions."""

    @patch("db.get_session")
    def test_search_past_experiments(self, mock_get_session):
        """Test synchronous search convenience function."""
        from memory.memory_search import search_past_experiments

        mock_db = MagicMock()
        mock_db.__enter__ = MagicMock(return_value=mock_db)
        mock_db.__exit__ = MagicMock(return_value=False)

        mock_result = MagicMock()
        mock_result.scalars.return_value.all.return_value = []
        mock_db.execute.return_value = mock_result

        mock_get_session.return_value = mock_db

        results = search_past_experiments("test query", top_k=3)
        assert results == []

    @pytest.mark.asyncio
    @patch("memory.memory_search.get_async_session")
    async def test_async_search_past_experiments(self, mock_get_async_session):
        """Test asynchronous search convenience function."""
        from memory.memory_search import async_search_past_experiments

        mock_db = AsyncMock()

        mock_result = MagicMock()
        mock_result.scalars.return_value.all.return_value = []
        mock_db.execute.return_value = mock_result

        mock_context = AsyncMock()
        mock_context.__aenter__.return_value = mock_db
        mock_context.__aexit__.return_value = None

        mock_get_async_session.return_value = mock_context

        results = await async_search_past_experiments("test query", top_k=3)
        assert results == []


class TestMemorySearchEdgeCases:
    """Tests for edge cases and error handling."""

    def test_format_result_empty_summary(self):
        """Test formatting with empty summary."""
        ms = MemorySearch(db=None)
        entry = {
            "session_id": "test",
            "original_query": "query",
            "final_summary": "",
            "timestamp": "2024-01-01",
            "goal_achieved": False,
            "experiment_state": {},
        }

        result = ms._format_result(50, entry)
        assert result["summary"] == ""

    def test_format_result_long_summary(self):
        """Test formatting with long summary."""
        ms = MemorySearch(db=None)
        long_summary = "x" * 1000
        entry = {
            "session_id": "test",
            "original_query": "query",
            "final_summary": long_summary,
            "timestamp": "2024-01-01",
            "goal_achieved": True,
            "experiment_state": {},
        }

        result = ms._format_result(50, entry)
        assert len(result["summary"]) <= 500

    def test_search_by_metric_no_metric_data(self):
        """Test metric search with missing metric data."""
        ms = MemorySearch(db=None)
        ms._index_data = [
            {
                "session_id": "s1",
                "original_query": "query",
                "final_summary": "summary",
                "timestamp": "2024-01-01",
                "goal_achieved": True,
                "experiment_state": {},  # No accuracy
            },
        ]

        results = ms.search_by_metric("accuracy", min_value=0.80)
        assert len(results) == 0

    def test_search_by_metric_high_threshold(self):
        """Test metric search with high threshold."""
        ms = MemorySearch(db=None)
        ms._index_data = [
            {
                "session_id": "s1",
                "original_query": "query",
                "final_summary": "summary",
                "timestamp": "2024-01-01",
                "goal_achieved": True,
                "experiment_state": {"best_accuracy": 0.70},
            },
        ]

        results = ms.search_by_metric("accuracy", min_value=0.99)
        assert len(results) == 0

    def test_get_improvement_history_empty(self):
        """Test improvement history with no history."""
        ms = MemorySearch(db=None)
        ms._index_data = [
            {
                "session_id": "s1",
                "original_query": "query",
                "final_summary": "summary",
                "timestamp": "2024-01-01",
                "goal_achieved": True,
                "experiment_state": {"improvement_history": []},
            },
        ]

        history = ms.get_improvement_history()
        assert len(history) == 0

    @patch("memory.memory_search.RAPIDFUZZ_AVAILABLE", True)
    @patch("memory.memory_search.fuzz")
    def test_search_boost_for_achieved_goal(self, mock_fuzz):
        """Test score boost for achieved goals."""
        mock_fuzz.token_set_ratio.return_value = 50

        ms = MemorySearch(db=None)
        ms._index_data = [
            {
                "session_id": "s1",
                "original_query": "test query",
                "final_summary": "summary",
                "timestamp": "2024-01-01",
                "goal_achieved": True,  # Should get boost
                "experiment_state": {},
            },
            {
                "session_id": "s2",
                "original_query": "test query",
                "final_summary": "summary",
                "timestamp": "2024-01-02",
                "goal_achieved": False,  # No boost
                "experiment_state": {},
            },
        ]

        results = ms.search_memory("test", threshold=30)
        # Both should match but achieved goal should score higher
        assert len(results) >= 1
