#!/usr/bin/env python3
"""
Tests for the database-backed memory search module.

Run with: pytest tests/root_migrated/test_memory_search.py -v
"""

import pytest

from db import close_db, get_session, init_db
from db.repositories import SessionRepository
from db.session import close_async_db, get_async_session, init_async_db
from memory.memory_search import (
    AsyncMemorySearch,
    MemorySearch,
    async_search_past_experiments,
    search_past_experiments,
)

# ============================================================================
# Test Fixtures
# ============================================================================


@pytest.fixture(autouse=True)
def clean_db_state():
    """Clean up database state before and after each test."""
    close_db()
    yield
    close_db()


@pytest.fixture
def temp_db_path(tmp_path):
    """Create a temporary database file path."""
    return str(tmp_path / "test_memory_search.db")


@pytest.fixture
def temp_db_url(temp_db_path):
    """Create a temporary database URL."""
    return f"sqlite:///{temp_db_path}"


@pytest.fixture
def configured_db(temp_db_url, monkeypatch):
    """Configure and initialize a temporary database."""
    monkeypatch.setenv("DATABASE_URL", temp_db_url)
    close_db()
    init_db()
    yield
    close_db()


@pytest.fixture
def db_with_sessions(configured_db):
    """Create test sessions in database."""
    with get_session() as db:
        repo = SessionRepository(db)

        # Session 1: Completed ML pipeline setup
        session1 = repo.create_session(
            session_id="test-session-1",
            original_query="Set up MLOps pipeline for cat-dog classifier",
            project_path="/projects/cat-dog",
        )
        session1.status = "completed"
        session1.result = "Successfully configured MLOps pipeline with MLflow tracking."
        exp1 = repo.create_experiment_state(
            agent_session_id=session1.id,
            experiment_name="cat-dog-classifier",
            target_accuracy=0.90,
        )
        exp1.current_accuracy = 0.92
        exp1.best_accuracy = 0.92
        exp1.stage = "deploy"
        exp1.current_config = {"learning_rate": 0.001, "batch_size": 32}

        # Session 2: Incomplete training
        session2 = repo.create_session(
            session_id="test-session-2",
            original_query="Train image classification model",
            project_path="/projects/images",
        )
        session2.status = "active"
        session2.result = "Training in progress..."
        exp2 = repo.create_experiment_state(
            agent_session_id=session2.id,
            experiment_name="image-classifier",
            target_accuracy=0.85,
        )
        exp2.current_accuracy = 0.75
        exp2.best_accuracy = 0.75
        exp2.stage = "training"
        exp2.current_config = {"learning_rate": 0.01, "batch_size": 64}
        exp2.improvement_history = [
            {
                "attempt": 1,
                "config_changes": {"learning_rate": 0.005},
                "accuracy_before": 0.70,
                "accuracy_after": 0.75,
                "timestamp": "2024-01-15T10:00:00",
            }
        ]

        # Session 3: Failed session
        session3 = repo.create_session(
            session_id="test-session-3",
            original_query="Deploy model to production",
            project_path="/projects/deploy",
        )
        session3.status = "failed"
        session3.result = "Deployment failed due to configuration error."

    yield


# ============================================================================
# MemorySearch Class Tests
# ============================================================================


class TestMemorySearch:
    """Tests for MemorySearch class with database backend."""

    def test_init_without_db(self):
        """Test initialization without database session."""
        ms = MemorySearch()
        assert ms.db is None
        assert ms.index_data == []

    def test_init_with_db(self, configured_db):
        """Test initialization with database session."""
        with get_session() as db:
            ms = MemorySearch(db)
            assert ms.db is not None

    def test_load_sessions_empty_db(self, configured_db):
        """Test loading sessions from empty database."""
        with get_session() as db:
            ms = MemorySearch(db)
            assert ms.index_data == []

    def test_load_sessions_with_data(self, db_with_sessions):
        """Test loading sessions with data in database."""
        with get_session() as db:
            ms = MemorySearch(db)
            assert len(ms.index_data) == 3

    def test_session_entry_structure(self, db_with_sessions):
        """Test that loaded session entries have correct structure."""
        with get_session() as db:
            ms = MemorySearch(db)
            entry = ms.index_data[0]

            assert "session_id" in entry
            assert "original_query" in entry
            assert "final_summary" in entry
            assert "status" in entry
            assert "goal_achieved" in entry
            assert "timestamp" in entry
            assert "experiment_state" in entry
            assert "project_path" in entry


class TestMemorySearchSearch:
    """Tests for MemorySearch search methods."""

    def test_search_memory_empty(self, configured_db):
        """Test searching empty database."""
        with get_session() as db:
            ms = MemorySearch(db)
            results = ms.search_memory("any query")
            assert results == []

    def test_search_memory_finds_match(self, db_with_sessions):
        """Test searching finds matching sessions."""
        with get_session() as db:
            ms = MemorySearch(db)
            results = ms.search_memory("cat-dog classifier", top_k=5)
            assert len(results) > 0
            assert any("cat-dog" in r["original_query"].lower() for r in results)

    def test_search_memory_respects_top_k(self, db_with_sessions):
        """Test that search respects top_k parameter."""
        with get_session() as db:
            ms = MemorySearch(db)
            results = ms.search_memory("model", top_k=1)
            assert len(results) <= 1

    def test_search_memory_respects_threshold(self, db_with_sessions):
        """Test that search respects threshold parameter."""
        with get_session() as db:
            ms = MemorySearch(db)
            high_threshold_results = ms.search_memory("xyz123abc", threshold=90)
            assert len(high_threshold_results) == 0

    def test_search_memory_result_structure(self, db_with_sessions):
        """Test search result structure."""
        with get_session() as db:
            ms = MemorySearch(db)
            results = ms.search_memory("classifier", threshold=30)
            if results:
                result = results[0]
                assert "score" in result
                assert "session_id" in result
                assert "original_query" in result
                assert "summary" in result
                assert "timestamp" in result
                assert "goal_achieved" in result
                assert "best_accuracy" in result
                assert "target_accuracy" in result
                assert "improvement_attempts" in result
                assert "stage" in result

    def test_search_memory_boosts_successful_sessions(self, db_with_sessions):
        """Test that successful sessions get a score boost."""
        with get_session() as db:
            ms = MemorySearch(db)
            results = ms.search_memory("pipeline", threshold=30)
            if len(results) > 1:
                completed_sessions = [r for r in results if r["goal_achieved"]]
                incomplete_sessions = [r for r in results if not r["goal_achieved"]]
                if completed_sessions and incomplete_sessions:
                    assert completed_sessions[0]["score"] >= incomplete_sessions[0]["score"] - 10


class TestMemorySearchByMetric:
    """Tests for search_by_metric method."""

    def test_search_by_accuracy(self, db_with_sessions):
        """Test searching by accuracy metric."""
        with get_session() as db:
            ms = MemorySearch(db)
            results = ms.search_by_metric(metric_name="accuracy", min_value=0.8)
            assert len(results) > 0
            assert all(r["metric_value"] >= 0.8 for r in results)

    def test_search_by_accuracy_sorted(self, db_with_sessions):
        """Test that results are sorted by metric descending."""
        with get_session() as db:
            ms = MemorySearch(db)
            results = ms.search_by_metric(metric_name="accuracy", min_value=0.0)
            if len(results) > 1:
                for i in range(len(results) - 1):
                    assert results[i]["metric_value"] >= results[i + 1]["metric_value"]

    def test_search_by_metric_respects_top_k(self, db_with_sessions):
        """Test that search_by_metric respects top_k."""
        with get_session() as db:
            ms = MemorySearch(db)
            results = ms.search_by_metric(metric_name="accuracy", min_value=0.0, top_k=1)
            assert len(results) <= 1

    def test_search_by_metric_no_results(self, db_with_sessions):
        """Test search_by_metric with high threshold returns empty."""
        with get_session() as db:
            ms = MemorySearch(db)
            results = ms.search_by_metric(metric_name="accuracy", min_value=0.99)
            assert len(results) == 0


class TestMemorySearchByConfig:
    """Tests for search_by_config method."""

    def test_search_by_config_finds_match(self, db_with_sessions):
        """Test searching by configuration value."""
        with get_session() as db:
            ms = MemorySearch(db)
            results = ms.search_by_config("learning_rate", 0.001)
            assert len(results) > 0

    def test_search_by_config_no_match(self, db_with_sessions):
        """Test searching by non-existent config returns empty."""
        with get_session() as db:
            ms = MemorySearch(db)
            results = ms.search_by_config("nonexistent_key", "value")
            assert len(results) == 0


class TestMemorySearchImprovementHistory:
    """Tests for get_improvement_history method."""

    def test_get_improvement_history(self, db_with_sessions):
        """Test getting improvement history."""
        with get_session() as db:
            ms = MemorySearch(db)
            history = ms.get_improvement_history()
            assert len(history) > 0
            assert all("attempt" in h for h in history)
            assert all("config_changes" in h for h in history)

    def test_get_improvement_history_by_experiment(self, db_with_sessions):
        """Test filtering improvement history by experiment name."""
        with get_session() as db:
            ms = MemorySearch(db)
            history = ms.get_improvement_history(experiment_name="image-classifier")
            assert len(history) > 0
            assert all(h["experiment_name"] == "image-classifier" for h in history)

    def test_get_improvement_history_sorted_by_improvement(self, db_with_sessions):
        """Test that improvement history is sorted by improvement."""
        with get_session() as db:
            ms = MemorySearch(db)
            history = ms.get_improvement_history()
            if len(history) > 1:
                for i in range(len(history) - 1):
                    assert history[i]["improvement"] >= history[i + 1]["improvement"]


# ============================================================================
# AsyncMemorySearch Tests
# ============================================================================


@pytest.fixture
def async_configured_db(temp_db_url, monkeypatch):
    """Configure and initialize a temporary async database."""
    import asyncio

    monkeypatch.setenv("DATABASE_URL", temp_db_url)
    asyncio.get_event_loop().run_until_complete(close_async_db())
    asyncio.get_event_loop().run_until_complete(init_async_db())
    yield
    asyncio.get_event_loop().run_until_complete(close_async_db())


@pytest.fixture
def async_db_with_sessions(async_configured_db, temp_db_url, monkeypatch):
    """Create test sessions in async database."""
    monkeypatch.setenv("DATABASE_URL", temp_db_url)
    close_db()
    init_db()

    with get_session() as db:
        repo = SessionRepository(db)

        session1 = repo.create_session(
            session_id="async-test-session-1",
            original_query="Set up async MLOps pipeline",
            project_path="/projects/async",
        )
        session1.status = "completed"
        session1.result = "Async pipeline configured successfully."
        exp1 = repo.create_experiment_state(
            agent_session_id=session1.id,
            experiment_name="async-classifier",
            target_accuracy=0.85,
        )
        exp1.current_accuracy = 0.88
        exp1.best_accuracy = 0.88
        exp1.stage = "evaluation"

    close_db()
    yield


class TestAsyncMemorySearch:
    """Tests for AsyncMemorySearch class."""

    @pytest.fixture(autouse=True)
    def cleanup(self):
        """Clean up async database state."""
        import asyncio

        asyncio.get_event_loop().run_until_complete(close_async_db())
        yield
        asyncio.get_event_loop().run_until_complete(close_async_db())

    @pytest.mark.asyncio
    async def test_init_without_db(self):
        """Test async initialization without database session."""
        ms = AsyncMemorySearch()
        assert ms.db is None
        assert ms.index_data == []

    @pytest.mark.asyncio
    async def test_load_data(self, async_db_with_sessions):
        """Test async data loading."""
        async with get_async_session() as db:
            ms = AsyncMemorySearch(db)
            await ms.load_data()
            assert len(ms.index_data) > 0

    @pytest.mark.asyncio
    async def test_search_memory(self, async_db_with_sessions):
        """Test async search_memory."""
        async with get_async_session() as db:
            ms = AsyncMemorySearch(db)
            results = await ms.search_memory("async pipeline", top_k=5)
            assert len(results) > 0

    @pytest.mark.asyncio
    async def test_search_by_metric(self, async_db_with_sessions):
        """Test async search_by_metric."""
        async with get_async_session() as db:
            ms = AsyncMemorySearch(db)
            results = await ms.search_by_metric(metric_name="accuracy", min_value=0.80)
            assert len(results) > 0

    @pytest.mark.asyncio
    async def test_search_by_config(self, async_db_with_sessions):
        """Test async search_by_config returns empty for non-matching."""
        async with get_async_session() as db:
            ms = AsyncMemorySearch(db)
            results = await ms.search_by_config("nonexistent", "value")
            assert len(results) == 0

    @pytest.mark.asyncio
    async def test_get_improvement_history(self, async_db_with_sessions):
        """Test async get_improvement_history."""
        async with get_async_session() as db:
            ms = AsyncMemorySearch(db)
            history = await ms.get_improvement_history()
            assert isinstance(history, list)


# ============================================================================
# Convenience Function Tests
# ============================================================================


class TestConvenienceFunctions:
    """Tests for convenience search functions."""

    def test_search_past_experiments(self, db_with_sessions):
        """Test sync convenience function."""
        results = search_past_experiments("classifier")
        assert isinstance(results, list)

    def test_search_past_experiments_empty_db(self, configured_db):
        """Test sync convenience function with empty database."""
        results = search_past_experiments("anything")
        assert results == []

    @pytest.mark.asyncio
    async def test_async_search_past_experiments(self, async_db_with_sessions):
        """Test async convenience function."""
        results = await async_search_past_experiments("pipeline")
        assert isinstance(results, list)


# ============================================================================
# Edge Cases and Error Handling
# ============================================================================


class TestEdgeCases:
    """Tests for edge cases and error handling."""

    def test_empty_query(self, db_with_sessions):
        """Test searching with empty query."""
        with get_session() as db:
            ms = MemorySearch(db)
            results = ms.search_memory("")
            assert isinstance(results, list)

    def test_special_characters_in_query(self, db_with_sessions):
        """Test searching with special characters."""
        with get_session() as db:
            ms = MemorySearch(db)
            results = ms.search_memory("test@#$%^&*()")
            assert isinstance(results, list)

    def test_very_long_query(self, db_with_sessions):
        """Test searching with very long query."""
        with get_session() as db:
            ms = MemorySearch(db)
            long_query = "a" * 1000
            results = ms.search_memory(long_query)
            assert isinstance(results, list)

    def test_unicode_query(self, db_with_sessions):
        """Test searching with unicode characters."""
        with get_session() as db:
            ms = MemorySearch(db)
            results = ms.search_memory("测试模型 トレーニング")
            assert isinstance(results, list)

    def test_negative_top_k(self, db_with_sessions):
        """Test with negative top_k value."""
        with get_session() as db:
            ms = MemorySearch(db)
            results = ms.search_memory("test", top_k=-1)
            assert isinstance(results, list)

    def test_zero_threshold(self, db_with_sessions):
        """Test with zero threshold (should return more results)."""
        with get_session() as db:
            ms = MemorySearch(db)
            results = ms.search_memory("model", threshold=0)
            assert isinstance(results, list)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
