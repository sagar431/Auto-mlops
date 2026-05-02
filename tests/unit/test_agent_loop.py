#!/usr/bin/env python3
"""
Tests for agent/agent_loop.py - MLOps agent orchestration loop.

Run with: pytest tests/unit/test_agent_loop.py -v
"""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from agent.agent_loop import (
    AgentLoop,
    Route,
    StepExecutionError,
    StepExecutionTracker,
    StepType,
    run_mlops_agent,
)
from workflow.registry import WorkflowStatus

# ============================================================================
# Route Constants Tests
# ============================================================================


class TestRouteConstants:
    """Tests for Route constants."""

    def test_route_summarize(self):
        """Test SUMMARIZE route constant."""
        assert Route.SUMMARIZE == "summarize"

    def test_route_decision(self):
        """Test DECISION route constant."""
        assert Route.DECISION == "decision"

    def test_route_improve(self):
        """Test IMPROVE route constant."""
        assert Route.IMPROVE == "improve"

    def test_route_deploy(self):
        """Test DEPLOY route constant."""
        assert Route.DEPLOY == "deploy"


# ============================================================================
# StepType Constants Tests
# ============================================================================


class TestStepTypeConstants:
    """Tests for StepType constants."""

    def test_step_type_root(self):
        """Test ROOT step type constant."""
        assert StepType.ROOT == "ROOT"

    def test_step_type_code(self):
        """Test CODE step type constant."""
        assert StepType.CODE == "CODE"

    def test_step_type_improve(self):
        """Test IMPROVE step type constant."""
        assert StepType.IMPROVE == "IMPROVE"

    def test_step_type_deploy(self):
        """Test DEPLOY step type constant."""
        assert StepType.DEPLOY == "DEPLOY"


# ============================================================================
# StepExecutionError Tests
# ============================================================================


class TestStepExecutionError:
    """Tests for StepExecutionError exception."""

    def test_create_step_execution_error(self):
        """Test creating StepExecutionError."""
        error = StepExecutionError("step_1", "Something went wrong")
        assert error.step_id == "step_1"
        assert error.error_message == "Something went wrong"
        assert "step_1" in str(error)
        assert "Something went wrong" in str(error)

    def test_step_execution_error_inheritance(self):
        """Test StepExecutionError is an Exception."""
        error = StepExecutionError("step_1", "Error")
        assert isinstance(error, Exception)

    def test_step_execution_error_message_format(self):
        """Test error message format."""
        error = StepExecutionError("test_step", "Test error")
        expected = "Step 'test_step' failed: Test error"
        assert str(error) == expected


# ============================================================================
# StepExecutionTracker Tests
# ============================================================================


class TestStepExecutionTracker:
    """Tests for StepExecutionTracker class."""

    def test_create_tracker_with_defaults(self):
        """Test creating tracker with default values."""
        tracker = StepExecutionTracker()
        assert tracker.max_steps == 15
        assert tracker.max_retries == 3
        assert tracker.tries == 0
        assert tracker.attempts == {}

    def test_create_tracker_with_custom_values(self):
        """Test creating tracker with custom values."""
        tracker = StepExecutionTracker(max_steps=10, max_retries=5)
        assert tracker.max_steps == 10
        assert tracker.max_retries == 5

    def test_increment(self):
        """Test incrementing tries counter."""
        tracker = StepExecutionTracker()
        assert tracker.tries == 0
        tracker.increment()
        assert tracker.tries == 1
        tracker.increment()
        assert tracker.tries == 2

    def test_record_failure(self):
        """Test recording step failures."""
        tracker = StepExecutionTracker()
        tracker.record_failure("step_1")
        assert tracker.attempts["step_1"] == 1
        tracker.record_failure("step_1")
        assert tracker.attempts["step_1"] == 2
        tracker.record_failure("step_2")
        assert tracker.attempts["step_2"] == 1

    def test_retry_step_id_no_failures(self):
        """Test retry step ID with no failures."""
        tracker = StepExecutionTracker()
        assert tracker.retry_step_id("step_1") == "step_1"

    def test_retry_step_id_with_failures(self):
        """Test retry step ID generation with failures."""
        tracker = StepExecutionTracker()
        tracker.record_failure("step_1")
        assert tracker.retry_step_id("step_1") == "step_1F1"
        tracker.record_failure("step_1")
        assert tracker.retry_step_id("step_1") == "step_1F2"

    def test_should_continue_within_limits(self):
        """Test should_continue when within limits."""
        tracker = StepExecutionTracker(max_steps=3)
        assert tracker.should_continue() is True
        tracker.increment()
        assert tracker.should_continue() is True
        tracker.increment()
        assert tracker.should_continue() is True

    def test_should_continue_at_limit(self):
        """Test should_continue when at limit."""
        tracker = StepExecutionTracker(max_steps=3)
        tracker.increment()
        tracker.increment()
        tracker.increment()
        assert tracker.should_continue() is False

    def test_has_exceeded_retries_within_limit(self):
        """Test has_exceeded_retries when within limit."""
        tracker = StepExecutionTracker(max_retries=2)
        assert tracker.has_exceeded_retries("step_1") is False
        tracker.record_failure("step_1")
        assert tracker.has_exceeded_retries("step_1") is False

    def test_has_exceeded_retries_at_limit(self):
        """Test has_exceeded_retries when at limit."""
        tracker = StepExecutionTracker(max_retries=2)
        tracker.record_failure("step_1")
        tracker.record_failure("step_1")
        assert tracker.has_exceeded_retries("step_1") is True

    def test_is_circuit_open_initially_false(self):
        """Test circuit is initially closed."""
        tracker = StepExecutionTracker()
        assert tracker.is_circuit_open() is False

    def test_get_circuit_stats(self):
        """Test getting circuit stats."""
        tracker = StepExecutionTracker()
        stats = tracker.get_circuit_stats()
        assert "state" in stats
        assert "total_calls" in stats
        assert "successful_calls" in stats
        assert "failed_calls" in stats
        assert "rejected_calls" in stats
        assert "consecutive_failures" in stats
        assert "consecutive_successes" in stats
        assert stats["state"] == "closed"
        assert stats["total_calls"] == 0

    def test_reset_circuit(self):
        """Test resetting circuit breaker."""
        tracker = StepExecutionTracker()
        tracker.circuit_breaker._stats.total_calls = 100
        tracker.circuit_breaker._stats.failed_calls = 50
        tracker.reset_circuit()
        stats = tracker.get_circuit_stats()
        assert stats["total_calls"] == 0
        assert stats["failed_calls"] == 0

    def test_circuit_breaker_property(self):
        """Test circuit breaker property access."""
        tracker = StepExecutionTracker()
        assert tracker.circuit_breaker is not None
        assert tracker.circuit_breaker.name == "step_execution"


# ============================================================================
# AgentLoop Initialization Tests
# ============================================================================


class TestAgentLoopInit:
    """Tests for AgentLoop initialization."""

    @pytest.fixture
    def mock_prompts_dir(self, tmp_path):
        """Create temporary prompts directory with test prompts."""
        prompts_dir = tmp_path / "prompts"
        prompts_dir.mkdir()

        (prompts_dir / "perception_prompt.txt").write_text("Perception: {query}")
        (prompts_dir / "decision_prompt.txt").write_text("Decision: {query}")
        (prompts_dir / "summarizer_prompt.txt").write_text("Summarize: {query}")
        (prompts_dir / "improvement_prompt.txt").write_text(
            "Improve: target={target_accuracy} current={current_accuracy}"
        )

        return str(prompts_dir)

    def test_create_agent_loop_with_defaults(self, mock_prompts_dir):
        """Test creating AgentLoop with default values."""
        agent = AgentLoop(prompts_dir=mock_prompts_dir)
        assert agent.status == "idle"
        assert agent.profile == "default"
        assert agent.on_event is None
        assert agent.tools_module is None

    def test_create_agent_loop_with_custom_profile(self, mock_prompts_dir):
        """Test creating AgentLoop with custom profile."""
        agent = AgentLoop(prompts_dir=mock_prompts_dir, profile="custom")
        assert agent.profile == "custom"

    def test_create_agent_loop_with_on_event(self, mock_prompts_dir):
        """Test creating AgentLoop with event callback."""

        async def callback(event_type, data):
            pass

        agent = AgentLoop(prompts_dir=mock_prompts_dir, on_event=callback)
        assert agent.on_event is callback

    def test_create_agent_loop_with_tools_module(self, mock_prompts_dir):
        """Test creating AgentLoop with custom tools module."""
        mock_tools = MagicMock()
        agent = AgentLoop(prompts_dir=mock_prompts_dir, tools_module=mock_tools)
        assert agent.tools_module is mock_tools

    def test_load_prompt_returns_content(self, mock_prompts_dir, tmp_path):
        """Test _load_prompt returns file content."""
        agent = AgentLoop(prompts_dir=mock_prompts_dir)
        prompt_file = tmp_path / "test.txt"
        prompt_file.write_text("Test content")
        result = agent._load_prompt(prompt_file)
        assert result == "Test content"

    def test_load_prompt_missing_file_returns_empty(self, mock_prompts_dir, tmp_path):
        """Test _load_prompt returns empty string for missing file."""
        agent = AgentLoop(prompts_dir=mock_prompts_dir)
        result = agent._load_prompt(tmp_path / "nonexistent.txt")
        assert result == ""


# ============================================================================
# AgentLoop Session Initialization Tests
# ============================================================================


class TestAgentLoopSessionInit:
    """Tests for AgentLoop session initialization."""

    @pytest.fixture
    def agent(self, tmp_path):
        """Create AgentLoop with test prompts."""
        prompts_dir = tmp_path / "prompts"
        prompts_dir.mkdir()
        (prompts_dir / "perception_prompt.txt").write_text("Perception: {query}")
        (prompts_dir / "decision_prompt.txt").write_text("Decision: {query}")
        (prompts_dir / "summarizer_prompt.txt").write_text("Summarize: {query}")
        (prompts_dir / "improvement_prompt.txt").write_text("Improve")
        return AgentLoop(prompts_dir=str(prompts_dir))

    def test_initialize_session_creates_session_id(self, agent):
        """Test that _initialize_session creates a session ID."""
        agent._initialize_session("Test query", "/test/path", 0.85)
        assert agent.session_id is not None
        assert len(agent.session_id) > 0

    def test_initialize_session_sets_query(self, agent):
        """Test that _initialize_session sets the query."""
        agent._initialize_session("Test query", "/test/path", 0.85)
        assert agent.query == "Test query"

    def test_initialize_session_creates_context(self, agent):
        """Test that _initialize_session creates context manager."""
        agent._initialize_session("Test query", "/test/path", 0.85)
        assert agent.ctx is not None
        assert agent.ctx.project_path == "/test/path"

    def test_initialize_session_sets_accuracy_threshold(self, agent):
        """Test that _initialize_session sets accuracy threshold."""
        agent._initialize_session("Test query", "/test/path", 0.90)
        assert agent.ctx.experiment_state.target_accuracy == 0.90

    def test_initialize_session_creates_agent_session(self, agent):
        """Test that _initialize_session creates AgentSession."""
        agent._initialize_session("Test query", "/test/path", 0.85)
        assert agent.session is not None
        assert agent.session.original_query == "Test query"

    def test_initialize_session_initializes_placeholders(self, agent):
        """Test that _initialize_session initializes placeholders."""
        agent._initialize_session("Test query", "/test/path", 0.85)
        assert agent.p_out == {}
        assert agent.code_variants == {}
        assert agent.next_step_id == "0"
        assert agent.final_output == ""


# ============================================================================
# AgentLoop Routing Tests
# ============================================================================


class TestAgentLoopRouting:
    """Tests for AgentLoop routing logic."""

    @pytest.fixture
    def agent(self, tmp_path):
        """Create AgentLoop with test prompts."""
        prompts_dir = tmp_path / "prompts"
        prompts_dir.mkdir()
        (prompts_dir / "perception_prompt.txt").write_text("Perception")
        (prompts_dir / "decision_prompt.txt").write_text("Decision")
        (prompts_dir / "summarizer_prompt.txt").write_text("Summarize")
        (prompts_dir / "improvement_prompt.txt").write_text("Improve")
        return AgentLoop(prompts_dir=str(prompts_dir))

    def test_should_summarize_true_when_goal_achieved(self, agent):
        """Test _should_summarize returns True when goal achieved."""
        agent.p_out = {"original_goal_achieved": True, "route": "decision"}
        assert agent._should_summarize() is True

    def test_should_summarize_true_when_route_is_summarize(self, agent):
        """Test _should_summarize returns True when route is summarize."""
        agent.p_out = {"original_goal_achieved": False, "route": Route.SUMMARIZE}
        assert agent._should_summarize() is True

    def test_should_summarize_false_when_not_achieved(self, agent):
        """Test _should_summarize returns False when not achieved."""
        agent.p_out = {"original_goal_achieved": False, "route": Route.DECISION}
        assert agent._should_summarize() is False

    def test_needs_improvement_when_below_threshold(self, agent):
        """Test _needs_improvement returns True when below threshold."""
        agent._initialize_session("Test", "/test", 0.90)
        agent.ctx.experiment_state.current_accuracy = 0.80
        agent.ctx.experiment_state.stage = "evaluation"
        assert agent._needs_improvement() is True

    def test_needs_improvement_false_when_at_threshold(self, agent):
        """Test _needs_improvement returns False when at threshold."""
        agent._initialize_session("Test", "/test", 0.85)
        agent.ctx.experiment_state.current_accuracy = 0.85
        agent.ctx.experiment_state.stage = "evaluation"
        assert agent._needs_improvement() is False

    def test_needs_improvement_false_when_no_accuracy(self, agent):
        """Test _needs_improvement returns False when no accuracy recorded."""
        agent._initialize_session("Test", "/test", 0.85)
        agent.ctx.experiment_state.stage = "evaluation"
        assert agent._needs_improvement() is False

    def test_needs_deployment_when_route_is_deploy(self, agent):
        """Test _needs_deployment returns True when route is deploy."""
        agent.p_out = {"route": Route.DEPLOY}
        assert agent._needs_deployment() is True

    def test_needs_deployment_when_stage_is_deploy(self, agent):
        """Test _needs_deployment returns True when stage is deploy."""
        agent.p_out = {"route": Route.DECISION, "pipeline_stage": "deploy"}
        assert agent._needs_deployment() is True

    def test_needs_deployment_when_deployment_target_set(self, agent):
        """Test _needs_deployment returns True when deployment target set."""
        agent.p_out = {"route": Route.DECISION, "entities": {"deployment_target": "gradio"}}
        assert agent._needs_deployment() is True

    def test_needs_deployment_false_when_not_deploying(self, agent):
        """Test _needs_deployment returns False when not deploying."""
        agent.p_out = {"route": Route.DECISION, "pipeline_stage": "training", "entities": {}}
        assert agent._needs_deployment() is False


# ============================================================================
# AgentLoop Event Emission Tests
# ============================================================================


class TestAgentLoopEventEmission:
    """Tests for AgentLoop event emission."""

    @pytest.fixture
    def agent_with_event_handler(self, tmp_path):
        """Create AgentLoop with event handler."""
        prompts_dir = tmp_path / "prompts"
        prompts_dir.mkdir()
        (prompts_dir / "perception_prompt.txt").write_text("Perception")
        (prompts_dir / "decision_prompt.txt").write_text("Decision")
        (prompts_dir / "summarizer_prompt.txt").write_text("Summarize")
        (prompts_dir / "improvement_prompt.txt").write_text("Improve")

        events = []

        async def capture_event(event_type, data):
            events.append({"type": event_type, "data": data})

        agent = AgentLoop(prompts_dir=str(prompts_dir), on_event=capture_event)
        agent._events = events
        return agent

    @pytest.mark.asyncio
    async def test_emit_calls_callback(self, agent_with_event_handler):
        """Test _emit calls the event callback."""
        await agent_with_event_handler._emit("test_event", {"key": "value"})
        assert len(agent_with_event_handler._events) == 1
        assert agent_with_event_handler._events[0]["type"] == "test_event"
        assert agent_with_event_handler._events[0]["data"]["key"] == "value"

    @pytest.mark.asyncio
    async def test_emit_with_empty_data(self, agent_with_event_handler):
        """Test _emit with empty data."""
        await agent_with_event_handler._emit("test_event")
        assert len(agent_with_event_handler._events) == 1
        assert agent_with_event_handler._events[0]["data"] == {}

    @pytest.mark.asyncio
    async def test_emit_handles_callback_error(self, tmp_path):
        """Test _emit handles callback error gracefully."""
        prompts_dir = tmp_path / "prompts"
        prompts_dir.mkdir()
        (prompts_dir / "perception_prompt.txt").write_text("Perception")
        (prompts_dir / "decision_prompt.txt").write_text("Decision")
        (prompts_dir / "summarizer_prompt.txt").write_text("Summarize")
        (prompts_dir / "improvement_prompt.txt").write_text("Improve")

        async def failing_callback(event_type, data):
            raise ValueError("Callback failed")

        agent = AgentLoop(prompts_dir=str(prompts_dir), on_event=failing_callback)
        # Should not raise
        await agent._emit("test_event", {"key": "value"})

    @pytest.mark.asyncio
    async def test_emit_without_callback(self, tmp_path):
        """Test _emit without callback does nothing."""
        prompts_dir = tmp_path / "prompts"
        prompts_dir.mkdir()
        (prompts_dir / "perception_prompt.txt").write_text("Perception")
        (prompts_dir / "decision_prompt.txt").write_text("Decision")
        (prompts_dir / "summarizer_prompt.txt").write_text("Summarize")
        (prompts_dir / "improvement_prompt.txt").write_text("Improve")

        agent = AgentLoop(prompts_dir=str(prompts_dir))
        # Should not raise
        await agent._emit("test_event", {"key": "value"})


# ============================================================================
# AgentLoop Pick Next Step Tests
# ============================================================================


class TestAgentLoopPickNextStep:
    """Tests for AgentLoop _pick_next_step method."""

    @pytest.fixture
    def agent(self, tmp_path):
        """Create AgentLoop with test prompts."""
        prompts_dir = tmp_path / "prompts"
        prompts_dir.mkdir()
        (prompts_dir / "perception_prompt.txt").write_text("Perception")
        (prompts_dir / "decision_prompt.txt").write_text("Decision")
        (prompts_dir / "summarizer_prompt.txt").write_text("Summarize")
        (prompts_dir / "improvement_prompt.txt").write_text("Improve")
        return AgentLoop(prompts_dir=str(prompts_dir))

    def test_pick_next_step_returns_first_pending(self, agent):
        """Test _pick_next_step returns first pending step."""
        agent._initialize_session("Test", "/test", 0.85)
        agent.ctx.add_step("0", "First step", "CODE", from_node="ROOT")
        agent.ctx.add_step("1", "Second step", "CODE", from_node="ROOT")
        result = agent._pick_next_step()
        assert result == "0"

    def test_pick_next_step_skips_completed(self, agent):
        """Test _pick_next_step skips completed steps."""
        agent._initialize_session("Test", "/test", 0.85)
        agent.ctx.add_step("0", "First step", "CODE", from_node="ROOT")
        agent.ctx.add_step("1", "Second step", "CODE", from_node="ROOT")
        agent.ctx.mark_step_completed("0")
        result = agent._pick_next_step()
        assert result == "1"

    def test_pick_next_step_returns_none_when_all_completed(self, agent):
        """Test _pick_next_step returns None when all completed."""
        agent._initialize_session("Test", "/test", 0.85)
        agent.ctx.add_step("0", "First step", "CODE", from_node="ROOT")
        agent.ctx.mark_step_completed("0")
        result = agent._pick_next_step()
        assert result is None

    def test_pick_next_step_returns_none_when_no_steps(self, agent):
        """Test _pick_next_step returns None when no steps."""
        agent._initialize_session("Test", "/test", 0.85)
        result = agent._pick_next_step()
        assert result is None


# ============================================================================
# AgentLoop Run Integration Tests
# ============================================================================


class TestAgentLoopRun:
    """Integration tests for AgentLoop.run method."""

    @pytest.fixture
    def mock_agent(self, tmp_path, mock_llm):
        """Create AgentLoop with mocked dependencies."""
        prompts_dir = tmp_path / "prompts"
        prompts_dir.mkdir()
        (prompts_dir / "perception_prompt.txt").write_text("Perception")
        (prompts_dir / "decision_prompt.txt").write_text("Decision")
        (prompts_dir / "summarizer_prompt.txt").write_text("Summarize")
        (prompts_dir / "improvement_prompt.txt").write_text("Improve")

        agent = AgentLoop(prompts_dir=str(prompts_dir))
        return agent

    @pytest.mark.asyncio
    async def test_run_sets_status_to_running(self, mock_agent, mock_llm):
        """Test run sets status to running."""
        mock_llm.json_response = {
            "entities": {},
            "pipeline_stage": "setup",
            "route": "summarize",
            "original_goal_achieved": True,
            "confidence": 0.9,
            "reasoning": "Goal achieved",
        }

        with patch.object(mock_agent.summarizer, "summarize", new_callable=AsyncMock) as mock_sum:
            mock_sum.return_value = {"summary_markdown": "Test summary"}
            await mock_agent.run("Test query", "/test/path")
            # Status should be success at the end
            assert mock_agent.status in ("running", "success")

    @pytest.mark.asyncio
    async def test_run_initializes_session(self, mock_agent, mock_llm):
        """Test run initializes session."""
        mock_llm.json_response = {
            "entities": {},
            "pipeline_stage": "setup",
            "route": "summarize",
            "original_goal_achieved": True,
            "confidence": 0.9,
            "reasoning": "Goal achieved",
        }

        with patch.object(mock_agent.summarizer, "summarize", new_callable=AsyncMock) as mock_sum:
            mock_sum.return_value = {"summary_markdown": "Test summary"}
            await mock_agent.run("Test query", "/test/path", 0.85)
            assert mock_agent.session is not None
            assert mock_agent.ctx is not None

    @pytest.mark.asyncio
    async def test_run_returns_summary_when_goal_achieved(self, mock_agent, mock_llm):
        """Test run returns summary when goal achieved."""
        mock_llm.json_response = {
            "entities": {},
            "pipeline_stage": "setup",
            "route": "summarize",
            "original_goal_achieved": True,
            "confidence": 0.9,
            "reasoning": "Goal achieved",
        }

        with patch.object(mock_agent.summarizer, "summarize", new_callable=AsyncMock) as mock_sum:
            mock_sum.return_value = {"summary_markdown": "Final summary"}
            result = await mock_agent.run("Test query", "/test/path")
            assert result == "Final summary"
            assert mock_agent.status == "success"

    @pytest.mark.asyncio
    async def test_run_selects_setup_workflow_before_perception_or_decision(self, mock_agent):
        """Test registry workflow selection runs before prompt-authored planning."""
        expected_step_ids = [
            step.step_id for step in mock_agent.workflow_registry.get("setup_pipeline").steps
        ]

        with (
            patch.object(
                mock_agent.perception,
                "run",
                new_callable=AsyncMock,
                side_effect=AssertionError("perception should not run before selection"),
            ) as mock_perception,
            patch.object(mock_agent.decision, "run", new_callable=AsyncMock) as mock_decision,
            patch("agent.agent_loop.execute_step", new_callable=AsyncMock) as mock_execute_step,
        ):
            result = await mock_agent.run("Set up MLOps for this project", "/test/path")

        assert mock_agent.workflow_selection.workflow_id == "setup_pipeline"
        assert mock_agent.workflow_selection.status is WorkflowStatus.PENDING
        assert mock_agent.ctx.get_pending_steps() == expected_step_ids
        for step in mock_agent.workflow_registry.get("setup_pipeline").steps:
            runtime_step = mock_agent.ctx.graph.nodes[step.step_id]["data"]
            assert runtime_step.tool in step.tool_functions
        assert "setup_pipeline" in result
        mock_perception.assert_not_awaited()
        mock_decision.assert_not_awaited()
        mock_execute_step.assert_not_awaited()


# ============================================================================
# run_mlops_agent Convenience Function Tests
# ============================================================================


class TestRunMlopsAgent:
    """Tests for run_mlops_agent convenience function."""

    @pytest.mark.asyncio
    async def test_run_mlops_agent_creates_agent(self, mock_llm):
        """Test run_mlops_agent creates and runs agent."""
        mock_llm.json_response = {
            "entities": {},
            "pipeline_stage": "setup",
            "route": "summarize",
            "original_goal_achieved": True,
            "confidence": 0.9,
            "reasoning": "Goal achieved",
        }

        with (
            patch(
                "agent.agent_loop.Summarizer.summarize", new_callable=AsyncMock
            ) as mock_summarize,
            patch("agent.agent_loop.Perception.run", new_callable=AsyncMock) as mock_perception,
        ):
            mock_perception.return_value = {
                "entities": {},
                "pipeline_stage": "setup",
                "route": "summarize",
                "original_goal_achieved": True,
                "confidence": 0.9,
                "reasoning": "Goal achieved",
            }
            mock_summarize.return_value = {"summary_markdown": "Test result"}
            result = await run_mlops_agent("Test query", "/test/path", 0.85)
            assert result == "Test result"

    @pytest.mark.asyncio
    async def test_run_mlops_agent_passes_event_callback(self, mock_llm):
        """Test run_mlops_agent passes event callback."""
        mock_llm.json_response = {
            "entities": {},
            "pipeline_stage": "setup",
            "route": "summarize",
            "original_goal_achieved": True,
            "confidence": 0.9,
            "reasoning": "Goal achieved",
        }

        events = []

        async def capture_event(event_type, data):
            events.append(event_type)

        with (
            patch(
                "agent.agent_loop.Summarizer.summarize", new_callable=AsyncMock
            ) as mock_summarize,
            patch("agent.agent_loop.Perception.run", new_callable=AsyncMock) as mock_perception,
        ):
            mock_perception.return_value = {
                "entities": {},
                "pipeline_stage": "setup",
                "route": "summarize",
                "original_goal_achieved": True,
                "confidence": 0.9,
                "reasoning": "Goal achieved",
            }
            mock_summarize.return_value = {"summary_markdown": "Test result"}
            await run_mlops_agent("Test query", on_event=capture_event)
            assert "status" in events
            assert "phase" in events


# ============================================================================
# AgentLoop Handle Failure Tests
# ============================================================================


class TestAgentLoopHandleFailure:
    """Tests for AgentLoop _handle_failure method."""

    @pytest.fixture
    def agent(self, tmp_path):
        """Create AgentLoop with test prompts."""
        prompts_dir = tmp_path / "prompts"
        prompts_dir.mkdir()
        (prompts_dir / "perception_prompt.txt").write_text("Perception")
        (prompts_dir / "decision_prompt.txt").write_text("Decision")
        (prompts_dir / "summarizer_prompt.txt").write_text("Summarize")
        (prompts_dir / "improvement_prompt.txt").write_text("Improve")
        return AgentLoop(prompts_dir=str(prompts_dir))

    @pytest.mark.asyncio
    async def test_handle_failure_sets_status_failed(self, agent, mock_llm):
        """Test _handle_failure preserves failed status while generating a summary."""
        mock_llm.json_response = {"summary_markdown": "Failure summary"}

        agent._initialize_session("Test", "/test", 0.85)
        agent.p_out = {}

        with patch.object(agent.summarizer, "summarize", new_callable=AsyncMock) as mock_sum:
            mock_sum.return_value = {"summary_markdown": "Failure summary"}
            await agent._handle_failure()
            assert agent.status == "failed"
            assert agent.session.status == "failed"

    @pytest.mark.asyncio
    async def test_handle_failure_marks_session_completed(self, agent, mock_llm):
        """Test _handle_failure marks session as completed with failure."""
        mock_llm.json_response = {"summary_markdown": "Failure summary"}

        agent._initialize_session("Test", "/test", 0.85)
        agent.p_out = {}

        with patch.object(agent.summarizer, "summarize", new_callable=AsyncMock) as mock_sum:
            mock_sum.return_value = {"summary_markdown": "Failure summary"}
            await agent._handle_failure()
            assert agent.session.status == "failed"

    @pytest.mark.asyncio
    async def test_handle_failure_returns_summary(self, agent, mock_llm):
        """Test _handle_failure still generates a summary."""
        mock_llm.json_response = {"summary_markdown": "Failure summary"}

        agent._initialize_session("Test", "/test", 0.85)
        agent.p_out = {}

        with patch.object(agent.summarizer, "summarize", new_callable=AsyncMock) as mock_sum:
            mock_sum.return_value = {"summary_markdown": "Failure summary"}
            result = await agent._handle_failure()
            assert result == "Failure summary"
