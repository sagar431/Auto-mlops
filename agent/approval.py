"""
Approval utilities for human-in-the-loop workflows.

Stores and retrieves approval decisions from the session event log in the database.
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass

from db import AsyncSessionRepository, get_async_session

APPROVAL_DECISION_EVENT = "approval_decision"


@dataclass
class ApprovalDecision:
    """Represents a human approval decision."""

    approval_id: str
    approved: bool
    reason: str | None
    timestamp: str


async def get_latest_decision(
    session_id: str, approval_id: str | None = None
) -> ApprovalDecision | None:
    """Fetch the latest approval decision for a session (optionally filtered by approval_id)."""
    async with get_async_session() as db:
        repo = AsyncSessionRepository(db)
        session = await repo.get_session_by_id(session_id)
        if not session:
            return None

        events = list(session.events or [])
        for event in reversed(events):
            if event.get("type") != APPROVAL_DECISION_EVENT:
                continue
            data = event.get("data", {})
            if approval_id and data.get("approval_id") != approval_id:
                continue
            return ApprovalDecision(
                approval_id=data.get("approval_id", ""),
                approved=bool(data.get("approved", False)),
                reason=data.get("reason"),
                timestamp=event.get("timestamp", ""),
            )
    return None


async def wait_for_approval(
    session_id: str,
    approval_id: str,
    timeout_seconds: int = 300,
    poll_interval: float = 2.0,
) -> ApprovalDecision | None:
    """Wait for an approval decision to appear in the session event log."""
    start = time.monotonic()
    while True:
        decision = await get_latest_decision(session_id, approval_id)
        if decision:
            return decision

        if timeout_seconds > 0 and (time.monotonic() - start) >= timeout_seconds:
            return None

        await asyncio.sleep(poll_interval)
