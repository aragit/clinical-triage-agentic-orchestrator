"""
FastAPI webhook endpoint for the clinical triage orchestrator.

Production equivalent: a Dialogflow CX fulfillment webhook that receives
incoming patient messages, runs them through the triage pipeline, and
returns a response.

Post-mortem fix: The original implementation was SYNCHRONOUS — the LLM
cognition step could take 2-8 seconds, causing Dialogflow's 5-second
webhook timeout to fire and drop the response.  The fix is to use
FastAPI BackgroundTasks: the endpoint immediately returns a <150ms
"processing" acknowledgment, while the actual triage runs in the
background.  A polling endpoint lets the UI retrieve the completed result.
"""

from __future__ import annotations

import logging
import time
import uuid
from typing import Any

from fastapi import APIRouter, BackgroundTasks
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)
router = APIRouter()


# ── Request / Response schemas ──────────────────────────────────────

class DialogflowRequest(BaseModel):
    """Simulated Dialogflow CX webhook request payload."""

    session_id: str = Field(
        default_factory=lambda: uuid.uuid4().hex[:16],
        description="Dialogflow session ID",
    )
    query_text: str = Field(
        min_length=1,
        max_length=10000,
        description="Patient's free-text input",
    )
    patient_id: str = Field(default="", description="Optional patient identifier")
    language_code: str = Field(default="en")


class TriageResponse(BaseModel):
    """Synchronous triage response (returned when result is ready)."""

    session_id: str
    status: str = "completed"
    response: dict[str, Any] = {}
    state_transition: str | None = None
    latency_ms: float = 0.0


class ProcessingAck(BaseModel):
    """Immediate acknowledgment returned while triage runs in background."""

    session_id: str
    status: str = "processing"
    message: str = "Triage request accepted. Processing in background."
    poll_url: str = ""


# ── In-memory result store (production: Redis / Cloud Tasks) ────────

_results: dict[str, dict[str, Any]] = {}


# ── Background triage worker ────────────────────────────────────────

def _run_triage_background(
    session_id: str,
    user_input: str,
    patient_id: str = "",
) -> None:
    """Runs in a FastAPI BackgroundTask — the actual triage pipeline.

    This function imports state lazily to avoid circular imports with
    the main app module.
    """
    from src.api.main import state

    start = time.time()
    try:
        # Ensure session exists
        if state.episodic_store.get_session(session_id) is None:
            state.episodic_store.create_session(patient_id=patient_id, session_id=session_id)

        result = state.agent.execute_triage_turn(session_id, user_input)
        result["latency_ms"] = round((time.time() - start) * 1000, 1)
        result["status"] = "completed"
        _results[session_id] = result
        logger.info(
            "Triage completed for %s in %.1fms",
            session_id,
            result["latency_ms"],
        )
    except Exception as exc:
        logger.error("Triage failed for %s: %s", session_id, exc)
        _results[session_id] = {
            "session_id": session_id,
            "status": "error",
            "error": str(exc),
            "latency_ms": round((time.time() - start) * 1000, 1),
        }


# ── Endpoints ───────────────────────────────────────────────────────

@router.post("/fulfillment", response_model=ProcessingAck)
async def fulfillment(
    req: DialogflowRequest,
    background_tasks: BackgroundTasks,
) -> ProcessingAck:
    """Dialogflow CX fulfillment webhook — async pattern.

    POST /webhook/fulfillment

    Accepts a simulated Dialogflow request and immediately returns a
    <150ms acknowledgment.  The actual triage pipeline runs in a
    BackgroundTask so the webhook never times out.

    The UI polls GET /webhook/results/{session_id} for the result.
    """
    # Create session if needed
    from src.api.main import state

    if state.episodic_store.get_session(req.session_id) is None:
        state.episodic_store.create_session(
            patient_id=req.patient_id,
            session_id=req.session_id,
        )

    # Offload triage to background — this is the post-mortem fix
    background_tasks.add_task(
        _run_triage_background,
        session_id=req.session_id,
        user_input=req.query_text,
        patient_id=req.patient_id,
    )

    return ProcessingAck(
        session_id=req.session_id,
        poll_url=f"/webhook/results/{req.session_id}",
    )


@router.get("/results/{session_id}", response_model=TriageResponse)
async def get_result(session_id: str) -> TriageResponse:
    """Poll for completed triage results.

    GET /webhook/results/{session_id}

    Returns 202 (processing) if the background task hasn't finished yet.
    Returns 200 with the full result once complete.
    """
    result = _results.get(session_id)
    if result is None or result.get("status") == "processing":
        from fastapi.responses import JSONResponse
        return JSONResponse(
            status_code=202,
            content={
                "session_id": session_id,
                "status": "processing",
                "message": "Triage still in progress",
            },
        )

    return TriageResponse(
        session_id=session_id,
        status=result.get("status", "completed"),
        response=result.get("response", {}),
        state_transition=result.get("state_transition"),
        latency_ms=result.get("latency_ms", 0.0),
    )


@router.post("/triage", response_model=TriageResponse)
async def triage_sync(req: DialogflowRequest) -> TriageResponse:
    """Synchronous triage endpoint (for testing / CLI).

    POST /webhook/triage

    Runs the full pipeline inline and returns the result.  Use this for
    testing or when you know the LLM server is fast enough.  The async
    /fulfillment endpoint is the production pattern.
    """
    from src.api.main import state

    start = time.time()

    if state.episodic_store.get_session(req.session_id) is None:
        state.episodic_store.create_session(patient_id=req.patient_id, session_id=req.session_id)

    result = state.agent.execute_triage_turn(req.session_id, req.query_text)
    latency = round((time.time() - start) * 1000, 1)

    return TriageResponse(
        session_id=req.session_id,
        status="completed",
        response=result.get("response", {}),
        state_transition=result.get("state_transition"),
        latency_ms=latency,
    )


@router.get("/session/{session_id}")
async def get_session_state(session_id: str) -> dict[str, Any]:
    """Inspect the current episodic state for a session.

    GET /webhook/session/{session_id}

    Returns the full session state including FSM node, conversation
    history, extracted symptoms, and matched guidelines.
    """
    from src.api.main import state

    session = state.episodic_store.get_session(session_id)
    if session is None:
        from fastapi.responses import JSONResponse
        return JSONResponse(
            status_code=404,
            content={"error": f"Session {session_id!r} not found"},
        )

    return {
        "session_id": session.session_id,
        "patient_id": session.patient_id,
        "current_state": session.current_state.value,
        "turns": len(session.turns),
        "extracted_symptoms": session.extracted_symptoms,
        "matched_guidelines": len(session.matched_guidelines),
        "risk_score": session.risk_score,
        "triage_result": session.triage_result,
    }


@router.get("/sessions")
async def list_sessions() -> dict[str, Any]:
    """List all active sessions.

    GET /webhook/sessions
    """
    from src.api.main import state

    sids = state.episodic_store.list_sessions()
    sessions = []
    for sid in sids:
        s = state.episodic_store.get_session(sid)
        if s:
            sessions.append({
                "session_id": s.session_id,
                "patient_id": s.patient_id,
                "state": s.current_state.value,
                "turns": len(s.turns),
            })
    return {"count": len(sessions), "sessions": sessions}
