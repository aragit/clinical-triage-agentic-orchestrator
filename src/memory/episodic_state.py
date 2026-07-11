"""
Episodic session state backed by an in-process dictionary.

Production equivalent: Cloud Memorystore (Redis) holding per-user session
keys, conversation history, and finite-state-machine node positions.
This local twin replaces Redis with a plain dict so the orchestrator runs
with zero infrastructure while preserving the exact same key-based API
contract used in production.
"""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class TriageState(str, Enum):
    """Finite-state nodes for the clinical triage FSM."""

    INTAKE = "intake"
    SYMPTOM_EXTRACTION = "symptom_extraction"
    GUIDELINE_LOOKUP = "guideline_lookup"
    RISK_ASSESSMENT = "risk_assessment"
    TRIAGE_DECISION = "triage_decision"
    ESCALATION = "escalation"
    RESOLVED = "resolved"


@dataclass
class Turn:
    """A single conversational turn."""

    role: str  # "user" | "assistant" | "system"
    content: str
    timestamp: float = field(default_factory=time.time)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class Session:
    """Complete episodic state for one clinical triage session."""

    session_id: str
    patient_id: str = ""
    current_state: TriageState = TriageState.INTAKE
    turns: list[Turn] = field(default_factory=list)
    extracted_symptoms: list[str] = field(default_factory=list)
    matched_guidelines: list[dict[str, Any]] = field(default_factory=list)
    risk_score: float = 0.0
    triage_result: str = ""
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "session_id": self.session_id,
            "patient_id": self.patient_id,
            "current_state": self.current_state.value,
            "turns": [
                {
                    "role": t.role,
                    "content": t.content,
                    "timestamp": t.timestamp,
                    "metadata": t.metadata,
                }
                for t in self.turns
            ],
            "extracted_symptoms": self.extracted_symptoms,
            "matched_guidelines": self.matched_guidelines,
            "risk_score": self.risk_score,
            "triage_result": self.triage_result,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "metadata": self.metadata,
        }


# ── Valid state transitions (FSM guard) ────────────────────────────

_VALID_TRANSITIONS: dict[TriageState, set[TriageState]] = {
    TriageState.INTAKE: {TriageState.SYMPTOM_EXTRACTION},
    TriageState.SYMPTOM_EXTRACTION: {TriageState.GUIDELINE_LOOKUP},
    TriageState.GUIDELINE_LOOKUP: {TriageState.RISK_ASSESSMENT},
    TriageState.RISK_ASSESSMENT: {TriageState.TRIAGE_DECISION},
    TriageState.TRIAGE_DECISION: {TriageState.RESOLVED, TriageState.ESCALATION},
    TriageState.ESCALATION: {TriageState.RESOLVED},
    TriageState.RESOLVED: set(),
}


class StateTransitionError(Exception):
    """Raised when the FSM rejects an illegal state transition."""


# ── Session store ───────────────────────────────────────────────────

class EpisodicStore:
    """In-process session store replacing Redis for local development.

    API contract mirrors the production Redis client:
      - ``create_session() -> session_id``
      - ``get_session(session_id) -> Session | None``
      - ``transition(session_id, target_state) -> Session``
      - ``append_turn(session_id, role, content) -> Turn``
      - ``update_symptoms(session_id, symptoms) -> None``
      - ``update_risk(session_id, score, result) -> None``
    """

    def __init__(self, ttl_seconds: int = 3600) -> None:
        self._store: dict[str, Session] = {}
        self._ttl = ttl_seconds

    # ── CRUD ────────────────────────────────────────────────────────

    def create_session(
        self,
        patient_id: str = "",
        initial_state: TriageState = TriageState.INTAKE,
        session_id: str = "",
    ) -> str:
        sid = session_id or uuid.uuid4().hex[:16]
        self._store[sid] = Session(session_id=sid, patient_id=patient_id, current_state=initial_state)
        return sid

    def get_session(self, session_id: str) -> Session | None:
        session = self._store.get(session_id)
        if session is None:
            return None
        if time.time() - session.updated_at > self._ttl:
            del self._store[session_id]
            return None
        return session

    def delete_session(self, session_id: str) -> bool:
        return self._store.pop(session_id, None) is not None

    def list_sessions(self) -> list[str]:
        return list(self._store.keys())

    # ── FSM transitions ─────────────────────────────────────────────

    def transition(self, session_id: str, target_state: TriageState) -> Session:
        session = self._get_or_raise(session_id)
        allowed = _VALID_TRANSITIONS.get(session.current_state, set())
        if target_state not in allowed:
            raise StateTransitionError(
                f"Cannot transition from {session.current_state.value} "
                f"to {target_state.value}.  "
                f"Allowed: {[s.value for s in allowed]}"
            )
        session.current_state = target_state
        session.updated_at = time.time()
        return session

    # ── Conversation history ────────────────────────────────────────

    def append_turn(
        self,
        session_id: str,
        role: str,
        content: str,
        metadata: dict[str, Any] | None = None,
    ) -> Turn:
        session = self._get_or_raise(session_id)
        turn = Turn(role=role, content=content, metadata=metadata or {})
        session.turns.append(turn)
        session.updated_at = time.time()
        return turn

    def get_turns(self, session_id: str, last_n: int | None = None) -> list[Turn]:
        session = self._get_or_raise(session_id)
        if last_n is None:
            return list(session.turns)
        return list(session.turns[-last_n:])

    # ── Clinical data ───────────────────────────────────────────────

    def update_symptoms(self, session_id: str, symptoms: list[str]) -> None:
        session = self._get_or_raise(session_id)
        session.extracted_symptoms = symptoms
        session.updated_at = time.time()

    def update_guidelines(self, session_id: str, guidelines: list[dict[str, Any]]) -> None:
        session = self._get_or_raise(session_id)
        session.matched_guidelines = guidelines
        session.updated_at = time.time()

    def update_risk(
        self, session_id: str, score: float, result: str = ""
    ) -> None:
        session = self._get_or_raise(session_id)
        session.risk_score = score
        session.triage_result = result
        session.updated_at = time.time()

    # ── Internal ────────────────────────────────────────────────────

    def _get_or_raise(self, session_id: str) -> Session:
        session = self.get_session(session_id)
        if session is None:
            raise KeyError(f"Session {session_id!r} not found or expired")
        return session
