"""
Core agentic triage loop — Perception → Memory → OPA → Cognition → Action.

This is the orchestrator brain.  It ties together every Phase 1 and Phase 2
module into a single deterministic execution pipeline:

  Step A (Perception):  Retrieve episodic history from the session store.
  Step B (OPA):         Run ClinicalGuardrail.  Emergency = instant bypass.
  Step C (Memory):      Fetch related clinical guidelines from Qdrant.
  Step D (Executor):    Run ClinicalEntityExtractor for SNOMED/ICD-10 codes.
  Step E (Cognition):   Pass all context to instructor-wrapped LLM, forcing
                        the response into the DiagnosticCoT schema.
  Step F (Action):      Update episodic state with the next FSM node.

The LLM is ONLY invoked after passing the OPA guardrail and receiving
enriched context from Memory + Executor.  This enforces strict
deterministic boundaries on probabilistic model outputs — the
neuro-symbolic bridge.
"""

from __future__ import annotations

import logging
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field

from src.core.opa_policies import ClinicalGuardrail, GuardrailResult, RouteAction
from src.memory.episodic_state import (
    EpisodicStore,
    Session,
    StateTransitionError,
    TriageState,
)
from src.memory.vector_store import HybridVectorStore
from src.tools.healthcare_nl import ClinicalEntityExtractor, ExtractionResult

logger = logging.getLogger(__name__)


# ── DiagnosticCoT schema ────────────────────────────────────────────
# The instructor wrapper forces the LLM to output EXACTLY this structure.
# This is the grammar-constrained sampling that makes the local LLM
# behave deterministically — no free-form hallucinated text allowed.


class UrgencyLevel(str, Enum):
    EMERGENT = "emergent"
    URGENT = "urgent"
    SEMI_URGENT = "semi-urgent"
    NON_URGENT = "non-urgent"
    DEFERRABLE = "deferrable"


class NextStateAction(str, Enum):
    """Maps directly to the triage FSM states (Dialogflow CX nodes)."""

    SYMPTOM_EXTRACTION = "symptom_extraction"
    GUIDELINE_LOOKUP = "guideline_lookup"
    RISK_ASSESSMENT = "risk_assessment"
    TRIAGE_DECISION = "triage_decision"
    ESCALATION = "escalation"
    RESOLVED = "resolved"


# Mapping from NextStateAction → TriageState
_ACTION_TO_STATE: dict[NextStateAction, TriageState] = {
    NextStateAction.SYMPTOM_EXTRACTION: TriageState.SYMPTOM_EXTRACTION,
    NextStateAction.GUIDELINE_LOOKUP: TriageState.GUIDELINE_LOOKUP,
    NextStateAction.RISK_ASSESSMENT: TriageState.RISK_ASSESSMENT,
    NextStateAction.TRIAGE_DECISION: TriageState.TRIAGE_DECISION,
    NextStateAction.ESCALATION: TriageState.ESCALATION,
    NextStateAction.RESOLVED: TriageState.RESOLVED,
}


class DiagnosticCoT(BaseModel):
    """Structured Chain-of-Thought diagnostic output.

    This schema is enforced by instructor at inference time.  The LLM
    MUST fill every field — no free-form text, no hallucinated fields.
    This is the neuro-symbolic bridge: the model's probabilistic reasoning
    is captured inside a deterministic, validated schema.
    """

    clinical_observations: list[str] = Field(
        description=(
            "Objective clinical observations extracted from the patient's "
            "input and session context.  Each item is a single observable fact."
        )
    )
    step_by_step_rationale: list[str] = Field(
        min_length=1,
        description=(
            "Explicit chain-of-thought reasoning steps.  Each step must "
            "follow logically from the previous one.  The model MUST think "
            "through the differential diagnosis before concluding."
        ),
    )
    urgency_level: UrgencyLevel = Field(
        description=(
            "Final urgency classification based on the reasoning chain. "
            "Must be one of: emergent, urgent, semi-urgent, non-urgent, deferrable."
        )
    )
    next_state_action: NextStateAction = Field(
        description=(
            "Deterministic FSM action.  Must map to exactly one Dialogflow CX "
            "state node.  The orchestrator will transition the session to this state."
        )
    )
    extracted_symptoms: list[str] = Field(
        description="Symptoms identified during this reasoning turn"
    )
    recommended_department: str = Field(
        description="Suggested department routing: ER | urgent_care | primary_care | telehealth | self_care"
    )
    confidence: float = Field(
        ge=0.0, le=1.0,
        description="Model's self-assessed confidence in the diagnostic reasoning"
    )

    def to_dict(self) -> dict[str, Any]:
        return {
            "clinical_observations": self.clinical_observations,
            "step_by_step_rationale": self.step_by_step_rationale,
            "urgency_level": self.urgency_level.value,
            "next_state_action": self.next_state_action.value,
            "extracted_symptoms": self.extracted_symptoms,
            "recommended_department": self.recommended_department,
            "confidence": self.confidence,
        }


# ── Triage agent ────────────────────────────────────────────────────

class TriageAgent:
    """Agentic orchestrator: Perception → Memory → OPA → Cognition → Action.

    Wires together all subsystems into the canonical triage loop.
    The LLM is ONLY called after passing OPA guardrails and receiving
    enriched context — deterministic gates protect probabilistic reasoning.
    """

    def __init__(
        self,
        episodic_store: EpisodicStore,
        vector_store: HybridVectorStore,
        llm_client: Any,  # LLMClient from llm_client.py (type-erased to avoid import cycle)
        entity_extractor: ClinicalEntityExtractor | None = None,
        guardrail: ClinicalGuardrail | None = None,
    ) -> None:
        self._episodic = episodic_store
        self._vectors = vector_store
        self._llm = llm_client
        self._extractor = entity_extractor or ClinicalEntityExtractor()
        self._guardrail = guardrail or ClinicalGuardrail()

    def execute_triage_turn(
        self, session_id: str, user_input: str
    ) -> dict[str, Any]:
        """Execute one complete triage turn through the agentic pipeline.

        Returns a dict with all intermediate artifacts so the caller can
        render the response or debug the pipeline.
        """
        result: dict[str, Any] = {
            "session_id": session_id,
            "user_input": user_input,
            "steps": {},
        }

        # ── Step A: Perception — retrieve episodic history ──────────
        session = self._episodic.get_session(session_id)
        if session is None:
            result["error"] = f"Session {session_id!r} not found"
            return result

        conversation_history = [
            {"role": t.role, "content": t.content} for t in session.turns
        ]
        current_state = session.current_state
        result["steps"]["A_perception"] = {
            "state": current_state.value,
            "history_turns": len(conversation_history),
            "existing_symptoms": session.extracted_symptoms,
        }

        # ── Step B: OPA guardrails ─────────────────────────────────
        guardrail_result = self._guardrail.evaluate(user_input, session_id)
        result["steps"]["B_guardrail"] = guardrail_result.to_dict()

        if guardrail_result.emergency_override:
            # EMERGENCY BYPASS — skip LLM entirely
            result["response"] = self._emergency_response(guardrail_result)
            result["state_transition"] = "escalation"
            return result

        if not guardrail_result.is_safe:
            result["response"] = {
                "message": guardrail_result.reason,
                "action": guardrail_result.action.value,
            }
            result["state_transition"] = None
            return result

        # ── Step C: Memory — vector store retrieval ─────────────────
        guidelines = self._vectors.search_hybrid(user_input, limit=5)
        result["steps"]["C_memory"] = {
            "guidelines_found": len(guidelines),
            "top_guideline": guidelines[0]["title"] if guidelines else None,
        }

        # Store matched guidelines in session
        self._episodic.update_guidelines(session_id, guidelines)

        # ── Step D: Executor — entity extraction ────────────────────
        extraction: ExtractionResult = self._extractor.extract(user_input)
        symptoms = [e.term.lower() for e in extraction.entities if e.category == "symptom"]
        all_terms = [e.term for e in extraction.entities]

        result["steps"]["D_executor"] = extraction.to_dict()

        # Update session with extracted symptoms
        if symptoms:
            existing = set(session.extracted_symptoms)
            existing.update(symptoms)
            self._episodic.update_symptoms(session_id, sorted(existing))

        # ── Step E: Cognition — instructor-forced CoT reasoning ─────
        # Build the enriched prompt with all context from A-D
        context_parts = [
            f"## Current triage state: {current_state.value}",
            f"## Conversation history ({len(conversation_history)} turns):",
        ]
        for turn in conversation_history[-6:]:  # last 6 turns for context
            context_parts.append(f"  [{turn['role']}] {turn['content']}")
        context_parts.append(f"## Current patient input: {user_input}")

        if all_terms:
            context_parts.append(
                f"## Extracted clinical entities: {', '.join(all_terms)}"
            )
        if guidelines:
            guids = "; ".join(
                f"{g['title']} ({g.get('guideline_id', 'n/a')})" for g in guidelines[:3]
            )
            context_parts.append(f"## Matched guidelines: {guids}")
        if session.extracted_symptoms:
            context_parts.append(
                f"## Accumulated symptoms: {', '.join(session.extracted_symptoms)}"
            )

        # Add severity hints
        if extraction.severity_hints:
            context_parts.append(
                f"## Severity signals: {', '.join(extraction.severity_hints)}"
            )

        enriched_prompt = "\n".join(context_parts)

        try:
            cot_response: DiagnosticCoT = self._llm.chat.completions.create(
                model=self._llm._model,
                response_model=DiagnosticCoT,
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "You are a clinical triage reasoning engine.  You MUST "
                            "produce structured Chain-of-Thought output.  Think "
                            "step-by-step through the differential diagnosis before "
                            "classifying urgency.  Every field in the schema is "
                            "required — no shortcuts."
                        ),
                    },
                    {"role": "user", "content": enriched_prompt},
                ],
                max_tokens=1024,
                temperature=0.1,
            )
            result["steps"]["E_cognition"] = cot_response.to_dict()
        except Exception as exc:
            logger.error("LLM call failed: %s", exc)
            result["error"] = f"LLM inference failed: {exc}"
            result["steps"]["E_cognition"] = {"error": str(exc)}
            return result

        # ── Step F: Action — update FSM state ───────────────────────
        next_state = _ACTION_TO_STATE.get(cot_response.next_state_action)
        if next_state is None:
            result["error"] = f"Invalid next_state_action: {cot_response.next_state_action}"
            return result

        try:
            self._episodic.transition(session_id, next_state)
            self._episodic.append_turn(session_id, "user", user_input)
            self._episodic.append_turn(
                session_id,
                "assistant",
                cot_response.model_dump_json(),
                metadata={"triage_turn": True},
            )
        except StateTransitionError as exc:
            # FSM rejected the transition — log but don't crash
            logger.warning("State transition rejected: %s", exc)
            result["warning"] = str(exc)

        result["response"] = {
            "observations": cot_response.clinical_observations,
            "rationale": cot_response.step_by_step_rationale,
            "urgency": cot_response.urgency_level.value,
            "department": cot_response.recommended_department,
            "confidence": cot_response.confidence,
            "symptoms": cot_response.extracted_symptoms,
        }
        result["state_transition"] = cot_response.next_state_action.value

        return result

    # ── Emergency bypass ────────────────────────────────────────────

    def _emergency_response(self, guardrail: GuardrailResult) -> dict[str, Any]:
        """Hardcoded emergency response — NO LLM involved."""
        return {
            "message": (
                "EMERGENCY: Your input indicates a potentially life-threatening "
                "condition.  Do NOT wait for an AI assessment.  "
                "Call emergency services (911) immediately or go to the "
                "nearest emergency room."
            ),
            "triggered_rules": guardrail.triggered_rules,
            "action": "route_to_emergency",
            "llm_bypassed": True,
        }
