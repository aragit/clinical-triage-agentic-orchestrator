"""
Local LLM client wrapping llama-cpp-python's OpenAI-compatible server
with instructor for grammar-constrained structured output.

Production equivalent: Vertex AI Gemini / PaLM endpoint with
response-schema enforcement.  This local twin achieves the same
deterministic, schema-validated output by combining:

  1. llama-cpp-python server (port 8000) — CPU-native Qwen 2.5 GGUF
  2. OpenAI SDK pointed at localhost — API-compatible wire protocol
  3. instructor.from_openai() — injects grammar/schema into sampling

The result is a perception-to-cognition engine that runs entirely on a
laptop with no cloud keys, no GPU, and strict Pydantic contract enforcement.
"""

from __future__ import annotations

import logging
from typing import Any

import instructor
import openai
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

# ── Configuration ───────────────────────────────────────────────────

LOCAL_LLM_BASE_URL = "http://localhost:8000/v1"
LOCAL_LLM_MODEL = "local-model"  # llama-cpp-server registers any model as this
LOCAL_LLM_API_KEY = "not-needed"  # llama-cpp-server doesn't check keys


# ── Structured output schemas ───────────────────────────────────────

class SymptomExtraction(BaseModel):
    """Structured clinical symptom extraction from free-text patient input."""

    symptoms: list[str] = Field(
        description="List of identified clinical symptoms"
    )
    severity: str = Field(
        description="Overall severity assessment: mild | moderate | severe | critical"
    )
    body_systems: list[str] = Field(
        description="Affected body systems (e.g., respiratory, cardiac, neurological)"
    )
    onset: str = Field(
        description="When symptoms started: acute (<24h), subacute (1-7d), chronic (>7d)"
    )
    reasoning: str = Field(
        description="Brief chain-of-thought explaining the extraction decisions"
    )


class RiskAssessment(BaseModel):
    """Clinical risk scoring for triage prioritization."""

    risk_level: str = Field(
        description="Triage level: emergent | urgent | semi-urgent | non-urgent | deferrable"
    )
    risk_score: float = Field(
        ge=0.0, le=1.0,
        description="Normalized risk score 0.0 (lowest) to 1.0 (highest)"
    )
    requires_escalation: bool = Field(
        description="Whether the case must be escalated to a human clinician"
    )
    reasoning: str = Field(
        description="Chain-of-thought explaining the risk determination"
    )
    recommended_action: str = Field(
        description="Immediate recommended clinical action"
    )


class TriageDecision(BaseModel):
    """Final triage output combining assessment with actionable routing."""

    department: str = Field(
        description="Recommended department: ER | urgent_care | primary_care | telehealth | self_care"
    )
    urgency_minutes: int = Field(
        ge=0,
        description="Maximum minutes before the patient should be seen"
    )
    reasoning: str = Field(
        description="Chain-of-thought summarising the full triage rationale"
    )
    contraindications: list[str] = Field(
        default_factory=list,
        description="Any identified contraindications for standard treatment"
    )
    follow_up: str = Field(
        description="Recommended follow-up action or timeframe"
    )


# ── Client ──────────────────────────────────────────────────────────

class LLMClient:
    """Instructor-wrapped OpenAI client for structured clinical reasoning.

    Usage::

        client = LLMClient()
        result = client.extract_symptoms("Patient reports chest tightness and shortness of breath...")
        print(result.symptoms)  # ["chest tightness", "shortness of breath"]
    """

    def __init__(
        self,
        base_url: str = LOCAL_LLM_BASE_URL,
        model: str = LOCAL_LLM_MODEL,
        api_key: str = LOCAL_LLM_API_KEY,
    ) -> None:
        self._model = model
        self._raw_client = openai.OpenAI(base_url=base_url, api_key=api_key)
        self._client = instructor.from_openai(self._raw_client)

    # ── Health check ────────────────────────────────────────────────

    def is_server_reachable(self) -> bool:
        """Verify the llama-cpp server is up and responding."""
        try:
            self._raw_client.models.list()
            return True
        except openai.APIConnectionError:
            return False
        except Exception as exc:  # noqa: BLE001
            logger.warning("LLM health check failed: %s", exc)
            return False

    def health_check(self) -> dict[str, Any]:
        """Return structured health status for diagnostics."""
        reachable = self.is_server_reachable()
        return {
            "status": "healthy" if reachable else "unreachable",
            "base_url": self._raw_client.base_url,
            "model": self._model,
        }

    # ── Structured extraction ───────────────────────────────────────

    def extract_symptoms(
        self,
        patient_text: str,
        system_prompt: str | None = None,
        **kwargs: Any,
    ) -> SymptomExtraction:
        """Extract structured symptoms from free-text clinical input."""
        if system_prompt is None:
            system_prompt = (
                "You are a clinical NLP engine. Extract structured symptom data "
                "from the patient's free-text input. Think step-by-step before "
                "outputting your structured assessment."
            )
        return self._client.chat.completions.create(
            model=self._model,
            response_model=SymptomExtraction,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": patient_text},
            ],
            max_tokens=1024,
            temperature=0.1,  # low temperature for clinical determinism
            **kwargs,
        )

    def assess_risk(
        self,
        symptoms: list[str],
        patient_context: str = "",
        system_prompt: str | None = None,
        **kwargs: Any,
    ) -> RiskAssessment:
        """Generate a clinical risk assessment from extracted symptoms."""
        if system_prompt is None:
            system_prompt = (
                "You are a clinical triage risk engine. Assess the urgency and "
                "risk level of the presented symptoms. Think step-by-step through "
                "differential diagnoses before providing your risk determination."
            )
        symptom_text = ", ".join(symptoms)
        user_msg = f"Symptoms: {symptom_text}"
        if patient_context:
            user_msg += f"\nPatient context: {patient_context}"

        return self._client.chat.completions.create(
            model=self._model,
            response_model=RiskAssessment,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_msg},
            ],
            max_tokens=1024,
            temperature=0.1,
            **kwargs,
        )

    def make_triage_decision(
        self,
        symptoms: list[str],
        risk_assessment: RiskAssessment,
        matched_guidelines: list[dict[str, Any]] | None = None,
        system_prompt: str | None = None,
        **kwargs: Any,
    ) -> TriageDecision:
        """Produce the final triage routing decision."""
        if system_prompt is None:
            system_prompt = (
                "You are a clinical triage decision engine. Given extracted symptoms, "
                "risk assessment, and matched clinical guidelines, produce a definitive "
                "triage routing decision. Think step-by-step and explain your reasoning."
            )
        parts = [
            f"Symptoms: {', '.join(symptoms)}",
            f"Risk level: {risk_assessment.risk_level} (score: {risk_assessment.risk_score})",
            f"Escalation required: {risk_assessment.requires_escalation}",
        ]
        if matched_guidelines:
            guids = "; ".join(
                g.get("title", "unnamed") for g in matched_guidelines[:5]
            )
            parts.append(f"Matched guidelines: {guids}")

        return self._client.chat.completions.create(
            model=self._model,
            response_model=TriageDecision,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": "\n".join(parts)},
            ],
            max_tokens=1024,
            temperature=0.1,
            **kwargs,
        )


# ── Quick verification script ───────────────────────────────────────

def _verify() -> None:
    """Run a connectivity test against the local llama-cpp server."""
    client = LLMClient()
    status = client.health_check()
    print(f"LLM server status: {status}")
    if status["status"] == "healthy":
        print("Server is reachable. Structured output schemas loaded.")
        print("  SymptomExtraction fields:", list(SymptomExtraction.model_fields))
        print("  RiskAssessment fields:", list(RiskAssessment.model_fields))
        print("  TriageDecision fields:", list(TriageDecision.model_fields))
    else:
        print(
            "Server not reachable. Start it with:\n"
            "  python -m llama_cpp.server --model models/Qwen2.5-3B-Instruct-Q4_K_M.gguf --port 8000"
        )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    _verify()
