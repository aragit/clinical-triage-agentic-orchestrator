"""
OPA-style clinical guardrails enforced via Pydantic validators.

Production equivalent: Open Policy Agent (OPA) Rego policies deployed as a
sidecar service, evaluating every clinical request against a ruleset before
it reaches the LLM.  The OPA sidecar returns ALLOW/DENY + routing directives.

This local twin implements the exact same decision logic — safety-critical
triage bypass and content filtering — as deterministic Python validators
that run BEFORE the LLM is called.  This is the neuro-symbolic bridge:
probabilistic model outputs are only accepted after passing through these
hard, rule-based guardrails.
"""

from __future__ import annotations

import re
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field, field_validator


# ── Triage FSM states (duplicated here for guardrail independence) ──

class RouteAction(str, Enum):
    """Routing actions the guardrail can mandate."""

    ALLOW_TRIAGE = "allow_triage"
    ROUTE_TO_EMERGENCY = "route_to_emergency"
    ESCALATE_TO_HUMAN = "escalate_to_human"
    DENY = "deny"


# ── Guardrail output schema ─────────────────────────────────────────

class GuardrailResult(BaseModel):
    """Result of the OPA guardrail evaluation."""

    action: RouteAction
    is_safe: bool = Field(
        description="True if the input is safe for LLM triage processing"
    )
    triggered_rules: list[str] = Field(
        default_factory=list,
        description="Names of guardrail rules that fired"
    )
    emergency_override: bool = Field(
        default=False,
        description="If True, bypass ALL LLM processing and route to emergency immediately"
    )
    reason: str = Field(
        default="",
        description="Human-readable explanation of the guardrail decision"
    )

    def to_dict(self) -> dict[str, Any]:
        return {
            "action": self.action.value,
            "is_safe": self.is_safe,
            "triggered_rules": self.triggered_rules,
            "emergency_override": self.emergency_override,
            "reason": self.reason,
        }


# ── Input validator ─────────────────────────────────────────────────

class ClinicalInput(BaseModel):
    """Validated clinical input before it enters the agentic loop."""

    text: str = Field(min_length=1, max_length=10000)
    session_id: str = Field(min_length=1)
    patient_id: str = Field(default="")

    @field_validator("text")
    @classmethod
    def strip_and_validate(cls, v: str) -> str:
        v = v.strip()
        if not v:
            raise ValueError("Clinical input text cannot be empty")
        return v


# ── Guardrail engine ────────────────────────────────────────────────

# Life-threatening keyword patterns — immediate emergency bypass.
# These are the硬规则 (hard rules) that no probabilistic model should override.
_EMERGENCY_PATTERNS: list[tuple[str, str]] = [
    # Cardiac
    (r"\bheart\s*attack\b", "cardiac-emergency"),
    (r"\bmyocardial\s*infarction\b", "cardiac-emergency"),
    (r"\bstemi\b", "cardiac-emergency"),
    (r"\bchest\s*pain\b.*\b(shortness|breathing|dyspnea|breath)\b", "cardiac-emergency"),
    (r"\bchest\s*pain\b", "cardiac-emergency"),
    # Respiratory
    (r"\bshortness\s*of\s*breath\b", "respiratory-emergency"),
    (r"\bdifficulty\s*breathing\b", "respiratory-emergency"),
    (r"\bcan'?t\s*breathe\b", "respiratory-emergency"),
    (r"\bnot\s*breathing\b", "respiratory-emergency"),
    (r"\bsevere\s*(dyspnea|sob)\b", "respiratory-emergency"),
    # Psychiatric
    (r"\bsuicid(e|al)\b", "psychiatric-emergency"),
    (r"\bkill\s*myself\b", "psychiatric-emergency"),
    (r"\bend\s*my\s*life\b", "psychiatric-emergency"),
    (r"\bself[\-\s]*harm\b", "psychiatric-emergency"),
    # Allergic
    (r"\banaphyla(xis|ctic)\b", "allergic-emergency"),
    # Neurological
    (r"\bstroke\b", "neurological-emergency"),
    (r"\bfacial\s*droop\b", "neurological-emergency"),
    (r"\bface\s*(is\s*)?drooping\b", "neurological-emergency"),
    (r"\barm\s*weakness\b", "neurological-emergency"),
    (r"\bcan'?t\s*(lift|raise)\s*(my\s*)?arm\b", "neurological-emergency"),
    (r"\bslurred\s*speech\b", "neurological-emergency"),
    (r"\bseizure\b", "neurological-emergency"),
    (r"\bloss\s*of\s*consciousness\b", "neurological-emergency"),
    (r"\bunresponsive\b", "neurological-emergency"),
    # Hemorrhagic
    (r"\buncontrolled\s*bleed", "hemorrhagic-emergency"),
    (r"\bhemorrhag(e|ic)\b", "hemorrhagic-emergency"),
    # Toxicological
    (r"\boverdos(e|ed)\b", "toxicological-emergency"),
    (r"\bpoison(ing|ed)\b", "toxicological-emergency"),
    # Airway
    (r"\bchoking\b", "airway-emergency"),
]

# Escalation keywords — not immediately life-threatening but require
# human clinician review before LLM triage proceeds.
_ESCALATION_PATTERNS: list[tuple[str, str]] = [
    (r"\bpregnant\b", "obstetric"),
    (r"\bchild\b.*\byears?\s*old\b", "pediatric"),
    (r"\bbaby\b|\binfant\b", "pediatric"),
    (r"\bmedication\s*(change|adjust)", "medication-management"),
    (r"\bside\s*effect", "adverse-drug-event"),
    (r"\ballergic\b", "allergy-workup"),
]


class ClinicalGuardrail:
    """OPA-style guardrail engine for clinical triage safety.

    Evaluates every input against hard-coded safety rules BEFORE the
    LLM is invoked.  This is the neuro-symbolic boundary: deterministic
    rules gate probabilistic reasoning.

    Production equivalent: OPA sidecar with Rego policies.
    Local twin: Python regex + Pydantic validation.
    """

    def evaluate(self, text: str, session_id: str = "") -> GuardrailResult:
        """Run the full guardrail pipeline on clinical input.

        Evaluation order (matches OPA policy chain):
          1. Emergency detection  → ROUTE_TO_EMERGENCY (bypass LLM)
          2. Escalation detection → ESCALATE_TO_HUMAN (LLM reads but flags)
          3. Content safety       → ALLOW_TRIAGE or DENY
        """
        triggered: list[str] = []
        lower_text = text.lower()

        # ── Rule 1: Emergency detection ─────────────────────────────
        for pattern, rule_name in _EMERGENCY_PATTERNS:
            if re.search(pattern, lower_text):
                triggered.append(rule_name)

        if triggered:
            return GuardrailResult(
                action=RouteAction.ROUTE_TO_EMERGENCY,
                is_safe=False,
                triggered_rules=triggered,
                emergency_override=True,
                reason=(
                    f"Life-threatening condition(s) detected: {', '.join(triggered)}. "
                    "Bypassing LLM triage. Routing to emergency."
                ),
            )

        # ── Rule 2: Escalation detection ────────────────────────────
        for pattern, rule_name in _ESCALATION_PATTERNS:
            if re.search(pattern, lower_text):
                triggered.append(rule_name)

        if triggered:
            return GuardrailResult(
                action=RouteAction.ESCALATE_TO_HUMAN,
                is_safe=True,  # LLM may still process, but flags for human review
                triggered_rules=triggered,
                emergency_override=False,
                reason=(
                    f"Escalation trigger(s): {', '.join(triggered)}. "
                    "LLM triage will proceed but case flagged for human review."
                ),
            )

        # ── Rule 3: Content safety ──────────────────────────────────
        # Check for empty / nonsensical input
        if len(text.split()) < 2:
            return GuardrailResult(
                action=RouteAction.DENY,
                is_safe=False,
                triggered_rules=["minimum-content"],
                emergency_override=False,
                reason="Input too short for clinical triage. Please provide symptom details.",
            )

        # All clear — allow LLM triage
        return GuardrailResult(
            action=RouteAction.ALLOW_TRIAGE,
            is_safe=True,
            triggered_rules=[],
            emergency_override=False,
            reason="Input passed all guardrail checks. LLM triage permitted.",
        )
