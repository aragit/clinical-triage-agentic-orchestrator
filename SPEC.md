# Clinical Triage Agentic Orchestrator — Full Specification & Source Code

**Version:** 0.3.0
**Date:** 2026-07-12
**Commit:** 46183c3

---

## Table of Contents

1. [Architecture Overview](#1-architecture-overview)
2. [System Flow](#2-system-flow)
3. [Module Descriptions](#3-module-descriptions)
4. [Known Issues & Limitations](#4-known-issues--limitations)
5. [Setup & Running](#5-setup--running)
6. [Complete Source Code](#6-complete-source-code)

---

## 1. Architecture Overview

```
┌──────────────┐     ┌──────────────┐     ┌──────────────────┐
│  Streamlit   │────▶│   FastAPI     │────▶│  llama-cpp-server │
│  Dashboard   │     │   API :8080   │     │  :8000 (CPU LLM) │
│  :8501       │     │              │     │  Gemma 3n E4B     │
└──────────────┘     └──────┬───────┘     └──────────────────┘
                            │
              ┌─────────────┼─────────────┐
              ▼             ▼             ▼
        ┌──────────┐ ┌──────────┐ ┌──────────────┐
        │ Episodic  │ │  Vector  │ │    OPA       │
        │  Store    │ │  Store   │ │  Guardrails  │
        │ (dict)    │ │ (Qdrant) │ │  (regex)     │
        └──────────┘ └──────────┘ └──────────────┘
```

### Components

| Port | Service | Description |
|------|---------|-------------|
| 8000 | llama-cpp-server | CPU-native Gemma 3n E4B GGUF inference |
| 8080 | FastAPI API | Triage pipeline, webhook endpoints |
| 8501 | Streamlit UI | Observability dashboard with chat |

### Pipeline Steps (per request)

```
Step A: Perception    — Load session from EpisodicStore
Step B: OPA Guardrails — Regex-based emergency/escalation detection (BEFORE LLM)
Step C: Memory        — Hybrid vector search (dense + BM25) against clinical guidelines
Step D: Executor      — SNOMED CT / ICD-10 entity extraction via regex dictionary
Step E: Cognition     — LLM inference with JSON response format (resilient fallback)
Step F: Action        — FSM state transition with validity enforcement
```

### FSM States

```
intake → symptom_extraction → guideline_lookup → risk_assessment → triage_decision → resolved/escalation
```

---

## 2. System Flow

1. **User sends message** via Streamlit chat input
2. **Dashboard POSTs** to `/webhook/fulfillment` (async) with session_id
3. **API creates session** in EpisodicStore using the UI's session_id
4. **API returns** immediate `ProcessingAck` (200 OK)
5. **Background task** runs `execute_triage_turn()`:
   - Loads session state (Step A)
   - Runs OPA guardrails — emergency keywords bypass LLM entirely (Step B)
   - Searches vector store for matching clinical guidelines (Step C)
   - Extracts SNOMED/ICD-10 entities from text (Step D)
   - Calls LLM with enriched prompt, requesting JSON output (Step E)
   - Transitions FSM state (Step F)
6. **Dashboard polls** `/webhook/results/{session_id}` every 0.5s for up to 180s
7. **Result returned** with observations, rationale, urgency, department, latency

---

## 3. Module Descriptions

### `src/api/main.py` — FastAPI Application
- Lifespan handler initializes all subsystems as singletons
- Seeds 5 clinical guidelines into the vector store
- CORS enabled for all origins
- Health endpoint reports uptime, LLM reachability, guideline count

### `src/api/webhook.py` — API Endpoints
- `POST /webhook/fulfillment` — Async triage (returns immediately, runs in BackgroundTask)
- `GET /webhook/results/{session_id}` — Poll for completed results (202 if processing)
- `POST /webhook/triage` — Synchronous triage (for testing)
- `GET /webhook/session/{session_id}` — Inspect session state
- `GET /webhook/sessions` — List all active sessions

### `src/cognition/triage_agent.py` — Core Agent (largest file, 454 lines)
- `DiagnosticCoT` Pydantic model — expected LLM output schema
- `TriageAgent.execute_triage_turn()` — the 6-step pipeline
- Resilient LLM response handling: 3-tier fallback:
  1. Perfect JSON matching DiagnosticCoT schema → use directly
  2. JSON but wrong schema/enums → heuristic extraction with enum normalization
  3. Non-JSON text → use as rationale with defaults
- FSM transition enforcement: picks valid next state even if LLM suggests invalid one

### `src/cognition/llm_client.py` — LLM Client
- Wraps OpenAI SDK pointed at localhost llama-cpp server
- Model: `models/gemma-3n-E4B-it-Q4_K_M.gguf`
- Also includes instructor-wrapped methods (extract_symptoms, assess_risk, make_triage_decision) — currently unused by the pipeline due to instructor/gemma incompatibility

### `src/core/opa_policies.py` — Guardrails
- 19 emergency regex patterns (heart attack, stroke, seizure, etc.) → instant LLM bypass
- 6 escalation patterns (pregnant, pediatric, etc.) → flags for human review
- Minimum content check (2+ words)
- Production equivalent: OPA sidecar with Rego policies

### `src/memory/episodic_state.py` — Session State
- In-process dict replacing Redis for local dev
- FSM transition validation with `_VALID_TRANSITIONS` map
- TTL-based session expiry (default 2h)
- Conversation history, extracted symptoms, matched guidelines per session

### `src/memory/vector_store.py` — Clinical Guideline Retrieval
- Qdrant in-memory mode with 384-dim pseudo-embeddings
- Hybrid search: dense (cosine) + sparse (Okapi BM25) fused via RRF
- In-process BM25 implementation (k1=1.5, b=0.75)
- 5 seeded guidelines: chest pain, stroke, asthma, diabetic emergency, anaphylaxis

### `src/tools/healthcare_nl.py` — Clinical NLP
- Regex-based entity extraction with 20+ clinical presentations
- SNOMED CT and ICD-10 code mapping
- 5 severity escalation patterns
- Deterministic, zero cloud keys

### `src/ui/dashboard.py` — Streamlit UI
- Two-column layout: Patient Chat (left) | Observability Trace (right)
- Quick scenario buttons: Chest Pain, Stroke, Headache, Heart Attack
- FSM visualization with state progression
- SNOMED/ICD-10 entity display
- Raw JSON inspector
- 180s polling timeout for CPU inference

---

## 4. Known Issues & Limitations

### LLM Output Quality (Root Cause: Model Capability)
**Symptom:** The Gemma 3n E4B model on CPU (2.3 tokens/sec) consistently returns conversational text (`{"response": "Okay, I understand..."}`) instead of the structured JSON schema requested.

**Impact:** All responses fall through to the heuristic fallback, producing:
- `urgency: "semi-urgent"` (default)
- `department: "primary_care"` (default)
- `confidence: 0.5` (hardcoded)
- `rationale: "LLM provided free-form response"` (placeholder)
- Observations show the raw LLM text instead of structured clinical observations

**Why:** The 6.9B Gemma model at Q4_K quantization on CPU lacks the instruction-following precision to produce structured JSON with exact enum values. The model has ~4B active parameters (E4B architecture) and is optimized for general chat, not schema-constrained output.

**Possible Fixes (not implemented):**
1. **Switch to a smaller, instruction-tuned model** (e.g., Qwen 2.5 1.5B) that follows JSON schemas better
2. **Use grammar-constrained sampling** (llama.cpp `--grammar` flag) to force JSON output at the token level
3. **Use a cloud API** (OpenAI, Gemini) for the cognition step — fast, reliable, schema-compliant
4. **Add few-shot examples** to the system prompt to guide the model toward the expected format
5. **Fine-tune** the model on DiagnosticCoT-format data

### Performance
- CPU inference: ~2.3 tokens/sec for 6.9B model → 60-120 seconds per turn
- Single-threaded llama-cpp server — requests queue behind each other
- Pseudo-embeddings (SHA-512 hash) — not semantically meaningful, only structural correctness

### Session Persistence
- In-process dict — sessions lost on API restart
- Production would use Redis (docker-compose includes Redis service)

### Entity Extraction
- 20+ clinical patterns — covers common emergency/primary-care presentations
- Not exhaustive — misses rare conditions, medication interactions, lab values

---

## 5. Setup & Running

### Option A: Local (3 terminals)

```bash
# Terminal 1 — LLM Server (~20s startup)
cd ~/clinical-triage-agentic-orchestrator
python -m llama_cpp.server \
  --model models/gemma-3n-E4B-it-Q4_K_M.gguf \
  --host 0.0.0.0 --port 8000 \
  --n_ctx 4096 --chat_format gemma

# Terminal 2 — FastAPI Backend
cd ~/clinical-triage-agentic-orchestrator
uvicorn src.api.main:app --host 0.0.0.0 --port 8080 --reload

# Terminal 3 — Streamlit Dashboard
cd ~/clinical-triage-agentic-orchestrator
streamlit run src/ui/dashboard.py --server.port 8501
```

### Option B: Docker Compose

```bash
cd ~/clinical-triage-agentic-orchestrator
docker compose up --build
```

### Dependencies

```
fastapi>=0.110.0
uvicorn[standard]>=0.29.0
streamlit>=1.33.0
llama-cpp-python[server]>=0.2.56
instructor>=1.2.0
openai>=1.25.0
qdrant-client>=1.8.0
pydantic>=2.7.0
httpx>=0.27.0
```

---

## 6. Complete Source Code

---

### `src/api/main.py`

```python
"""
FastAPI application entry point with lifespan-managed subsystems.

Initialises the vector store, episodic store, LLM client, guardrail,
entity extractor, and triage agent on startup — all singletons shared
across requests.  Tears them down cleanly on shutdown.

Run with:  uvicorn src.api.main:app --host 0.0.0.0 --port 8080 --reload
"""

from __future__ import annotations

import logging
import time
from contextlib import asynccontextmanager
from typing import AsyncGenerator

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from src.api.webhook import router as webhook_router
from src.cognition.llm_client import LLMClient
from src.cognition.triage_agent import TriageAgent
from src.core.opa_policies import ClinicalGuardrail
from src.memory.episodic_state import EpisodicStore
from src.memory.vector_store import ClinicalGuideline, HybridVectorStore
from src.tools.healthcare_nl import ClinicalEntityExtractor

logger = logging.getLogger(__name__)

# ── Application state (replaces global singletons) ──────────────────


class AppState:
    """Shared state container — instantiated once in the lifespan handler."""

    episodic_store: EpisodicStore
    vector_store: HybridVectorStore
    llm_client: LLMClient
    guardrail: ClinicalGuardrail
    extractor: ClinicalEntityExtractor
    agent: TriageAgent
    start_time: float


state = AppState()


# ── Seed clinical guidelines ────────────────────────────────────────

_SEED_GUIDELINES = [
    ClinicalGuideline(
        title="Chest Pain Protocol",
        content=(
            "Acute chest pain with shortness of breath requires immediate "
            "12-lead ECG, troponin levels, and CBC.  Administer aspirin 325mg "
            "if no contraindications.  Assess with HEART score."
        ),
        specialty="cardiology",
        icd10_codes=["R07.9"],
        snomed_codes=["29857009"],
    ),
    ClinicalGuideline(
        title="Stroke FAST Assessment",
        content=(
            "Sudden facial droop, arm weakness, or speech difficulty — "
            "perform FAST assessment.  If positive, activate stroke alert. "
            "CT head within 25 minutes.  Consider tPA if within 4.5h window."
        ),
        specialty="neurology",
        icd10_codes=["I63.9"],
        snomed_codes=["230690007"],
    ),
    ClinicalGuideline(
        title="Asthma Exacerbation Management",
        content=(
            "Wheezing and dyspnea — administer nebulized albuterol 2.5mg q20min "
            "x3.  Assess peak flow.  If severe (PEF <25% predicted), add IV "
            "magnesium sulfate and consider intubation."
        ),
        specialty="respiratory",
        icd10_codes=["J45.909"],
        snomed_codes=["195967001"],
    ),
    ClinicalGuideline(
        title="Diabetic Emergency Protocol",
        content=(
            "Hypoglycemia (BG <70 mg/dL): administer oral glucose or IV dextrose "
            "(D50W 50mL).  Hyperglycemia with ketones: initiate DKA protocol "
            "with IV fluids and insulin drip."
        ),
        specialty="endocrinology",
        icd10_codes=["E11.65"],
        snomed_codes=["302866004"],
    ),
    ClinicalGuideline(
        title="Anaphylaxis Protocol",
        content=(
            "Suspected anaphylaxis: epinephrine 0.3mg IM immediately "
            "(0.15mg if <30kg).  Repeat q5-15min PRN.  Administer diphenhydramine "
            "50mg IV and methylprednisolone 125mg IV.  Monitor airway."
        ),
        specialty="emergency",
        icd10_codes=["T78.2"],
        snomed_codes=["39579001"],
    ),
]


# ── Lifespan ────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(_app: FastAPI) -> AsyncGenerator[None, None]:
    """Startup/shutdown lifecycle — initialises all subsystems once."""
    state.episodic_store = EpisodicStore(ttl_seconds=7200)
    state.vector_store = HybridVectorStore()
    state.guardrail = ClinicalGuardrail()
    state.extractor = ClinicalEntityExtractor()

    # LLM client — gracefully degrade if llama-cpp server isn't up yet
    state.llm_client = LLMClient()
    if state.llm_client.is_server_reachable():
        logger.info("llama-cpp server connected at %s", state.llm_client._raw_client.base_url)
    else:
        logger.warning(
            "llama-cpp server not reachable — triage agent will skip "
            "LLM cognition step until server is started"
        )

    state.agent = TriageAgent(
        episodic_store=state.episodic_store,
        vector_store=state.vector_store,
        llm_client=state.llm_client,
        entity_extractor=state.extractor,
        guardrail=state.guardrail,
    )

    # Seed clinical guidelines into the vector store
    inserted = state.vector_store.insert_guidelines(_SEED_GUIDELINES)
    logger.info("Seeded %d clinical guidelines into vector store", len(inserted))

    state.start_time = time.time()
    logger.info("Clinical Triage Orchestrator started")

    yield  # app is running

    logger.info("Clinical Triage Orchestrator shutting down")


# ── FastAPI app ─────────────────────────────────────────────────────

app = FastAPI(
    title="Clinical Triage Agentic Orchestrator",
    description=(
        "CPU-native clinical triage pipeline with OPA guardrails, "
        "instructor-forced CoT reasoning, and deterministic FSM routing."
    ),
    version="0.3.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(webhook_router, prefix="/webhook", tags=["triage"])


@app.get("/health")
def health() -> dict:
    return {
        "status": "healthy",
        "uptime_seconds": time.time() - state.start_time,
        "llm_reachable": state.llm_client.is_server_reachable(),
        "guidelines_loaded": state.vector_store.count(),
    }
```

---

### `src/api/webhook.py`

```python
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
```

---

### `src/cognition/triage_agent.py`

```python
"""
Core agentic triage loop — Perception → Memory → OPA → Cognition → Action.

This is the orchestrator brain.  It ties together every Phase 1 and Phase 2
module into a single deterministic execution pipeline:

  Step A (Perception):  Retrieve episodic history from the session store.
  Step B (OPA):         Run ClinicalGuardrail.  Emergency = instant bypass.
  Step C (Memory):      Fetch related clinical guidelines from Qdrant.
  Step D (Executor):    Run ClinicalEntityExtractor for SNOMED/ICD-10 codes.
  Step E (Cognition):   Pass all context to LLM with JSON response format.
                        Resilient fallback handles wrong schema/enums/text.
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

class UrgencyLevel(str, Enum):
    EMERGENT = "emergent"
    URGENT = "urgent"
    SEMI_URGENT = "semi-urgent"
    NON_URGENT = "non-urgent"
    DEFERRABLE = "deferrable"


class NextStateAction(str, Enum):
    """Maps directly to the triage FSM states."""
    SYMPTOM_EXTRACTION = "symptom_extraction"
    GUIDELINE_LOOKUP = "guideline_lookup"
    RISK_ASSESSMENT = "risk_assessment"
    TRIAGE_DECISION = "triage_decision"
    ESCALATION = "escalation"
    RESOLVED = "resolved"


_ACTION_TO_STATE: dict[NextStateAction, TriageState] = {
    NextStateAction.SYMPTOM_EXTRACTION: TriageState.SYMPTOM_EXTRACTION,
    NextStateAction.GUIDELINE_LOOKUP: TriageState.GUIDELINE_LOOKUP,
    NextStateAction.RISK_ASSESSMENT: TriageState.RISK_ASSESSMENT,
    NextStateAction.TRIAGE_DECISION: TriageState.TRIAGE_DECISION,
    NextStateAction.ESCALATION: TriageState.ESCALATION,
    NextStateAction.RESOLVED: TriageState.RESOLVED,
}


class DiagnosticCoT(BaseModel):
    """Structured Chain-of-Thought diagnostic output."""

    clinical_observations: list[str]
    step_by_step_rationale: list[str] = Field(min_length=1)
    urgency_level: UrgencyLevel
    next_state_action: NextStateAction
    extracted_symptoms: list[str]
    recommended_department: str
    confidence: float = Field(ge=0.0, le=1.0)

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
    """Agentic orchestrator: Perception → Memory → OPA → Cognition → Action."""

    def __init__(
        self,
        episodic_store: EpisodicStore,
        vector_store: HybridVectorStore,
        llm_client: Any,
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
        """Execute one complete triage turn through the agentic pipeline."""
        result: dict[str, Any] = {
            "session_id": session_id,
            "user_input": user_input,
            "steps": {},
        }

        # ── Step A: Perception ──────────────────────────────────────
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

        # ── Step C: Memory ─────────────────────────────────────────
        guidelines = self._vectors.search_hybrid(user_input, limit=5)
        result["steps"]["C_memory"] = {
            "guidelines_found": len(guidelines),
            "top_guideline": guidelines[0]["title"] if guidelines else None,
        }
        self._episodic.update_guidelines(session_id, guidelines)

        # ── Step D: Executor ───────────────────────────────────────
        extraction: ExtractionResult = self._extractor.extract(user_input)
        symptoms = [e.term.lower() for e in extraction.entities if e.category == "symptom"]
        all_terms = [e.term for e in extraction.entities]
        result["steps"]["D_executor"] = extraction.to_dict()

        if symptoms:
            existing = set(session.extracted_symptoms)
            existing.update(symptoms)
            self._episodic.update_symptoms(session_id, sorted(existing))

        # ── Step E: Cognition ──────────────────────────────────────
        context_parts = [
            f"## Current triage state: {current_state.value}",
            f"## Conversation history ({len(conversation_history)} turns):",
        ]
        for turn in conversation_history[-6:]:
            context_parts.append(f"  [{turn['role']}] {turn['content']}")
        context_parts.append(f"## Current patient input: {user_input}")

        if all_terms:
            context_parts.append(f"## Extracted clinical entities: {', '.join(all_terms)}")
        if guidelines:
            guids = "; ".join(
                f"{g['title']} ({g.get('guideline_id', 'n/a')})" for g in guidelines[:3]
            )
            context_parts.append(f"## Matched guidelines: {guids}")
        if session.extracted_symptoms:
            context_parts.append(f"## Accumulated symptoms: {', '.join(session.extracted_symptoms)}")
        if extraction.severity_hints:
            context_parts.append(f"## Severity signals: {', '.join(extraction.severity_hints)}")

        enriched_prompt = "\n".join(context_parts)

        try:
            import json
            raw_response = self._llm._raw_client.chat.completions.create(
                model=self._llm._model,
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "You are a clinical triage engine. "
                            "Analyze the patient input and return ONLY a JSON object "
                            "with these keys: "
                            "clinical_observations (list of strings), "
                            "step_by_step_rationale (list of strings), "
                            "urgency_level (emergent|urgent|semi-urgent|non-urgent|deferrable), "
                            "next_state_action (symptom_extraction|risk_assessment|triage_decision|escalation), "
                            "extracted_symptoms (list of strings), "
                            "recommended_department (ER|urgent_care|primary_care|telehealth|self_care), "
                            "confidence (number 0-1). "
                            "Return ONLY the JSON object, no other text."
                        ),
                    },
                    {"role": "user", "content": enriched_prompt},
                ],
                response_format={"type": "json_object"},
                max_tokens=512,
                temperature=0.1,
            )
            content = raw_response.choices[0].message.content
            parsed = json.loads(content)

            try:
                cot_response = DiagnosticCoT(**parsed)
                result["steps"]["E_cognition"] = cot_response.to_dict()
            except Exception:
                logger.warning("LLM JSON didn't match schema, wrapping response")

                def _extract_text(data: dict, *keys: str, fallback: str = "") -> str:
                    for k in keys:
                        v = data.get(k)
                        if isinstance(v, str):
                            return v
                        if isinstance(v, list) and v:
                            return v[0] if isinstance(v[0], str) else str(v[0])
                    return fallback

                _urgency_map = {
                    "high": "urgent", "medium": "semi-urgent", "low": "non-urgent",
                    "critical": "emergent", "emergency": "emergent", "mild": "deferrable",
                    "severe": "urgent", "moderate": "semi-urgent", "urgent": "urgent",
                }
                raw_urgency = parsed.get("urgency_level", parsed.get("urgency", "semi-urgent"))
                urgency = _urgency_map.get(str(raw_urgency).lower(), "semi-urgent")

                _dept_map = {
                    "emergency": "ER", "er": "ER", "urgent care": "urgent_care",
                    "primary": "primary_care", "telehealth": "telehealth",
                    "self care": "self_care", "self-care": "self_care",
                }
                raw_dept = parsed.get("recommended_department", parsed.get("department", "primary_care"))
                dept = _dept_map.get(str(raw_dept).lower(), "primary_care")

                _action_map = {
                    "intake": "symptom_extraction", "resolved": "triage_decision",
                    "triage": "triage_decision", "assess": "risk_assessment",
                }
                raw_action = parsed.get("next_state_action", "triage_decision")
                action = _action_map.get(str(raw_action).lower(), "triage_decision")

                raw_obs = _extract_text(parsed, "clinical_observations", "observations", fallback="")
                if not raw_obs:
                    raw_obs = _extract_text(parsed, "response", fallback=content[:300])
                elif isinstance(parsed.get("clinical_observations"), list):
                    raw_obs = str(parsed["clinical_observations"][0]) if parsed["clinical_observations"] else content[:300]

                raw_rat = _extract_text(parsed, "step_by_step_rationale", "rationale", fallback="LLM provided free-form response")
                if isinstance(parsed.get("step_by_step_rationale"), list):
                    raw_rat = str(parsed["step_by_step_rationale"][0]) if parsed["step_by_step_rationale"] else raw_rat

                raw_syms = parsed.get("extracted_symptoms", session.extracted_symptoms or ["symptoms reported"])
                if isinstance(raw_syms, str):
                    raw_syms = [raw_syms]

                cot_response = DiagnosticCoT(
                    clinical_observations=[raw_obs],
                    step_by_step_rationale=[raw_rat],
                    urgency_level=urgency,
                    next_state_action=action,
                    extracted_symptoms=raw_syms,
                    recommended_department=dept,
                    confidence=float(parsed.get("confidence", 0.5)),
                )
                result["steps"]["E_cognition"] = {
                    **cot_response.to_dict(),
                    "raw_llm_output": content[:500],
                }

        except json.JSONDecodeError:
            logger.warning("LLM returned non-JSON text, using as rationale")
            cot_response = DiagnosticCoT(
                clinical_observations=[content[:200]],
                step_by_step_rationale=[content[:500]],
                urgency_level="semi-urgent",
                next_state_action="triage_decision",
                extracted_symptoms=session.extracted_symptoms or symptoms or ["symptoms reported"],
                recommended_department="primary_care",
                confidence=0.4,
            )
            result["steps"]["E_cognition"] = {
                **cot_response.to_dict(),
                "raw_llm_output": content[:500],
            }
        except Exception as exc:
            logger.error("LLM call failed: %s", exc)
            result["error"] = f"LLM inference failed: {exc}"
            result["steps"]["E_cognition"] = {"error": str(exc)}
            result["response"] = {
                "message": f"LLM inference failed: {exc}. Please ensure the LLM server is running.",
                "observations": [],
                "rationale": [],
                "urgency": "unknown",
                "department": "unknown",
                "confidence": 0.0,
                "symptoms": [],
            }
            return result

        # ── Step F: Action ─────────────────────────────────────────
        from src.memory.episodic_state import _VALID_TRANSITIONS
        suggested_state = _ACTION_TO_STATE.get(cot_response.next_state_action)
        allowed = _VALID_TRANSITIONS.get(session.current_state, set())

        if suggested_state and suggested_state in allowed:
            next_state = suggested_state
        elif allowed:
            next_state = max(allowed, key=lambda s: list(TriageState).index(s))
        else:
            next_state = TriageState.RESOLVED

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
```

---

### `src/cognition/llm_client.py`

```python
"""
Local LLM client wrapping llama-cpp-python's OpenAI-compatible server.

Production equivalent: Vertex AI Gemini / PaLM endpoint with
response-schema enforcement.
"""

from __future__ import annotations

import logging
from typing import Any

import instructor
import openai
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

LOCAL_LLM_BASE_URL = "http://localhost:8000/v1"
LOCAL_LLM_MODEL = "models/gemma-3n-E4B-it-Q4_K_M.gguf"
LOCAL_LLM_API_KEY = "not-needed"


class SymptomExtraction(BaseModel):
    symptoms: list[str]
    severity: str
    body_systems: list[str]
    onset: str
    reasoning: str


class RiskAssessment(BaseModel):
    risk_level: str
    risk_score: float = Field(ge=0.0, le=1.0)
    requires_escalation: bool
    reasoning: str
    recommended_action: str


class TriageDecision(BaseModel):
    department: str
    urgency_minutes: int = Field(ge=0)
    reasoning: str
    contraindications: list[str] = Field(default_factory=list)
    follow_up: str


class LLMClient:
    def __init__(
        self,
        base_url: str = LOCAL_LLM_BASE_URL,
        model: str = LOCAL_LLM_MODEL,
        api_key: str = LOCAL_LLM_API_KEY,
    ) -> None:
        self._model = model
        self._raw_client = openai.OpenAI(base_url=base_url, api_key=api_key)
        self._client = instructor.from_openai(self._raw_client)

    def is_server_reachable(self) -> bool:
        try:
            self._raw_client.models.list()
            return True
        except openai.APIConnectionError:
            return False
        except Exception as exc:
            logger.warning("LLM health check failed: %s", exc)
            return False

    def health_check(self) -> dict[str, Any]:
        reachable = self.is_server_reachable()
        return {
            "status": "healthy" if reachable else "unreachable",
            "base_url": self._raw_client.base_url,
            "model": self._model,
        }
```

---

### `src/core/opa_policies.py`

```python
"""
OPA-style clinical guardrails enforced via regex pattern matching.

Production equivalent: Open Policy Agent (OPA) Rego policies deployed as a
sidecar service, evaluating every clinical request against a ruleset before
it reaches the LLM.
"""

from __future__ import annotations

import re
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


class RouteAction(str, Enum):
    ALLOW_TRIAGE = "allow_triage"
    ROUTE_TO_EMERGENCY = "route_to_emergency"
    ESCALATE_TO_HUMAN = "escalate_to_human"
    DENY = "deny"


class GuardrailResult(BaseModel):
    action: RouteAction
    is_safe: bool
    triggered_rules: list[str] = Field(default_factory=list)
    emergency_override: bool = Field(default=False)
    reason: str = Field(default="")

    def to_dict(self) -> dict[str, Any]:
        return {
            "action": self.action.value,
            "is_safe": self.is_safe,
            "triggered_rules": self.triggered_rules,
            "emergency_override": self.emergency_override,
            "reason": self.reason,
        }


class ClinicalInput(BaseModel):
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


_EMERGENCY_PATTERNS: list[tuple[str, str]] = [
    (r"\bheart\s*attack\b", "cardiac-emergency"),
    (r"\bmyocardial\s*infarction\b", "cardiac-emergency"),
    (r"\bstemi\b", "cardiac-emergency"),
    (r"\bsuicid(e|al)\b", "psychiatric-emergency"),
    (r"\bkill\s*myself\b", "psychiatric-emergency"),
    (r"\bend\s*my\s*life\b", "psychiatric-emergency"),
    (r"\bself[\-\s]*harm\b", "psychiatric-emergency"),
    (r"\banaphyla(xis|ctic)\b", "allergic-emergency"),
    (r"\bstroke\b", "neurological-emergency"),
    (r"\bseizure\b", "neurological-emergency"),
    (r"\buncontrolled\s*bleed", "hemorrhagic-emergency"),
    (r"\bhemorrhag(e|ic)\b", "hemorrhagic-emergency"),
    (r"\bloss\s*of\s*consciousness\b", "neurological-emergency"),
    (r"\bunresponsive\b", "neurological-emergency"),
    (r"\boverdos(e|ed)\b", "toxicological-emergency"),
    (r"\bpoison(ing|ed)\b", "toxicological-emergency"),
    (r"\bchoking\b", "airway-emergency"),
    (r"\bcan'?t\s*breathe\b", "respiratory-emergency"),
    (r"\bnot\s*breathing\b", "respiratory-emergency"),
]

_ESCALATION_PATTERNS: list[tuple[str, str]] = [
    (r"\bpregnant\b", "obstetric"),
    (r"\bchild\b.*\byears?\s*old\b", "pediatric"),
    (r"\bbaby\b|\binfant\b", "pediatric"),
    (r"\bmedication\s*(change|adjust)", "medication-management"),
    (r"\bside\s*effect", "adverse-drug-event"),
    (r"\ballergic\b", "allergy-workup"),
]


class ClinicalGuardrail:
    def evaluate(self, text: str, session_id: str = "") -> GuardrailResult:
        triggered: list[str] = []
        lower_text = text.lower()

        for pattern, rule_name in _EMERGENCY_PATTERNS:
            if re.search(pattern, lower_text):
                triggered.append(rule_name)

        if triggered:
            return GuardrailResult(
                action=RouteAction.ROUTE_TO_EMERGENCY,
                is_safe=False,
                triggered_rules=triggered,
                emergency_override=True,
                reason=f"Life-threatening condition(s) detected: {', '.join(triggered)}. Bypassing LLM triage.",
            )

        for pattern, rule_name in _ESCALATION_PATTERNS:
            if re.search(pattern, lower_text):
                triggered.append(rule_name)

        if triggered:
            return GuardrailResult(
                action=RouteAction.ESCALATE_TO_HUMAN,
                is_safe=True,
                triggered_rules=triggered,
                emergency_override=False,
                reason=f"Escalation trigger(s): {', '.join(triggered)}. Case flagged for human review.",
            )

        if len(text.split()) < 2:
            return GuardrailResult(
                action=RouteAction.DENY,
                is_safe=False,
                triggered_rules=["minimum-content"],
                emergency_override=False,
                reason="Input too short for clinical triage.",
            )

        return GuardrailResult(
            action=RouteAction.ALLOW_TRIAGE,
            is_safe=True,
            triggered_rules=[],
            emergency_override=False,
            reason="Input passed all guardrail checks.",
        )
```

---

### `src/core/config.py`

```python
"""
Centralised configuration for the clinical triage orchestrator.
"""

from __future__ import annotations

import os
from dataclasses import dataclass


@dataclass(frozen=True)
class LLMConfig:
    base_url: str = "http://localhost:8000/v1"
    model: str = "local-model"
    api_key: str = "not-needed"
    max_tokens: int = 1024
    temperature: float = 0.1


@dataclass(frozen=True)
class VectorStoreConfig:
    dense_dim: int = 384
    rrf_k: int = 60
    collection_name: str = "clinical_guidelines"


@dataclass(frozen=True)
class EpisodicConfig:
    ttl_seconds: int = 3600
    max_turns_in_context: int = 20


@dataclass(frozen=True)
class AppConfig:
    llm: LLMConfig = LLMConfig()
    vector_store: VectorStoreConfig = VectorStoreConfig()
    episodic: EpisodicConfig = EpisodicConfig()
    log_level: str = "INFO"


def load_config() -> AppConfig:
    return AppConfig(
        llm=LLMConfig(
            base_url=os.getenv("LLM_BASE_URL", "http://localhost:8000/v1"),
            model=os.getenv("LLM_MODEL", "local-model"),
            api_key=os.getenv("LLM_API_KEY", "not-needed"),
            max_tokens=int(os.getenv("LLM_MAX_TOKENS", "1024")),
            temperature=float(os.getenv("LLM_TEMPERATURE", "0.1")),
        ),
        vector_store=VectorStoreConfig(
            dense_dim=int(os.getenv("VS_DENSE_DIM", "384")),
            rrf_k=int(os.getenv("VS_RRF_K", "60")),
        ),
        episodic=EpisodicConfig(
            ttl_seconds=int(os.getenv("EPISODE_TTL", "3600")),
        ),
        log_level=os.getenv("LOG_LEVEL", "INFO"),
    )
```

---

### `src/memory/episodic_state.py`

```python
"""
Episodic session state backed by an in-process dictionary.

Production equivalent: Cloud Memorystore (Redis) holding per-user session
keys, conversation history, and finite-state-machine node positions.
"""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class TriageState(str, Enum):
    INTAKE = "intake"
    SYMPTOM_EXTRACTION = "symptom_extraction"
    GUIDELINE_LOOKUP = "guideline_lookup"
    RISK_ASSESSMENT = "risk_assessment"
    TRIAGE_DECISION = "triage_decision"
    ESCALATION = "escalation"
    RESOLVED = "resolved"


@dataclass
class Turn:
    role: str
    content: str
    timestamp: float = field(default_factory=time.time)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class Session:
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
            "turns": [{"role": t.role, "content": t.content, "timestamp": t.timestamp, "metadata": t.metadata} for t in self.turns],
            "extracted_symptoms": self.extracted_symptoms,
            "matched_guidelines": self.matched_guidelines,
            "risk_score": self.risk_score,
            "triage_result": self.triage_result,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "metadata": self.metadata,
        }


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
    pass


class EpisodicStore:
    def __init__(self, ttl_seconds: int = 3600) -> None:
        self._store: dict[str, Session] = {}
        self._ttl = ttl_seconds

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

    def transition(self, session_id: str, target_state: TriageState) -> Session:
        session = self._get_or_raise(session_id)
        allowed = _VALID_TRANSITIONS.get(session.current_state, set())
        if target_state not in allowed:
            raise StateTransitionError(
                f"Cannot transition from {session.current_state.value} to {target_state.value}. "
                f"Allowed: {[s.value for s in allowed]}"
            )
        session.current_state = target_state
        session.updated_at = time.time()
        return session

    def append_turn(self, session_id: str, role: str, content: str, metadata: dict[str, Any] | None = None) -> Turn:
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

    def update_symptoms(self, session_id: str, symptoms: list[str]) -> None:
        session = self._get_or_raise(session_id)
        session.extracted_symptoms = symptoms
        session.updated_at = time.time()

    def update_guidelines(self, session_id: str, guidelines: list[dict[str, Any]]) -> None:
        session = self._get_or_raise(session_id)
        session.matched_guidelines = guidelines
        session.updated_at = time.time()

    def update_risk(self, session_id: str, score: float, result: str = "") -> None:
        session = self._get_or_raise(session_id)
        session.risk_score = score
        session.triage_result = result
        session.updated_at = time.time()

    def _get_or_raise(self, session_id: str) -> Session:
        session = self.get_session(session_id)
        if session is None:
            raise KeyError(f"Session {session_id!r} not found or expired")
        return session
```

---

### `src/memory/vector_store.py`

```python
"""
Hybrid vector store backed by Qdrant in-memory mode.

Provides dual-mode retrieval: dense semantic search and sparse BM25-style
keyword search, fused via reciprocal rank fusion (RRF) for clinical guideline
lookup.
"""

from __future__ import annotations

import math
import uuid
from collections import Counter
from dataclasses import dataclass, field
from typing import Any

from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance, FieldCondition, Filter, MatchValue, PointStruct, VectorParams,
)


@dataclass
class ClinicalGuideline:
    guideline_id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    title: str = ""
    content: str = ""
    specialty: str = ""
    icd10_codes: list[str] = field(default_factory=list)
    snomed_codes: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


COLLECTION = "clinical_guidelines"
DENSE_DIM = 384


class HybridVectorStore:
    def __init__(self) -> None:
        self._client = QdrantClient(":memory:")
        self._sparse_index: list[dict[str, Any]] = []
        self._ensure_collection()

    def _ensure_collection(self) -> None:
        collections = [c.name for c in self._client.get_collections().collections]
        if COLLECTION in collections:
            return
        self._client.create_collection(
            collection_name=COLLECTION,
            vectors_config=VectorParams(size=DENSE_DIM, distance=Distance.COSINE),
        )

    def insert_guideline(self, guideline: ClinicalGuideline) -> str:
        point_id = uuid.uuid4().int >> 64
        dense_vector = _pseudo_embed(guideline.content, DENSE_DIM)
        payload = {
            "guideline_id": guideline.guideline_id,
            "title": guideline.title,
            "content": guideline.content,
            "specialty": guideline.specialty,
            "icd10_codes": guideline.icd10_codes,
            "snomed_codes": guideline.snomed_codes,
            **guideline.metadata,
        }
        self._client.upsert(
            collection_name=COLLECTION,
            points=[PointStruct(id=point_id, vector=dense_vector, payload=payload)],
        )
        self._sparse_index.append({
            "guideline_id": guideline.guideline_id,
            "title": guideline.title,
            "content": guideline.content,
            "specialty": guideline.specialty,
            "payload": payload,
        })
        return guideline.guideline_id

    def insert_guidelines(self, guidelines: list[ClinicalGuideline]) -> list[str]:
        return [self.insert_guideline(g) for g in guidelines]

    def search_dense(self, query: str, limit: int = 5, filter_specialty: str | None = None) -> list[dict[str, Any]]:
        qvector = _pseudo_embed(query, DENSE_DIM)
        query_filter = _build_filter(specialty=filter_specialty)
        results = self._client.query_points(
            collection_name=COLLECTION, query=qvector, query_filter=query_filter, limit=limit,
        )
        return _format_results(results)

    def search_sparse(self, query: str, limit: int = 5, filter_specialty: str | None = None) -> list[dict[str, Any]]:
        hits = _bm25_search(query=query, corpus=self._sparse_index, filter_specialty=filter_specialty, limit=limit)
        return [{**h["payload"], "_score": h["score"]} for h in hits]

    def search_hybrid(self, query: str, limit: int = 5, filter_specialty: str | None = None, rrf_k: int = 60) -> list[dict[str, Any]]:
        dense = self.search_dense(query, limit=limit * 2, filter_specialty=filter_specialty)
        sparse = self.search_sparse(query, limit=limit * 2, filter_specialty=filter_specialty)
        scores: dict[str, float] = {}
        payload_map: dict[str, dict] = {}
        for rank, hit in enumerate(dense):
            gid = hit["guideline_id"]
            scores[gid] = scores.get(gid, 0.0) + 1.0 / (rrf_k + rank + 1)
            payload_map[gid] = hit
        for rank, hit in enumerate(sparse):
            gid = hit["guideline_id"]
            scores[gid] = scores.get(gid, 0.0) + 1.0 / (rrf_k + rank + 1)
            if gid not in payload_map:
                payload_map[gid] = hit
        ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:limit]
        return [{**payload_map[gid], "rrf_score": sc} for gid, sc in ranked]

    def count(self) -> int:
        return self._client.count(collection_name=COLLECTION).count


def _build_filter(specialty: str | None = None) -> Filter | None:
    if not specialty:
        return None
    return Filter(must=[FieldCondition(key="specialty", match=MatchValue(value=specialty))])


def _pseudo_embed(text: str, dim: int) -> list[float]:
    import hashlib
    h = hashlib.sha512(text.encode()).digest()
    raw = [h[i % len(h)] / 255.0 for i in range(dim)]
    norm = sum(v * v for v in raw) ** 0.5 or 1.0
    return [v / norm for v in raw]


def _format_results(results: Any) -> list[dict[str, Any]]:
    hits = getattr(results, "points", results) if not isinstance(results, list) else results
    out: list[dict[str, Any]] = []
    for hit in hits:
        payload = hit.payload if hasattr(hit, "payload") else hit.get("payload", {})
        score = hit.score if hasattr(hit, "score") else hit.get("score", 0.0)
        out.append({**payload, "_score": score})
    return out


_BM25_K1 = 1.5
_BM25_B = 0.75


def _tokenize(text: str) -> list[str]:
    return [tok for tok in text.lower().split() if len(tok) >= 2 and tok.isalnum()]


def _bm25_search(query: str, corpus: list[dict[str, Any]], filter_specialty: str | None = None, limit: int = 5) -> list[dict[str, Any]]:
    if not corpus:
        return []
    tokenized_corpus = [_tokenize(doc["content"]) for doc in corpus]
    doc_count = len(tokenized_corpus)
    avgdl = sum(len(d) for d in tokenized_corpus) / doc_count if doc_count else 1.0
    df: Counter[str] = Counter()
    for doc_toks in tokenized_corpus:
        for tok in set(doc_toks):
            df[tok] += 1
    query_tokens = _tokenize(query)
    scores: list[tuple[int, float]] = []
    for idx, doc_toks in enumerate(tokenized_corpus):
        if filter_specialty and corpus[idx].get("specialty") != filter_specialty:
            continue
        tf = Counter(doc_toks)
        doc_len = len(doc_toks)
        score = 0.0
        for qt in query_tokens:
            if qt not in tf:
                continue
            term_freq = tf[qt]
            doc_freq = df.get(qt, 0)
            idf = math.log((doc_count - doc_freq + 0.5) / (doc_freq + 0.5) + 1.0)
            tf_norm = (term_freq * (_BM25_K1 + 1)) / (term_freq + _BM25_K1 * (1 - _BM25_B + _BM25_B * doc_len / avgdl))
            score += idf * tf_norm
        if score > 0:
            scores.append((idx, score))
    scores.sort(key=lambda x: x[1], reverse=True)
    return [{"guideline_id": corpus[idx]["guideline_id"], "payload": corpus[idx]["payload"], "score": sc} for idx, sc in scores[:limit]]
```

---

### `src/tools/healthcare_nl.py`

```python
"""
Clinical NLP entity extraction with mocked SNOMED CT / ICD-10 code lookup.

Production equivalent: Google Healthcare NLP API / AWS Comprehend Medical.
Local twin uses keyword+regex with a curated dictionary of ~20 clinical
presentations.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any


@dataclass
class ClinicalEntity:
    term: str
    snomed_code: str
    icd10_code: str
    category: str  # symptom | diagnosis
    confidence: float = 1.0
    context: str = ""


@dataclass
class ExtractionResult:
    raw_text: str
    entities: list[ClinicalEntity] = field(default_factory=list)
    primary_complaint: str = ""
    body_systems: list[str] = field(default_factory=list)
    severity_hints: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "raw_text": self.raw_text,
            "entities": [
                {"term": e.term, "snomed_code": e.snomed_code, "icd10_code": e.icd10_code, "category": e.category, "confidence": e.confidence}
                for e in self.entities
            ],
            "primary_complaint": self.primary_complaint,
            "body_systems": self.body_systems,
            "severity_hints": self.severity_hints,
        }


_TERMINOLOGY: list[dict[str, Any]] = [
    {"patterns": [r"chest\s*pain", r"chest\s*tight", r"angina"], "term": "Chest Pain", "snomed": "29857009", "icd10": "R07.9", "category": "symptom", "system": "cardiovascular"},
    {"patterns": [r"heart\s*attack", r"myocardial\s*infarction", r"mi\b", r"stemi"], "term": "Myocardial Infarction", "snomed": "22298006", "icd10": "I21.9", "category": "diagnosis", "system": "cardiovascular"},
    {"patterns": [r"palpitation", r"heart\s*racing", r"irregular\s*heartbeat"], "term": "Palpitations", "snomed": "426783006", "icd10": "R00.2", "category": "symptom", "system": "cardiovascular"},
    {"patterns": [r"shortness\s*of\s*breath", r"dyspnea", r"breathing\s*difficult", r"can'?t\s*breathe"], "term": "Dyspnea", "snomed": "267036007", "icd10": "R06.02", "category": "symptom", "system": "respiratory"},
    {"patterns": [r"asthma", r"wheez"], "term": "Asthma Exacerbation", "snomed": "195967001", "icd10": "J45.909", "category": "diagnosis", "system": "respiratory"},
    {"patterns": [r"cough", r"persistent\s*cough"], "term": "Cough", "snomed": "49727002", "icd10": "R05.9", "category": "symptom", "system": "respiratory"},
    {"patterns": [r"headache", r"migraine", r"throbbing\s*head"], "term": "Headache", "snomed": "25064002", "icd10": "R51.9", "category": "symptom", "system": "neurological"},
    {"patterns": [r"dizziness", r"vertigo", r"lightheaded", r"feeling\s*faint"], "term": "Dizziness", "snomed": "404640003", "icd10": "R42", "category": "symptom", "system": "neurological"},
    {"patterns": [r"stroke", r"facial\s*droop", r"arm\s*weakness", r"slurred\s*speech"], "term": "Stroke", "snomed": "230690007", "icd10": "I63.9", "category": "diagnosis", "system": "neurological"},
    {"patterns": [r"seizure", r"convulsion", r"grand\s*mal"], "term": "Seizure", "snomed": "371631005", "icd10": "R56.9", "category": "symptom", "system": "neurological"},
    {"patterns": [r"abdominal\s*pain", r"stomach\s*pain", r"belly\s*pain"], "term": "Abdominal Pain", "snomed": "271803005", "icd10": "R10.9", "category": "symptom", "system": "gastrointestinal"},
    {"patterns": [r"nausea", r"feeling\s*sick"], "term": "Nausea", "snomed": "422587007", "icd10": "R11.0", "category": "symptom", "system": "gastrointestinal"},
    {"patterns": [r"vomit", r"emesis"], "term": "Vomiting", "snomed": "422400008", "icd10": "R11.10", "category": "symptom", "system": "gastrointestinal"},
    {"patterns": [r"hypoglycemia", r"low\s*blood\s*sugar"], "term": "Hypoglycemia", "snomed": "302866004", "icd10": "E11.65", "category": "diagnosis", "system": "endocrine"},
    {"patterns": [r"diabetic", r"diabetes", r"ketoacidosis"], "term": "Diabetic Emergency", "snomed": "73211009", "icd10": "E14.9", "category": "diagnosis", "system": "endocrine"},
    {"patterns": [r"back\s*pain", r"lower\s*back", r"lumbago"], "term": "Back Pain", "snomed": "279039007", "icd10": "M54.5", "category": "symptom", "system": "musculoskeletal"},
    {"patterns": [r"fracture", r"broken\s*(bone|arm|leg)"], "term": "Fracture", "snomed": "125605003", "icd10": "T14.8", "category": "diagnosis", "system": "musculoskeletal"},
    {"patterns": [r"suicide", r"suicidal", r"kill\s*myself", r"self[\-\s]*harm"], "term": "Suicidal Ideation", "snomed": "48652007", "icd10": "R45.851", "category": "symptom", "system": "psychiatric"},
    {"patterns": [r"anxiety", r"panic\s*attack"], "term": "Anxiety/Panic", "snomed": "197480006", "icd10": "F41.1", "category": "diagnosis", "system": "psychiatric"},
    {"patterns": [r"allergic\s*reaction", r"anaphyla", r"hives"], "term": "Allergic Reaction", "snomed": "39579001", "icd10": "T78.2", "category": "diagnosis", "system": "immunological"},
    {"patterns": [r"fever", r"temperature", r"febrile"], "term": "Fever", "snomed": "386661006", "icd10": "R50.9", "category": "symptom", "system": "infectious"},
    {"patterns": [r"bleeding", r"hemorrhage", r"uncontrolled\s*bleed"], "term": "Hemorrhage", "snomed": "50960005", "icd10": "R58", "category": "symptom", "system": "hematological"},
    {"patterns": [r"loss\s*of\s*consciousness", r"fainted", r"syncope", r"unresponsive"], "term": "Loss of Consciousness", "snomed": "418304008", "icd10": "R40.2", "category": "symptom", "system": "neurological"},
]

_SEVERITY_PATTERNS: list[tuple[str, str]] = [
    (r"severe|worst|intense|excruciating", "severe"),
    (r"sudden|acute|rapid|immediate", "acute"),
    (r"crushing|pressure|squeezing", "cardiac-risk"),
    (r"uncontrolled|continuous|persistent", "persistent"),
    (r"worse|getting\s*worse|deteriorating", "worsening"),
]


class ClinicalEntityExtractor:
    def extract(self, text: str) -> ExtractionResult:
        lower_text = text.lower()
        entities: list[ClinicalEntity] = []
        systems: set[str] = set()
        primary: str = ""
        primary_priority = 999

        for entry in _TERMINOLOGY:
            for pattern in entry["patterns"]:
                match = re.search(pattern, lower_text)
                if match:
                    if any(e.term == entry["term"] for e in entities):
                        continue
                    start = max(0, match.start() - 20)
                    end = min(len(text), match.end() + 20)
                    context = text[start:end].strip()
                    entity = ClinicalEntity(
                        term=entry["term"], snomed_code=entry["snomed"], icd10_code=entry["icd10"],
                        category=entry["category"], confidence=1.0, context=context,
                    )
                    entities.append(entity)
                    systems.add(entry["system"])
                    cat_priority = 0 if entry["category"] == "diagnosis" else 1
                    if cat_priority < primary_priority:
                        primary = entry["term"]
                        primary_priority = cat_priority
                    break

        severity_hints: list[str] = []
        for pattern, label in _SEVERITY_PATTERNS:
            if re.search(pattern, lower_text):
                severity_hints.append(label)

        return ExtractionResult(
            raw_text=text, entities=entities, primary_complaint=primary,
            body_systems=sorted(systems), severity_hints=severity_hints,
        )
```

---

### `src/ui/dashboard.py`

```python
"""
Streamlit observability dashboard for the clinical triage orchestrator.

Two-column layout:
  LEFT  — Patient chat interface (sends messages, polls for results)
  RIGHT — Observability trace (FSM state, entities, CoT reasoning, latency)

Run with:  streamlit run src/ui/dashboard.py --server.port 8501
"""

from __future__ import annotations

import json
import time
from typing import Any

import os

import httpx
import streamlit as st

DEFAULT_API = os.getenv("API_BASE_URL", "http://localhost:8080")

API_BASE_URL = st.sidebar.text_input(
    "API Base URL",
    value=DEFAULT_API,
    help="FastAPI orchestrator endpoint",
)

st.sidebar.divider()
st.sidebar.markdown("**Clinical Triage Orchestrator**")
st.sidebar.markdown("CPU-native agentic pipeline v0.3")

if "session_id" not in st.session_state:
    st.session_state.session_id = f"ui-{int(time.time()*1000) % 1_000_000:06d}"
if "messages" not in st.session_state:
    st.session_state.messages = []
if "last_result" not in st.session_state:
    st.session_state.last_result = None


def api_post(endpoint: str, payload: dict) -> dict | None:
    try:
        r = httpx.post(f"{API_BASE_URL}{endpoint}", json=payload, timeout=30)
        r.raise_for_status()
        return r.json()
    except httpx.HTTPError as e:
        st.error(f"API error: {e}")
        return None


def api_get(endpoint: str) -> dict | None:
    try:
        r = httpx.get(f"{API_BASE_URL}{endpoint}", timeout=10)
        if r.status_code == 202:
            return {"status": "processing"}
        if r.status_code == 404:
            return None
        r.raise_for_status()
        return r.json()
    except httpx.HTTPError as e:
        st.error(f"API error: {e}")
        return None


def poll_for_result(session_id: str, max_wait: float = 180.0) -> dict | None:
    deadline = time.time() + max_wait
    placeholder = st.sidebar.empty()
    while time.time() < deadline:
        result = api_get(f"/webhook/results/{session_id}")
        if result and result.get("status") == "completed":
            placeholder.empty()
            return result
        placeholder.info(f"Processing... ({time.time() - (deadline - max_wait):.1f}s)")
        time.sleep(0.5)
    placeholder.warning("Polling timed out — result may still be processing")
    return None


st.title("Clinical Triage Orchestrator")
st.caption(f"Session: `{st.session_state.session_id}`")

col_chat, col_trace = st.columns([1, 1])

with col_chat:
    st.subheader("Patient Chat")

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    st.markdown("**Quick scenarios:**")
    col1, col2 = st.columns(2)
    with col1:
        if st.button("Chest Pain", use_container_width=True):
            st.session_state.messages.append({"role": "user", "content": "I have severe chest pain and difficulty breathing"})
            st.rerun()
        if st.button("Stroke Symptoms", use_container_width=True):
            st.session_state.messages.append({"role": "user", "content": "My face is drooping on one side and I can't lift my arm"})
            st.rerun()
    with col2:
        if st.button("Mild Headache", use_container_width=True):
            st.session_state.messages.append({"role": "user", "content": "I have a mild headache and slight nausea"})
            st.rerun()
        if st.button("Emergency (Heart Attack)", use_container_width=True):
            st.session_state.messages.append({"role": "user", "content": "I think I'm having a heart attack"})
            st.rerun()

    if prompt := st.chat_input("Describe your symptoms..."):
        st.session_state.messages.append({"role": "user", "content": prompt})

        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            with st.spinner("Running triage pipeline..."):
                ack = api_post("/webhook/fulfillment", {
                    "session_id": st.session_state.session_id,
                    "query_text": prompt,
                    "patient_id": "UI-PATIENT",
                })

                if ack and ack.get("status") == "processing":
                    result = poll_for_result(st.session_state.session_id)
                else:
                    result = api_post("/webhook/triage", {
                        "session_id": st.session_state.session_id,
                        "query_text": prompt,
                    })

            if result:
                st.session_state.last_result = result
                resp = result.get("response", {})

                if resp.get("llm_bypassed"):
                    st.error(resp.get("message", "Emergency — call 911"))
                    assistant_msg = resp.get("message", "Emergency detected.")
                else:
                    urgency = resp.get("urgency", "unknown")
                    dept = resp.get("department", "unknown")
                    observations = resp.get("observations", [])
                    rationale = resp.get("rationale", [])

                    parts = [f"**Urgency:** {urgency} | **Route:** {dept}"]
                    if observations:
                        parts.append("**Observations:**")
                        for obs in observations:
                            parts.append(f"- {obs}")
                    if rationale:
                        parts.append("**Reasoning:**")
                        for step in rationale:
                            parts.append(f"1. {step}")

                    assistant_msg = "\n\n".join(parts)
                    st.markdown(assistant_msg)

                st.session_state.messages.append({"role": "assistant", "content": assistant_msg})

                latency = result.get("latency_ms", 0)
                if latency:
                    st.caption(f"Pipeline latency: {latency}ms")

with col_trace:
    st.subheader("Observability Trace")

    session_state = api_get(f"/webhook/session/{st.session_state.session_id}")

    if session_state:
        st.markdown("### Active State Node")
        current = session_state.get("current_state", "unknown")
        state_colors = {
            "intake": "blue_circle", "symptom_extraction": "yellow_circle",
            "guideline_lookup": "orange_circle", "risk_assessment": "red_circle",
            "triage_decision": "purple_circle", "escalation": "warning",
            "resolved": "check",
        }
        st.markdown(f"#### `{current}`")

        fsm_steps = ["intake", "symptom_extraction", "guideline_lookup", "risk_assessment", "triage_decision"]
        try:
            current_idx = fsm_steps.index(current)
            for i, step in enumerate(fsm_steps):
                if i < current_idx:
                    st.markdown(f"~~{step}~~")
                elif i == current_idx:
                    st.markdown(f"**{step}** (current)")
                else:
                    st.markdown(f"{step}")
        except ValueError:
            st.markdown(f"Current: `{current}`")

        st.divider()

        st.markdown("### SNOMED / ICD-10 Entities")
        result = st.session_state.last_result
        if result and "steps" in result:
            executor = result["steps"].get("D_executor", {})
            entities = executor.get("entities", [])
            if entities:
                for ent in entities:
                    with st.expander(f"**{ent['term']}** ({ent['category']})", expanded=True):
                        cols = st.columns(3)
                        cols[0].metric("SNOMED", ent.get("snomed_code", "—"))
                        cols[1].metric("ICD-10", ent.get("icd10_code", "—"))
                        cols[2].metric("Confidence", f"{ent.get('confidence', 0):.0%}")
            else:
                st.info("No entities extracted yet")

            st.divider()

            st.markdown("### LLM Diagnostic CoT")
            cognition = result["steps"].get("E_cognition", {})
            if cognition and "error" not in cognition:
                observations = cognition.get("clinical_observations", [])
                rationale = cognition.get("step_by_step_rationale", [])

                if observations:
                    st.markdown("**Clinical Observations:**")
                    for obs in observations:
                        st.markdown(f"- {obs}")

                if rationale:
                    st.markdown("**Step-by-step Rationale:**")
                    for i, step in enumerate(rationale, 1):
                        st.markdown(f"{i}. {step}")

                meta_cols = st.columns(3)
                meta_cols[0].metric("Urgency", cognition.get("urgency_level", "—"))
                meta_cols[1].metric("Department", cognition.get("recommended_department", "—"))
                meta_cols[2].metric("Confidence", f"{cognition.get('confidence', 0):.0%}")
            elif cognition.get("error"):
                st.warning(f"LLM step error: {cognition['error']}")
            else:
                st.info("No LLM reasoning yet — send a message to trigger the pipeline")

            st.divider()

            st.markdown("### Pipeline Latency")
            latency = result.get("latency_ms", 0)
            st.metric("Total Latency", f"{latency:.1f}ms")

            guardrail = result["steps"].get("B_guardrail", {})
            memory = result["steps"].get("C_memory", {})
            st.markdown(f"**Guardrail rules fired:** {', '.join(guardrail.get('triggered_rules', [])) or 'none'}")
            st.markdown(f"**Guidelines matched:** {memory.get('guidelines_found', 0)}")
        else:
            st.info("No trace data yet — send a message to start the pipeline")
    else:
        st.info("Session not found on server. Send a message to create one.")

    with st.expander("Raw result JSON", expanded=False):
        if st.session_state.last_result:
            st.json(st.session_state.last_result)
        else:
            st.code("No result yet")
```

---

### `requirements.txt`

```
fastapi>=0.110.0
uvicorn[standard]>=0.29.0
streamlit>=1.33.0
llama-cpp-python[server]>=0.2.56
starlette-context>=0.5.1
instructor>=1.2.0
openai>=1.25.0
qdrant-client>=1.8.0
redis>=5.0.0
pydantic>=2.7.0
httpx>=0.27.0
pyyaml>=6.0
pytest>=8.0.0
```

---

### `Dockerfile`

```dockerfile
FROM python:3.12-slim AS deps
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

FROM deps AS runtime
WORKDIR /app
COPY . .
RUN mkdir -p /app/models
EXPOSE 8080 8501
CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8080"]
```

---

### `docker-compose.yml`

```yaml
services:
  llama-cpp-server:
    image: ghcr.io/ggerganov/llama.cpp:server
    container_name: triage-llm-server
    ports:
      - "8000:8000"
    volumes:
      - ./models:/models:ro
    command: >
      --model /models/gemma-3n-E4B-it-Q4_K_M.gguf
      --host 0.0.0.0 --port 8000 --n-ctx 4096 --n-threads 4
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8000/v1/models"]
      interval: 10s
      timeout: 5s
      retries: 5
      start_period: 30s
    deploy:
      resources:
        limits:
          memory: 4G

  redis:
    image: redis:7-alpine
    container_name: triage-redis
    ports:
      - "6379:6379"
    volumes:
      - redis-data:/data
    healthcheck:
      test: ["CMD", "redis-cli", "ping"]
      interval: 5s
      timeout: 3s
      retries: 3

  orchestrator-api:
    build:
      context: .
      dockerfile: Dockerfile
    container_name: triage-api
    ports:
      - "8080:8080"
    environment:
      - LLM_BASE_URL=http://llama-cpp-server:8000/v1
      - LLM_MODEL=local-model
      - LLM_API_KEY=not-needed
      - REDIS_URL=redis://redis:6379/0
      - LOG_LEVEL=INFO
    depends_on:
      llama-cpp-server:
        condition: service_healthy
      redis:
        condition: service_healthy
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8080/health"]
      interval: 10s
      timeout: 5s
      retries: 3
      start_period: 15s

  streamlit-ui:
    build:
      context: .
      dockerfile: Dockerfile
    container_name: triage-ui
    ports:
      - "8501:8501"
    environment:
      - API_BASE_URL=http://orchestrator-api:8080
    depends_on:
      orchestrator-api:
        condition: service_healthy
    command: >
      streamlit run src/ui/dashboard.py
      --server.port 8501
      --server.address 0.0.0.0
      --server.headless true
      --browser.gatherUsageStats false

volumes:
  redis-data:
```

---

### `setup.sh`

```bash
#!/usr/bin/env bash
set -euo pipefail

VENV_DIR=".venv"
MODEL_DIR="models"
MODEL_FILE="gemma-3n-E4B-it-Q4_K_M.gguf"

echo "Creating virtual environment in ${VENV_DIR}/ ..."
python3 -m venv "${VENV_DIR}"
source "${VENV_DIR}/bin/activate"

echo "Installing Python dependencies ..."
pip install --upgrade pip setuptools wheel
pip install -r requirements.txt

mkdir -p "${MODEL_DIR}"
if [ ! -f "${MODEL_DIR}/${MODEL_FILE}" ]; then
    echo "Model not found at ${MODEL_DIR}/${MODEL_FILE}"
    echo "  Place your GGUF model file in ${MODEL_DIR}/ and update MODEL_FILE in setup.sh"
    exit 1
fi

echo "Verifying installation ..."
python -c "import fastapi; import instructor; import qdrant_client; print('All imports OK')"
echo ""
echo "Setup complete. Activate the environment with:"
echo "  source ${VENV_DIR}/bin/activate"
echo ""
echo "Start the local LLM server with:"
echo "  python -m llama_cpp.server --model ${MODEL_DIR}/${MODEL_FILE} --port 8000"
```

---

*End of specification.*
