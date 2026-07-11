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
