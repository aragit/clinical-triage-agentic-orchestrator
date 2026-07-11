"""
Centralised configuration for the clinical triage orchestrator.

All tuneable knobs live here.  In production these would be served by
GCP Secret Manager / Vertex AI config; the local twin reads them from
env vars or falls back to sane defaults.
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
