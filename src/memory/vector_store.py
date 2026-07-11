"""
Hybrid vector store backed by Qdrant in-memory mode.

Provides dual-mode retrieval: dense semantic search and sparse BM25-style
keyword search, fused via reciprocal rank fusion (RRF) for clinical guideline
lookup.  The in-memory backend requires zero infrastructure — perfect for the
local simulation twin while mirroring the exact API surface used in production
Qdrant clusters.
"""

from __future__ import annotations

import math
import uuid
from collections import Counter
from dataclasses import dataclass, field
from typing import Any

from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    FieldCondition,
    Filter,
    MatchValue,
    PointStruct,
    VectorParams,
)


# ── Schema ──────────────────────────────────────────────────────────

@dataclass
class ClinicalGuideline:
    """A single clinical guideline document."""

    guideline_id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    title: str = ""
    content: str = ""
    specialty: str = ""
    icd10_codes: list[str] = field(default_factory=list)
    snomed_codes: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


# ── Collection config ───────────────────────────────────────────────

COLLECTION = "clinical_guidelines"
DENSE_DIM = 384  # all-MiniLM-L6-v2 output dim (default in Qdrant cloud)
SPARSE_FIELD = "content"  # field indexed for BM25-style retrieval


# ── Hybrid store ────────────────────────────────────────────────────

class HybridVectorStore:
    """Qdrant in-memory hybrid (dense + sparse) vector store.

    Production equivalent: GCP Vertex AI Vector Search / managed Qdrant.
    This local twin uses ``QdrantClient(":memory:")`` so the entire
    retrieval pipeline runs on a laptop with zero cloud dependencies.

    Sparse search uses an in-process Python BM25 implementation because
    Qdrant's in-memory mode does not support payload text indexes.  The
    production version would use Qdrant's native sparse vector support
    or a dedicated Elasticsearch / OpenSearch cluster.
    """

    def __init__(self) -> None:
        self._client = QdrantClient(":memory:")
        self._sparse_index: list[dict[str, Any]] = []  # in-process BM25 corpus
        self._ensure_collection()

    # ── collection bootstrap ────────────────────────────────────────

    def _ensure_collection(self) -> None:
        collections = [c.name for c in self._client.get_collections().collections]
        if COLLECTION in collections:
            return
        self._client.create_collection(
            collection_name=COLLECTION,
            vectors_config=VectorParams(
                size=DENSE_DIM,
                distance=Distance.COSINE,
            ),
        )

    # ── ingest ──────────────────────────────────────────────────────

    def insert_guideline(self, guideline: ClinicalGuideline) -> str:
        """Insert a clinical guideline.  Returns the guideline ID."""
        point_id = uuid.uuid4().int >> 64  # unsigned 64-bit

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
            points=[
                PointStruct(id=point_id, vector=dense_vector, payload=payload),
            ],
        )
        # Maintain the in-process sparse index for BM25 retrieval.
        self._sparse_index.append({
            "guideline_id": guideline.guideline_id,
            "title": guideline.title,
            "content": guideline.content,
            "specialty": guideline.specialty,
            "payload": payload,
        })

        return guideline.guideline_id

    def insert_guidelines(self, guidelines: list[ClinicalGuideline]) -> list[str]:
        """Batch insert.  Returns list of guideline IDs."""
        return [self.insert_guideline(g) for g in guidelines]

    # ── retrieval ───────────────────────────────────────────────────

    def search_dense(
        self,
        query: str,
        limit: int = 5,
        filter_specialty: str | None = None,
    ) -> list[dict[str, Any]]:
        """Pure dense (semantic) search via Qdrant."""
        qvector = _pseudo_embed(query, DENSE_DIM)
        query_filter = _build_filter(specialty=filter_specialty)
        results = self._client.query_points(
            collection_name=COLLECTION,
            query=qvector,
            query_filter=query_filter,
            limit=limit,
        )
        return _format_results(results)

    def search_sparse(
        self,
        query: str,
        limit: int = 5,
        filter_specialty: str | None = None,
    ) -> list[dict[str, Any]]:
        """BM25 keyword search against the in-process sparse index.

        Implements Okapi BM25 scoring (k1=1.5, b=0.75) so the local twin
        produces ranked keyword results identical in spirit to what a
        production Elasticsearch or Qdrant sparse-vector cluster would return.
        """
        hits = _bm25_search(
            query=query,
            corpus=self._sparse_index,
            filter_specialty=filter_specialty,
            limit=limit,
        )
        return [{**h["payload"], "_score": h["score"]} for h in hits]

    def search_hybrid(
        self,
        query: str,
        limit: int = 5,
        filter_specialty: str | None = None,
        rrf_k: int = 60,
    ) -> list[dict[str, Any]]:
        """Hybrid search fusing dense + sparse via Reciprocal Rank Fusion.

        This mirrors the production pattern where Vertex AI Vector Search
        handles dense retrieval and a separate BM25 index handles keyword
        recall — combined server-side with RRF.
        """
        dense = self.search_dense(query, limit=limit * 2, filter_specialty=filter_specialty)
        sparse = self.search_sparse(query, limit=limit * 2, filter_specialty=filter_specialty)

        # RRF scoring: score(d) = sum 1/(k + rank_i(d)) across engines
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

    # ── stats ───────────────────────────────────────────────────────

    def count(self) -> int:
        return self._client.count(collection_name=COLLECTION).count


# ── Helpers ─────────────────────────────────────────────────────────


def _build_filter(specialty: str | None = None) -> Filter | None:
    if not specialty:
        return None
    return Filter(
        must=[FieldCondition(key="specialty", match=MatchValue(value=specialty))]
    )


def _pseudo_embed(text: str, dim: int) -> list[float]:
    """Deterministic pseudo-embedding for offline testing.

    In production this would be a real sentence-transformer call.
    Here we hash-character-fingerprint so identical texts always produce
    identical vectors — enough for structural correctness of the pipeline.
    """
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


# ── In-process BM25 ────────────────────────────────────────────────

_BM25_K1 = 1.5
_BM25_B = 0.75


def _tokenize(text: str) -> list[str]:
    """Lowercase word-tokenizer with 2+ char minimum."""
    return [
        tok
        for tok in text.lower().split()
        if len(tok) >= 2 and tok.isalnum()
    ]


def _bm25_search(
    query: str,
    corpus: list[dict[str, Any]],
    filter_specialty: str | None = None,
    limit: int = 5,
) -> list[dict[str, Any]]:
    """Okapi BM25 over the in-process corpus."""
    if not corpus:
        return []

    # Pre-tokenize the corpus
    tokenized_corpus = [_tokenize(doc["content"]) for doc in corpus]
    doc_count = len(tokenized_corpus)
    avgdl = sum(len(d) for d in tokenized_corpus) / doc_count if doc_count else 1.0

    # Document frequency for each term
    df: Counter[str] = Counter()
    for doc_toks in tokenized_corpus:
        for tok in set(doc_toks):
            df[tok] += 1

    query_tokens = _tokenize(query)
    scores: list[tuple[int, float]] = []

    for idx, doc_toks in enumerate(tokenized_corpus):
        # Optional specialty filter
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
            tf_norm = (term_freq * (_BM25_K1 + 1)) / (
                term_freq
                + _BM25_K1 * (1 - _BM25_B + _BM25_B * doc_len / avgdl)
            )
            score += idf * tf_norm

        if score > 0:
            scores.append((idx, score))

    scores.sort(key=lambda x: x[1], reverse=True)
    return [
        {
            "guideline_id": corpus[idx]["guideline_id"],
            "payload": corpus[idx]["payload"],
            "score": sc,
        }
        for idx, sc in scores[:limit]
    ]
