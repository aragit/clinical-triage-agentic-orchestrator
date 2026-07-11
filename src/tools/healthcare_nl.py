"""
Clinical NLP entity extraction with mocked SNOMED CT / ICD-10 code lookup.

Production equivalent: a Vertex AI NLP pipeline calling Google Healthcare
NLP API (or AWS Comprehend Medical) for entity extraction, then resolving
entities against a FHIR-backed terminology server.

This local twin implements the exact same contract — free-text in,
structured clinical entities out — but uses a deterministic keyword+regex
engine with a curated dictionary of ~40 common emergency/primary-care
presentations.  This is the Executor/Tools layer of the agentic loop.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any


# ── Clinical entity schema ──────────────────────────────────────────

@dataclass
class ClinicalEntity:
    """A single extracted clinical entity with coded references."""

    term: str
    snomed_code: str
    icd10_code: str
    category: str  # symptom | diagnosis | medication | procedure
    confidence: float = 1.0  # 0.0-1.0, deterministic in local twin
    context: str = ""  # surrounding text snippet


@dataclass
class ExtractionResult:
    """Full extraction output matching the production NLP API contract."""

    raw_text: str
    entities: list[ClinicalEntity] = field(default_factory=list)
    primary_complaint: str = ""
    body_systems: list[str] = field(default_factory=list)
    severity_hints: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "raw_text": self.raw_text,
            "entities": [
                {
                    "term": e.term,
                    "snomed_code": e.snomed_code,
                    "icd10_code": e.icd10_code,
                    "category": e.category,
                    "confidence": e.confidence,
                }
                for e in self.entities
            ],
            "primary_complaint": self.primary_complaint,
            "body_systems": self.body_systems,
            "severity_hints": self.severity_hints,
        }


# ── Terminology dictionary ──────────────────────────────────────────
# Curated SNOMED CT + ICD-10 mappings for common clinical presentations.
# In production this would be a FHIR terminology server lookup.

_TERMINOLOGY: list[dict[str, Any]] = [
    # Cardiac
    {
        "patterns": [r"chest\s*pain", r"chest\s*tight", r"angina", r"chest\s*discomfort"],
        "term": "Chest Pain",
        "snomed": "29857009",
        "icd10": "R07.9",
        "category": "symptom",
        "system": "cardiovascular",
    },
    {
        "patterns": [r"heart\s*attack", r"myocardial\s*infarction", r"mi\b", r"stemi", r"nstemi"],
        "term": "Myocardial Infarction",
        "snomed": "22298006",
        "icd10": "I21.9",
        "category": "diagnosis",
        "system": "cardiovascular",
    },
    {
        "patterns": [r"palpitation", r"heart\s*racing", r"heart\s*pounding", r"irregular\s*heartbeat"],
        "term": "Palpitations",
        "snomed": "426783006",
        "icd10": "R00.2",
        "category": "symptom",
        "system": "cardiovascular",
    },
    # Respiratory
    {
        "patterns": [r"shortness\s*of\s*breath", r"dyspnea", r"breathing\s*difficult", r"sob\b", r"can'?t\s*breathe"],
        "term": "Dyspnea",
        "snomed": "267036007",
        "icd10": "R06.02",
        "category": "symptom",
        "system": "respiratory",
    },
    {
        "patterns": [r"asthma", r"asthma\s*attack", r"bronchospasm", r"wheez"],
        "term": "Asthma Exacerbation",
        "snomed": "195967001",
        "icd10": "J45.909",
        "category": "diagnosis",
        "system": "respiratory",
    },
    {
        "patterns": [r"cough(?:ing)?", r"persistent\s*cough", r"hacking\s*cough"],
        "term": "Cough",
        "snomed": "49727002",
        "icd10": "R05.9",
        "category": "symptom",
        "system": "respiratory",
    },
    # Neurological
    {
        "patterns": [r"headache", r"migraine", r"head\s*pain", r"throbbing\s*head"],
        "term": "Headache",
        "snomed": "25064002",
        "icd10": "R51.9",
        "category": "symptom",
        "system": "neurological",
    },
    {
        "patterns": [r"dizziness", r"vertigo", r"lightheaded", r"dizzy", r"feeling\s*faint"],
        "term": "Dizziness",
        "snomed": "404640003",
        "icd10": "R42",
        "category": "symptom",
        "system": "neurological",
    },
    {
        "patterns": [r"stroke", r"facial\s*droop", r"arm\s*weakness", r"slurred\s*speech", r"fast\s*assessment"],
        "term": "Stroke",
        "snomed": "230690007",
        "icd10": "I63.9",
        "category": "diagnosis",
        "system": "neurological",
    },
    {
        "patterns": [r"seizure", r"convulsion", r"grand\s*mal", r"tonic[\-\s]*clonic"],
        "term": "Seizure",
        "snomed": "371631005",
        "icd10": "R56.9",
        "category": "symptom",
        "system": "neurological",
    },
    # Gastrointestinal
    {
        "patterns": [r"abdominal\s*pain", r"stomach\s*pain", r"belly\s*pain", r"stomach\s*ache", r"abdominal\s*cramp"],
        "term": "Abdominal Pain",
        "snomed": "271803005",
        "icd10": "R10.9",
        "category": "symptom",
        "system": "gastrointestinal",
    },
    {
        "patterns": [r"nausea", r"feeling\s*sick", r"queasy"],
        "term": "Nausea",
        "snomed": "422587007",
        "icd10": "R11.0",
        "category": "symptom",
        "system": "gastrointestinal",
    },
    {
        "patterns": [r"vomit", r"emesis", r"throwing\s*up"],
        "term": "Vomiting",
        "snomed": "422400008",
        "icd10": "R11.10",
        "category": "symptom",
        "system": "gastrointestinal",
    },
    # Endocrine
    {
        "patterns": [r"hypoglycemia", r"low\s*blood\s*sugar", r"blood\s*sugar\s*drop", r"sugar\s*crash"],
        "term": "Hypoglycemia",
        "snomed": "302866004",
        "icd10": "E11.65",
        "category": "diagnosis",
        "system": "endocrine",
    },
    {
        "patterns": [r"diabetic", r"diabetes", r"hyperglycemia", r"high\s*blood\s*sugar", r"ketoacidosis"],
        "term": "Diabetic Emergency",
        "snomed": "73211009",
        "icd10": "E14.9",
        "category": "diagnosis",
        "system": "endocrine",
    },
    # Musculoskeletal
    {
        "patterns": [r"back\s*pain", r"lower\s*back", r"lumbago", r"spinal\s*pain"],
        "term": "Back Pain",
        "snomed": "279039007",
        "icd10": "M54.5",
        "category": "symptom",
        "system": "musculoskeletal",
    },
    {
        "patterns": [r"fracture", r"broken\s*(bone|arm|leg|rib|wrist|ankle)"],
        "term": "Fracture",
        "snomed": "125605003",
        "icd10": "T14.8",
        "category": "diagnosis",
        "system": "musculoskeletal",
    },
    # Mental health
    {
        "patterns": [r"suicide", r"suicidal", r"kill\s*myself", r"end\s*my\s*life", r"want\s*to\s*die", r"self[\-\s]*harm"],
        "term": "Suicidal Ideation",
        "snomed": "48652007",
        "icd10": "R45.851",
        "category": "symptom",
        "system": "psychiatric",
    },
    {
        "patterns": [r"anxiety", r"panic\s*attack", r"anxious", r"panicking"],
        "term": "Anxiety/Panic",
        "snomed": "197480006",
        "icd10": "F41.1",
        "category": "diagnosis",
        "system": "psychiatric",
    },
    # Allergic
    {
        "patterns": [r"allergic\s*reaction", r"anaphyla", r"hives", r"swollen\s*(throat|tongue|lip)"],
        "term": "Allergic Reaction",
        "snomed": "39579001",
        "icd10": "T78.2",
        "category": "diagnosis",
        "system": "immunological",
    },
    # Infectious
    {
        "patterns": [r"fever", r"temperature", r"high\s*temp", r"febrile"],
        "term": "Fever",
        "snomed": "386661006",
        "icd10": "R50.9",
        "category": "symptom",
        "system": "infectious",
    },
    # Trauma
    {
        "patterns": [r"bleeding", r"hemorrhage", r"blood\s*loss", r"uncontrolled\s*bleed"],
        "term": "Hemorrhage",
        "snomed": "50960005",
        "icd10": "R58",
        "category": "symptom",
        "system": "hematological",
    },
    {
        "patterns": [r"loss\s*of\s*consciousness", r"fainted", r"passed\s*out", r"syncope", r"unresponsive"],
        "term": "Loss of Consciousness",
        "snomed": "418304008",
        "icd10": "R40.2",
        "category": "symptom",
        "system": "neurological",
    },
]

# Severity escalation keywords — presence bumps severity hints.
_SEVERITY_PATTERNS: list[tuple[str, str]] = [
    (r"severe|worst|intense|excruciating|unbearable", "severe"),
    (r"sudden|acute|rapid|immediate|emergency", "acute"),
    (r"crushing|pressure|squeezing", "cardiac-risk"),
    (r"uncontrolled|continuous|persistent", "persistent"),
    (r"worse|getting\s*worse|deteriorating", "worsening"),
]


# ── Extractor ───────────────────────────────────────────────────────

class ClinicalEntityExtractor:
    """Deterministic clinical entity extraction with SNOMED/ICD-10 coding.

    Production equivalent: Google Healthcare NLP API / AWS Comprehend Medical
    backed by a FHIR terminology server for code resolution.

    This local twin provides the exact same output schema — structured
    clinical entities with standardized codes — using a keyword+regex engine
    against a curated terminology dictionary.  This is deterministic and
    requires zero cloud keys or GPU.
    """

    def extract(self, text: str) -> ExtractionResult:
        """Extract clinical entities from free-text input.

        Returns an ExtractionResult with coded entities, body systems,
        and severity hints — the same contract as the production NLP API.
        """
        lower_text = text.lower()
        entities: list[ClinicalEntity] = []
        systems: set[str] = set()
        primary: str = ""
        primary_priority = 999  # lower = higher priority for primary complaint

        for entry in _TERMINOLOGY:
            for pattern in entry["patterns"]:
                match = re.search(pattern, lower_text)
                if match:
                    # Check for duplicates
                    if any(e.term == entry["term"] for e in entities):
                        continue

                    # Grab context window around match
                    start = max(0, match.start() - 20)
                    end = min(len(text), match.end() + 20)
                    context = text[start:end].strip()

                    entity = ClinicalEntity(
                        term=entry["term"],
                        snomed_code=entry["snomed"],
                        icd10_code=entry["icd10"],
                        category=entry["category"],
                        confidence=1.0,
                        context=context,
                    )
                    entities.append(entity)
                    systems.add(entry["system"])

                    # Primary complaint selection: diagnoses > symptoms,
                    # prioritised by order in the terminology list
                    cat_priority = 0 if entry["category"] == "diagnosis" else 1
                    if cat_priority < primary_priority:
                        primary = entry["term"]
                        primary_priority = cat_priority
                    break  # one match per entry is enough

        # Severity hints
        severity_hints: list[str] = []
        for pattern, label in _SEVERITY_PATTERNS:
            if re.search(pattern, lower_text):
                severity_hints.append(label)

        return ExtractionResult(
            raw_text=text,
            entities=entities,
            primary_complaint=primary,
            body_systems=sorted(systems),
            severity_hints=severity_hints,
        )
