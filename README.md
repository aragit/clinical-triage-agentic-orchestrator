<p align="center">
  <h1 align="center">Clinical Triage Agentic Orchestrator</h1>
  <p align="center">An enterprise-grade, localized Neuro-Symbolic Agentic AI Orchestrator for high-stakes clinical triage.</p>
</p>

| Layer | Technology |
|---|---|
| **Backend Framework** | FastAPI + Uvicorn |
| **Frontend / Observability** | Streamlit |
| **LLM Inference** | llama-cpp-python (local GGUF) |
| **Structured Output** | instructor + Pydantic |
| **Model** | Gemma 3n E4B (Q4_K_M quantized) |
| **Vector Store** | Qdrant (in-memory, Hybrid BM25 + Dense + RRF) |
| **Policy Engine** | Custom OPA-style deterministic guardrails |
| **Clinical NLP** | SNOMED CT / ICD-10-CM entity extraction |
| **State Management** | Atomic FSM (7 nodes, session-scoped) |
| **Async Processing** | FastAPI BackgroundTasks |
| **Containerization** | Docker + Docker Compose |
| **Language** | Python 3.12+ |

---

Operating entirely on consumer-grade hardware (Intel i5 CPU, 16GB RAM) without external cloud APIs or GPU dependencies, this system implements a **deterministic-to-probabilistic execution boundary**. By combining an isolated policy guardrail, real-time medical ontology extraction (**SNOMED CT / ICD-10-CM**), an in-memory hybrid vector database, an atomic Finite State Machine (FSM), and grammar-constrained LLM decoding, this architecture guarantees safety, eliminates hallucination risks for critical presentations, and maintains a zero-latency fast-path for medical emergencies.

---

## Architecture & Data Flow

The system employs a **dual-pathway execution pattern** (Fast-Path vs. Slow-Path) to process clinical input safely and efficiently:

```
                        ┌─────────────────────────────┐
                        │   Patient Input              │
                        │   (Streamlit UI / Webhook)   │
                        └──────────────┬──────────────┘
                                       │
                                       ▼
                        ┌─────────────────────────────┐
                        │   Step A: Perception         │
                        │   Episodic History Retrieval │
                        └──────────────┬──────────────┘
                                       │
                                       ▼
                        ┌─────────────────────────────┐
                        │   Step B: OPA Policy Engine  │
                        │   Deterministic Guardrails   │
                        └──────────────┬──────────────┘
                                       │
                 ┌─────────────────────┴─────────────────────┐
                 │                                           │
                 ▼ EMERGENCY                                 ▼ NON-EMERGENT
    ┌─────────────────────────────┐           ┌─────────────────────────────┐
    │   FAST-PATH: SHORT-CIRCUIT  │           │   SLOW-PATH: COGNITIVE LOOP │
    └──────────────┬──────────────┘           └──────────────┬──────────────┘
                   │                                         │
                   ▼                                         ▼
    ┌─────────────────────────────┐           ┌─────────────────────────────┐
    │  Step D: Ontology Extraction│           │  Step C: Context Retrieval  │
    │  SNOMED CT / ICD-10 Coding  │           │  Qdrant Hybrid Dense+BM25   │
    └──────────────┬──────────────┘           └──────────────┬──────────────┘
                   │                                         │
                   ▼                                         ▼
    ┌─────────────────────────────┐           ┌─────────────────────────────┐
    │  Step F: State Persistence  │           │  Step D: Ontology Extraction│
    │  Transition to ESCALATION   │           │  Extract Clinical Entities  │
    └──────────────┬──────────────┘           └──────────────┬──────────────┘
                   │                                         │
                   ▼                                         ▼
    ┌─────────────────────────────┐           ┌─────────────────────────────┐
    │  Immediate Response         │           │  Step E: Structured Cognition│
    │  Latency: < 30ms            │           │  llama-cpp + Gemma 3n        │
    └─────────────────────────────┘           │  instructor Grammar Forcing  │
                                              └──────────────┬──────────────┘
                                                             │
                                                             ▼
                                              ┌─────────────────────────────┐
                                              │  Step F: FSM State Move     │
                                              │  Triage Decision            │
                                              └──────────────┬──────────────┘
                                                             │
                                                             ▼
                                              ┌─────────────────────────────┐
                                              │  Structured Output JSON     │
                                              │  Latency: ~1.2s - 2.5s      │
                                              └─────────────────────────────┘
```

---

## Core Components

1. **Perception Layer (`src/api/` & `src/ui/`)**: A high-performance FastAPI backend paired with a comprehensive Streamlit observability dashboard. Implements asynchronous webhook decoupling via FastAPI `BackgroundTasks` to achieve low-latency (<150ms) HTTP network acknowledgments, protecting against connection drops in consumer channels like Dialogflow CX.

2. **Deterministic Guardrails (`src/core/opa_policies.py`)**: An isolated rule engine written in strict adherence to Open Policy Agent concepts. It scans text for acute clinical triggers (cardiovascular, respiratory, neurological, or psychiatric distress) prior to hitting the LLM tier.

3. **Medical Ontology Engine (`src/tools/healthcare_nl.py`)**: A deterministic terminology extractor mapping presentation variables to standardized healthcare vocabularies (**SNOMED CT Concepts** and **ICD-10-CM Codes**). It guarantees structured downstream tracking packets, even during emergency overrides.

4. **Episodic Memory Store (`src/memory/episodic_state.py`)**: An atomic Finite State Machine (FSM) managing 7 distinct clinical state nodes (`intake`, `symptom_extraction`, `guideline_lookup`, `risk_assessment`, `triage_decision`, `escalation`, `resolved`). It prevents conversational state drift and enforces structural integrity over session lifetimes.

5. **Context Memory Engine (`src/memory/vector_store.py`)**: An in-memory Qdrant client implementing **Hybrid Search** (Dense Vector Embedding simulation + Sparse BM25 text tracking) joined via Reciprocal Rank Fusion (RRF) to pull clinical protocols in real time.

6. **Structured Cognition Loop (`src/cognition/triage_agent.py`)**: The probabilistic core that wraps a local `llama-cpp-python` engine (running a quantized `Gemma 3n E4B` model) using the `instructor` framework. It utilizes context-free grammars to force pure JSON extraction into a typed Pydantic schema containing deep Chain-of-Thought (`DiagnosticCoT`) fields.

---

## Production Verification & Performance Metrics

### Scenario: Acute Emergency Ingestion
When a user provides an active high-risk presentation string:
> **Patient:** *"I have severe chest pain and difficulty breathing"*

The deterministic policy engine intercepts the input instantly, extracts relevant medical codes, transitions the state machine, and bypasses the local LLM entirely:

* **Total Pipeline Latency:** **24.4ms** (compared to 90+ seconds for raw CPU LLM generation).
* **FSM Transition:** Successfully moves state persistence from `intake` directly to `escalation (current)`.
* **Ontology Extraction Alignment:** Resolves the complaint to **SNOMED ID: `29857009` (Chest Pain)** and **ICD-10-CM ID: `R07.9`** with a 100% confidence index.
* **LLM Short-Circuit Execution:** Logs `llm_bypassed: true`, saving critical compute cycles and removing prompt injection risks during life-threatening events.

#### Raw Engine Output Verification Payload:
```json
{
  "session_id": "ui-519898",
  "status": "completed",
  "response": {
    "message": "EMERGENCY: Your input indicates a potentially life-threatening condition. Do NOT wait for an AI assessment. Call emergency services (911) immediately or go to the nearest emergency room.",
    "triggered_rules": [
      "cardiac-emergency",
      "cardiac-emergency",
      "respiratory-emergency"
    ],
    "action": "route_to_emergency",
    "llm_bypassed": true,
    "extracted_entities": {
      "raw_text": "I have severe chest pain and difficulty breathing",
      "entities": [
        {
          "term": "Chest Pain",
          "snomed_code": "29857009",
          "icd10_code": "R07.9",
          "category": "symptom",
          "confidence": 1
        }
      ],
      "primary_complaint": "Chest Pain",
      "body_systems": ["cardiovascular"],
      "severity_hints": ["severe"]
    }
  },
  "state_transition": "escalation",
  "latency_ms": 24.4
}
```

### UI Walkthrough

![Triage Dashboard — Initial State](assets/triage_1.png)

![Triage Dashboard — Emergency Bypass](assets/triage2.png)

![Triage Dashboard — Trace & Entities](assets/triage3.png)

---

## Repository Structure

```
clinical-triage-agentic-orchestrator/
├── Dockerfile                  # Multi-stage production Python runtime
├── docker-compose.yml          # Local infra stack orchestration
├── requirements.txt            # Dependency manifest
├── setup.sh                    # Automation environment and dependency compiler
├── README.md                   # Project documentation
└── src/
    ├── __init__.py
    ├── core/
    │   ├── __init__.py
    │   ├── config.py           # Frozen AppConfig system containing ENV configurations
    │   └── opa_policies.py     # Deterministic Policy Guardrails
    ├── memory/
    │   ├── __init__.py
    │   ├── vector_store.py     # In-memory Qdrant Hybrid RRF context client
    │   └── episodic_state.py   # Atomic State Machine (FSM) session engine
    ├── cognition/
    │   ├── __init__.py
    │   ├── llm_client.py       # Instructor Client bindings for Local Llama-cpp
    │   └── triage_agent.py     # Main Neuro-Symbolic Pipeline Orchestrator
    ├── tools/
    │   ├── __init__.py
    │   └── healthcare_nl.py    # SNOMED CT & ICD-10 Ontology Mapping Engine
    ├── api/
    │   ├── __init__.py
    │   ├── main.py             # FastAPI App Lifespan and Dependency Singleton Manager
    │   └── webhook.py          # Decoupled Asynchronous /fulfillment endpoints
    └── ui/
        ├── __init__.py
        └── dashboard.py        # Streamlit Dual-Column Observability GUI
```

---

## Local Installation & Environment Setup

This project is built to run natively on Linux/Ubuntu systems utilizing an isolated environment profile.

### Prerequisites

Ensure your local host has Python 3.12+ and compiler essentials installed:

```bash
sudo apt-get update && sudo apt-get install -y build-essential python3-dev
```

### 1. Scripted Environment Bootstrapping

Clone the target repository and execute the automated environment config compiler:

```bash
git clone https://github.com/aragit/clinical-triage-agentic-orchestrator.git
cd clinical-triage-agentic-orchestrator
chmod +x setup.sh
./setup.sh
```

> **Note:** The script prepares the `.venv`, compiles base binaries, and builds project structures.

### 2. Positioning the Model File

Download a quantized GGUF format model (Highly recommended: `gemma-3n-E4B-it-Q4_K_M.gguf` or an equivalent open-source SLM) and place it directly into your local `models/` directory:

```bash
mkdir -p models/
# Move or download your target GGUF file here
mv /path/to/gemma-3n-E4B-it-Q4_K_M.gguf models/
```

---

## Running the Orchestrator Stack

### Method A: Native Process Isolation (Recommended for Active Local Dev)

Open three separate terminal windows to monitor individual process standard outputs:

**Terminal 1: Start the Local LLM Core Inference Engine**

```bash
source .venv/bin/activate
python -m llama_cpp.server \
  --model models/gemma-3n-E4B-it-Q4_K_M.gguf \
  --host 0.0.0.0 \
  --port 8000 \
  --n_ctx 4096 \
  --chat_format gemma \
  --threads 4
```

> **Tip:** Clamping `--threads` directly to match your physical CPU cores maximizes execution performance on local CPUs.

**Terminal 2: Run the High-Performance FastAPI Backend Gateway**

```bash
source .venv/bin/activate
uvicorn src.api.main:app --host 0.0.0.0 --port 8080 --reload
```

**Terminal 3: Boot the Streamlit Operational Dashboard**

```bash
source .venv/bin/activate
streamlit run src/ui/dashboard.py --server.port 8501
```

### Method B: Containerized Docker Compose Scaffolding

To spin up the entire isolated framework—including the network gateway, local server definitions, and UI containers—using multi-stage configurations:

```bash
docker compose up --build
```

Access the systems via the following local allocations:

| Service | URL |
|---|---|
| Streamlit Operational Center | http://localhost:8501 |
| FastAPI Swagger UI | http://localhost:8080/docs |
| LLM Inference Server | http://localhost:8000 |

---

## Observability & Verification Testing

1. Direct your browser window context to the Streamlit layout dashboard (http://localhost:8501).
2. Examine the right-hand column: The **Active State Node** list shows the active FSM tracking, initialized at `intake`.
3. Use the chat window to enter a critical presentation phrase (e.g., *"I have severe chest pain"*).
4. Review the structural updates: The trace transitions immediately to `escalation`, the ontology matrix registers the correct SNOMED/ICD codes, and the pipeline displays sub-30ms performance execution.
5. Provide a benign presentation description (e.g., *"My left big toe has a mild itch since last night"*). The interface will correctly process this through the slow-path, engaging the local LLM core to parse structured JSON arrays via instructor.

---

## Engineering Principles

1. **The Safe-Fail Ingestion Boundary** — An isolated, deterministic policy engine (mirroring Open Policy Agent concepts) intercepts requests prior to reaching the probabilistic LLM layer. High-risk vectors are instantly mitigated with 0% chance of model hallucination.

2. **Asynchronous Webhook Decoupling** — To accommodate strict real-time third-party engine constraints (like Dialogflow's <5-second limits), the intake transaction is decoupled via FastAPI `BackgroundTasks`. The system immediately provides a low-latency acknowledgment to the network while processing complex local data graphs out-of-band.

3. **Structured Grammar Forcing** — A reliable neuro-symbolic bridge using `instructor` over local `llama-cpp-python` setups. This configuration forces open-source quantized models (like Gemma 3n) to strictly follow typed Pydantic structures on consumer CPU limits.

4. **Resilient Local State Failure Routing** — Strict fallback mechanisms ensure that if structural parsing errors occur during low-compute generation windows, the system automatically over-triages to prevent critical health downgrades.

---

## Evolutionary Roadmap: Plan-Then-Execute (PTE) Architecture

This repository serves as a high-performance baseline for localized, resource-constrained agentic execution. To transition from the current rigid, linear pipeline to a highly adaptive **Plan-Then-Execute (PTE)** framework, the architecture decouples high-level cognitive planning from low-level deterministic tool execution. 

Under this paradigm, the Large Language Model (LLM) acts purely as a **Workflow Compiler** that ingests unstructured patient data and outputs an explicit Directed Acyclic Graph (DAG) plan. A **Central Tool Executor**, written in native Python, then parses this graph topology, resolves data dependencies, and executes the necessary deterministic clinical tools in a safe, isolated sandbox.

### Core Engineering Improvements Checklist

To implement this decentralized, plan-driven execution pattern, the following architectural upgrades must be introduced to the codebase:

#### 1. Structured DAG Plan Blueprint (`src/cognition/schemas.py`)
Define a strict, strongly-typed schema using Pydantic to govern the LLM compiler's output. The model must emit a clean JSON structural plan mapping execution nodes to strict array-driven dependencies:
*   `TaskNode`: Contains a distinct `task_id`, targeted `tool_name`, and explicit runtime parameter kwargs.
*   `ExecutionPlan`: Declares a flat list of `TaskNode` objects and a dependency dictionary tracking node execution constraints.

#### 2. Centralized Clinical Tool Registry (`src/tools/registry.py`)
Move away from implicit code executions by wrapping capabilities into stateless, individual tool classes executed exclusively by the central master handler:
*   **Clinical Risk Calculator:** Runs pure algorithmic math formulas to compute formal medical risks (e.g., TIMI or GCS scores) from extracted integers, removing calculation tasks from the LLM.
*   **EHR Document Summarizer:** Takes verbose medical history logs and extracts structural data segments using local context tokens.
*   **Ontology Mapper Engine:** Resolves presentation strings directly to validated **SNOMED CT**, **ICD-10-CM**, and **RxNorm** code bases.
*   **Guideline Vector Fetcher:** Connects directly to the Qdrant instance to isolate targeted disease parameters.
*   **Drug-Drug Interaction Guard:** Verifies medication compatibility matrices before output generation.

#### 3. State-Aware Task Context Memory (`src/memory/context.py`)
Build a thread-safe data registry within the central executor to act as a shared context canvas during execution. The engine must support variable resolution strings (e.g., input values referencing upstream outputs like `"$ref: task_summarizer.extracted_vitals"`). The executor intercepts these tokens and dynamically resolves the variables prior to tool invocation.

#### 4. Topological Execution Loop (`src/cognition/dag_executor.py`)
Utilize Python's native `graphlib.TopologicalSorter` to parse the compiled JSON execution plan. The loop maps out linear dependencies, safely tracks task failures, and uses a concurrent thread or async pool to run independent tasks in parallel, maximizing local CPU throughput.

#### 5. Grounded Cognitive Synthesis (`src/cognition/synthesis.py`)
Establish a final completion step where the executor aggregates all tool runtime data arrays (the computed risk matrix, guidelines, and validated medical codes) into an immutable text frame. The orchestrator invokes the local model one final time to synthesize this data into an empathetic, clinically sound conversation string for the frontend user interface.

---

## Contributing & Development Standards

We welcome high-quality contributions from the open-source and clinical AI engineering communities. Because this orchestrator governs a safety-critical clinical triage domain, all contributions must uphold strict deterministic boundaries, architectural encapsulation, and performance budgets (e.g., maintaining sub-30ms latencies for emergency pathways on low-compute local hardware).

Please review our strict development guidelines before opening an issue or submitting a Pull Request.

---

### Git Branching Strategy & Workflow

We employ a structured feature-branch workflow. Direct commits to the `main` or `master` branches are restricted.

1. **Fork & Clone:** Fork the repository and create your feature branch from the latest `main`:
   ```bash
   git checkout -b feat/your-feature-name
   # For bug fixes: git checkout -b fix/issue-description
   # For refactoring: git checkout -b refactor/target-module
   ```

2. **Conventional Commits:** We enforce the [Conventional Commits](https://www.conventionalcommits.org/) specification. Every commit message must cleanly signal intention and module context. Examples:

   ```
   feat(cognition): integrate pydantic-driven dag blueprint validator

   fix(ui): align raw json viewer inside widescreen layout containers

   docs(readme): add detailed implementation roadmap for plan-then-execute evolution

   perf(core): optimize compiled regex evaluation network in opa fallback module
   ```

### Code Style, Linting & Static Analysis

To maintain code consistency across pipelines, we enforce formatting and static typing checks. Prior to submitting code, install and run our linting suite:

```bash
pip install ruff mypy
```

**Formatting & Linting (`ruff`):** We replace Black, Flake8, and isort with `ruff`. Ensure your code complies with our ruleset:

```bash
ruff format src/
ruff check src/ --fix
```

**Static Type Verification (`mypy`):** Every public function signature must be fully typed (no loose `Any` declarations unless explicitly necessary for dynamic JSON payload dictionary mapping).

```bash
mypy src/ --strict
```

### Testing Protocols & Performance Benchmarking

Changes to the core execution loop must not regress pipeline latency or state machine integrity.

- **Unit Testing:** Write descriptive assertions inside the `tests/` directory utilizing `pytest`. Ensure that your changes do not break existing context extraction parameters:

  ```bash
  pip install pytest pytest-asyncio
  pytest tests/
  ```

- **FSM State Verification:** Ensure your code modifications do not introduce illegal state mutations inside `src/memory/episodic_state.py`. Every transaction must follow the 7 allowed clinical state nodes.

- **Latency Benchmarking:** If you modify components within Step B (OPA Guardrails) or Step D (Ontology Engine), you must profile the execution path. Run the local test suite on standard CPU parameters to verify that emergency short-circuit responses execute in <30ms.

### Pull Request (PR) Submission Checklist

When submitting a Pull Request, ensure your description layout addresses the following criteria:

- [ ] **Architectural Alignment:** Does this code conform to our local, CPU-native, self-contained constraint? (e.g., No external cloud API calls, no heavy framework wrappers like LangChain/LangGraph).
- [ ] **Validation Pass:** `ruff format`, `ruff check`, and `mypy --strict` execution profiles all report a 100% success rate with zero errors.
- [ ] **Test Coverage:** New logic paths, tool extensions, or heuristic custom fallbacks are fully covered by deterministic test fixtures.
- [ ] **State Safety:** Verified that the changes introduce no conversational drift or broken loop vulnerabilities inside the finite state machine tracking.
- [ ] **Documentation Sync:** Corresponding structural specs in `SPEC.md` or local module docstrings are updated to reflect configuration edits or schema expansions.

---

## License

This project is licensed under the **MIT License**. You are free to use, modify, and distribute this software. See the [LICENSE](LICENSE) file for full details.
