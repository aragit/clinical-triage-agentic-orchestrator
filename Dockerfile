# ── Multi-stage Dockerfile for the Clinical Triage Orchestrator ─────
# Stage 1: Python dependencies
# Stage 2: Runtime (API + UI)

FROM python:3.12-slim AS deps

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# ── Runtime ─────────────────────────────────────────────────────────

FROM deps AS runtime

WORKDIR /app
COPY . .

# Models directory — mounted as a volume at runtime
RUN mkdir -p /app/models

EXPOSE 8080 8501

# Default: run the FastAPI server
CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8080"]
