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

# ── Configuration ───────────────────────────────────────────────────

DEFAULT_API = os.getenv("API_BASE_URL", "http://localhost:8080")

API_BASE_URL = st.sidebar.text_input(
    "API Base URL",
    value=DEFAULT_API,
    help="FastAPI orchestrator endpoint",
)

st.sidebar.divider()
st.sidebar.markdown("**Clinical Triage Orchestrator**")
st.sidebar.markdown("CPU-native agentic pipeline v0.3")

# ── Session init ────────────────────────────────────────────────────

if "session_id" not in st.session_state:
    st.session_state.session_id = f"ui-{int(time.time()*1000) % 1_000_000:06d}"
if "messages" not in st.session_state:
    st.session_state.messages = []
if "last_result" not in st.session_state:
    st.session_state.last_result = None

# ── API helpers ─────────────────────────────────────────────────────


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
    """Poll the async endpoint until result is ready or timeout."""
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


# ── Page layout ─────────────────────────────────────────────────────

st.title("Clinical Triage Orchestrator")
st.caption(f"Session: `{st.session_state.session_id}`")

col_chat, col_trace = st.columns([1, 1])

# ═══════════════════════════════════════════════════════════════════
# LEFT COLUMN: Patient Chat
# ═══════════════════════════════════════════════════════════════════

with col_chat:
    st.subheader("Patient Chat")

    # Display conversation history
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    # Quick-launch clinical scenarios
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

    # Chat input
    if prompt := st.chat_input("Describe your symptoms..."):
        st.session_state.messages.append({"role": "user", "content": prompt})

        with st.chat_message("user"):
            st.markdown(prompt)

        # Send to API (async pattern)
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
                    # Fallback to sync endpoint
                    result = api_post("/webhook/triage", {
                        "session_id": st.session_state.session_id,
                        "query_text": prompt,
                    })

            if result:
                st.session_state.last_result = result
                resp = result.get("response", {})

                if resp.get("llm_bypassed"):
                    # Emergency bypass
                    st.error(resp.get("message", "Emergency — call 911"))
                    assistant_msg = resp.get("message", "Emergency detected.")
                else:
                    # Normal triage response
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

# ═══════════════════════════════════════════════════════════════════
# RIGHT COLUMN: Observability Trace
# ═══════════════════════════════════════════════════════════════════

with col_trace:
    st.subheader("Observability Trace")

    # Fetch current session state
    session_state = api_get(f"/webhook/session/{st.session_state.session_id}")

    if session_state:
        current = session_state.get("current_state", "unknown")

        # ── FSM Pipeline ────────────────────────────────────────────
        state_icons = {
            "intake": "🔵", "symptom_extraction": "🟡",
            "guideline_lookup": "🟠", "risk_assessment": "🔴",
            "triage_decision": "🟣", "escalation": "🚨", "resolved": "✅",
        }
        fsm_steps = ["intake", "symptom_extraction", "guideline_lookup",
                      "risk_assessment", "triage_decision", "escalation", "resolved"]
        icon = state_icons.get(current, "⚪")

        st.markdown(f"#### {icon} `{current}`")
        try:
            idx = fsm_steps.index(current)
            parts = []
            for i, step in enumerate(fsm_steps):
                if i < idx:
                    parts.append(f"✅ `{step}`")
                elif i == idx:
                    parts.append(f"▶️ **{step}**")
                else:
                    parts.append(f"⬜ `{step}`")
            st.markdown(" → ".join(parts))
        except ValueError:
            st.markdown(f"Current: `{current}`")

        st.divider()

        # ── Clinical Entities ───────────────────────────────────────
        st.markdown("#### Clinical Entities")
        result = st.session_state.last_result
        if result:
            resp = result.get("response", {})

            entities = []
            extracted = resp.get("extracted_entities", {})
            if extracted:
                entities = extracted.get("entities", [])
            elif "steps" in result:
                executor = result["steps"].get("D_executor", {})
                entities = executor.get("entities", [])

            if entities:
                for ent in entities:
                    st.markdown(f"**{ent['term']}** `{ent['category']}`")
                    cols = st.columns(3)
                    cols[0].metric("SNOMED", ent.get("snomed_code", "—"))
                    cols[1].metric("ICD-10", ent.get("icd10_code", "—"))
                    cols[2].metric("Confidence", f"{ent.get('confidence', 0):.0%}")
            else:
                st.caption("No entities extracted yet")

            st.divider()

            # ── Pipeline Summary ────────────────────────────────────
            st.markdown("#### Pipeline")
            latency = result.get("latency_ms", 0)

            # Collect rules fired
            rules = []
            if "steps" in result:
                guardrail = result["steps"].get("B_guardrail", {})
                rules = guardrail.get("triggered_rules", [])
            elif resp.get("triggered_rules"):
                rules = resp.get("triggered_rules", [])

            metric_cols = st.columns(2)
            metric_cols[0].metric("Latency", f"{latency:.1f}ms")
            metric_cols[1].metric("Rules fired", len(rules))

            if rules:
                # Deduplicate while preserving order
                seen = set()
                unique_rules = []
                for r in rules:
                    if r not in seen:
                        seen.add(r)
                        unique_rules.append(r)
                st.caption(", ".join(unique_rules))

            st.divider()

            # ── Diagnostic CoT ──────────────────────────────────────
            st.markdown("#### Diagnostic CoT")
            cognition = {}
            if "steps" in result:
                cognition = result["steps"].get("E_cognition", {})

            if cognition and "error" not in cognition:
                observations = cognition.get("clinical_observations", [])
                rationale = cognition.get("step_by_step_rationale", [])

                if observations:
                    for obs in observations:
                        st.markdown(f"- {obs}")
                if rationale:
                    st.markdown("**Reasoning:**")
                    for i, step in enumerate(rationale, 1):
                        st.markdown(f"{i}. {step}")

                meta_cols = st.columns(3)
                meta_cols[0].metric("Urgency", cognition.get("urgency_level", "—"))
                meta_cols[1].metric("Dept", cognition.get("recommended_department", "—"))
                meta_cols[2].metric("Confidence", f"{cognition.get('confidence', 0):.0%}")
            elif cognition.get("error"):
                st.warning(f"LLM error: {cognition['error']}")
            elif resp.get("llm_bypassed"):
                st.info("LLM bypassed — emergency rules triggered, no CoT needed")
            else:
                st.caption("No LLM reasoning yet — send a message to start")

            # ── Raw JSON ────────────────────────────────────────────
            with st.expander("Raw result JSON", expanded=False):
                st.json(result)
        else:
            st.info("No trace data yet — send a message to start the pipeline")
    else:
        st.info("Session not found on server. Send a message to create one.")
