"""
Weather Agent Station — Streamlit UI

A production-ready monitoring dashboard for the Weather Documentation
Pipeline, showing both v1 (LangGraph) and v2 (Deep Agent) side-by-side
with live agent reasoning, tool calls, and LaTeX output.

Run:
    streamlit run app.py
"""

import sys
import os
import json
import time
import threading
import queue
from datetime import datetime, timezone

sys.path.insert(0, os.path.dirname(__file__))

import streamlit as st

# ---------------------------------------------------------------------------
#  Page config & custom CSS
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="Weather Agent Station",
    page_icon="https://cdn-icons-png.flaticon.com/512/1163/1163661.png",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
/* ── Global ──────────────────────────────────────────── */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');

.stApp {
    font-family: 'Inter', sans-serif;
}

/* ── Header ──────────────────────────────────────────── */
.station-header {
    background: linear-gradient(135deg, #0f0c29 0%, #302b63 50%, #24243e 100%);
    padding: 1.8rem 2rem;
    border-radius: 16px;
    margin-bottom: 1.5rem;
    color: white;
    position: relative;
    overflow: hidden;
}
.station-header::before {
    content: '';
    position: absolute;
    top: -50%;
    right: -20%;
    width: 300px;
    height: 300px;
    background: radial-gradient(circle, rgba(99,102,241,0.15) 0%, transparent 70%);
    border-radius: 50%;
}
.station-header h1 {
    font-size: 1.8rem;
    font-weight: 700;
    margin: 0;
    letter-spacing: -0.5px;
}
.station-header p {
    font-size: 0.9rem;
    opacity: 0.7;
    margin: 0.3rem 0 0 0;
}
.station-header .live-dot {
    display: inline-block;
    width: 8px;
    height: 8px;
    background: #10b981;
    border-radius: 50%;
    margin-right: 6px;
    animation: pulse-dot 2s infinite;
}
@keyframes pulse-dot {
    0%, 100% { opacity: 1; box-shadow: 0 0 0 0 rgba(16,185,129,0.7); }
    50% { opacity: 0.7; box-shadow: 0 0 0 6px rgba(16,185,129,0); }
}

/* ── Metric cards ────────────────────────────────────── */
.metric-card {
    background: linear-gradient(135deg, #1e1b4b 0%, #312e81 100%);
    border: 1px solid rgba(99,102,241,0.2);
    border-radius: 12px;
    padding: 1rem 1.2rem;
    color: white;
    text-align: center;
}
.metric-card .value {
    font-size: 1.6rem;
    font-weight: 700;
    color: #a5b4fc;
}
.metric-card .label {
    font-size: 0.75rem;
    text-transform: uppercase;
    letter-spacing: 1px;
    opacity: 0.6;
    margin-top: 0.2rem;
}

/* ── Thinking bubble ─────────────────────────────────── */
.thinking-bubble {
    background: #1a1a2e;
    border-left: 3px solid #6366f1;
    border-radius: 0 8px 8px 0;
    padding: 0.6rem 1rem;
    margin: 0.3rem 0;
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.78rem;
    line-height: 1.5;
    color: #c4b5fd;
    max-height: 300px;
    overflow-y: auto;
}

/* ── Tool call card ──────────────────────────────────── */
.tool-card {
    background: #0f172a;
    border: 1px solid #1e3a5f;
    border-radius: 8px;
    padding: 0.5rem 0.8rem;
    margin: 0.3rem 0;
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.75rem;
}
.tool-card .tool-name {
    color: #38bdf8;
    font-weight: 600;
}
.tool-card .tool-args {
    color: #94a3b8;
}
.tool-card .tool-result {
    color: #4ade80;
    border-top: 1px solid #1e3a5f;
    margin-top: 0.3rem;
    padding-top: 0.3rem;
}

/* ── Node badge ──────────────────────────────────────── */
.node-badge {
    display: inline-block;
    padding: 0.2rem 0.7rem;
    border-radius: 999px;
    font-size: 0.72rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.5px;
}
.node-orchestrator { background: #312e81; color: #a5b4fc; }
.node-retriever { background: #064e3b; color: #6ee7b7; }
.node-weather { background: #78350f; color: #fcd34d; }
.node-writer { background: #1e3a5f; color: #7dd3fc; }
.node-fact-checker { background: #4c1d95; color: #c4b5fd; }
.node-fix-fact { background: #701a75; color: #f0abfc; }
.node-formatter { background: #164e63; color: #67e8f9; }
.node-agent { background: #1e40af; color: #93c5fd; }

/* ── Status indicator ────────────────────────────────── */
.status-running {
    color: #fbbf24;
    font-weight: 600;
}
.status-done {
    color: #34d399;
    font-weight: 600;
}
.status-error {
    color: #f87171;
    font-weight: 600;
}

/* ── Version tab ─────────────────────────────────────── */
.version-label {
    background: linear-gradient(135deg, #6366f1, #8b5cf6);
    color: white;
    padding: 0.3rem 0.8rem;
    border-radius: 6px;
    font-size: 0.75rem;
    font-weight: 600;
    display: inline-block;
    margin-bottom: 0.5rem;
}

/* ── LaTeX preview box ───────────────────────────────── */
.latex-preview {
    background: #fffef5;
    border: 1px solid #e5e2c8;
    border-radius: 8px;
    padding: 1rem;
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.72rem;
    color: #1a1a1a;
    max-height: 500px;
    overflow-y: auto;
    white-space: pre-wrap;
    word-wrap: break-word;
}

/* ── Pipeline flow ───────────────────────────────────── */
.pipeline-flow {
    display: flex;
    align-items: center;
    gap: 0.3rem;
    flex-wrap: wrap;
    padding: 0.5rem 0;
}
.pipeline-step {
    padding: 0.3rem 0.6rem;
    border-radius: 6px;
    font-size: 0.7rem;
    font-weight: 500;
    background: #1e293b;
    color: #64748b;
    border: 1px solid #334155;
}
.pipeline-step.active {
    background: #312e81;
    color: #a5b4fc;
    border-color: #6366f1;
    animation: pulse-step 1.5s infinite;
}
.pipeline-step.done {
    background: #064e3b;
    color: #6ee7b7;
    border-color: #10b981;
}
.pipeline-step.redirected {
    background: #78350f;
    color: #fcd34d;
    border-color: #f59e0b;
}
.pipeline-arrow {
    color: #475569;
    font-size: 0.8rem;
}

/* ── Voice input ─────────────────────────────────────── */
[data-testid="stAudioInput"] {
    border: 2px solid #6366f1;
    border-radius: 12px;
    padding: 0.3rem;
}

/* ── Cloud wake word ─────────────────────────────────── */
.cloud-banner {
    background: linear-gradient(135deg, #0c4a6e 0%, #1e3a5f 50%, #312e81 100%);
    border: 1px solid rgba(56,189,248,0.3);
    border-radius: 12px;
    padding: 1rem;
    text-align: center;
    margin: 0.5rem 0;
    position: relative;
    overflow: hidden;
}
.cloud-banner::before {
    content: '';
    position: absolute;
    top: -30%;
    left: -20%;
    width: 200px;
    height: 200px;
    background: radial-gradient(circle, rgba(56,189,248,0.1) 0%, transparent 70%);
    border-radius: 50%;
}
.cloud-banner .cloud-name {
    font-size: 1.4rem;
    font-weight: 700;
    color: #7dd3fc;
    letter-spacing: 1px;
}
.cloud-banner .cloud-sub {
    font-size: 0.7rem;
    color: #94a3b8;
    margin-top: 0.2rem;
}
.cloud-heard {
    background: linear-gradient(135deg, #064e3b 0%, #065f46 100%);
    border: 1px solid #10b981;
    border-radius: 10px;
    padding: 0.8rem;
    text-align: center;
    animation: cloud-glow 1.5s ease-in-out;
}
@keyframes cloud-glow {
    0% { box-shadow: 0 0 0 0 rgba(16,185,129,0.5); }
    50% { box-shadow: 0 0 20px 4px rgba(16,185,129,0.3); }
    100% { box-shadow: 0 0 0 0 rgba(16,185,129,0); }
}
.cloud-heard .heard-text {
    color: #6ee7b7;
    font-weight: 600;
    font-size: 0.9rem;
}
.cloud-heard .heard-query {
    color: #a7f3d0;
    font-size: 0.8rem;
    margin-top: 0.3rem;
    font-style: italic;
}
.wake-variants {
    display: flex;
    flex-wrap: wrap;
    gap: 0.3rem;
    justify-content: center;
    margin-top: 0.4rem;
}
.wake-chip {
    background: #1e293b;
    border: 1px solid #334155;
    border-radius: 999px;
    padding: 0.15rem 0.5rem;
    font-size: 0.65rem;
    color: #7dd3fc;
    font-family: 'JetBrains Mono', monospace;
}
.cloud-no-wake {
    background: #1c1917;
    border: 1px solid #78350f;
    border-radius: 10px;
    padding: 0.6rem;
    text-align: center;
}
.cloud-no-wake .nw-text {
    color: #fbbf24;
    font-size: 0.8rem;
    font-weight: 500;
}
.cloud-no-wake .nw-sub {
    color: #a8a29e;
    font-size: 0.7rem;
    margin-top: 0.2rem;
}
</style>
""", unsafe_allow_html=True)


# ---------------------------------------------------------------------------
#  DSPy init (once)
# ---------------------------------------------------------------------------

@st.cache_resource
def init_dspy():
    import dspy
    from config.settings import DEEPSEEK_API_KEY, DEEPSEEK_BASE_URL, DEEPSEEK_MODEL
    lm = dspy.LM(
        model=f"openai/{DEEPSEEK_MODEL}",
        api_key=DEEPSEEK_API_KEY,
        api_base=DEEPSEEK_BASE_URL,
        temperature=0.0,
        max_tokens=1000,
    )
    dspy.configure(lm=lm)
    return True


# ---------------------------------------------------------------------------
#  Build graphs (cached)
# ---------------------------------------------------------------------------

@st.cache_resource
def get_v1_graph():
    from agent.graph import build_graph
    return build_graph()


@st.cache_resource
def get_v2_agent():
    from agent_v2.deep_agent import build_deep_agent
    return build_deep_agent()


# ---------------------------------------------------------------------------
#  Streaming callback for capturing agent reasoning
# ---------------------------------------------------------------------------

class StreamingTracer:
    """Thread-safe collector of agent reasoning events for the UI."""

    def __init__(self):
        self.events: list[dict] = []
        self._lock = threading.Lock()

    def add(self, event_type: str, node: str, **data):
        with self._lock:
            self.events.append({
                "type": event_type,
                "node": node,
                "time": datetime.now(timezone.utc).strftime("%H:%M:%S"),
                **data,
            })

    def get_all(self) -> list[dict]:
        with self._lock:
            return list(self.events)

    def clear(self):
        with self._lock:
            self.events.clear()


# ---------------------------------------------------------------------------
#  V1 runner — stream node-by-node
# ---------------------------------------------------------------------------

def run_v1_streaming(query: str, tracer: StreamingTracer):
    """Run v1 pipeline, streaming events to tracer."""
    graph = get_v1_graph()
    from agent.state import WeatherAgentState

    config = {
        "configurable": {"thread_id": f"ui_v1_{int(time.time())}"},
    }

    initial = WeatherAgentState(user_query=query).model_dump()

    tracer.add("start", "pipeline", message=f"Starting v1 pipeline for: {query}")

    t0 = time.time()

    for event in graph.stream(initial, config=config, stream_mode="updates"):
        for node_name, node_output in event.items():
            elapsed = time.time() - t0

            # Log reasoning based on node
            if node_name == "orchestration_agent":
                step = node_output.get("current_step", "?")
                tracer.add("thinking", "orchestrator",
                           message=f"Decision: route to '{step}'",
                           detail=f"Evaluated state -> next_step = {step}",
                           elapsed=f"{elapsed:.1f}s")

            elif node_name == "retriever_agent":
                co = node_output.get("crag_output")
                rdem = node_output.get("crag_rdem")
                if co and hasattr(co, "action"):
                    tracer.add("tool_call", "retriever",
                               tool="CRAGPipeline.run()",
                               args=query,
                               result=f"action={co.action}, score={co.max_score:.2f}, "
                                      f"sources={len(co.sources)}",
                               elapsed=f"{elapsed:.1f}s")
                    tracer.add("thinking", "retriever",
                               message=f"CRAG returned '{co.action}' with score {co.max_score:.2f}",
                               detail=co.answer[:500] if co.answer else "",
                               elapsed=f"{elapsed:.1f}s")
                if rdem and hasattr(rdem, "message"):
                    is_redirect = rdem.error_type == "relevance_redirect"
                    tracer.add(
                        "redirect" if is_redirect else "thinking",
                        "retriever",
                        message=(
                            "Quality gate: stale data detected — redirecting to live web search"
                            if is_redirect
                            else f"Quality gate: {rdem.error_type}"
                        ),
                        detail=rdem.message,
                        elapsed=f"{elapsed:.1f}s",
                    )

            elif node_name == "weather_consultant":
                co = node_output.get("crag_output")
                if co and hasattr(co, "answer"):
                    tracer.add("tool_call", "weather_consultant",
                               tool="web_search (Tavily)",
                               args=query,
                               result=f"Got {len(co.sources)} sources via live web search",
                               elapsed=f"{elapsed:.1f}s")
                    tracer.add("thinking", "weather_consultant",
                               message="Generated answer from web search results",
                               detail=co.answer[:500] if co.answer else "",
                               elapsed=f"{elapsed:.1f}s")

            elif node_name == "writer_agent":
                draft = node_output.get("draft_document", "")
                tracer.add("thinking", "writer",
                           message=f"Drafted weather report ({len(draft)} chars)",
                           detail=draft[:600] if draft else "",
                           elapsed=f"{elapsed:.1f}s")

            elif node_name == "writer_fact_checker":
                fc = node_output.get("fact_check_result")
                if fc and hasattr(fc, "is_factual"):
                    temps = fc.verified_temperatures or {}
                    temp_str = ", ".join(f"{c}C->{f}F" for c, f in temps.items())
                    tracer.add("tool_call", "fact_checker",
                               tool="celsius_to_fahrenheit",
                               args="(multiple temperatures)",
                               result=temp_str or "(no temps verified)",
                               elapsed=f"{elapsed:.1f}s")
                    tracer.add("thinking", "fact_checker",
                               message=f"Verdict: {'PASS' if fc.is_factual else 'FAIL'}",
                               detail=f"Issues: {fc.issues}" if fc.issues else "No issues found",
                               elapsed=f"{elapsed:.1f}s")

                attempts = node_output.get("fact_fix_attempts")
                if attempts is not None and attempts > 0:
                    tracer.add("thinking", "fact_checker",
                               message=f"Fix attempt {attempts}/3",
                               elapsed=f"{elapsed:.1f}s")

            elif node_name == "fix_fact_agent":
                draft = node_output.get("draft_document", "")
                tracer.add("thinking", "fix_fact",
                           message=f"Corrected draft ({len(draft)} chars)",
                           detail=draft[:400] if draft else "",
                           elapsed=f"{elapsed:.1f}s")

            elif node_name == "formatter":
                latex = node_output.get("final_latex_document", "")
                tracer.add("thinking", "formatter",
                           message=f"Generated LaTeX document ({len(latex)} chars)",
                           detail=latex[:400] if latex else "",
                           elapsed=f"{elapsed:.1f}s")

    total_time = time.time() - t0

    # Get final state
    final = graph.get_state(config)
    final_values = final.values if final else {}

    tracer.add("done", "pipeline",
               message=f"Pipeline complete in {total_time:.1f}s",
               elapsed=f"{total_time:.1f}s")

    return final_values, total_time


# ---------------------------------------------------------------------------
#  V2 runner — stream Deep Agent events
# ---------------------------------------------------------------------------

def run_v2_streaming(query: str, tracer: StreamingTracer):
    """Run v2 Deep Agent pipeline, streaming events to tracer."""
    agent = get_v2_agent()

    config = {
        "configurable": {"thread_id": f"ui_v2_{int(time.time())}"},
    }

    tracer.add("start", "pipeline", message=f"Starting v2 Deep Agent for: {query}")

    t0 = time.time()
    last_node = None

    for event in agent.stream(
        {"messages": [{"role": "user", "content": query}]},
        config=config,
        stream_mode="updates",
    ):
        for node_name, node_output in event.items():
            elapsed = time.time() - t0

            if node_name != last_node:
                tracer.add("node_enter", "agent",
                           message=f"Entering: {node_name}",
                           elapsed=f"{elapsed:.1f}s")
                last_node = node_name

            messages = node_output.get("messages", [])
            for msg in messages:
                content = getattr(msg, "content", "")
                tool_calls = getattr(msg, "tool_calls", [])
                msg_type = type(msg).__name__

                if tool_calls:
                    for tc in tool_calls:
                        args_str = json.dumps(tc.get("args", {}), default=str)
                        tracer.add("tool_call", "agent",
                                   tool=tc.get("name", "?"),
                                   args=args_str[:300],
                                   elapsed=f"{elapsed:.1f}s")

                elif msg_type == "ToolMessage":
                    tool_name = getattr(msg, "name", "?")
                    tracer.add("tool_result", "agent",
                               tool=tool_name,
                               result=str(content)[:500],
                               elapsed=f"{elapsed:.1f}s")

                elif content and msg_type == "AIMessage":
                    if "\\documentclass" in content:
                        tracer.add("output", "agent",
                                   message="Generated LaTeX document",
                                   detail=content[:300],
                                   elapsed=f"{elapsed:.1f}s")
                    else:
                        tracer.add("thinking", "agent",
                                   message=content[:200],
                                   detail=content[:600] if len(content) > 200 else "",
                                   elapsed=f"{elapsed:.1f}s")

    total_time = time.time() - t0

    final = agent.get_state(config)
    final_values = final.values if final else {}

    tracer.add("done", "pipeline",
               message=f"Deep Agent complete in {total_time:.1f}s",
               elapsed=f"{total_time:.1f}s")

    return final_values, total_time


# ---------------------------------------------------------------------------
#  UI Helper: render thinking events
# ---------------------------------------------------------------------------

NODE_CSS_MAP = {
    "orchestrator": "node-orchestrator",
    "retriever": "node-retriever",
    "weather_consultant": "node-weather",
    "writer": "node-writer",
    "fact_checker": "node-fact-checker",
    "fix_fact": "node-fix-fact",
    "formatter": "node-formatter",
    "agent": "node-agent",
    "pipeline": "node-orchestrator",
}

NODE_ICONS = {
    "orchestrator": "🎯",
    "retriever": "🔍",
    "weather_consultant": "🌐",
    "writer": "✍️",
    "fact_checker": "✅",
    "fix_fact": "🔧",
    "formatter": "📄",
    "agent": "🤖",
    "pipeline": "⚡",
}


def render_events(events: list[dict], container):
    """Render reasoning events into a streamlit container."""
    for ev in events:
        node = ev.get("node", "")
        css_class = NODE_CSS_MAP.get(node, "node-agent")
        icon = NODE_ICONS.get(node, "💭")
        elapsed = ev.get("elapsed", "")
        elapsed_tag = f" <span style='color:#64748b;font-size:0.65rem;'>[{elapsed}]</span>" if elapsed else ""

        if ev["type"] == "start":
            container.markdown(
                f"<div style='padding:0.4rem;color:#fbbf24;font-weight:600;'>"
                f"⚡ {ev.get('message', '')}{elapsed_tag}</div>",
                unsafe_allow_html=True,
            )

        elif ev["type"] == "done":
            container.markdown(
                f"<div style='padding:0.4rem;color:#34d399;font-weight:600;'>"
                f"✅ {ev.get('message', '')}</div>",
                unsafe_allow_html=True,
            )

        elif ev["type"] == "node_enter":
            container.markdown(
                f"<div style='padding:0.2rem 0;'>"
                f"<span class='node-badge {css_class}'>{icon} {node}</span>"
                f" <span style='color:#94a3b8;font-size:0.75rem;'>"
                f"{ev.get('message', '')}</span>{elapsed_tag}</div>",
                unsafe_allow_html=True,
            )

        elif ev["type"] == "thinking":
            msg = ev.get("message", "")
            detail = ev.get("detail", "")
            container.markdown(
                f"<div style='margin:0.2rem 0;'>"
                f"<span class='node-badge {css_class}'>{icon} {node}</span>"
                f"{elapsed_tag}"
                f"</div>",
                unsafe_allow_html=True,
            )
            detail_html = detail.replace("<", "&lt;").replace(">", "&gt;").replace("\n", "<br>") if detail else ""
            msg_html = msg.replace("<", "&lt;").replace(">", "&gt;")
            inner = f"<strong>{msg_html}</strong>"
            if detail_html:
                inner += f"<br><span style='color:#94a3b8;font-size:0.72rem;'>{detail_html[:800]}</span>"
            container.markdown(
                f"<div class='thinking-bubble'>{inner}</div>",
                unsafe_allow_html=True,
            )

        elif ev["type"] in ("tool_call", "tool_result"):
            tool = ev.get("tool", "?")
            args = ev.get("args", "")
            result = ev.get("result", "")
            args_html = str(args).replace("<", "&lt;").replace(">", "&gt;")
            result_html = str(result).replace("<", "&lt;").replace(">", "&gt;") if result else ""
            card = f"<div class='tool-card'>"
            if ev["type"] == "tool_call":
                card += f"<span class='tool-name'>CALL</span> {tool}({args_html[:200]})"
            else:
                card += f"<span class='tool-name'>RESULT</span> {tool}"
            if result_html:
                card += f"<div class='tool-result'>{result_html[:400]}</div>"
            card += f"</div>"
            container.markdown(
                f"<div style='margin:0.2rem 0;'>"
                f"<span class='node-badge {css_class}'>{icon} {node}</span>"
                f"{elapsed_tag}</div>{card}",
                unsafe_allow_html=True,
            )

        elif ev["type"] == "redirect":
            msg = ev.get("message", "")
            detail = ev.get("detail", "")
            msg_html = msg.replace("<", "&lt;").replace(">", "&gt;")
            detail_html = detail.replace("<", "&lt;").replace(">", "&gt;").replace("\n", "<br>") if detail else ""
            inner = f"<strong>{msg_html}</strong>"
            if detail_html:
                inner += f"<br><span style='color:#94a3b8;font-size:0.72rem;'>{detail_html[:600]}</span>"
            container.markdown(
                f"<div style='margin:0.2rem 0;'>"
                f"<span class='node-badge {css_class}'>🔀 {node}</span>"
                f"{elapsed_tag}</div>"
                f"<div style='background:#1a1a2e;border-left:3px solid #f59e0b;"
                f"border-radius:0 8px 8px 0;padding:0.6rem 1rem;margin:0.3rem 0;"
                f"font-family:JetBrains Mono,monospace;font-size:0.78rem;"
                f"line-height:1.5;color:#fbbf24;'>{inner}</div>",
                unsafe_allow_html=True,
            )

        elif ev["type"] == "output":
            container.markdown(
                f"<div style='margin:0.2rem 0;'>"
                f"<span class='node-badge {css_class}'>{icon} {node}</span>"
                f" <span style='color:#34d399;font-weight:600;'>"
                f"{ev.get('message', '')}</span>{elapsed_tag}</div>",
                unsafe_allow_html=True,
            )


# ---------------------------------------------------------------------------
#  V1 pipeline flow visualization
# ---------------------------------------------------------------------------

V1_STEPS = [
    ("Orchestrator", "orchestrator"),
    ("Retriever", "retriever"),
    ("Weather Fallback", "weather_consultant"),
    ("Writer", "writer"),
    ("Fact Checker", "fact_checker"),
    ("Fix Fact", "fix_fact"),
    ("Formatter", "formatter"),
]


def render_pipeline_flow(audit_trail: list, active_step: str = ""):
    done_nodes = set()
    redirected_nodes = set()
    if audit_trail:
        for step in audit_trail:
            name = step.get("node_name", "") if isinstance(step, dict) else getattr(step, "node_name", "")
            status = step.get("status", "") if isinstance(step, dict) else getattr(step, "status", "")
            done_nodes.add(name)
            if status == "redirected":
                redirected_nodes.add(name)

    html_parts = []
    for label, node_key in V1_STEPS:
        if node_key in redirected_nodes:
            cls = "pipeline-step redirected"
            label = f"🔀 {label}"
        elif node_key in done_nodes:
            cls = "pipeline-step done"
        elif node_key == active_step:
            cls = "pipeline-step active"
        else:
            cls = "pipeline-step"
        html_parts.append(f"<span class='{cls}'>{label}</span>")
        html_parts.append("<span class='pipeline-arrow'>→</span>")

    html_parts.append("<span class='pipeline-step'>Done</span>")

    return f"<div class='pipeline-flow'>{''.join(html_parts)}</div>"


# ---------------------------------------------------------------------------
#  MAIN APP
# ---------------------------------------------------------------------------

def main():
    init_dspy()

    # ── Header ────────────────────────────────────────────
    st.markdown("""
    <div class="station-header">
        <h1><span class="live-dot"></span> Weather Agent Station &mdash; <span style="color:#7dd3fc;">Cloud</span></h1>
        <p>Multi-Agent Weather Documentation Pipeline &mdash; Production Monitoring Dashboard &mdash; Voice Activated</p>
    </div>
    """, unsafe_allow_html=True)

    # ── Sidebar ───────────────────────────────────────────
    with st.sidebar:
        st.markdown("### Configuration")

        input_mode = st.radio(
            "Input Mode",
            ["Text", "Voice"],
            horizontal=True,
        )

        query = ""

        if input_mode == "Text":
            query = st.text_input(
                "Weather Query",
                value="What is the weather in Berlin this week?",
                placeholder="Enter any weather question...",
            )

            presets = [
                "What is the weather in Berlin this week?",
                "Current weather in Tokyo, Japan",
                "5-day forecast for London",
                "Weather warnings for Miami, Florida",
                "Is it raining in Paris right now?",
            ]
            preset = st.selectbox("Quick Presets", ["(custom)"] + presets)
            if preset != "(custom)":
                query = preset

        else:
            from wake_word import detect_wake_word, get_variants_display
            from config.settings import GROQ_API_KEY as _groq_key

            variants = get_variants_display()
            chips = "".join(f"<span class='wake-chip'>\"{v}\"</span>" for v in variants)
            st.markdown(
                f"<div class='cloud-banner'>"
                f"<div class='cloud-name'>Cloud</div>"
                f"<div class='cloud-sub'>Weather Agent &mdash; Voice Activated</div>"
                f"<div class='wake-variants'>{chips}</div>"
                f"</div>",
                unsafe_allow_html=True,
            )

            if not _groq_key:
                st.warning(
                    "**GROQ_API_KEY** not set in `.env`. "
                    "Get a free key at [console.groq.com/keys]"
                    "(https://console.groq.com/keys) to enable voice input."
                )

            audio_data = st.audio_input(
                "🎙️ Say: \"Hey Cloud, what's the weather in...\"",
                key="voice_input",
            )

            if audio_data is not None:
                audio_bytes = audio_data.getvalue()
                st.audio(audio_bytes, format="audio/wav")

                if _groq_key:
                    with st.spinner("Cloud is listening..."):
                        try:
                            from stt import transcribe_audio
                            raw_text = transcribe_audio(audio_bytes)
                        except Exception as e:
                            st.error(f"Transcription failed: {e}")
                            raw_text = ""

                    if raw_text:
                        result = detect_wake_word(raw_text)

                        if result.detected:
                            st.markdown(
                                f"<div class='cloud-heard'>"
                                f"<div class='heard-text'>"
                                f"Cloud heard you! (via \"{result.variant}\")"
                                f"</div>"
                                f"<div class='heard-query'>"
                                f"\"{result.query}\"</div>"
                                f"</div>",
                                unsafe_allow_html=True,
                            )
                            query = result.query
                        else:
                            st.markdown(
                                f"<div class='cloud-no-wake'>"
                                f"<div class='nw-text'>"
                                f"No wake word detected</div>"
                                f"<div class='nw-sub'>"
                                f"Using full transcription as query</div>"
                                f"</div>",
                                unsafe_allow_html=True,
                            )
                            st.caption(f"Heard: \"{raw_text}\"")
                            query = raw_text
                else:
                    st.info("Add GROQ_API_KEY to `.env` to transcribe audio.")

        st.divider()
        st.markdown("### Pipeline Version")
        version = st.radio(
            "Select version to run",
            ["Both (side-by-side)", "v1 — LangGraph", "v2 — Deep Agent"],
            index=0,
        )

        st.divider()
        st.markdown("### About")
        st.markdown("""
        **v1 — LangGraph StateGraph**
        8-node graph with deterministic routing, 
        ReAct agents, self-healing fact-check loop.

        **v2 — Deep Agent**
        Single autonomous agent with built-in 
        planning, context management, subagent 
        delegation.
        """)

        run_btn = st.button("Run Pipeline", type="primary", use_container_width=True)

    # ── Main area ─────────────────────────────────────────

    if run_btn and not query:
        st.warning("Please enter a query or record a voice question first.")
        run_btn = False

    if not run_btn:
        # Show idle state
        cols = st.columns(4)
        idle_metrics = [
            ("Pipeline Status", "IDLE", "Waiting for query"),
            ("Agents Available", "7 + 1", "v1 nodes + v2 agent"),
            ("Tools", "3", "CRAG, C→F, Web Search"),
            ("Output Format", "LaTeX", "Compilable .tex"),
        ]
        for col, (label, value, desc) in zip(cols, idle_metrics):
            with col:
                st.markdown(
                    f"<div class='metric-card'>"
                    f"<div class='value'>{value}</div>"
                    f"<div class='label'>{label}</div>"
                    f"</div>",
                    unsafe_allow_html=True,
                )

        st.markdown("---")
        st.info("Enter a weather query and click **Run Pipeline** to start.")
        return

    # ── Run pipelines ─────────────────────────────────────

    run_v1 = version in ["Both (side-by-side)", "v1 — LangGraph"]
    run_v2 = version in ["Both (side-by-side)", "v2 — Deep Agent"]

    if run_v1 and run_v2:
        col1, col2 = st.columns(2)
    elif run_v1:
        col1 = st.container()
        col2 = None
    else:
        col1 = None
        col2 = st.container()

    # --- V1 ---
    if run_v1 and col1:
        with col1:
            st.markdown("<div class='version-label'>v1 — LangGraph StateGraph</div>",
                        unsafe_allow_html=True)

            status_v1 = st.empty()
            status_v1.markdown("<span class='status-running'>RUNNING...</span>",
                               unsafe_allow_html=True)

            flow_v1 = st.empty()
            flow_v1.markdown(render_pipeline_flow([]), unsafe_allow_html=True)

            reasoning_v1 = st.container()
            reasoning_v1.markdown("##### Agent Reasoning")

            tracer_v1 = StreamingTracer()

            with st.spinner("v1 pipeline running..."):
                try:
                    result_v1, time_v1 = run_v1_streaming(query, tracer_v1)
                except Exception as e:
                    tracer_v1.add("done", "pipeline", message=f"ERROR: {e}")
                    result_v1, time_v1 = {}, 0

            # Render events
            events_v1 = tracer_v1.get_all()
            with reasoning_v1:
                render_events(events_v1, reasoning_v1)

            # Update status
            audit_trail = result_v1.get("audit_trail", [])
            flow_v1.markdown(render_pipeline_flow(audit_trail), unsafe_allow_html=True)

            if result_v1.get("final_latex_document"):
                status_v1.markdown("<span class='status-done'>COMPLETE</span>",
                                   unsafe_allow_html=True)
            else:
                status_v1.markdown("<span class='status-error'>FINISHED (no LaTeX)</span>",
                                   unsafe_allow_html=True)

            # Metrics
            m1, m2, m3 = st.columns(3)
            m1.markdown(f"<div class='metric-card'><div class='value'>{time_v1:.1f}s</div>"
                        f"<div class='label'>Total Time</div></div>", unsafe_allow_html=True)

            crag_out = result_v1.get("crag_output")
            if crag_out:
                action = crag_out.action if hasattr(crag_out, "action") else crag_out.get("action", "?")
                score = crag_out.max_score if hasattr(crag_out, "max_score") else crag_out.get("max_score", 0)
                m2.markdown(f"<div class='metric-card'><div class='value'>{action}</div>"
                            f"<div class='label'>CRAG Action</div></div>", unsafe_allow_html=True)
                m3.markdown(f"<div class='metric-card'><div class='value'>{score:.2f}</div>"
                            f"<div class='label'>Confidence</div></div>", unsafe_allow_html=True)
            else:
                m2.markdown(f"<div class='metric-card'><div class='value'>N/A</div>"
                            f"<div class='label'>CRAG Action</div></div>", unsafe_allow_html=True)
                m3.markdown(f"<div class='metric-card'><div class='value'>N/A</div>"
                            f"<div class='label'>Confidence</div></div>", unsafe_allow_html=True)

            # LaTeX output
            latex_v1 = result_v1.get("final_latex_document", "")
            if latex_v1:
                with st.expander("LaTeX Output", expanded=True):
                    st.code(latex_v1, language="latex")

            # Audit trail
            if audit_trail:
                with st.expander("Audit Trail"):
                    for step in audit_trail:
                        if isinstance(step, dict):
                            name = step.get("node_name", "?")
                            status = step.get("status", "?")
                        else:
                            name = getattr(step, "node_name", "?")
                            status = getattr(step, "status", "?")
                        if status == "success":
                            icon, color = "✅", ""
                        elif status == "redirected":
                            icon, color = "🔀", "orange"
                        elif status == "forced_proceed":
                            icon, color = "⚠️", ""
                        else:
                            icon, color = "❌", ""
                        label = f":{color}[{status}]" if color else status
                        st.markdown(f"{icon} **{name}** — {label}")

            # Errors
            errors = result_v1.get("global_error_log", [])
            if errors:
                with st.expander(f"Errors ({len(errors)})"):
                    for err in errors:
                        if isinstance(err, dict):
                            st.error(f"**{err.get('node', '?')}**: {err.get('message', '?')}")
                        else:
                            st.error(f"**{err.node}**: {err.message}")

    # --- V2 ---
    if run_v2 and col2:
        with col2:
            st.markdown("<div class='version-label'>v2 — Deep Agent</div>",
                        unsafe_allow_html=True)

            status_v2 = st.empty()
            status_v2.markdown("<span class='status-running'>RUNNING...</span>",
                               unsafe_allow_html=True)

            reasoning_v2 = st.container()
            reasoning_v2.markdown("##### Agent Reasoning")

            tracer_v2 = StreamingTracer()

            with st.spinner("v2 Deep Agent running..."):
                try:
                    result_v2, time_v2 = run_v2_streaming(query, tracer_v2)
                except Exception as e:
                    tracer_v2.add("done", "pipeline", message=f"ERROR: {e}")
                    result_v2, time_v2 = {}, 0

            events_v2 = tracer_v2.get_all()
            with reasoning_v2:
                render_events(events_v2, reasoning_v2)

            # Check final output
            messages = result_v2.get("messages", [])
            final_content = ""
            if messages:
                final_content = getattr(messages[-1], "content", "")

            has_latex = "\\documentclass" in final_content if final_content else False

            if has_latex:
                status_v2.markdown("<span class='status-done'>COMPLETE</span>",
                                   unsafe_allow_html=True)
            else:
                status_v2.markdown("<span class='status-error'>FINISHED</span>",
                                   unsafe_allow_html=True)

            # Metrics
            m1, m2, m3 = st.columns(3)
            m1.markdown(f"<div class='metric-card'><div class='value'>{time_v2:.1f}s</div>"
                        f"<div class='label'>Total Time</div></div>", unsafe_allow_html=True)

            tool_calls = sum(1 for e in events_v2 if e["type"] == "tool_call")
            thinking_steps = sum(1 for e in events_v2 if e["type"] == "thinking")
            m2.markdown(f"<div class='metric-card'><div class='value'>{tool_calls}</div>"
                        f"<div class='label'>Tool Calls</div></div>", unsafe_allow_html=True)
            m3.markdown(f"<div class='metric-card'><div class='value'>{thinking_steps}</div>"
                        f"<div class='label'>Thinking Steps</div></div>", unsafe_allow_html=True)

            # LaTeX output
            if has_latex:
                doc_start = final_content.find("\\documentclass")
                latex_v2 = final_content[doc_start:]
                with st.expander("LaTeX Output", expanded=True):
                    st.code(latex_v2, language="latex")

            # Files written
            files = result_v2.get("files", {})
            if files:
                with st.expander("Files Created by Agent"):
                    for path in files:
                        st.markdown(f"📁 `{path}`")

    # ── Comparison bar (if both ran) ──────────────────────
    if run_v1 and run_v2:
        st.markdown("---")
        st.markdown("### Performance Comparison")

        c1, c2, c3 = st.columns(3)

        with c1:
            t1 = time_v1 if 'time_v1' in dir() or 'time_v1' in locals() else 0
            t2 = time_v2 if 'time_v2' in dir() or 'time_v2' in locals() else 0
            faster = "v1" if t1 < t2 else "v2"
            diff = abs(t1 - t2)
            st.metric("Latency Winner", f"{faster}", f"{diff:.1f}s faster")

        with c2:
            ev1 = len(tracer_v1.get_all()) if 'tracer_v1' in locals() else 0
            ev2 = len(tracer_v2.get_all()) if 'tracer_v2' in locals() else 0
            st.metric("v1 Events", ev1)

        with c3:
            st.metric("v2 Events", ev2)


if __name__ == "__main__":
    main()
