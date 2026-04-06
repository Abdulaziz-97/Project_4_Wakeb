"""
Run this script to execute the full LangGraph multi-agent weather pipeline.

Usage:
    python run_agent.py
"""

import sys
import os

sys.path.insert(0, os.path.dirname(__file__))

import dspy
from config.settings import DEEPSEEK_API_KEY, DEEPSEEK_BASE_URL, DEEPSEEK_MODEL


def setup_dspy():
    """Configure DSPy -- MUST be called before CRAGPipeline is instantiated."""
    lm = dspy.LM(
        model=f"openai/{DEEPSEEK_MODEL}",
        api_key=DEEPSEEK_API_KEY,
        api_base=DEEPSEEK_BASE_URL,
        temperature=0.0,
        max_tokens=1000,
    )
    dspy.configure(lm=lm)


def _get(obj, key, default=None):
    """Access a field on a Pydantic model or a dict transparently."""
    if isinstance(obj, dict):
        return obj.get(key, default)
    return getattr(obj, key, default)


def print_result(result, label="RESULT"):
    """Pretty-print a pipeline result with all sections."""
    print("=" * 70)
    print(f"  {label}")
    print("=" * 70)

    # -- LaTeX --
    latex = result.get("final_latex_document")
    if latex:
        preview = latex[:500] + ("..." if len(latex) > 500 else "")
        print(f"\n[LaTeX Document] ({len(latex)} chars)")
        print(preview)
    else:
        print("\n[LaTeX Document] No output generated")

    # -- CRAG output --
    crag = result.get("crag_output")
    if crag:
        action = _get(crag, "action", "N/A")
        score = _get(crag, "max_score", 0.0)
        sources = _get(crag, "sources", [])
        confidence = _get(crag, "is_high_confidence", False)

        print(f"\n[CRAG Retrieval]")
        print(f"  Action:     {action}")
        print(f"  Max Score:  {score:.2f}")
        print(f"  Confidence: {'HIGH' if confidence else 'LOW'}")
        print(f"  Sources ({len(sources)}):")
        for i, src in enumerate(sources, 1):
            print(f"    [{i}] {src}")
    else:
        print("\n[CRAG Retrieval] N/A")

    # -- Audit trail --
    trail = result.get("audit_trail", [])
    print(f"\n[Audit Trail] ({len(trail)} steps)")
    for step in trail:
        status = _get(step, "status", "?")
        name = _get(step, "node_name", "?")
        icon = "OK" if status == "success" else "!!" if status == "forced_proceed" else "XX"
        print(f"  {icon} [{name}] {status}")

    # -- Errors --
    errors = result.get("global_error_log", [])
    print(f"\n[Errors] {len(errors)} total")
    for err in errors:
        node = _get(err, "node", "?")
        etype = _get(err, "error_type", "?")
        msg = _get(err, "message", "")
        print(f"  - [{node}] {etype}: {msg[:120]}")

    # -- Fact check --
    fc = result.get("fact_check_result")
    if fc:
        factual = _get(fc, "is_factual", None)
        issues = _get(fc, "issues", [])
        temps = _get(fc, "verified_temperatures", {})
        print(f"\n[Fact Check]")
        print(f"  Factual: {factual}")
        print(f"  Issues:  {issues}")
        print(f"  Verified Temps: {temps}")

    print("\n" + "=" * 70)


def main():
    setup_dspy()

    from agent.graph import build_graph
    from agent.state import WeatherAgentState

    graph = build_graph()

    config = {"configurable": {"thread_id": "abdulaziz_session_1"}}

    initial_state = WeatherAgentState(
        user_query="What is the weather in Berlin this week?",
    ).model_dump()

    result = graph.invoke(initial_state, config=config)
    print_result(result, label="QUERY 1: Berlin Weather")

    # Follow-up turn — reset pipeline fields so the new query runs fresh,
    # but keep the same thread_id so conversation memory (messages) persists.
    result2 = graph.invoke(
        {
            "user_query": "What about Hamburg?",
            "messages": [],
            "crag_output": None,
            "crag_rdem": None,
            "draft_document": None,
            "fact_check_result": None,
            "fact_fix_attempts": 0,
            "writer_rdem": None,
            "final_latex_document": None,
            "current_step": "retrieve",
            "audit_trail": [],
            "global_error_log": [],
        },
        config=config,
    )
    print_result(result2, label="QUERY 2 (follow-up): Hamburg Weather")


if __name__ == "__main__":
    main()
