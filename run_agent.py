\
\
\
\
\
\
   

import sys
import os

sys.path.insert(0, os.path.dirname(__file__))

import dspy
from config.settings import DEEPSEEK_API_KEY, DEEPSEEK_BASE_URL, DEEPSEEK_MODEL


def setup_dspy():
    lm = dspy.LM(
        model=f"openai/{DEEPSEEK_MODEL}",
        api_key=DEEPSEEK_API_KEY,
        api_base=DEEPSEEK_BASE_URL,
        temperature=0.0,
        max_tokens=2000,
    )
    dspy.configure(lm=lm)


def _get(obj, key, default=None):
    if isinstance(obj, dict):
        return obj.get(key, default)
    return getattr(obj, key, default)


def print_result(result, label="RESULT"):
    print("=" * 70)
    print(f"  {label}")
    print("=" * 70)

    latex = result.get("final_latex_document")
    if latex:
        preview = latex[:500] + ("..." if len(latex) > 500 else "")
        print(f"\n[Output] ({len(latex)} chars)")
        print(preview)
    else:
        print("\n[Output] No output generated")

    crag = result.get("crag_output")
    if crag:
        action = _get(crag, "action", "N/A")
        score = _get(crag, "max_score", 0.0)
        sources = _get(crag, "sources", [])
        print(f"\n[CRAG] action={action}  score={score:.2f}  sources={len(sources)}")
        for i, src in enumerate(sources, 1):
            print(f"  [{i}] {src}")

    trail = result.get("audit_trail", [])
    print(f"\n[Audit] {len(trail)} steps")
    for step in trail:
        icon = {"success": "OK", "forced_proceed": "!!"}.get(_get(step, "status", ""), "XX")
        print(f"  {icon} [{_get(step, 'node_name', '?')}] {_get(step, 'status', '?')}")

    errors = result.get("global_error_log", [])
    if errors:
        print(f"\n[Errors] {len(errors)}")
        for err in errors:
            print(f"  - [{_get(err, 'node', '?')}] {_get(err, 'error_type', '?')}: "
                  f"{str(_get(err, 'message', ''))[:120]}")

    print("\n" + "=" * 70)


def main():
    query = sys.argv[1] if len(sys.argv) > 1 else "What is the weather in Riyadh today?"

    setup_dspy()

    from agent.graph import build_graph
    from agent.state import WeatherAgentState
    from agent.logger import init_run

    init_run()
    graph = build_graph()

    config = {"configurable": {"thread_id": "cli_session"}}
    initial_state = WeatherAgentState(user_query=query).model_dump()

    result = graph.invoke(initial_state, config=config)
    print_result(result, label=f"QUERY: {query}")


if __name__ == "__main__":
    main()
