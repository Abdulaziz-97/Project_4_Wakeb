"""Run the pipeline and dump full LaTeX output."""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))

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

from agent.graph import build_graph
from agent.state import WeatherAgentState

graph = build_graph()


def _get(obj, key, default=None):
    if isinstance(obj, dict):
        return obj.get(key, default)
    return getattr(obj, key, default)


def show(result, label, tex_file):
    print("=" * 70)
    print(f"  {label}")
    print("=" * 70)

    latex = result.get("final_latex_document", "No output")
    with open(tex_file, "w", encoding="utf-8") as f:
        f.write(latex)
    print(latex)

    print("\n" + "-" * 70)
    crag = result.get("crag_output")
    if crag:
        print(f"CRAG Action:  {_get(crag, 'action')}")
        print(f"Max Score:    {_get(crag, 'max_score', 0)}")
        conf = _get(crag, 'is_high_confidence', False)
        print(f"Confidence:   {'HIGH' if conf else 'LOW'}")
        sources = _get(crag, "sources", [])
        print(f"Sources ({len(sources)}):")
        for i, s in enumerate(sources, 1):
            print(f"  [{i}] {s}")

    trail = result.get("audit_trail", [])
    print(f"\nAudit Trail ({len(trail)} steps):")
    for step in trail:
        name = _get(step, "node_name", "?")
        status = _get(step, "status", "?")
        icon = "OK" if status == "success" else "!!" if status == "forced_proceed" else "XX"
        print(f"  {icon} [{name}] {status}")

    errors = result.get("global_error_log", [])
    print(f"\nErrors: {len(errors)}")
    for err in errors:
        print(f"  - [{_get(err,'node')}] {_get(err,'error_type')}: {_get(err,'message','')[:120]}")

    fc = result.get("fact_check_result")
    if fc:
        print(f"\nFact Check: factual={_get(fc,'is_factual')}, issues={_get(fc,'issues',[])}")

    print(f"\nSaved to: {tex_file}")
    print("=" * 70)


config = {"configurable": {"thread_id": "demo_live_v2"}}

state = WeatherAgentState(
    user_query="What is the weather in Berlin this week?",
).model_dump()

result = graph.invoke(state, config=config)
show(result, "QUERY: Berlin Weather This Week", "output_berlin.tex")
