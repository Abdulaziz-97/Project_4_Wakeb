from langgraph.graph import StateGraph, START, END
from langgraph.types import RetryPolicy
from langgraph.checkpoint.memory import MemorySaver

from agent.state import WeatherAgentState
from agent.nodes.orchestrator import orchestration_agent
from agent.nodes.retriever import retriever_agent
from agent.nodes.weather_consultant import weather_consultant
from agent.nodes.writer import writer_agent
from agent.nodes.fact_checker import writer_fact_checker
from agent.nodes.fix_fact import fix_fact_agent
from agent.nodes.formatter import formatter


def build_graph():
    builder = StateGraph(WeatherAgentState)

    # ── Add Nodes ───────────────────────────────────────────────
    builder.add_node("orchestration_agent", orchestration_agent)

    builder.add_node("retriever_agent", retriever_agent)

    builder.add_node(
        "weather_consultant",
        weather_consultant,
        retry=RetryPolicy(
            max_attempts=3,
            initial_interval=1.0,
            backoff_factor=2.0,
            retry_on=(ConnectionError, TimeoutError, Exception),
        ),
    )

    _default_retry = RetryPolicy(
        max_attempts=2,
        initial_interval=1.0,
        backoff_factor=2.0,
        retry_on=(ConnectionError, TimeoutError, Exception),
    )

    builder.add_node("writer_agent", writer_agent, retry=_default_retry)
    builder.add_node("writer_fact_checker", writer_fact_checker, retry=_default_retry)
    builder.add_node("fix_fact_agent", fix_fact_agent, retry=_default_retry)
    builder.add_node("formatter", formatter, retry=_default_retry)

    # ── Add Edges ───────────────────────────────────────────────

    # Entry point
    builder.add_edge(START, "orchestration_agent")

    # Orchestrator routing
    def route_from_orchestrator(state: WeatherAgentState) -> str:
        routes = {
            "retrieve": "retriever_agent",
            "weather_fallback": "weather_consultant",
            "write": "writer_agent",
            "fact_check": "writer_fact_checker",
            "format": "formatter",
            "done": END,
        }
        return routes.get(state.current_step, "retriever_agent")

    builder.add_conditional_edges("orchestration_agent", route_from_orchestrator)

    # Retriever: exception → fallback, success → orchestrator
    def route_from_retriever(state: WeatherAgentState) -> str:
        if state.crag_rdem is not None:
            return "weather_consultant"
        return "orchestration_agent"

    builder.add_conditional_edges("retriever_agent", route_from_retriever)

    # Weather consultant always returns to orchestrator
    builder.add_edge("weather_consultant", "orchestration_agent")

    # Writer always goes to fact checker
    builder.add_edge("writer_agent", "writer_fact_checker")

    # Fact checker routing (the fix loop)
    def route_from_fact_checker(state: WeatherAgentState) -> str:
        if state.fact_check_result and state.fact_check_result.is_factual:
            return "orchestration_agent"
        if state.fact_fix_attempts >= 3:
            return "orchestration_agent"
        return "fix_fact_agent"

    builder.add_conditional_edges("writer_fact_checker", route_from_fact_checker)

    # Fix fact loops back to fact checker
    builder.add_edge("fix_fact_agent", "writer_fact_checker")

    # Formatter ends the graph
    builder.add_edge("formatter", END)

    # ── Compile with Memory ─────────────────────────────────────
    memory = MemorySaver()
    graph = builder.compile(checkpointer=memory)
    return graph
