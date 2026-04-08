from langgraph.graph import StateGraph, START, END
from langgraph.types import RetryPolicy
from langgraph.checkpoint.memory import MemorySaver

from agent.state import WeatherAgentState
from agent.nodes.orchestrator import orchestration_agent
from agent.nodes.retriever import retriever_agent
from agent.nodes.validation_agent import validation_agent
from agent.nodes.weather_consultant import weather_consultant
from agent.nodes.writer import writer_agent
from agent.nodes.formatter import formatter


def build_graph():
    builder = StateGraph(WeatherAgentState)

    # ── Add Nodes ───────────────────────────────────────────────
    builder.add_node("orchestration_agent", orchestration_agent)
    builder.add_node("retriever_agent", retriever_agent)
    builder.add_node("validation_agent", validation_agent)

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
            "format": "formatter",
            "done": END,
        }
        return routes.get(state.current_step, "retriever_agent")

    builder.add_conditional_edges("orchestration_agent", route_from_orchestrator)

    # Retriever → Validation Agent (lightweight check)
    builder.add_edge("retriever_agent", "validation_agent")

    # Validation routing:
    #   VALID   → orchestrator (skip consultant, save ~100s)
    #   INVALID → consultant (fix the answer)
    def route_from_validator(state: WeatherAgentState) -> str:
        if state.validation_passed:
            return "orchestration_agent"
        return "weather_consultant"

    builder.add_conditional_edges("validation_agent", route_from_validator)

    # Weather consultant always returns to orchestrator
    builder.add_edge("weather_consultant", "orchestration_agent")

    # Writer goes directly to formatter
    builder.add_edge("writer_agent", "formatter")

    # Formatter ends the graph
    builder.add_edge("formatter", END)

    # ── Compile with Memory ─────────────────────────────────────
    memory = MemorySaver()
    graph = builder.compile(checkpointer=memory)
    return graph