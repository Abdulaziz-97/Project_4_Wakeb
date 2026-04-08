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

    builder.add_edge(START, "orchestration_agent")

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

    builder.add_edge("retriever_agent", "validation_agent")

    def route_from_validator(state: WeatherAgentState) -> str:
        if state.validation_passed:
            return "orchestration_agent"
        return "weather_consultant"

    builder.add_conditional_edges("validation_agent", route_from_validator)

    builder.add_edge("weather_consultant", "orchestration_agent")
    builder.add_edge("writer_agent", "formatter")
    builder.add_edge("formatter", END)

    memory = MemorySaver()
    return builder.compile(checkpointer=memory)
