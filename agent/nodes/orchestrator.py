from datetime import datetime

from langchain_core.messages import AIMessage

from agent.state import WeatherAgentState, AgentStep
from agent.logger import log_node, get_logger

_STEP_DESCRIPTIONS = {
    "retrieve": "Routing to retriever to fetch weather data via CRAG.",
    "weather_fallback": "CRAG data was insufficient. Routing to weather consultant for live web search.",
    "write": "Data retrieved successfully. Routing to writer to draft the weather report.",
    "format": "Draft ready. Routing to formatter for final output.",
    "done": "Pipeline complete. Final document is ready.",
}


@log_node
def orchestration_agent(state: WeatherAgentState) -> dict:
    step_status = "success"

    if state.final_latex_document is not None:
        next_step = "done"
    elif state.draft_document is not None:
        next_step = "format"
    elif state.crag_output is not None and state.validation_passed is True:
        next_step = "write"
    elif state.crag_rdem is not None:
        next_step = "weather_fallback"
    else:
        next_step = "retrieve"

    description = _STEP_DESCRIPTIONS.get(next_step, f"Routing to {next_step}.")
    status_msg = f"[Orchestrator] Query: '{state.user_query}' | {description}"

    logger = get_logger()
    logger.info(f"  [orchestrator] Decision: {next_step} — {description}")

    return {
        "current_step": next_step,
        "messages": [AIMessage(content=status_msg)],
        "audit_trail": state.audit_trail + [
            AgentStep(
                node_name="orchestrator",
                status=step_status,
                timestamp=datetime.utcnow().isoformat(),
            )
        ],
    }
