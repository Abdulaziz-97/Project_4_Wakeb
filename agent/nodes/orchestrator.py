from datetime import datetime

from langchain_core.messages import AIMessage

from agent.state import WeatherAgentState, AgentStep, RDEMError
from agent.logger import log_node, get_logger

_STEP_DESCRIPTIONS = {
    "retrieve": "Routing to retriever to fetch weather data via CRAG.",
    "weather_fallback": "CRAG data was insufficient. Routing to weather consultant for live web search.",
    "write": "Data retrieved successfully. Routing to writer to draft the weather report.",
    "fact_check": "Draft ready. Routing to fact checker for verification.",
    "format": "Fact check passed. Routing to formatter for LaTeX output.",
    "done": "Pipeline complete. Final LaTeX document is ready.",
}


@log_node
def orchestration_agent(state: WeatherAgentState) -> dict:
    step_status = "success"

    if state.final_latex_document is not None:
        next_step = "done"
    elif state.fact_check_result is not None and state.fact_check_result.is_factual:
        next_step = "format"
    elif state.fact_check_result is not None and not state.fact_check_result.is_factual:
        if state.fact_fix_attempts >= 3:
            next_step = "format"
            step_status = "forced_proceed"
        else:
            next_step = "fact_check"
    elif state.draft_document is not None and state.fact_check_result is None:
        next_step = "fact_check"
    elif state.crag_output is not None:
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
