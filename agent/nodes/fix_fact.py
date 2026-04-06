from datetime import datetime

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage
from langgraph.prebuilt import create_react_agent

from config.settings import DEEPSEEK_API_KEY, DEEPSEEK_BASE_URL, DEEPSEEK_MODEL
from agent.state import WeatherAgentState, AgentStep
from agent.tools.weather_tools import celsius_to_fahrenheit
from agent.logger import log_node, create_tracer

llm = ChatOpenAI(
    model=DEEPSEEK_MODEL,
    api_key=DEEPSEEK_API_KEY,
    base_url=DEEPSEEK_BASE_URL,
    temperature=0.0,
    timeout=60,
)

_react_agent = create_react_agent(
    model=llm,
    tools=[celsius_to_fahrenheit],
    prompt=(
        "You are a fact-correction agent with ReAct capabilities.\n"
        "You have a celsius_to_fahrenheit tool. Use it to recalculate any "
        "temperature conversions that were flagged as incorrect.\n\n"
        "Fix ALL of the listed issues. Do not change any correct information.\n"
        "Return ONLY the corrected report text. No explanations."
    ),
)


@log_node
def fix_fact_agent(state: WeatherAgentState) -> dict:
    fc = state.fact_check_result
    temp_detail = ""
    if fc and fc.verified_temperatures:
        temp_lines = [f"  {c}C -> {f}F" for c, f in fc.verified_temperatures.items()]
        temp_detail = "\nVerified temperatures (tool-computed ground truth):\n" + "\n".join(temp_lines)

    issues_detail = ""
    if fc and fc.issues:
        issues_detail = "\nSpecific issues found:\n" + "\n".join(f"  - {i}" for i in fc.issues)

    user_msg = (
        f"Issues to fix: {state.writer_rdem.message}\n"
        f"Suggestion: {state.writer_rdem.suggestion}\n"
        f"{issues_detail}{temp_detail}\n\n"
        f"Ground truth (CRAG original answer): {state.crag_output.answer}\n\n"
        f"Current draft to fix:\n{state.draft_document}"
    )

    tracer = create_tracer("fix_fact")
    result = _react_agent.invoke(
        {"messages": [HumanMessage(content=user_msg)]},
        config={"callbacks": [tracer]},
    )
    response = result["messages"][-1]

    return {
        "draft_document": response.content,
        "fact_check_result": None,
        "audit_trail": state.audit_trail + [
            AgentStep(
                node_name="fix_fact",
                status="success",
                timestamp=datetime.utcnow().isoformat(),
            )
        ],
    }
