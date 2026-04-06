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
    temperature=0.1,
    timeout=60,
)

_react_agent = create_react_agent(
    model=llm,
    tools=[celsius_to_fahrenheit],
    prompt=(
        "You are a professional weather report writer.\n"
        "You have a celsius_to_fahrenheit tool — use it for EVERY temperature "
        "conversion. Never guess Fahrenheit values.\n\n"
        "CRITICAL RULES:\n"
        "1. You MUST produce a full structured weather report using the data "
        "provided. NEVER refuse to write. NEVER say 'I cannot help' or "
        "'data is unavailable'. Work with whatever data you have.\n"
        "2. Extract ALL concrete data points from the source (temperatures, "
        "humidity, wind, conditions) and present them clearly.\n"
        "3. Use your celsius_to_fahrenheit tool for every temperature.\n"
        "4. Include numbered source citations [1], [2], etc.\n\n"
        "Report structure:\n"
        "## Weather Report: {location}\n"
        "### Current Conditions\n"
        "### Temperature Summary (use a table)\n"
        "### Forecast\n"
        "### Weather Warnings\n"
        "### Sources\n\n"
        "If confidence is low (action was 'incorrect' or 'web_only'), add a "
        "note: 'Data sourced from live web search.'\n\n"
        "Be factual. Include every number from the source data."
    ),
)


@log_node
def writer_agent(state: WeatherAgentState) -> dict:
    crag = state.crag_output

    user_msg = (
        f"CRAG action: {crag.action}, confidence: {crag.max_score:.2f}\n"
        f"Sources: {crag.sources}\n\n"
        f"Weather query: {state.user_query}\n\n"
        f"Retrieved data to use (write a report based on ALL of this):\n"
        f"{crag.answer}"
    )

    tracer = create_tracer("writer")
    result = _react_agent.invoke(
        {"messages": [HumanMessage(content=user_msg)]},
        config={"callbacks": [tracer]},
    )
    response = result["messages"][-1]

    return {
        "draft_document": response.content,
        "fact_check_result": None,
        "fact_fix_attempts": 0,
        "writer_rdem": None,
        "audit_trail": state.audit_trail + [
            AgentStep(
                node_name="writer",
                status="success",
                timestamp=datetime.utcnow().isoformat(),
            )
        ],
    }
