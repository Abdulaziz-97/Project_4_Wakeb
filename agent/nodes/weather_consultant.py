import re
from datetime import datetime

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage
from langgraph.prebuilt import create_react_agent

from config.settings import DEEPSEEK_API_KEY, DEEPSEEK_BASE_URL, DEEPSEEK_MODEL
from agent.state import WeatherAgentState, CRAGOutput, RDEMError, AgentStep
from agent.tools.weather_tools import web_search
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
    tools=[web_search],
    prompt=(
        "You are a weather consultant agent — the fallback when the primary "
        "retrieval system could not provide relevant, current data.\n"
        "You have a web_search tool. Use it to find CURRENT weather data for "
        "the user's query. Call the tool multiple times with different queries "
        "to get comprehensive coverage:\n"
        "  1. Current conditions (e.g. 'Berlin current weather temperature')\n"
        "  2. Weekly forecast (e.g. 'Berlin 7 day weather forecast')\n"
        "  3. Warnings/advisories if relevant\n\n"
        "After gathering data, compile a comprehensive answer that includes:\n"
        "- Concrete temperature values in Celsius\n"
        "- Sky conditions, humidity, wind\n"
        "- Multi-day forecast if available\n"
        "- Source URLs for every piece of data\n\n"
        "Be factual. Include numbers. Do not guess."
    ),
)

_URL_PATTERN = re.compile(r"https?://[^\s\]\)\"',]+")


@log_node
def weather_consultant(state: WeatherAgentState) -> dict:
    crag_context = ""
    if state.crag_output and state.crag_output.answer:
        crag_context = (
            "\n\nPrevious retrieval attempt returned this data (it may be "
            "partially useful — verify and augment, do NOT just repeat it):\n"
            f"{state.crag_output.answer[:1000]}"
        )

    tracer = create_tracer("weather_consultant")
    try:
        result = _react_agent.invoke(
            {"messages": [HumanMessage(
                content=f"Find current weather information for: {state.user_query}{crag_context}"
            )]},
            config={"callbacks": [tracer]},
        )
        response = result["messages"][-1]
    except Exception as e:
        rdem = RDEMError(
            node="weather_consultant",
            error_type="api_fail",
            message=f"Weather consultant ReAct agent failed: {e}",
            suggestion="Proceed with empty context. Writer will note data unavailability.",
            attempt=1,
        )
        empty_output = CRAGOutput(
            answer="Weather data is currently unavailable.",
            sources=[],
            action="web_only",
            scores=[],
            max_score=0.0,
            is_high_confidence=False,
        )
        return {
            "crag_output": empty_output,
            "crag_rdem": None,
            "global_error_log": state.global_error_log + [rdem],
            "audit_trail": state.audit_trail + [
                AgentStep(
                    node_name="weather_consultant",
                    status="error",
                    error=rdem,
                    timestamp=datetime.utcnow().isoformat(),
                )
            ],
        }

    sources = set()
    for msg in result["messages"]:
        content = getattr(msg, "content", "")
        if isinstance(content, str):
            for url in _URL_PATTERN.findall(content):
                url = url.rstrip(".,;:)")
                sources.add(url)

    source_list = sorted(sources)

    crag_output = CRAGOutput(
        answer=response.content,
        sources=[f"[web] {s}" for s in source_list],
        action="web_only",
        scores=[],
        max_score=0.0,
        is_high_confidence=False,
    )

    return {
        "crag_output": crag_output,
        "crag_rdem": None,
        "messages": [response],
        "audit_trail": state.audit_trail + [
            AgentStep(
                node_name="weather_consultant",
                status="success",
                timestamp=datetime.utcnow().isoformat(),
            )
        ],
    }
