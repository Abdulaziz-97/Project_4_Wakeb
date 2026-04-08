import re
from datetime import datetime
from crag.answer_ingest import ingest_answer
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


def _build_consultant_prompt() -> str:
    today = datetime.now().strftime("%A, %B %d, %Y")
    return (
        f"Today's date is {today}.\n\n"
        "You are a weather consultant. You receive a USER QUERY and possibly "
        "a PREVIOUS ANSWER from a local vector store.\n\n"
        "RULES:\n"
        "1. If the previous answer already has correct, current data for the "
        "   right location and time period — return it as-is. Say: "
        "'Verified: answer is accurate and current.' Do NOT search.\n"
        "2. Otherwise, call web_search ONCE with a single comprehensive query "
        "   that covers current conditions AND the weekly forecast "
        "   (e.g. 'Dammam Saudi Arabia weather this week forecast temperature'). "
        "   Do NOT make multiple search calls.\n"
        "3. After receiving the search result, compile a final answer that includes:\n"
        "   - Concrete temperatures in Celsius\n"
        "   - Sky conditions, humidity, wind\n"
        "   - Multi-day forecast if available\n"
        "   - Source URLs\n\n"
        "Be factual. Include numbers. Do not guess."
    )


_react_agent = create_react_agent(
    model=llm,
    tools=[web_search],
    prompt=_build_consultant_prompt(),
)

_URL_PATTERN = re.compile(r"https?://[^\s\]\)\"',]+")
_TEMP_PATTERN = re.compile(r"-?\d+\.?\d*\s*°?[CcFf]")


def _quick_validate(query: str, answer: str) -> bool:
                                                                                  
    if not answer or len(answer.strip()) < 50:
        return False

    lower = answer.lower()
    query_lower = query.lower()

    has_temp = bool(_TEMP_PATTERN.search(answer))
    has_weather_words = any(w in lower for w in [
        "temperature", "forecast", "humidity", "wind", "rain",
        "cloud", "sunny", "snow", "storm", "celsius", "fahrenheit",
    ])

    query_words = {w for w in query_lower.split() if len(w) > 3}
    has_location = any(w in lower for w in query_words) if query_words else True

    is_error = any(w in lower for w in [
        "unavailable", "could not", "failed to", "no results",
        "i don't have", "i cannot",
    ])

    return (has_temp or has_weather_words) and has_location and not is_error


@log_node
def weather_consultant(state: WeatherAgentState) -> dict:
    crag_context = ""
    if state.crag_output and state.crag_output.answer:
        action_hint = {
            "correct": "This answer came from the local vector store (ingested forecast data).",
            "ambiguous": "This answer is a MIX of local documents + web search.",
            "incorrect": "Local documents were irrelevant. This answer came from web search.",
            "web_only": "This answer came entirely from web search.",
        }.get(state.crag_output.action, "")
        crag_context = (
            f"\n\nSOURCE INFO: {action_hint}\n"
            f"\nPREVIOUS ANSWER (evaluate — keep good parts, fix bad parts):\n"
            f"{state.crag_output.answer[:1500]}"
        )

    tracer = create_tracer("weather_consultant")
    try:
        result = _react_agent.invoke(
            {"messages": [HumanMessage(
                content=f"USER QUERY: {state.user_query}{crag_context}"
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
            "validation_passed": True,
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

    crag_output = CRAGOutput(
        answer=response.content,
        sources=[f"[web] {s}" for s in sorted(sources)],
        action="web_only",
        scores=[],
        max_score=0.0,
        is_high_confidence=False,
    )

    is_good = _quick_validate(state.user_query, crag_output.answer)
    if is_good:
        ingest_answer(state.user_query, crag_output, node_name="weather_consultant")

    return {
        "crag_output": crag_output,
        "crag_rdem": None,
        "validation_passed": is_good,
        "messages": [response],
        "audit_trail": state.audit_trail + [
            AgentStep(
                node_name="weather_consultant",
                status="success" if is_good else "unverified",
                timestamp=datetime.utcnow().isoformat(),
            )
        ],
    }
