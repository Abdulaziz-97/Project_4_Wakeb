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

_react_agent = create_react_agent(
    model=llm,
    tools=[web_search],  
    prompt=(
        "You are a weather data quality assistant. Be FAST and EFFICIENT.\n\n"
        "USER QUERY: Check the query and previous answer (if any).\n\n"
        "SEARCH LIMIT: You can make AT MOST 1 combined search.\n"
        "Call web_search ONCE with all info you need in one query.\n\n"
        "STRATEGY:\n"
        "1. If previous answer is recent and looks good → use it as-is\n"
        "2. If outdated/missing data → make ONE combined search:\n"
        "   '[location] current weather temperature forecast [time period]'\n"
        "3. Compile final answer from that one search result\n\n"
        "FINAL ANSWER MUST HAVE:\n"
        "- Temperatures in Celsius (current + forecast)\n"
        "- Conditions (rain, sun, wind, humidity)\n"
        "- Time period covered (today, this week, etc.)\n"
        "- Source URLs\n\n"
        "BE FAST. ONE SEARCH ONLY."
    ),
)

_URL_PATTERN = re.compile(r"https?://[^\s\]\)\"',]+")


@log_node
def weather_consultant(state: WeatherAgentState) -> dict:
    crag_context = ""
    if state.crag_output and state.crag_output.answer:
        action_hint = {
            "correct": "This answer came from local documents only (may be outdated).",
            "ambiguous": "This answer is a MIX of local documents + web search. Some parts may be good, some bad.",
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
            #Save this web answer to vector store!
    ingest_answer(state.user_query, crag_output, node_name="weather_consultant")

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