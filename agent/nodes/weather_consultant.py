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
        "You are a weather consultant agent — the fallback when the primary "
        "retrieval system could not provide relevant, current data.\n\n"
        "You will receive a USER QUERY and possibly a PREVIOUS ANSWER that "
        "was rejected by the quality gate.\n\n"
        "If a previous answer is provided:\n"
        "1. First EVALUATE it — identify what is wrong (outdated data? "
        "wrong location? missing forecast? incomplete?)\n"
        "2. Then search specifically to FIX those gaps\n"
        "3. Keep any parts of the previous answer that are correct and current\n\n"
        "If no previous answer is provided, do a fresh search.\n\n"
        "You have a web_search tool. Call it multiple times:\n"
        "  1. Current conditions (e.g. 'Berlin current weather temperature')\n"
        "  2. Weekly forecast (e.g. 'Berlin 7 day weather forecast')\n"
        "  3. Warnings/advisories if relevant\n\n"
        "Your final answer MUST include:\n"
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
            f"\n\nPREVIOUS ANSWER (rejected by quality gate — evaluate what's wrong "
            f"and fix it):\n{state.crag_output.answer[:1500]}"
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