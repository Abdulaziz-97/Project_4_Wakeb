"""
Validation Agent — lightweight LLM check on retrieved answers.
Receives {query, answer, date, sources} and decides: VALID or INVALID.
No tools, no searching — just judgment.

"""
from datetime import datetime

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage

from config.settings import DEEPSEEK_API_KEY, DEEPSEEK_BASE_URL, DEEPSEEK_MODEL
from agent.state import WeatherAgentState, CRAGOutput, RDEMError, AgentStep
from agent.logger import log_node, create_tracer, get_logger

_judge = ChatOpenAI(
    model=DEEPSEEK_MODEL,
    api_key=DEEPSEEK_API_KEY,
    base_url=DEEPSEEK_BASE_URL,
    temperature=0.0,
    timeout=30,
)


@log_node
def validation_agent(state: WeatherAgentState) -> dict:
    """
    Validate the retrieved answer against the query and current date.
    """
    logger = get_logger()


    # If no answer to validate, mark as invalid
    if not state.crag_output or not state.crag_output.answer:
        logger.info("  [validator] No answer to validate — routing to consultant")

        return {
            "validation_passed": False,
            "audit_trail": state.audit_trail + [
                AgentStep(
                    node_name="validation_agent",
                    status="skip",
                    timestamp=datetime.utcnow().isoformat(),
                )
            ],
        }

    today = datetime.utcnow().strftime("%A, %B %d, %Y")
    sources_str = "\n".join(state.crag_output.sources[:10]) if state.crag_output.sources else "No sources"

    prompt = (
        "You are a weather answer validator. Your ONLY job is to check if "
        "an answer is valid. You have NO tools — just judge the data.\n\n"
        f"TODAY'S DATE: {today}\n"
        f"USER QUERY: {state.user_query}\n\n"
        f"SOURCES:\n{sources_str}\n\n"
        f"ANSWER:\n{state.crag_output.answer[:2000]}\n\n"
        "IMPORTANT: Ignore any disclaimers, hedging, or statements like "
        "'recommend checking other sources'. Judge ONLY the actual data.\n\n"
        "Check these 4 things:\n"
        "1. LOCATION — Does the answer match the location in the query?\n"
        "2. DATA — Does it contain concrete weather data (temperatures, conditions)?\n"
        "3. CURRENT — Are the dates in the answer from today or this week "
        f"({today})? April 2026 IS current.\n"
        "4. COVERAGE — Does it cover what was asked (e.g. 'this week' = "
        "multi-day forecast)?\n\n"
        "Respond in this EXACT format:\n"
        "Location: yes/no\n"
        "Data: yes/no\n"
        "Current: yes/no\n"
        "Coverage: yes/no\n"
        "VERDICT: VALID or INVALID\n"
        "REASON: one sentence explaining why"
    )

    tracer = create_tracer("validation_agent")
    response = _judge.invoke(
        [HumanMessage(content=prompt)],
        config={"callbacks": [tracer]},
    )

    # Parse all fields
    lines = response.content.strip().splitlines()
    checks = {}
    verdict_line = ""
    reason_line = ""
    for line in lines:
        stripped = line.strip()
        upper = stripped.upper()
        if upper.startswith("LOCATION:"):
            checks["location"] = "YES" in upper
        elif upper.startswith("DATA:"):
            checks["data"] = "YES" in upper
        elif upper.startswith("CURRENT:"):
            checks["current"] = "YES" in upper
        elif upper.startswith("COVERAGE:"):
            checks["coverage"] = "YES" in upper
        elif upper.startswith("VERDICT:"):
            verdict_line = upper
        elif upper.startswith("REASON:"):
            reason_line = stripped

    is_valid = "VALID" in verdict_line and "INVALID" not in verdict_line

    logger.info(
        f"  [validator] Verdict: {'VALID' if is_valid else 'INVALID'} "
        f"| location={checks.get('location')} data={checks.get('data')} "
        f"current={checks.get('current')} coverage={checks.get('coverage')}"
    )
    if reason_line:
        logger.info(f"  [validator] {reason_line}")

    return {
        "validation_passed": is_valid,
        "audit_trail": state.audit_trail + [
            AgentStep(
                node_name="validation_agent",
                status="valid" if is_valid else "invalid",
                timestamp=datetime.utcnow().isoformat(),
            )
        ],
    }