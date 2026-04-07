from datetime import datetime

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage

from crag.pipeline import CRAGPipeline
from config.settings import DEEPSEEK_API_KEY, DEEPSEEK_BASE_URL, DEEPSEEK_MODEL
from agent.state import WeatherAgentState, CRAGOutput, RDEMError, AgentStep
from agent.logger import log_node, create_tracer, get_logger

_pipeline = CRAGPipeline()

_ACTION_LABELS = {
    "correct": "high-confidence local retrieval",
    "ambiguous": "mixed local + web search",
    "incorrect": "low local relevance — system used live web search to answer",
    "web_only": "no local data — system used live web search only",
}

_judge = ChatOpenAI(
    model=DEEPSEEK_MODEL,
    api_key=DEEPSEEK_API_KEY,
    base_url=DEEPSEEK_BASE_URL,
    temperature=0.0,
    timeout=60,
)


def _answer_is_relevant(query: str, answer: str, action: str) -> bool:
    """LLM quality gate -- the 'Is the retrieved info relevant?' diamond."""
    logger = get_logger()
    readable_action = _ACTION_LABELS.get(action, action)
    today = datetime.utcnow().strftime("%Y-%m-%d")
    prompt = (
        "You are a relevance judge for weather answers.\n\n"
        f"TODAY'S DATE: {today}\n\n"
        f"USER QUERY: {query}\n\n"
        f"ANSWER:\n{answer[:2000]}\n\n"
        "IMPORTANT: Ignore any disclaimers, hedging, or statements like "
        "'recommend checking other sources' or 'date discrepancy'. "
        "Judge ONLY the actual weather data provided.\n\n"
        "Think step by step:\n"
        "1. Does the answer mention the correct location?\n"
        "2. Does it contain concrete weather data (temperatures, conditions)?\n"
        "3. Do the dates in the data match today's date or this week? "
        f"(Today is {today}, so April 2026 IS current.)\n\n"
        "If ALL three are yes, it is relevant.\n\n"
        "Respond in this EXACT format:\n"
        "Location: yes/no\n"
        "Has data: yes/no\n"
        "Current: yes/no\n"
        "VERDICT: RELEVANT or NOT_RELEVANT"
    )
    tracer = create_tracer("retriever_quality_gate")
    response = _judge.invoke(
        [HumanMessage(content=prompt)],
        config={"callbacks": [tracer]},
    )
    # Parse verdict from last line only
    lines = response.content.strip().splitlines()
    last_line = lines[-1].upper() if lines else ""
    verdict = "RELEVANT" in last_line and "NOT_RELEVANT" not in last_line

    logger.info(
        f"  [retriever] Quality gate verdict: "
        f"{'RELEVANT' if verdict else 'NOT_RELEVANT'} "
        f"(action={action})"
    )
    logger.debug(f"  [retriever] Quality gate reasoning:\n{response.content}")
    return verdict


@log_node
def retriever_agent(state: WeatherAgentState) -> dict:
    logger = get_logger()

    # Step 1 -- Run CRAG
    logger.info(f"  [retriever] Calling CRAGPipeline.run('{state.user_query}')")
    try:
        result = _pipeline.run(state.user_query)
    except Exception as e:
        logger.error(f"  [retriever] CRAGPipeline EXCEPTION: {e}")
        rdem = RDEMError(
            node="retriever",
            error_type="crag_exception",
            message=f"CRAGPipeline.run() raised: {type(e).__name__}: {e}",
            suggestion="Route to weather_consultant for direct Tavily fallback.",
            attempt=1,
        )
        return {
            "crag_rdem": rdem,
            "crag_output": None,
            "global_error_log": state.global_error_log + [rdem],
            "audit_trail": state.audit_trail + [
                AgentStep(
                    node_name="retriever",
                    status="error",
                    error=rdem,
                    timestamp=datetime.utcnow().isoformat(),
                )
            ],
        }

    # Step 2 -- Log raw CRAG result
    logger.debug(f"  [retriever] CRAG raw result:")
    logger.debug(f"    action:  {result['action']}")
    logger.debug(f"    scores:  {result['scores']}")
    logger.debug(f"    sources: {result['sources']}")
    logger.debug(f"    answer:  {result['answer'][:300]}...")

    crag_output = CRAGOutput(
        answer=result["answer"],
        sources=result["sources"],
        action=result["action"],
        scores=result["scores"],
        max_score=max(result["scores"]) if result["scores"] else 0.0,
        is_high_confidence=(result["action"] == "correct"),
    )

    # Step 3 -- Quality gate
    relevant = _answer_is_relevant(
        state.user_query, crag_output.answer, crag_output.action
    )

    if not relevant:
        rdem = RDEMError(
            node="retriever",
            error_type="relevance_low",
            message=(
                f"CRAG returned action='{crag_output.action}' with score "
                f"{crag_output.max_score:.2f}, but the LLM quality gate "
                f"determined the answer does not adequately address the query."
            ),
            suggestion="Route to weather_consultant for fresh web search.",
            attempt=1,
        )
        return {
            "crag_output": crag_output,
            "crag_rdem": rdem,
            "global_error_log": state.global_error_log + [rdem],
            "audit_trail": state.audit_trail + [
                AgentStep(
                    node_name="retriever",
                    status="error",
                    error=rdem,
                    timestamp=datetime.utcnow().isoformat(),
                )
            ],
        }

    step = AgentStep(
        node_name="retriever",
        status="success",
        timestamp=datetime.utcnow().isoformat(),
    )
    return {
        "crag_output": crag_output,
        "crag_rdem": None,
        "audit_trail": state.audit_trail + [step],
    }