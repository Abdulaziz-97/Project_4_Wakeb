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
    prompt = (
        "You are a strict relevance judge. Decide if the ANSWER below "
        "actually provides useful, up-to-date information for the USER QUERY.\n\n"
        "IMPORTANT: Focus ONLY on the content of the answer itself. "
        "Ignore how the data was retrieved. If the answer contains concrete, "
        "current weather data (temperatures, conditions, forecasts) that "
        "addresses the user's query, it IS relevant.\n\n"
        "Reject the answer ONLY if:\n"
        "- The data is clearly outdated / historical and the user asks for "
        "current info (e.g. data from a past year)\n"
        "- The answer explicitly says it cannot help, or only suggests "
        "checking other sources without providing data\n"
        "- The answer contains NO concrete weather data at all\n"
        "- The answer is about a completely different location\n\n"
        f"USER QUERY: {query}\n\n"
        f"Retrieval method: {readable_action}\n\n"
        f"ANSWER:\n{answer}\n\n"
        "Reply with EXACTLY one word: RELEVANT or NOT_RELEVANT"
    )
    tracer = create_tracer("retriever_quality_gate")
    response = _judge.invoke(
        [HumanMessage(content=prompt)],
        config={"callbacks": [tracer]},
    )
    verdict = "NOT_RELEVANT" not in response.content.upper()
    logger.info(
        f"  [retriever] Quality gate verdict: "
        f"{'RELEVANT' if verdict else 'NOT_RELEVANT'} "
        f"(action={action})"
    )
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
            error_type="relevance_redirect",
            message=(
                f"CRAG returned action='{crag_output.action}' with score "
                f"{crag_output.max_score:.2f}. Quality gate detected stale/insufficient "
                f"data — redirecting to live web search for fresh results."
            ),
            suggestion="Route to weather_consultant for fresh web search.",
            attempt=1,
        )
        return {
            "crag_output": crag_output,
            "crag_rdem": rdem,
            "audit_trail": state.audit_trail + [
                AgentStep(
                    node_name="retriever",
                    status="redirected",
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
