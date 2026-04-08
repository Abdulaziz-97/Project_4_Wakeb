from datetime import datetime
from crag.pipeline import CRAGPipeline
from agent.state import WeatherAgentState, CRAGOutput, RDEMError, AgentStep
from agent.logger import log_node, get_logger

# FAST pipeline: evaluate_metrics=False (no blocking RAGAS)
_pipeline = CRAGPipeline(evaluate_metrics=False)


@log_node
def retriever_agent(state: WeatherAgentState) -> dict:
    logger = get_logger()
    normalized_query = " ".join(state.user_query.split()).lower()

    logger.info(f"  [retriever] Calling CRAGPipeline.run('{normalized_query}')")
    try:
        # Run FAST CRAG without metrics
        # evaluate_ragas_async=True will start RAGAS in background thread
        result = _pipeline.run(normalized_query, evaluate_ragas_async=True)
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

    logger.debug(f"  [retriever] CRAG result:")
    logger.debug(f"    action:  {result['action']}")
    logger.debug(f"    sources: {result['sources']}")
    logger.debug(f"    answer:  {result['answer'][:300]}...")

    crag_output = CRAGOutput(
        answer=result["answer"],
        sources=result["sources"],
        action=result["action"],
        scores=result["scores"],
        max_score=max(result["scores"]) if result["scores"] else 0.0,
        is_high_confidence=(result["action"] == "correct"),
        is_ingested=result.get("is_ingested", False),
        ingested_at=result.get("ingested_at", ""),
    )

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