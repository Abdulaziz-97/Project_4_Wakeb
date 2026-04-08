"""
Run 10 diverse weather scenarios through the full pipeline.

- Full logging to logs/run_<timestamp>.log
- All LaTeX documents saved to output_all_scenarios.tex
- Console shows live progress per scenario
"""

import sys
import os
import time

sys.path.insert(0, os.path.dirname(__file__))

import dspy
from config.settings import DEEPSEEK_API_KEY, DEEPSEEK_BASE_URL, DEEPSEEK_MODEL

lm = dspy.LM(
    model=f"openai/{DEEPSEEK_MODEL}",
    api_key=DEEPSEEK_API_KEY,
    api_base=DEEPSEEK_BASE_URL,
    temperature=0.0,
    max_tokens=1000,
)
dspy.configure(lm=lm)

from agent.graph import build_graph
from agent.state import WeatherAgentState
from agent.logger import init_run, get_logger

SCENARIOS = [
    "What is the weather in arar last week?",
    # "Current temperature and forecast for Tokyo, Japan",
    # "Is it going to rain in London tomorrow?",
    # "Weather conditions in Dubai right now",
    # "What is the 5-day forecast for New York City?",
    # "Current weather in Riyadh, Saudi Arabia",
    # "Will there be snow in Moscow this week?",
    # "Weather and humidity in Mumbai today",
    # "What is the wind speed in Chicago right now?",
    # "Weekly weather forecast for Sydney, Australia",
]


def _get(obj, key, default=None):
    if isinstance(obj, dict):
        return obj.get(key, default)
    return getattr(obj, key, default)


def main():
    run_id = init_run()
    logger = get_logger()

    graph = build_graph()
    all_latex = []
    summary_rows = []

    ragas_samples = []  # collected after each scenario for post-run evaluation

    for idx, query in enumerate(SCENARIOS, 1):
        thread_id = f"scenario_{run_id}_{idx}"
        config = {"configurable": {"thread_id": thread_id}}

        logger.info("")
        logger.info("#" * 70)
        logger.info(f"#  SCENARIO {idx}/10: {query}")
        logger.info("#" * 70)

        state = WeatherAgentState(user_query=query).model_dump()

        t0 = time.time()
        try:
            result = graph.invoke(state, config=config)
        except Exception as exc:
            elapsed = time.time() - t0
            logger.error(f"SCENARIO {idx} FAILED after {elapsed:.1f}s: {exc}")
            summary_rows.append({
                "idx": idx, "query": query, "status": "FAILED",
                "action": "N/A", "sources": 0, "steps": 0,
                "errors": 1, "elapsed": elapsed,
            })
            continue

        elapsed = time.time() - t0

        latex = result.get("final_latex_document", "")
        crag = result.get("crag_output")
        action = _get(crag, "action", "N/A") if crag else "N/A"
        sources = _get(crag, "sources", []) if crag else []
        answer = _get(crag, "answer", "") if crag else ""
        trail = result.get("audit_trail", [])
        errors = result.get("global_error_log", [])
        fc = result.get("fact_check_result")

        # Log summary
        logger.info(f"SCENARIO {idx} COMPLETED in {elapsed:.1f}s")
        logger.info(f"  CRAG action: {action}")
        logger.info(f"  Sources: {len(sources)}")
        logger.info(f"  Audit steps: {len(trail)}")
        logger.info(f"  Errors: {len(errors)}")
        logger.info(f"  Factual: {_get(fc, 'is_factual', '?') if fc else '?'}")
        logger.info(f"  LaTeX length: {len(latex)} chars")

        # Collect pipeline outputs for post-run RAGAS evaluation
        if answer:
            contexts = [s.replace("[web] ", "").replace("[doc] ", "") for s in sources[:3]]
            contexts = [c for c in contexts if c.strip()]
            if not contexts:
                # No source URLs — ingested/correct path: the answer IS the document
                contexts = [answer]
            ragas_samples.append({"query": query, "answer": answer, "contexts": contexts})

        # Collect
        all_latex.append(
            f"% {'='*66}\n"
            f"% SCENARIO {idx}: {query}\n"
            f"% Action: {action} | Sources: {len(sources)} | "
            f"Steps: {len(trail)} | Errors: {len(errors)}\n"
            f"% {'='*66}\n\n"
            f"{latex}\n\n"
            f"\\clearpage\n\n"
        )

        summary_rows.append({
            "idx": idx, "query": query, "status": "OK",
            "action": action, "sources": len(sources),
            "steps": len(trail), "errors": len(errors),
            "elapsed": elapsed,
        })

    # --- Save all LaTeX ---
    tex_path = "output_all_scenarios.tex"
    combined = "All 10 scenario outputs -- auto-generated\n"
    combined += f"Run ID: {run_id}\n\n"
    for block in all_latex:
        combined += block

    combined = (
        combined
        .replace("\u00b0", r"\textdegree{}")
        .replace("\u2014", "---")
        .replace("\u2013", "--")
        .replace("\u00b1", r"\textpm{}")
        .replace("\ufffd", "?")
    )

    with open(tex_path, "w", encoding="utf-8") as f:
        f.write(combined)

    logger.info("")
    logger.info("=" * 70)
    logger.info("  FINAL SUMMARY")
    logger.info("=" * 70)
    logger.info(f"{'#':>3}  {'Status':>6}  {'Action':<10} {'Src':>3} {'Stp':>3} "
                f"{'Err':>3} {'Time':>6}  Query")
    logger.info("-" * 70)
    for row in summary_rows:
        logger.info(
            f"{row['idx']:>3}  {row['status']:>6}  {row['action']:<10} "
            f"{row['sources']:>3} {row['steps']:>3} {row['errors']:>3} "
            f"{row['elapsed']:>5.0f}s  {row['query'][:40]}"
        )
    logger.info("-" * 70)

    total_time = sum(r["elapsed"] for r in summary_rows)
    ok_count = sum(1 for r in summary_rows if r["status"] == "OK")
    logger.info(f"Total: {ok_count}/10 succeeded in {total_time:.0f}s")
    logger.info(f"LaTeX saved to: {tex_path}")
    logger.info(f"Full log saved to: logs/run_{run_id}.log")

    # ── RAGAS Evaluation (post-run, no pipeline involvement) ─────
    if ragas_samples:
        import numpy as np
        from crag.metrics import RAGASEvaluator, MetricsLogger

        logger.info("")
        logger.info("=" * 70)
        logger.info("  RAGAS EVALUATION  (pipeline already finished)")
        logger.info("=" * 70)

        evaluator = RAGASEvaluator()
        metrics_logger = MetricsLogger()
        all_metrics = []

        for sample in ragas_samples:
            m = evaluator.evaluate(
                query=sample["query"],
                answer=sample["answer"],
                contexts=sample["contexts"],
            )
            metrics_logger.log_metrics(sample["query"], m)
            all_metrics.append(m)
            logger.info(
                f"  [{sample['query'][:40]:<40}]  "
                f"faith={m['faithfulness']:.2f}  "
                f"rel={m['answer_relevancy']:.2f}  "
                f"prec={m['context_precision']:.2f}  "
                f"score={m['overall_rag_score']:.2f}"
            )

        faithfulness = [m["faithfulness"] for m in all_metrics]
        relevancy    = [m["answer_relevancy"] for m in all_metrics]
        precision    = [m["context_precision"] for m in all_metrics]
        overall      = [m["overall_rag_score"] for m in all_metrics]

        logger.info("-" * 70)
        logger.info(f"  {'Metric':<22} {'Mean':>6}  {'Min':>6}  {'Max':>6}")
        logger.info(f"  {'-'*44}")
        for name, scores in [
            ("Faithfulness",      faithfulness),
            ("Answer Relevancy",  relevancy),
            ("Context Precision", precision),
            ("Overall RAG Score", overall),
        ]:
            logger.info(
                f"  {name:<22} {np.mean(scores):>6.3f}  "
                f"{np.min(scores):>6.3f}  {np.max(scores):>6.3f}"
            )
        logger.info("=" * 70)
    else:
        logger.warning("  No RAGAS samples collected (all scenarios failed).")


if __name__ == "__main__":
    main()
