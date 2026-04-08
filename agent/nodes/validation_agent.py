"""
Validation Agent — lightweight check on retrieved answers.
Receives {query, answer, date, sources} and decides: VALID or INVALID.

Three check levels:
  1. FRESHNESS gate  — is the ingested answer still fresh enough for the
     time window the user asked about? (no LLM, instant)
  2. FAST PATH       — high-confidence + fresh ingested answer → skip LLM
  3. SLOW PATH       — full LLM judgment (location, data, currency, coverage)
"""
import re
from datetime import datetime

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage

from config.settings import DEEPSEEK_API_KEY, DEEPSEEK_BASE_URL, DEEPSEEK_MODEL
from agent.state import WeatherAgentState, AgentStep
from agent.logger import log_node, create_tracer, get_logger

_judge = ChatOpenAI(
    model=DEEPSEEK_MODEL,
    api_key=DEEPSEEK_API_KEY,
    base_url=DEEPSEEK_BASE_URL,
    temperature=0.0,
    timeout=30,
)

# ── Time-sensitivity detection ──────────────────────────────────────
# Maps query patterns to max allowed age (in hours) for cached answers.
_TIME_RULES = [
    (re.compile(r"\b(today|tonight|right now|current(ly)?)\b", re.I), 6),
    (re.compile(r"\b(tomorrow)\b", re.I), 12),
    (re.compile(r"\b(this week|next \d+ days|5.day|7.day)\b", re.I), 48),
    (re.compile(r"\b(next week)\b", re.I), 72),
    (re.compile(r"\b(last week|last month|yesterday|historical)\b", re.I), 168),
]
_DEFAULT_MAX_AGE_HOURS = 24


def _max_age_for_query(query: str) -> int:
    """Return max allowed age (hours) for a cached answer given this query."""
    for pattern, hours in _TIME_RULES:
        if pattern.search(query):
            return hours
    return _DEFAULT_MAX_AGE_HOURS


@log_node
def validation_agent(state: WeatherAgentState) -> dict:
    """
    Validate the retrieved answer against the query and current date.

    1. Freshness gate — reject stale ingested answers instantly.
    2. Fast path — skip LLM for fresh high-confidence ingested answers.
    3. Slow path — LLM judges location, data, currency, coverage.
    """
    logger = get_logger()

    # ── No answer at all → INVALID ──────────────────────────────
    if not state.crag_output or not state.crag_output.answer:
        logger.info("  [validator] No answer to validate — routing to consultant")
        return _build_result(state, is_valid=False, status="skip")

    crag = state.crag_output

    # ── FRESHNESS GATE (ingested answers only) ──────────────────
    if crag.is_ingested and crag.ingested_at:
        max_hours = _max_age_for_query(state.user_query)
        try:
            ts = datetime.fromisoformat(crag.ingested_at)
            age = datetime.utcnow() - ts
            age_hours = age.total_seconds() / 3600

            if age_hours > max_hours:
                logger.info(
                    f"  [validator] STALE — ingested {age_hours:.1f}h ago, "
                    f"max allowed {max_hours}h for this query. Routing to consultant."
                )
                return _build_result(state, is_valid=False, status="stale")

            # Fresh + high confidence → FAST PATH (no LLM call)
            if crag.is_high_confidence:
                logger.info(
                    f"  [validator] FAST PATH — fresh ingested answer "
                    f"({age_hours:.1f}h old < {max_hours}h, "
                    f"score={crag.max_score:.2f}). Skipping LLM call."
                )
                return _build_result(state, is_valid=True, status="fast_pass")
        except (ValueError, TypeError):
            logger.warning("  [validator] Could not parse ingested_at — falling through to LLM")

    # ── FAST PATH: very high retrieval score on non-ingested docs ─
    if crag.max_score > 0.85 and not crag.is_ingested:
        logger.info(
            f"  [validator] FAST PATH — very high retrieval score "
            f"({crag.max_score:.2f} > 0.85). Skipping LLM call."
        )
        return _build_result(state, is_valid=True, status="fast_pass_score")

    # ── SLOW PATH: LLM validation ──────────────────────────────
    today = datetime.utcnow().strftime("%A, %B %d, %Y")
    sources_str = (
        "\n".join(crag.sources[:10]) if crag.sources else "No sources"
    )

    prompt = (
        "You are a weather answer validator. Your ONLY job is to check if "
        "an answer is valid. You have NO tools — just judge the data.\n\n"
        f"TODAY'S DATE: {today}\n"
        f"USER QUERY: {state.user_query}\n\n"
        f"SOURCES:\n{sources_str}\n\n"
        f"ANSWER:\n{crag.answer[:2000]}\n\n"
        "IMPORTANT: Ignore any disclaimers, hedging, or statements like "
        "'recommend checking other sources'. Judge ONLY the actual data.\n\n"
        "Check these 4 things:\n"
        "1. LOCATION — Does the answer match the location in the query?\n"
        "2. DATA — Does it contain concrete weather data (temperatures, conditions)?\n"
        "3. CURRENT — Is this data fresh enough for what the user asked?\n"
        f"   - 'today'/'now' = must be from today ({today})\n"
        f"   - 'this week' = must be from this week\n"
        f"   - 'last week' = historical, older data is fine\n"
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

    try:
        tracer = create_tracer("validation_agent")
        response = _judge.invoke(
            [HumanMessage(content=prompt)],
            config={"callbacks": [tracer]},
        )
    except Exception as e:
        logger.error(f"  [validator] LLM call failed: {e}")
        return _build_result(state, is_valid=False, status="error")

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

    return _build_result(
        state,
        is_valid=is_valid,
        status="valid" if is_valid else "invalid",
    )


def _build_result(state: WeatherAgentState, is_valid: bool, status: str) -> dict:
    """Build the standard return dict for the validation agent."""
    return {
        "validation_passed": is_valid,
        "audit_trail": state.audit_trail + [
            AgentStep(
                node_name="validation_agent",
                status=status,
                timestamp=datetime.utcnow().isoformat(),
            )
        ],
    }
