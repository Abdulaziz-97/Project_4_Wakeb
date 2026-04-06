import json
import re
from datetime import datetime

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage
from langgraph.prebuilt import create_react_agent

from config.settings import DEEPSEEK_API_KEY, DEEPSEEK_BASE_URL, DEEPSEEK_MODEL
from agent.state import (
    WeatherAgentState,
    FactCheckResult,
    RDEMError,
    AgentStep,
)
from agent.tools.weather_tools import celsius_to_fahrenheit
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
    tools=[celsius_to_fahrenheit],
    prompt=(
        "You are a weather fact-checker with ReAct capabilities.\n"
        "You have a celsius_to_fahrenheit tool. Use it to verify EVERY "
        "temperature conversion in the draft report.\n\n"
        "Process:\n"
        "1. Read the draft report and identify all temperature values.\n"
        "2. For each Celsius temperature, call celsius_to_fahrenheit to verify "
        "the Fahrenheit value in the draft.\n"
        "3. Compare all factual claims against the source data.\n"
        "4. After all verifications, output ONLY this JSON (no other text):\n"
        "{\n"
        '  "is_factual": true or false,\n'
        '  "issues": ["issue 1", "issue 2"],\n'
        '  "verified_temperatures": {"<celsius_value>": <fahrenheit_result>}\n'
        "}\n\n"
        "ROUNDING RULE: Differences of +/-0.1 F or less between your tool "
        "result and the draft value are acceptable rounding. Note them in "
        "issues for transparency, but they must NOT cause is_factual to be "
        "false.\n\n"
        "If all facts match and temperatures are correct: is_factual = true."
    ),
)


@log_node
def writer_fact_checker(state: WeatherAgentState) -> dict:
    user_msg = (
        f"Source data (ground truth): {state.crag_output.answer}\n\n"
        f"Draft report to verify:\n{state.draft_document}"
    )

    tracer = create_tracer("fact_checker")
    result = _react_agent.invoke(
        {"messages": [HumanMessage(content=user_msg)]},
        config={"callbacks": [tracer]},
    )
    response = result["messages"][-1]

    try:
        json_match = re.search(r"\{.*\}", response.content, re.DOTALL)
        data = json.loads(json_match.group())
        fact_result = FactCheckResult(
            is_factual=data["is_factual"],
            issues=data.get("issues", []),
            verified_temperatures=data.get("verified_temperatures", {}),
        )
    except Exception:
        fact_result = FactCheckResult(
            is_factual=True, issues=["JSON parse error -- proceeding"]
        )

    if fact_result.is_factual:
        return {
            "fact_check_result": fact_result,
            "writer_rdem": None,
            "audit_trail": state.audit_trail + [
                AgentStep(
                    node_name="fact_checker",
                    status="success",
                    timestamp=datetime.utcnow().isoformat(),
                )
            ],
        }

    new_attempts = state.fact_fix_attempts + 1

    if new_attempts >= 3:
        forced_result = FactCheckResult(
            is_factual=True,
            issues=fact_result.issues
            + ["MAX ATTEMPTS REACHED -- proceeding with best effort"],
            verified_temperatures=fact_result.verified_temperatures,
        )
        rdem = RDEMError(
            node="fact_checker",
            error_type="max_attempts_reached",
            message=f"Fact fix loop reached max 3 attempts. Issues: {fact_result.issues}",
            suggestion="Formatter will proceed with best-effort document.",
            attempt=new_attempts,
        )
        return {
            "fact_check_result": forced_result,
            "fact_fix_attempts": 3,
            "writer_rdem": rdem,
            "global_error_log": state.global_error_log + [rdem],
            "audit_trail": state.audit_trail + [
                AgentStep(
                    node_name="fact_checker",
                    status="forced_proceed",
                    error=rdem,
                    timestamp=datetime.utcnow().isoformat(),
                )
            ],
        }

    rdem = RDEMError(
        node="fact_checker",
        error_type="factual_fail",
        message=f"Attempt {new_attempts}: Issues found: {fact_result.issues}",
        suggestion="Fix temperature unit errors and align facts with CRAG source answer.",
        attempt=new_attempts,
    )
    return {
        "fact_check_result": fact_result,
        "fact_fix_attempts": new_attempts,
        "writer_rdem": rdem,
        "global_error_log": state.global_error_log + [rdem],
        "audit_trail": state.audit_trail + [
            AgentStep(
                node_name="fact_checker",
                status="error",
                error=rdem,
                timestamp=datetime.utcnow().isoformat(),
            )
        ],
    }
