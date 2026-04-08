from datetime import datetime

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage

from config.settings import DEEPSEEK_API_KEY, DEEPSEEK_BASE_URL, DEEPSEEK_MODEL
from agent.state import WeatherAgentState, AgentStep
from agent.logger import log_node, create_tracer

llm = ChatOpenAI(
    model=DEEPSEEK_MODEL,
    api_key=DEEPSEEK_API_KEY,
    base_url=DEEPSEEK_BASE_URL,
    temperature=0.0,
    timeout=60,
)


@log_node
def fix_fact_agent(state: WeatherAgentState) -> dict:
    fc = state.fact_check_result
    
    issues_detail = ""
    if fc and fc.issues:
        issues_detail = "\nIssues to fix:\n" + "\n".join(f"  - {i}" for i in fc.issues)

    prompt = (
        f"Fix the following issues in the draft report. Do not change any correct information.\n"
        f"{issues_detail}\n\n"
        f"Ground truth (CRAG original answer):\n{state.crag_output.answer}\n\n"
        f"Current draft to fix:\n{state.draft_document}\n\n"
        f"Return ONLY the corrected report text. No explanations."
    )

    tracer = create_tracer("fix_fact")
    response = llm.invoke(
        [HumanMessage(content=prompt)],
        config={"callbacks": [tracer]},
    )

    return {
        "draft_document": response.content,
        "fact_check_result": None,
        "audit_trail": state.audit_trail + [
            AgentStep(
                node_name="fix_fact",
                status="success",
                timestamp=datetime.utcnow().isoformat(),
            )
        ],
    }
