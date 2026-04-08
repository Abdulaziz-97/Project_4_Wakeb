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
    temperature=0.1,
    timeout=60,
)


@log_node
def writer_agent(state: WeatherAgentState) -> dict:
    crag = state.crag_output

    prompt = (
        "Convert this weather data into a simple report. Use bullet points only.\n\n"
        "Format:\n"
        "## Weather: [location]\n"
        "- Current: [conditions and temp]\n"
        "- Forecast: [next days]\n"
        "- Sources: [list]\n\n"
        f"Data:\n{crag.answer}"
    )

    tracer = create_tracer("writer")
    response = llm.invoke(
        [HumanMessage(content=prompt)],
        config={"callbacks": [tracer]},
    )

    return {
        "draft_document": response.content,
        "fact_check_result": None,
        "fact_fix_attempts": 0,
        "writer_rdem": None,
        "audit_trail": state.audit_trail + [
            AgentStep(
                node_name="writer",
                status="success",
                timestamp=datetime.utcnow().isoformat(),
            )
        ],
    }
