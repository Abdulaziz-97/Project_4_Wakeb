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
    max_tokens=2000,
)


def _voice_prompt_en(crag_data: str) -> str:
    return (
        "You are a voice assistant. Give a SHORT spoken weather summary.\n"
        "Rules:\n"
        "- 2-4 sentences MAX\n"
        "- No references, no URLs, no sources\n"
        "- No markdown, no bullet points, no headers\n"
        "- Use natural spoken language (e.g. 'It is 35 degrees and sunny')\n"
        "- Include temperature, conditions, and any notable info\n\n"
        f"Data:\n{crag_data}"
    )


def _voice_prompt_ar(crag_data: str) -> str:
    return (
        "انت مساعد صوتي للطقس. اعطني ملخص قصير عن حالة الطقس.\n"
        "القواعد:\n"
        "- جملتين الى اربع جمل بالكثير\n"
        "- لا تحط مراجع ولا روابط ولا مصادر\n"
        "- لا تستخدم ماركداون ولا نقاط ولا عناوين\n"
        "- استخدم اللهجة السعودية العامية (مثلا: 'الجو اليوم حار وصحو، الحرارة 35 درجة')\n"
        "- اذكر درجة الحرارة وحالة الجو واي شي مهم\n"
        "- الرد لازم يكون باللهجة السعودية مو فصحى\n\n"
        f"البيانات:\n{crag_data}"
    )


def _text_prompt(crag_data: str) -> str:
    return (
        "You are a professional meteorologist. Write a detailed, well-structured weather report.\n"
        "Use the data below — include EVERY available metric, do not skip anything.\n\n"
        "FORMAT RULES (strictly follow):\n"
        "- Main title: ## Weather Report: [City, Country]\n"
        "- Section headings: ### [Section Name]\n"
        "- Bullet items: - **Label:** value\n"
        "- Use **bold** for all key numbers and conditions\n"
        "- No URLs, no source citations inside the report body\n\n"
        "REQUIRED SECTIONS (include all that have data):\n\n"
        "### Overview\n"
        "Write 2-3 sentences describing the overall weather picture in plain English.\n\n"
        "### Current Conditions\n"
        "- **Temperature:** high / low (°C)\n"
        "- **Conditions:** [sky description]\n"
        "- **Wind:** speed and direction\n"
        "- **Precipitation:** probability and expected amount (mm)\n"
        "- **Humidity:** percentage (if available)\n"
        "- **UV Index:** value — include sun safety advice if high\n"
        "- **Sunrise / Sunset:** times\n\n"
        "### Multi-Day Forecast\n"
        "For EACH day in the data, write a sub-entry:\n"
        "- **[Day / Date]:** [conditions], High **X°C** / Low **Y°C**, Wind **Z km/h**, Rain **N%**\n\n"
        "### Practical Recommendations\n"
        "- Give 3-4 specific, actionable tips based on the actual conditions "
        "(clothing, activity suitability, sun protection, rain gear, etc.)\n\n"
        f"WEATHER DATA:\n{crag_data}"
    )


@log_node
def writer_agent(state: WeatherAgentState) -> dict:
    crag = state.crag_output
    crag_data = crag.answer if crag and crag.answer else "No weather data available."

    if state.voice_mode and state.voice_language == "ar":
        prompt = _voice_prompt_ar(crag_data)
    elif state.voice_mode:
        prompt = _voice_prompt_en(crag_data)
    else:
        prompt = _text_prompt(crag_data)

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
