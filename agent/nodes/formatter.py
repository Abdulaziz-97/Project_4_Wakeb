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


def _build_references_block(sources: list[str]) -> str:
    if not sources:
        return "No sources available."
    lines = []
    for i, src in enumerate(sources, 1):
        lines.append(f"  [{i}] {src}")
    return "\n".join(lines)


@log_node
def formatter(state: WeatherAgentState) -> dict:
    crag = state.crag_output
    ref_block = _build_references_block(crag.sources)

    prompt = (
        "You are a LaTeX formatting agent.\n"
        "Convert the following weather report to valid, compilable LaTeX.\n\n"
        "Requirements:\n"
        "- Use \\documentclass{article} with \\usepackage{booktabs}, "
        "\\usepackage{geometry}, \\usepackage{hyperref}\n"
        "- \\geometry{margin=2.5cm}\n"
        "- Use \\section{} for major sections\n"
        "- Use \\textbf{} for important values (temperatures, wind speeds)\n"
        "- Put temperature data in a \\begin{tabular} table with "
        "\\toprule/\\midrule/\\bottomrule\n"
        "- Add a MANDATORY \\section{References} at the end. This section must "
        "list EVERY source below as a numbered item using \\begin{enumerate}. "
        "If a source is a URL, wrap it with \\url{}. If it contains metadata "
        "like [pdf] or [web], keep that label. Every reference the CRAG system "
        "retrieved MUST appear here:\n"
        f"{ref_block}\n\n"
        "- Below the references, add an italic note:\n"
        f"  \\textit{{Data confidence: {crag.action} "
        f"(score: {crag.max_score:.2f}). "
        f"Retrieval action: {crag.action}.}}\n\n"
        "- If there are inline citations like [1], [2] in the report text, "
        "make sure they correspond to the numbered references above.\n"
        "- Output ONLY the LaTeX code, starting from \\documentclass. "
        "Do NOT wrap it in markdown code fences.\n\n"
        f"Report to convert:\n{state.draft_document}"
    )

    tracer = create_tracer("formatter")
    response = llm.invoke(
        [HumanMessage(content=prompt)],
        config={"callbacks": [tracer]},
    )

    latex = response.content
    if latex.startswith("```"):
        lines = latex.splitlines()
        if lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        latex = "\n".join(lines)

    return {
        "final_latex_document": latex,
        "current_step": "done",
        "audit_trail": state.audit_trail + [
            AgentStep(
                node_name="formatter",
                status="success",
                timestamp=datetime.utcnow().isoformat(),
            )
        ],
    }
