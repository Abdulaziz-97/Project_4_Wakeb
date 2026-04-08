from datetime import datetime

from agent.state import WeatherAgentState, AgentStep
from agent.logger import log_node


@log_node
def formatter(state: WeatherAgentState) -> dict:
    """Simple formatter: wrap markdown in minimal LaTeX without LLM call."""
    crag = state.crag_output
    
    # Build references
    ref_lines = []
    for i, src in enumerate(crag.sources, 1):
        ref_lines.append(f"\\item {src}")
    refs = "\n".join(ref_lines) if ref_lines else "No sources available."
    
    # Simple LaTeX wrapper (no LLM call = instant)
    latex = (
        "\\documentclass{article}\n"
        "\\usepackage{booktabs}\n"
        "\\usepackage{geometry}\n"
        "\\usepackage{hyperref}\n"
        "\\geometry{margin=2.5cm}\n"
        "\\begin{document}\n\n"
        f"{state.draft_document}\n\n"
        "\\section{References}\n"
        "\\begin{enumerate}\n"
        f"{refs}\n"
        "\\end{enumerate}\n\n"
        f"\\textit{{Data confidence: {crag.action} (score: {crag.max_score:.2f}). "
        f"Retrieval action: {crag.action}.}}\n\n"
        "\\end{document}\n"
    )

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
