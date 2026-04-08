from datetime import datetime

from agent.state import WeatherAgentState, AgentStep
from agent.logger import log_node


@log_node
def formatter(state: WeatherAgentState) -> dict:
    crag = state.crag_output
    draft = state.draft_document or "No content available."

    if state.voice_mode:
        output = draft
    else:
        ref_lines = []
        if crag and crag.sources:
            for src in crag.sources:
                clean = str(src).strip()
                if clean:
                    escaped = clean.replace("{", "\\{").replace("}", "\\}").replace("&", "\\&").replace("#", "\\#").replace("_", "\\_").replace("%", "\\%")
                    ref_lines.append(f"\\item {{{escaped}}}")
        refs = "\n".join(ref_lines) if ref_lines else "\\item {No sources available.}"

        action = crag.action if crag else "unknown"
        max_score = crag.max_score if crag else 0.0

        output = (
            "\\documentclass{article}\n"
            "\\usepackage{booktabs}\n"
            "\\usepackage{geometry}\n"
            "\\usepackage{hyperref}\n"
            "\\geometry{margin=2.5cm}\n"
            "\\begin{document}\n\n"
            f"{draft}\n\n"
            "\\section{References}\n"
            "\\begin{enumerate}\n"
            f"{refs}\n"
            "\\end{enumerate}\n\n"
            f"\\textit{{Data confidence: {action} (score: {max_score:.2f}). "
            f"Retrieval action: {action}.}}\n\n"
            "\\end{document}\n"
        )

    return {
        "final_latex_document": output,
        "current_step": "done",
        "audit_trail": state.audit_trail + [
            AgentStep(
                node_name="formatter",
                status="success",
                timestamp=datetime.utcnow().isoformat(),
            )
        ],
    }
