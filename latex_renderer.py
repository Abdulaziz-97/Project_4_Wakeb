\
\
\
   
import re


def _strip_latex_cmd(text: str, cmd: str) -> str:
                                                                                 
    return re.sub(r"\\" + cmd + r"\{([^}]*)\}", r"\1", text)


def _extract_between(text: str, start_marker: str, end_marker: str) -> str:
    try:
        s = text.index(start_marker) + len(start_marker)
        e = text.index(end_marker, s)
        return text[s:e].strip()
    except ValueError:
        return ""


def _parse_draft(draft: str) -> str:
                                                                           
    lines = draft.splitlines()
    html_parts = []
    in_list = False

    for raw in lines:
        line = raw.strip()
        if not line:
            if in_list:
                html_parts.append("</ul>")
                in_list = False
            continue

        if line.startswith("## "):
            if in_list:
                html_parts.append("</ul>")
                in_list = False
            title = line[3:].strip()
            html_parts.append(f'<h2 class="wr-title">{title}</h2>')

        elif line.startswith("### "):
            if in_list:
                html_parts.append("</ul>")
                in_list = False
            title = line[4:].strip()
            html_parts.append(f'<h3 class="wr-sub">{title}</h3>')

        elif line.startswith("- ") or line.startswith("* "):
            if not in_list:
                html_parts.append('<ul class="wr-list">')
                in_list = True
            item = line[2:].strip()
            item = re.sub(r"\*\*(.+?)\*\*", r"<strong>\1</strong>", item)
            item = re.sub(r"\*(.+?)\*", r"<em>\1</em>", item)
            html_parts.append(f"  <li>{item}</li>")

        elif re.match(r"^\d+\.\s", line):
            if not in_list:
                html_parts.append('<ul class="wr-list">')
                in_list = True
            item = re.sub(r"^\d+\.\s+", "", line)
            html_parts.append(f"  <li>{item}</li>")

        else:
            if in_list:
                html_parts.append("</ul>")
                in_list = False
            para = re.sub(r"\*\*(.+?)\*\*", r"<strong>\1</strong>", line)
            para = re.sub(r"\*(.+?)\*", r"<em>\1</em>", para)
            html_parts.append(f'<p class="wr-para">{para}</p>')

    if in_list:
        html_parts.append("</ul>")

    return "\n".join(html_parts)


def _parse_references(enumerate_body: str) -> list[list[str]]:
                                                                                  
    raw = []
    for m in re.finditer(r"\\item\s*(?:\[.*?\])?\s*\{([^}]+)\}", enumerate_body):
        src = m.group(1).strip()
        src = src.replace(r"\{", "{").replace(r"\}", "}").replace(r"\&", "&") \
                 .replace(r"\#", "#").replace(r"\_", "_").replace(r"\%", "%")
        if src and src.lower() not in ("no sources available.", ""):
            raw.append(src)
    if not raw:
        for m in re.finditer(r"\\item\s+(.+)", enumerate_body):
            src = m.group(1).strip().lstrip("[]").strip()
            if src:
                raw.append(src)

    _SUB_FIELDS = ("location:", "forecast", "data:", "retrieved:", "url:", "page", "section:")
    groups: list[list[str]] = []
    current: list[str] = []
    for line in raw:
        is_sub = any(line.lower().startswith(f) for f in _SUB_FIELDS)
        if not is_sub and current:
            groups.append(current)
            current = [line]
        elif not is_sub:
            current = [line]
        else:
            current.append(line)
    if current:
        groups.append(current)
    return groups if groups else [[s] for s in raw]


def _parse_confidence(textit_body: str) -> tuple[str, str, str]:
                                                                                   
    action, score, retrieval = "", "", ""
    m = re.search(r"Data confidence:\s*(\w+)\s*\(score:\s*([\d.]+)\)", textit_body)
    if m:
        action = m.group(1)
        score = m.group(2)
    m2 = re.search(r"Retrieval action:\s*(\w+)", textit_body)
    if m2:
        retrieval = m2.group(1)
    return action, score, retrieval


_BADGE_COLORS = {
    "correct": ("#064e3b", "#10b981"),
    "ambiguous": ("#78350f", "#f59e0b"),
    "incorrect": ("#7f1d1d", "#f87171"),
    "unknown": ("#1e293b", "#64748b"),
}


def render(latex: str) -> str:
\
\
\
       
    doc_body = _extract_between(latex, r"\begin{document}", r"\end{document}")
    if not doc_body:
        doc_body = latex

    refs_body = _extract_between(doc_body, r"\begin{enumerate}", r"\end{enumerate}")
    textit_body = ""
    m = re.search(r"\\textit\{([^}]+)\}", doc_body)
    if m:
        textit_body = m.group(1)

    draft = doc_body
    for marker in [r"\section{References}", r"\begin{enumerate}"]:
        if marker in draft:
            draft = draft[:draft.index(marker)]
    draft = draft.strip()

    draft_html = _parse_draft(draft)
    refs = _parse_references(refs_body)
    action, score, retrieval = _parse_confidence(textit_body)

    badge_bg, badge_fg = _BADGE_COLORS.get(action.lower(), _BADGE_COLORS["unknown"])

    refs_html = ""
    if refs:
        source_blocks = []
        for i, group in enumerate(refs, 1):
            header = group[0] if group else ""
            sub_lines = group[1:] if len(group) > 1 else []

            sub_html_parts = []
            for line in sub_lines:
                if line.lower().startswith("url:"):
                    url = line[4:].strip()
                    sub_html_parts.append(
                        f'<div class="wr-ref-sub">'
                        f'<span class="wr-ref-key">URL</span> '
                        f'<a href="{url}" target="_blank" class="wr-ref-url">{url}</a>'
                        f'</div>'
                    )
                else:
                    key, _, val = line.partition(":")
                    if val:
                        sub_html_parts.append(
                            f'<div class="wr-ref-sub">'
                            f'<span class="wr-ref-key">{key.strip()}</span> {val.strip()}'
                            f'</div>'
                        )
                    else:
                        sub_html_parts.append(
                            f'<div class="wr-ref-sub">{line}</div>'
                        )

                                                                               
            if " | http" in header:
                title, _, url = header.partition(" | ")
                header_html = (
                    f'<div class="wr-ref-title">'
                    f'<span class="wr-ref-num">{i}</span> {title}'
                    f'</div>'
                    f'<div class="wr-ref-sub">'
                    f'<span class="wr-ref-key">URL</span> '
                    f'<a href="{url}" target="_blank" class="wr-ref-url">{url}</a>'
                    f'</div>'
                )
            else:
                header_html = (
                    f'<div class="wr-ref-title">'
                    f'<span class="wr-ref-num">{i}</span> {header}'
                    f'</div>'
                )

            source_blocks.append(
                f'<div class="wr-ref-block">{header_html}{"".join(sub_html_parts)}</div>'
            )

        refs_html = f"""
<div class="wr-refs">
  <div class="wr-refs-title">Sources</div>
  {"".join(source_blocks)}
</div>"""

    confidence_html = ""
    if action:
        confidence_html = f"""
<div class="wr-meta">
  <span class="wr-badge" style="background:{badge_bg};color:{badge_fg};">
    {action.upper()}
  </span>
  <span class="wr-score">Score: {score}</span>
  <span class="wr-retrieval">Retrieval: {retrieval}</span>
</div>"""

    css = """
<style>
.wr-card {
    background: linear-gradient(160deg, #0f172a 0%, #1e293b 100%);
    border: 1px solid #334155; border-radius: 14px;
    padding: 1.8rem 2.2rem; margin: 0.6rem 0;
    color: #e2e8f0; font-family: 'Inter', sans-serif;
}
.wr-title {
    font-size: 1.35rem; font-weight: 700; color: #f8fafc;
    border-bottom: 2px solid #6366f1; padding-bottom: 0.6rem;
    margin: 0 0 1.2rem 0; letter-spacing: -0.3px;
}
.wr-sub {
    font-size: 0.78rem; font-weight: 700; color: #6366f1;
    margin: 1.4rem 0 0.5rem 0; text-transform: uppercase;
    letter-spacing: 1.2px; display: flex; align-items: center; gap: 0.5rem;
}
.wr-sub::after {
    content: ""; flex: 1; height: 1px; background: #1e293b;
}
.wr-list {
    list-style: none; padding: 0; margin: 0 0 0.4rem 0;
}
.wr-list li {
    padding: 0.4rem 0 0.4rem 1.2rem; position: relative;
    color: #cbd5e1; font-size: 0.91rem; border-bottom: 1px solid #1e293b11;
    line-height: 1.5;
}
.wr-list li::before {
    content: "▸"; position: absolute; left: 0; color: #6366f1;
}
.wr-list li strong { color: #f1f5f9; }
.wr-para {
    color: #94a3b8; font-size: 0.93rem; margin: 0.3rem 0 0.8rem 0;
    line-height: 1.65; font-style: italic;
    border-left: 3px solid #6366f130; padding-left: 0.8rem;
}
.wr-refs {
    margin-top: 1.5rem; border-top: 1px solid #1e293b; padding-top: 1rem;
}
.wr-refs-title {
    font-size: 0.65rem; text-transform: uppercase; letter-spacing: 1px;
    color: #64748b; margin-bottom: 0.7rem; font-weight: 700;
}
.wr-ref-block {
    background: #0f172a; border: 1px solid #1e293b; border-radius: 8px;
    padding: 0.65rem 0.9rem; margin-bottom: 0.5rem;
}
.wr-ref-title {
    font-size: 0.85rem; color: #94a3b8; font-weight: 600;
    margin-bottom: 0.3rem; display: flex; align-items: center; gap: 0.4rem;
}
.wr-ref-num {
    background: #1e293b; color: #64748b; font-size: 0.65rem; font-weight: 700;
    width: 18px; height: 18px; border-radius: 50%; display: inline-flex;
    align-items: center; justify-content: center; flex-shrink: 0;
}
.wr-ref-sub {
    font-size: 0.75rem; color: #475569; padding: 0.1rem 0 0.1rem 1.4rem;
    display: flex; gap: 0.4rem; align-items: baseline;
}
.wr-ref-key {
    color: #334155; font-size: 0.65rem; font-weight: 700; text-transform: uppercase;
    letter-spacing: 0.5px; min-width: 70px; flex-shrink: 0;
}
.wr-ref-url {
    color: #6366f1; text-decoration: none; word-break: break-all; font-size: 0.72rem;
}
.wr-ref-url:hover { text-decoration: underline; }
.wr-meta {
    margin-top: 1rem; display: flex; align-items: center;
    gap: 0.8rem; flex-wrap: wrap;
}
.wr-badge {
    font-size: 0.65rem; font-weight: 700; letter-spacing: 1px;
    padding: 0.2rem 0.6rem; border-radius: 4px;
    text-transform: uppercase;
}
.wr-score, .wr-retrieval {
    font-size: 0.75rem; color: #475569;
}
</style>
"""

    html = f"""{css}
<div class="wr-card">
  {draft_html}
  {refs_html}
  {confidence_html}
</div>"""

    return html
