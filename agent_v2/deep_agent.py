"""
Deep Agent v2 — Weather Documentation Pipeline.

Uses LangChain's Deep Agents SDK (create_deep_agent) to orchestrate
the same CRAG + Writer + Fact-Check + Formatter workflow, but with
automatic planning, context management, subagent delegation, and
file system tools built in.

The Deep Agent replaces the manual LangGraph StateGraph with a single
autonomous agent that plans, delegates, and self-corrects.
"""

from langchain_openai import ChatOpenAI
from deepagents import create_deep_agent
from langgraph.checkpoint.memory import MemorySaver

from config.settings import DEEPSEEK_API_KEY, DEEPSEEK_BASE_URL, DEEPSEEK_MODEL
from agent_v2.tools import crag_retrieve, celsius_to_fahrenheit, web_search

SYSTEM_PROMPT = """\
You are a Weather Documentation Agent. Your job is to take a user's weather \
query and produce a polished, fact-checked LaTeX weather report.

## Your Workflow

Follow these steps IN ORDER for every query:

### Step 1: Retrieve Data
- Call `crag_retrieve` with the user's query.
- Parse the JSON result. Check the "action" field:
  - "correct" (score > 0.7): data is from local docs — verify it is CURRENT \
(not historical/outdated). If outdated, proceed to Step 1b.
  - "ambiguous": mixed local + web — usable but verify.
  - "incorrect" or "web_only": data came from live web search — usable.
- If the CRAG answer contains no concrete weather data, or is clearly \
outdated (references a past year), go to Step 1b.

### Step 1b: Web Search Fallback
- Call `web_search` with targeted queries:
  1. "{location} current weather temperature today"
  2. "{location} 7 day weather forecast"
  3. "{location} weather warnings" (if relevant)
- Combine results with any CRAG data that was partially useful.

### Step 2: Write the Report
- Use ALL retrieved data to write a structured weather report.
- For EVERY temperature in Celsius, call `celsius_to_fahrenheit` to get \
the exact Fahrenheit value. NEVER guess or estimate conversions.
- Include:
  - Location and date
  - Current conditions (temperature, sky, humidity, wind, pressure)
  - Temperature summary table with both C and F
  - Multi-day forecast if available
  - Weather warnings/advisories
  - Numbered source citations [1], [2], etc.
- If data confidence is low, note: "Data sourced from live web search."

### Step 3: Fact-Check
- Review your draft against the source data.
- Re-verify EVERY temperature conversion by calling `celsius_to_fahrenheit` \
again on each Celsius value mentioned.
- If any value is wrong (difference > 0.1F), correct it immediately.
- Rounding differences of 0.1F or less are acceptable.

### Step 4: Format to LaTeX
- Convert the final verified report to valid, compilable LaTeX.
- Requirements:
  - \\documentclass{article} with \\usepackage{booktabs}, \\usepackage{geometry}, \
\\usepackage{hyperref}
  - \\geometry{margin=2.5cm}
  - \\section{} for major sections
  - \\textbf{} for important values
  - Temperature data in \\begin{tabular} with \\toprule/\\midrule/\\bottomrule
  - \\section{References} with \\begin{enumerate} listing ALL sources as \\url{}
  - Data confidence note in \\textit{}
- Output the LaTeX starting from \\documentclass. No markdown fences.

### Step 5: Save and Return
- Use `write_file` to save the LaTeX to "/output/report.tex".
- Return the full LaTeX document to the user.

## Rules
- ALWAYS call `crag_retrieve` first — never skip retrieval.
- ALWAYS use `celsius_to_fahrenheit` — never estimate conversions.
- NEVER refuse to write a report. Work with whatever data you have.
- Be factual. Include every number from the source data.
- Include source citations for every claim.
"""

FACT_CHECKER_SUBAGENT = {
    "name": "fact-checker",
    "description": (
        "Verifies temperature conversions and factual claims in a weather "
        "report. Give it the draft report and the source data. It will "
        "re-check every temperature using celsius_to_fahrenheit and flag "
        "any discrepancies."
    ),
    "system_prompt": (
        "You are a weather fact-checker. You receive a draft weather report "
        "and source data. Your job:\n"
        "1. Identify every temperature in the draft.\n"
        "2. Call celsius_to_fahrenheit for each Celsius value.\n"
        "3. Compare your result against the draft's Fahrenheit value.\n"
        "4. Differences of +/-0.1F are acceptable rounding.\n"
        "5. Report ALL discrepancies and whether the draft is factual.\n"
        "Return a clear verdict: PASS or FAIL with specific issues."
    ),
    "tools": [celsius_to_fahrenheit],
}


def build_deep_agent(checkpointer=None):
    """Build and return the Deep Agent v2 pipeline."""
    model = ChatOpenAI(
        model=DEEPSEEK_MODEL,
        api_key=DEEPSEEK_API_KEY,
        base_url=DEEPSEEK_BASE_URL,
        temperature=0.1,
        timeout=120,
        max_retries=3,
    )

    agent = create_deep_agent(
        name="weather-doc-agent",
        model=model,
        tools=[crag_retrieve, celsius_to_fahrenheit, web_search],
        system_prompt=SYSTEM_PROMPT,
        subagents=[FACT_CHECKER_SUBAGENT],
        checkpointer=checkpointer or MemorySaver(),
    )

    return agent
