"""
run_agent_v2.py — Entry point for the Deep Agent v2 weather pipeline.

Uses LangChain's deepagents SDK instead of the manual LangGraph StateGraph.
The Deep Agent handles planning, tool calling, fact-checking, and formatting
autonomously via a single agent loop with subagent delegation.

Usage:
    python run_agent_v2.py
"""

import sys
import os

sys.path.insert(0, os.path.dirname(__file__))

import dspy
from datetime import datetime

from config.settings import DEEPSEEK_API_KEY, DEEPSEEK_BASE_URL, DEEPSEEK_MODEL


def setup_dspy():
    """Configure DSPy BEFORE any CRAG imports."""
    lm = dspy.LM(
        model=f"openai/{DEEPSEEK_MODEL}",
        api_key=DEEPSEEK_API_KEY,
        api_base=DEEPSEEK_BASE_URL,
        temperature=0.0,
        max_tokens=1000,
    )
    dspy.configure(lm=lm)


def print_result(result, label="RESULT"):
    """Pretty-print the agent's final output."""
    print("\n" + "=" * 70)
    print(f"  {label}")
    print("=" * 70)

    messages = result.get("messages", [])
    if not messages:
        print("  (No messages returned)")
        return

    final_msg = messages[-1]
    content = getattr(final_msg, "content", str(final_msg))

    if "\\documentclass" in content:
        doc_start = content.find("\\documentclass")
        preamble = content[:doc_start].strip()
        latex = content[doc_start:].strip()

        if preamble:
            print("\n  Agent Commentary:")
            print("  " + "-" * 40)
            for line in preamble.split("\n")[:10]:
                print(f"    {line}")

        print("\n  LaTeX Document Preview (first 60 lines):")
        print("  " + "-" * 40)
        lines = latex.split("\n")
        for line in lines[:60]:
            print(f"    {line}")
        if len(lines) > 60:
            print(f"    ... ({len(lines) - 60} more lines)")

        print(f"\n  Total LaTeX length: {len(latex)} characters")
    else:
        preview = content[:2000]
        print(f"\n{preview}")
        if len(content) > 2000:
            print(f"\n  ... ({len(content) - 2000} more characters)")

    # Files written by the agent
    files = result.get("files", {})
    if files:
        print("\n  Files created by agent:")
        for path in files:
            print(f"    - {path}")

    print("\n" + "=" * 70)


def main():
    setup_dspy()

    from agent_v2.deep_agent import build_deep_agent

    agent = build_deep_agent()

    config = {"configurable": {"thread_id": "v2_session_1"}}

    print("=" * 70)
    print("  WEATHER AGENT v2 (Deep Agent)")
    print(f"  Started: {datetime.utcnow().isoformat()}")
    print("=" * 70)

    # --- Query 1 ---
    query1 = "What is the weather in Berlin this week?"
    print(f"\n  Query: {query1}\n")

    result = agent.invoke(
        {"messages": [{"role": "user", "content": query1}]},
        config=config,
    )
    print_result(result, label=f"QUERY 1: {query1}")

    # Save LaTeX output if present
    messages = result.get("messages", [])
    if messages:
        final_content = getattr(messages[-1], "content", "")
        if "\\documentclass" in final_content:
            doc_start = final_content.find("\\documentclass")
            latex = final_content[doc_start:]
            # Sanitize unicode for LaTeX
            latex = (
                latex
                .replace("\u00b0", r"\textdegree{}")
                .replace("\u2014", "---")
                .replace("\u2013", "--")
                .replace("\u00b1", r"\textpm{}")
            )
            out_path = "output_v2_berlin.tex"
            with open(out_path, "w", encoding="utf-8") as f:
                f.write(latex)
            print(f"\n  Saved LaTeX to: {out_path}")

    # --- Query 2 (follow-up) ---
    query2 = "What about Hamburg?"
    print(f"\n  Follow-up Query: {query2}\n")

    result2 = agent.invoke(
        {"messages": [{"role": "user", "content": query2}]},
        config=config,
    )
    print_result(result2, label=f"QUERY 2 (follow-up): {query2}")


if __name__ == "__main__":
    main()
