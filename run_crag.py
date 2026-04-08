\
\
\
\
\
\
   

import sys
import os

sys.path.insert(0, os.path.dirname(__file__))

import dspy
from config.settings import DEEPSEEK_API_KEY, DEEPSEEK_BASE_URL, DEEPSEEK_MODEL
from crag.pipeline import CRAGPipeline


def setup_dspy():
    lm = dspy.LM(
        model=f"openai/{DEEPSEEK_MODEL}",
        api_key=DEEPSEEK_API_KEY,
        api_base=DEEPSEEK_BASE_URL,
        temperature=0.0,
        max_tokens=1000,
    )
    dspy.configure(lm=lm)


def main():
    setup_dspy()

    from agent.logger import init_run
    init_run()

    pipeline = CRAGPipeline()

    queries = (
        [sys.argv[1]]
        if len(sys.argv) > 1
        else [
            "What is the weather in Riyadh today?",
            "What is the weekly forecast for Jeddah?",
        ]
    )

    for query in queries:
        print("=" * 70)
        print(f"QUERY: {query}")
        print("=" * 70)
        result = pipeline.run(query)
        print(f"ACTION:  {result['action']}")
        print(f"SCORES:  {result['scores']}")
        print(f"SOURCES: {result['sources']}")
        print(f"ANSWER:\n{result['answer']}")
        print()


if __name__ == "__main__":
    main()
