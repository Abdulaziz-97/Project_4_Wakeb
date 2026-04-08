"""
Run 5 weather queries quickly to validate the system works.
Tests with SearchLimiter, async RAGAS, and metrics.
"""

import time
import json
import sys
from datetime import datetime
from pathlib import Path

# Setup path
sys.path.insert(0, str(Path(__file__).parent))

import dspy
from config.settings import DEEPSEEK_API_KEY, DEEPSEEK_BASE_URL, DEEPSEEK_MODEL
from agent.graph import build_graph

# Configure DSPy BEFORE graph initialization
dspy.configure(lm=dspy.LM(
    model=f"openai/{DEEPSEEK_MODEL}",
    api_key=DEEPSEEK_API_KEY,
    api_base=DEEPSEEK_BASE_URL,
))

# Build graph
graph = build_graph()

# 5 test queries across different regions
QUERIES = [
    "What is the weather in rio de janeiro?",
    "What is the weather in paris tomorrow?",
    "Weather forecast for tokyo next week",
    "Is it raining in london?",
    "Temperature in dubai uae",
]

def run_5_tests():
    """Run 5 quick tests to validate system."""
    
    print(f"\n{'='*100}")
    print(f"QUICK TEST - 5 WEATHER QUERIES")
    print(f"Start Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*100}\n")
    
    results = {
        "timestamp": datetime.now().isoformat(),
        "queries_run": 0,
        "success": 0,
        "errors": 0,
        "results": [],
        "summary": {
            "total_time": 0,
            "avg_time": 0,
        }
    }
    
    total_start = time.time()
    
    for i, query in enumerate(QUERIES, 1):
        print(f"\n[{i}/5] Query: {query}")
        print(f"{'-'*80}")
        
        query_start = time.time()
        try:
            # Run through graph
            result = graph.invoke(
                {
                    "messages": [{"role": "user", "content": query}],
                    "user_query": query,
                },
                config={"configurable": {"thread_id": f"test_5_{i}"}}
            )
            
            query_time = time.time() - query_start
            
            # Extract results
            crag_output = result.get("crag_output", {})
            action = crag_output.get("action", "unknown") if isinstance(crag_output, dict) else getattr(crag_output, "action", "unknown")
            sources = crag_output.get("sources", []) if isinstance(crag_output, dict) else getattr(crag_output, "sources", [])
            answer = crag_output.get("answer", "") if isinstance(crag_output, dict) else getattr(crag_output, "answer", "")
            
            # Show results
            print(f"✓ Status: SUCCESS")
            print(f"  Time: {query_time:.2f}s")
            print(f"  Action: {action}")
            print(f"  Sources Retrieved: {len(sources)}")
            print(f"  Answer Preview: {answer[:150]}...")
            
            # Check for RAGAS metrics
            ragas = result.get("ragas_metrics_async", {})
            if ragas:
                print(f"  RAGAS Metrics:")
                print(f"    - Faithfulness: {ragas.get('faithfulness', 'N/A'):.3f}")
                print(f"    - Answer Relevance: {ragas.get('answer_relevance', 'N/A'):.3f}")
                print(f"    - Context Precision: {ragas.get('context_precision', 'N/A'):.3f}")
                print(f"    - Overall Score: {ragas.get('overall_rag_score', 'N/A'):.3f}")
            else:
                print(f"  RAGAS: Computing in background...")
            
            results["results"].append({
                "query": query,
                "time": round(query_time, 2),
                "status": "success",
                "action": action,
                "sources_count": len(sources),
                "ragas": ragas if ragas else "pending"
            })
            results["success"] += 1
            results["summary"]["total_time"] += query_time
            
        except Exception as e:
            query_time = time.time() - query_start
            print(f"✗ Status: ERROR")
            print(f"  Time: {query_time:.2f}s")
            print(f"  Error: {str(e)[:200]}")
            
            results["results"].append({
                "query": query,
                "time": round(query_time, 2),
                "status": "error",
                "error": str(e)[:100],
            })
            results["errors"] += 1
    
    total_time = time.time() - total_start
    results["queries_run"] = len(QUERIES)
    results["summary"]["total_time"] = round(total_time, 2)
    results["summary"]["avg_time"] = round(total_time / len(QUERIES), 2)
    
    # Print summary
    print(f"\n{'='*100}")
    print(f"SUMMARY")
    print(f"{'='*100}")
    print(f"Total Queries: {results['queries_run']}")
    print(f"Success: {results['success']}")
    print(f"Errors: {results['errors']}")
    print(f"Total Time: {results['summary']['total_time']:.2f}s")
    print(f"Average Time per Query: {results['summary']['avg_time']:.2f}s")
    print(f"{'='*100}\n")
    
    # Save results
    log_file = Path(__file__).parent / "logs" / f"test_5_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    log_file.parent.mkdir(exist_ok=True, parents=True)
    
    with open(log_file, "w") as f:
        json.dump(results, f, indent=2)
    
    print(f"✓ Results saved to: {log_file}\n")
    
    return results

if __name__ == "__main__":
    run_5_tests()