"""
Run 100 weather queries to benchmark retrieval quality.
Collects timing, success rates, and RAGAS metrics.
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
from agent.state import WeatherAgentState
from agent.logger import init_run, get_logger

# Configure DSPy
dspy.configure(lm=dspy.LM(
    model=f"openai/{DEEPSEEK_MODEL}",
    api_key=DEEPSEEK_API_KEY,
    api_base=DEEPSEEK_BASE_URL,
))

# Build the graph
graph = build_graph() 

# 100 diverse weather queries
QUERIES = [
    # Brazil (10)
    "What is the weather in brazil this week?",
    "What is the weather in rio de janeiro?",
    "Weather forecast for sao paulo next week",
    "Is it raining in brasilia today?",
    "Current weather conditions in amazon region",
    "Temperature in salvador brazil",
    "Weather in curitiba this weekend",
    "Forecast for belo horizonte next 5 days",
    "How is the weather in fortaleza?",
    "Weather conditions in manaus brazil",
    
    # Europe (10)
    "What is the weather in paris tomorrow?",
    "London weather forecast next week",
    "Is it snowing in berlin?",
    "Weather in madrid this week",
    "Current conditions in rome italy",
    "Amsterdam weather forecast",
    "Temperature in zurich switzerland",
    "Weather in barcelona spain",
    "Forecast for stockholm sweden",
    "Current weather in dublin ireland",
    
    # Asia (10)
    "What is the weather in tokyo?",
    "Weather forecast for hong kong",
    "Is it raining in bangkok thailand?",
    "New delhi weather next week",
    "Current conditions in singapore",
    "Weather in mumbai india",
    "Forecast for shanghai china",
    "Temperature in seoul south korea",
    "Weather in jakarta indonesia",
    "Current weather in dubai uae",
    
    # Americas (10)
    "Weather forecast for new york",
    "What is the weather in los angeles?",
    "Toronto canada weather",
    "Forecast for mexico city",
    "Weather in vancouver canada",
    "Current conditions in miami florida",
    "Weather forecast for chicago",
    "Temperature in denver colorado",
    "Weather in san francisco",
    "Forecast for buenos aires argentina",
    
    # Africa (10)
    "Weather in cairo egypt",
    "What is the weather in johannesburg?",
    "Lagos nigeria weather forecast",
    "Current conditions in cape town",
    "Weather in nairobi kenya",
    "Temperature in marrakech morocco",
    "Forecast for accra ghana",
    "Weather in dakar senegal",
    "Current weather in khartoum sudan",
    "Weather forecast for lagos nigeria",
    
    # Oceania (10)
    "Weather in sydney australia",
    "What is the weather in melbourne?",
    "Auckland new zealand forecast",
    "Current conditions in brisbane",
    "Weather in perth australia",
    "Temperature in fiji islands",
    "Forecast for auckland new zealand",
    "Weather in honolulu hawaii",
    "Current weather in samoa",
    "Weather forecast for wellington nz",
    
    # Middle East (10)
    "Weather in istanbul turkey",
    "What is the weather in tehran iran?",
    "Baghdad iraq weather forecast",
    "Current conditions in riyadh saudi arabia",
    "Weather in amman jordan",
    "Temperature in beirut lebanon",
    "Forecast for doha qatar",
    "Weather in muscat oman",
    "Current weather in kuwait city",
    "Weather forecast for jeddah saudi arabia",
    
    # More specific queries (24)
    "Will it rain in london this week?",
    "Is it too hot in mumbai to go outside?",
    "How cold is it in moscow right now?",
    "Best weather this week in europe",
    "Worst weather forecast for next month",
    "Temperature swing in new york",
    "Humidity levels in tropical cities",
    "Storm forecast for florida",
    "Heat wave in australia",
    "Snowfall prediction in canada",
    "Seasonal weather pattern in india",
    "Monsoon forecast for southeast asia",
    "Winter weather in scandinavia",
    "Spring weather in japan",
    "Summer forecast in mediterranean",
    "Autumn weather in united states",
    "Weather comparison tokyo vs bangkok",
    "Air quality and weather in beijing",
    "Cyclone forecast for pacific",
    "Drought conditions in africa",
    "Flood risk in south asia",
    "Tornado season in us midwest",
    "Weather for outdoor activities",
    "Best time to visit x for weather",
]

def run_benchmark():
    """Run 100 weather queries and collect metrics."""
    
    print(f"\n{'='*100}")
    print(f"WEATHER PIPELINE BENCHMARK - 100 TEST SCENARIOS")
    print(f"Start Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*100}\n")
    
    results = {
        "metadata": {
            "total_queries": len(QUERIES),
            "start_time": datetime.now().isoformat(),
            "model": DEEPSEEK_MODEL,
        },
        "results": [],
        "summary": {
            "total_time": 0,
            "avg_time": 0,
            "min_time": float('inf'),
            "max_time": 0,
            "success_count": 0,
            "error_count": 0,
            "ragas": {
                "avg_faithfulness": 0,
                "avg_answer_relevance": 0,
                "avg_context_precision": 0,
                "avg_overall_score": 0,
            }
        }
    }
    
    total_start = time.time()
    faithfulness_scores = []
    relevance_scores = []
    precision_scores = []
    overall_scores = []
    
    # Run each query
    for i, query in enumerate(QUERIES, 1):
        print(f"[{i:3d}/100] Running: {query[:60]}...", end=" ", flush=True)
        
        query_start = time.time()
        try:
            # Run through graph WITH config for checkpointer
            result = graph.invoke(
                {
                    "messages": [{"role": "user", "content": query}],
                    "user_query": query,
                },
                config={"configurable": {"thread_id": f"benchmark_{i}"}}
            )
            query_time = time.time() - query_start
            
            # Extract RAGAS metrics if available (will be in background)
            ragas_metrics = result.get("ragas_metrics_async", {})
            
            # Safely extract crag_output
            crag_output = result.get("crag_output")
            action = "unknown"
            sources_count = 0
            
            if crag_output:
                action = crag_output.get("action", "unknown") if isinstance(crag_output, dict) else getattr(crag_output, "action", "unknown")
                sources = crag_output.get("sources", []) if isinstance(crag_output, dict) else getattr(crag_output, "sources", [])
                sources_count = len(sources) if sources else 0
            
            result_entry = {
                "query": query,
                "time": round(query_time, 2),
                "status": "success",
                "action": action,
                "sources": sources_count,
            }
            
            if ragas_metrics:
                result_entry["ragas"] = {
                    "faithfulness": round(ragas_metrics.get("faithfulness", 0), 3),
                    "answer_relevance": round(ragas_metrics.get("answer_relevance", 0), 3),
                    "context_precision": round(ragas_metrics.get("context_precision", 0), 3),
                    "overall_score": round(ragas_metrics.get("overall_rag_score", 0), 3),
                }
                faithfulness_scores.append(ragas_metrics.get("faithfulness", 0))
                relevance_scores.append(ragas_metrics.get("answer_relevance", 0))
                precision_scores.append(ragas_metrics.get("context_precision", 0))
                overall_scores.append(ragas_metrics.get("overall_rag_score", 0))
            
            results["results"].append(result_entry)
            results["summary"]["success_count"] += 1
            results["summary"]["total_time"] += query_time
            results["summary"]["min_time"] = min(results["summary"]["min_time"], query_time)
            results["summary"]["max_time"] = max(results["summary"]["max_time"], query_time)
            
            print(f"✓ {query_time:.1f}s")
            
        except Exception as e:
            query_time = time.time() - query_start
            error_msg = str(e)[:50]
            print(f"✗ ERROR: {error_msg}")
            
            results["results"].append({
                "query": query,
                "time": round(query_time, 2),
                "status": "error",
                "error": str(e)[:100],
            })
            results["summary"]["error_count"] += 1
    
    # Calculate summary statistics
    total_time = time.time() - total_start
    results["summary"]["total_time"] = round(total_time, 2)
    results["summary"]["end_time"] = datetime.now().isoformat()
    
    if results["summary"]["success_count"] > 0:
        results["summary"]["avg_time"] = round(
            results["summary"]["total_time"] / results["summary"]["success_count"], 2
        )
    
    if faithfulness_scores:
        results["summary"]["ragas"]["avg_faithfulness"] = round(sum(faithfulness_scores) / len(faithfulness_scores), 3)
        results["summary"]["ragas"]["avg_answer_relevance"] = round(sum(relevance_scores) / len(relevance_scores), 3)
        results["summary"]["ragas"]["avg_context_precision"] = round(sum(precision_scores) / len(precision_scores), 3)
        results["summary"]["ragas"]["avg_overall_score"] = round(sum(overall_scores) / len(overall_scores), 3)
    
    # Print summary
    print(f"\n{'='*100}")
    print(f"BENCHMARK RESULTS SUMMARY")
    print(f"{'='*100}\n")
    
    total_queries = len(QUERIES)
    success_count = results["summary"]["success_count"]
    error_count = results["summary"]["error_count"]
    
    print(f"Total Queries:        {total_queries}")
    print(f"Successful:           {success_count}")
    print(f"Failed:               {error_count}")
    
    if total_queries > 0:
        success_rate = (success_count / total_queries) * 100
        print(f"Success Rate:         {success_rate:.1f}%\n")
    else:
        print(f"Success Rate:         0%\n")
    
    total_time_sec = results["summary"]["total_time"]
    avg_time = results["summary"]["avg_time"]
    min_time = results["summary"]["min_time"]
    max_time = results["summary"]["max_time"]
    
    print(f"Total Time:           {total_time_sec:.1f}s ({total_time_sec/60:.1f} min)")
    print(f"Average per Query:    {avg_time:.2f}s")
    
    if min_time != float('inf'):
        print(f"Min Time:             {min_time:.2f}s")
    else:
        print(f"Min Time:             N/A")
    
    if max_time > 0:
        print(f"Max Time:             {max_time:.2f}s\n")
    else:
        print(f"Max Time:             N/A\n")
    
    print(f"RAGAS Quality Metrics (Average):")
    print(f"  Faithfulness:       {results['summary']['ragas']['avg_faithfulness']:.3f}")
    print(f"  Answer Relevance:   {results['summary']['ragas']['avg_answer_relevance']:.3f}")
    print(f"  Context Precision:  {results['summary']['ragas']['avg_context_precision']:.3f}")
    print(f"  Overall RAG Score:  {results['summary']['ragas']['avg_overall_score']:.3f}\n")
    
    # Save results
    output_file = Path("logs") / f"benchmark_100_tests_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    output_file.parent.mkdir(exist_ok=True)
    
    with open(output_file, "w") as f:
        json.dump(results, f, indent=2)
    
    print(f"Results saved to: {output_file}")
    print(f"{'='*100}\n")
    
    return results

if __name__ == "__main__":
    results = run_benchmark()