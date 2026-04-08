"""Quick test for async RAGAS evaluation."""
import time
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Configure DSPy BEFORE importing pipeline
import dspy
from config.settings import DEEPSEEK_API_KEY, DEEPSEEK_BASE_URL, DEEPSEEK_MODEL

# Configure DSPy with DeepSeek (same as project)
dspy.configure(lm=dspy.LM(
    model=f"openai/{DEEPSEEK_MODEL}",
    api_key=DEEPSEEK_API_KEY,
    api_base=DEEPSEEK_BASE_URL,
))

from crag.pipeline import CRAGPipeline

# Initialize pipeline
pipeline = CRAGPipeline(evaluate_metrics=False)

# Test query
query = "What is the weather in brazil this week?"

print(f"\n{'='*80}")
print(f"TEST: RAGAS Async Evaluation")
print(f"{'='*80}\n")

# Run CRAG with async RAGAS
print(f"[TEST] Running CRAG with async RAGAS for: '{query}'")
start = time.time()

result = pipeline.run(query, evaluate_ragas_async=True)

crag_time = time.time() - start
print(f"[TEST] CRAG completed in {crag_time:.1f}s (should be ~11-15s, NOT 30-35s)")
print(f"[TEST] CRAG action: {result['action']}")
print(f"[TEST] CRAG sources: {len(result['sources'])} sources")
print(f"[TEST] CRAG answer: {result['answer'][:150]}...\n")

# RAGAS should be running in background now
print(f"[TEST] Waiting for async RAGAS to complete (~10-15s)...")

# Poll for RAGAS results
ragas_found = False
for i in range(30):  # Wait up to 30 seconds
    time.sleep(0.5)
    if "ragas_metrics_async" in result:
        ragas_found = True
        ragas_time = time.time() - start
        print(f"\n[TEST] ✓ RAGAS completed in {ragas_time:.1f}s total (background)")
        print(f"[TEST] RAGAS Metrics:")
        for key, value in result["ragas_metrics_async"].items():
            if value is not None:
                print(f"        {key}: {value:.3f}")
        break

if not ragas_found:
    print(f"[TEST] ⚠ RAGAS still running (check logs after ~15s)")

print(f"\n[TEST] Total execution: {time.time() - start:.1f}s")
print(f"{'='*80}\n")