# Weather Documentation Pipeline -- Technical Reference

## Table of Contents

1. [System Overview](#1-system-overview)
2. [Architecture](#2-architecture)
3. [Project Structure](#3-project-structure)
4. [Layer 1: Data Ingestion](#4-layer-1-data-ingestion)
5. [Layer 2: Corrective RAG (CRAG)](#5-layer-2-corrective-rag-crag)
6. [Layer 3: Multi-Agent Orchestration](#6-layer-3-multi-agent-orchestration)
7. [State Model](#7-state-model)
8. [Graph Wiring and Execution Flow](#8-graph-wiring-and-execution-flow)
9. [Agent Nodes -- Deep Dive](#9-agent-nodes----deep-dive)
10. [Tools](#10-tools)
11. [Logging and Observability](#11-logging-and-observability)
12. [Configuration](#12-configuration)
13. [Entry Points and Scripts](#13-entry-points-and-scripts)
14. [Error Handling (RDEM)](#14-error-handling-rdem)
15. [Data Flow: End-to-End Walkthrough](#15-data-flow-end-to-end-walkthrough)
16. [Key Design Decisions](#16-key-design-decisions)

---

## 1. System Overview

This system takes a natural-language weather query (e.g., "What is the weather in Berlin this week?") and produces a **fully formatted LaTeX document** with verified data, source citations, and temperature conversions. It does this through three layers:

| Layer | Technology | Purpose |
|-------|-----------|---------|
| **Data Ingestion** | Loaders + ChromaDB | Download documents (PDF, web, JSON, CSV), chunk them, embed with `all-MiniLM-L6-v2`, store in ChromaDB |
| **Corrective RAG** | DSPy + Tavily | Retrieve relevant chunks, evaluate relevance, fall back to web search if needed, generate a cited answer |
| **Agent Orchestration** | LangGraph + LangChain | Route the CRAG output through Writer, Fact Checker, Fix Fact (self-healing loop), and Formatter agents |

The LLM backend for **everything** is **DeepSeek** (`deepseek-chat`) accessed through OpenAI-compatible APIs. DSPy uses it for CRAG; LangChain `ChatOpenAI` uses it for all agent nodes.

---

## 2. Architecture

The system has two distinct execution modes:

**Mode 1: Standalone CRAG** (`run_crag.py`) -- query goes directly to the CRAG pipeline, which returns an answer with citations. No agents involved.

**Mode 2: Full Agent Pipeline** (`run_agent.py`) -- the CRAG pipeline is wrapped inside a LangGraph multi-agent workflow. The agents draft a report, fact-check it, fix errors, and format it to LaTeX.

### High-Level Data Flow (Mode 2)

```
User Query
    |
    v
[Orchestrator] ---------> [Retriever]
    ^                         |
    |                    CRAGPipeline.run()
    |                         |
    |                   +-----+------+
    |                   |            |
    |               Success    Quality Gate
    |               (RELEVANT)   (NOT_RELEVANT)
    |                   |            |
    |                   |     [Weather Consultant]
    |                   |     (ReAct + web_search)
    |                   |            |
    |                   +-----+------+
    |                         |
    |                    crag_output
    |                         |
[Orchestrator] ---------> [Writer Agent]
    ^                    (ReAct + celsius_to_fahrenheit)
    |                         |
    |                    draft_document
    |                         |
    |                  [Fact Checker]
    |                  (ReAct + celsius_to_fahrenheit)
    |                         |
    |                  +------+------+
    |                  |             |
    |              is_factual    NOT factual
    |                  |             |
    |                  |       [Fix Fact Agent] --+
    |                  |       (ReAct + c2f)      |
    |                  |             |            |
    |                  |        (loops back, max 3)
    |                  |             |
    |                  +------+------+
    |                         |
[Orchestrator] ---------> [Formatter]
    ^                    (LLM -> LaTeX)
    |                         |
    |                  final_latex_document
    |                         |
[Orchestrator] ---------> END
```

---

## 3. Project Structure

```
Project_4_Wakeb/
|-- .env                        # API keys (DEEPSEEK_API_KEY, TAVILY_API_KEY)
|-- requirements.txt            # Python dependencies
|-- ingest.py                   # Download & index documents into ChromaDB
|-- run_crag.py                 # Standalone CRAG test script
|-- run_agent.py                # Full agent pipeline entry point
|-- run_10_scenarios.py         # Batch-run 10 queries, save all LaTeX
|-- run_latex_demo.py           # Single-query demo with verbose output
|
|-- config/
|   +-- settings.py             # All configuration constants
|
|-- loaders/                    # Document loaders (one per format)
|   |-- pdf_loader.py           # PDF -> [{text, metadata}]
|   |-- web_loader.py           # HTML -> [{text, metadata}]
|   |-- markdown_loader.py      # Markdown -> [{text, metadata}]
|   +-- file_loader.py          # JSON/CSV download utilities
|
|-- chunking/
|   +-- chunker.py              # Character-level sliding window chunker
|
|-- vectorstore/
|   +-- chroma_store.py         # ChromaDB wrapper (add, query, count, reset)
|
|-- crag/                       # Corrective RAG (DSPy-based)
|   |-- signatures.py           # DSPy Signature definitions
|   |-- modules.py              # DSPy Module wrappers
|   +-- pipeline.py             # CRAGPipeline class (the main RAG engine)
|
|-- agent/                      # LangGraph multi-agent layer
|   |-- state.py                # Pydantic v2 state models
|   |-- graph.py                # StateGraph assembly and compilation
|   |-- logger.py               # Logging, @log_node decorator, AgentTracer
|   |-- nodes/
|   |   |-- orchestrator.py     # Deterministic router
|   |   |-- retriever.py        # Wraps CRAGPipeline + quality gate
|   |   |-- weather_consultant.py # ReAct fallback with web_search tool
|   |   |-- writer.py           # ReAct report writer with c2f tool
|   |   |-- fact_checker.py     # ReAct fact checker with c2f tool
|   |   |-- fix_fact.py         # ReAct fact corrector with c2f tool
|   |   +-- formatter.py        # LLM-based LaTeX converter
|   +-- tools/
|       +-- weather_tools.py    # celsius_to_fahrenheit + web_search (cached)
|
|-- data/
|   |-- berlin_forecast.json    # Sample downloaded data
|   +-- chroma_db/              # Persisted ChromaDB (SQLite + HNSW shards)
|
|-- logs/                       # Run logs (one per execution)
+-- output_all_scenarios.tex    # Aggregated LaTeX from 10-scenario runs
```

---

## 4. Layer 1: Data Ingestion

**Entry point:** `python ingest.py`

### 4.1 Loaders

Each loader returns a list of `{text: str, metadata: dict}` dicts.

| Loader | File | Input | How It Works |
|--------|------|-------|-------------|
| `load_pdf(source)` | `loaders/pdf_loader.py` | URL or local path | Downloads PDF if URL, uses `pypdf.PdfReader` to extract text per page. Metadata includes `source`, `source_type: "pdf"`, `page` number. |
| `load_web_page(url)` | `loaders/web_loader.py` | URL | Fetches HTML with `requests`, parses with `BeautifulSoup`, removes `<script>`, `<style>`, `<nav>`, `<footer>`, `<header>` tags. Returns cleaned text with `source_type: "web"`. |
| `load_markdown(source)` | `loaders/markdown_loader.py` | URL or local path | Splits by `## ` headers into sections. Each section becomes a separate document with `source_type: "markdown"` and `section` index. |
| `load_json_url(url)` | `loaders/file_loader.py` | URL | Downloads JSON, saves to `data/berlin_forecast.json`. Returns parsed dict (not indexed into ChromaDB). |
| `load_csv_url(url)` | `loaders/file_loader.py` | URL | Downloads CSV, strips `#`-comment lines, returns `pd.DataFrame`. |

### 4.2 Chunker

**File:** `chunking/chunker.py`

Character-level sliding window:
- **Chunk size:** 512 characters (configurable via `config/settings.py`)
- **Overlap:** 64 characters
- Preserves original metadata on each chunk, adds `chunk_index`
- Skips empty/whitespace-only chunks

### 4.3 Vector Store

**File:** `vectorstore/chroma_store.py`

Wraps ChromaDB's `PersistentClient`:

| Method | What It Does |
|--------|-------------|
| `__init__(persist_dir, collection_name)` | Opens (or creates) a ChromaDB collection with cosine distance metric. Loads `SentenceTransformer("all-MiniLM-L6-v2")` for embedding. |
| `add_documents(chunks)` | Encodes texts with the embedder, assigns auto-incrementing IDs (`doc_0`, `doc_1`, ...), stores in ChromaDB. |
| `query(query_text, k=3)` | Encodes the query, retrieves top-k nearest documents. Returns `[{text, metadata, distance}]`. |
| `count()` | Number of documents in the collection. |
| `reset()` | Deletes and recreates the collection. |

**Default collection:** `weather_docs` (for RAG documents)
**Cache collection:** `web_search_cache` (for cached Tavily results)

### 4.4 Ingestion Flow

```
ingest.py
  |
  |-- load_pdf("https://...noaa...pdf")
  |     -> chunk_documents(docs)
  |     -> store.add_documents(chunks)
  |
  |-- load_web_page("https://open-meteo.com/en/docs")
  |     -> chunk_documents(docs)
  |     -> store.add_documents(chunks)
  |
  |-- load_json_url("https://api.open-meteo.com/...")
  |     -> saved to data/berlin_forecast.json (not indexed)
  |
  +-- load_csv_url("https://api.open-meteo.com/...&format=csv")
        -> saved to data/berlin_hourly.csv (not indexed)
```

---

## 5. Layer 2: Corrective RAG (CRAG)

**Files:** `crag/signatures.py`, `crag/modules.py`, `crag/pipeline.py`

CRAG is the retrieval engine. It uses **DSPy** to define LLM-powered modules that evaluate, refine, rewrite, and generate.

### 5.1 DSPy Signatures

Defined in `crag/signatures.py`. Each is a typed contract for an LLM call:

| Signature | Inputs | Output | Purpose |
|-----------|--------|--------|---------|
| `RetrievalEvaluator` | `query`, `document` | `relevance_score: float` | Score how relevant a document is (0.0 -- 1.0) |
| `KnowledgeRefiner` | `document` | `key_points: str` | Extract key points as bullet points |
| `QueryRewriter` | `query` | `rewritten_query: str` | Rewrite query for web search |
| `ResponseGenerator` | `query`, `references` | `answer: str` | Generate a cited answer using numbered references |

### 5.2 DSPy Modules

Defined in `crag/modules.py`. Each wraps a signature with `dspy.Predict`:

| Module | Signature Used | Returns |
|--------|---------------|---------|
| `Evaluator` | `RetrievalEvaluator` | `float` (clamped 0.0--1.0) |
| `Refiner` | `KnowledgeRefiner` | `str` (key points) |
| `Rewriter` | `QueryRewriter` | `str` (rewritten query) |
| `Generator` | `ResponseGenerator` | `str` (cited answer) |

### 5.3 CRAGPipeline

**File:** `crag/pipeline.py`

The central retrieval class. Its `run(query)` method returns:

```python
{
    "answer": str,      # LLM-generated cited answer
    "sources": list[str],  # Formatted source strings
    "action": str,      # "correct" | "ambiguous" | "incorrect" | "web_only"
    "scores": list[float]  # Per-document relevance scores
}
```

#### Routing Logic

```
query
  |
  v
ChromaDB.query(query, k=3)
  |
  +-- No documents? --> _web_search_path() --> action="web_only"
  |
  v
Evaluator(query, doc) for each doc --> scores[]
  |
  v
max(scores)
  |
  +-- > 0.7  --> _correct_path()   --> action="correct"
  |             Uses best local doc only.
  |             Generator(query, [best_doc])
  |
  +-- < 0.3  --> _incorrect_path() --> action="incorrect"
  |             Discards local docs entirely.
  |             Rewriter(query) -> Tavily search
  |             Generator(query, [web_results])
  |
  +-- 0.3-0.7 -> _ambiguous_path() --> action="ambiguous"
                  Keeps best doc + adds web search.
                  Refiner(best_doc) + Rewriter(query) -> Tavily
                  Generator(query, [refined_local + web_results])
```

#### Web Search

Internal method `_do_web_search(query)`:
1. `Rewriter` rewrites the query for better web search results
2. `TavilyClient.search(rewritten_query, max_results=3)` fetches results
3. Returns `(contents: list[str], sources: list[str])`

**Critical:** DSPy must be configured (`setup_dspy()`) **before** `CRAGPipeline` is instantiated, because DSPy modules bind to the configured LM at creation time.

---

## 6. Layer 3: Multi-Agent Orchestration

**Framework:** LangGraph (StateGraph)
**LLM:** DeepSeek via LangChain `ChatOpenAI`
**Agent type:** ReAct (via `langgraph.prebuilt.create_react_agent`)

### 6.1 Why ReAct?

Four of the seven nodes use `create_react_agent` -- a built-in LangGraph utility that implements the Thought-Action-Observation loop. The agent:

1. **Thinks** about what to do (LLM generates reasoning)
2. **Acts** by calling a tool (e.g., `celsius_to_fahrenheit`)
3. **Observes** the tool result
4. Repeats until it has enough information, then produces a final answer

This is superior to simple reflex agents because the LLM can decide **when** and **how many times** to use tools based on the data it sees.

### 6.2 Node Summary

| Node | Type | Tools | Purpose |
|------|------|-------|---------|
| `orchestration_agent` | Deterministic router | None | Reads state, computes next step, emits static message |
| `retriever_agent` | CRAG wrapper + LLM quality gate | None | Calls `CRAGPipeline.run()`, validates relevance |
| `weather_consultant` | ReAct agent | `web_search` | Fallback when CRAG data is insufficient |
| `writer_agent` | ReAct agent | `celsius_to_fahrenheit` | Drafts structured weather report |
| `writer_fact_checker` | ReAct agent | `celsius_to_fahrenheit` | Verifies temperature conversions and facts |
| `fix_fact_agent` | ReAct agent | `celsius_to_fahrenheit` | Corrects flagged issues in the draft |
| `formatter` | Direct LLM call | None | Converts markdown report to LaTeX |

---

## 7. State Model

**File:** `agent/state.py`

All state is defined as **Pydantic v2 BaseModel** classes. LangGraph validates state against `WeatherAgentState` before every node.

### 7.1 WeatherAgentState (Top-Level)

```python
class WeatherAgentState(BaseModel):
    # Conversation memory (accumulates via add_messages reducer)
    messages: Annotated[list, add_messages] = Field(default_factory=list)
    user_query: str = Field(default="")

    # Step 1: Retrieval
    crag_output: Optional[CRAGOutput] = None
    crag_rdem: Optional[RDEMError] = None

    # Step 2: Writing
    draft_document: Optional[str] = None
    fact_check_result: Optional[FactCheckResult] = None
    fact_fix_attempts: int = Field(default=0, ge=0, le=3)  # Pydantic hard cap
    writer_rdem: Optional[RDEMError] = None

    # Step 3: Formatting
    final_latex_document: Optional[str] = None

    # Orchestration
    current_step: str = Field(default="retrieve")

    # Observability
    audit_trail: list[AgentStep] = Field(default_factory=list)
    global_error_log: list[RDEMError] = Field(default_factory=list)
```

**Key design choices:**
- `messages` uses `Annotated[list, add_messages]` -- LangGraph's message accumulator. New messages are **appended**, not replaced.
- `fact_fix_attempts` has `le=3` -- Pydantic itself enforces the hard cap on fact-fix loops at the schema level.
- Every field has a default so nodes can return partial dicts (only changed fields).
- `model_config = ConfigDict(arbitrary_types_allowed=True)` is required for LangChain message objects.

### 7.2 Supporting Models

| Model | Fields | Purpose |
|-------|--------|---------|
| `CRAGOutput` | `answer`, `sources`, `action`, `scores`, `max_score`, `is_high_confidence` | Mirror of `CRAGPipeline.run()` output. Has a `@model_validator` that auto-computes `max_score = max(scores)`. |
| `FactCheckResult` | `is_factual`, `issues`, `verified_temperatures` | Structured fact-check output. `verified_temperatures` maps Celsius string to computed Fahrenheit float. |
| `RDEMError` | `node`, `error_type`, `message`, `suggestion`, `attempt` | Structured error carrier. Error types: `crag_exception`, `relevance_low`, `factual_fail`, `api_fail`, `max_attempts_reached`. |
| `AgentStep` | `node_name`, `status`, `error`, `timestamp` | Audit trail entry. Status is one of: `success`, `error`, `forced_proceed`, `skipped`. |

---

## 8. Graph Wiring and Execution Flow

**File:** `agent/graph.py`

### 8.1 Nodes and Edges

```python
# Entry
START --> orchestration_agent

# Orchestrator routes based on current_step
orchestration_agent --[conditional]--> retriever_agent      (if "retrieve")
                                  --> weather_consultant    (if "weather_fallback")
                                  --> writer_agent          (if "write")
                                  --> writer_fact_checker   (if "fact_check")
                                  --> formatter             (if "format")
                                  --> END                   (if "done")

# Retriever routes based on outcome
retriever_agent --[conditional]--> weather_consultant    (if crag_rdem is set)
                               --> orchestration_agent   (if success)

# Fixed edges
weather_consultant --> orchestration_agent
writer_agent --> writer_fact_checker

# Fact checker routes based on result
writer_fact_checker --[conditional]--> orchestration_agent  (if factual or attempts >= 3)
                                  --> fix_fact_agent        (if not factual)

fix_fact_agent --> writer_fact_checker  (loop back)
formatter --> END
```

### 8.2 Retry Policies

| Node | Retry Policy |
|------|-------------|
| `weather_consultant` | 3 attempts, 1s initial, 2x backoff |
| `writer_agent` | 2 attempts, 1s initial, 2x backoff |
| `writer_fact_checker` | 2 attempts, 1s initial, 2x backoff |
| `fix_fact_agent` | 2 attempts, 1s initial, 2x backoff |
| `formatter` | 2 attempts, 1s initial, 2x backoff |

### 8.3 Memory

Uses `MemorySaver` (in-memory checkpointer). State is checkpointed after every node. Conversation continuity across queries is maintained via `thread_id`.

When starting a **follow-up query** on the same thread, the caller must explicitly reset pipeline-specific fields (`crag_output`, `draft_document`, `fact_check_result`, `final_latex_document`, etc.) to `None` while keeping the same `thread_id`. This preserves conversation memory (`messages`) while starting a fresh pipeline run.

---

## 9. Agent Nodes -- Deep Dive

### 9.1 Orchestrator (`orchestrator.py`)

**Type:** Deterministic router (no LLM call)

Evaluates state fields in priority order to determine `next_step`:

```
1. final_latex_document exists?  --> "done"
2. fact_check passed?            --> "format"
3. fact_check failed, attempts >= 3? --> "format" (forced_proceed)
4. fact_check failed, attempts < 3?  --> "fact_check"
5. draft_document exists, no fact_check yet? --> "fact_check"
6. crag_output exists?           --> "write"
7. crag_rdem exists?             --> "weather_fallback"
8. else                          --> "retrieve"
```

Returns a static `AIMessage` with the routing decision (no LLM needed) and appends an `AgentStep` to the audit trail.

### 9.2 Retriever (`retriever.py`)

**Type:** CRAG wrapper + LLM quality gate

Two-phase process:

**Phase 1: Run CRAG**
```python
result = _pipeline.run(state.user_query)
# result = {answer, sources, action, scores}
```

If `CRAGPipeline.run()` throws an exception, sets `crag_rdem` with `error_type="crag_exception"` and routes to weather consultant.

**Phase 2: Quality Gate**
An LLM judge (`_judge`) evaluates whether the CRAG answer actually addresses the query:
- Checks for outdated data (e.g., 2022 data when user asks about today)
- Checks the answer contains concrete weather data
- Checks location match
- Uses human-readable action labels (not raw "incorrect") to avoid biasing the judge

If `NOT_RELEVANT`: sets `crag_rdem` with `error_type="relevance_low"`, triggering the weather consultant fallback. The `crag_output` is still stored so the consultant can use it as context.

### 9.3 Weather Consultant (`weather_consultant.py`)

**Type:** ReAct agent with `web_search` tool
**When called:** Only when `crag_rdem` is set (CRAG exception or quality gate rejection)

The ReAct agent autonomously:
1. Formulates multiple search queries (current conditions, forecast, warnings)
2. Calls `web_search` tool for each (which checks ChromaDB cache first)
3. Compiles results into a comprehensive answer

If `state.crag_output` exists from the retriever's attempt, it's included in the prompt as context to avoid re-fetching identical data.

Source URLs are extracted from all messages in the ReAct loop using regex, deduplicated, and stored in `crag_output.sources`.

### 9.4 Writer (`writer.py`)

**Type:** ReAct agent with `celsius_to_fahrenheit` tool

Takes `crag_output.answer` and produces a structured markdown report:
- Uses `celsius_to_fahrenheit` for every temperature (never guesses)
- Includes numbered source citations `[1]`, `[2]`
- Forced to never refuse -- must produce a report with whatever data is available
- Notes when data comes from web search (low confidence)

### 9.5 Fact Checker (`fact_checker.py`)

**Type:** ReAct agent with `celsius_to_fahrenheit` tool

Verifies the `draft_document` against `crag_output.answer` (ground truth):
1. Identifies all temperature values in the draft
2. Calls `celsius_to_fahrenheit` for each to verify conversions
3. Compares factual claims against source data
4. Outputs structured JSON: `{is_factual, issues, verified_temperatures}`

**Rounding rule:** Differences of +/-0.1F are acceptable and do not cause `is_factual = false`.

**Routing after fact check:**
- `is_factual = true` --> return to orchestrator (routes to formatter)
- `is_factual = false, attempts < 3` --> route to `fix_fact_agent`
- `is_factual = false, attempts >= 3` --> force `is_factual = true`, log `forced_proceed`, route to formatter

### 9.6 Fix Fact Agent (`fix_fact.py`)

**Type:** ReAct agent with `celsius_to_fahrenheit` tool

Receives:
- `writer_rdem.message` (summary of issues)
- `fact_check_result.issues` (specific issue list)
- `fact_check_result.verified_temperatures` (tool-computed ground truth)
- `crag_output.answer` (original data)
- `draft_document` (current draft to fix)

Returns corrected `draft_document`, resets `fact_check_result` to `None` so the checker runs fresh.

### 9.7 Formatter (`formatter.py`)

**Type:** Direct LLM call (not ReAct -- no tools needed)

Converts `draft_document` to compilable LaTeX with:
- `\documentclass{article}` + `booktabs`, `geometry`, `hyperref`
- `\section{}` for each report section
- `\begin{tabular}` for temperature data
- Mandatory `\section{References}` with all CRAG sources as `\url{}`
- Data confidence note in `\textit{}`
- Strips markdown code fences if the LLM wraps output

---

## 10. Tools

**File:** `agent/tools/weather_tools.py`

### 10.1 celsius_to_fahrenheit

```python
@tool
def celsius_to_fahrenheit(celsius: float) -> float:
```

- Formula: `(celsius * 9/5) + 32`
- Validates input: raises `ToolException` if below absolute zero (-273.15C)
- Used by: Writer, Fact Checker, Fix Fact agents

### 10.2 web_search (with smart caching)

```python
@tool
def web_search(query: str) -> str:
```

Three-step process:

**Step 1: Check cache**
- Queries ChromaDB collection `web_search_cache` for similar queries
- Similarity threshold: cosine distance < 0.25
- TTL: 6 hours (entries older than this are discarded)
- If cache hit: returns cached content immediately (no API call)

**Step 2: Live Tavily search** (only if no cache hit)
- `TavilyClient.search(query, max_results=3)`
- Returns formatted results with source URLs

**Step 3: Save to cache**
- Stores each Tavily result in ChromaDB with metadata:
  - `source`: URL
  - `query`: original search query
  - `cached_at`: UTC timestamp
  - `type`: "web_search_cache"

Used by: Weather Consultant agent only.

---

## 11. Logging and Observability

**File:** `agent/logger.py`

### 11.1 Log Initialization

`init_run(run_id)` creates:
- A file handler writing to `logs/run_{run_id}.log`
- A console handler (same format)
- Both at DEBUG level

Format: `[HH:MM:SS] LEVEL | message`

### 11.2 @log_node Decorator

Every graph node function is wrapped with `@log_node`, which:

**On entry:**
- Logs `>>> ENTER {node_name}`
- Snapshots key state fields (query, step, CRAG action/score, draft length, etc.)

**On exit:**
- Logs `<<< EXIT {node_name} ({elapsed}s)`
- Logs every key in the output dict with truncated previews

**On exception:**
- Logs the exception with elapsed time, then re-raises

### 11.3 AgentTracer (LangChain Callback)

The `AgentTracer` class extends `BaseCallbackHandler` and is attached to every LLM/ReAct invocation. It captures:

| Event | What's Logged |
|-------|--------------|
| `on_chat_model_start` | Full LLM input: system prompt, human message, prior AI messages, tool results |
| `on_llm_end` | Full LLM output: AI response text, tool call requests (name + args) |
| `on_tool_start` | Tool name and input arguments |
| `on_tool_end` | Tool output/result |
| `on_tool_error` | Tool error message |
| `on_retry` | Retry attempt number |

This provides complete visibility into the ReAct reasoning loop: every thought, every tool call, every observation.

### 11.4 Audit Trail

Every node appends an `AgentStep` to `state.audit_trail`:

```python
AgentStep(
    node_name="writer",
    status="success",       # or "error", "forced_proceed"
    error=None,             # RDEMError if applicable
    timestamp="2026-04-06T18:03:33.123456"
)
```

Printed at the end of each run as a summary table.

---

## 12. Configuration

**File:** `config/settings.py`

| Constant | Value | Purpose |
|----------|-------|---------|
| `DEEPSEEK_API_KEY` | From `.env` | DeepSeek API authentication |
| `TAVILY_API_KEY` | From `.env` | Tavily web search authentication |
| `DEEPSEEK_BASE_URL` | `https://api.deepseek.com` | OpenAI-compatible endpoint |
| `DEEPSEEK_MODEL` | `deepseek-chat` | Model name for all LLM calls |
| `EMBEDDING_MODEL` | `all-MiniLM-L6-v2` | Sentence transformer for ChromaDB |
| `CHROMA_PERSIST_DIR` | `data/chroma_db` | ChromaDB storage path |
| `CHROMA_COLLECTION` | `weather_docs` | Default collection name |
| `CHUNK_SIZE` | 512 | Characters per chunk |
| `CHUNK_OVERLAP` | 64 | Character overlap between chunks |
| `HIGH_RELEVANCE` | 0.7 | Threshold for "correct" CRAG action |
| `LOW_RELEVANCE` | 0.3 | Threshold for "incorrect" CRAG action |
| `RETRIEVAL_K` | 3 | Number of documents to retrieve |

### Document Sources

Configured in `DOCUMENT_SOURCES` dict:
- **PDF:** NOAA monthly climate briefing
- **Web:** Open-Meteo API docs (forecast + historical)
- **Files:** Berlin forecast JSON + CSV from Open-Meteo API

---

## 13. Entry Points and Scripts

### 13.1 `ingest.py`

Downloads all document sources and indexes them into ChromaDB.

```bash
python ingest.py
```

### 13.2 `run_crag.py`

Tests the standalone CRAG pipeline with 3 hardcoded queries. Calls `setup_dspy()` first.

```bash
python run_crag.py
```

### 13.3 `run_agent.py`

Runs the full agent pipeline with a Berlin weather query + Hamburg follow-up.

```bash
python run_agent.py
```

Calls `setup_dspy()` before importing `agent.graph` (which imports `retriever.py`, which instantiates `CRAGPipeline()` at module level).

**Follow-up query handling:** The second `graph.invoke()` explicitly resets all pipeline fields (`crag_output=None`, `draft_document=None`, etc.) while keeping the same `thread_id` for conversation memory.

### 13.4 `run_10_scenarios.py`

Runs 10 diverse weather queries sequentially:
1. Berlin, 2. Tokyo, 3. London, 4. Dubai, 5. New York, 6. Riyadh, 7. Moscow, 8. Mumbai, 9. Chicago, 10. Sydney

For each:
- Uses a unique `thread_id`
- Initializes logging via `init_run()`
- Collects LaTeX output
- Prints a summary matrix at the end

Saves all LaTeX to `output_all_scenarios.tex` (with Unicode sanitization to LaTeX commands).

---

## 14. Error Handling (RDEM)

**RDEM = Retrieval-Document-Error Model**

All errors are structured as `RDEMError` objects with 5 fields:

| Field | Purpose |
|-------|---------|
| `node` | Which node raised the error |
| `error_type` | Category (see below) |
| `message` | Human-readable explanation |
| `suggestion` | Recovery guidance for the next node |
| `attempt` | Which retry attempt this is |

### Error Types

| Type | When It Occurs | Recovery |
|------|---------------|----------|
| `crag_exception` | `CRAGPipeline.run()` throws Python exception | Route to weather consultant |
| `relevance_low` | Quality gate rejects CRAG answer | Route to weather consultant |
| `factual_fail` | Fact checker finds issues in draft | Route to fix_fact_agent (up to 3x) |
| `api_fail` | Weather consultant's ReAct agent fails | Proceed with empty data |
| `max_attempts_reached` | Fact fix loop hit 3 iterations | Force proceed to formatter |

Errors are appended to both:
- `state.global_error_log` (session-level)
- `state.audit_trail[n].error` (step-level)

---

## 15. Data Flow: End-to-End Walkthrough

Query: **"What is the weather in Berlin this week?"**

### Step 1: Initialization
`run_agent.py` calls `setup_dspy()` (configures DSPy with DeepSeek), then `build_graph()` (imports all nodes, instantiates `CRAGPipeline` and ReAct agents at module level).

### Step 2: Orchestrator (1st call)
State: `user_query="What is the weather in Berlin this week?"`, everything else is default.
Decision: `next_step = "retrieve"` (nothing populated yet).

### Step 3: Retriever
Calls `CRAGPipeline.run("What is the weather in Berlin this week?")`.
- ChromaDB has Berlin docs from 2022 (ingested via `ingest.py`).
- Evaluator scores: `[0.80, 0.75, 0.72]` --> `action="correct"` (max > 0.7).
- CRAG generates a cited answer using the 2022 data.

**Quality Gate:** The LLM judge evaluates whether the answer is current.
- The data references July 2022. The user asks about "this week" (April 2026).
- Verdict: `NOT_RELEVANT`.
- Sets `crag_rdem` with `error_type="relevance_low"`.

### Step 4: Weather Consultant
Triggered because `crag_rdem is not None`.
ReAct agent receives the query + the CRAG answer as context.
- **Action 1:** `web_search("Berlin current weather temperature conditions")` --> Tavily returns current data
- **Action 2:** `web_search("Berlin 7 day weather forecast April 2026")` --> more data
- **Action 3:** `web_search("Berlin weather warnings")` --> additional info
- Compiles all into a comprehensive answer with current temperatures, forecast, sources.

### Step 5: Orchestrator (2nd call)
State: `crag_output` is populated (from weather consultant), `crag_rdem` is cleared.
Decision: `next_step = "write"`.

### Step 6: Writer Agent
ReAct agent receives the CRAG answer and source list.
- **Action 1:** `celsius_to_fahrenheit(10.2)` --> 50.36
- **Action 2:** `celsius_to_fahrenheit(7.6)` --> 45.68
- ... (for every temperature in the data)
- Produces a structured markdown report with tables, sections, citations.

### Step 7: Fact Checker
ReAct agent receives the draft report + source data.
- Re-calls `celsius_to_fahrenheit` for each temperature to verify.
- Compares all claims against source data.
- Result: `{is_factual: true, issues: ["Minor rounding: 50.36F vs 50.4F"], verified_temperatures: {...}}`

### Step 8: Orchestrator (3rd call)
State: `fact_check_result.is_factual = True`.
Decision: `next_step = "format"`.

### Step 9: Formatter
LLM converts the markdown report to LaTeX with `\documentclass{article}`, `booktabs` tables, `\section{References}`, data confidence note.

### Step 10: Orchestrator (4th call)
State: `final_latex_document` is populated.
Decision: `next_step = "done"` --> routes to END.

**Total: ~120-180 seconds per query** depending on API latency and number of web searches.

---

## 16. Key Design Decisions

### Why DSPy for CRAG but LangChain for Agents?
DSPy provides declarative Signature-based LLM programming ideal for the structured evaluate/refine/rewrite/generate pipeline. LangChain + LangGraph provides the stateful agent framework with tool support, memory, and graph-based orchestration. Each framework is used where it excels.

### Why a Deterministic Orchestrator?
The orchestrator was originally an LLM-based agent that generated narrative responses. This caused hallucinations (inventing city names), added ~2-6s latency per call (called 4x per run), and consumed API tokens for zero functional benefit since routing is deterministic based on state fields. Replacing it with pure `if/elif` logic eliminated all three problems.

### Why a Quality Gate After CRAG?
CRAG's relevance scoring is based on embedding similarity, which can be high even for semantically similar but temporally irrelevant data (e.g., "Berlin weather" from 2022 matches "Berlin weather this week" with score 0.80). The LLM quality gate adds temporal and content-level relevance checking that embeddings cannot provide.

### Why Cache Web Search Results?
Tavily API calls cost money and add latency. For repeated or similar queries within a 6-hour window, cached results are served from ChromaDB instantly. The cache uses the same embedding model as the main store, with a cosine distance threshold of 0.25 for similarity matching.

### Why Pydantic v2 BaseModel (Not TypedDict)?
Pydantic provides runtime validation (e.g., `fact_fix_attempts: int = Field(ge=0, le=3)` enforces the loop cap at the schema level), computed fields (`@model_validator`), and clean serialization (`model_dump()`). LangGraph works with both, but Pydantic gives stronger guarantees.

### Why ReAct Over Simple Reflex?
Reflex agents execute a fixed sequence of actions. ReAct agents reason about **which** tools to call and **how many times** based on the data they see. The writer agent might call `celsius_to_fahrenheit` 3 times or 15 times depending on how many temperatures appear in the source data. This flexibility is impossible with a fixed pipeline.

### Why LaTeX as Output?
LaTeX produces publication-quality documents with proper tables (`booktabs`), mathematical formatting, and bibliography support. It's the standard format for technical documentation and can be compiled to PDF with `pdflatex`.
