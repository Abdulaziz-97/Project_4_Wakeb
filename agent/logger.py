\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
   

import logging
import os
import json
import textwrap
from datetime import datetime
from functools import wraps
from typing import Any

from langchain_core.callbacks import BaseCallbackHandler

_LOG_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "logs")
_RUN_ID: str | None = None


                                                                        
               
                                                                        

def init_run(run_id: str | None = None) -> str:
    global _RUN_ID
    _RUN_ID = run_id or datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    os.makedirs(_LOG_DIR, exist_ok=True)

    logger = logging.getLogger("weather_agent")
    logger.setLevel(logging.DEBUG)
    logger.handlers.clear()

    fmt = logging.Formatter(
        "[%(asctime)s] %(levelname)-5s | %(message)s",
        datefmt="%H:%M:%S",
    )

    ch = logging.StreamHandler()
    ch.setLevel(logging.DEBUG)
    ch.setFormatter(fmt)
    logger.addHandler(ch)

    fh = logging.FileHandler(
        os.path.join(_LOG_DIR, f"run_{_RUN_ID}.log"),
        encoding="utf-8",
    )
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(fmt)
    logger.addHandler(fh)

    logger.info("=" * 80)
    logger.info(f"  RUN {_RUN_ID}  |  {datetime.utcnow().isoformat()}")
    logger.info("=" * 80)

    return _RUN_ID


def get_logger() -> logging.Logger:
    return logging.getLogger("weather_agent")


                                                                        
                                                                   
                                                                        

def _indent(text: str, prefix: str = "        ") -> str:
                                    
    return textwrap.indent(str(text), prefix)


def _trunc(text: str, n: int = 1500) -> str:
    if len(text) > n:
        return text[:n] + f"\n        ... [{len(text)} chars total, truncated]"
    return text


class AgentTracer(BaseCallbackHandler):
\
\
\
\
       

    def __init__(self, node_name: str):
        super().__init__()
        self.node_name = node_name
        self.logger = logging.getLogger("weather_agent")
        self.llm_call = 0

                                                                   

    def on_chat_model_start(self, serialized, messages, **kwargs):
        self.llm_call += 1
        tag = f"[{self.node_name}] LLM Call #{self.llm_call}"
        self.logger.debug("")
        self.logger.debug(f"    {tag}  INPUT  {'~' * 40}")

        for msg_group in messages:
            for msg in msg_group:
                role = type(msg).__name__.replace("Message", "").upper()
                content = getattr(msg, "content", "")

                if role == "SYSTEM":
                    self.logger.debug(f"      SYSTEM:")
                    self.logger.debug(_indent(_trunc(content)))
                elif role == "HUMAN":
                    self.logger.debug(f"      HUMAN:")
                    self.logger.debug(_indent(_trunc(content)))
                elif role == "AI":
                    self.logger.debug(f"      AI (prior):")
                    self.logger.debug(_indent(_trunc(content)))
                    tool_calls = getattr(msg, "tool_calls", [])
                    if tool_calls:
                        for tc in tool_calls:
                            self.logger.debug(
                                f"        -> ToolCall: {tc['name']}({json.dumps(tc['args'])})"
                            )
                elif role == "TOOL":
                    name = getattr(msg, "name", "?")
                    self.logger.debug(f"      TOOL RESULT ({name}):")
                    self.logger.debug(_indent(_trunc(content)))
                else:
                    self.logger.debug(f"      {role}:")
                    self.logger.debug(_indent(_trunc(str(content))))

    def on_llm_end(self, response, **kwargs):
        tag = f"[{self.node_name}] LLM Call #{self.llm_call}"

        try:
            msg = response.generations[0][0].message
        except (IndexError, AttributeError):
            self.logger.debug(f"    {tag}  OUTPUT: {_trunc(str(response))}")
            return

        content = getattr(msg, "content", "")
        tool_calls = getattr(msg, "tool_calls", [])

        self.logger.debug(f"    {tag}  OUTPUT {'~' * 39}")

        if content:
            self.logger.debug(f"      AI RESPONSE:")
            self.logger.debug(_indent(_trunc(content)))

        if tool_calls:
            self.logger.debug(f"      TOOL CALLS REQUESTED:")
            for tc in tool_calls:
                self.logger.debug(
                    f"        Action: {tc['name']}({json.dumps(tc['args'])})"
                )
                self.logger.debug(f"        ID: {tc.get('id', '?')}")
        elif not content:
            self.logger.debug(f"      (empty response)")

                                                                   

    def on_tool_start(self, serialized, input_str, **kwargs):
        name = serialized.get("name", kwargs.get("name", "?"))
        self.logger.debug("")
        self.logger.debug(
            f"    [{self.node_name}] TOOL EXECUTE: {name}"
        )
        self.logger.debug(f"      Input: {_trunc(str(input_str), 800)}")

    def on_tool_end(self, output, **kwargs):
        self.logger.debug(f"    [{self.node_name}] TOOL RESULT:")
        self.logger.debug(_indent(_trunc(str(output), 800)))

    def on_tool_error(self, error, **kwargs):
        self.logger.debug(f"    [{self.node_name}] TOOL ERROR: {error}")

                                                                   

    def on_chain_start(self, serialized, inputs, **kwargs):
        if not serialized:
            return
        name = serialized.get("name", "")
        if name and name not in ("RunnableSequence", "RunnableLambda", ""):
            self.logger.debug(f"    [{self.node_name}] Chain: {name}")

    def on_retry(self, retry_state, **kwargs):
        self.logger.debug(
            f"    [{self.node_name}] RETRY: attempt {retry_state.attempt_number}"
        )


def create_tracer(node_name: str) -> AgentTracer:
                                                       
    return AgentTracer(node_name)


                                                                        
                                                 
                                                                        

def _state_snapshot(state) -> dict:
                                                                
    snap = {}
    for key in ("user_query", "current_step", "fact_fix_attempts"):
        val = getattr(state, key, None)
        if val is not None:
            snap[key] = val

    co = getattr(state, "crag_output", None)
    if co:
        snap["crag_output"] = (
            f"action={co.action} score={co.max_score:.2f} "
            f"sources={len(co.sources)} answer={len(co.answer)} chars"
        )

    if getattr(state, "crag_rdem", None):
        r = state.crag_rdem
        snap["crag_rdem"] = f"{r.error_type}: {r.message[:120]}"

    d = getattr(state, "draft_document", None)
    if d:
        snap["draft_document"] = f"{len(d)} chars"

    fc = getattr(state, "fact_check_result", None)
    if fc:
        snap["fact_check"] = f"factual={fc.is_factual} issues={fc.issues}"

    wr = getattr(state, "writer_rdem", None)
    if wr:
        snap["writer_rdem"] = f"{wr.error_type}: {wr.message[:120]}"

    ftx = getattr(state, "final_latex_document", None)
    if ftx:
        snap["final_latex"] = f"{len(ftx)} chars"

    snap["msgs"] = len(getattr(state, "messages", []))
    snap["trail"] = len(getattr(state, "audit_trail", []))
    snap["errors"] = len(getattr(state, "global_error_log", []))
    return snap


def log_node(func):
                                                                   

    @wraps(func)
    def wrapper(state):
        logger = get_logger()
        name = func.__name__
        t0 = datetime.utcnow()

        logger.info("")
        logger.info("=" * 80)
        logger.info(f"  >>> ENTER  {name}")
        logger.info("=" * 80)

        snap = _state_snapshot(state)
        logger.debug(f"  [{name}] STATE ON ENTRY:")
        for k, v in snap.items():
            logger.debug(f"    {k}: {v}")

        try:
            result = func(state)
        except Exception as exc:
            elapsed = (datetime.utcnow() - t0).total_seconds()
            logger.error(f"  [{name}] EXCEPTION after {elapsed:.1f}s: {exc}")
            raise

        elapsed = (datetime.utcnow() - t0).total_seconds()

        logger.info("")
        logger.info(f"  <<< EXIT   {name}  ({elapsed:.1f}s)")
        logger.info("-" * 80)

        logger.debug(f"  [{name}] OUTPUT keys: {list(result.keys())}")
        for key, val in result.items():
            if key == "messages":
                for m in val:
                    c = getattr(m, "content", str(m))
                    logger.debug(f"    messages += [{type(m).__name__}]")
                    logger.debug(_indent(_trunc(c, 600)))
            elif key == "audit_trail":
                logger.debug(f"    audit_trail: {len(val)} steps")
            elif key == "global_error_log":
                logger.debug(f"    global_error_log: {len(val)} entries")
            elif key == "crag_output" and val is not None:
                if hasattr(val, "model_dump"):
                    d = val.model_dump()
                    logger.debug(f"    crag_output.action: {d.get('action')}")
                    logger.debug(f"    crag_output.max_score: {d.get('max_score')}")
                    logger.debug(f"    crag_output.sources: {d.get('sources')}")
                    ans = d.get("answer", "")
                    logger.debug(f"    crag_output.answer:")
                    logger.debug(_indent(_trunc(ans, 800)))
                else:
                    logger.debug(f"    crag_output: {_trunc(str(val), 500)}")
            elif key == "draft_document" and val:
                logger.debug(f"    draft_document ({len(val)} chars):")
                logger.debug(_indent(_trunc(val, 800)))
            elif key == "final_latex_document" and val:
                logger.debug(f"    final_latex_document ({len(val)} chars):")
                logger.debug(_indent(_trunc(val, 600)))
            elif key == "fact_check_result" and val is not None:
                if hasattr(val, "model_dump"):
                    logger.debug(f"    fact_check_result: {val.model_dump()}")
                else:
                    logger.debug(f"    fact_check_result: {val}")
            else:
                logger.debug(f"    {key}: {_trunc(str(val), 400)}")

        return result

    return wrapper
