from pydantic import BaseModel, Field, ConfigDict, model_validator
from typing import Annotated, Optional

from langgraph.graph.message import add_messages


class RDEMError(BaseModel):
    """Structured error carrier used by all agents."""

    node: str
    error_type: str
    message: str
    suggestion: str
    attempt: int = 1


class CRAGOutput(BaseModel):
    """Direct mirror of CRAGPipeline.run() return dict."""

    answer: str
    sources: list[str]
    action: str
    scores: list[float]
    max_score: float = 0.0
    is_high_confidence: bool = False
    is_ingested: bool = False
    ingested_at: str = ""  # ISO timestamp — when the cached answer was stored

    @model_validator(mode="after")
    def _compute_max_score(self):
        if self.scores:
            self.max_score = max(self.scores)
        else:
            self.max_score = 0.0
        return self


class FactCheckResult(BaseModel):
    is_factual: bool
    issues: list[str] = Field(default_factory=list)


class AgentStep(BaseModel):
    """Audit trail entry."""

    node_name: str
    status: str
    error: Optional[RDEMError] = None
    timestamp: str = ""


class WeatherAgentState(BaseModel):
    """Top-level graph state — passed to StateGraph(WeatherAgentState)."""

    messages: Annotated[list, add_messages] = Field(default_factory=list)
    user_query: str = Field(default="")

    # Step 1: Retrieval
    crag_output: Optional[CRAGOutput] = None
    crag_rdem: Optional[RDEMError] = None
    validation_passed: Optional[bool] = None

    # Step 2: Writing
    draft_document: Optional[str] = None
    fact_check_result: Optional[FactCheckResult] = None
    fact_fix_attempts: int = Field(default=0, ge=0, le=3)
    writer_rdem: Optional[RDEMError] = None

    # Step 3: Formatting
    final_latex_document: Optional[str] = None

    # Orchestration
    current_step: str = Field(default="retrieve")

    # Observability
    audit_trail: list[AgentStep] = Field(default_factory=list)
    global_error_log: list[RDEMError] = Field(default_factory=list)

    model_config = ConfigDict(arbitrary_types_allowed=True)