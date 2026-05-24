from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field, model_validator

from .finding import ContestedFinding, FindingSchema


class TaskRequest(BaseModel):
    task_id: str
    agent_name: str
    # Full self-contained prompt — must contain enough context for agent to act autonomously.
    prompt: str
    # Structured metadata for programmatic use by the agent.
    context: dict[str, Any] = Field(default_factory=dict)
    timeout_ms: int = 30_000

    @model_validator(mode="after")
    def prompt_must_contain_query(self) -> "TaskRequest":
        original_query = self.context.get("original_query", "")
        if original_query and original_query not in self.prompt:
            raise ValueError(
                f"Explicit context violation: prompt for {self.agent_name} does not "
                f"contain the original query. All context must be injected explicitly."
            )
        return self


class AgentOutput(BaseModel):
    agent_name: str
    task_id: str
    status: Literal["success"] = "success"
    findings: list[FindingSchema]
    metadata: dict[str, Any] = Field(default_factory=dict)
    # metadata keys: execution_time_ms, query_used, token_usage


class BenchmarkResult(BaseModel):
    mode: Literal["sequential", "parallel"]
    web_agent_ms: float
    doc_agent_ms: float
    synthesis_ms: float
    total_ms: float
    latency_improvement_pct: float = 0.0  # only set on parallel result


class FinalReport(BaseModel):
    query: str
    well_established: list[FindingSchema]
    contested: list[ContestedFinding]
    coverage_gaps: list[str]
    synthesis_paragraphs: list[str]
    # "paragraph_N" → ["task-001.finding-2", ...]
    trace_map: dict[str, list[str]] = Field(default_factory=dict)
    benchmark_sequential: BenchmarkResult | None = None
    benchmark_parallel: BenchmarkResult | None = None
    telemetry: list[Any] = Field(default_factory=list)
    errors_encountered: list[Any] = Field(default_factory=list)
