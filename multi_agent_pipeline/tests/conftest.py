"""
Shared fixtures for the multi-agent pipeline test suite.
All agents here are deterministic mocks — no network calls, no API keys required.
"""
from __future__ import annotations

import asyncio
import uuid
from datetime import date
from typing import Union

import pytest
import pytest_asyncio

from multi_agent_pipeline.agents.base_agent import BaseAgent
from multi_agent_pipeline.observability.telemetry import TelemetryCollector
from multi_agent_pipeline.orchestration.registry import AgentRegistry
from multi_agent_pipeline.orchestration.task_runner import TaskRunner
from multi_agent_pipeline.schemas.errors import ErrorDetail, ErrorOutput, FailureType
from multi_agent_pipeline.schemas.finding import FindingSchema, SourceSchema
from multi_agent_pipeline.schemas.task import AgentOutput, TaskRequest
from multi_agent_pipeline.synthesis.engine import SynthesisEngine


# ---------------------------------------------------------------------------
# Shared constants
# ---------------------------------------------------------------------------

SAMPLE_QUERY = "What are the latest AI chip export restrictions?"
SAMPLE_CONTEXT = "Regulatory research for US-China semiconductor policy comparison."


def make_task(
    agent_name: str = "WebResearchAgent",
    query: str = SAMPLE_QUERY,
    timeout_ms: int = 5000,
) -> TaskRequest:
    task_id = f"task-{uuid.uuid4().hex[:8]}"
    return TaskRequest(
        task_id=task_id,
        agent_name=agent_name,
        prompt=(
            f"Research Question (original): {query}\n\n"
            f"Agent Role: {agent_name}\n\n"
            "Return structured JSON findings."
        ),
        context={
            "original_query": query,
            "user_context": SAMPLE_CONTEXT,
        },
        timeout_ms=timeout_ms,
    )


def make_finding(
    task_id: str,
    index: int = 1,
    claim: str = "AI chip exports increased.",
    publisher: str = "Test Publisher",
    pub_date: date = date(2025, 1, 1),
    confidence: float = 0.85,
) -> FindingSchema:
    return FindingSchema(
        finding_id=f"{task_id}.finding-{index}",
        claim=claim,
        evidence_excerpt=f"Evidence for: {claim}",
        source=SourceSchema(
            type="url",
            location=f"https://example.gov/report-{index}",
            publisher=publisher,
            publication_date=pub_date,
            credibility_score=0.90,
        ),
        confidence=confidence,
    )


# ---------------------------------------------------------------------------
# Mock agents
# ---------------------------------------------------------------------------

class MockWebAgent(BaseAgent):
    """Returns 2 deterministic findings in ~50ms."""
    name = "WebResearchAgent"

    def __init__(self, delay_ms: int = 50, telemetry: TelemetryCollector | None = None):
        super().__init__(telemetry)
        self._delay_ms = delay_ms

    async def _run(self, task: TaskRequest) -> AgentOutput:
        await asyncio.sleep(self._delay_ms / 1000)
        return AgentOutput(
            agent_name=self.name,
            task_id=task.task_id,
            findings=[
                make_finding(task.task_id, 1, "US imposed new AI chip export controls in 2025."),
                make_finding(task.task_id, 2, "Export licensing requirements now cover H100-class GPUs."),
            ],
            metadata={"execution_time_ms": self._delay_ms, "query_used": SAMPLE_QUERY},
        )


class MockDocAgent(BaseAgent):
    """Returns 2 deterministic findings in ~40ms."""
    name = "DocumentAnalysisAgent"

    def __init__(self, delay_ms: int = 40, telemetry: TelemetryCollector | None = None):
        super().__init__(telemetry)
        self._delay_ms = delay_ms

    async def _run(self, task: TaskRequest) -> AgentOutput:
        await asyncio.sleep(self._delay_ms / 1000)
        return AgentOutput(
            agent_name=self.name,
            task_id=task.task_id,
            findings=[
                make_finding(
                    task.task_id, 1,
                    "Internal policy document confirms chip export ban effective Q2 2025.",
                    publisher="Internal Policy Memo",
                    pub_date=date(2025, 3, 15),
                ),
                make_finding(
                    task.task_id, 2,
                    "Compliance checklist requires pre-approval for all dual-use AI hardware.",
                    publisher="Internal Policy Memo",
                    pub_date=date(2025, 3, 15),
                ),
            ],
            metadata={"execution_time_ms": self._delay_ms, "query_used": SAMPLE_QUERY},
        )


class SlowWebAgent(MockWebAgent):
    """200ms delay — used to measure parallel speedup."""
    def __init__(self, telemetry: TelemetryCollector | None = None):
        super().__init__(delay_ms=200, telemetry=telemetry)


class SlowDocAgent(MockDocAgent):
    """180ms delay — used to measure parallel speedup."""
    def __init__(self, telemetry: TelemetryCollector | None = None):
        super().__init__(delay_ms=180, telemetry=telemetry)


class TimeoutAgent(BaseAgent):
    """Simulates a timeout after configurable ms."""
    name = "WebResearchAgent"

    def __init__(self, delay_ms: int = 5000, telemetry: TelemetryCollector | None = None):
        super().__init__(telemetry)
        self._delay_ms = delay_ms

    async def _run(self, task: TaskRequest) -> AgentOutput:
        await asyncio.sleep(self._delay_ms / 1000)
        raise asyncio.TimeoutError()


class MalformedResponseAgent(BaseAgent):
    """Returns JSON garbage — triggers MALFORMED_RESPONSE."""
    name = "WebResearchAgent"

    async def _run(self, task: TaskRequest) -> AgentOutput:
        from multi_agent_pipeline.agents.base_agent import SchemaValidationError
        raise SchemaValidationError("Response was not valid JSON: {{{ garbage }")


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def sample_query() -> str:
    return SAMPLE_QUERY


@pytest.fixture
def telemetry() -> TelemetryCollector:
    return TelemetryCollector()


@pytest.fixture
def mock_web_agent(telemetry: TelemetryCollector) -> MockWebAgent:
    return MockWebAgent(telemetry=telemetry)


@pytest.fixture
def mock_doc_agent(telemetry: TelemetryCollector) -> MockDocAgent:
    return MockDocAgent(telemetry=telemetry)


@pytest.fixture
def slow_web_agent(telemetry: TelemetryCollector) -> SlowWebAgent:
    return SlowWebAgent(telemetry=telemetry)


@pytest.fixture
def slow_doc_agent(telemetry: TelemetryCollector) -> SlowDocAgent:
    return SlowDocAgent(telemetry=telemetry)


@pytest.fixture
def timeout_agent(telemetry: TelemetryCollector) -> TimeoutAgent:
    return TimeoutAgent(delay_ms=5000, telemetry=telemetry)


@pytest.fixture
def malformed_agent(telemetry: TelemetryCollector) -> MalformedResponseAgent:
    return MalformedResponseAgent(telemetry=telemetry)


@pytest.fixture
def registry_fast(
    mock_web_agent: MockWebAgent, mock_doc_agent: MockDocAgent
) -> AgentRegistry:
    reg = AgentRegistry()
    reg.register("WebResearchAgent", mock_web_agent)
    reg.register("DocumentAnalysisAgent", mock_doc_agent)
    return reg


@pytest.fixture
def registry_slow(
    slow_web_agent: SlowWebAgent, slow_doc_agent: SlowDocAgent
) -> AgentRegistry:
    reg = AgentRegistry()
    reg.register("WebResearchAgent", slow_web_agent)
    reg.register("DocumentAnalysisAgent", slow_doc_agent)
    return reg


@pytest.fixture
def runner_fast(registry_fast: AgentRegistry) -> TaskRunner:
    return TaskRunner(registry_fast)


@pytest.fixture
def runner_slow(registry_slow: AgentRegistry) -> TaskRunner:
    return TaskRunner(registry_slow)


@pytest.fixture
def synthesis_engine() -> SynthesisEngine:
    return SynthesisEngine()


@pytest.fixture
def two_tasks() -> list[TaskRequest]:
    return [
        make_task("WebResearchAgent"),
        make_task("DocumentAnalysisAgent"),
    ]


@pytest.fixture
def conflicting_outputs() -> tuple[AgentOutput, AgentOutput]:
    """Two outputs reporting conflicting market growth figures."""
    task_id_a = f"task-{uuid.uuid4().hex[:8]}"
    task_id_b = f"task-{uuid.uuid4().hex[:8]}"
    output_a = AgentOutput(
        agent_name="WebResearchAgent",
        task_id=task_id_a,
        findings=[
            FindingSchema(
                finding_id=f"{task_id_a}.finding-1",
                claim="AI chip market grew 12% year-over-year in 2024.",
                evidence_excerpt="OECD report confirms 12% semiconductor market growth.",
                source=SourceSchema(
                    type="url",
                    location="https://oecd.org/report-2025",
                    publisher="OECD",
                    publication_date=date(2025, 1, 15),
                    credibility_score=0.95,
                ),
                confidence=0.91,
            )
        ],
        metadata={"execution_time_ms": 1000, "query_used": "AI chip market growth"},
    )
    output_b = AgentOutput(
        agent_name="DocumentAnalysisAgent",
        task_id=task_id_b,
        findings=[
            FindingSchema(
                finding_id=f"{task_id_b}.finding-1",
                claim="AI chip market grew 19% year-over-year in 2024.",
                evidence_excerpt="McKinsey analysis estimates 19% YoY growth in AI silicon revenue.",
                source=SourceSchema(
                    type="document",
                    location="mckinsey_2025.pdf",
                    document_name="mckinsey_2025.pdf",
                    publisher="McKinsey Global Institute",
                    publication_date=date(2025, 2, 10),
                    credibility_score=0.85,
                ),
                confidence=0.88,
            )
        ],
        metadata={"execution_time_ms": 800, "query_used": "AI chip market growth"},
    )
    return output_a, output_b
