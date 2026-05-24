"""
Tests 4 + 5: Error propagation and partial continuation.
Test 4: Forced timeout → structured ErrorOutput with correct FailureType.
Test 5: Partial results from failed agent are preserved in final report.
"""
from __future__ import annotations

import asyncio
import uuid
from datetime import date

import pytest

from multi_agent_pipeline.orchestration.registry import AgentRegistry
from multi_agent_pipeline.orchestration.task_runner import TaskRunner
from multi_agent_pipeline.schemas.errors import ErrorOutput, FailureType
from multi_agent_pipeline.schemas.finding import FindingSchema, SourceSchema
from multi_agent_pipeline.schemas.task import AgentOutput, TaskRequest
from multi_agent_pipeline.synthesis.engine import SynthesisEngine

from .conftest import (
    MockDocAgent,
    TimeoutAgent,
    MalformedResponseAgent,
    make_task,
    SAMPLE_QUERY,
)


# ---------------------------------------------------------------------------
# Test 4: Timeout produces structured ErrorOutput
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_timeout_produces_error_output(telemetry):
    """A timed-out agent returns ErrorOutput with FailureType.TIMEOUT."""
    agent = TimeoutAgent(delay_ms=5000, telemetry=telemetry)
    task = make_task("WebResearchAgent", timeout_ms=100)
    result = await agent.execute(task)

    assert isinstance(result, ErrorOutput)
    assert result.status == "error"
    assert result.error.failure_type == FailureType.TIMEOUT


@pytest.mark.asyncio
async def test_timeout_error_is_retryable(telemetry):
    """TIMEOUT failures must be marked retryable."""
    agent = TimeoutAgent(delay_ms=5000, telemetry=telemetry)
    task = make_task("WebResearchAgent", timeout_ms=100)
    result = await agent.execute(task)

    assert isinstance(result, ErrorOutput)
    assert result.error.retryable is True


@pytest.mark.asyncio
async def test_timeout_captures_elapsed_time(telemetry):
    """elapsed_ms in ErrorDetail must reflect actual elapsed time."""
    agent = TimeoutAgent(delay_ms=5000, telemetry=telemetry)
    task = make_task("WebResearchAgent", timeout_ms=150)
    result = await agent.execute(task)

    assert isinstance(result, ErrorOutput)
    assert result.error.elapsed_ms >= 100  # at least the timeout duration


@pytest.mark.asyncio
async def test_malformed_response_not_retryable(telemetry):
    """Schema validation failures are not retryable."""
    agent = MalformedResponseAgent(telemetry=telemetry)
    task = make_task("WebResearchAgent", timeout_ms=5000)
    result = await agent.execute(task)

    assert isinstance(result, ErrorOutput)
    assert result.error.failure_type == FailureType.MALFORMED_RESPONSE
    assert result.error.retryable is False


@pytest.mark.asyncio
async def test_error_output_records_attempted_query(telemetry):
    """ErrorDetail.attempted_query should reference the original query."""
    agent = TimeoutAgent(delay_ms=5000, telemetry=telemetry)
    task = make_task("WebResearchAgent", query=SAMPLE_QUERY, timeout_ms=100)
    result = await agent.execute(task)

    assert isinstance(result, ErrorOutput)
    assert SAMPLE_QUERY in result.error.attempted_query


# ---------------------------------------------------------------------------
# Test 5: Partial continuation — pipeline continues despite agent failure
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_pipeline_continues_after_agent_failure(telemetry):
    """Final report is generated even when one agent fails."""
    registry = AgentRegistry()
    registry.register("WebResearchAgent", TimeoutAgent(delay_ms=5000, telemetry=telemetry))
    registry.register("DocumentAnalysisAgent", MockDocAgent(telemetry=telemetry))

    runner = TaskRunner(registry)
    tasks = [
        make_task("WebResearchAgent", timeout_ms=100),
        make_task("DocumentAnalysisAgent", timeout_ms=5000),
    ]

    results, _ = await runner.run_parallel(tasks)
    engine = SynthesisEngine()
    report = engine.synthesize(query=SAMPLE_QUERY, results=results)

    # Report is produced despite one failure
    assert report is not None
    assert report.query == SAMPLE_QUERY


@pytest.mark.asyncio
async def test_error_captured_in_final_report(telemetry):
    """errors_encountered in FinalReport contains the failed agent's ErrorOutput."""
    registry = AgentRegistry()
    registry.register("WebResearchAgent", TimeoutAgent(delay_ms=5000, telemetry=telemetry))
    registry.register("DocumentAnalysisAgent", MockDocAgent(telemetry=telemetry))

    runner = TaskRunner(registry)
    tasks = [
        make_task("WebResearchAgent", timeout_ms=100),
        make_task("DocumentAnalysisAgent", timeout_ms=5000),
    ]

    results, _ = await runner.run_parallel(tasks)
    engine = SynthesisEngine()
    report = engine.synthesize(query=SAMPLE_QUERY, results=results)

    assert len(report.errors_encountered) == 1
    assert report.errors_encountered[0].error.failure_type == FailureType.TIMEOUT


@pytest.mark.asyncio
async def test_coverage_gap_annotated_on_failure(telemetry):
    """Coverage gap annotation is added for each failed agent."""
    registry = AgentRegistry()
    registry.register("WebResearchAgent", TimeoutAgent(delay_ms=5000, telemetry=telemetry))
    registry.register("DocumentAnalysisAgent", MockDocAgent(telemetry=telemetry))

    runner = TaskRunner(registry)
    tasks = [
        make_task("WebResearchAgent", timeout_ms=100),
        make_task("DocumentAnalysisAgent", timeout_ms=5000),
    ]

    results, _ = await runner.run_parallel(tasks)
    engine = SynthesisEngine()
    report = engine.synthesize(query=SAMPLE_QUERY, results=results)

    assert len(report.coverage_gaps) >= 1
    assert any("WebResearchAgent" in gap for gap in report.coverage_gaps)


@pytest.mark.asyncio
async def test_successful_agent_findings_present_despite_sibling_failure(telemetry):
    """DocumentAnalysisAgent findings appear in the report even when WebResearchAgent fails."""
    registry = AgentRegistry()
    registry.register("WebResearchAgent", TimeoutAgent(delay_ms=5000, telemetry=telemetry))
    registry.register("DocumentAnalysisAgent", MockDocAgent(telemetry=telemetry))

    runner = TaskRunner(registry)
    tasks = [
        make_task("WebResearchAgent", timeout_ms=100),
        make_task("DocumentAnalysisAgent", timeout_ms=5000),
    ]

    results, _ = await runner.run_parallel(tasks)
    engine = SynthesisEngine()
    report = engine.synthesize(query=SAMPLE_QUERY, results=results)

    all_findings = report.well_established + [
        f for cf in report.contested for f in cf.sources
    ]
    assert len(all_findings) > 0
