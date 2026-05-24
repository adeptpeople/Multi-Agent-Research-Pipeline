"""
Test 3: Structured output contract.
Every finding must satisfy the full provenance schema.
"""
from __future__ import annotations

from datetime import date

import pytest

from multi_agent_pipeline.schemas.finding import FindingSchema, SourceSchema
from multi_agent_pipeline.schemas.task import AgentOutput
from pydantic import ValidationError

from .conftest import make_finding, make_task, SAMPLE_QUERY


@pytest.mark.asyncio
async def test_successful_agent_output_satisfies_schema(runner_fast, two_tasks):
    """All findings from mock agents satisfy the output contract."""
    results, _ = await runner_fast.run_parallel(two_tasks)
    for result in results:
        assert result.status == "success"
        for finding in result.findings:
            assert finding.claim, "claim must be non-empty"
            assert finding.evidence_excerpt, "evidence_excerpt must be non-empty"
            assert finding.source is not None, "source must be present"
            assert finding.source.publisher, "publisher must be non-empty"
            assert finding.source.publication_date is not None
            assert 0.0 <= finding.confidence <= 1.0
            assert 0.0 <= finding.source.credibility_score <= 1.0


def test_finding_with_missing_claim_raises():
    """FindingSchema requires a non-empty claim."""
    with pytest.raises(ValidationError):
        FindingSchema(
            finding_id="task-001.finding-1",
            claim="",  # empty
            evidence_excerpt="some evidence",
            source=SourceSchema(
                type="url",
                location="https://example.gov",
                publisher="Test",
                publication_date=date(2025, 1, 1),
                credibility_score=0.9,
            ),
            confidence=0.8,
        )


def test_finding_with_out_of_range_confidence_raises():
    """Confidence must be in [0.0, 1.0]."""
    with pytest.raises(ValidationError):
        FindingSchema(
            finding_id="task-001.finding-1",
            claim="Some claim",
            evidence_excerpt="Some evidence",
            source=SourceSchema(
                type="url",
                location="https://example.gov",
                publisher="Test",
                publication_date=date(2025, 1, 1),
                credibility_score=0.9,
            ),
            confidence=1.5,  # out of range
        )


def test_source_with_invalid_credibility_raises():
    """credibility_score must be in [0.0, 1.0]."""
    with pytest.raises(ValidationError):
        SourceSchema(
            type="url",
            location="https://example.gov",
            publisher="Test",
            publication_date=date(2025, 1, 1),
            credibility_score=2.0,  # out of range
        )


def test_validate_provenance_detects_missing_fields():
    """validate_provenance() returns False when source has no locatable reference."""
    # Empty location + no document_name → provenance incomplete at the logical level
    finding = FindingSchema(
        finding_id="task-001.finding-1",
        claim="Some claim",
        evidence_excerpt="Some excerpt",
        source=SourceSchema(
            type="document",
            location="",           # empty location
            document_name=None,    # and no document_name → no locatable source
            publisher="Test",
            publication_date=date(2025, 1, 1),
            credibility_score=0.9,
        ),
        confidence=0.8,
    )
    assert finding.validate_provenance() is False


def test_validate_provenance_passes_for_complete_finding():
    """validate_provenance() returns True for a complete finding."""
    finding = make_finding("task-001", 1)
    assert finding.validate_provenance() is True


def test_agent_output_finding_ids_are_unique():
    """All finding_ids within an AgentOutput must be unique."""
    task_id = "task-test"
    findings = [make_finding(task_id, i) for i in range(1, 4)]
    ids = [f.finding_id for f in findings]
    assert len(ids) == len(set(ids))
