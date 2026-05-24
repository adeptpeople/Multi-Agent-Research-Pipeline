"""
Test 7: Provenance traceability.
Every synthesized statement must map back to originating findings via trace_map.
"""
from __future__ import annotations

import pytest

from multi_agent_pipeline.synthesis.provenance import ProvenanceTracker, ProvenanceLossError
from multi_agent_pipeline.synthesis.engine import SynthesisEngine

from .conftest import make_finding, SAMPLE_QUERY


class TestProvenanceTracker:
    def test_register_and_retrieve_trace(self):
        """Registered paragraph IDs appear in trace_map."""
        tracker = ProvenanceTracker()
        tracker.register_paragraph("paragraph_1", ["task-001.finding-1", "task-002.finding-1"])
        assert "paragraph_1" in tracker.trace_map
        assert "task-001.finding-1" in tracker.trace_map["paragraph_1"]

    def test_validate_passes_when_all_findings_traced(self):
        """validate() does not raise when all well-established findings are in trace_map."""
        tracker = ProvenanceTracker()
        finding = make_finding("task-001", 1)
        tracker.register_paragraph("paragraph_1", [finding.finding_id])
        tracker.validate([finding])  # should not raise

    def test_validate_raises_on_orphaned_finding(self):
        """validate() raises ProvenanceLossError if a finding is not in any paragraph."""
        tracker = ProvenanceTracker()
        finding = make_finding("task-001", 1)
        # Register a paragraph that does NOT include this finding
        tracker.register_paragraph("paragraph_1", ["task-999.finding-1"])

        with pytest.raises(ProvenanceLossError, match="Provenance loss detected"):
            tracker.validate([finding])

    def test_completeness_pct_100_when_all_traced(self):
        """completeness_pct returns 100.0 when all findings are covered."""
        tracker = ProvenanceTracker()
        findings = [make_finding("task-001", i) for i in range(1, 4)]
        tracker.register_paragraph("paragraph_1", [f.finding_id for f in findings])
        assert tracker.completeness_pct(findings) == 100.0

    def test_completeness_pct_partial(self):
        """completeness_pct returns correct percentage for partial coverage."""
        tracker = ProvenanceTracker()
        findings = [make_finding("task-001", i) for i in range(1, 5)]
        # Only register 2 out of 4
        tracker.register_paragraph("paragraph_1", [findings[0].finding_id, findings[1].finding_id])
        pct = tracker.completeness_pct(findings)
        assert pct == 50.0

    def test_empty_findings_returns_100(self):
        """completeness_pct returns 100.0 when there are no findings."""
        tracker = ProvenanceTracker()
        assert tracker.completeness_pct([]) == 100.0


class TestSynthesisProvenance:
    @pytest.mark.asyncio
    async def test_trace_map_covers_all_well_established(self, runner_fast, two_tasks, synthesis_engine):
        """Every well-established finding ID appears in trace_map after synthesis."""
        results, _ = await runner_fast.run_parallel(two_tasks)
        report = synthesis_engine.synthesize(query=SAMPLE_QUERY, results=results)

        referenced = {fid for ids in report.trace_map.values() for fid in ids}
        for finding in report.well_established:
            assert finding.finding_id in referenced, (
                f"Finding {finding.finding_id} not traceable in trace_map"
            )

    @pytest.mark.asyncio
    async def test_trace_map_is_non_empty(self, runner_fast, two_tasks, synthesis_engine):
        """trace_map must have at least one entry when findings exist."""
        results, _ = await runner_fast.run_parallel(two_tasks)
        report = synthesis_engine.synthesize(query=SAMPLE_QUERY, results=results)

        all_findings = report.well_established + [f for cf in report.contested for f in cf.sources]
        if all_findings:
            assert len(report.trace_map) > 0

    @pytest.mark.asyncio
    async def test_synthesis_paragraphs_correspond_to_trace_keys(self, runner_fast, two_tasks, synthesis_engine):
        """Number of trace_map entries matches number of synthesis paragraphs."""
        results, _ = await runner_fast.run_parallel(two_tasks)
        report = synthesis_engine.synthesize(query=SAMPLE_QUERY, results=results)

        assert len(report.trace_map) == len(report.synthesis_paragraphs)

    @pytest.mark.asyncio
    async def test_provenance_completeness_pct_is_100(self, runner_fast, two_tasks, synthesis_engine):
        """Completeness metric must be 100% — no finding orphaned from trace_map."""
        results, _ = await runner_fast.run_parallel(two_tasks)
        report = synthesis_engine.synthesize(query=SAMPLE_QUERY, results=results)

        from multi_agent_pipeline.synthesis.provenance import ProvenanceTracker
        tracker = ProvenanceTracker()
        for para_id, finding_ids in report.trace_map.items():
            tracker.register_paragraph(para_id, finding_ids)

        pct = tracker.completeness_pct(report.well_established)
        assert pct == 100.0, f"Provenance completeness was {pct}%, expected 100%"
