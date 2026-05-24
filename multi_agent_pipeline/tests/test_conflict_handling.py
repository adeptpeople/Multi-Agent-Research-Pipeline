"""
Test 6: Conflicting evidence preservation.
Validates that contradictory claims from credible sources are both retained,
never silently averaged or arbitrarily chosen.
"""
from __future__ import annotations

import pytest

from multi_agent_pipeline.synthesis.conflict_detector import ConflictDetector
from multi_agent_pipeline.synthesis.engine import SynthesisEngine

from .conftest import SAMPLE_QUERY


class TestConflictDetection:
    def test_conflicting_percentages_detected(self, conflicting_outputs):
        """12% vs 19% growth rate should be flagged as contested."""
        output_a, output_b = conflicting_outputs
        all_findings = output_a.findings + output_b.findings

        detector = ConflictDetector()
        well_established, contested = detector.detect(all_findings)

        assert len(contested) >= 1, "Expected at least one contested finding"

    def test_both_conflicting_values_preserved(self, conflicting_outputs):
        """Both 12% and 19% values must appear in contested sources — none dropped."""
        output_a, output_b = conflicting_outputs
        all_findings = output_a.findings + output_b.findings

        detector = ConflictDetector()
        _, contested = detector.detect(all_findings)

        all_contested_claims = " ".join(f.claim for cf in contested for f in cf.sources)
        assert "12%" in all_contested_claims, "12% value was silently dropped"
        assert "19%" in all_contested_claims, "19% value was silently dropped"

    def test_contested_includes_both_publishers(self, conflicting_outputs):
        """Both OECD and McKinsey must appear in contested sources."""
        output_a, output_b = conflicting_outputs
        all_findings = output_a.findings + output_b.findings

        detector = ConflictDetector()
        _, contested = detector.detect(all_findings)

        publishers = {f.source.publisher for cf in contested for f in cf.sources}
        assert "OECD" in publishers
        assert "McKinsey Global Institute" in publishers

    def test_non_conflicting_findings_go_to_well_established(self, conflicting_outputs):
        """When there's only one finding per topic, it goes to well_established."""
        from .conftest import make_finding
        sole_finding = make_finding("task-sole", 1, "Export licenses cover H100 GPUs.")
        detector = ConflictDetector()
        well_established, contested = detector.detect([sole_finding])
        assert len(well_established) == 1
        assert len(contested) == 0

    def test_synthesis_report_contains_contested_section(self, conflicting_outputs):
        """FinalReport.contested is non-empty for conflicting outputs."""
        output_a, output_b = conflicting_outputs
        engine = SynthesisEngine()
        report = engine.synthesize(
            query="AI chip market growth rate 2024",
            results=[output_a, output_b],
        )
        assert len(report.contested) >= 1

    def test_synthesis_does_not_average_contested_values(self, conflicting_outputs):
        """Report text must not contain averaged value (e.g. 15.5%)."""
        output_a, output_b = conflicting_outputs
        engine = SynthesisEngine()
        report = engine.synthesize(
            query="AI chip market growth rate 2024",
            results=[output_a, output_b],
        )
        full_text = " ".join(report.synthesis_paragraphs)
        # Averaged value would be ~15.5% — must NOT appear
        assert "15.5%" not in full_text
        assert "15%" not in full_text or ("12%" in full_text and "19%" in full_text)

    def test_contested_finding_has_explanation(self, conflicting_outputs):
        """ContestedFinding must include a non-empty explanation."""
        output_a, output_b = conflicting_outputs
        all_findings = output_a.findings + output_b.findings

        detector = ConflictDetector()
        _, contested = detector.detect(all_findings)

        for cf in contested:
            assert cf.explanation, "ContestedFinding.explanation must be non-empty"
