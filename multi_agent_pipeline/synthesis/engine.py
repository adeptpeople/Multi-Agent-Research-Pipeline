from __future__ import annotations

import time
from typing import Union

from ..schemas.errors import ErrorOutput
from ..schemas.finding import ContestedFinding, FindingSchema
from ..schemas.task import AgentOutput, BenchmarkResult, FinalReport, TaskRequest
from ..observability.telemetry import TelemetryRecord
from .conflict_detector import ConflictDetector
from .provenance import ProvenanceTracker


AgentResult = Union[AgentOutput, ErrorOutput]


class SynthesisEngine:
    def __init__(self) -> None:
        self.conflict_detector = ConflictDetector()

    def synthesize(
        self,
        query: str,
        results: list[AgentResult],
        benchmark_seq: BenchmarkResult | None = None,
        benchmark_par: BenchmarkResult | None = None,
        telemetry: list[TelemetryRecord] | None = None,
    ) -> FinalReport:
        t0 = time.monotonic()

        # Separate successful outputs from errors
        successes: list[AgentOutput] = []
        errors: list[ErrorOutput] = []
        for r in results:
            if isinstance(r, AgentOutput):
                successes.append(r)
            else:
                errors.append(r)

        # Collect all findings (include partial results from errors at lower confidence)
        all_findings: list[FindingSchema] = []
        for output in successes:
            all_findings.extend(output.findings)
        for err in errors:
            for partial in err.error.partial_results:
                # Mark partial results with lower confidence
                adjusted = partial.model_copy(
                    update={"confidence": min(partial.confidence, 0.5)}
                )
                all_findings.append(adjusted)

        # Detect conflicts
        well_established, contested = self.conflict_detector.detect(all_findings)

        # Build synthesis paragraphs with inline citations + track provenance
        tracker = ProvenanceTracker()
        paragraphs: list[str] = []

        # Well-established section
        if well_established:
            para_parts: list[str] = []
            finding_ids_in_para: list[str] = []
            for f in well_established:
                pub_date = f.source.publication_date.strftime("%b %Y")
                citation = f"{f.source.publisher} ({pub_date})"
                para_parts.append(f"{citation} reports: {f.claim}.")
                finding_ids_in_para.append(f.finding_id)

            para = " ".join(para_parts)
            para_id = f"paragraph_{len(paragraphs) + 1}"
            paragraphs.append(para)
            tracker.register_paragraph(para_id, finding_ids_in_para)

        # Contested section
        for cf in contested:
            parts: list[str] = []
            ids: list[str] = []
            for source_finding in cf.sources:
                pub_date = source_finding.source.publication_date.strftime("%b %Y")
                parts.append(
                    f"{source_finding.source.publisher} ({pub_date}) reports: {source_finding.claim}"
                )
                ids.append(source_finding.finding_id)

            contested_para = (
                f"[CONTESTED — {cf.topic}] "
                + "; while ".join(parts)
                + f". {cf.explanation}"
            )
            para_id = f"paragraph_{len(paragraphs) + 1}"
            paragraphs.append(contested_para)
            tracker.register_paragraph(para_id, ids)

        # Coverage gaps from errors
        coverage_gaps: list[str] = []
        for err in errors:
            agent = err.error.agent
            ftype = err.error.failure_type.value
            query_used = err.error.attempted_query[:80]
            coverage_gaps.append(
                f"Coverage Gap: {agent} encountered {ftype} while researching "
                f'"{query_used}". Data from this source is incomplete or missing.'
            )

        # Validate provenance — hard invariant
        tracker.validate(well_established)

        synthesis_ms = int((time.monotonic() - t0) * 1000)
        if benchmark_par is not None:
            benchmark_par = benchmark_par.model_copy(update={"synthesis_ms": synthesis_ms})

        return FinalReport(
            query=query,
            well_established=well_established,
            contested=contested,
            coverage_gaps=coverage_gaps,
            synthesis_paragraphs=paragraphs,
            trace_map=tracker.trace_map,
            benchmark_sequential=benchmark_seq,
            benchmark_parallel=benchmark_par,
            telemetry=telemetry or [],
            errors_encountered=errors,
        )
