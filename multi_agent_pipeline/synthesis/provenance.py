from __future__ import annotations

from ..schemas.finding import FindingSchema


class ProvenanceLossError(Exception):
    """Raised when a synthesized report cannot trace all findings."""


class ProvenanceTracker:
    def __init__(self) -> None:
        self._trace_map: dict[str, list[str]] = {}

    def register_paragraph(self, paragraph_id: str, finding_ids: list[str]) -> None:
        self._trace_map[paragraph_id] = list(finding_ids)

    @property
    def trace_map(self) -> dict[str, list[str]]:
        return dict(self._trace_map)

    def validate(self, well_established: list[FindingSchema]) -> None:
        """Raise ProvenanceLossError if any well-established finding has no trace."""
        referenced = {fid for ids in self._trace_map.values() for fid in ids}
        orphaned = [f.finding_id for f in well_established if f.finding_id not in referenced]
        if orphaned:
            raise ProvenanceLossError(
                f"Provenance loss detected. Findings not traceable in trace_map: {orphaned}"
            )

    def completeness_pct(self, all_findings: list[FindingSchema]) -> float:
        if not all_findings:
            return 100.0
        referenced = {fid for ids in self._trace_map.values() for fid in ids}
        covered = sum(1 for f in all_findings if f.finding_id in referenced)
        return round(covered / len(all_findings) * 100, 1)
