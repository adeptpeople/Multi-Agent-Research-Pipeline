from __future__ import annotations

from datetime import date
from typing import Literal, Optional

from pydantic import BaseModel, Field


class SourceSchema(BaseModel):
    type: Literal["url", "document"]
    location: str  # URL or file path
    document_name: Optional[str] = None
    publisher: str
    publication_date: date
    credibility_score: float = Field(ge=0.0, le=1.0)


class FindingSchema(BaseModel):
    finding_id: str  # format: "task-<id>.finding-<n>"
    claim: str = Field(min_length=1)
    evidence_excerpt: str = Field(min_length=1)
    source: SourceSchema
    confidence: float = Field(ge=0.0, le=1.0)

    def validate_provenance(self) -> bool:
        return bool(
            self.claim
            and self.evidence_excerpt
            and self.source
            and (self.source.location or self.source.document_name)
        )


class ContestedFinding(BaseModel):
    topic: str
    sources: list[FindingSchema]  # All conflicting findings — never collapsed
    explanation: str              # e.g. "Different sampling periods and methodologies"
