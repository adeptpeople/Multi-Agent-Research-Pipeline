from __future__ import annotations

import json
import time
from datetime import date

import anthropic

from ..config import ANTHROPIC_API_KEY, MODEL_NAME
from ..observability.telemetry import TelemetryCollector
from ..schemas.finding import FindingSchema, SourceSchema
from ..schemas.task import AgentOutput, TaskRequest
from .base_agent import BaseAgent, SchemaValidationError

_SYSTEM_PROMPT = """\
You are a document analysis agent. You extract factual claims from provided documents \
and return structured findings with full provenance metadata.

RULES:
- Extract only explicit claims present in the document — do not infer or extrapolate.
- For each claim, include a verbatim or near-verbatim excerpt from the document.
- Include the document name and, if available, the publication date from the document.
- Return ONLY valid JSON matching the schema below. No prose, no markdown fences.

OUTPUT SCHEMA (JSON array of findings):
[
  {
    "claim": "concise factual claim extracted from document",
    "evidence_excerpt": "verbatim or close excerpt from the document",
    "document_name": "filename or title",
    "publisher": "author or organization",
    "publication_date": "YYYY-MM-DD or best estimate",
    "confidence": 0.0-1.0
  }
]
"""


class DocumentAnalysisAgent(BaseAgent):
    name = "DocumentAnalysisAgent"

    def __init__(self, telemetry: TelemetryCollector | None = None) -> None:
        super().__init__(telemetry)
        self._client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)

    async def _run(self, task: TaskRequest) -> AgentOutput:
        t0 = time.monotonic()
        query = task.context.get("original_query", task.prompt)
        doc_content = task.context.get("document_content", "")
        doc_name = task.context.get("document_name", "uploaded_document")

        full_prompt = (
            f"{task.prompt}\n\n"
            f"--- DOCUMENT: {doc_name} ---\n"
            f"{doc_content}\n"
            f"--- END DOCUMENT ---"
        )

        response = self._client.messages.create(
            model=MODEL_NAME,
            max_tokens=4096,
            system=_SYSTEM_PROMPT,
            messages=[{"role": "user", "content": full_prompt}],
        )

        token_usage = response.usage.input_tokens + response.usage.output_tokens
        raw_text = "".join(
            block.text for block in response.content if hasattr(block, "text")
        )
        findings = self._parse_findings(raw_text, task.task_id, doc_name)
        elapsed = int((time.monotonic() - t0) * 1000)

        return AgentOutput(
            agent_name=self.name,
            task_id=task.task_id,
            findings=findings,
            metadata={
                "execution_time_ms": elapsed,
                "query_used": query,
                "token_usage": token_usage,
                "document_name": doc_name,
            },
        )

    def _parse_findings(
        self, raw: str, task_id: str, doc_name: str
    ) -> list[FindingSchema]:
        raw = raw.strip()
        if raw.startswith("```"):
            raw = raw.split("```", 2)[1]
            if raw.startswith("json"):
                raw = raw[4:]
        raw = raw.strip()

        try:
            data = json.loads(raw)
        except json.JSONDecodeError:
            start = raw.find("[")
            end = raw.rfind("]") + 1
            if start == -1 or end == 0:
                raise SchemaValidationError(
                    f"Could not parse JSON from document analysis response: {raw[:200]}"
                )
            data = json.loads(raw[start:end])

        findings = []
        for i, item in enumerate(data):
            pub_date_raw = item.get("publication_date", "2025-01-01")
            try:
                pub_date = date.fromisoformat(pub_date_raw)
            except (ValueError, TypeError):
                pub_date = date(2025, 1, 1)

            findings.append(
                FindingSchema(
                    finding_id=f"{task_id}.finding-{i + 1}",
                    claim=item["claim"],
                    evidence_excerpt=item["evidence_excerpt"],
                    source=SourceSchema(
                        type="document",
                        location=item.get("document_name", doc_name),
                        document_name=item.get("document_name", doc_name),
                        publisher=item.get("publisher", "Unknown"),
                        publication_date=pub_date,
                        credibility_score=0.80,
                    ),
                    confidence=float(item.get("confidence", 0.80)),
                )
            )
        return findings
