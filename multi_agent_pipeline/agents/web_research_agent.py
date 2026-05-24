from __future__ import annotations

import json
import time
import uuid
from datetime import date, datetime

import anthropic

from ..config import ANTHROPIC_API_KEY, CREDIBILITY_BY_DOMAIN, MODEL_NAME
from ..observability.telemetry import TelemetryCollector
from ..schemas.finding import FindingSchema, SourceSchema
from ..schemas.task import AgentOutput, TaskRequest
from .base_agent import BaseAgent, SchemaValidationError

_SYSTEM_PROMPT = """\
You are an authoritative web research agent. Your task is to research a given question \
and return structured findings with full provenance metadata.

RULES:
- Only cite authoritative sources: government agencies (.gov), standards bodies, \
  research institutions (.edu), major journals, reputable news (Reuters, BBC, AP).
- Every finding MUST include an exact excerpt from the source.
- Every finding MUST include the publisher name and publication date.
- Prefer sources published within the last 12 months.
- Return ONLY valid JSON matching the schema below. No prose, no markdown fences.

OUTPUT SCHEMA (JSON array of findings):
[
  {
    "claim": "concise factual claim",
    "evidence_excerpt": "exact quote or close paraphrase from source",
    "url": "https://...",
    "publisher": "Publisher Name",
    "publication_date": "YYYY-MM-DD",
    "confidence": 0.0-1.0
  }
]
"""


def _credibility_for_url(url: str) -> float:
    lower = url.lower()
    for key, score in CREDIBILITY_BY_DOMAIN.items():
        if key in lower:
            return score
    return CREDIBILITY_BY_DOMAIN["default"]


class WebResearchAgent(BaseAgent):
    name = "WebResearchAgent"

    def __init__(self, telemetry: TelemetryCollector | None = None) -> None:
        super().__init__(telemetry)
        self._client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)

    async def _run(self, task: TaskRequest) -> AgentOutput:
        t0 = time.monotonic()
        query = task.context.get("original_query", task.prompt)

        response = self._client.messages.create(
            model=MODEL_NAME,
            max_tokens=4096,
            system=_SYSTEM_PROMPT,
            tools=[{"type": "web_search_20250305", "name": "web_search"}],
            messages=[{"role": "user", "content": task.prompt}],
        )

        token_usage = response.usage.input_tokens + response.usage.output_tokens
        raw_text = self._extract_text(response)
        findings = self._parse_findings(raw_text, task.task_id)
        elapsed = int((time.monotonic() - t0) * 1000)

        return AgentOutput(
            agent_name=self.name,
            task_id=task.task_id,
            findings=findings,
            metadata={
                "execution_time_ms": elapsed,
                "query_used": query,
                "token_usage": token_usage,
            },
        )

    def _extract_text(self, response: anthropic.types.Message) -> str:
        parts: list[str] = []
        for block in response.content:
            if hasattr(block, "text"):
                parts.append(block.text)
        return "\n".join(parts)

    def _parse_findings(self, raw: str, task_id: str) -> list[FindingSchema]:
        raw = raw.strip()
        # Strip markdown code fences if present
        if raw.startswith("```"):
            raw = raw.split("```", 2)[1]
            if raw.startswith("json"):
                raw = raw[4:]
        raw = raw.strip()

        try:
            data = json.loads(raw)
        except json.JSONDecodeError:
            # Attempt to extract a JSON array from mixed text
            start = raw.find("[")
            end = raw.rfind("]") + 1
            if start == -1 or end == 0:
                raise SchemaValidationError(f"Could not parse JSON from response: {raw[:200]}")
            data = json.loads(raw[start:end])

        findings = []
        for i, item in enumerate(data):
            pub_date_raw = item.get("publication_date", "2025-01-01")
            try:
                pub_date = date.fromisoformat(pub_date_raw)
            except (ValueError, TypeError):
                pub_date = date(2025, 1, 1)

            url = item.get("url", "https://unknown.example.com")
            findings.append(
                FindingSchema(
                    finding_id=f"{task_id}.finding-{i + 1}",
                    claim=item["claim"],
                    evidence_excerpt=item["evidence_excerpt"],
                    source=SourceSchema(
                        type="url",
                        location=url,
                        publisher=item.get("publisher", "Unknown"),
                        publication_date=pub_date,
                        credibility_score=_credibility_for_url(url),
                    ),
                    confidence=float(item.get("confidence", 0.75)),
                )
            )
        return findings
