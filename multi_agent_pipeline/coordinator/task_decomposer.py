from __future__ import annotations

import json
import uuid
from typing import Any

import anthropic

from ..config import ANTHROPIC_API_KEY, MODEL_NAME
from ..schemas.task import TaskRequest

_DECOMPOSE_SYSTEM = """\
You are a research coordinator. Given a research question, decompose it into 2-4 \
focused sub-questions. Assign each sub-question to either WebResearchAgent \
(for live web search) or DocumentAnalysisAgent (for uploaded documents).

Return ONLY valid JSON array. No prose, no markdown fences.

SCHEMA:
[
  {
    "agent_name": "WebResearchAgent" | "DocumentAnalysisAgent",
    "sub_question": "focused sub-question",
    "rationale": "why this agent for this question"
  }
]
"""


class QueryDecomposer:
    def __init__(self) -> None:
        self._client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)

    def decompose(
        self,
        query: str,
        user_context: str = "",
        has_documents: bool = False,
    ) -> list[TaskRequest]:
        """Decompose query into TaskRequests with fully explicit context payloads."""
        prompt = (
            f"Research Question: {query}\n"
            f"User Context: {user_context}\n"
            f"Has uploaded documents: {has_documents}"
        )
        response = self._client.messages.create(
            model=MODEL_NAME,
            max_tokens=1024,
            system=_DECOMPOSE_SYSTEM,
            messages=[{"role": "user", "content": prompt}],
        )
        raw = "".join(b.text for b in response.content if hasattr(b, "text")).strip()
        if raw.startswith("```"):
            raw = raw.split("```", 2)[1]
            if raw.startswith("json"):
                raw = raw[4:]
            raw = raw.strip()

        subtasks = json.loads(raw)

        requests: list[TaskRequest] = []
        for i, subtask in enumerate(subtasks):
            task_id = f"task-{uuid.uuid4().hex[:8]}"
            agent_name = subtask["agent_name"]
            sub_question = subtask["sub_question"]

            # Explicit context injection — every field needed for autonomous execution
            full_prompt = self._build_agent_prompt(
                original_query=query,
                sub_question=sub_question,
                agent_name=agent_name,
                user_context=user_context,
                task_id=task_id,
            )

            requests.append(
                TaskRequest(
                    task_id=task_id,
                    agent_name=agent_name,
                    prompt=full_prompt,
                    context={
                        "original_query": query,          # Hard guard: must appear in prompt
                        "sub_question": sub_question,
                        "user_context": user_context,
                        "task_index": i,
                        "rationale": subtask.get("rationale", ""),
                    },
                )
            )
        return requests

    @staticmethod
    def _build_agent_prompt(
        original_query: str,
        sub_question: str,
        agent_name: str,
        user_context: str,
        task_id: str,
    ) -> str:
        # All context is explicitly included — agent has NO access to anything outside this string.
        return f"""Research Question (original): {original_query}

Sub-Question (your focus): {sub_question}

Task ID: {task_id}
Agent Role: {agent_name}

User Context: {user_context if user_context else "General research — prioritize accuracy and recency."}

Required Output:
- Return structured JSON findings array only.
- Every finding MUST include: claim, evidence_excerpt, source URL or document name,
  publisher, publication_date (YYYY-MM-DD), confidence (0.0-1.0).
- Preferred sources: government agencies, standards bodies, research institutions,
  reputable news organizations, academic journals.
- Time Range: Prioritize sources published within the last 12 months.
- Do NOT include claims without supporting evidence excerpts.
- Do NOT return prose — only the JSON array.
"""
