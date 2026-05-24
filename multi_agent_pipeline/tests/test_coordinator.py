"""
Test 1: Coordinator explicit context passing.
Validates that every TaskRequest prompt contains the original query
and that no hidden context inheritance is possible.
"""
from __future__ import annotations

import pytest

from multi_agent_pipeline.schemas.task import TaskRequest
from .conftest import SAMPLE_QUERY, SAMPLE_CONTEXT, make_task


class TestExplicitContextPassing:
    def test_task_request_prompt_must_contain_original_query(self):
        """Pydantic validator rejects prompts that don't include the original query."""
        with pytest.raises(ValueError, match="Explicit context violation"):
            TaskRequest(
                task_id="task-001",
                agent_name="WebResearchAgent",
                prompt="Search for AI chips.",  # missing original_query
                context={"original_query": SAMPLE_QUERY},
            )

    def test_valid_task_request_contains_query(self):
        """Valid TaskRequest passes when prompt explicitly contains the query."""
        task = make_task("WebResearchAgent", query=SAMPLE_QUERY)
        assert SAMPLE_QUERY in task.prompt

    def test_all_required_context_fields_present(self):
        """TaskRequest context must have original_query key."""
        task = make_task("WebResearchAgent", query=SAMPLE_QUERY)
        assert "original_query" in task.context
        assert task.context["original_query"] == SAMPLE_QUERY

    def test_no_hidden_state_between_tasks(self):
        """Two TaskRequests for different agents are fully independent."""
        task_web = make_task("WebResearchAgent", query=SAMPLE_QUERY)
        task_doc = make_task("DocumentAnalysisAgent", query=SAMPLE_QUERY)

        # Different task IDs — no shared mutable state
        assert task_web.task_id != task_doc.task_id
        # Each prompt is self-contained
        assert SAMPLE_QUERY in task_web.prompt
        assert SAMPLE_QUERY in task_doc.prompt
        # Context dicts are separate objects
        assert task_web.context is not task_doc.context

    def test_empty_context_no_query_passes_without_validation(self):
        """If context has no original_query, validator skips (allows legacy tasks)."""
        task = TaskRequest(
            task_id="task-legacy",
            agent_name="WebResearchAgent",
            prompt="Research AI exports.",
            context={},  # no original_query key → validator skips
        )
        assert task.task_id == "task-legacy"

    def test_prompt_payload_completeness(self):
        """Prompt must include all structural sections expected by agents."""
        task = make_task("WebResearchAgent", query=SAMPLE_QUERY)
        # Must contain the query
        assert SAMPLE_QUERY in task.prompt
        # Must name the agent role
        assert "WebResearchAgent" in task.prompt

    def test_context_query_matches_prompt_query(self):
        """Structural integrity: context.original_query == what's embedded in prompt."""
        task = make_task("DocumentAnalysisAgent", query=SAMPLE_QUERY)
        assert task.context["original_query"] in task.prompt
