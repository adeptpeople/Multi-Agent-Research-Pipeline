from __future__ import annotations

import time
from typing import Union

from ..agents.document_analysis_agent import DocumentAnalysisAgent
from ..agents.web_research_agent import WebResearchAgent
from ..observability.telemetry import TelemetryCollector
from ..orchestration.registry import AgentRegistry
from ..orchestration.task_runner import TaskRunner
from ..schemas.errors import ErrorOutput
from ..schemas.task import AgentOutput, FinalReport, TaskRequest
from ..synthesis.engine import SynthesisEngine
from .task_decomposer import QueryDecomposer

AgentResult = Union[AgentOutput, ErrorOutput]


class CoordinatorAgent:
    """
    Orchestrator for the multi-agent research pipeline.

    Allowed tools: [Task]  — coordinator interacts with subagents ONLY via Task dispatch.
    Context model: EXPLICIT — every subagent prompt is fully self-contained.
                   No agent has access to any shared state outside its TaskRequest.
    """

    def __init__(self) -> None:
        self.telemetry = TelemetryCollector()

        # Build agent registry
        self._registry = AgentRegistry()
        self._registry.register("WebResearchAgent", WebResearchAgent(self.telemetry))
        self._registry.register("DocumentAnalysisAgent", DocumentAnalysisAgent(self.telemetry))

        self._runner = TaskRunner(self._registry)
        self._decomposer = QueryDecomposer()
        self._synthesis = SynthesisEngine()

    async def research(
        self,
        query: str,
        user_context: str = "",
        documents: list[dict] | None = None,
    ) -> FinalReport:
        """
        Main entry point. Accepts a research query and optional document list.
        Returns a fully synthesized FinalReport with provenance, conflicts, and benchmarks.
        """
        documents = documents or []
        has_documents = bool(documents)

        # Step 1: Decompose query into explicit TaskRequests
        tasks: list[TaskRequest] = self._decomposer.decompose(
            query=query,
            user_context=user_context,
            has_documents=has_documents,
        )

        # Inject document content into DocumentAnalysisAgent tasks
        for task in tasks:
            if task.agent_name == "DocumentAnalysisAgent" and documents:
                doc = documents[0]
                task.context["document_content"] = doc.get("content", "")
                task.context["document_name"] = doc.get("name", "uploaded_document")

        # Step 2: Sequential run (benchmark baseline)
        # Clone tasks with fresh IDs for sequential run
        seq_tasks = [
            TaskRequest(
                task_id=f"{t.task_id}-seq",
                agent_name=t.agent_name,
                prompt=t.prompt,
                context={**t.context, "original_query": query},
                timeout_ms=t.timeout_ms,
            )
            for t in tasks
        ]
        t_synth_seq_start = time.monotonic()
        seq_results, seq_timing = await self._runner.run_sequential(seq_tasks)
        synthesis_ms_seq = int((time.monotonic() - t_synth_seq_start) * 1000)

        # Step 3: Parallel run (production path)
        par_results, par_timing = await self._runner.run_parallel(tasks)

        # Step 4: Build benchmark comparison
        t_synth_start = time.monotonic()
        bench_seq, bench_par = TaskRunner.build_benchmark(
            seq_timing=seq_timing,
            par_timing=par_timing,
            tasks=tasks,
            synthesis_ms_seq=synthesis_ms_seq,
            synthesis_ms_par=0,  # will be updated by synthesis engine
        )

        # Step 5: Synthesize parallel results into final report
        report = self._synthesis.synthesize(
            query=query,
            results=par_results,
            benchmark_seq=bench_seq,
            benchmark_par=bench_par,
            telemetry=self.telemetry.records,
        )
        return report

    def register_agent(self, name: str, agent: object) -> None:
        """Extension point: register additional specialized agents."""
        self._registry.register(name, agent)  # type: ignore[arg-type]
