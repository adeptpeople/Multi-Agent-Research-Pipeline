from __future__ import annotations

import asyncio
import time
from typing import Union

from ..schemas.errors import ErrorDetail, ErrorOutput, FailureType
from ..schemas.task import AgentOutput, BenchmarkResult, TaskRequest
from .registry import AgentRegistry

AgentResult = Union[AgentOutput, ErrorOutput]


class TaskRunner:
    def __init__(self, registry: AgentRegistry) -> None:
        self.registry = registry

    async def run_parallel(
        self, tasks: list[TaskRequest]
    ) -> tuple[list[AgentResult], dict[str, float]]:
        """Execute all tasks concurrently. Returns results + per-task timing."""
        timing: dict[str, float] = {}

        async def _timed(task: TaskRequest) -> AgentResult:
            t0 = time.monotonic()
            result = await self._dispatch(task)
            timing[task.task_id] = (time.monotonic() - t0) * 1000
            return result

        t_wall_start = time.monotonic()
        raw = await asyncio.gather(*[_timed(t) for t in tasks], return_exceptions=True)
        timing["__wall__"] = (time.monotonic() - t_wall_start) * 1000

        return [self._normalize(r, task) for r, task in zip(raw, tasks)], timing

    async def run_sequential(
        self, tasks: list[TaskRequest]
    ) -> tuple[list[AgentResult], dict[str, float]]:
        """Execute tasks one at a time. Returns results + per-task timing."""
        results: list[AgentResult] = []
        timing: dict[str, float] = {}
        t_wall_start = time.monotonic()

        for task in tasks:
            t0 = time.monotonic()
            result = await self._dispatch(task)
            timing[task.task_id] = (time.monotonic() - t0) * 1000
            results.append(result)

        timing["__wall__"] = (time.monotonic() - t_wall_start) * 1000
        return results, timing

    async def _dispatch(self, task: TaskRequest) -> AgentResult:
        agent = self.registry.get(task.agent_name)
        return await agent.execute(task)

    @staticmethod
    def _normalize(result: object, task: TaskRequest) -> AgentResult:
        if isinstance(result, (AgentOutput, ErrorOutput)):
            return result
        # asyncio.gather returned a raw exception
        exc = result
        elapsed_ms = int(task.timeout_ms)
        failure_type = (
            FailureType.TIMEOUT
            if isinstance(exc, asyncio.TimeoutError)
            else FailureType.PARSER_EXCEPTION
        )
        return ErrorOutput(
            agent_name=task.agent_name,
            task_id=task.task_id,
            error=ErrorDetail(
                failure_type=failure_type,
                agent=task.agent_name,
                attempted_query=task.context.get("original_query", ""),
                partial_results=[],
                retryable=failure_type == FailureType.TIMEOUT,
                elapsed_ms=elapsed_ms,
            ),
        )

    @staticmethod
    def build_benchmark(
        seq_timing: dict[str, float],
        par_timing: dict[str, float],
        tasks: list[TaskRequest],
        synthesis_ms_seq: float,
        synthesis_ms_par: float,
    ) -> tuple[BenchmarkResult, BenchmarkResult]:
        web_task_id = next((t.task_id for t in tasks if "Web" in t.agent_name), None)
        doc_task_id = next((t.task_id for t in tasks if "Document" in t.agent_name), None)

        seq = BenchmarkResult(
            mode="sequential",
            web_agent_ms=seq_timing.get(web_task_id or "", 0.0),
            doc_agent_ms=seq_timing.get(doc_task_id or "", 0.0),
            synthesis_ms=synthesis_ms_seq,
            total_ms=seq_timing["__wall__"] + synthesis_ms_seq,
        )
        par_total = par_timing["__wall__"] + synthesis_ms_par
        seq_total = seq.total_ms
        improvement = ((seq_total - par_total) / seq_total * 100) if seq_total > 0 else 0.0

        par = BenchmarkResult(
            mode="parallel",
            web_agent_ms=par_timing.get(web_task_id or "", 0.0),
            doc_agent_ms=par_timing.get(doc_task_id or "", 0.0),
            synthesis_ms=synthesis_ms_par,
            total_ms=par_total,
            latency_improvement_pct=round(improvement, 1),
        )
        return seq, par
