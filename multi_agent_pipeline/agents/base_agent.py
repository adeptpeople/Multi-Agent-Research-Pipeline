from __future__ import annotations

import asyncio
import time
from abc import ABC, abstractmethod
from typing import Union

from ..config import MAX_RETRIES, RETRY_DELAY_MS
from ..observability.telemetry import TelemetryCollector, TelemetryRecord
from ..schemas.errors import ErrorDetail, ErrorOutput, FailureType
from ..schemas.task import AgentOutput, TaskRequest


class SchemaValidationError(Exception):
    pass


class BaseAgent(ABC):
    name: str = "BaseAgent"

    def __init__(self, telemetry: TelemetryCollector | None = None) -> None:
        self.telemetry = telemetry or TelemetryCollector()

    async def execute(self, task: TaskRequest) -> Union[AgentOutput, ErrorOutput]:
        start_dt = TelemetryCollector.now()
        t0 = time.monotonic()
        retry_count = 0
        last_error: ErrorOutput | None = None

        for attempt in range(MAX_RETRIES):
            try:
                result = await asyncio.wait_for(
                    self._run(task),
                    timeout=task.timeout_ms / 1000.0,
                )
                self._validate_output(result)
                elapsed = int((time.monotonic() - t0) * 1000)
                self.telemetry.record(
                    TelemetryRecord(
                        task_id=task.task_id,
                        agent_name=self.name,
                        start_time=start_dt,
                        end_time=TelemetryCollector.now(),
                        latency_ms=elapsed,
                        status="success",
                        token_usage=result.metadata.get("token_usage", 0),
                        retry_count=retry_count,
                        context_payload_size_bytes=len(task.prompt.encode()),
                    )
                )
                return result

            except asyncio.TimeoutError:
                elapsed = int((time.monotonic() - t0) * 1000)
                last_error = self._make_error(
                    task, FailureType.TIMEOUT, elapsed, retryable=True
                )
                retry_count += 1

            except SchemaValidationError as exc:
                elapsed = int((time.monotonic() - t0) * 1000)
                last_error = self._make_error(
                    task, FailureType.MALFORMED_RESPONSE, elapsed, retryable=False,
                    detail=str(exc),
                )
                break  # schema violations are not retryable

            except Exception as exc:
                elapsed = int((time.monotonic() - t0) * 1000)
                failure_type = self._classify_exception(exc)
                retryable = failure_type in (
                    FailureType.TIMEOUT,
                    FailureType.RATE_LIMIT,
                    FailureType.SOURCE_FETCH_FAILURE,
                )
                last_error = self._make_error(task, failure_type, elapsed, retryable=retryable)
                if not retryable:
                    break
                retry_count += 1

            if retry_count < MAX_RETRIES:
                await asyncio.sleep(RETRY_DELAY_MS / 1000.0)

        elapsed = int((time.monotonic() - t0) * 1000)
        assert last_error is not None
        self.telemetry.record(
            TelemetryRecord(
                task_id=task.task_id,
                agent_name=self.name,
                start_time=start_dt,
                end_time=TelemetryCollector.now(),
                latency_ms=elapsed,
                status="error",
                retry_count=retry_count,
                error_type=last_error.error.failure_type.value,
                context_payload_size_bytes=len(task.prompt.encode()),
            )
        )
        return last_error

    @abstractmethod
    async def _run(self, task: TaskRequest) -> AgentOutput:
        ...

    def _validate_output(self, output: AgentOutput) -> None:
        for finding in output.findings:
            if not finding.validate_provenance():
                raise SchemaValidationError(
                    f"Finding {finding.finding_id} missing required provenance fields"
                )

    def _make_error(
        self,
        task: TaskRequest,
        failure_type: FailureType,
        elapsed_ms: int,
        retryable: bool,
        detail: str = "",
    ) -> ErrorOutput:
        return ErrorOutput(
            agent_name=self.name,
            task_id=task.task_id,
            error=ErrorDetail(
                failure_type=failure_type,
                agent=self.name,
                attempted_query=task.context.get("original_query", task.prompt[:120]),
                partial_results=[],
                retryable=retryable,
                elapsed_ms=elapsed_ms,
            ),
        )

    @staticmethod
    def _classify_exception(exc: Exception) -> FailureType:
        name = type(exc).__name__.lower()
        if "timeout" in name:
            return FailureType.TIMEOUT
        if "json" in name or "decode" in name or "parse" in name:
            return FailureType.MALFORMED_RESPONSE
        if "ratelimit" in name or "rate" in name:
            return FailureType.RATE_LIMIT
        if "connection" in name or "client" in name or "fetch" in name:
            return FailureType.SOURCE_FETCH_FAILURE
        return FailureType.PARSER_EXCEPTION
