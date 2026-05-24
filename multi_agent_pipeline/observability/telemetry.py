from __future__ import annotations

import threading
from datetime import datetime, timezone
from typing import Any, Literal, Optional

from pydantic import BaseModel


class TelemetryRecord(BaseModel):
    task_id: str
    agent_name: str
    start_time: datetime
    end_time: datetime
    latency_ms: int
    status: Literal["success", "error"]
    token_usage: int = 0
    retry_count: int = 0
    error_type: Optional[str] = None
    context_payload_size_bytes: int = 0


class TelemetryCollector:
    def __init__(self) -> None:
        self._records: list[TelemetryRecord] = []
        self._lock = threading.Lock()

    def record(self, r: TelemetryRecord) -> None:
        with self._lock:
            self._records.append(r)

    @property
    def records(self) -> list[TelemetryRecord]:
        with self._lock:
            return list(self._records)

    def summary(self) -> dict[str, Any]:
        recs = self.records
        if not recs:
            return {}
        total = len(recs)
        successes = sum(1 for r in recs if r.status == "success")
        timeouts = sum(1 for r in recs if r.error_type == "timeout")
        latencies = [r.latency_ms for r in recs]
        return {
            "total_tasks": total,
            "success_rate": round(successes / total, 3),
            "failure_rate": round((total - successes) / total, 3),
            "timeout_rate": round(timeouts / total, 3),
            "avg_latency_ms": round(sum(latencies) / len(latencies), 1),
            "max_latency_ms": max(latencies),
            "min_latency_ms": min(latencies),
            "total_tokens": sum(r.token_usage for r in recs),
            "total_retries": sum(r.retry_count for r in recs),
        }

    def provenance_completeness_pct(
        self,
        trace_map: dict[str, list[str]],
        all_finding_ids: list[str],
    ) -> float:
        if not all_finding_ids:
            return 100.0
        referenced = {fid for ids in trace_map.values() for fid in ids}
        covered = sum(1 for fid in all_finding_ids if fid in referenced)
        return round(covered / len(all_finding_ids) * 100, 1)

    @staticmethod
    def now() -> datetime:
        return datetime.now(tz=timezone.utc)
