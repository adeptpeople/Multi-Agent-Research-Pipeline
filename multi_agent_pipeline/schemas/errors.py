from __future__ import annotations

from enum import Enum
from typing import Literal

from pydantic import BaseModel

from .finding import FindingSchema


class FailureType(str, Enum):
    TIMEOUT = "timeout"
    RATE_LIMIT = "rate_limit"
    MALFORMED_RESPONSE = "malformed_response"
    TOOL_UNAVAILABLE = "tool_unavailable"
    SOURCE_FETCH_FAILURE = "source_fetch_failure"
    PARSER_EXCEPTION = "parser_exception"


class ErrorDetail(BaseModel):
    failure_type: FailureType
    agent: str
    attempted_query: str
    partial_results: list[FindingSchema] = []
    retryable: bool
    elapsed_ms: int


class ErrorOutput(BaseModel):
    agent_name: str
    task_id: str
    status: Literal["error"] = "error"
    error: ErrorDetail
