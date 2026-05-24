"""
Test 2: Parallel execution latency improvement.
Measures that parallel execution is faster than sequential with slow mock agents.
"""
from __future__ import annotations

import asyncio
import time

import pytest

from .conftest import make_task, SAMPLE_QUERY


@pytest.mark.asyncio
async def test_parallel_faster_than_sequential(runner_slow, two_tasks):
    """Parallel wall time must be less than sequential wall time."""
    # Sequential
    t0 = time.monotonic()
    await runner_slow.run_sequential(two_tasks)
    seq_ms = (time.monotonic() - t0) * 1000

    # Parallel (fresh tasks to avoid state reuse)
    fresh_tasks = [make_task(t.agent_name) for t in two_tasks]
    t1 = time.monotonic()
    await runner_slow.run_parallel(fresh_tasks)
    par_ms = (time.monotonic() - t1) * 1000

    assert par_ms < seq_ms, (
        f"Parallel ({par_ms:.0f}ms) was not faster than sequential ({seq_ms:.0f}ms)"
    )


@pytest.mark.asyncio
async def test_latency_improvement_is_measurable(runner_slow, two_tasks):
    """Improvement percentage must be > 0%."""
    fresh_tasks = [make_task(t.agent_name) for t in two_tasks]

    t0 = time.monotonic()
    await runner_slow.run_sequential(two_tasks)
    seq_ms = (time.monotonic() - t0) * 1000

    t1 = time.monotonic()
    await runner_slow.run_parallel(fresh_tasks)
    par_ms = (time.monotonic() - t1) * 1000

    improvement = (seq_ms - par_ms) / seq_ms * 100
    assert improvement > 0, f"Expected positive improvement, got {improvement:.1f}%"


@pytest.mark.asyncio
async def test_parallel_executes_all_tasks(runner_fast, two_tasks):
    """All tasks must produce results in parallel mode."""
    results, timing = await runner_fast.run_parallel(two_tasks)
    assert len(results) == len(two_tasks)


@pytest.mark.asyncio
async def test_sequential_executes_all_tasks(runner_fast, two_tasks):
    """All tasks must produce results in sequential mode."""
    results, timing = await runner_fast.run_sequential(two_tasks)
    assert len(results) == len(two_tasks)


@pytest.mark.asyncio
async def test_parallel_timing_captures_wall_time(runner_slow, two_tasks):
    """Timing dict must include __wall__ key."""
    fresh_tasks = [make_task(t.agent_name) for t in two_tasks]
    _, timing = await runner_slow.run_parallel(fresh_tasks)
    assert "__wall__" in timing
    assert timing["__wall__"] > 0


@pytest.mark.asyncio
async def test_sequential_timing_captures_wall_time(runner_slow, two_tasks):
    """Sequential timing must sum to roughly the sum of individual task times."""
    _, timing = await runner_slow.run_sequential(two_tasks)
    assert "__wall__" in timing
    individual_sum = sum(v for k, v in timing.items() if k != "__wall__")
    # Wall time ≈ sum of individual times in sequential mode (within 50ms tolerance)
    assert abs(timing["__wall__"] - individual_sum) < 50
