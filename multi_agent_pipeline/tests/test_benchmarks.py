"""
Benchmark test: Sequential vs Parallel latency table.
Produces a printable benchmark summary.
"""
from __future__ import annotations

import asyncio
import time

import pytest

from multi_agent_pipeline.orchestration.task_runner import TaskRunner
from multi_agent_pipeline.synthesis.engine import SynthesisEngine

from .conftest import make_task, SlowWebAgent, SlowDocAgent, SAMPLE_QUERY
from multi_agent_pipeline.orchestration.registry import AgentRegistry
from multi_agent_pipeline.observability.telemetry import TelemetryCollector


@pytest.mark.asyncio
async def test_benchmark_table_sequential_vs_parallel():
    """
    Runs both modes and prints the benchmark table.
    Validates that parallel is faster and improvement is computable.
    """
    telemetry = TelemetryCollector()
    registry = AgentRegistry()
    registry.register("WebResearchAgent", SlowWebAgent(telemetry=telemetry))
    registry.register("DocumentAnalysisAgent", SlowDocAgent(telemetry=telemetry))
    runner = TaskRunner(registry)
    engine = SynthesisEngine()

    tasks_seq = [make_task("WebResearchAgent"), make_task("DocumentAnalysisAgent")]
    tasks_par = [make_task("WebResearchAgent"), make_task("DocumentAnalysisAgent")]

    # Sequential run
    t0 = time.monotonic()
    seq_results, seq_timing = await runner.run_sequential(tasks_seq)
    synth_t0 = time.monotonic()
    engine.synthesize(SAMPLE_QUERY, seq_results)
    synthesis_ms_seq = (time.monotonic() - synth_t0) * 1000
    total_seq_ms = (time.monotonic() - t0) * 1000

    # Parallel run
    t1 = time.monotonic()
    par_results, par_timing = await runner.run_parallel(tasks_par)
    synth_t1 = time.monotonic()
    engine.synthesize(SAMPLE_QUERY, par_results)
    synthesis_ms_par = (time.monotonic() - synth_t1) * 1000
    total_par_ms = (time.monotonic() - t1) * 1000

    bench_seq, bench_par = TaskRunner.build_benchmark(
        seq_timing=seq_timing,
        par_timing=par_timing,
        tasks=tasks_par,
        synthesis_ms_seq=synthesis_ms_seq,
        synthesis_ms_par=synthesis_ms_par,
    )

    improvement = (total_seq_ms - total_par_ms) / total_seq_ms * 100

    # Print benchmark table
    print("\n")
    print("=" * 70)
    print(f"{'BENCHMARK TABLE':^70}")
    print("=" * 70)
    header = f"{'Mode':<14} {'Web (ms)':>10} {'Doc (ms)':>10} {'Synth (ms)':>12} {'Total (ms)':>12}"
    print(header)
    print("-" * 70)
    print(
        f"{'Sequential':<14} {seq_timing.get(tasks_seq[0].task_id, 0):>10.0f} "
        f"{seq_timing.get(tasks_seq[1].task_id, 0):>10.0f} "
        f"{synthesis_ms_seq:>12.0f} {total_seq_ms:>12.0f}"
    )
    print(
        f"{'Parallel':<14} {par_timing.get(tasks_par[0].task_id, 0):>10.0f} "
        f"{par_timing.get(tasks_par[1].task_id, 0):>10.0f} "
        f"{synthesis_ms_par:>12.0f} {total_par_ms:>12.0f}"
    )
    print("-" * 70)
    print(f"{'Improvement':^14} {improvement:>10.1f}%")
    print("=" * 70)

    # Assertions
    assert total_par_ms < total_seq_ms, "Parallel must be faster than sequential"
    assert improvement > 0, f"Expected positive improvement, got {improvement:.1f}%"
    assert bench_par.latency_improvement_pct >= 0
