# Multi-Agent Research Pipeline

A production-grade multi-agent research system built on the Anthropic Claude API. Accepts a research query, fans it out to specialized subagents in parallel, detects conflicts between sources, and returns a fully synthesized report with provenance tracing and performance benchmarks.

## Architecture

```
User Query ──► CoordinatorAgent
                    │
                    ▼  [Task Tool Only — Explicit Context]
             QueryDecomposer
                    │
        ┌───────────┴───────────┐
        ▼ (parallel)           ▼ (parallel)
 WebResearchAgent      DocumentAnalysisAgent
 (Claude + web_search) (Claude + doc content)
        │                      │
        └───────────┬──────────┘
                    ▼
             SynthesisEngine
        ┌───────────┼───────────┐
        ▼           ▼           ▼
 ConflictDetector ProvenanceTracker TelemetryCollector
        │           │
        └───────────┘
                    ▼
             FinalReport
 (well_established │ contested │ gaps │ trace_map)
```

**Design invariants:**
- The coordinator communicates with subagents via `Task` dispatch only — no shared state.
- Every subagent prompt is fully self-contained (explicit context model).
- Conflicts between sources are preserved, never averaged.

## Features

- **Parallel execution** — web research and document analysis run concurrently via `asyncio.gather`.
- **Conflict detection** — numeric divergence >5% between sources is flagged as contested, not silently merged.
- **Provenance tracing** — every claim in the synthesis is mapped back to its source finding.
- **Fault tolerance** — agent timeouts result in partial continuation; the pipeline synthesizes whatever succeeded.
- **Benchmarking** — sequential vs. parallel latency is measured and reported on every run.
- **Structured telemetry** — per-agent latency, token usage, and retry counts are collected via `TelemetryCollector`.

## Project Structure

```
multi_agent_pipeline/
├── agents/
│   ├── base_agent.py            # Abstract base with retry logic
│   ├── web_research_agent.py    # Claude + web_search tool
│   └── document_analysis_agent.py
├── coordinator/
│   ├── coordinator.py           # Main orchestrator (CoordinatorAgent)
│   └── task_decomposer.py       # Splits query into TaskRequests
├── orchestration/
│   ├── registry.py              # AgentRegistry — name → agent instance
│   └── task_runner.py           # run_parallel / run_sequential + benchmarking
├── synthesis/
│   ├── engine.py                # SynthesisEngine — assembles FinalReport
│   ├── conflict_detector.py     # Numeric + semantic conflict detection
│   └── provenance.py            # Paragraph → finding_id trace map
├── schemas/
│   ├── task.py                  # TaskRequest, AgentOutput, FinalReport
│   ├── finding.py               # FindingSchema, SourceSchema
│   └── errors.py                # ErrorOutput, ErrorDetail, FailureType
├── observability/
│   └── telemetry.py             # TelemetryCollector, TelemetryRecord
├── tests/                       # pytest suite (asyncio)
└── config.py                    # API key, model, timeouts, thresholds
main.py                          # Demo entry point (3 scenarios)
```

## Quick Start

### Prerequisites

- Python 3.11+
- An [Anthropic API key](https://console.anthropic.com/)

### Installation

```bash
pip install -r requirements.txt
```

### Run

```bash
export ANTHROPIC_API_KEY=your_key_here
python main.py
```

Without an API key, demos 2 and 3 (mock agents) still run. Demo 1 (live Claude calls) requires a valid key.

### Run tests

```bash
pytest
```

## Demo Scenarios

`main.py` runs three scenarios in sequence:

| Demo | Description |
|------|-------------|
| 1 — Full pipeline | Live query decomposed into parallel web + document research, synthesized into a report |
| 2 — Failure simulation | `WebResearchAgent` times out; pipeline continues with the document agent's findings |
| 3 — Conflict scenario | Two authoritative sources report different growth figures; both are preserved in the `contested` section |

## Configuration

All tunables live in `multi_agent_pipeline/config.py`:

| Setting | Default | Description |
|---------|---------|-------------|
| `MODEL_NAME` | `claude-sonnet-4-6` | Claude model used by all agents |
| `DEFAULT_TIMEOUT_MS` | `30000` | Per-agent execution timeout |
| `MAX_RETRIES` | `3` | Retry attempts on transient failures |
| `NUMERIC_CONFLICT_THRESHOLD` | `0.05` | Relative difference that triggers a conflict flag |

## Output Schema

`FinalReport` fields:

- `well_established` — findings corroborated across sources (high confidence)
- `contested` — findings where sources disagree; each side is preserved with attribution
- `synthesis_paragraphs` — narrative summary with inline provenance
- `coverage_gaps` — aspects of the query not addressed by any agent
- `trace_map` — paragraph ID → list of finding IDs (full provenance chain)
- `benchmark_sequential` / `benchmark_parallel` — latency comparison (ms)
- `telemetry` — per-agent execution records

## Dependencies

| Package | Purpose |
|---------|---------|
| `anthropic` | Claude API client |
| `pydantic` | Schema validation for all data models |
| `aiohttp` | Async HTTP for web research |
| `structlog` | Structured logging |
| `rich` | Terminal output rendering |
| `pytest` + `pytest-asyncio` | Test suite |
