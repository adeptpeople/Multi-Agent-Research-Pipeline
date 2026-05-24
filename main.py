"""
Production Multi-Agent Research Pipeline — Demo Entry Point

Three demonstration scenarios:
  1. Full pipeline   — parallel web + document research
  2. Failure sim     — timeout mid-run, partial continuation
  3. Conflict demo   — contradictory sources, contested findings section
"""
from __future__ import annotations

import asyncio
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))

from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich import box

from multi_agent_pipeline.coordinator.coordinator import CoordinatorAgent
from multi_agent_pipeline.schemas.task import FinalReport

console = Console()


# ---------------------------------------------------------------------------
# ASCII Architecture Diagram
# ---------------------------------------------------------------------------

ARCH_DIAGRAM = """
┌─────────────────────────────────────────────────────────────────┐
│              MULTI-AGENT RESEARCH PIPELINE ARCHITECTURE         │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│   User Query ──► CoordinatorAgent                               │
│                      │                                          │
│                      ▼  [Task Tool Only — Explicit Context]     │
│               QueryDecomposer                                   │
│                      │                                          │
│          ┌───────────┴───────────┐                              │
│          ▼ (parallel)           ▼ (parallel)                   │
│   WebResearchAgent      DocumentAnalysisAgent                   │
│   (Claude + web_search) (Claude + doc content)                 │
│          │                      │                               │
│          └───────────┬──────────┘                               │
│                      ▼                                          │
│               SynthesisEngine                                   │
│          ┌───────────┼───────────┐                              │
│          ▼           ▼           ▼                              │
│   ConflictDetector ProvenanceTracker TelemetryCollector         │
│          │           │                                          │
│          └───────────┘                                          │
│                      ▼                                          │
│               FinalReport                                       │
│   (well_established │ contested │ gaps │ trace_map)            │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
"""

SEQUENCE_DIAGRAM = """
Sequence: Parallel Execution Flow
──────────────────────────────────────────────────────────────────
User            Coordinator         WebAgent        DocAgent
 │                   │                  │               │
 │──research(Q)──►   │                  │               │
 │                   │──decompose(Q)──► QueryDecomposer │
 │                   │◄── [tasks] ──────│               │
 │                   │                  │               │
 │                   │──Task(web_task)──►               │
 │                   │──Task(doc_task)──────────────────►
 │                   │   [parallel]     │               │
 │                   │◄─ AgentOutput ───│               │
 │                   │◄─────────────────── AgentOutput ─│
 │                   │                  │               │
 │                   │──synthesize()──► SynthesisEngine  │
 │                   │◄── FinalReport ──│               │
 │◄── FinalReport ───│                  │               │
──────────────────────────────────────────────────────────────────
"""


# ---------------------------------------------------------------------------
# Rendering helpers
# ---------------------------------------------------------------------------

def print_report(report: FinalReport, title: str) -> None:
    console.print(Panel(f"[bold cyan]{title}[/bold cyan]"))

    # Well-established findings
    if report.well_established:
        console.print("\n[bold green]WELL-ESTABLISHED FINDINGS[/bold green]")
        for f in report.well_established:
            pub = f.source.publication_date.strftime("%b %Y")
            console.print(
                f"  • [{f.source.publisher}, {pub}] {f.claim} "
                f"(confidence: {f.confidence:.0%})"
            )

    # Contested findings
    if report.contested:
        console.print("\n[bold yellow]CONTESTED FINDINGS[/bold yellow]")
        for cf in report.contested:
            console.print(f"  Topic: {cf.topic}")
            for s in cf.sources:
                pub = s.source.publication_date.strftime("%b %Y")
                console.print(f"    - {s.source.publisher} ({pub}): {s.claim}")
            console.print(f"    Note: {cf.explanation}")

    # Synthesis paragraphs
    if report.synthesis_paragraphs:
        console.print("\n[bold blue]SYNTHESIS[/bold blue]")
        for i, para in enumerate(report.synthesis_paragraphs, 1):
            console.print(f"  [{i}] {para}")

    # Coverage gaps
    if report.coverage_gaps:
        console.print("\n[bold red]COVERAGE GAPS[/bold red]")
        for gap in report.coverage_gaps:
            console.print(f"  ⚠ {gap}")

    # Provenance trace map
    if report.trace_map:
        console.print("\n[bold magenta]PROVENANCE TRACE MAP[/bold magenta]")
        for para_id, finding_ids in report.trace_map.items():
            console.print(f"  {para_id} → {finding_ids}")

    # Benchmark table
    if report.benchmark_sequential and report.benchmark_parallel:
        table = Table(title="Benchmark: Sequential vs Parallel", box=box.SIMPLE_HEAVY)
        table.add_column("Mode", style="bold")
        table.add_column("Web Agent (ms)")
        table.add_column("Doc Agent (ms)")
        table.add_column("Synthesis (ms)")
        table.add_column("Total (ms)")
        table.add_column("Improvement")

        seq = report.benchmark_sequential
        par = report.benchmark_parallel
        table.add_row(
            "Sequential",
            f"{seq.web_agent_ms:.0f}",
            f"{seq.doc_agent_ms:.0f}",
            f"{seq.synthesis_ms:.0f}",
            f"{seq.total_ms:.0f}",
            "—",
        )
        table.add_row(
            "Parallel",
            f"{par.web_agent_ms:.0f}",
            f"{par.doc_agent_ms:.0f}",
            f"{par.synthesis_ms:.0f}",
            f"{par.total_ms:.0f}",
            f"[green]{par.latency_improvement_pct:.1f}%[/green]",
        )
        console.print(table)

    # Telemetry summary
    if report.telemetry:
        console.print("\n[bold]TELEMETRY RECORDS[/bold]")
        for rec in report.telemetry:
            status_color = "green" if rec.status == "success" else "red"
            console.print(
                f"  [{status_color}]{rec.agent_name}[/{status_color}] "
                f"task={rec.task_id} latency={rec.latency_ms}ms "
                f"tokens={rec.token_usage} retries={rec.retry_count}"
            )


# ---------------------------------------------------------------------------
# Demo 1: Full pipeline
# ---------------------------------------------------------------------------

async def demo_full_pipeline() -> None:
    console.rule("[bold cyan]DEMO 1: Full Pipeline — Parallel Research[/bold cyan]")
    console.print(ARCH_DIAGRAM)
    console.print(SEQUENCE_DIAGRAM)

    coordinator = CoordinatorAgent()
    report = await coordinator.research(
        query="What are the latest AI chip export restrictions and their impact on semiconductor markets?",
        user_context=(
            "Comparative regulatory analysis. "
            "Focus on US, EU, and Asia-Pacific regulatory developments from 2024-2025."
        ),
    )
    print_report(report, "Full Pipeline Research Report")


# ---------------------------------------------------------------------------
# Demo 2: Failure simulation
# ---------------------------------------------------------------------------

async def demo_failure_simulation() -> None:
    console.rule("[bold red]DEMO 2: Failure Simulation — Timeout + Partial Continuation[/bold red]")

    import asyncio
    import uuid
    from datetime import date
    from multi_agent_pipeline.agents.base_agent import BaseAgent
    from multi_agent_pipeline.schemas.task import AgentOutput, TaskRequest
    from multi_agent_pipeline.schemas.finding import FindingSchema, SourceSchema
    from multi_agent_pipeline.schemas.errors import ErrorDetail, ErrorOutput, FailureType
    from multi_agent_pipeline.orchestration.registry import AgentRegistry
    from multi_agent_pipeline.orchestration.task_runner import TaskRunner
    from multi_agent_pipeline.synthesis.engine import SynthesisEngine

    # A mock web agent that times out
    class TimeoutWebAgent(BaseAgent):
        name = "WebResearchAgent"
        async def _run(self, task: TaskRequest) -> AgentOutput:
            await asyncio.sleep(100)  # will be cancelled by timeout
            raise asyncio.TimeoutError()

    # A mock doc agent that succeeds with one finding
    class SuccessDocAgent(BaseAgent):
        name = "DocumentAnalysisAgent"
        async def _run(self, task: TaskRequest) -> AgentOutput:
            return AgentOutput(
                agent_name=self.name,
                task_id=task.task_id,
                findings=[
                    FindingSchema(
                        finding_id=f"{task.task_id}.finding-1",
                        claim="EU enacted the AI Act with provisions restricting high-risk AI chip exports.",
                        evidence_excerpt="The AI Act (2024) establishes export control frameworks for dual-use AI hardware.",
                        source=SourceSchema(
                            type="document",
                            location="eu_ai_act_summary.pdf",
                            document_name="eu_ai_act_summary.pdf",
                            publisher="European Commission",
                            publication_date=date(2024, 8, 1),
                            credibility_score=0.95,
                        ),
                        confidence=0.92,
                    )
                ],
                metadata={"execution_time_ms": 120, "query_used": task.context.get("original_query", "")},
            )

    registry = AgentRegistry()
    registry.register("WebResearchAgent", TimeoutWebAgent())
    registry.register("DocumentAnalysisAgent", SuccessDocAgent())

    runner = TaskRunner(registry)
    task_id_web = f"task-{uuid.uuid4().hex[:8]}"
    task_id_doc = f"task-{uuid.uuid4().hex[:8]}"
    query = "What are the latest AI chip export restrictions and their impact on semiconductor markets?"

    tasks = [
        TaskRequest(
            task_id=task_id_web,
            agent_name="WebResearchAgent",
            prompt=f"Research Question (original): {query}\n\nSub-Question: US chip export ban details 2025",
            context={"original_query": query},
            timeout_ms=500,
        ),
        TaskRequest(
            task_id=task_id_doc,
            agent_name="DocumentAnalysisAgent",
            prompt=f"Research Question (original): {query}\n\nSub-Question: EU AI Act chip provisions",
            context={"original_query": query},
            timeout_ms=5000,
        ),
    ]

    results, timing = await runner.run_parallel(tasks)
    engine = SynthesisEngine()
    report = engine.synthesize(query=query, results=results)

    console.print("\n[bold]Simulated: WebResearchAgent timed out mid-run[/bold]")
    print_report(report, "Failure Simulation Report (Partial Continuation)")


# ---------------------------------------------------------------------------
# Demo 3: Conflict scenario
# ---------------------------------------------------------------------------

async def demo_conflict_scenario() -> None:
    console.rule("[bold yellow]DEMO 3: Conflicting Sources — Evidence Reconciliation[/bold yellow]")

    import uuid
    from datetime import date
    from multi_agent_pipeline.schemas.task import AgentOutput
    from multi_agent_pipeline.schemas.finding import FindingSchema, SourceSchema
    from multi_agent_pipeline.synthesis.engine import SynthesisEngine

    query = "What is the current growth rate of the AI chip market?"
    task_id_a = f"task-{uuid.uuid4().hex[:8]}"
    task_id_b = f"task-{uuid.uuid4().hex[:8]}"

    output_a = AgentOutput(
        agent_name="WebResearchAgent",
        task_id=task_id_a,
        findings=[
            FindingSchema(
                finding_id=f"{task_id_a}.finding-1",
                claim="AI chip market grew 12% year-over-year in 2024.",
                evidence_excerpt="According to the OECD Digital Economy Outlook 2025, AI semiconductor revenue grew by 12% in 2024.",
                source=SourceSchema(
                    type="url",
                    location="https://oecd.org/digital-economy-outlook-2025",
                    publisher="OECD",
                    publication_date=date(2025, 1, 15),
                    credibility_score=0.95,
                ),
                confidence=0.91,
            )
        ],
        metadata={"execution_time_ms": 1200, "query_used": query},
    )

    output_b = AgentOutput(
        agent_name="DocumentAnalysisAgent",
        task_id=task_id_b,
        findings=[
            FindingSchema(
                finding_id=f"{task_id_b}.finding-1",
                claim="AI chip market grew 19% year-over-year in 2024.",
                evidence_excerpt="McKinsey Global Institute analysis (Feb 2025) estimates 19% YoY growth in AI silicon revenue.",
                source=SourceSchema(
                    type="document",
                    location="mckinsey_ai_chips_2025.pdf",
                    document_name="mckinsey_ai_chips_2025.pdf",
                    publisher="McKinsey Global Institute",
                    publication_date=date(2025, 2, 10),
                    credibility_score=0.85,
                ),
                confidence=0.88,
            )
        ],
        metadata={"execution_time_ms": 800, "query_used": query},
    )

    engine = SynthesisEngine()
    report = engine.synthesize(query=query, results=[output_a, output_b])

    console.print("\n[bold]Two authoritative sources report conflicting growth rates[/bold]")
    console.print("  • OECD (Jan 2025): 12% growth")
    console.print("  • McKinsey (Feb 2025): 19% growth")
    console.print("  → System must preserve BOTH values, not average them\n")
    print_report(report, "Conflict Scenario Report")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

async def main() -> None:
    console.print(
        Panel.fit(
            "[bold]Production Multi-Agent Research Pipeline[/bold]\n"
            "Coordinator-driven · Explicit context · Parallel execution\n"
            "Provenance-aware synthesis · Fault-tolerant · Observable",
            border_style="bright_blue",
        )
    )

    if not os.environ.get("ANTHROPIC_API_KEY"):
        console.print(
            "[bold red]WARNING:[/bold red] ANTHROPIC_API_KEY not set. "
            "Demos 2 and 3 use mock agents and will work. "
            "Demo 1 requires a valid API key.\n"
        )
        # Run only the non-API demos
        await demo_failure_simulation()
        await demo_conflict_scenario()
    else:
        await demo_full_pipeline()
        await demo_failure_simulation()
        await demo_conflict_scenario()


if __name__ == "__main__":
    asyncio.run(main())
