#!/usr/bin/env python3
"""Interactive CLI for the AI Companion with long-term memory."""

from __future__ import annotations

import logging
import sys
import uuid
from typing import Optional

import typer
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from config import DB_PATH, GEMINI_API_KEY, GROQ_API_KEY, LLM_MODEL
from core.pipeline import CompanionPipeline
from core.persona import DEFAULT_PERSONA
from core.schema import FactStatus
from core.tools import default_tool_registry

app = typer.Typer(add_completion=False, help="CLI AI Companion")
console = Console()
logger = logging.getLogger(__name__)


def _setup_logging(verbose: bool) -> None:
    level = logging.DEBUG if verbose else logging.WARNING
    logging.basicConfig(level=level, format="%(levelname)s %(name)s: %(message)s")


def _print_welcome(session_id: str) -> None:
    console.print(
        Panel(
            f"[bold]{DEFAULT_PERSONA.name}[/bold] — your companion with long-term memory.\n\n"
            f"Session: [dim]{session_id[:8]}...[/dim]\n"
            f"Database: [dim]{DB_PATH}[/dim]\n"
            f"Model: [dim]{LLM_MODEL}[/dim]\n\n"
            "Commands: [cyan]/memories[/cyan] [cyan]/history[/cyan] "
            "[cyan]/reset[/cyan] [cyan]/exit[/cyan]",
            title="CLI AI Companion",
            border_style="green",
        )
    )
    if not (GEMINI_API_KEY or GROQ_API_KEY):
        console.print(
            "[yellow]Warning:[/yellow] No Gemini or Groq API key set. "
            "Using heuristic extraction and fallback responses.\n"
        )


def _print_memories(pipeline: CompanionPipeline) -> None:
    facts = pipeline.get_active_memories()
    if not facts:
        console.print("[dim]No active memories stored yet.[/dim]")
        return
    table = Table(title="Active Memories", show_header=True, header_style="bold cyan")
    table.add_column("Category", style="magenta", max_width=12)
    table.add_column("Fact", max_width=50)
    table.add_column("Conf.", justify="right", max_width=6)
    table.add_column("Updated", max_width=20)
    for fact in facts:
        table.add_row(
            str(fact.category.value),
            fact.text[:80],
            f"{fact.confidence:.2f}",
            fact.updated_at.strftime("%Y-%m-%d %H:%M"),
        )
    console.print(table)
    superseded = pipeline.db.count_facts(status=FactStatus.SUPERSEDED)
    console.print(f"[dim]Superseded facts in DB: {superseded}[/dim]")


def _print_memory_matches(pipeline: CompanionPipeline, query: str) -> None:
    matches = [memory.fact for memory in pipeline.retriever.search(query, top_k=20) if memory.score >= 0.45]
    if not matches:
        console.print("[dim]No matching active memories found.[/dim]")
        return
    table = Table(title=f"Memory Search: {query}", show_header=True, header_style="bold cyan")
    table.add_column("ID", style="dim", no_wrap=True)
    table.add_column("Category", style="magenta", max_width=12)
    table.add_column("Fact", max_width=60)
    table.add_column("Updated", max_width=20)
    for fact in matches:
        table.add_row(
            fact.id,
            str(fact.category.value),
            fact.text[:100],
            fact.updated_at.strftime("%Y-%m-%d %H:%M"),
        )
    console.print(table)


def _print_history(pipeline: CompanionPipeline) -> None:
    turns = pipeline.get_session_history()
    if not turns:
        console.print("[dim]No conversation history for this session.[/dim]")
        return
    for turn in turns:
        role = turn["role"]
        style = "bold green" if role == "user" else "bold blue"
        label = "You" if role == "user" else DEFAULT_PERSONA.name
        console.print(f"[{style}]{label}:[/{style}] {turn['content']}")


def _print_tools() -> None:
    table = Table(title="Agent Tools", show_header=True, header_style="bold cyan")
    table.add_column("Name", style="magenta")
    table.add_column("Description")
    for tool in default_tool_registry().list_tools():
        table.add_row(tool.name, tool.description)
    console.print(table)


def _resolve_memory_id(pipeline: CompanionPipeline, identifier: str) -> str | None:
    facts = pipeline.get_active_memories()
    if identifier.isdigit():
        index = int(identifier)
        if 1 <= index <= len(facts):
            return facts[index - 1].id
        return None
    matches = [fact.id for fact in facts if fact.id.startswith(identifier)]
    return matches[0] if len(matches) == 1 else None


def _handle_memories_command(command: str, pipeline: CompanionPipeline) -> None:
    parts = command.strip().split(maxsplit=2)
    if len(parts) == 1:
        _print_memories(pipeline)
        return
    action = parts[1].lower()
    if action == "search" and len(parts) == 3 and parts[2].strip():
        _print_memory_matches(pipeline, parts[2].strip())
        return
    if action == "forget" and len(parts) == 3:
        fact_id = _resolve_memory_id(pipeline, parts[2].strip())
        if fact_id and pipeline.db.forget_fact(fact_id):
            console.print(f"[yellow]Memory {fact_id} marked as forgotten.[/yellow]")
        else:
            console.print("[red]Active memory not found.[/red]")
        return
    console.print("[red]Usage:[/red] /memories [search <query>|forget <id>]")


def _handle_command(command: str, pipeline: CompanionPipeline) -> bool:
    """Returns True if the REPL should continue, False to exit."""
    user_input = command.strip()
    cmd = user_input.lower()
    if cmd in ("/exit", "/quit", "/q"):
        console.print("[dim]Goodbye.[/dim]")
        return False
    if user_input == "/memories" or user_input.startswith("/memories "):
        _handle_memories_command(user_input, pipeline)
        return True
    if cmd == "/tools":
        _print_tools()
        return True
    if cmd == "/history":
        _print_history(pipeline)
        return True
    if cmd == "/reset":
        pipeline.reset_session()
        console.print("[yellow]Session history cleared. Memories persist in database.[/yellow]")
        return True
    console.print(f"[red]Unknown command:[/red] {command}")
    return True


def run_repl(pipeline: CompanionPipeline) -> None:
    _print_welcome(pipeline.session_id)
    while True:
        try:
            user_input = console.input("[bold green]You:[/bold green] ").strip()
        except (EOFError, KeyboardInterrupt):
            console.print("\n[dim]Goodbye.[/dim]")
            break

        if not user_input:
            continue
        if user_input.startswith("/"):
            if not _handle_command(user_input, pipeline):
                break
            continue

        console.print(f"[bold blue]{DEFAULT_PERSONA.name}:[/bold blue] ", end="")
        try:
            stream_gen = pipeline.process_turn(user_input, stream=True)
            parts: list[str] = []
            for chunk in stream_gen:
                console.print(chunk, end="")
                parts.append(chunk)
            console.print()
        except Exception as exc:
            logger.exception("Turn failed")
            console.print(f"\n[red]Error processing turn:[/red] {exc}")


@app.command()
def main(
    session_id: Optional[str] = typer.Option(None, help="Resume a specific session ID"),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Enable debug logging"),
) -> None:
    """Launch the interactive AI Companion CLI."""
    _setup_logging(verbose)
    sid = session_id or str(uuid.uuid4())
    try:
        pipeline = CompanionPipeline(session_id=sid)
    except Exception as exc:
        console.print(f"[red]Failed to initialize:[/red] {exc}")
        raise typer.Exit(code=1) from exc
    run_repl(pipeline)


if __name__ == "__main__":
    app()
