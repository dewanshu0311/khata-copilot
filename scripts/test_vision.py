"""
Phase 1 CLI test: photo of a handwritten page in -> clean validated JSON out.

Usage (from repo root):
    python scripts/test_vision.py sample_data/page1.jpg
    python scripts/test_vision.py                 # no path -> mock demo page

Set KHATA_MOCK=1 (or leave GEMINI_API_KEY unset) to run fully offline.
"""
from __future__ import annotations

import sys
from pathlib import Path

# Windows consoles default stdout to cp1252, which can't encode ₹/⚑ and crashes.
# Force UTF-8 so the demo prints cleanly on any terminal.
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8")
    except Exception:
        pass

# Make the repo root importable when run as a plain script.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from rich.console import Console
from rich.table import Table

from agents.vision_agent import extract_page
from core.config import REVIEW_CONFIDENCE_THRESHOLD

console = Console()


def _confidence_style(c: float) -> str:
    if c >= REVIEW_CONFIDENCE_THRESHOLD:
        return "green"
    if c >= 0.5:
        return "yellow"
    return "red"


def main() -> int:
    image_path = sys.argv[1] if len(sys.argv) > 1 else "sample_data/mock_page.jpg"
    console.rule(f"[bold]Vision Agent[/bold] — reading {image_path}")

    result = extract_page(image_path)

    if result.degraded:
        badge = "[yellow]DEGRADED / MOCK[/yellow]" if not result.error else f"[red]FAILED[/red]"
        console.print(f"Status: {badge}")
        if result.error:
            console.print(f"[red]Error:[/red] {result.error}")

    table = Table(title="Extracted entries", show_lines=False)
    table.add_column("#", justify="right", style="dim")
    table.add_column("Name")
    table.add_column("Amount (₹)", justify="right")
    table.add_column("Date")
    table.add_column("Status")
    table.add_column("Conf.", justify="right")
    table.add_column("Review?", justify="center")

    for i, e in enumerate(result.entries, 1):
        needs_review = e.confidence <= REVIEW_CONFIDENCE_THRESHOLD
        table.add_row(
            str(i), e.name, f"{e.amount:,.2f}", e.date or "—", e.status,
            f"[{_confidence_style(e.confidence)}]{e.confidence:.2f}[/]",
            "[red]⚑[/red]" if needs_review else "",
        )
    console.print(table)

    flagged = result.flagged_entries(REVIEW_CONFIDENCE_THRESHOLD)
    console.print(
        f"Entries: [bold]{len(result.entries)}[/bold]   "
        f"Flagged for review (≤{REVIEW_CONFIDENCE_THRESHOLD}): [bold red]{len(flagged)}[/bold red]   "
        f"Overall confidence: [bold]{result.overall_confidence:.2f}[/bold]"
    )
    console.print(
        f"Computed total: [bold]₹{result.computed_total:,.2f}[/bold]   "
        f"Written total on page: "
        + (f"[bold]₹{result.written_total:,.2f}[/bold]" if result.written_total is not None else "[dim]none[/dim]")
    )
    if result.notes:
        console.print(f"[dim]Notes:[/dim] {result.notes}")

    console.rule("[bold]Raw validated JSON[/bold]")
    console.print_json(result.model_dump_json())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
