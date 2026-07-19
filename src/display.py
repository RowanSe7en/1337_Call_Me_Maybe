import json
from typing import Any

from rich import print as rprint
from rich.columns import Columns
from rich.panel import Panel
from rich.table import Table
from rich.rule import Rule


_LOGO: str = ("\n[bold green]  ██████╗ █████╗ ██╗     ██╗        ███╗   ███╗███████╗"
    "   ███╗   ███╗ █████╗ ██╗   ██╗██████╗ ███████╗\n"
    " ██╔════╝██╔══██╗██║     ██║        ████╗ ████║██╔════╝"
    "   ████╗ ████║██╔══██╗╚██╗ ██╔╝██╔══██╗██╔════╝\n"
    " ██║     ███████║██║     ██║        ██╔████╔██║█████╗  "
    "   ██╔████╔██║███████║ ╚████╔╝ ██████╔╝█████╗\n"
    " ██║     ██╔══██║██║     ██║        ██║╚██╔╝██║██╔══╝  "
    "   ██║╚██╔╝██║██╔══██║  ╚██╔╝  ██╔══██╗██╔══╝\n"
    " ╚██████╗██║  ██║███████╗███████╗   ██║ ╚═╝ ██║███████╗"
    "   ██║ ╚═╝ ██║██║  ██║   ██║   ██████╔╝███████╗\n"
    "  ╚═════╝╚═╝  ╚═╝╚══════╝╚══════╝   ╚═╝     ╚═╝╚══════╝"
    "   ╚═╝     ╚═╝╚═╝  ╚═╝   ╚═╝   ╚═════╝ ╚══════╝"
    "[/bold green]"
)


def print_logo() -> None:
    rprint(_LOGO)


def print_result(result: dict[str, Any], elapsed: float, total: float) -> None:
    left_text = (
        f"[bold cyan]prompt: [/bold cyan]{result['prompt']}\n"
    )
    right_text = json.dumps(result, indent=2)

    rprint(
        Columns(
            [
                Panel(left_text, title="Parsed", border_style="green"),
                Panel(right_text, title="JSON", border_style="blue"),
            ]
        )
    )
    rprint(
        Panel(
            f"[cyan]step:[/cyan] {elapsed:.2f}s   "
            f"[cyan]total:[/cyan] {total:.2f}s",
            border_style="green",
            expand=False,
            title="Timing",
        )
    )

    rprint(Rule(style="dim"))


def print_error(prompt: str, reason: str) -> None:
    rprint(
        Panel(
            f"[bold]Prompt:[/bold] {prompt}\n"
            f"[red]{reason}[/red]",
            title="[red]Generation error[/red]",
            border_style="red",
            expand=False,
        )
    )


def print_summary(total_prompts: int, successes: int, total_time: float) -> None:
    table = Table(title="Run Summary", border_style="green")
    table.add_column("Metric", style="cyan", no_wrap=True)
    table.add_column("Value", style="bold white")

    table.add_row("Prompts processed", str(total_prompts))
    table.add_row("Successful", str(successes))
    table.add_row("Failed", str(total_prompts - successes))
    table.add_row(
        "Accuracy",
        f"{successes / total_prompts * 100:.1f}%" if total_prompts else "N/A",
    )
    table.add_row("Total time", f"{total_time:.2f}s")
    table.add_row(
        "Avg time / prompt",
        f"{total_time / total_prompts:.2f}s" if total_prompts else "N/A",
    )

    rprint(table)
