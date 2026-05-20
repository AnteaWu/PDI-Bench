import logging
import numpy as np
from rich.console import Console
from rich.logging import RichHandler
from rich.panel import Panel
from rich.table import Table

console = Console()

class PDILogger:
    """Rich-styled logger for PDI-Eval."""
    def __init__(self):
        logging.basicConfig(
            level="INFO",
            format="%(message)s",
            datefmt="[%X]",
            handlers=[RichHandler(rich_tracebacks=True, console=console)]
        )
        self.log = logging.getLogger("pdi_eval")

    def info(self, msg: str):
        self.log.info(msg)

    def success(self, msg: str):
        self.log.info(f"[bold green][OK] {msg}[/bold green]")

    def warning(self, msg: str):
        self.log.warning(f"[yellow]{msg}[/yellow]")

    def error(self, msg: str):
        self.log.error(f"[bold red][ERR] {msg}[/bold red]")

    def pdi_report(self, report: dict):
        """Print final audit summary table."""
        table = Table(title="[bold blue]PDI-Eval Final Audit Report[/bold blue]")
        table.add_column("Indicator", justify="right", style="cyan")
        table.add_column("Value", justify="left", style="magenta")
        
        table.add_row("PDI Score", str(report['pdi_score']))
        table.add_row("Grade", report['grade'])
        table.add_section()
        
        for k, v in report['breakdown'].items():
            # Only scalar values; skip sequence histories
            if isinstance(v, (float, int)) or (isinstance(v, np.ndarray) and v.size == 1):
                table.add_row(f"{k}", str(v))
            
        color = "green" if report['pdi_score'] < 0.2 else ("yellow" if report['pdi_score'] < 0.5 else "red")
        
        console.print(Panel(table, title="Final Verdict", subtitle=f"Status: {report['grade']}", border_style=color))

pdi_logger = PDILogger()
