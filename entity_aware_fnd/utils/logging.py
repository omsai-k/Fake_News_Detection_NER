from rich import print
from rich.console import Console
from rich.table import Table
import logging

_console = Console()

def setup_logging(level: str = "INFO"):
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format='[%(asctime)s] [%(levelname)s] %(name)s: %(message)s'
    )
    logging.getLogger('pykeen').setLevel(logging.WARNING)


def banner(title: str):
    _console.rule(f"[bold cyan]{title}")


def dict_table(d: dict, title: str = "Config"):
    table = Table(title=title)
    table.add_column("Key")
    table.add_column("Value")
    for k, v in d.items():
        table.add_row(str(k), str(v))
    _console.print(table)
