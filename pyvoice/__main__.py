"""Command-line interface."""

import click

from pyvoice.logging import configure_logging
from pyvoice.server import server


@click.command()
@click.version_option()
def main() -> None:
    """pyvoice."""
    configure_logging(server)
    server.start_io()


if __name__ == "__main__":
    main(prog_name="pyvoice")  # pragma: no cover
