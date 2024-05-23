"""Command-line interface."""

import logging

import click

from pyvoice.logging import configure_logging
from pyvoice.server import server


@click.command()
@click.version_option()
@click.option("--verbose", "-v", is_flag=True, help="Enable verbose logging")
def main(verbose: bool) -> None:
    """pyvoice."""
    configure_logging(server, logging.DEBUG if verbose else logging.INFO)
    server.start_io()


if __name__ == "__main__":
    main(prog_name="pyvoice")  # pragma: no cover
