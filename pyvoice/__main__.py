"""Command-line interface."""
import click

from .server import server


@click.command()
@click.version_option()
def main() -> None:
    """pyvoice."""
    server.start_io()


if __name__ == "__main__":
    main(prog_name="pyvoice")  # pragma: no cover
