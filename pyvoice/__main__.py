"""Command-line interface."""
import click


@click.command()
@click.version_option()
def main() -> None:
    """pyvoice."""


if __name__ == "__main__":
    main(prog_name="pyvoice")  # pragma: no cover
