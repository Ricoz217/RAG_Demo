"""Command-line interface for the Hybrid RAG demo."""

from typing import Annotated

import typer

from rag_demo import __version__

app = typer.Typer(
    name="rag-demo",
    help="A minimal, observable Hybrid RAG retrieval demo.",
    no_args_is_help=True,
)


def _version_callback(value: bool) -> None:
    if value:
        typer.echo(__version__)
        raise typer.Exit


@app.callback()
def main(
    version: Annotated[
        bool,
        typer.Option(
            "--version",
            callback=_version_callback,
            is_eager=True,
            help="Show the package version and exit.",
        ),
    ] = False,
) -> None:
    """Run the observable Hybrid RAG retrieval demo."""
