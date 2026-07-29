"""Command-line interface for the Hybrid RAG demo."""

from typing import Annotated

import typer
from rich.console import Console
from rich.table import Table

from rag_demo import __version__
from rag_demo.asyncio_compat import run_async
from rag_demo.config import Settings
from rag_demo.db import Database, DatabaseStatus
from rag_demo.migrations import apply_migrations

app = typer.Typer(
    name="rag-demo",
    help="A minimal, observable Hybrid RAG retrieval demo.",
    no_args_is_help=True,
)
db_app = typer.Typer(help="Initialize and inspect PostgreSQL.")
app.add_typer(db_app, name="db")
console = Console()


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


@db_app.command("init")
def db_init() -> None:
    """Apply pending PostgreSQL migrations."""
    settings = Settings()
    applied = run_async(apply_migrations(settings.database_url.get_secret_value()))

    if applied:
        console.print(f"Applied migrations: {', '.join(applied)}")
    else:
        console.print("Database schema is up to date.")


async def _database_status(settings: Settings) -> DatabaseStatus:
    async with Database(settings.database_url.get_secret_value()) as database:
        return await database.status()


@db_app.command("status")
def db_status() -> None:
    """Display PostgreSQL, pgvector, schema, and corpus status."""
    status = run_async(_database_status(Settings()))
    table = Table(title="Database status")
    table.add_column("Property")
    table.add_column("Value")
    table.add_row("User", status.current_user)
    table.add_row("Database", status.current_database)
    table.add_row("PostgreSQL", status.server_version)
    table.add_row("pgvector", status.vector_version)
    table.add_row("Embedding column", status.embedding_column_type)
    table.add_row("HNSW", "HNSW ready" if status.hnsw_index_present else "HNSW missing")
    table.add_row("Documents", str(status.document_count))
    table.add_row("Chunks", str(status.chunk_count))
    table.add_row("Migrations", ", ".join(status.applied_migrations))
    console.print(table)
