from typer.testing import CliRunner

from rag_demo import __version__
from rag_demo.cli import app

runner = CliRunner()


def test_cli_reports_package_version() -> None:
    result = runner.invoke(app, ["--version"])

    assert result.exit_code == 0
    assert result.stdout.strip() == __version__


def test_cli_help_identifies_the_demo() -> None:
    result = runner.invoke(app, ["--help"])

    assert result.exit_code == 0
    assert "observable Hybrid RAG retrieval demo" in result.stdout
