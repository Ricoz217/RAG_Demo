from io import StringIO

import pytest
from rich.console import Console
from typer.testing import CliRunner

from rag_demo import __version__
from rag_demo.api import create_app
from rag_demo.cli import _print_confidence, app
from rag_demo.search_service import SearchConfidence, SearchConfidenceStatus

runner = CliRunner()


def test_cli_reports_package_version() -> None:
    result = runner.invoke(app, ["--version"])

    assert result.exit_code == 0
    assert result.stdout.strip() == __version__


def test_cli_help_identifies_the_demo() -> None:
    result = runner.invoke(app, ["--help"])

    assert result.exit_code == 0
    assert "observable Hybrid RAG retrieval demo" in result.stdout


def test_ingest_idempotency_key_is_optional_in_cli_help() -> None:
    result = runner.invoke(app, ["ingest", "--help"])

    assert result.exit_code == 0
    assert "--idempotency-key" in result.stdout
    assert "Override the generated stable" in result.stdout
    assert "ingestion key." in result.stdout


def test_ingest_idempotency_key_is_optional_in_rest_schema() -> None:
    operation = create_app().openapi()["paths"]["/v1/ingest"]["post"]
    idempotency_parameter = next(
        parameter for parameter in operation["parameters"] if parameter["name"] == "Idempotency-Key"
    )

    assert idempotency_parameter["required"] is False


def test_search_exposes_opt_in_rewrite_in_cli_and_rest() -> None:
    cli = runner.invoke(app, ["search", "--help"])
    search_schema = create_app().openapi()["components"]["schemas"]["SearchBody"]

    assert cli.exit_code == 0
    assert "--rewrite" in cli.stdout
    assert search_schema["properties"]["rewrite"]["default"] is False


def test_cli_exposes_rewrite_effect_comparison_command() -> None:
    result = runner.invoke(app, ["compare-rewrite", "--help"])

    assert result.exit_code == 0
    assert "without Rewrite" in result.stdout


def test_cli_prints_low_confidence_warning(monkeypatch: pytest.MonkeyPatch) -> None:
    stream = StringIO()
    monkeypatch.setattr(
        "rag_demo.cli.console",
        Console(file=stream, force_terminal=False, color_system=None),
    )

    _print_confidence(
        SearchConfidence(
            status=SearchConfidenceStatus.LOW,
            score=-6.0,
            threshold=-4.0,
            warning="Top reranker score is low; results may be unreliable.",
        )
    )

    output = stream.getvalue()
    assert "WARNING" in output
    assert "results may be unreliable" in output
    assert "-6.000" in output
