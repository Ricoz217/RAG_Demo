import pytest
from typer.testing import CliRunner

from rag_demo.cli import app

pytestmark = pytest.mark.integration
runner = CliRunner()


def test_db_init_and_status_use_the_configured_database() -> None:
    initialized = runner.invoke(app, ["db", "init"])
    status = runner.invoke(app, ["db", "status"])

    assert initialized.exit_code == 0
    assert "Database schema is up to date." in initialized.stdout
    assert status.exit_code == 0
    assert "rag_app" in status.stdout
    assert "rag_demo" in status.stdout
    assert "0.8.5" in status.stdout
    assert "vector(1024)" in status.stdout
    assert "HNSW ready" in status.stdout
