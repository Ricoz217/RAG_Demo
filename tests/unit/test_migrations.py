from pathlib import Path

import pytest

from rag_demo.migrations import MigrationError, discover_migrations


def _write_migration(directory: Path, name: str, sql: str) -> None:
    (directory / name).write_text(sql, encoding="utf-8")


def test_discover_migrations_returns_ordered_versions_and_checksums(tmp_path: Path) -> None:
    _write_migration(tmp_path, "002_create_tables.sql", "SELECT 2;\n")
    _write_migration(tmp_path, "001_enable_vector.sql", "SELECT 1;\n")

    migrations = discover_migrations(tmp_path)

    assert [migration.version for migration in migrations] == ["001", "002"]
    assert [migration.name for migration in migrations] == [
        "001_enable_vector",
        "002_create_tables",
    ]
    assert all(len(migration.checksum) == 64 for migration in migrations)


def test_discover_migrations_checksum_changes_with_content(tmp_path: Path) -> None:
    path = tmp_path / "001_enable_vector.sql"
    path.write_text("SELECT 1;\n", encoding="utf-8")
    original_checksum = discover_migrations(tmp_path)[0].checksum

    path.write_text("SELECT 2;\n", encoding="utf-8")

    assert discover_migrations(tmp_path)[0].checksum != original_checksum


def test_discover_migrations_rejects_duplicate_versions(tmp_path: Path) -> None:
    _write_migration(tmp_path, "001_first.sql", "SELECT 1;\n")
    _write_migration(tmp_path, "001_second.sql", "SELECT 2;\n")

    with pytest.raises(MigrationError, match="duplicate migration version 001"):
        discover_migrations(tmp_path)


def test_discover_migrations_rejects_invalid_filename(tmp_path: Path) -> None:
    _write_migration(tmp_path, "create_tables.sql", "SELECT 1;\n")

    with pytest.raises(MigrationError, match="invalid migration filename"):
        discover_migrations(tmp_path)
