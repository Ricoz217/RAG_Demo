"""Discover and apply ordered PostgreSQL migration files."""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from pathlib import Path

from psycopg import AsyncConnection
from psycopg.rows import dict_row

DEFAULT_MIGRATIONS_DIR = Path(__file__).resolve().parents[2] / "migrations"
MIGRATION_FILENAME = re.compile(r"^(?P<version>\d{3})_(?P<label>[a-z0-9_]+)\.sql$")


class MigrationError(RuntimeError):
    """Raised when migrations are invalid or disagree with database history."""


@dataclass(frozen=True, slots=True)
class Migration:
    """One immutable SQL migration loaded from disk."""

    version: str
    name: str
    path: Path
    sql: str
    checksum: str


def discover_migrations(directory: Path = DEFAULT_MIGRATIONS_DIR) -> tuple[Migration, ...]:
    """Load validated migrations in version order."""
    try:
        paths = sorted(directory.glob("*.sql"))
    except OSError as exc:
        raise MigrationError(f"cannot read migrations directory: {directory}") from exc

    migrations: list[Migration] = []
    versions: set[str] = set()

    for path in paths:
        match = MIGRATION_FILENAME.fullmatch(path.name)
        if match is None:
            raise MigrationError(f"invalid migration filename: {path.name}")

        version = match.group("version")
        if version in versions:
            raise MigrationError(f"duplicate migration version {version}")

        try:
            sql = path.read_text(encoding="utf-8")
        except OSError as exc:
            raise MigrationError(f"cannot read migration: {path}") from exc

        versions.add(version)
        migrations.append(
            Migration(
                version=version,
                name=path.stem,
                path=path,
                sql=sql,
                checksum=hashlib.sha256(sql.encode("utf-8")).hexdigest(),
            )
        )

    return tuple(migrations)


async def apply_migrations(
    conninfo: str,
    directory: Path = DEFAULT_MIGRATIONS_DIR,
) -> tuple[str, ...]:
    """Apply every pending migration atomically and return applied names."""
    migrations = discover_migrations(directory)
    applied_names: list[str] = []

    connection = await AsyncConnection.connect(conninfo, row_factory=dict_row)
    async with connection:
        async with connection.transaction():
            await connection.execute(
                """
                CREATE TABLE IF NOT EXISTS schema_migrations (
                    version text PRIMARY KEY,
                    name text NOT NULL,
                    checksum text NOT NULL,
                    applied_at timestamptz NOT NULL DEFAULT now()
                )
                """
            )
            await connection.execute(
                "SELECT pg_advisory_xact_lock(hashtext('rag_demo_schema_migrations'))"
            )

            cursor = await connection.execute(
                "SELECT version, name, checksum FROM schema_migrations ORDER BY version"
            )
            applied = {row["version"]: row for row in await cursor.fetchall()}

            for migration in migrations:
                existing = applied.get(migration.version)
                if existing is not None:
                    if existing["name"] != migration.name:
                        raise MigrationError(
                            f"migration {migration.version} name changed "
                            f"from {existing['name']} to {migration.name}"
                        )
                    if existing["checksum"] != migration.checksum:
                        raise MigrationError(
                            f"migration {migration.name} checksum differs from database history"
                        )
                    continue

                await connection.execute(migration.sql)
                await connection.execute(
                    """
                    INSERT INTO schema_migrations (version, name, checksum)
                    VALUES (%s, %s, %s)
                    """,
                    (migration.version, migration.name, migration.checksum),
                )
                applied_names.append(migration.name)

    return tuple(applied_names)
