"""Asynchronous PostgreSQL pool and database status inspection."""

from __future__ import annotations

import re
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any, Self

from pgvector.psycopg import register_vector_async
from psycopg import AsyncConnection
from psycopg.rows import dict_row
from psycopg_pool import AsyncConnectionPool

from rag_demo.models.operations import DatabaseStatus

Row = dict[str, Any]

__all__ = ["Database", "DatabaseStatus", "Row"]


async def _configure_connection(connection: AsyncConnection[Any]) -> None:
    await register_vector_async(connection)


class Database:
    """Own and expose a reusable Psycopg asynchronous connection pool."""

    def __init__(
        self,
        conninfo: str,
        *,
        min_size: int = 1,
        max_size: int = 4,
    ) -> None:
        self._pool = AsyncConnectionPool(
            conninfo,
            min_size=min_size,
            max_size=max_size,
            open=False,
            kwargs={"row_factory": dict_row},  # row 模式，返回字典
            configure=_configure_connection,
            name="rag-demo",
        )

    async def open(self) -> None:
        await self._pool.open(wait=True)

    async def close(self) -> None:
        await self._pool.close()

    async def __aenter__(self) -> Self:
        await self.open()
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: object | None,
    ) -> None:
        await self.close()

    @asynccontextmanager
    async def connection(self) -> AsyncIterator[AsyncConnection[Row]]:
        """Borrow a pgvector-configured connection from the pool."""
        async with self._pool.connection() as connection:
            yield connection

    async def status(self) -> DatabaseStatus:
        """Read database, extension, schema, index, and row-count facts."""
        async with self.connection() as connection:
            identity_cursor = await connection.execute(
                """
                SELECT
                    current_user AS current_user,
                    current_database() AS current_database,
                    current_setting('server_version') AS server_version
                """
            )  # 设置 cursor 上下文
            identity = await identity_cursor.fetchone()
            if identity is None:
                raise RuntimeError("database identity query returned no row")

            vector_cursor = await connection.execute(
                "SELECT extversion FROM pg_extension WHERE extname = 'vector'"
            )  # 检查当前 PG 是否有 pg_vector 插件
            vector_row = await vector_cursor.fetchone()
            if vector_row is None:
                raise RuntimeError("pgvector extension is not installed")

            # 获取系统信息，进行筛选
            # 找符合当前 schema  当前是 'public'
            # 找 schema 中名为 "chunks" 的表
            # 找 chunks 中名为 "embedding" 的列
            # 同时未标记为已删除
            # 这里是为了获取列信息
            type_cursor = await connection.execute(
                """
                SELECT format_type(attribute.atttypid, attribute.atttypmod) AS column_type
                FROM pg_attribute AS attribute
                JOIN pg_class AS relation ON relation.oid = attribute.attrelid
                JOIN pg_namespace AS namespace ON namespace.oid = relation.relnamespace
                WHERE namespace.nspname = current_schema()
                  AND relation.relname = 'chunks'
                  AND attribute.attname = 'embedding'
                  AND NOT attribute.attisdropped
                """
            )
            type_row = await type_cursor.fetchone()
            if type_row is None:
                raise RuntimeError("chunks.embedding column does not exist")
            dimensions = _parse_vector_dimensions(type_row["column_type"])  # 用正则获取维度

            facts_cursor = await connection.execute(
                """
                SELECT
                    EXISTS (
                        SELECT 1
                        FROM pg_indexes
                        WHERE schemaname = current_schema()
                          AND tablename = 'chunks'
                          AND indexname = 'chunks_embedding_hnsw'
                          AND indexdef ILIKE '%USING hnsw%'
                          AND indexdef ILIKE '%vector_cosine_ops%'
                    ) AS hnsw_index_present,
                    (SELECT count(*) FROM documents) AS document_count,
                    (SELECT count(*) FROM chunks) AS chunk_count
                """
            )
            facts = await facts_cursor.fetchone()
            if facts is None:
                raise RuntimeError("database facts query returned no row")

            migrations_cursor = await connection.execute(
                "SELECT name FROM schema_migrations ORDER BY version"
            )
            migrations = tuple(row["name"] for row in await migrations_cursor.fetchall())

        return DatabaseStatus(
            current_user=identity["current_user"],
            current_database=identity["current_database"],
            server_version=identity["server_version"],
            vector_version=vector_row["extversion"],
            embedding_dimensions=dimensions,
            hnsw_index_present=facts["hnsw_index_present"],
            document_count=facts["document_count"],
            chunk_count=facts["chunk_count"],
            applied_migrations=migrations,
        )


def _parse_vector_dimensions(column_type: str) -> int:
    match = re.fullmatch(r"vector\((\d+)\)", column_type)
    if match is None:
        raise RuntimeError(f"unexpected chunks.embedding type: {column_type}")
    return int(match.group(1))
