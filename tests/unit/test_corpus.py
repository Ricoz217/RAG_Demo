import asyncio
import subprocess
from pathlib import Path

import pytest

from rag_demo.corpus import download_corpus, inspect_corpus


def _git(*args: str, cwd: Path) -> None:
    subprocess.run(
        ["git", *args],
        cwd=cwd,
        check=True,
        capture_output=True,
        text=True,
        encoding="utf-8",
    )


@pytest.mark.asyncio
async def test_download_corpus_uses_shallow_sparse_clone_and_records_revision(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source"
    source.mkdir()
    _git("init", "-b", "main", cwd=source)
    _git("config", "user.name", "RAG Test", cwd=source)
    _git("config", "user.email", "rag@example.test", cwd=source)
    zh = source / "docs" / "zh" / "docs"
    en = source / "docs" / "en" / "docs"
    unrelated = source / "scripts"
    zh.mkdir(parents=True)
    en.mkdir(parents=True)
    unrelated.mkdir()
    (zh / "body.md").write_text("# 请求体", encoding="utf-8")
    (en / "body.md").write_text("# Request Body", encoding="utf-8")
    (unrelated / "tool.py").write_text("print('not sparse')", encoding="utf-8")
    _git("add", ".", cwd=source)
    _git("commit", "-m", "fixture", cwd=source)
    expected_commit = (
        await asyncio.to_thread(
            subprocess.run,
            ["git", "rev-parse", "HEAD"],
            cwd=source,
            check=True,
            capture_output=True,
            text=True,
            encoding="utf-8",
        )
    ).stdout.strip()
    destination = tmp_path / "clone"

    downloaded = await download_corpus(
        repository_url=str(source),
        destination=destination,
        sparse_paths=("docs/zh/docs", "docs/en/docs"),
    )
    repeated = await download_corpus(
        repository_url=str(source),
        destination=destination,
        sparse_paths=("docs/zh/docs", "docs/en/docs"),
    )

    assert downloaded.commit == expected_commit
    assert downloaded.zh_markdown_count == 1
    assert downloaded.en_markdown_count == 1
    assert (destination / "docs" / "zh" / "docs" / "body.md").is_file()
    assert not (destination / "scripts" / "tool.py").exists()
    assert repeated == downloaded
    assert inspect_corpus(destination) == downloaded


@pytest.mark.asyncio
async def test_download_corpus_refuses_non_repository_destination(tmp_path: Path) -> None:
    destination = tmp_path / "occupied"
    destination.mkdir()
    (destination / "user-file.txt").write_text("keep", encoding="utf-8")

    with pytest.raises(FileExistsError, match="not a Git repository"):
        await download_corpus(
            repository_url="https://example.test/repository.git",
            destination=destination,
            sparse_paths=("docs",),
        )

    assert (destination / "user-file.txt").read_text(encoding="utf-8") == "keep"
