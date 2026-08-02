"""Download and inspect sparse Git-backed Markdown corpora."""

from __future__ import annotations

import asyncio
import subprocess
from pathlib import Path

from rag_demo.models.operations import CorpusInfo

__all__ = [
    "CorpusGitError",
    "CorpusInfo",
    "FASTAPI_REPOSITORY_URL",
    "FASTAPI_SPARSE_PATHS",
    "download_corpus",
    "download_fastapi",
    "inspect_corpus",
]

FASTAPI_REPOSITORY_URL = "https://github.com/fastapi/fastapi.git"
FASTAPI_SPARSE_PATHS = ("docs/zh/docs", "docs/en/docs")


class CorpusGitError(RuntimeError):
    """Raised when a corpus Git operation fails."""


async def download_fastapi(
    destination: Path = Path("data/corpus/fastapi"),
) -> CorpusInfo:
    """Shallow sparse-clone the official FastAPI repository."""
    return await download_corpus(
        repository_url=FASTAPI_REPOSITORY_URL,
        destination=destination,
        sparse_paths=FASTAPI_SPARSE_PATHS,
    )


async def download_corpus(
    *,
    repository_url: str,
    destination: Path,
    sparse_paths: tuple[str, ...],
) -> CorpusInfo:
    """Clone once or inspect an existing checkout without overwriting files."""
    if not repository_url.strip():
        raise ValueError("repository_url must not be empty")
    if not sparse_paths:
        raise ValueError("sparse_paths must not be empty")
    return await asyncio.to_thread(
        _download_corpus_sync,
        repository_url,
        destination,
        sparse_paths,
    )


def inspect_corpus(path: Path) -> CorpusInfo:
    """Read the repository URL, commit, and Markdown counts."""
    repository = path.resolve()
    if not (repository / ".git").exists():
        raise FileNotFoundError(f"corpus is not a Git checkout: {repository}")
    repository_url = _git_output(repository, "remote", "get-url", "origin")
    commit = _git_output(repository, "rev-parse", "HEAD")
    return CorpusInfo(
        path=repository,
        repository_url=repository_url,
        commit=commit,
        zh_markdown_count=_markdown_count(repository / "docs" / "zh" / "docs"),
        en_markdown_count=_markdown_count(repository / "docs" / "en" / "docs"),
    )


def _download_corpus_sync(
    repository_url: str,
    destination: Path,
    sparse_paths: tuple[str, ...],
) -> CorpusInfo:
    target = destination.resolve()
    if target.exists():
        if (target / ".git").exists():
            if not all((target / path).exists() for path in sparse_paths):
                if _git_output(target, "status", "--porcelain"):
                    raise CorpusGitError(
                        "existing corpus checkout has local changes; sparse paths were not modified"
                    )
                _run_git(target, "sparse-checkout", "set", *sparse_paths)
            return inspect_corpus(target)
        if any(target.iterdir()):
            raise FileExistsError(f"destination exists and is not a Git repository: {target}")
    target.parent.mkdir(parents=True, exist_ok=True)
    _run_git(
        None,
        "clone",
        "--depth",
        "1",
        "--filter=blob:none",
        "--sparse",
        repository_url,
        str(target),
    )
    _run_git(target, "sparse-checkout", "set", *sparse_paths)
    return inspect_corpus(target)


def _markdown_count(path: Path) -> int:
    if not path.is_dir():
        return 0
    return sum(1 for candidate in path.rglob("*.md") if candidate.is_file())


def _git_output(repository: Path, *arguments: str) -> str:
    return _run_git(repository, *arguments).stdout.strip()


def _run_git(
    repository: Path | None,
    *arguments: str,
) -> subprocess.CompletedProcess[str]:
    command = ["git"]
    if repository is not None:
        command.extend(("-C", str(repository)))
    command.extend(arguments)
    try:
        return subprocess.run(
            command,
            check=True,
            capture_output=True,
            text=True,
            encoding="utf-8",
        )
    except subprocess.CalledProcessError as exc:
        detail = (exc.stderr or "").strip() or "unknown Git error"
        raise CorpusGitError(f"git {arguments[0]} failed: {detail}") from exc
