"""Document parsing and chunking data contracts."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path


class BlockKind(StrEnum):
    """Markdown block categories relevant to chunking."""

    PARAGRAPH = "paragraph"
    LIST = "list"
    CODE = "code"


@dataclass(frozen=True, slots=True)
class ParsedBlock:
    """One source-preserving Markdown block under a heading path."""

    kind: BlockKind
    text: str
    heading_path: tuple[str, ...]
    section_root: str


@dataclass(frozen=True, slots=True)
class ParsedMarkdownDocument:
    """Normalized Markdown structure before chunking."""

    title: str | None
    first_h1: str | None
    frontmatter: Mapping[str, object]
    source_path: Path
    content_hash: str
    blocks: tuple[ParsedBlock, ...]


@dataclass(frozen=True, slots=True)
class Chunk:
    """One retrieval unit derived from a Markdown document."""

    chunk_index: int
    chunker_version: str
    heading_path: tuple[str, ...]
    content_raw: str
    retrieval_text: str
    char_count: int
    content_hash: str
