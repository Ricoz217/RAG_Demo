"""Structure-aware Markdown parsing for the ingestion pipeline."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path

from markdown_it import MarkdownIt
from markdown_it.token import Token


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

    title: str
    source_path: Path
    content_hash: str
    blocks: tuple[ParsedBlock, ...]


_MARKDOWN = MarkdownIt("commonmark")
_CONTAINER_BLOCKS = {
    "blockquote_open": BlockKind.PARAGRAPH,
    "bullet_list_open": BlockKind.LIST,
    "ordered_list_open": BlockKind.LIST,
}
_LEAF_BLOCKS = {
    "code_block": BlockKind.CODE,
    "fence": BlockKind.CODE,
    "html_block": BlockKind.PARAGRAPH,
}


def parse_markdown(markdown: str, *, source_path: Path) -> ParsedMarkdownDocument:
    """Parse Markdown while retaining source text and heading breadcrumbs."""
    normalized = _normalize_line_endings(markdown)
    lines = normalized.splitlines(keepends=True)
    tokens = _MARKDOWN.parse(normalized)

    title = source_path.stem
    found_title = False
    section_root = title
    headings: dict[int, str] = {}
    blocks: list[ParsedBlock] = []

    for index, token in enumerate(tokens):
        if token.type == "heading_open" and token.level == 0:
            level = int(token.tag.removeprefix("h"))
            heading = _heading_text(tokens, index)

            if level == 1 and not found_title:
                title = heading
                found_title = True
                section_root = heading
                headings.clear()
                continue

            _drop_heading_level_and_children(headings, level)
            headings[level] = heading
            if level == 1:
                section_root = heading
            continue

        kind = _block_kind(token)
        if kind is None or token.map is None:
            continue

        text = _source_slice(lines, token.map)
        if not text:
            continue
        blocks.append(
            ParsedBlock(
                kind=kind,
                text=text,
                heading_path=tuple(headings[level] for level in sorted(headings)),
                section_root=section_root,
            )
        )

    return ParsedMarkdownDocument(
        title=title,
        source_path=source_path,
        content_hash=_sha256(normalized),
        blocks=tuple(blocks),
    )


def _normalize_line_endings(text: str) -> str:
    return text.replace("\r\n", "\n").replace("\r", "\n")


def _sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _heading_text(tokens: list[Token], heading_index: int) -> str:
    try:
        inline = tokens[heading_index + 1]
    except IndexError as exc:
        raise ValueError("heading token has no inline content") from exc
    if inline.type != "inline":
        raise ValueError("heading token is not followed by inline content")
    return inline.content.strip()


def _drop_heading_level_and_children(headings: dict[int, str], level: int) -> None:
    for existing_level in tuple(headings):
        if existing_level >= level:
            del headings[existing_level]


def _block_kind(token: Token) -> BlockKind | None:
    if token.level == 0 and token.type == "paragraph_open":
        return BlockKind.PARAGRAPH
    if token.level == 0 and token.type in _CONTAINER_BLOCKS:
        return _CONTAINER_BLOCKS[token.type]
    if token.level == 0 and token.type in _LEAF_BLOCKS:
        return _LEAF_BLOCKS[token.type]
    return None


def _source_slice(lines: list[str], line_map: list[int]) -> str:
    start, end = line_map
    return "".join(lines[start:end]).strip("\n")
