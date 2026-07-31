"""Heading-aware Markdown chunking with bounded same-section overlap."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from itertools import groupby

from rag_demo.markdown_parser import ParsedBlock, ParsedMarkdownDocument

CHUNKER_VERSION = "markdown-structure-v2"


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


class MarkdownChunker:  # 惰性拆分，用的时候才拆一次，不保存
    """Chunk consecutive blocks without mixing different heading paths."""

    def __init__(
        self,
        *,
        target_chars: int,
        max_chars: int,
        overlap_chars: int,
        version: str = CHUNKER_VERSION,
    ) -> None:
        if target_chars <= 0:
            raise ValueError("target_chars must be positive")
        if max_chars < target_chars:
            raise ValueError("max_chars must be at least target_chars")
        if overlap_chars < 0 or overlap_chars >= target_chars:
            raise ValueError("overlap_chars must be non-negative and smaller than target_chars")
        self._target_chars = target_chars
        self._max_chars = max_chars
        self._overlap_chars = overlap_chars
        self._version = version

    @property
    def version(self) -> str:
        return self._version

    @property
    def configuration(self) -> dict[str, int]:
        """Return parameters that affect chunk boundaries."""
        return {
            "target_chars": self._target_chars,
            "max_chars": self._max_chars,
            "overlap_chars": self._overlap_chars,
        }

    def chunk(self, document: ParsedMarkdownDocument) -> tuple[Chunk, ...]:
        """Create globally indexed chunks for a parsed document."""
        raw_chunks: list[tuple[tuple[str, ...], str]] = []

        for (_, heading_path), section_blocks in groupby(  # 提取同一段/同一主体的内容
            document.blocks,
            key=_section_key,
        ):
            section_texts = self._chunk_section(tuple(section_blocks))  # 将同一段尽量拆分，同时防止与其他段乱串掉
            raw_chunks.extend((heading_path, text) for text in section_texts)

        return tuple(
            self._build_chunk(document, index, heading_path, content_raw)
            for index, (heading_path, content_raw) in enumerate(raw_chunks)
        )

    def _chunk_section(self, blocks: tuple[ParsedBlock, ...]) -> tuple[str, ...]:
        completed: list[str] = []
        current = ""

        for block in blocks:
            text = block.text
            if len(text) > self._max_chars:  # 拆分超长
                if current:
                    completed.append(current)
                    current = ""
                completed.extend(self._split_oversized(text))
                continue

            if not current:
                current = text
                continue

            candidate = f"{current}\n\n{text}"
            if len(current) < self._target_chars and len(candidate) <= self._max_chars:  # 字数太少，不够，继续合并
                current = candidate
                continue

            completed.append(current)
            current = self._start_with_overlap(current, text)

        if current:
            completed.append(current)
        return tuple(completed)

    def _start_with_overlap(self, previous: str, next_text: str) -> str:
        if self._overlap_chars == 0:
            return next_text
        overlap = previous[-self._overlap_chars :]
        candidate = f"{overlap}\n\n{next_text}"
        return candidate if len(candidate) <= self._max_chars else next_text

    def _split_oversized(self, text: str) -> tuple[str, ...]:
        segments: list[str] = []
        start = 0

        while start < len(text):
            end = min(start + self._max_chars, len(text))
            segments.append(text[start:end])
            if end == len(text):
                break
            start = end - self._overlap_chars  # 这里产生 overlap

        return tuple(segments)

    def _build_chunk(
        self,
        document: ParsedMarkdownDocument,
        chunk_index: int,
        heading_path: tuple[str, ...],
        content_raw: str,
    ) -> Chunk:
        breadcrumb = " / ".join(heading_path) if heading_path else "（文档正文）"
        context = []
        if document.title is not None:
            context.append(f"[文档：{document.title}]")
        context.extend((f"[章节：{breadcrumb}]", content_raw))
        retrieval_text = "\n".join(context)
        return Chunk(
            chunk_index=chunk_index,
            chunker_version=self._version,
            heading_path=heading_path,
            content_raw=content_raw,
            retrieval_text=retrieval_text,
            char_count=len(content_raw),
            content_hash=hashlib.sha256(retrieval_text.encode("utf-8")).hexdigest(),
        )


def _section_key(block: ParsedBlock) -> tuple[str, tuple[str, ...]]:
    return block.section_root, block.heading_path
