# /// script
# requires-python = ">=3.12"
# dependencies = [
#   "beautifulsoup4>=4.15.0",
#   "ebooklib>=0.20",
#   "markdownify>=1.2.3",
# ]
# ///
"""Convert an unencrypted EPUB novel to one chapter-structured Markdown file.

Usage:
    uv run tools/epub_to_markdown.py novel.epub -o novel.md
"""

from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from bs4 import BeautifulSoup
from ebooklib import ITEM_DOCUMENT, epub
from markdownify import markdownify


@dataclass(frozen=True, slots=True)
class ConversionResult:
    destination: Path
    title: str
    authors: tuple[str, ...]
    chapter_count: int


def convert_epub(
    source: Path,
    destination: Path | None = None,
    *,
    overwrite: bool = False,
) -> ConversionResult:
    source = source.resolve()
    if not source.is_file():
        raise FileNotFoundError(source)
    if source.suffix.casefold() != ".epub":
        raise ValueError(f"source must be an EPUB file: {source}")

    destination = source.with_suffix(".md") if destination is None else destination.resolve()
    if destination.exists() and not overwrite:
        raise FileExistsError(f"destination already exists: {destination}")

    book: Any = epub.read_epub(str(source), options={"ignore_ncx": True})
    title = _first_metadata(book, "title") or source.stem
    authors = tuple(_metadata_values(book, "creator"))
    language = _first_metadata(book, "language")

    chapters: list[str] = []
    for item_id, _linear in book.spine:
        item: Any = book.get_item_with_id(item_id)
        if item is None or item.get_type() != ITEM_DOCUMENT or "nav" in set(item.properties):
            continue
        chapter = _xhtml_to_markdown(item.get_content())
        if chapter:
            chapters.append(chapter)

    if not chapters:
        raise ValueError(f"EPUB contains no readable chapters: {source}")

    output = _render_markdown(
        title=title,
        authors=authors,
        language=language,
        source_name=source.name,
        chapters=chapters,
    )
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(output, encoding="utf-8", newline="\n")
    return ConversionResult(destination, title, authors, len(chapters))


def _metadata_values(book: Any, name: str) -> list[str]:
    values: list[str] = []
    for value, _attributes in book.get_metadata("DC", name):
        text = str(value).strip()
        if text:
            values.append(text)
    return values


def _first_metadata(book: Any, name: str) -> str | None:
    values = _metadata_values(book, name)
    return values[0] if values else None


def _xhtml_to_markdown(content: bytes) -> str:
    soup = BeautifulSoup(content, "html.parser")
    for element in soup.select("script, style, nav, noscript, svg, img"):
        element.decompose()

    body = soup.body
    if body is None:
        return ""

    # The combined file reserves H1 for the book title.
    for heading in body.find_all(("h6", "h5", "h4", "h3", "h2", "h1")):
        if heading.name is None:
            continue
        level = int(heading.name[1])
        heading.name = f"h{min(level + 1, 6)}"

    if body.find(("h1", "h2", "h3", "h4", "h5", "h6")) is None:
        chapter_title = soup.title.get_text(" ", strip=True) if soup.title else ""
        if chapter_title:
            heading = soup.new_tag("h2")
            heading.string = chapter_title
            body.insert(0, heading)

    converted = str(markdownify(str(body), heading_style="ATX", bullets="-"))
    converted = "\n".join(line.rstrip() for line in converted.splitlines())
    return re.sub(r"\n{3,}", "\n\n", converted).strip()


def _render_markdown(
    *,
    title: str,
    authors: tuple[str, ...],
    language: str | None,
    source_name: str,
    chapters: list[str],
) -> str:
    metadata = [
        "---",
        f"title: {json.dumps(title, ensure_ascii=False)}",
        f"authors: {json.dumps(authors, ensure_ascii=False)}",
    ]
    if language is not None:
        metadata.append(f"language: {json.dumps(language, ensure_ascii=False)}")
    metadata.extend(
        (
            'source_format: "epub"',
            f"source_file: {json.dumps(source_name, ensure_ascii=False)}",
            "---",
        )
    )
    body = "\n\n".join((f"# {title}", *chapters))
    return "\n".join(metadata) + f"\n\n{body}\n"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path, help="unencrypted EPUB file")
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        help="output Markdown path; defaults to the EPUB path with a .md suffix",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="replace an existing output file",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    result = convert_epub(args.source, args.output, overwrite=args.overwrite)
    print(f"Title: {result.title}")
    print(f"Authors: {', '.join(result.authors) or '(unknown)'}")
    print(f"Chapters: {result.chapter_count}")
    print(f"Markdown: {result.destination}")


if __name__ == "__main__":
    main()
