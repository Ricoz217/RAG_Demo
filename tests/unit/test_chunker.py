from pathlib import Path

import pytest

from rag_demo.chunker import MarkdownChunker
from rag_demo.markdown_parser import parse_markdown


def test_chunker_preserves_complete_code_block_within_maximum() -> None:
    code = "```python\n" + "\n".join(f"print({index})" for index in range(8)) + "\n```"
    document = parse_markdown(
        f"# 示例\n\n## 代码\n\n代码说明。\n\n{code}\n",
        source_path=Path("code.md"),
    )
    chunker = MarkdownChunker(target_chars=60, max_chars=140, overlap_chars=15)

    chunks = chunker.chunk(document)

    chunks_with_code = [chunk for chunk in chunks if "```python" in chunk.content_raw]
    assert len(chunks_with_code) == 1
    assert code in chunks_with_code[0].content_raw
    assert all(chunk.char_count <= 140 for chunk in chunks)


def test_chunker_adds_retrieval_context_and_stable_metadata() -> None:
    document = parse_markdown(
        "# FastAPI 教程\n\n## 请求体\n\n使用模型声明 JSON 对象。",
        source_path=Path("tutorial/body.md"),
    )

    chunks = MarkdownChunker(target_chars=100, max_chars=150, overlap_chars=20).chunk(document)

    assert len(chunks) == 1
    assert chunks[0].chunk_index == 0
    assert chunks[0].heading_path == ("请求体",)
    assert chunks[0].retrieval_text == (
        "[文档：FastAPI 教程]\n[章节：请求体]\n使用模型声明 JSON 对象。"
    )
    assert chunks[0].char_count == len(chunks[0].content_raw)
    assert len(chunks[0].content_hash) == 64


def test_chunker_never_combines_different_heading_paths() -> None:
    document = parse_markdown(
        """# 标题

## 第一节

第一节内容。

## 第二节

第二节内容。
""",
        source_path=Path("sections.md"),
    )

    chunks = MarkdownChunker(target_chars=100, max_chars=150, overlap_chars=20).chunk(document)

    assert [chunk.heading_path for chunk in chunks] == [("第一节",), ("第二节",)]
    assert "第一节内容" not in chunks[1].content_raw


def test_chunker_overlap_stays_inside_the_same_section() -> None:
    first_section = "甲" * 45 + "FIRST_END"
    second_paragraph = "乙" * 45
    document = parse_markdown(
        (
            "# 标题\n\n"
            "## 第一节\n\n"
            f"{first_section}\n\n"
            f"{second_paragraph}\n\n"
            "## 第二节\n\n"
            "SECOND_START\n"
        ),
        source_path=Path("overlap.md"),
    )
    chunker = MarkdownChunker(target_chars=40, max_chars=70, overlap_chars=12)

    chunks = chunker.chunk(document)

    first_section_chunks = [chunk for chunk in chunks if chunk.heading_path == ("第一节",)]
    second_section_chunk = next(chunk for chunk in chunks if chunk.heading_path == ("第二节",))
    assert len(first_section_chunks) >= 2
    assert first_section_chunks[0].content_raw[-12:] in first_section_chunks[1].content_raw
    assert "FIRST_END" not in second_section_chunk.content_raw


def test_chunker_splits_oversized_code_block_without_exceeding_maximum() -> None:
    code_body = "\n".join(f"line_{index:03d} = {index}" for index in range(60))
    document = parse_markdown(
        f"# 标题\n\n## 大代码块\n\n```python\n{code_body}\n```\n",
        source_path=Path("large-code.md"),
    )

    chunks = MarkdownChunker(target_chars=100, max_chars=140, overlap_chars=20).chunk(document)

    assert len(chunks) > 1
    assert all(chunk.char_count <= 140 for chunk in chunks)
    assert all(chunk.heading_path == ("大代码块",) for chunk in chunks)
    assert any("line_000" in chunk.content_raw for chunk in chunks)
    assert any("line_059" in chunk.content_raw for chunk in chunks)


def test_chunk_content_hash_is_stable_for_equivalent_line_endings() -> None:
    chunker = MarkdownChunker(target_chars=100, max_chars=150, overlap_chars=20)

    windows = chunker.chunk(
        parse_markdown("# 标题\r\n\r\n正文。\r\n", source_path=Path("stable.md"))
    )
    unix = chunker.chunk(parse_markdown("# 标题\n\n正文。\n", source_path=Path("stable.md")))

    assert windows == unix


@pytest.mark.parametrize(
    ("target_chars", "max_chars", "overlap_chars", "message"),
    [
        (0, 100, 10, "target_chars"),
        (100, 99, 10, "max_chars"),
        (100, 150, -1, "overlap_chars"),
        (100, 150, 100, "overlap_chars"),
    ],
)
def test_chunker_rejects_invalid_limits(
    target_chars: int,
    max_chars: int,
    overlap_chars: int,
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        MarkdownChunker(
            target_chars=target_chars,
            max_chars=max_chars,
            overlap_chars=overlap_chars,
        )


def test_chunker_can_disable_overlap() -> None:
    document = parse_markdown(
        "# 标题\n\n## 章节\n\nAAAAAA\n\nBBBBBB",
        source_path=Path("no-overlap.md"),
    )

    chunks = MarkdownChunker(target_chars=5, max_chars=10, overlap_chars=0).chunk(document)

    assert [chunk.content_raw for chunk in chunks] == ["AAAAAA", "BBBBBB"]
