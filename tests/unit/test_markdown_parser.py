from pathlib import Path

from rag_demo.markdown_parser import BlockKind, parse_markdown


def test_parser_extracts_title_heading_path_lists_and_code_blocks() -> None:
    markdown = """# FastAPI 教程

文档导言。

## 请求体

使用 Pydantic 模型声明请求体。

- 字段一
- 字段二

```python
class Item:
    name: str
```
"""

    document = parse_markdown(markdown, source_path=Path("tutorial/body.md"))

    assert document.title == "FastAPI 教程"
    assert document.source_path == Path("tutorial/body.md")
    assert [block.kind for block in document.blocks] == [
        BlockKind.PARAGRAPH,
        BlockKind.PARAGRAPH,
        BlockKind.LIST,
        BlockKind.CODE,
    ]
    assert document.blocks[0].heading_path == ()
    assert all(block.heading_path == ("请求体",) for block in document.blocks[1:])
    assert document.blocks[2].text == "- 字段一\n- 字段二"
    assert document.blocks[3].text.startswith("```python\n")
    assert document.blocks[3].text.endswith("\n```")


def test_parser_tracks_nested_headings_and_resets_at_same_or_higher_level() -> None:
    markdown = """# 主标题

## 安全

### OAuth2

OAuth2 内容。

## 依赖项

依赖项内容。

# 附录

附录内容。
"""

    document = parse_markdown(markdown, source_path=Path("guide.md"))

    assert [block.heading_path for block in document.blocks] == [
        ("安全", "OAuth2"),
        ("依赖项",),
        ("附录",),
    ]
    assert document.blocks[0].section_root == "主标题"
    assert document.blocks[2].section_root == "附录"


def test_parser_uses_filename_when_document_has_no_heading() -> None:
    document = parse_markdown("只有正文。", source_path=Path("request-body.md"))

    assert document.title == "request-body"
    assert document.blocks[0].heading_path == ()


def test_document_hash_is_stable_across_line_endings() -> None:
    source_path = Path("stable.md")

    windows = parse_markdown("# 标题\r\n\r\n正文。\r\n", source_path=source_path)
    unix = parse_markdown("# 标题\n\n正文。\n", source_path=source_path)

    assert windows.content_hash == unix.content_hash
    assert windows.blocks == unix.blocks
