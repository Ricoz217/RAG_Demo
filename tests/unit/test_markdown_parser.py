from pathlib import Path

from rag_demo.markdown_parser import BlockKind, parse_markdown


def test_parser_extracts_frontmatter_heading_path_lists_and_code_blocks() -> None:
    markdown = """---
title: FastAPI 教程
description: FastAPI 请求体示例。
tags:
  - fastapi
  - pydantic
---

# FastAPI 教程

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
    assert document.first_h1 == "FastAPI 教程"
    assert document.frontmatter == {
        "title": "FastAPI 教程",
        "description": "FastAPI 请求体示例。",
        "tags": ["fastapi", "pydantic"],
    }
    assert document.source_path == Path("tutorial/body.md")
    assert [block.kind for block in document.blocks] == [
        BlockKind.PARAGRAPH,
        BlockKind.PARAGRAPH,
        BlockKind.LIST,
        BlockKind.CODE,
    ]
    assert document.blocks[0].heading_path == ("FastAPI 教程",)
    assert all(
        block.heading_path == ("FastAPI 教程", "请求体")
        for block in document.blocks[1:]
    )
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
        ("主标题", "安全", "OAuth2"),
        ("主标题", "依赖项"),
        ("附录",),
    ]
    assert document.blocks[0].section_root == "主标题"
    assert document.blocks[2].section_root == "附录"


def test_parser_keeps_source_path_but_does_not_infer_title_from_filename() -> None:
    document = parse_markdown("只有正文。", source_path=Path("request-body.md"))

    assert document.title is None
    assert document.first_h1 is None
    assert document.frontmatter == {}
    assert document.source_path == Path("request-body.md")
    assert document.blocks[0].heading_path == ()


def test_parser_uses_skill_name_when_frontmatter_has_no_title() -> None:
    markdown = """---
name: python-patterns
description: Pythonic patterns and type hints.
origin: ECC
---

# Python Development Patterns

Introduction.
"""

    document = parse_markdown(markdown, source_path=Path("SKILL.md"))

    assert document.title == "python-patterns"
    assert document.first_h1 == "Python Development Patterns"
    assert document.frontmatter["description"] == "Pythonic patterns and type hints."
    assert document.blocks[0].heading_path == ("Python Development Patterns",)
    assert document.blocks[0].text == "Introduction."


def test_parser_preserves_preamble_without_assigning_it_to_first_h1() -> None:
    markdown = """你是一个QQ群机器人 Agent 的主控

# 任务

任务正文。

# 规则与说明

规则正文。
"""

    document = parse_markdown(
        markdown,
        source_path=Path("group_chat_system_0710.md"),
    )

    assert document.title is None
    assert document.first_h1 == "任务"
    assert [block.heading_path for block in document.blocks] == [
        (),
        ("任务",),
        ("规则与说明",),
    ]
    assert document.blocks[0].text == "你是一个QQ群机器人 Agent 的主控"


def test_document_hash_is_stable_across_line_endings() -> None:
    source_path = Path("stable.md")

    windows = parse_markdown("# 标题\r\n\r\n正文。\r\n", source_path=source_path)
    unix = parse_markdown("# 标题\n\n正文。\n", source_path=source_path)

    assert windows.content_hash == unix.content_hash
    assert windows.blocks == unix.blocks
