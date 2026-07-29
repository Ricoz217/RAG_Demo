from rag_demo.bm25_retriever import tokenize_bm25


def test_tokenizer_segments_chinese_for_search() -> None:
    tokens = tokenize_bm25("怎样让接口接收一个 JSON 对象？")

    assert "接口" in tokens
    assert "接收" in tokens
    assert "json" in tokens
    assert "对象" in tokens


def test_tokenizer_lowercases_and_preserves_technical_identifiers() -> None:
    tokens = tokenize_bm25("OAuth2PasswordBearer、HTTP 422、request_body 与 foo.bar()")

    assert "oauth2passwordbearer" in tokens
    assert "http" in tokens
    assert "422" in tokens
    assert "request_body" in tokens
    assert "foo.bar" in tokens


def test_tokenizer_ignores_punctuation_and_whitespace() -> None:
    assert tokenize_bm25(" \n\t，。！？ ") == ()
