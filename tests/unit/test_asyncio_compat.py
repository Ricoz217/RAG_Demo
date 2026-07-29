from rag_demo.asyncio_compat import run_async


async def _answer() -> int:
    return 42


def test_run_async_returns_coroutine_result() -> None:
    assert run_async(_answer()) == 42
