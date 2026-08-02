import asyncio

from rag_demo.asyncio_compat import create_compatible_event_loop, run_async


async def _answer() -> int:
    return 42


def test_run_async_returns_coroutine_result() -> None:
    assert run_async(_answer()) == 42


def test_create_compatible_event_loop_returns_a_new_loop() -> None:
    loop = create_compatible_event_loop()
    try:
        assert isinstance(loop, asyncio.AbstractEventLoop)
        assert loop.is_closed() is False
    finally:
        loop.close()
