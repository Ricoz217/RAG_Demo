"""Windows-compatible asyncio entry points for Psycopg."""

from __future__ import annotations

import asyncio
import selectors
import sys
from collections.abc import Coroutine
from typing import Any


def run_async[ResultT](coroutine: Coroutine[Any, Any, ResultT]) -> ResultT:
    """Run a coroutine with an event loop supported by Psycopg on Windows."""
    if sys.platform == "win32":
        return asyncio.run(coroutine, loop_factory=_windows_selector_loop)
    return asyncio.run(coroutine)


def _windows_selector_loop() -> asyncio.AbstractEventLoop:
    return asyncio.SelectorEventLoop(selectors.SelectSelector())
