"""Windows-compatible asyncio entry points for Psycopg."""

from __future__ import annotations

import asyncio
import selectors
import sys
from collections.abc import Coroutine
from typing import Any


def run_async[ResultT](coroutine: Coroutine[Any, Any, ResultT]) -> ResultT:
    """Run a coroutine with an event loop supported by Psycopg on Windows."""
    return asyncio.run(coroutine, loop_factory=create_compatible_event_loop)


def create_compatible_event_loop() -> asyncio.AbstractEventLoop:
    """Create an event loop compatible with Psycopg on the current platform."""
    if sys.platform == "win32":
        return asyncio.SelectorEventLoop(selectors.SelectSelector())
    return asyncio.new_event_loop()
