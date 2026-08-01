import subprocess
import sys
from pathlib import Path

import pytest

from rag_demo import AsyncRAG

pytestmark = pytest.mark.integration


@pytest.mark.asyncio
async def test_python_sdk_runs_real_hybrid_search() -> None:
    async with AsyncRAG() as rag:
        response = await rag.search(
            "FastAPI 如何接收 JSON 请求体？",
            use_reranker=False,
            final_top_k=2,
        )

    assert response.results
    assert len(response.results) <= 2
    assert all(candidate.source_path for candidate in response.results)


def test_sync_python_sdk_works_in_a_fresh_windows_process() -> None:
    project_root = Path(__file__).resolve().parents[2]
    script = """
from rag_demo import RAG

with RAG() as rag:
    response = rag.search(
        "FastAPI 如何接收 JSON 请求体？",
        use_reranker=False,
        final_top_k=1,
    )
    print(len(response.results))
"""

    completed = subprocess.run(
        [sys.executable, "-c", script],
        cwd=project_root,
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr
    assert completed.stdout.strip() == "1"
