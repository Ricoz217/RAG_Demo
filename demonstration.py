import asyncio
import json
import traceback
import shlex

from concurrent.futures import ThreadPoolExecutor
from dataclasses import is_dataclass, asdict
from pathlib import Path
from hashlib import blake2b

from rag_demo.sdk import (
    AsyncRAG,
    Settings,
    QuerySearchResponse,
    IngestResult,
    BM25BuildResult,
    BM25Retriever,
    DoctorReport
)
from rag_demo.asyncio_compat import run_async


# _RAG = AsyncRAG()  # 构建一个默认单例
_RAG = AsyncRAG(settings=Settings(
    bm25_index_path="data/indexes/bm25_demo_02"
))

_EXECUTOR = ThreadPoolExecutor()


async def async_input(prompt: str = '>') -> str:
    def sync_input() -> str:
        return input(f"{prompt} ")

    loop = asyncio.get_running_loop()

    # noinspection PyTypeChecker
    future = loop.run_in_executor(_EXECUTOR, sync_input)
    return await future

def indent_print(obj: object):
    """固定打印 indent = 2 的结构化信息"""
    def _default_json(_obj: object):
        if hasattr(_obj, "to_json") and callable(_obj.to_json):
            return _obj.to_json()

        elif hasattr(_obj, "to_dict") and callable(_obj.to_dict):
            return _obj.to_dict()

        elif is_dataclass(_obj):
            return asdict(_obj)

        else:
            raise TypeError("无法将对象打印为结构化信息")

    if obj is None:
        return

    print(json.dumps(obj, ensure_ascii=False, indent=2, default=_default_json))


async def main():
    async with _RAG as rag_engine:
        await rag_engine.init_database()
        async def search(query: str = "", rewrite: str = "") -> QuerySearchResponse:
            """查询"""
            rewrite = True if rewrite.lower() == "true" else False
            return await rag_engine.search_query(query=query, rewrite=rewrite)

        async def ingest(source: str) -> IngestResult:
            """入库"""
            source_path = Path(source)
            if not source_path.is_file():
                raise FileNotFoundError("Demo only support one file in single injection")

            if source_path.suffix.lower() != ".md":
                raise ValueError("Demo only support ingest Markdown file")

            source_repo = "https://localtest/"
            source_commit = blake2b(source_path.read_bytes(), digest_size=32).hexdigest()
            return await rag_engine.ingest(source=source_path, source_repo=source_repo, source_commit=source_commit)

        async def rebuild_bm25() -> BM25BuildResult:
            return await rag_engine.rebuild_bm25()

        async def reload_bm25() -> BM25Retriever:
            return await rag_engine.reload_bm25()

        async def doctor() -> DoctorReport:
            return await rag_engine.doctor()

        EXEC_MAPPING = {
            "query": search,
            "search": search,
            "ingest": ingest,
            "rebuild": rebuild_bm25,
            "reload": reload_bm25,
            "doctor": doctor
        }

        async def _run(*args):
            if cmd in EXEC_MAPPING:
                if params:
                    result = await EXEC_MAPPING[cmd](*args)

                else:
                    result = await EXEC_MAPPING[cmd]()

                loop = asyncio.get_running_loop()
                await loop.run_in_executor(_EXECUTOR, indent_print, result)

        try:
            while True:
                prompt = await async_input()
                splited = shlex.split(prompt)
                if not splited:
                    continue

                cmd = splited[0]
                params = splited[1:]
                if cmd not in EXEC_MAPPING:
                    print(f"unknown cmd: {cmd}")
                    continue

                asyncio.create_task(_run(*params))

        except asyncio.CancelledError:
            exit(0)

        except KeyboardInterrupt:
            exit(0)

        except Exception as E:
            print(E)
            print(traceback.format_exc())


if __name__ == '__main__':
    run_async(main())
