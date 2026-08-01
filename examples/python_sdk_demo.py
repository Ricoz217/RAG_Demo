"""Small synchronous Python SDK demo using the configured local infrastructure."""

from rag_demo import RAG


def main() -> None:
    with RAG() as rag:
        response = rag.search(
            "FastAPI 如何接收 JSON 请求体？",
            final_top_k=5,
        )

        for candidate in response.results:
            print(f"#{candidate.final_rank} {candidate.source_path}")
            print(f"heading: {' > '.join(candidate.heading_path) or '(root)'}")
            print(f"rerank: {candidate.rerank_score}")
            print(candidate.content_raw[:300])
            print("-" * 80)


if __name__ == "__main__":
    main()
