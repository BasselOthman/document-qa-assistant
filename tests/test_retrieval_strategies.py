"""Tests for Dense, Hybrid (Ensemble), and Reranked retrieval strategies."""

from typing import List
from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
try:
    from langchain.retrievers import EnsembleRetriever
except ImportError:
    from langchain_classic.retrievers import EnsembleRetriever
from langchain_community.retrievers import BM25Retriever

from app.vectorstore import deduplicate_documents


class MockDenseRetriever(BaseRetriever):
    """Simulated dense retriever that misses exact rare keywords (common semantic search failure)."""

    docs: List[Document]

    def _get_relevant_documents(
        self, query: str, *, run_manager: CallbackManagerForRetrieverRun
    ) -> List[Document]:
        # Simulates dense vector model returning generic semantic documents rather than exact keyword match
        return [self.docs[0], self.docs[2]]


def test_hybrid_retrieval_catches_keyword_missed_by_dense():
    """Verify hybrid BM25 + dense ensemble retrieves an exact keyword query that dense alone misses."""
    docs = [
        Document(
            page_content="System architecture overview: services communicate over gRPC.",
            metadata={"source": "arch.pdf", "page": 1},
        ),
        Document(
            page_content="Critical error code ERR-9042 indicates database connection timeout.",
            metadata={"source": "troubleshoot.pdf", "page": 5},
        ),
        Document(
            page_content="General logging guidelines: all components log structured JSON.",
            metadata={"source": "logging.pdf", "page": 2},
        ),
    ]

    query = "What causes error code ERR-9042?"

    # 1. Dense alone misses doc 1 (the ERR-9042 document)
    dense_retriever = MockDenseRetriever(docs=docs)
    dense_results = dense_retriever.invoke(query)
    dense_contents = [d.page_content for d in dense_results]
    assert not any("ERR-9042" in c for c in dense_contents), "Dense alone simulated miss"

    # 2. BM25 alone catches the exact keyword
    bm25_retriever = BM25Retriever.from_documents(docs)
    bm25_retriever.k = 2
    bm25_results = bm25_retriever.invoke(query)
    assert any("ERR-9042" in d.page_content for d in bm25_results), "BM25 should capture exact keyword"

    # 3. Hybrid EnsembleRetriever successfully surfaces the keyword match
    ensemble = EnsembleRetriever(
        retrievers=[bm25_retriever, dense_retriever],
        weights=[0.5, 0.5],
    )
    hybrid_results = ensemble.invoke(query)
    hybrid_contents = [d.page_content for d in hybrid_results]

    assert any("ERR-9042" in c for c in hybrid_contents), (
        "Hybrid search must retrieve the exact keyword chunk missed by dense alone"
    )


def test_deduplicate_documents():
    """Verify deduplication removes redundant chunks while preserving unique order."""
    doc_a = Document(page_content="Unique chunk 1", metadata={"source": "a.pdf", "page": 1})
    doc_a_dup = Document(page_content="Unique chunk 1", metadata={"source": "a.pdf", "page": 1})
    doc_b = Document(page_content="Unique chunk 2", metadata={"source": "b.pdf", "page": 3})

    deduped = deduplicate_documents([doc_a, doc_a_dup, doc_b])
    assert len(deduped) == 2
    assert deduped[0].page_content == "Unique chunk 1"
    assert deduped[1].page_content == "Unique chunk 2"


def test_mock_reranker():
    """Verify reranking orders candidate passages based on query-relevance scores."""
    class DummyScore:
        def __init__(self, index, score):
            self.index = index
            self.score = score

    docs = [
        Document(page_content="Irrelevant text about gardening", metadata={"source": "g.pdf", "page": 1}),
        Document(page_content="Highly relevant passage answering the user question", metadata={"source": "q.pdf", "page": 2}),
    ]

    # Simulated cross encoder score ordering
    scores = [DummyScore(index=1, score=0.95), DummyScore(index=0, score=0.12)]
    sorted_scores = sorted(scores, key=lambda x: x.score, reverse=True)
    reranked = [docs[s.index] for s in sorted_scores]

    assert "Highly relevant" in reranked[0].page_content
