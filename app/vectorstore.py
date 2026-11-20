"""Vector store, embeddings, and retrieval strategy integrations.

Provides:
- Local FastEmbed embeddings (BAAI/bge-small-en-v1.5)
- Local persistent Chroma vector store
- Sparse BM25 keyword retriever
- Dense, Hybrid (Ensemble), and Hybrid + Cross-Encoder Rerank strategies
"""

import os
import pickle
from typing import List, Optional
from langchain_core.documents import Document
from app.config import get_settings


class EmptyVectorStoreError(RuntimeError):
    """Raised when querying a vector store with no documents indexed."""
    pass


_embeddings = None
_vectorstore = None
_reranker = None


def get_embeddings():
    """Return a cached instance of FastEmbedEmbeddings."""
    global _embeddings
    if _embeddings is None:
        from langchain_community.embeddings import FastEmbedEmbeddings

        settings = get_settings()
        _embeddings = FastEmbedEmbeddings(model_name=settings.fastembed_model)
    return _embeddings


def get_vectorstore():
    """Return a Chroma vector store connected to the persistent directory."""
    global _vectorstore
    from langchain_chroma import Chroma

    settings = get_settings()
    embeddings = get_embeddings()
    _vectorstore = Chroma(
        collection_name=settings.chroma_collection,
        embedding_function=embeddings,
        persist_directory=settings.chroma_dir,
    )
    return _vectorstore


def get_document_count() -> int:
    """Return total number of chunks indexed in the persistent Chroma collection."""
    try:
        vs = get_vectorstore()
        return vs._collection.count()
    except Exception:
        return 0


def get_dense_retriever(top_k: int = 4):
    """Return dense vector similarity retriever."""
    vs = get_vectorstore()
    return vs.as_retriever(search_kwargs={"k": top_k})


def load_bm25_documents() -> List[Document]:
    """Load serialized document chunks for BM25 keyword retrieval."""
    settings = get_settings()
    if os.path.exists(settings.bm25_index_path):
        with open(settings.bm25_index_path, "rb") as f:
            return pickle.load(f)
    return []


def get_bm25_retriever(k: int = 4, docs: Optional[List[Document]] = None):
    """Return BM25 keyword retriever built from persisted chunks."""
    from langchain_community.retrievers import BM25Retriever

    if docs is None:
        docs = load_bm25_documents()

    if not docs:
        raise EmptyVectorStoreError(
            "BM25 index is empty. Please run 'python -m app.ingest' to build document indices."
        )

    retriever = BM25Retriever.from_documents(docs)
    retriever.k = k
    return retriever


def get_hybrid_retriever(top_k: int = 4, weights: Optional[List[float]] = None):
    """Return EnsembleRetriever combining dense vector search and BM25 sparse search."""
    try:
        from langchain.retrievers import EnsembleRetriever
    except ImportError:
        from langchain_classic.retrievers import EnsembleRetriever

    if weights is None:
        weights = [0.5, 0.5]

    dense = get_dense_retriever(top_k=top_k)
    bm25 = get_bm25_retriever(k=top_k)

    return EnsembleRetriever(
        retrievers=[bm25, dense],
        weights=weights,
    )


def deduplicate_documents(docs: List[Document]) -> List[Document]:
    """Deduplicate document chunks based on source, page, and content excerpt."""
    seen = set()
    unique_docs = []
    for doc in docs:
        key = (
            doc.metadata.get("source", ""),
            doc.metadata.get("page", 0),
            doc.page_content.strip()[:100],
        )
        if key not in seen:
            seen.add(key)
            unique_docs.append(doc)
    return unique_docs


def rerank_documents(query: str, docs: List[Document], top_k: int = 4) -> List[Document]:
    """Rerank candidate documents using FastEmbed's lightweight cross-encoder."""
    global _reranker
    if not docs:
        return []

    settings = get_settings()
    if _reranker is None:
        from fastembed.rerank.cross_encoder import TextCrossEncoder

        _reranker = TextCrossEncoder(model_name=settings.reranker_model)

    passages = [d.page_content for d in docs]
    results = list(_reranker.rerank(query, passages))
    if not results:
        return docs[:top_k]

    # Handle float scores (FastEmbed TextCrossEncoder) or structured score objects
    if isinstance(results[0], (int, float)):
        scored_docs = sorted(zip(docs, results), key=lambda x: x[1], reverse=True)
        return [doc for doc, _ in scored_docs[:top_k]]
    elif hasattr(results[0], "score") and hasattr(results[0], "index"):
        sorted_results = sorted(results, key=lambda x: x.score, reverse=True)
        return [docs[item.index] for item in sorted_results[:top_k]]
    elif hasattr(results[0], "score"):
        scored_docs = sorted(zip(docs, [r.score for r in results]), key=lambda x: x[1], reverse=True)
        return [doc for doc, _ in scored_docs[:top_k]]
    else:
        return docs[:top_k]


def retrieve_documents(
    query: str,
    top_k: int = 4,
    strategy: Optional[str] = None,
) -> List[Document]:
    """Unified retrieval interface executing the configured or requested strategy.

    Strategies:
    - 'dense': Vector similarity search via Chroma.
    - 'hybrid': Reciprocal Rank Fusion of BM25 sparse + Chroma dense retrieval.
    - 'hybrid_rerank': Hybrid retrieval of top-N candidates, followed by cross-encoder reranking to top_k.
    """
    settings = get_settings()
    active_strategy = strategy or settings.retrieval_strategy

    count = get_document_count()
    if count == 0:
        raise EmptyVectorStoreError(
            "Vector store is empty. Please run ingestion first using: python -m app.ingest"
        )

    if active_strategy == "dense":
        retriever = get_dense_retriever(top_k=top_k)
        docs = retriever.invoke(query)
        return deduplicate_documents(docs)[:top_k]

    elif active_strategy == "hybrid":
        retriever = get_hybrid_retriever(top_k=top_k)
        docs = retriever.invoke(query)
        return deduplicate_documents(docs)[:top_k]

    elif active_strategy == "hybrid_rerank":
        candidates_k = max(top_k, settings.rerank_candidates)
        hybrid_retriever = get_hybrid_retriever(top_k=candidates_k)
        candidate_docs = deduplicate_documents(hybrid_retriever.invoke(query))
        return rerank_documents(query, candidate_docs, top_k=top_k)

    else:
        raise ValueError(
            f"Unknown retrieval strategy: '{active_strategy}'. Must be 'dense', 'hybrid', or 'hybrid_rerank'."
        )
